"""Select priority leads from a completed prospective DFT run.

This stage implements the lead-selection rule described in the submitted
manuscript: retain DFT-window candidates with an independently verified
Grotthuss motif and an explicit human stability review, then rank the eligible
pool by decreasing molecular DFT proton affinity.

The molecular PA is the maximum over all enumerated protonation sites, matching
the reported workflow.  The best nitrogen-site PA is also reported as a
diagnostic because the molecular maximum can occur at a carbonyl or thione.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_config import load_pipeline_config


KJ_TO_KCAL = 1 / 4.184


def _as_bool(series: pd.Series, name: str) -> pd.Series:
    """Parse a boolean column without treating non-empty strings as true."""
    if pd.api.types.is_bool_dtype(series):
        return series
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false"})
    if invalid.any():
        bad = sorted(series[invalid].astype(str).unique())
        raise ValueError(f"{name} contains non-boolean values: {bad}")
    return normalized.eq("true")


def _load_dft_records(results_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open() as handle:
            record = json.load(handle)

        if record.get("status") != "OK":
            raise ValueError(f"DFT result is not OK: {path}")
        neutral_imag = record.get("neutral", {}).get("n_imaginary")
        protonated_imag = record.get("protonated_best", {}).get("n_imaginary")
        if neutral_imag != 0 or protonated_imag != 0:
            raise ValueError(
                f"Non-minimum geometry in {path}: neutral={neutral_imag}, "
                f"protonated={protonated_imag}"
            )

        sites = [site for site in record.get("all_sites", []) if site.get("status") == "OK"]
        n_sites = [site for site in sites if site.get("atom") == "N"]
        best_n_pa = max((site["pa_kjmol"] for site in n_sites), default=None)
        dft_pa_kcal = float(record["dft_pa"]) * KJ_TO_KCAL
        rows.append(
            {
                "mol_idx": record["mol_idx"],
                "smiles": record["smiles"],
                "pa_dft_kcalmol": dft_pa_kcal,
                "best_n_site_pa_kcalmol": (
                    float(best_n_pa) * KJ_TO_KCAL if best_n_pa is not None else None
                ),
                "best_site": record.get("best_site"),
                "best_site_element": str(record.get("best_site", ""))[:1],
                "dft_n_sites": record.get("n_sites"),
                "neutral_homo_ev": record["neutral"].get("HOMO_eV"),
                "neutral_lumo_ev": record["neutral"].get("LUMO_eV"),
                "neutral_gap_ev": record["neutral"].get("HOMO_LUMO_gap_eV"),
                "neutral_dipole_debye": record["neutral"].get("dipole_debye"),
            }
        )
    if not rows:
        raise ValueError(f"No JSON DFT results found in {results_dir}")
    return pd.DataFrame(rows)


def select_leads(
    results_dir: Path,
    screening_csv: Path,
    stability_review_csv: Path,
    output_dir: Path,
    config_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = load_pipeline_config(config_path)
    lead_config = config["lead_selection"]
    pa_low = float(config["pa_window_kcalmol"]["low"])
    pa_high = float(config["pa_window_kcalmol"]["high"])

    dft = _load_dft_records(results_dir)
    screening = pd.read_csv(screening_csv)
    review = pd.read_csv(stability_review_csv)

    required_review = {"smiles", "stability_verdict", "stability_reason"}
    missing_review = required_review - set(review.columns)
    if missing_review:
        raise ValueError(f"Stability review is missing columns: {sorted(missing_review)}")
    if review["smiles"].duplicated().any():
        raise ValueError("Stability review contains duplicate SMILES")

    dft_smiles = set(dft["smiles"])
    review_smiles = set(review["smiles"])
    if dft_smiles != review_smiles:
        missing = sorted(dft_smiles - review_smiles)
        extra = sorted(review_smiles - dft_smiles)
        raise ValueError(
            "Stability review must cover every DFT molecule exactly once; "
            f"missing={missing}, extra={extra}"
        )

    metadata_columns = [
        "smiles",
        "selection_rank",
        "mol_id",
        "pa_pm7_kcalmol",
        "pa_pred_kcalmol",
        "uncertainty",
        "sa_score",
        "motif_capable",
        "motif_reason",
        "final_verdict",
        "structural_concern",
    ]
    missing_metadata = set(metadata_columns) - set(screening.columns)
    if missing_metadata:
        raise ValueError(f"Screening metadata is missing columns: {sorted(missing_metadata)}")

    analysis = dft.merge(screening[metadata_columns], on="smiles", validate="one_to_one")
    analysis = analysis.merge(review, on="smiles", validate="one_to_one")
    analysis["motif_capable"] = _as_bool(analysis["motif_capable"], "motif_capable")
    analysis["ml_error_kcalmol"] = analysis["pa_pred_kcalmol"] - analysis["pa_dft_kcalmol"]
    analysis["pm7_error_kcalmol"] = analysis["pa_pm7_kcalmol"] - analysis["pa_dft_kcalmol"]
    analysis["dft_pa_in_window"] = analysis["pa_dft_kcalmol"].between(pa_low, pa_high)
    analysis["stability_eligible"] = analysis["stability_verdict"].isin(
        lead_config["eligible_stability_verdicts"]
    )

    eligible = analysis[analysis["stability_eligible"]].copy()
    if lead_config["require_grotthuss_motif"]:
        eligible = eligible[eligible["motif_capable"]]
    if lead_config["require_dft_pa_window"]:
        eligible = eligible[eligible["dft_pa_in_window"]]

    eligible = eligible.sort_values(
        ["pa_dft_kcalmol", "selection_rank"], ascending=[False, True]
    ).copy()
    n_leads = int(lead_config["n_leads"])
    top = eligible.head(n_leads).copy()
    top.insert(0, "lead_label", [f"Mol-{i}" for i in range(1, len(top) + 1)])

    analysis["lead_eligible"] = analysis["smiles"].isin(eligible["smiles"])
    lead_lookup = top.set_index("smiles")["lead_label"].to_dict()
    analysis["lead_label"] = analysis["smiles"].map(lead_lookup)
    analysis = analysis.sort_values("pa_dft_kcalmol", ascending=False)

    report = {
        "n_dft_results": int(len(analysis)),
        "n_dft_in_window": int(analysis["dft_pa_in_window"].sum()),
        "n_motif_capable": int(analysis["motif_capable"].sum()),
        "n_stability_eligible": int(analysis["stability_eligible"].sum()),
        "n_lead_eligible": int(analysis["lead_eligible"].sum()),
        "n_priority_leads": int(len(top)),
        "ranking_rule": "decreasing molecular B3LYP/def2-TZVP proton affinity",
        "pa_window_kcalmol": [pa_low, pa_high],
        "non_nitrogen_best_site_leads": top.loc[
            top["best_site_element"] != "N", "lead_label"
        ].tolist(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(output_dir / "lead_selection_analysis.csv", index=False)
    top.to_csv(output_dir / "top5_leads.csv", index=False)
    with (output_dir / "lead_selection_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return analysis, top, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--screening-csv", required=True, type=Path)
    parser.add_argument("--stability-review", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    _, top, report = select_leads(
        results_dir=args.results_dir,
        screening_csv=args.screening_csv,
        stability_review_csv=args.stability_review,
        output_dir=args.output_dir,
        config_path=args.config,
    )
    print(json.dumps(report, indent=2))
    print(top[["lead_label", "mol_idx", "smiles", "pa_dft_kcalmol"]].to_string(index=False))


if __name__ == "__main__":
    main()
