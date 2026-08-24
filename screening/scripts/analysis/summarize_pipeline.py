#!/usr/bin/env python3
"""Create the authoritative screening funnel and outcome summary.

The summary is derived from saved stage artifacts; it does not rerun PM7,
ML, LLM, Pareto selection, or DFT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SCREENING = SCRIPT_DIR.parent.parent
PROJECT = SCREENING.parent
EXECUTION = SCREENING / "scripts" / "execution"
sys.path.insert(0, str(EXECUTION))

from pipeline_config import PATHS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iter", type=int, default=1)
    ap.add_argument("--input-dir", type=Path)
    ap.add_argument("--build-report", type=Path)
    ap.add_argument("--dft-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--config", type=Path, default=PATHS.config)
    args = ap.parse_args()

    input_dir = args.input_dir or PATHS.iteration_dir(args.iter)
    build_report = (
        args.build_report or PATHS.data / "processed" / "build_report.json"
    )
    dft_dir = PATHS.dft_validation_dir(
        args.iter,
        explicit=args.dft_dir,
        required_files=("lead_selection_analysis.csv", "top5_leads.csv"),
    )
    output_dir = args.output_dir or input_dir

    candidates = pd.read_parquet(input_dir / "candidates.parquet")
    pm7_failed = pd.read_csv(input_dir / "pm7_failed.csv")
    pm7 = pd.read_parquet(input_dir / "pm7_results.parquet")
    features = pd.read_parquet(input_dir / "features.parquet")
    predictions = pd.read_parquet(input_dir / "predictions.parquet")
    molecular = pd.read_parquet(input_dir / "molecular_pa.parquet")
    verdicts = pd.read_parquet(input_dir / "llm_verdicts.parquet")
    pareto = pd.read_csv(input_dir / "pareto_selected.csv")
    dft = pd.read_csv(dft_dir / "lead_selection_analysis.csv")
    leads = pd.read_csv(dft_dir / "top5_leads.csv")
    build = json.loads(build_report.read_text())
    config = json.loads(args.config.read_text())

    low = float(config["pa_window_kcalmol"]["low"])
    high = float(config["pa_window_kcalmol"]["high"])
    eligible_verdicts = config["pareto"]["eligible_verdicts"]
    eligible = verdicts["final_verdict"].isin(eligible_verdicts)
    in_window = verdicts["pa_pred_kcalmol"].between(low, high)

    summary = {
        "filtered_library_size": int(build["n_molecules"]),
        "retrieved_candidates": int(len(candidates)),
        "pm7_successful_molecules": int(pm7["smiles"].nunique()),
        "pm7_failed_molecules": int(pm7_failed["smiles"].nunique()),
        "pm7_successful_sites": int(len(pm7)),
        "featurized_molecules": int(features["smiles"].nunique()),
        "featurized_sites": int(len(features)),
        "predicted_molecules": int(molecular["smiles"].nunique()),
        "predicted_sites": int(len(predictions)),
        "predicted_pa_window_molecules": int(
            molecular["pa_pred_kcalmol"].between(low, high).sum()
        ),
        "rule_accept": int((verdicts["rule_verdict"] == "accept").sum()),
        "rule_flag": int((verdicts["rule_verdict"] == "flag").sum()),
        "rule_reject": int((verdicts["rule_verdict"] == "reject").sum()),
        "llm_accept": int((verdicts["final_verdict"] == "accept").sum()),
        "llm_flag": int((verdicts["final_verdict"] == "flag").sum()),
        "llm_reject": int((verdicts["final_verdict"] == "reject").sum()),
        "llm_eligible_total": int(eligible.sum()),
        "llm_eligible_in_pa_window": int((eligible & in_window).sum()),
        "pareto_selected": int(len(pareto)),
        "dft_completed": int(len(dft)),
        "dft_in_pa_window": int(dft["dft_pa_in_window"].sum()),
        "stability_eligible": int(dft["stability_eligible"].sum()),
        "lead_eligible": int(dft["lead_eligible"].sum()),
        "priority_leads": int(len(leads)),
        "pa_window_kcalmol": [low, high],
        "prospective_ml_mae_kcalmol": float(dft["ml_error_kcalmol"].abs().mean()),
        "prospective_ml_bias_kcalmol": float(dft["ml_error_kcalmol"].mean()),
        "prospective_pm7_mae_kcalmol": float(dft["pm7_error_kcalmol"].abs().mean()),
        "eligible_verdicts": eligible_verdicts,
    }

    checks = {
        "candidate_smiles_are_unique": not candidates["smiles"].duplicated().any(),
        "pm7_success_and_failure_partition_candidates": (
            set(pm7["smiles"]) | set(pm7_failed["smiles"])
            == set(candidates["smiles"])
            and not (set(pm7["smiles"]) & set(pm7_failed["smiles"]))
        ),
        "features_match_pm7_rows": len(features) == len(pm7),
        "predictions_match_feature_rows": len(predictions) == len(features),
        "molecular_predictions_match_pm7_molecules": (
            molecular["smiles"].nunique() == pm7["smiles"].nunique()
        ),
        "verdicts_match_molecular_predictions": len(verdicts) == len(molecular),
        "pareto_count_matches_reported_selection": (
            len(pareto) == int(config["pareto"]["n_select"])
        ),
        "dft_count_matches_pareto": len(dft) == len(pareto),
        "lead_count_matches_config": (
            len(leads) == int(config["lead_selection"]["n_leads"])
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Screening artifact consistency checks failed: {failed}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps({"summary": summary, "checks": checks}, indent=2)
    )
    pd.DataFrame([summary]).to_csv(output_dir / "pipeline_summary.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
