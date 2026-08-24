"""
build_targets.py
================
Constructs ML targets by joining DFT and experimental PA values into
the feature files, then computes molecule-level delta targets.

Target definitions
------------------

NIST 1155-molecule dataset  (nist1185_features.parquet)
  Source: PM7 calculations + experimental PA from NIST (via DFT JSON files)
  Level:  molecule-level — one row per molecule (best PM7 site)
  Target: delta_pm7_exp = exp_pa_kjmol - pm7_best_pa_kjmol  (signed)
  Also records:
    pm7_best_pa_kjmol   — max PM7 PA across all sites for that molecule
    exp_pa_kjmol        — NIST experimental PA
    raw_pm7_error       — pm7_best_pa - exp_pa  (signed, for baseline plots)

k-means 251-molecule dataset  (kmeans251_features.parquet)
  Source: PM7 calculations + B3LYP/def2-TZVP DFT (from folder records)
  Level:  site-level — one row per (molecule, site)
  Target: delta_dft_pm7 = dft_pa_kjmol - pm7_pa_kjmol  (signed)
  Also records:
    dft_pa_kjmol        — B3LYP DFT PA for this site
    pm7_pa_kjmol        — PM7 PA for this site
    raw_pm7_error       — pm7_pa - dft_pa  (signed)

Join strategy
-------------
  NIST exp_pa:  neutral_smiles -> exp_pa from DFT dataset.json (json-source records)
                These come from the 1185 JSON files which contain NIST exp_pa values.
  k-means DFT:  neutral_smiles -> dft_pa per site from DFT dataset.json (folder-source)
                Joined on neutral_smiles + site match via protonated_smiles.

Outputs  (../data/targets/)
-------
  nist1155_ml.parquet / .csv   — molecule-level, ready for ML
  kmeans251_ml.parquet / .csv  — site-level, ready for ML
  target_report.json           — join stats, coverage, target distributions
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR   = next((p for p in (PROJECT_DIR / "data", PROJECT_DIR.parent / "data") if p.exists()), PROJECT_DIR / "data")
# ---------------------------------------------------------------------------
# Load processed datasets
# ---------------------------------------------------------------------------

def load_dft_exp_map(data_dir: Path, processed_dir: Path) -> dict[str, float]:
    """
    Build smiles -> exp_pa_kjmol from the Jin & Merz dataset Excel file.

    The Excel file (Merz-and-hogni-dataset.xlsx) contains the experimental
    NIST proton affinities for all 1185 molecules. EXP_PA is stored in kJ/mol
    despite the paper reporting in kcal/mol — confirmed by cross-checking
    against NIST (e.g. COF2 = 666.7 kJ/mol).

    We use ONLY the smiles and EXP_PA columns. All other columns (their 186
    descriptors) are ignored — we are not using their features.

    Expected location: DATA_DIR / "Merz-and-hogni-dataset.xlsx"
    (place the file in the data/ directory on the cluster)
    """
    excel_path = data_dir / "Merz-and-hogni-dataset.xlsx"
    if not excel_path.exists():
        # Fallback: try to load from DFT JSON records (only 1 of 1185 in test env)
        log.warning(f"Jin & Merz Excel not found at {excel_path}")
        log.warning("Falling back to DFT JSON records (incomplete — use Excel for full dataset)")
        dataset = json.loads((processed_dir / "dataset.json").read_text())
        smap = {}
        for rec in dataset.values():
            if rec["metadata"]["source"] != "json":
                continue
            exp = rec["labels"].get("exp_pa_kjmol")
            smi = rec["neutral"].get("smiles")
            if exp and smi:
                smap[smi] = float(exp)
        log.info(f"exp_pa map (fallback): {len(smap)} molecules from DFT json records")
        return smap

    df = pd.read_excel(excel_path)
    # EXP_PA is in kJ/mol (verified: 666.7 kJ/mol for COF2 matches NIST)
    smap = {row["smiles"]: float(row["EXP_PA"])
            for _, row in df.iterrows()
            if pd.notna(row["EXP_PA"]) and pd.notna(row["smiles"])}
    log.info(f"exp_pa map: {len(smap)} molecules from Jin & Merz Excel")
    return smap


def _canonicalize(smi: str) -> str:
    """Canonicalize SMILES via RDKit. Returns empty string on failure."""
    if not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    return Chem.MolToSmiles(mol)


def load_dft_site_map(processed_dir: Path) -> pd.DataFrame:
    """
    Build a DataFrame of DFT folder site-level PAs:
    (neutral_smiles, protonated_smiles) -> dft_pa_kjmol

    Used to join DFT site PAs onto k-means PM7 site rows.
    Protonated SMILES are canonicalized to match PM7 format.
    """
    dataset = json.loads((processed_dir / "dataset.json").read_text())
    rows = []
    for rec in dataset.values():
        if rec["metadata"]["source"] != "folder":
            continue
        neu_smi = rec["neutral"].get("smiles", "")
        for site in rec["all_sites"]:
            pa = site.get("pa_kjmol")
            if pa is None:
                continue
            rows.append({
                "neutral_smiles":    _canonicalize(neu_smi),
                "protonated_smiles": _canonicalize(site.get("protonated_smiles", "")),
                "dft_pa_kjmol":      float(pa),
                "dft_pa_kcalmol":    float(pa) * 0.239006,
            })
    df = pd.DataFrame(rows)
    log.info(f"DFT site map: {len(df)} sites from {df['neutral_smiles'].nunique()} molecules")
    return df


# ---------------------------------------------------------------------------
# NIST dataset — molecule-level target
# ---------------------------------------------------------------------------

def build_nist_targets(exp_map: dict[str, float], feature_dir: Path) -> pd.DataFrame:
    """
    From nist1185_features.parquet:
      1. Join exp_pa onto each row via neutral_smiles
      2. For each molecule, select the best-site row (max pm7_pa_kjmol)
      3. Compute delta_pm7_exp = exp_pa - pm7_best_pa (signed)

    Returns molecule-level DataFrame (one row per molecule).
    """
    df = pd.read_parquet(feature_dir / "nist1185_features.parquet")
    log.info(f"NIST features loaded: {len(df)} site rows, "
             f"{df['record_id'].nunique()} molecules")

    # Join exp_pa
    df["exp_pa_kjmol"] = df["neutral_smiles"].map(exp_map)
    n_matched = df["exp_pa_kjmol"].notna().sum()
    log.info(f"  exp_pa joined: {df['exp_pa_kjmol'].nunique()} unique values, "
             f"{n_matched}/{len(df)} rows matched "
             f"({df[df['exp_pa_kjmol'].notna()]['record_id'].nunique()} molecules)")

    # Drop molecules without exp_pa
    before = df["record_id"].nunique()
    df = df[df["exp_pa_kjmol"].notna()].copy()
    after = df["record_id"].nunique()
    if before > after:
        log.warning(f"  Dropped {before - after} molecules without exp_pa")

    # Per molecule: select best-site row (max pm7_pa_kjmol)
    # This aligns with how experimental PA is reported — the most basic site
    idx_best = df.groupby("record_id")["pm7_pa_kjmol"].idxmax()
    df_mol = df.loc[idx_best].copy()
    df_mol = df_mol.rename(columns={"pm7_pa_kjmol": "pm7_best_pa_kjmol",
                                     "pm7_pa_kcalmol": "pm7_best_pa_kcalmol"})

    # Compute targets — SIGNED correction: PA_pred = PA_PM7 + correction_ML
    # Using signed difference so the model learns both direction and magnitude.
    # PA_exp - PA_PM7 > 0 means PM7 underestimated (needs positive correction)
    # PA_exp - PA_PM7 < 0 means PM7 overestimated (needs negative correction)
    df_mol["delta_pm7_exp"]   = df_mol["exp_pa_kjmol"] - df_mol["pm7_best_pa_kjmol"]   # signed
    df_mol["raw_pm7_error"]   = df_mol["pm7_best_pa_kjmol"] - df_mol["exp_pa_kjmol"]   # signed (opposite convention for reference)
    df_mol["exp_pa_kcalmol"]  = df_mol["exp_pa_kjmol"] * 0.239006

    log.info(f"  NIST molecule-level dataset: {len(df_mol)} molecules")
    log.info(f"  delta_pm7_exp (signed)  mean={df_mol['delta_pm7_exp'].mean():.2f}  "
             f"std={df_mol['delta_pm7_exp'].std():.2f}  "
             f"min={df_mol['delta_pm7_exp'].min():.2f}  max={df_mol['delta_pm7_exp'].max():.2f}  kJ/mol")

    return df_mol.reset_index(drop=True)


# ---------------------------------------------------------------------------
# k-means dataset — site-level target
# ---------------------------------------------------------------------------

def build_kmeans_targets(
    dft_site_map: pd.DataFrame,
    feature_dir: Path,
) -> pd.DataFrame:
    """
    From kmeans251_features.parquet:
      1. Join DFT site PA via (neutral_smiles, protonated_smiles)
      2. Compute delta_dft_pm7 = dft_pa - pm7_pa per site (signed)

    Returns site-level DataFrame.
    """
    df = pd.read_parquet(feature_dir / "kmeans251_features.parquet")
    log.info(f"k-means features loaded: {len(df)} site rows, "
             f"{df['record_id'].nunique()} molecules")

    # Canonicalize both join keys on both sides. Unmatched rows are excluded;
    # never broadcast a molecule-level best-site target onto an unmatched site.
    df["neutral_smiles"] = df["neutral_smiles"].apply(_canonicalize)
    df["protonated_smiles"] = df["protonated_smiles"].apply(_canonicalize)

    # Join DFT PA on (neutral_smiles, protonated_smiles)
    # Drop existing empty DFT PA columns before merge to avoid _x/_y suffixes
    df = df.drop(columns=[c for c in ["dft_pa_kjmol", "dft_pa_kcalmol",
                                       "delta_dft_exp", "dft_correction"]
                           if c in df.columns])

    df = df.merge(
        dft_site_map[["neutral_smiles", "protonated_smiles", "dft_pa_kjmol", "dft_pa_kcalmol"]],
        on=["neutral_smiles", "protonated_smiles"],
        how="left",
    )

    n_exact = df["dft_pa_kjmol"].notna().sum()
    log.info(f"  Exact (neutral+protonated) join: {n_exact}/{len(df)} rows matched")

    # Drop rows without an exact site label. A molecule-level maximum is not a
    # valid substitute for a missing site-specific target.
    before = len(df)
    df = df[df["dft_pa_kjmol"].notna()].copy()
    if len(df) < before:
        log.warning(f"  Dropped {before - len(df)} rows without DFT PA match")

    # Compute targets — SIGNED correction: PA_pred = PA_PM7 + correction_ML
    df["delta_dft_pm7"]  = df["dft_pa_kjmol"] - df["pm7_pa_kjmol"]   # signed
    df["raw_pm7_error"]  = df["pm7_pa_kjmol"] - df["dft_pa_kjmol"]   # signed (opposite convention)

    log.info(f"  k-means site-level dataset: {len(df)} sites, "
             f"{df['record_id'].nunique()} molecules")
    log.info(f"  delta_dft_pm7 (signed)  mean={df['delta_dft_pm7'].mean():.2f}  "
             f"std={df['delta_dft_pm7'].std():.2f}  "
             f"min={df['delta_dft_pm7'].min():.2f}  max={df['delta_dft_pm7'].max():.2f}  kJ/mol")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Save and report
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame, name: str, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target_dir / f"{name}.parquet", index=False)
    df.to_csv(target_dir / f"{name}.csv", index=False)
    log.info(f"  Saved {name}.*  ({len(df)} rows × {len(df.columns)} cols)")


def parse_args() -> argparse.Namespace:
    """Parse portable input, output, and overwrite settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Scientific data root containing features/ and processed/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Target output directory. Defaults to DATA_DIR/targets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate both target tables without writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement of existing target outputs.",
    )
    return parser.parse_args()


def ensure_output_is_safe(target_dir: Path, force: bool) -> None:
    """Refuse accidental replacement of frozen target artifacts."""
    expected = [
        target_dir / "nist1155_ml.parquet",
        target_dir / "nist1155_ml.csv",
        target_dir / "kmeans251_ml.parquet",
        target_dir / "kmeans251_ml.csv",
        target_dir / "target_report.json",
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing target outputs in {target_dir}: "
            f"{names}. Use --output-dir for a new run or --force explicitly."
        )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    feature_dir = data_dir / "features"
    processed_dir = data_dir / "processed"
    target_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_dir / "targets"
    )
    if not args.dry_run:
        ensure_output_is_safe(target_dir, args.force)

    log.info("Loading DFT maps ...")
    exp_map = load_dft_exp_map(data_dir, processed_dir)
    dft_site_map = load_dft_site_map(processed_dir)

    log.info("\nBuilding NIST targets ...")
    df_nist = build_nist_targets(exp_map, feature_dir)
    if not args.dry_run:
        save(df_nist, "nist1155_ml", target_dir)

    log.info("\nBuilding k-means targets ...")
    df_km = build_kmeans_targets(dft_site_map, feature_dir)
    if not args.dry_run:
        save(df_km, "kmeans251_ml", target_dir)

    # Report
    report = {
        "nist1155": {
            "n_molecules":     int(len(df_nist)),
            "target":          "delta_pm7_exp = exp_pa - pm7_best_pa  kJ/mol  (signed)",
            "target_mean":     float(df_nist["delta_pm7_exp"].mean()),
            "target_std":      float(df_nist["delta_pm7_exp"].std()),
            "target_median":   float(df_nist["delta_pm7_exp"].median()),
            "target_max":      float(df_nist["delta_pm7_exp"].max()),
            "raw_pm7_mae":     float(df_nist["raw_pm7_error"].abs().mean()),
            "n_features":      len([c for c in df_nist.columns
                                    if c not in ["record_id","mol_id","source","dataset",
                                                  "neutral_smiles","protonated_smiles",
                                                  "site_idx","site_name","mordred_geom_source",
                                                  "exp_pa_kjmol","exp_pa_kcalmol",
                                                  "pm7_best_pa_kjmol","pm7_best_pa_kcalmol",
                                                  "delta_pm7_exp","raw_pm7_error",
                                                  "dft_pa_kjmol","dft_pa_kcalmol",
                                                  "delta_dft_exp","delta_pm7_exp","dft_correction"]]),
        },
        "kmeans251": {
            "n_sites":         int(len(df_km)),
            "n_molecules":     int(df_km["record_id"].nunique()),
            "target":          "delta_dft_pm7 = dft_pa - pm7_pa  kJ/mol  (signed, site-level)",
            "target_mean":     float(df_km["delta_dft_pm7"].mean()),
            "target_std":      float(df_km["delta_dft_pm7"].std()),
            "target_median":   float(df_km["delta_dft_pm7"].median()),
            "target_max":      float(df_km["delta_dft_pm7"].max()),
            "raw_pm7_mae":     float(df_km["raw_pm7_error"].abs().mean()),
        },
    }
    if not args.dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        report_path = target_dir / "target_report.json"
        report_path.write_text(json.dumps(report, indent=2))

    print("\n" + "="*60)
    print("  NIST 1155 dataset:")
    print(f"    Molecules      : {report['nist1155']['n_molecules']}")
    print(f"    Target (delta) : {report['nist1155']['target_mean']:.2f} ± "
          f"{report['nist1155']['target_std']:.2f} kJ/mol")
    print(f"    Raw PM7 MAE    : {report['nist1155']['raw_pm7_mae']:.2f} kJ/mol  (baseline)")
    print()
    print("  k-means 251 dataset:")
    print(f"    Sites          : {report['kmeans251']['n_sites']} "
          f"({report['kmeans251']['n_molecules']} molecules)")
    print(f"    Target (delta) : {report['kmeans251']['target_mean']:.2f} ± "
          f"{report['kmeans251']['target_std']:.2f} kJ/mol")
    print(f"    Raw PM7 MAE    : {report['kmeans251']['raw_pm7_mae']:.2f} kJ/mol  (baseline)")
    if args.dry_run:
        print("\n  Dry run: no target files were written.")
    else:
        print(f"\n  Outputs: {target_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()
