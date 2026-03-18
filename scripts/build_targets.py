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
  Target: delta_pm7_exp = |exp_pa_kjmol - pm7_best_pa_kjmol|
  Also records:
    pm7_best_pa_kjmol   — max PM7 PA across all sites for that molecule
    exp_pa_kjmol        — NIST experimental PA
    raw_pm7_error       — pm7_best_pa - exp_pa  (signed, for baseline plots)

k-means 251-molecule dataset  (kmeans251_features.parquet)
  Source: PM7 calculations + B3LYP/def2-TZVP DFT (from folder records)
  Level:  site-level — one row per (molecule, site)
  Target: delta_dft_pm7 = |dft_pa_site_kjmol - pm7_pa_site_kjmol|
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

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR.parent / "data"
FEAT_DIR   = DATA_DIR / "features"
TARGET_DIR = DATA_DIR / "targets"
PROC_DIR   = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# Load processed datasets
# ---------------------------------------------------------------------------

def load_dft_exp_map() -> dict[str, float]:
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
    excel_path = DATA_DIR / "Merz-and-hogni-dataset.xlsx"
    if not excel_path.exists():
        # Fallback: try to load from DFT JSON records (only 1 of 1185 in test env)
        log.warning(f"Jin & Merz Excel not found at {excel_path}")
        log.warning("Falling back to DFT JSON records (incomplete — use Excel for full dataset)")
        dataset = json.loads((PROC_DIR / "dataset.json").read_text())
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


def load_dft_site_map() -> pd.DataFrame:
    """
    Build a DataFrame of DFT folder site-level PAs:
    (neutral_smiles, protonated_smiles) -> dft_pa_kjmol

    Used to join DFT site PAs onto k-means PM7 site rows.
    """
    dataset = json.loads((PROC_DIR / "dataset.json").read_text())
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
                "neutral_smiles":    neu_smi,
                "protonated_smiles": site.get("protonated_smiles", ""),
                "dft_pa_kjmol":      float(pa),
                "dft_pa_kcalmol":    float(pa) * 0.239006,
            })
    df = pd.DataFrame(rows)
    log.info(f"DFT site map: {len(df)} sites from {df['neutral_smiles'].nunique()} molecules")
    return df


# ---------------------------------------------------------------------------
# NIST dataset — molecule-level target
# ---------------------------------------------------------------------------

def build_nist_targets(exp_map: dict[str, float]) -> pd.DataFrame:
    """
    From nist1185_features.parquet:
      1. Join exp_pa onto each row via neutral_smiles
      2. For each molecule, select the best-site row (max pm7_pa_kjmol)
      3. Compute delta_pm7_exp = |exp_pa - pm7_best_pa|

    Returns molecule-level DataFrame (one row per molecule).
    """
    df = pd.read_parquet(FEAT_DIR / "nist1185_features.parquet")
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

def build_kmeans_targets(dft_site_map: pd.DataFrame) -> pd.DataFrame:
    """
    From kmeans251_features.parquet:
      1. Join DFT site PA via (neutral_smiles, protonated_smiles)
      2. Compute delta_dft_pm7 = |dft_pa - pm7_pa| per site

    Returns site-level DataFrame.
    """
    df = pd.read_parquet(FEAT_DIR / "kmeans251_features.parquet")
    log.info(f"k-means features loaded: {len(df)} site rows, "
             f"{df['record_id'].nunique()} molecules")

    # Join DFT PA on (neutral_smiles, protonated_smiles)
    # PM7 protonated SMILES include explicit H; DFT SMILES may not — try exact first,
    # then fall back to neutral_smiles-only join using best DFT site
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

    # Fallback for unmatched rows: join on neutral_smiles, take max DFT PA for that mol
    if n_exact < len(df):
        dft_best = (dft_site_map.groupby("neutral_smiles")["dft_pa_kjmol"]
                    .max().reset_index()
                    .rename(columns={"dft_pa_kjmol": "dft_pa_kjmol_fallback"}))
        df = df.merge(dft_best, on="neutral_smiles", how="left")
        mask = df["dft_pa_kjmol"].isna()
        df.loc[mask, "dft_pa_kjmol"] = df.loc[mask, "dft_pa_kjmol_fallback"]
        df.loc[mask, "dft_pa_kcalmol"] = df.loc[mask, "dft_pa_kjmol_fallback"] * 0.239006
        df = df.drop(columns=["dft_pa_kjmol_fallback"])
        n_fallback = mask.sum()
        log.info(f"  Fallback (neutral only, best DFT site): {n_fallback} rows filled")

    # Drop rows still without DFT PA
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

def save(df: pd.DataFrame, name: str):
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TARGET_DIR / f"{name}.parquet", index=False)
    df.to_csv(TARGET_DIR / f"{name}.csv", index=False)
    log.info(f"  Saved {name}.*  ({len(df)} rows × {len(df.columns)} cols)")


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading DFT maps ...")
    exp_map      = load_dft_exp_map()
    dft_site_map = load_dft_site_map()

    log.info("\nBuilding NIST targets ...")
    df_nist = build_nist_targets(exp_map)
    save(df_nist, "nist1155_ml")

    log.info("\nBuilding k-means targets ...")
    df_km = build_kmeans_targets(dft_site_map)
    save(df_km, "kmeans251_ml")

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
    report_path = TARGET_DIR / "target_report.json"
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
    print(f"\n  Outputs: {TARGET_DIR.relative_to(SCRIPT_DIR.parent)}/")
    print("="*60)


if __name__ == "__main__":
    main()
