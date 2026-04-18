"""
verify_features.py
==================
Sanity-check that the ML-ready parquet files contain the feature groups
claimed in the manuscript (MACCS/Morgan/RDKit/Mordred/PM7/site).

Usage (from repo root):
  python scripts/analysis/verify_features.py

Or with an explicit data dir:
  python scripts/analysis/verify_features.py --data-dir data/targets
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent


NON_FEATURE_COLS = {
    "record_id", "mol_id", "source", "dataset",
    "neutral_smiles", "protonated_smiles",
    "site_idx", "site_name", "mordred_geom_source",
    "exp_pa_kjmol", "exp_pa_kcalmol",
    "dft_pa_kjmol", "dft_pa_kcalmol",
    "pm7_pa_kjmol", "pm7_pa_kcalmol",
    "pm7_best_pa_kjmol", "pm7_best_pa_kcalmol",
    "delta_dft_exp", "delta_pm7_exp", "dft_correction",
    "delta_dft_pm7", "raw_pm7_error",
}


def categorize_features(cols):
    cats = {
        "MACCS keys":               [],
        "Morgan fingerprints":      [],
        "RDKit (neutral)":          [],
        "RDKit (protonated)":       [],
        "RDKit (delta)":            [],
        "Mordred (neutral)":        [],
        "Mordred (protonated)":     [],
        "Mordred (delta)":          [],
        "PM7 physical (neutral)":   [],
        "PM7 physical (protonated)":[],
        "PM7 point_group":          [],
        "Site descriptors":         [],
        "Other":                    [],
    }
    for c in cols:
        if c.startswith("maccs_"):
            cats["MACCS keys"].append(c)
        elif c.startswith("morgan_"):
            cats["Morgan fingerprints"].append(c)
        elif "rdkit" in c and c.startswith("neutral_"):
            cats["RDKit (neutral)"].append(c)
        elif "rdkit" in c and c.startswith("protonated_"):
            cats["RDKit (protonated)"].append(c)
        elif "rdkit" in c and c.startswith("delta_"):
            cats["RDKit (delta)"].append(c)
        elif "mordred" in c and c.startswith("neutral_"):
            cats["Mordred (neutral)"].append(c)
        elif "mordred" in c and c.startswith("protonated_"):
            cats["Mordred (protonated)"].append(c)
        elif "mordred" in c and c.startswith("delta_"):
            cats["Mordred (delta)"].append(c)
        elif "pm7" in c and "point_group" in c:
            cats["PM7 point_group"].append(c)
        elif "pm7" in c and c.startswith("neutral_"):
            cats["PM7 physical (neutral)"].append(c)
        elif "pm7" in c and c.startswith("protonated_"):
            cats["PM7 physical (protonated)"].append(c)
        elif any(x in c for x in ["site_", "element_", "n_sites"]):
            cats["Site descriptors"].append(c)
        else:
            cats["Other"].append(c)
    return cats


def analyze_dataset(path, label):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  File: {path}")
    print(f"{'='*65}")

    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Feature columns only
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    # Also remove any remaining label-like columns
    feat_cols = [c for c in feat_cols if not any(
        c == x for x in {"delta_pm7_exp", "delta_dft_pm7", "raw_pm7_error",
                          "exp_pa_kjmol", "exp_pa_kcalmol",
                          "dft_pa_kjmol", "dft_pa_kcalmol",
                          "pm7_pa_kjmol", "pm7_pa_kcalmol",
                          "pm7_best_pa_kjmol", "pm7_best_pa_kcalmol"})]

    print(f"  Total feature columns: {len(feat_cols)}")

    cats = categorize_features(feat_cols)

    print(f"\n  {'Category':<35} {'Count':>6}  {'Example columns'}")
    print(f"  {'-'*75}")
    total = 0
    for cat, cols in cats.items():
        if cols:
            examples = ", ".join(cols[:3])
            if len(cols) > 3:
                examples += f", ... (+{len(cols)-3} more)"
            print(f"  {cat:<35} {len(cols):>6}  {examples}")
            total += len(cols)

    # Summary matching manuscript claims
    maccs   = len(cats["MACCS keys"])
    morgan  = len(cats["Morgan fingerprints"])
    rdkit_n = len(cats["RDKit (neutral)"])
    rdkit_p = len(cats["RDKit (protonated)"])
    rdkit_d = len(cats["RDKit (delta)"])
    rdkit_t = rdkit_n + rdkit_p + rdkit_d
    mord_n  = len(cats["Mordred (neutral)"])
    mord_p  = len(cats["Mordred (protonated)"])
    mord_d  = len(cats["Mordred (delta)"])
    mord_t  = mord_n + mord_p + mord_d
    pm7_n   = len(cats["PM7 physical (neutral)"])
    pm7_p   = len(cats["PM7 physical (protonated)"])
    pm7_pg  = len(cats["PM7 point_group"])
    pm7_t   = pm7_n + pm7_p + pm7_pg
    site    = len(cats["Site descriptors"])
    other   = len(cats["Other"])
    grand   = maccs + morgan + rdkit_t + mord_t + pm7_t + site + other

    print(f"\n  {'─'*65}")
    print(f"  SUMMARY vs MANUSCRIPT CLAIMS")
    print(f"  {'─'*65}")
    print(f"  {'Feature group':<35} {'Actual':>8}  {'Manuscript':>10}  Match?")
    print(f"  {'-'*65}")

    claims = {
        "MACCS keys":                    (maccs,  167),
        "Morgan fingerprints":           (morgan, 1024),
        "RDKit (neutral only)":          (rdkit_n, 140),
        "RDKit (protonated only)":       (rdkit_p, 140),
        "RDKit (delta only)":            (rdkit_d, 140),
        "RDKit total (3 states)":        (rdkit_t, 420),
        "Mordred (neutral only)":        (mord_n, 1826),
        "Mordred (protonated only)":     (mord_p, 1826),
        "Mordred total (2 states)":      (mord_t, 3652),
        "PM7 physical neutral":          (pm7_n,  13),
        "PM7 physical protonated":       (pm7_p,  13),
        "PM7 point_group one-hot":       (pm7_pg, 0),
        "PM7 total (all)":               (pm7_t,  26),
        "Site descriptors":              (site,   6),
        "GRAND TOTAL":                   (grand,  5295),
    }

    for label2, (actual, claimed) in claims.items():
        match = "✓" if actual == claimed else f"✗ (manuscript says {claimed})"
        print(f"  {label2:<35} {actual:>8}  {claimed:>10}  {match}")

    # Print all PM7 columns explicitly
    print(f"\n  PM7 neutral columns ({pm7_n}):")
    for c in sorted(cats["PM7 physical (neutral)"]):
        print(f"    {c}")
    print(f"\n  PM7 protonated columns ({pm7_p}):")
    for c in sorted(cats["PM7 physical (protonated)"]):
        print(f"    {c}")
    print(f"\n  PM7 point_group columns ({pm7_pg}):")
    for c in sorted(cats["PM7 point_group"]):
        print(f"    {c}")

    if other > 0:
        print(f"\n  Other/unclassified columns ({other}):")
        for c in cats["Other"]:
            print(f"    {c}")

    return grand


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None,
                        help="Directory containing nist1155_ml.parquet "
                             "and kmeans251_ml.parquet")
    args = parser.parse_args()

    # Try to find the data files
    search_dirs = [
        args.data_dir,
        ".",
        PROJECT_DIR,
        PROJECT_DIR / "data",
        PROJECT_DIR / "data" / "targets",
        PROJECT_DIR / "data" / "screening" / "iter1",
    ]

    found = {}
    for d in search_dirs:
        if d is None:
            continue
        base = Path(d)
        for fname in ["nist1155_ml.parquet", "nist1155_ml.csv",
                      "kmeans251_ml.parquet", "kmeans251_ml.csv"]:
            p = base / fname
            if p.exists():
                key = "nist" if "nist" in fname else "kmeans"
                if key not in found:
                    found[key] = p
                    print(f"Found {key}: {p}")

    if not found:
        print("\nERROR: Could not find nist1155_ml or kmeans251_ml files.")
        print("Searched in:", [d for d in search_dirs if d])
        print("\nPlease run with:")
        print("  python verify_features.py --data-dir /path/to/your/data")
        print("\nOr list files manually:")
        print("  find <repo_root> -name '*.parquet' 2>/dev/null")
        print("  find <repo_root> -name '*ml*.csv' 2>/dev/null")
        return

    for key, path in found.items():
        analyze_dataset(path, f"{key.upper()} dataset — {path.name}")

    print("\n" + "="*65)
    print("  Done. Check ✗ rows above for discrepancies with manuscript.")
    print("="*65)


if __name__ == "__main__":
    main()
