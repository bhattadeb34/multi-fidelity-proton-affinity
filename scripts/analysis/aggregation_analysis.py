"""
aggregation_analysis.py
=======================
Post-hoc analysis of max vs mean aggregation for molecular PA
from site-level CV predictions.

Reads predictions.csv from a k-means CV run, groups sites by molecule,
and compares max-aggregation vs mean-aggregation for deriving molecular PA.

Usage:
  python aggregation_analysis.py --results-dir ../../results/kmeans251
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KJMOL_TO_KCAL = 1 / 4.184


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=None,
                        help="Model to analyze (default: best single model in cv_results.json)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    model_name = args.model
    if model_name is None:
        cv = json.loads((results_dir / "cv_results.json").read_text())
        eligible = {
            k: v for k, v in cv["models"].items()
            if k not in {"PM7+bias", "VotingEnsemble"}
            and v.get("mae_pa_mean") is not None
        }
        model_name = min(eligible, key=lambda k: eligible[k]["mae_pa_mean"])
    preds = pd.read_csv(results_dir / "predictions.csv")
    preds = preds[preds["model"] == model_name].copy()

    if preds.empty:
        print(f"No predictions found for model '{model_name}'")
        return

    print(f"\nAggregation analysis for {model_name}")
    print(f"Total site predictions: {len(preds)}")
    print(f"Unique molecules: {preds['record_id'].nunique()}")
    print(f"Folds: {preds['fold'].nunique()}")

    # Group by (fold, molecule) to get per-molecule aggregations
    rows = []
    for (fold, mol_id), grp in preds.groupby(["fold", "record_id"]):
        n_sites = len(grp)
        # Corrected targets are site-specific.  Molecular reference PA is the
        # maximum across DFT protonation sites, not the first site's value.
        pa_true = grp["pa_true"].max()
        pa_pm7_sites = grp["pa_pm7"].values
        pa_pred_sites = grp["pa_pred"].values

        pa_pred_max = pa_pred_sites.max()
        pa_pred_mean = pa_pred_sites.mean()
        pa_pm7_max = pa_pm7_sites.max()
        pa_pm7_mean = pa_pm7_sites.mean()

        rows.append({
            "fold": fold,
            "record_id": mol_id,
            "n_sites": n_sites,
            "pa_true": pa_true,
            "pa_pred_max": pa_pred_max,
            "pa_pred_mean": pa_pred_mean,
            "pa_pm7_max": pa_pm7_max,
            "pa_pm7_mean": pa_pm7_mean,
            "error_max": (pa_pred_max - pa_true) * KJMOL_TO_KCAL,
            "error_mean": (pa_pred_mean - pa_true) * KJMOL_TO_KCAL,
            "error_pm7_max": (pa_pm7_max - pa_true) * KJMOL_TO_KCAL,
        })

    df = pd.DataFrame(rows)

    # Per-fold MAEs
    print(f"\n{'Fold':<6} {'Max-agg MAE':>14} {'Mean-agg MAE':>14} {'PM7-max MAE':>14} {'N mols':>8}")
    print("-" * 60)
    for fold in sorted(df["fold"].unique()):
        fdf = df[df["fold"] == fold]
        mae_max = fdf["error_max"].abs().mean()
        mae_mean = fdf["error_mean"].abs().mean()
        mae_pm7 = fdf["error_pm7_max"].abs().mean()
        print(f"{fold:<6} {mae_max:>14.2f} {mae_mean:>14.2f} {mae_pm7:>14.2f} {len(fdf):>8}")

    # Overall
    mae_max_all = df["error_max"].abs().mean()
    mae_mean_all = df["error_mean"].abs().mean()
    mae_pm7_all = df["error_pm7_max"].abs().mean()
    bias_max = df["error_max"].mean()
    bias_mean = df["error_mean"].mean()

    print("-" * 60)
    print(f"{'All':<6} {mae_max_all:>14.2f} {mae_mean_all:>14.2f} {mae_pm7_all:>14.2f} {len(df):>8}")
    print(f"\nBias (mean signed error, kcal/mol):")
    print(f"  Max-aggregation:  {bias_max:+.2f}")
    print(f"  Mean-aggregation: {bias_mean:+.2f}")

    # Breakdown by number of sites
    print(f"\nBreakdown by number of sites per molecule:")
    print(f"{'k':>4} {'N mols':>8} {'Max MAE':>10} {'Mean MAE':>10} {'Max bias':>10} {'Mean bias':>10}")
    print("-" * 55)
    for k in sorted(df["n_sites"].unique()):
        kdf = df[df["n_sites"] == k]
        print(f"{k:>4} {len(kdf):>8} "
              f"{kdf['error_max'].abs().mean():>10.2f} "
              f"{kdf['error_mean'].abs().mean():>10.2f} "
              f"{kdf['error_max'].mean():>10.2f} "
              f"{kdf['error_mean'].mean():>10.2f}")

    # Save
    out_path = results_dir / "aggregation_analysis.csv"
    df.to_csv(out_path, index=False)
    summary = {
        "model": model_name,
        "n_molecules": int(len(df)),
        "molecular_reference": "maximum site-specific DFT PA",
        "mae_max_aggregation_kcalmol": float(mae_max_all),
        "mae_mean_aggregation_kcalmol": float(mae_mean_all),
        "mae_pm7_max_kcalmol": float(mae_pm7_all),
        "bias_max_aggregation_kcalmol": float(bias_max),
        "bias_mean_aggregation_kcalmol": float(bias_mean),
    }
    (results_dir / "aggregation_analysis_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
