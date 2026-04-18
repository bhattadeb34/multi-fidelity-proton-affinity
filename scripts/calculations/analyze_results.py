"""
analyze_results.py
==================
Comprehensive analysis of CV results from train_models.py.

Prints and saves:
  1. Model performance summary (both datasets, sorted by MAE)
  2. Baseline comparison (raw PM7 vs best ML vs Jin & Merz)
  3. Per-fold consistency check
  4. Feature selection statistics
  5. Top features by importance (tree models)

Run
---
  python scripts/analyze_results.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
TARGET_DIR  = PROJECT_DIR / "data" / "targets"

KJMOL_TO_KCAL = 1 / 4.184

SEP  = "=" * 65
SEP2 = "-" * 65


def load_results(dataset: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    d = RESULTS_DIR / dataset
    cv     = json.loads((d / "cv_results.json").read_text())
    preds  = pd.read_csv(d / "predictions.csv")
    mae_df = pd.read_csv(d / "mae_summary.csv")
    return cv, preds, mae_df


def print_section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def analyze_dataset(dataset_name: str, target_name: str, baseline_label: str):
    print_section(f"{dataset_name.upper()}  —  target: {target_name}")

    cv, preds, mae_df = load_results(dataset_name)
    unit = cv.get("unit", "kcal/mol")

    # ---- 1. Model ranking ----
    print(f"\n  Model performance ({unit}, mean ± std across 5 folds):\n")
    print(f"  {'Model':<20} {'MAE (correction)':>18} {'MAE (PA)':>14} {'N features':>12}")
    print(f"  {SEP2}")
    for _, row in mae_df.sort_values("mae_delta_mean").iterrows():
        d = f"{row['mae_delta_mean']:.2f} ± {row['mae_delta_std']:.2f}"
        p = f"{row['mae_pa_mean']:.2f} ± {row['mae_pa_std']:.2f}" if pd.notna(row.get('mae_pa_mean')) else "—"
        n = f"{row['n_features_mean']:.0f}" if pd.notna(row.get('n_features_mean')) else "—"
        print(f"  {row['model']:<20} {d:>18} {p:>14} {n:>12}")

    best = mae_df.sort_values("mae_delta_mean").iloc[0]
    print(f"\n  Best model : {best['model']}")
    print(f"  Best MAE   : {best['mae_delta_mean']:.2f} ± {best['mae_delta_std']:.2f} {unit}")

    # ---- 2. Baseline comparison ----
    print(f"\n  {SEP2}")
    print(f"  Baseline comparison ({unit}):\n")

    # Load target file for baseline stats
    target_file = TARGET_DIR / f"{dataset_name}_ml.parquet"
    if target_file.exists():
        df_target = pd.read_parquet(target_file)

        if "delta_pm7_exp" in df_target.columns:
            raw_err = df_target["raw_pm7_error"].abs() * KJMOL_TO_KCAL
        elif "delta_dft_pm7" in df_target.columns:
            raw_err = df_target["delta_dft_pm7"].abs() * KJMOL_TO_KCAL
        else:
            raw_err = None

        if raw_err is not None:
            print(f"  Raw PM7 MAE (no correction) : {raw_err.mean():.2f} {unit}")

    print(f"  Best ML MAE (correction)    : {best['mae_delta_mean']:.2f} ± {best['mae_delta_std']:.2f} {unit}")

    if dataset_name.startswith("nist"):
        print(f"  Jin & Merz 2025 (direct PA) : 2.47 kcal/mol  [their Voting Regressor, no delta-learning]")
        improvement = raw_err.mean() - best['mae_delta_mean'] if raw_err is not None else None
        if improvement:
            pct = 100 * improvement / raw_err.mean()
            print(f"\n  ML correction improvement   : {improvement:.2f} {unit} ({pct:.0f}% reduction in PM7 error)")

    # ---- 3. Per-fold consistency ----
    print(f"\n  {SEP2}")
    print(f"  Per-fold MAE for best model ({best['model']}):\n")
    best_folds = cv["models"][best["model"]]["mae_delta_per_fold"]
    for i, v in enumerate(best_folds):
        bar = "█" * int(v * 3)
        print(f"    Fold {i+1}: {v:.2f} {unit}  {bar}")
    print(f"    Mean  : {np.mean(best_folds):.2f}  Std: {np.std(best_folds):.2f}")

    # ---- 4. Feature selection stats ----
    print(f"\n  {SEP2}")
    n_feat_per_fold = cv["models"][best["model"]]["n_features_mean"]
    print(f"  Feature selection: ~{n_feat_per_fold:.0f} features selected per fold (from 5295 total)")

    # ---- 5. Top features ----
    feat_imp_path = RESULTS_DIR / dataset_name / "feature_importance.csv"
    if feat_imp_path.exists():
        feat_imp = pd.read_csv(feat_imp_path)
        tree_models = ["ExtraTrees", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM", "CatBoost"]
        for model in tree_models:
            df_m = feat_imp[feat_imp["model"] == model].sort_values("importance", ascending=False).head(10)
            if len(df_m) > 0:
                print(f"\n  Top 10 features — {model}:")
                for _, row in df_m.iterrows():
                    bar = "█" * int(row["importance"] * 200)
                    print(f"    {row['feature']:<45} {row['importance']:.4f}  {bar}")
                break   # just show one tree model for brevity

    # ---- 6. Prediction quality ----
    print(f"\n  {SEP2}")
    print(f"  Prediction diagnostics (all folds, best model):\n")
    best_preds = preds[preds["model"] == best["model"]].copy()
    best_preds["error"] = best_preds["y_pred_delta"] - best_preds["y_true_delta"]
    best_preds["abs_error_kcal"] = best_preds["error"].abs() * KJMOL_TO_KCAL

    print(f"    Mean signed error : {best_preds['error'].mean() * KJMOL_TO_KCAL:+.3f} {unit}  (bias)")
    print(f"    Std of errors     : {best_preds['error'].std() * KJMOL_TO_KCAL:.3f} {unit}")
    print(f"    % within 2 kcal   : {100*(best_preds['abs_error_kcal'] < 2).mean():.1f}%")
    print(f"    % within 5 kcal   : {100*(best_preds['abs_error_kcal'] < 5).mean():.1f}%")
    print(f"    Worst predictions  :")
    worst = best_preds.nlargest(5, "abs_error_kcal")[["neutral_smiles", "y_true_delta", "y_pred_delta", "abs_error_kcal"]]
    for _, row in worst.iterrows():
        print(f"      {row['neutral_smiles'][:40]:<40}  true={row['y_true_delta']*KJMOL_TO_KCAL:.1f}  pred={row['y_pred_delta']*KJMOL_TO_KCAL:.1f}  err={row['abs_error_kcal']:.1f} kcal")


def main():
    print(f"\n{'#'*65}")
    print(f"  PROTON AFFINITY DELTA-LEARNING — RESULTS ANALYSIS")
    print(f"{'#'*65}")

    datasets = [
        ("nist1155",   "PA_exp − PA_PM7 (signed)",    "exp vs PM7"),
        ("kmeans251",  "PA_DFT − PA_PM7 (signed)",    "DFT vs PM7"),
    ]

    for dataset_name, target_name, baseline_label in datasets:
        result_path = RESULTS_DIR / dataset_name / "mae_summary.csv"
        if not result_path.exists():
            print(f"\n  Skipping {dataset_name} — results not found at {result_path}")
            continue
        analyze_dataset(dataset_name, target_name, baseline_label)

    # ---- Cross-dataset summary ----
    print_section("CROSS-DATASET SUMMARY")
    print()
    for dataset_name, _, _ in datasets:
        mae_path = RESULTS_DIR / dataset_name / "mae_summary.csv"
        if not mae_path.exists():
            continue
        mae_df = pd.read_csv(mae_path)
        best = mae_df.sort_values("mae_delta_mean").iloc[0]
        print(f"  {dataset_name:<15} best={best['model']:<18} "
              f"MAE={best['mae_delta_mean']:.2f}±{best['mae_delta_std']:.2f} kcal/mol")

    print(f"\n  Note: MAE delta = MAE PA (signed correction, mathematically consistent)")
    print(f"  PA_pred = PA_PM7 + correction_ML\n")


if __name__ == "__main__":
    main()
