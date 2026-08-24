"""
train_models.py
===============
5-fold cross-validated surrogate model training with in-fold feature selection.

Delta-learning framework
------------------------
  Target:   correction = PA_high_fidelity - PA_PM7  (signed)
  Predict:  PA_pred = PA_PM7 + correction_ML
  Evaluate: MAE(PA_pred, PA_high_fidelity)  and  MAE(correction_ML, correction_true)
  Note: signed correction means MAE_delta ≈ MAE_PA when PM7 baseline is consistent

Datasets
--------
  nist1155  — molecule-level, target = exp_pa - pm7_best_pa  (kJ/mol)
  kmeans251 — site-level,     target = dft_pa - pm7_pa       (kJ/mol)

Feature selection (per fold, train split only)
----------------------------------------------
  Stage 1: Variance filter     threshold = 0.01
  Stage 2: Target-corr pool    top 1,500 by training-fold |r|
  Stage 3: Correlation filter  Pearson |r| > 0.95, keep higher target corr
  Stage 4: LassoCV             alpha minimizing inner 5-fold CV MSE

Models (15)
-----------
  Linear:   Ridge, Lasso, ElasticNet, BayesianRidge, SVR
  Tree:     DecisionTree, RandomForest, ExtraTrees, GradientBoosting,
            AdaBoost, XGBoost, LightGBM, CatBoost
  Neural:   MLP
  Prob:     GPR  (skipped above 500 training rows for cubic-cost control)

Outputs  (../results/{dataset_name}/)
-------
  cv_results.json         — MAE mean ± std per model, feature counts per fold, selected features
  predictions.csv         — per-fold raw predictions (mol/site id, fold, true, pred,
                            pa_pm7, pa_pred, pa_true for final PA MAE computation)
  mae_summary.csv         — clean table: model, mae_delta_mean, mae_delta_std,
                            mae_pa_mean, mae_pa_std
  feature_importance.csv  — mean feature importances across folds (tree models)
  baseline_data.csv       — raw PM7 vs exp/DFT for baseline comparison plots

Usage
-----
  python train_models.py --dataset nist
  python train_models.py --dataset kmeans
  python train_models.py --dataset all
  python train_models.py --dataset nist --n-folds 5 --seed 42
"""

import json
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error

from model_factory import ModelFactory

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR   = Path(__file__).resolve().parent
KJMOL_TO_KCAL = 1 / 4.184   # all MAEs reported in kcal/mol for comparison with literature
PROJECT_DIR  = SCRIPT_DIR.parent.parent
DATA_DIR     = next((p for p in (PROJECT_DIR / "data", PROJECT_DIR.parent / "data") if p.exists()), PROJECT_DIR / "data")
TARGET_DIR   = DATA_DIR / "targets"
RESULTS_DIR  = PROJECT_DIR / "results"

MODEL_FACTORY = ModelFactory(random_state=42, n_estimators=200, gpr_max_samples=500)
DEFAULT_TARGET_CORRELATION_POOL_MAX = 1_500


# ---------------------------------------------------------------------------
# Non-feature columns  (never used as ML features)
# ---------------------------------------------------------------------------

NON_FEATURE_COLS = {
    "record_id", "mol_id", "source", "dataset",
    "neutral_smiles", "protonated_smiles",
    "site_idx", "site_name", "mordred_geom_source",
    # label columns
    "exp_pa_kjmol", "exp_pa_kcalmol",
    "dft_pa_kjmol", "dft_pa_kcalmol",
    "pm7_pa_kjmol", "pm7_pa_kcalmol",
    "pm7_best_pa_kjmol", "pm7_best_pa_kcalmol",
    "delta_dft_exp", "delta_pm7_exp", "dft_correction",
    # target columns added by build_targets.py
    "delta_pm7_exp", "delta_dft_pm7",
    "raw_pm7_error",
}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_models(n_train: int) -> dict:
    """Return a fresh estimator registry for one outer fold."""
    return MODEL_FACTORY.build(n_train=n_train, logger=log)


# ---------------------------------------------------------------------------
# Feature selection (applied to train split only)
# ---------------------------------------------------------------------------

def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    lasso_cv_folds: int = 5,
    target_correlation_pool_max: int = DEFAULT_TARGET_CORRELATION_POOL_MAX,
) -> tuple[np.ndarray, list[str]]:
    """
    Four-stage hierarchical feature selection on training data only.

    Stage 1 — Variance filter
    Stage 2 — Retain a configured number of features by absolute training-target
              correlation (a computational feature-pool limit)
    Stage 3 — Correlation filter (keep higher target correlation)
    Stage 4 — LassoCV at the alpha minimizing inner-CV MSE

    Returns (X_selected, selected_feature_names).
    """
    names = list(feature_names)
    X = X_train.copy()

    # Mordred emits descriptor columns that can be unavailable for every
    # molecule in a fold. Remove these explicitly before median imputation;
    # relying on downstream variance-filter warnings is equivalent
    # numerically but is version-dependent and obscures the feature count.
    X[~np.isfinite(X)] = np.nan
    usable = ~np.isnan(X).all(axis=0)
    if not usable.all():
        log.debug(f"    Removed {(~usable).sum()} all-missing features")
        X = X[:, usable]
        names = [name for name, keep in zip(names, usable) if keep]
    if not names:
        return X, names

    # Replace NaN with column median (computed on train only)
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    # Stage 1: Variance filter
    vt = VarianceThreshold(threshold=variance_threshold)
    X = vt.fit_transform(X)
    names = [n for n, keep in zip(names, vt.get_support()) if keep]
    log.debug(f"    After variance filter: {len(names)}")

    # Stages 2 + 3: construct a computationally tractable feature pool using
    # training-fold target correlations, then remove redundant features.
    if len(names) > 1:
        Xs = (X - np.nanmean(X, axis=0)) / np.where(np.nanstd(X, axis=0) > 0, np.nanstd(X, axis=0), 1.0)
        ys = (y_train - y_train.mean()) / (y_train.std() if y_train.std() > 0 else 1.0)
        target_corr = np.abs(Xs.T @ ys) / len(ys)
        target_corr = np.nan_to_num(target_corr)

        if target_correlation_pool_max < 1:
            raise ValueError("target_correlation_pool_max must be positive")
        order = np.argsort(-target_corr)[:target_correlation_pool_max]
        C = np.abs(np.corrcoef(Xs[:, order], rowvar=False))
        C = np.nan_to_num(C)
        np.fill_diagonal(C, 0)

        dropped = np.zeros(len(order), dtype=bool)
        for i in range(len(order)):
            if dropped[i]:
                continue
            hits = np.flatnonzero((C[i] > correlation_threshold) & (~dropped))
            hits = hits[hits > i]
            dropped[hits] = True

        surviving = order[~dropped]
        keep_mask = np.zeros(len(names), dtype=bool)
        keep_mask[surviving] = True
        X = X[:, keep_mask]
        names = [n for n, k in zip(names, keep_mask) if k]
        log.debug(f"    After correlation filter: {len(names)}")

    # Stage 4: retain nonzero coefficients at the LassoCV alpha minimizing
    # mean inner-CV MSE. A sensitivity analysis showed that this is equivalent
    # to the former 1-SD + minimum-20 fallback for the corrected k-means run.
    if len(names) > 0:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Inner CV to find alpha path
        lasso_cv = LassoCV(cv=lasso_cv_folds, max_iter=10000, n_jobs=-1, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso_cv.fit(X_scaled, y_train)

        selected_mask = lasso_cv.coef_ != 0

        # Numerical fallback only. This is not activated in either reported
        # dataset and does not impose a minimum feature-count hyperparameter.
        if selected_mask.sum() == 0:
            top_idx = np.argsort(np.abs(lasso_cv.coef_))[-1:]
            selected_mask = np.zeros(len(names), dtype=bool)
            selected_mask[top_idx] = True

        X = X[:, selected_mask]
        names = [n for n, s in zip(names, selected_mask) if s]
        log.debug(f"    After minimum-CV-error Lasso: {len(names)}")

    return X, names


def apply_feature_selection(
    X_test: np.ndarray,
    X_train_full: np.ndarray,
    feature_names_full: list[str],
    selected_names: list[str],
) -> np.ndarray:
    """
    Apply previously selected feature names to the test set.
    Also imputes NaN with train column medians.
    """
    name_to_idx = {n: i for i, n in enumerate(feature_names_full)}
    sel_idx = [name_to_idx[n] for n in selected_names]
    X = X_test[:, sel_idx].copy()
    X[~np.isfinite(X)] = np.nan

    # Impute with train medians
    train_sel = X_train_full[:, sel_idx].copy()
    train_sel[~np.isfinite(train_sel)] = np.nan
    col_medians = np.nanmedian(train_sel, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    return X


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------

def run_cv(
    df: pd.DataFrame,
    target_col: str,
    pa_pm7_col: str,
    pa_true_col: str,
    dataset_name: str,
    n_folds: int = 5,
    seed: int = 42,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    target_correlation_pool_max: int = DEFAULT_TARGET_CORRELATION_POOL_MAX,
    unit_scale: float = KJMOL_TO_KCAL,
    unit_label: str = "kcal/mol",
    group_col: str | None = None,
    model_names: set[str] | None = None,
) -> dict:
    """
    Run cross-validation with fold-local feature selection.

    Parameters
    ----------
    df            : ML-ready DataFrame (from build_targets.py output)
    target_col    : column name for the delta target to model
    pa_pm7_col    : column name for PM7 PA (to compute PA_pred = PM7 + delta)
    pa_true_col   : column name for the high-fidelity PA (for final PA MAE)
    dataset_name  : used for logging and output paths
    group_col     : if set, use GroupKFold with this column as group key
    model_names   : optional estimator allowlist for a focused sensitivity run
    """
    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS
                    and c != target_col and c != pa_pm7_col and c != pa_true_col
                    and not c.startswith("delta_") and not c.startswith("raw_")]
    # Also exclude any remaining label-like columns
    feature_cols = [c for c in feature_cols if c not in {
        "exp_pa_kjmol","exp_pa_kcalmol","dft_pa_kjmol","dft_pa_kcalmol",
        "pm7_pa_kjmol","pm7_pa_kcalmol","pm7_best_pa_kjmol","pm7_best_pa_kcalmol",
    }]

    log.info(f"\n{'='*55}")
    log.info(f"  Dataset      : {dataset_name}")
    log.info(f"  Samples      : {len(df)}")
    log.info(f"  Target       : {target_col}")
    log.info(f"  Features     : {len(feature_cols)}")
    log.info(f"  Folds        : {n_folds}")
    log.info(f"{'='*55}")

    X_all = df[feature_cols].values.astype(np.float64)
    y_all = df[target_col].values.astype(np.float64)
    pa_pm7_all  = df[pa_pm7_col].values if pa_pm7_col in df.columns else None
    pa_true_all = df[pa_true_col].values if pa_true_col in df.columns else None

    if group_col and group_col not in df.columns:
        raise ValueError(
            f"Requested GroupKFold column {group_col!r} is absent from "
            f"dataset {dataset_name!r}; refusing to fall back to row-wise KFold."
        )

    if group_col:
        groups = df[group_col].values
        kf = GroupKFold(n_splits=n_folds)
        splitter = kf.split(X_all, y_all, groups=groups)
        log.info(f"  Using GroupKFold on '{group_col}' "
                 f"({len(np.unique(groups))} groups)")
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splitter = kf.split(X_all)

    # Storage
    all_pred_rows   = []
    fold_results    = {m: {"mae_delta": [], "mae_pa": [],
                            "n_features": []} for m in []}
    feat_importance = {}

    # Store selected features for each fold
    fold_selected_features = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter):
        log.info(f"\n  Fold {fold_idx+1}/{n_folds}  "
                 f"(train={len(train_idx)}, test={len(test_idx)})")

        X_train_raw, X_test_raw = X_all[train_idx], X_all[test_idx]
        y_train,     y_test     = y_all[train_idx], y_all[test_idx]

        # PM7+bias baseline: predict delta = mean(train delta) for all test
        train_mean_delta = y_train.mean()
        y_pred_bias = np.full_like(y_test, train_mean_delta)
        mae_delta_bias = mean_absolute_error(y_test, y_pred_bias) * unit_scale
        if pa_pm7_all is not None and pa_true_all is not None:
            pa_pred_bias = pa_pm7_all[test_idx] + y_pred_bias
            mae_pa_bias = mean_absolute_error(pa_true_all[test_idx], pa_pred_bias) * unit_scale
        else:
            mae_pa_bias = np.nan
        if "PM7+bias" not in fold_results:
            fold_results["PM7+bias"] = {"mae_delta": [], "mae_pa": [], "n_features": []}
        fold_results["PM7+bias"]["mae_delta"].append(mae_delta_bias)
        fold_results["PM7+bias"]["mae_pa"].append(mae_pa_bias)
        fold_results["PM7+bias"]["n_features"].append(0)

        # Feature selection on train split only
        log.info(f"    Feature selection ...")
        X_train_sel, sel_names = select_features(
            X_train_raw, y_train, feature_cols,
            variance_threshold, correlation_threshold,
            target_correlation_pool_max=target_correlation_pool_max,
        )
        
        # Track the selected features for this fold
        fold_selected_features.append(sel_names)

        X_test_sel = apply_feature_selection(
            X_test_raw, X_train_raw, feature_cols, sel_names
        )
        log.info(f"    Selected {len(sel_names)} features")

        # Get models for this fold's train size
        models = get_models(n_train=len(train_idx))
        if model_names is not None:
            unknown = sorted(model_names - models.keys())
            if unknown:
                raise ValueError(f"Unknown requested model names: {unknown}")
            models = {
                name: model for name, model in models.items() if name in model_names
            }

        # Initialise fold result storage
        for mname in models:
            if mname not in fold_results:
                fold_results[mname] = {"mae_delta": [], "mae_pa": [],
                                        "n_features": []}

        for mname, model in models.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train_sel, y_train)
                    y_pred = model.predict(X_test_sel)

                # No clipping — signed correction can be positive or negative
                # PA_pred = PA_PM7 + correction_ML  works correctly with signed values

                # MAE on delta — convert kJ/mol -> kcal/mol for reporting
                mae_delta = mean_absolute_error(y_test, y_pred) * unit_scale

                # MAE on PA (PA_pred = PA_PM7 + delta_ML)
                if pa_pm7_all is not None and pa_true_all is not None:
                    pa_pm7_test  = pa_pm7_all[test_idx]
                    pa_true_test = pa_true_all[test_idx]
                    pa_pred      = pa_pm7_test + y_pred
                    mae_pa       = mean_absolute_error(pa_true_test, pa_pred) * unit_scale
                else:
                    pa_pred      = np.full_like(y_pred, np.nan)
                    pa_pm7_test  = np.full_like(y_pred, np.nan)
                    pa_true_test = np.full_like(y_pred, np.nan)
                    mae_pa       = np.nan

                fold_results[mname]["mae_delta"].append(mae_delta)
                fold_results[mname]["mae_pa"].append(mae_pa)
                fold_results[mname]["n_features"].append(len(sel_names))

                # Save per-sample predictions
                for i, idx in enumerate(test_idx):
                    row_id = df.iloc[idx].get("record_id",
                             df.iloc[idx].get("neutral_smiles", str(idx)))
                    all_pred_rows.append({
                        "fold":           fold_idx + 1,
                        "model":          mname,
                        "sample_idx":     int(idx),
                        "record_id":      str(row_id),
                        "neutral_smiles": df.iloc[idx].get("neutral_smiles", ""),
                        "y_true_delta":   float(y_test[i]),
                        "y_pred_delta":   float(y_pred[i]),
                        "pa_pm7":         float(pa_pm7_test[i]) if pa_pm7_all is not None else np.nan,
                        "pa_pred":        float(pa_pred[i]),
                        "pa_true":        float(pa_true_test[i]) if pa_true_all is not None else np.nan,
                    })

                # Feature importances (tree models)
                raw_model = (model.named_steps.get("model") or model
                             if hasattr(model, "named_steps") else model)
                if hasattr(raw_model, "feature_importances_"):
                    imp = raw_model.feature_importances_
                    if mname not in feat_importance:
                        feat_importance[mname] = {}
                    for fname, fval in zip(sel_names, imp):
                        feat_importance[mname][fname] = (
                            feat_importance[mname].get(fname, 0) + fval / n_folds
                        )

            except Exception as e:
                log.warning(f"    {mname} failed fold {fold_idx+1}: {e}")
                fold_results[mname]["mae_delta"].append(np.nan)
                fold_results[mname]["mae_pa"].append(np.nan)
                fold_results[mname]["n_features"].append(0)

        log.info(f"    Fold {fold_idx+1} done  "
                 + "  ".join(f"{m}={np.mean(fold_results[m]['mae_delta']):.2f} {unit_label}"
                              for m in list(models)[:4]))

    # Aggregate results
    summary_rows = []
    
    # Store the tracked features into the output dictionary
    cv_out = {
        "dataset": dataset_name,
        "unit": unit_label,
        "cv_strategy": "GroupKFold" if group_col else "KFold",
        "group_col": group_col,
        "model_allowlist": sorted(model_names) if model_names else None,
        "feature_selection": {
            "variance_threshold": variance_threshold,
            "target_correlation_pool_max": target_correlation_pool_max,
            "redundancy_correlation_threshold": correlation_threshold,
            "lasso_alpha_rule": "minimum mean inner-CV MSE",
            "lasso_inner_cv_folds": 5,
        },
        "selected_features_per_fold": fold_selected_features,
        "models": {}
    }

    for mname, res in fold_results.items():
        mae_d = [v for v in res["mae_delta"] if not np.isnan(v)]
        mae_p = [v for v in res["mae_pa"]    if not np.isnan(v)]
        nf    = [v for v in res["n_features"] if v > 0]

        cv_out["models"][mname] = {
            "mae_delta_per_fold":  res["mae_delta"],
            "mae_delta_mean":      float(np.mean(mae_d)) if mae_d else None,
            "mae_delta_std":       float(np.std(mae_d))  if mae_d else None,
            "mae_pa_per_fold":     res["mae_pa"],
            "mae_pa_mean":         float(np.mean(mae_p)) if mae_p else None,
            "mae_pa_std":          float(np.std(mae_p))  if mae_p else None,
            "n_features_mean":     float(np.mean(nf))    if nf else None,
        }

        summary_rows.append({
            "model":           mname,
            "mae_delta_mean":  round(float(np.mean(mae_d)), 4) if mae_d else None,
            "mae_delta_std":   round(float(np.std(mae_d)),  4) if mae_d else None,
            "mae_pa_mean":     round(float(np.mean(mae_p)), 4) if mae_p else None,
            "mae_pa_std":      round(float(np.std(mae_p)),  4) if mae_p else None,
            "n_features_mean": round(float(np.mean(nf)),    1) if nf else None,
        })

    cv_out["run_at"] = datetime.now(timezone.utc).isoformat()

    # Save outputs
    (out_dir / "cv_results.json").write_text(json.dumps(cv_out, indent=2))

    pd.DataFrame(all_pred_rows).to_csv(out_dir / "predictions.csv", index=False)

    df_summary = pd.DataFrame(summary_rows).sort_values("mae_delta_mean")
    df_summary.to_csv(out_dir / "mae_summary.csv", index=False)

    # Feature importances
    if feat_importance:
        imp_rows = []
        for mname, imps in feat_importance.items():
            for fname, fval in sorted(imps.items(), key=lambda x: -x[1])[:100]:
                imp_rows.append({"model": mname, "feature": fname,
                                  "importance": round(fval, 6)})
        pd.DataFrame(imp_rows).to_csv(out_dir / "feature_importance.csv", index=False)

    # Baseline data for plots (raw PM7 vs true PA)
    if pa_pm7_all is not None and pa_true_all is not None:
        baseline_df = pd.DataFrame({
            "record_id":     df.get("record_id", pd.Series(range(len(df)))),
            "neutral_smiles":df.get("neutral_smiles", pd.Series([""] * len(df))),
            "pa_pm7":        pa_pm7_all,
            "pa_true":       pa_true_all,
            "raw_pm7_error": pa_pm7_all - pa_true_all,
        })
        baseline_df.to_csv(out_dir / "baseline_data.csv", index=False)

    # Print summary
    print(f"\n  Results for {dataset_name}  (MAE in {unit_label}):")
    print(f"  {'Model':<18} {'MAE delta':>12} {'MAE PA':>12} {'N features':>12}")
    print(f"  {'─'*57}")
    for row in sorted(summary_rows, key=lambda r: r["mae_delta_mean"] or 999):
        d = f"{row['mae_delta_mean']:.2f}±{row['mae_delta_std']:.2f}" if row['mae_delta_mean'] else "N/A"
        p = f"{row['mae_pa_mean']:.2f}±{row['mae_pa_std']:.2f}"       if row['mae_pa_mean']    else "N/A"
        n = f"{row['n_features_mean']:.0f}"                            if row['n_features_mean'] else "N/A"
        print(f"  {row['model']:<18} {d:>12} {p:>12} {n:>12}")

    log.info(f"  Outputs → {out_dir.relative_to(SCRIPT_DIR.parent.parent)}/")
    return cv_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train surrogate delta-learning models with 5-fold CV."
    )
    parser.add_argument("--dataset", default="all",
                        choices=["all", "nist", "kmeans"])
    parser.add_argument("--n-folds",  type=int, default=5)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--var-threshold",  type=float, default=0.01)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument(
        "--target-corr-pool-max",
        type=int,
        default=DEFAULT_TARGET_CORRELATION_POOL_MAX,
        help="Maximum training-fold features entering redundancy filtering",
    )
    parser.add_argument(
        "--group-col", type=str, default=None,
        help=("Override the k-means GroupKFold column (default: record_id). "
              "NIST remains row-wise because it has one row per molecule."),
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("all", "nist"):
        log.info("Loading NIST 1155 dataset ...")
        df_nist = pd.read_parquet(TARGET_DIR / "nist1155_ml.parquet")
        run_cv(
            df              = df_nist,
            target_col      = "delta_pm7_exp",
            pa_pm7_col      = "pm7_best_pa_kjmol",
            pa_true_col     = "exp_pa_kjmol",
            dataset_name    = "nist1155",
            n_folds         = args.n_folds,
            seed            = args.seed,
            variance_threshold  = args.var_threshold,
            correlation_threshold = args.corr_threshold,
            target_correlation_pool_max = args.target_corr_pool_max,
        )

    if args.dataset in ("all", "kmeans"):
        log.info("Loading k-means 251 dataset ...")
        df_km = pd.read_parquet(TARGET_DIR / "kmeans251_ml.parquet")
        run_cv(
            df              = df_km,
            target_col      = "delta_dft_pm7",
            pa_pm7_col      = "pm7_pa_kjmol",
            pa_true_col     = "dft_pa_kjmol",
            dataset_name    = "kmeans251",
            n_folds         = args.n_folds,
            seed            = args.seed,
            variance_threshold  = args.var_threshold,
            correlation_threshold = args.corr_threshold,
            target_correlation_pool_max = args.target_corr_pool_max,
            # Multiple protonation-site rows belong to each molecule. Grouping
            # is mandatory by default so the documented reproduction command
            # cannot silently leak a molecule across train and test folds.
            group_col       = args.group_col or "record_id",
        )

    print(f"\n  All results saved to: results/")


if __name__ == "__main__":
    main()
