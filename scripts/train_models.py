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
  nist1155  — molecule-level, target = |exp_pa - pm7_best_pa|  (kJ/mol)
  kmeans251 — site-level,     target = |dft_pa - pm7_pa|       (kJ/mol)

Feature selection (per fold, train split only)
----------------------------------------------
  Stage 1: Variance filter     threshold = 0.01
  Stage 2: Correlation filter  Pearson |r| > 0.95, keep higher target corr
  Stage 3: LassoCV             inner 5-fold CV to select alpha
  Stage 4: 1-SE rule           parsimony — fewest features within 1 SE of best

Models (16)
-----------
  Linear:   Ridge, Lasso, ElasticNet, BayesianRidge, SVR
  Tree:     DecisionTree, RandomForest, ExtraTrees, GradientBoosting,
            AdaBoost, XGBoost, LightGBM, CatBoost
  Neural:   MLP
  Prob:     GPR  (skipped if n_train > GPR_MAX_SAMPLES for speed)
  Meta:     VotingEnsemble (XGB + LGBM + GBM, uniform weights)

Outputs  (../results/{dataset_name}/)
-------
  cv_results.json         — MAE mean ± std per model, feature counts per fold
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
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import (Ridge, Lasso, ElasticNet, BayesianRidge,
                                   LassoCV)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor, AdaBoostRegressor,
                               VotingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import mean_absolute_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logging.warning("xgboost not installed — XGBRegressor skipped")

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    logging.warning("lightgbm not installed — LGBMRegressor skipped")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logging.warning("catboost not installed — CatBoostRegressor skipped")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR   = Path(__file__).parent
KJMOL_TO_KCAL = 1 / 4.184   # all MAEs reported in kcal/mol for comparison with literature
DATA_DIR     = SCRIPT_DIR.parent / "data"
TARGET_DIR   = DATA_DIR / "targets"
RESULTS_DIR  = SCRIPT_DIR.parent / "results"

GPR_MAX_SAMPLES = 500   # GPR skipped above this size (O(n³) complexity)


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
    """
    Return dict of model_name -> estimator.
    GPR is excluded if n_train > GPR_MAX_SAMPLES.
    VotingEnsemble only included if all three base models are available.
    """
    models = {
        "Ridge":         Ridge(),
        "Lasso":         Lasso(max_iter=10000),
        "ElasticNet":    ElasticNet(max_iter=10000),
        "BayesianRidge": BayesianRidge(),
        "SVR":           Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))]),
        "DecisionTree":  DecisionTreeRegressor(random_state=42),
        "RandomForest":  RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "ExtraTrees":    ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "AdaBoost":      AdaBoostRegressor(n_estimators=200, random_state=42),
        "MLP":           Pipeline([("scaler", StandardScaler()),
                                    ("mlp", MLPRegressor(hidden_layer_sizes=(256, 128, 64),
                                                          max_iter=500, random_state=42))]),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=200, random_state=42,
                                          verbosity=0, n_jobs=-1)
    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(n_estimators=200, random_state=42,
                                             n_jobs=-1, verbose=-1)
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostRegressor(iterations=200, random_state=42,
                                                verbose=False)

    # GPR — skip for large datasets
    if n_train <= GPR_MAX_SAMPLES:
        models["GPR"] = GaussianProcessRegressor(
            kernel=DotProduct() + WhiteKernel(), random_state=42, normalize_y=True
        )
    else:
        log.info(f"  GPR skipped (n_train={n_train} > {GPR_MAX_SAMPLES})")

    # Voting ensemble (requires all three base gradient boosters)
    base_for_voting = []
    if HAS_XGB:
        base_for_voting.append(("xgb", XGBRegressor(n_estimators=200, random_state=42,
                                                      verbosity=0, n_jobs=-1)))
    if HAS_LGBM:
        base_for_voting.append(("lgbm", LGBMRegressor(n_estimators=200, random_state=42,
                                                        n_jobs=-1, verbose=-1)))
    base_for_voting.append(("gbm", GradientBoostingRegressor(n_estimators=200,
                                                               random_state=42)))
    if len(base_for_voting) >= 2:
        models["VotingEnsemble"] = VotingRegressor(estimators=base_for_voting)

    return models


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
) -> tuple[np.ndarray, list[str]]:
    """
    Four-stage hierarchical feature selection on training data only.

    Stage 1 — Variance filter
    Stage 2 — Correlation filter (keep higher target correlation)
    Stage 3 — LassoCV (inner CV to find optimal alpha)
    Stage 4 — 1-SE rule (parsimony)

    Returns (X_selected, selected_feature_names).
    """
    names = list(feature_names)
    X = X_train.copy()

    # Replace NaN with column median (computed on train only)
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    # Stage 1: Variance filter
    vt = VarianceThreshold(threshold=variance_threshold)
    X = vt.fit_transform(X)
    names = [n for n, keep in zip(names, vt.get_support()) if keep]
    log.debug(f"    After variance filter: {len(names)}")

    # Stage 2: Correlation filter
    if len(names) > 1:
        df_tmp = pd.DataFrame(X, columns=names)
        target_corr = df_tmp.corrwith(pd.Series(y_train)).abs()
        corr_matrix = df_tmp.corr().abs()

        to_drop = set()
        # Upper triangle
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            if cols[i] in to_drop:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in to_drop:
                    continue
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    # Drop the one with lower target correlation
                    if target_corr.get(cols[i], 0) >= target_corr.get(cols[j], 0):
                        to_drop.add(cols[j])
                    else:
                        to_drop.add(cols[i])

        keep_mask = [n not in to_drop for n in names]
        X = X[:, keep_mask]
        names = [n for n, k in zip(names, keep_mask) if k]
        log.debug(f"    After correlation filter: {len(names)}")

    # Stage 3 + 4: LassoCV with 1-SE rule
    if len(names) > 0:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Inner CV to find alpha path
        lasso_cv = LassoCV(cv=lasso_cv_folds, max_iter=10000, n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso_cv.fit(X_scaled, y_train)

        # 1-SE rule: find the most regularized alpha within 1 SE of best
        # lasso_cv.mse_path_ shape: (n_alphas, n_folds)
        mean_mse   = lasso_cv.mse_path_.mean(axis=1)
        std_mse    = lasso_cv.mse_path_.std(axis=1)
        best_idx   = np.argmin(mean_mse)
        threshold  = mean_mse[best_idx] + std_mse[best_idx]
        # Among alphas with mse <= threshold, pick the largest alpha (most sparse)
        valid      = mean_mse <= threshold
        alphas     = lasso_cv.alphas_
        alpha_1se  = alphas[valid].max()

        lasso_1se = Lasso(alpha=alpha_1se, max_iter=10000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso_1se.fit(X_scaled, y_train)

        selected_mask = lasso_1se.coef_ != 0
        # Fallback: if 1-SE selects 0 features, use the best alpha features
        if selected_mask.sum() == 0:
            lasso_best = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            lasso_best.fit(X_scaled, y_train)
            selected_mask = lasso_best.coef_ != 0

        # Final fallback: if still 0, keep top-10 by |coef|
        if selected_mask.sum() == 0:
            top_k = min(10, len(names))
            top_idx = np.argsort(np.abs(lasso_cv.coef_))[-top_k:]
            selected_mask = np.zeros(len(names), dtype=bool)
            selected_mask[top_idx] = True

        X = X[:, selected_mask]
        names = [n for n, s in zip(names, selected_mask) if s]
        log.debug(f"    After Lasso 1-SE: {len(names)}")

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

    # Impute with train medians
    train_sel = X_train_full[:, sel_idx]
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
    unit_scale: float = KJMOL_TO_KCAL,   # convert stored kJ/mol -> kcal/mol for reporting
    unit_label: str = "kcal/mol",
) -> dict:
    """
    Run 5-fold CV with in-fold feature selection and all 16 models.

    Parameters
    ----------
    df            : ML-ready DataFrame (from build_targets.py output)
    target_col    : column name for the delta target to model
    pa_pm7_col    : column name for PM7 PA (to compute PA_pred = PM7 + delta)
    pa_true_col   : column name for the high-fidelity PA (for final PA MAE)
    dataset_name  : used for logging and output paths
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

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Storage
    all_pred_rows   = []
    fold_results    = {m: {"mae_delta": [], "mae_pa": [],
                            "n_features": []} for m in []}
    feat_importance = {}

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        log.info(f"\n  Fold {fold_idx+1}/{n_folds}  "
                 f"(train={len(train_idx)}, test={len(test_idx)})")

        X_train_raw, X_test_raw = X_all[train_idx], X_all[test_idx]
        y_train,     y_test     = y_all[train_idx], y_all[test_idx]

        # Feature selection on train split only
        log.info(f"    Feature selection ...")
        X_train_sel, sel_names = select_features(
            X_train_raw, y_train, feature_cols,
            variance_threshold, correlation_threshold,
        )
        X_test_sel = apply_feature_selection(
            X_test_raw, X_train_raw, feature_cols, sel_names
        )
        log.info(f"    Selected {len(sel_names)} features")

        # Get models for this fold's train size
        models = get_models(n_train=len(train_idx))

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
    cv_out = {"dataset": dataset_name, "unit": unit_label, "models": {}}

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

    log.info(f"  Outputs → {out_dir.relative_to(SCRIPT_DIR.parent)}/")
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
        )

    print(f"\n  All results saved to: results/")


if __name__ == "__main__":
    main()
