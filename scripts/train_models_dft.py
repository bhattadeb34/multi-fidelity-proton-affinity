"""
train_models_dft.py
===================
Identical to train_models.py but augments the feature set with
B3LYP/def2-TZVP DFT quantum descriptors not available from PM7:

  dft_neutral_ZPE_kjmol       — zero-point energy (neutral)
  dft_neutral_H_total_Ha      — absolute B3LYP enthalpy (neutral)
  dft_neutral_n_basis         — basis set size (neutral)
  dft_neutral_n_electrons     — electron count (neutral)
  dft_neutral_n_imaginary     — imaginary freq count (neutral)
  dft_neutral_freq_min_cm     — lowest vibrational frequency (neutral)
  dft_neutral_freq_max_cm     — highest vibrational frequency (neutral)
  dft_neutral_n_low_freq      — modes below 100 cm-1 (neutral)
  dft_prot_*                  — same 8 features for protonated state
  dft_delta_ZPE_kjmol         — ΔZPE (protonated - neutral)
  dft_delta_HOMO_LUMO_gap_eV  — Δgap from B3LYP (vs PM7 in base features)
  dft_delta_dipole_debye      — Δdipole from B3LYP

Purpose
-------
Ablation study: does adding high-fidelity DFT descriptors improve the
PM7 correction model over PM7-only features?

Results saved to:
  results/nist1155_dft/
  results/kmeans251_dft/

These can be directly compared against:
  results/nist1155/      (PM7-only baseline)
  results/kmeans251/     (PM7-only baseline)

Usage
-----
  python scripts/train_models_dft.py --dataset all
  python scripts/train_models_dft.py --dataset nist
  python scripts/train_models_dft.py --dataset kmeans
"""

import json
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import (Ridge, Lasso, ElasticNet, BayesianRidge, LassoCV)
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

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR    = Path(__file__).parent
DATA_DIR      = SCRIPT_DIR.parent / "data"
TARGET_DIR    = DATA_DIR / "targets"
RESULTS_DIR   = SCRIPT_DIR.parent / "results"
KJMOL_TO_KCAL = 1 / 4.184
GPR_MAX_SAMPLES = 500

# ---------------------------------------------------------------------------
# DFT extra features
# ---------------------------------------------------------------------------

DFT_EXTRA_FEATURES = [
    "dft_neutral_ZPE_kjmol",
    "dft_neutral_H_total_Ha",
    "dft_neutral_n_basis",
    "dft_neutral_n_electrons",
    "dft_neutral_n_imaginary",
    "dft_neutral_freq_min_cm",
    "dft_neutral_freq_max_cm",
    "dft_neutral_n_low_freq",
    "dft_prot_ZPE_kjmol",
    "dft_prot_H_total_Ha",
    "dft_prot_n_basis",
    "dft_prot_n_electrons",
    "dft_prot_n_imaginary",
    "dft_prot_freq_min_cm",
    "dft_prot_freq_max_cm",
    "dft_prot_n_low_freq",
    "dft_delta_ZPE_kjmol",
    "dft_delta_HOMO_LUMO_gap_eV",
    "dft_delta_dipole_debye",
]


def load_dft_features() -> pd.DataFrame:
    """
    Extract DFT-specific features from dataset.json (folder + json source records).
    Returns a DataFrame with neutral_smiles, protonated_smiles, and all DFT_EXTRA_FEATURES.
    """
    dataset_path = DATA_DIR / "processed" / "dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset.json not found at {dataset_path}")

    dataset = json.loads(dataset_path.read_text())
    rows = []

    for rec in dataset.values():
        if rec["metadata"]["source"] not in ("folder", "json"):
            continue

        neu     = rec["neutral"]
        neu_smi = neu.get("smiles", "")
        if not neu_smi:
            continue

        for site in rec.get("all_sites", []):
            prot_smi = site.get("protonated_smiles", "")
            rows.append({
                "neutral_smiles":              neu_smi,
                "protonated_smiles":            prot_smi,
                "dft_neutral_ZPE_kjmol":        neu.get("ZPE_kjmol"),
                "dft_neutral_H_total_Ha":       neu.get("H_total_Ha"),
                "dft_neutral_n_basis":          neu.get("n_basis"),
                "dft_neutral_n_electrons":      neu.get("n_electrons"),
                "dft_neutral_n_imaginary":      neu.get("n_imaginary"),
                "dft_neutral_freq_min_cm":      neu.get("freq_min_cm"),
                "dft_neutral_freq_max_cm":      neu.get("freq_max_cm"),
                "dft_neutral_n_low_freq":       neu.get("n_low_freq"),
                "dft_prot_ZPE_kjmol":           site.get("ZPE_kjmol"),
                "dft_prot_H_total_Ha":          site.get("H_total_Ha"),
                "dft_prot_n_basis":             site.get("n_basis"),
                "dft_prot_n_electrons":         site.get("n_electrons"),
                "dft_prot_n_imaginary":         site.get("n_imaginary"),
                "dft_prot_freq_min_cm":         site.get("freq_min_cm"),
                "dft_prot_freq_max_cm":         site.get("freq_max_cm"),
                "dft_prot_n_low_freq":          site.get("n_low_freq"),
                "dft_delta_ZPE_kjmol":          site.get("delta_ZPE_kjmol"),
                "dft_delta_HOMO_LUMO_gap_eV":   site.get("delta_HOMO_LUMO_gap_eV"),
                "dft_delta_dipole_debye":       site.get("delta_dipole_debye"),
            })

    df = pd.DataFrame(rows)
    log.info(f"DFT features loaded: {len(df)} rows, "
             f"{df['neutral_smiles'].nunique()} unique molecules")
    return df


def augment_with_dft(df: pd.DataFrame, dft_df: pd.DataFrame,
                     join_cols: list[str]) -> pd.DataFrame:
    """
    Left-join DFT extra features onto df using join_cols.
    For molecules without an exact protonated_smiles match,
    falls back to neutral_smiles-only join using the best (max H_total) DFT site.
    """
    # Exact join on both neutral + protonated SMILES
    df_aug = df.merge(
        dft_df[join_cols + DFT_EXTRA_FEATURES].drop_duplicates(subset=join_cols),
        on=join_cols, how="left",
    )
    n_exact = df_aug["dft_neutral_ZPE_kjmol"].notna().sum()
    log.info(f"  DFT join (exact): {n_exact}/{len(df_aug)} rows matched")

    # Fallback: neutral-only join for unmatched rows
    unmatched = df_aug["dft_neutral_ZPE_kjmol"].isna()
    if unmatched.sum() > 0 and "neutral_smiles" in join_cols:
        best_per_mol = (
            dft_df.sort_values("dft_neutral_H_total_Ha", ascending=False)
            .drop_duplicates(subset=["neutral_smiles"])
            [["neutral_smiles"] + DFT_EXTRA_FEATURES]
            .rename(columns={c: c + "_fb" for c in DFT_EXTRA_FEATURES})
        )
        df_aug = df_aug.merge(best_per_mol, on="neutral_smiles", how="left")
        for col in DFT_EXTRA_FEATURES:
            mask = df_aug[col].isna() & df_aug[col + "_fb"].notna()
            df_aug.loc[mask, col] = df_aug.loc[mask, col + "_fb"]
            df_aug.drop(columns=[col + "_fb"], inplace=True)

        n_fallback = df_aug["dft_neutral_ZPE_kjmol"].notna().sum() - n_exact
        log.info(f"  DFT join (fallback neutral-only): {n_fallback} additional rows")

    n_total = df_aug["dft_neutral_ZPE_kjmol"].notna().sum()
    n_missing = len(df_aug) - n_total
    if n_missing > 0:
        log.warning(f"  {n_missing} rows have no DFT features — will be NaN (handled by imputation)")

    return df_aug


# ---------------------------------------------------------------------------
# Non-feature columns (identical to train_models.py)
# ---------------------------------------------------------------------------

NON_FEATURE_COLS = {
    "record_id", "mol_id", "source", "dataset",
    "neutral_smiles", "protonated_smiles",
    "site_idx", "site_name", "mordred_geom_source",
    "exp_pa_kjmol", "exp_pa_kcalmol",
    "dft_pa_kjmol", "dft_pa_kcalmol",
    "pm7_pa_kjmol", "pm7_pa_kcalmol",
    "pm7_best_pa_kjmol", "pm7_best_pa_kcalmol",
    "delta_dft_exp", "delta_pm7_exp", "dft_correction",
    "delta_pm7_exp", "delta_dft_pm7",
    "raw_pm7_error",
}


# ---------------------------------------------------------------------------
# Models (identical to train_models.py)
# ---------------------------------------------------------------------------

def get_models(n_train: int) -> dict:
    models = {
        "Ridge":            Ridge(),
        "Lasso":            Lasso(max_iter=10000),
        "ElasticNet":       ElasticNet(max_iter=10000),
        "BayesianRidge":    BayesianRidge(),
        "SVR":              Pipeline([("scaler", StandardScaler()),
                                      ("svr", SVR(kernel="rbf"))]),
        "DecisionTree":     DecisionTreeRegressor(random_state=42),
        "RandomForest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "ExtraTrees":       ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "AdaBoost":         AdaBoostRegressor(n_estimators=200, random_state=42),
        "MLP":              Pipeline([("scaler", StandardScaler()),
                                      ("mlp", MLPRegressor(
                                          hidden_layer_sizes=(256, 128, 64),
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
    if n_train <= GPR_MAX_SAMPLES:
        models["GPR"] = GaussianProcessRegressor(
            kernel=DotProduct() + WhiteKernel(), random_state=42, normalize_y=True)
    else:
        log.info(f"  GPR skipped (n_train={n_train} > {GPR_MAX_SAMPLES})")

    base = []
    if HAS_XGB:
        base.append(("xgb", XGBRegressor(n_estimators=200, random_state=42,
                                          verbosity=0, n_jobs=-1)))
    if HAS_LGBM:
        base.append(("lgbm", LGBMRegressor(n_estimators=200, random_state=42,
                                             n_jobs=-1, verbose=-1)))
    base.append(("gbm", GradientBoostingRegressor(n_estimators=200, random_state=42)))
    if len(base) >= 2:
        models["VotingEnsemble"] = VotingRegressor(estimators=base)

    return models


# ---------------------------------------------------------------------------
# Feature selection (identical to train_models.py)
# ---------------------------------------------------------------------------

def select_features(X_train, y_train, feature_names,
                    variance_threshold=0.01, correlation_threshold=0.95,
                    lasso_cv_folds=5):
    names = list(feature_names)
    X = X_train.copy()

    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    vt = VarianceThreshold(threshold=variance_threshold)
    X = vt.fit_transform(X)
    names = [n for n, keep in zip(names, vt.get_support()) if keep]

    if len(names) > 1:
        df_tmp = pd.DataFrame(X, columns=names)
        target_corr = df_tmp.corrwith(pd.Series(y_train)).abs()
        corr_matrix = df_tmp.corr().abs()
        to_drop = set()
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            if cols[i] in to_drop:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in to_drop:
                    continue
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    if target_corr.get(cols[i], 0) >= target_corr.get(cols[j], 0):
                        to_drop.add(cols[j])
                    else:
                        to_drop.add(cols[i])
        keep_mask = [n not in to_drop for n in names]
        X = X[:, keep_mask]
        names = [n for n, k in zip(names, keep_mask) if k]

    if len(names) > 0:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lasso_cv = LassoCV(cv=lasso_cv_folds, max_iter=10000, n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso_cv.fit(X_scaled, y_train)

        mean_mse = lasso_cv.mse_path_.mean(axis=1)
        std_mse  = lasso_cv.mse_path_.std(axis=1)
        best_idx = np.argmin(mean_mse)
        threshold = mean_mse[best_idx] + std_mse[best_idx]
        valid = mean_mse <= threshold
        alpha_1se = lasso_cv.alphas_[valid].max()

        lasso_1se = Lasso(alpha=alpha_1se, max_iter=10000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso_1se.fit(X_scaled, y_train)

        selected_mask = lasso_1se.coef_ != 0
        if selected_mask.sum() == 0:
            lasso_best = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lasso_best.fit(X_scaled, y_train)
            selected_mask = lasso_best.coef_ != 0
        if selected_mask.sum() == 0:
            top_k = min(10, len(names))
            top_idx = np.argsort(np.abs(lasso_cv.coef_))[-top_k:]
            selected_mask = np.zeros(len(names), dtype=bool)
            selected_mask[top_idx] = True

        X = X[:, selected_mask]
        names = [n for n, s in zip(names, selected_mask) if s]

    return X, names


def apply_feature_selection(X_test, X_train_full, feature_names_full, selected_names):
    name_to_idx = {n: i for i, n in enumerate(feature_names_full)}
    sel_idx = [name_to_idx[n] for n in selected_names]
    X = X_test[:, sel_idx].copy()
    train_sel = X_train_full[:, sel_idx]
    col_medians = np.nanmedian(train_sel, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    return X


# ---------------------------------------------------------------------------
# CV loop (identical to train_models.py)
# ---------------------------------------------------------------------------

def run_cv(df, target_col, pa_pm7_col, pa_true_col, dataset_name,
           n_folds=5, seed=42, variance_threshold=0.01,
           correlation_threshold=0.95, unit_scale=KJMOL_TO_KCAL,
           unit_label="kcal/mol"):

    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS
                    and c != target_col and c != pa_pm7_col and c != pa_true_col
                    and not c.startswith("delta_") and not c.startswith("raw_")]
    feature_cols = [c for c in feature_cols if c not in {
        "exp_pa_kjmol", "exp_pa_kcalmol", "dft_pa_kjmol", "dft_pa_kcalmol",
        "pm7_pa_kjmol", "pm7_pa_kcalmol", "pm7_best_pa_kjmol", "pm7_best_pa_kcalmol",
    }]

    log.info(f"\n{'='*55}")
    log.info(f"  Dataset      : {dataset_name}")
    log.info(f"  Samples      : {len(df)}")
    log.info(f"  Target       : {target_col}")
    log.info(f"  Features     : {len(feature_cols)}")
    log.info(f"  DFT extras   : {sum(1 for c in feature_cols if c.startswith('dft_'))}")
    log.info(f"  Folds        : {n_folds}")
    log.info(f"{'='*55}")

    X_all       = df[feature_cols].values.astype(np.float64)
    y_all       = df[target_col].values.astype(np.float64)
    pa_pm7_all  = df[pa_pm7_col].values  if pa_pm7_col  in df.columns else None
    pa_true_all = df[pa_true_col].values if pa_true_col in df.columns else None

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_pred_rows = []
    fold_results  = {}
    feat_importance = {}

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        log.info(f"\n  Fold {fold_idx+1}/{n_folds}  "
                 f"(train={len(train_idx)}, test={len(test_idx)})")

        X_train_raw = X_all[train_idx]
        X_test_raw  = X_all[test_idx]
        y_train     = y_all[train_idx]
        y_test      = y_all[test_idx]

        log.info("    Feature selection ...")
        X_train_sel, sel_names = select_features(
            X_train_raw, y_train, feature_cols,
            variance_threshold, correlation_threshold)
        X_test_sel = apply_feature_selection(
            X_test_raw, X_train_raw, feature_cols, sel_names)
        log.info(f"    Selected {len(sel_names)} features  "
                 f"(DFT: {sum(1 for n in sel_names if n.startswith('dft_'))})")

        models = get_models(n_train=len(train_idx))
        for mname in models:
            if mname not in fold_results:
                fold_results[mname] = {"mae_delta": [], "mae_pa": [], "n_features": []}

        for mname, model in models.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train_sel, y_train)
                    y_pred = model.predict(X_test_sel)

                mae_delta = mean_absolute_error(y_test, y_pred) * unit_scale

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

                for i, idx in enumerate(test_idx):
                    all_pred_rows.append({
                        "fold":           fold_idx + 1,
                        "model":          mname,
                        "sample_idx":     int(idx),
                        "record_id":      str(df.iloc[idx].get("record_id", idx)),
                        "neutral_smiles": df.iloc[idx].get("neutral_smiles", ""),
                        "y_true_delta":   float(y_test[i]),
                        "y_pred_delta":   float(y_pred[i]),
                        "pa_pm7":         float(pa_pm7_test[i]) if pa_pm7_all is not None else np.nan,
                        "pa_pred":        float(pa_pred[i]),
                        "pa_true":        float(pa_true_test[i]) if pa_true_all is not None else np.nan,
                    })

                raw_model = (model.named_steps.get("model") or model
                             if hasattr(model, "named_steps") else model)
                if hasattr(raw_model, "feature_importances_"):
                    imp = raw_model.feature_importances_
                    if mname not in feat_importance:
                        feat_importance[mname] = {}
                    for fname, fval in zip(sel_names, imp):
                        feat_importance[mname][fname] = (
                            feat_importance[mname].get(fname, 0) + fval / n_folds)

            except Exception as e:
                log.warning(f"    {mname} failed fold {fold_idx+1}: {e}")
                fold_results[mname]["mae_delta"].append(np.nan)
                fold_results[mname]["mae_pa"].append(np.nan)
                fold_results[mname]["n_features"].append(0)

        log.info(f"    Fold {fold_idx+1} done  "
                 + "  ".join(f"{m}={np.mean(fold_results[m]['mae_delta']):.2f}"
                              for m in list(models)[:4]))

    summary_rows = []
    cv_out = {"dataset": dataset_name, "unit": unit_label,
              "feature_set": "PM7+DFT", "models": {}}

    for mname, res in fold_results.items():
        mae_d = [v for v in res["mae_delta"] if not np.isnan(v)]
        mae_p = [v for v in res["mae_pa"]    if not np.isnan(v)]
        nf    = [v for v in res["n_features"] if v > 0]
        cv_out["models"][mname] = {
            "mae_delta_per_fold": res["mae_delta"],
            "mae_delta_mean":     float(np.mean(mae_d)) if mae_d else None,
            "mae_delta_std":      float(np.std(mae_d))  if mae_d else None,
            "mae_pa_per_fold":    res["mae_pa"],
            "mae_pa_mean":        float(np.mean(mae_p)) if mae_p else None,
            "mae_pa_std":         float(np.std(mae_p))  if mae_p else None,
            "n_features_mean":    float(np.mean(nf))    if nf else None,
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
    (out_dir / "cv_results.json").write_text(json.dumps(cv_out, indent=2))
    pd.DataFrame(all_pred_rows).to_csv(out_dir / "predictions.csv", index=False)
    df_summary = pd.DataFrame(summary_rows).sort_values("mae_delta_mean")
    df_summary.to_csv(out_dir / "mae_summary.csv", index=False)

    if feat_importance:
        imp_rows = []
        for mname, imps in feat_importance.items():
            for fname, fval in sorted(imps.items(), key=lambda x: -x[1])[:100]:
                imp_rows.append({"model": mname, "feature": fname,
                                  "importance": round(fval, 6)})
        pd.DataFrame(imp_rows).to_csv(out_dir / "feature_importance.csv", index=False)

    if pa_pm7_all is not None and pa_true_all is not None:
        pd.DataFrame({
            "record_id":      df.get("record_id", pd.Series(range(len(df)))),
            "neutral_smiles": df.get("neutral_smiles", pd.Series([""] * len(df))),
            "pa_pm7":         pa_pm7_all,
            "pa_true":        pa_true_all,
            "raw_pm7_error":  pa_pm7_all - pa_true_all,
        }).to_csv(out_dir / "baseline_data.csv", index=False)

    print(f"\n  Results for {dataset_name}  (MAE in {unit_label}, PM7+DFT features):")
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
        description="Train PM7+DFT augmented surrogate models (ablation study).")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "nist", "kmeans"])
    parser.add_argument("--n-folds",        type=int,   default=5)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--var-threshold",  type=float, default=0.01)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading DFT extra features from dataset.json ...")
    dft_df = load_dft_features()

    if args.dataset in ("all", "nist"):
        log.info("Loading NIST 1155 dataset ...")
        df_nist = pd.read_parquet(TARGET_DIR / "nist1155_ml.parquet")
        df_nist = augment_with_dft(df_nist, dft_df, join_cols=["neutral_smiles"])
        run_cv(
            df           = df_nist,
            target_col   = "delta_pm7_exp",
            pa_pm7_col   = "pm7_best_pa_kjmol",
            pa_true_col  = "exp_pa_kjmol",
            dataset_name = "nist1155_dft",
            n_folds      = args.n_folds,
            seed         = args.seed,
            variance_threshold    = args.var_threshold,
            correlation_threshold = args.corr_threshold,
        )

    if args.dataset in ("all", "kmeans"):
        log.info("Loading k-means 251 dataset ...")
        df_km = pd.read_parquet(TARGET_DIR / "kmeans251_ml.parquet")
        df_km = augment_with_dft(df_km, dft_df,
                                  join_cols=["neutral_smiles", "protonated_smiles"])
        run_cv(
            df           = df_km,
            target_col   = "delta_dft_pm7",
            pa_pm7_col   = "pm7_pa_kjmol",
            pa_true_col  = "dft_pa_kjmol",
            dataset_name = "kmeans251_dft",
            n_folds      = args.n_folds,
            seed         = args.seed,
            variance_threshold    = args.var_threshold,
            correlation_threshold = args.corr_threshold,
        )

    print("\n  All results saved to: results/nist1155_dft/  and  results/kmeans251_dft/")
    print("  Compare against:      results/nist1155/       and  results/kmeans251/")


if __name__ == "__main__":
    main()
