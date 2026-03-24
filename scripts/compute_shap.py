"""
compute_shap.py
===============
SHAP analysis for the best model (ExtraTrees) on both datasets.
Requires retraining the model on the full dataset (no CV) to get
a single model object for SHAP computation.

Outputs (saved to results/{dataset}/shap/):
  shap_values.npy          — SHAP values array (n_samples × n_features)
  shap_feature_names.json  — corresponding feature names
  shap_summary.pdf/.png    — beeswarm summary plot (top 20 features)
  shap_bar.pdf/.png        — mean |SHAP| bar chart (top 20)
  shap_dependence_*.pdf    — dependence plots for top 5 features

Usage
-----
  python scripts/compute_shap.py --dataset nist
  python scripts/compute_shap.py --dataset kmeans
  python scripts/compute_shap.py --dataset all
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR    = Path(__file__).parent
PROJECT_DIR   = SCRIPT_DIR.parent
DATA_DIR      = PROJECT_DIR / "data"
TARGET_DIR    = DATA_DIR / "targets"
RESULTS_DIR   = PROJECT_DIR / "results"
FIG_DIR       = PROJECT_DIR / "figures"
KJMOL_TO_KCAL = 1 / 4.184

TICK_SIZE, LABEL_SIZE, SPINE_LW = 18, 22, 1.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.linewidth": SPINE_LW,
    "xtick.major.width": SPINE_LW, "ytick.major.width": SPINE_LW,
    "xtick.labelsize": TICK_SIZE,  "ytick.labelsize": TICK_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})

NON_FEATURE_COLS = {
    "record_id", "mol_id", "source", "dataset",
    "neutral_smiles", "protonated_smiles",
    "site_idx", "site_name", "mordred_geom_source",
    "exp_pa_kjmol", "exp_pa_kcalmol", "dft_pa_kjmol", "dft_pa_kcalmol",
    "pm7_pa_kjmol", "pm7_pa_kcalmol", "pm7_best_pa_kjmol", "pm7_best_pa_kcalmol",
    "delta_dft_exp", "delta_pm7_exp", "dft_correction",
    "delta_pm7_exp", "delta_dft_pm7", "raw_pm7_error",
}


def load_dataset(dataset_name: str) -> tuple[pd.DataFrame, str, list[str]]:
    if dataset_name == "nist":
        df = pd.read_parquet(TARGET_DIR / "nist1155_ml.parquet")
        target_col = "delta_pm7_exp"
    else:
        df = pd.read_parquet(TARGET_DIR / "kmeans251_ml.parquet")
        target_col = "delta_dft_pm7"

    feature_cols = [c for c in df.columns
                    if c not in NON_FEATURE_COLS
                    and c != target_col
                    and not c.startswith("delta_")
                    and not c.startswith("raw_")
                    and c not in {"exp_pa_kjmol","exp_pa_kcalmol","dft_pa_kjmol",
                                   "dft_pa_kcalmol","pm7_pa_kjmol","pm7_pa_kcalmol",
                                   "pm7_best_pa_kjmol","pm7_best_pa_kcalmol"}]
    return df, target_col, feature_cols


def prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    X = df[feature_cols].values.astype(np.float64)
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    return X


def run_shap_analysis(dataset_name: str):
    try:
        import shap
    except ImportError:
        log.error("shap not installed. Run: pip install shap")
        return

    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    log.info(f"Running SHAP analysis for {dataset_name} ...")
    out_dir = RESULTS_DIR / (f"nist1155" if dataset_name == "nist" else "kmeans251") / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_stem = f"shap_{dataset_name}"

    df, target_col, feature_cols = load_dataset(dataset_name)
    X_all = prepare_features(df, feature_cols)
    y_all = df[target_col].values.astype(np.float64)

    log.info(f"  Dataset: {len(df)} samples, {len(feature_cols)} features")

    # Feature selection on full dataset (mirrors CV pipeline)
    log.info("  Running feature selection on full dataset ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Stage 1: variance
        vt = VarianceThreshold(threshold=0.01)
        X_sel = vt.fit_transform(X_all)
        names = [n for n, keep in zip(feature_cols, vt.get_support()) if keep]

        # Stage 2: correlation
        df_tmp = pd.DataFrame(X_sel, columns=names)
        target_corr = df_tmp.corrwith(pd.Series(y_all)).abs()
        corr_matrix = df_tmp.corr().abs()
        to_drop = set()
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            if cols[i] in to_drop: continue
            for j in range(i+1, len(cols)):
                if cols[j] in to_drop: continue
                if corr_matrix.iloc[i,j] > 0.95:
                    if target_corr.get(cols[i],0) >= target_corr.get(cols[j],0):
                        to_drop.add(cols[j])
                    else:
                        to_drop.add(cols[i])
        keep_mask = [n not in to_drop for n in names]
        X_sel = X_sel[:, keep_mask]
        names = [n for n, k in zip(names, keep_mask) if k]

        # Stage 3+4: Lasso 1-SE
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sel)
        lasso_cv = LassoCV(cv=5, max_iter=10000, n_jobs=-1)
        lasso_cv.fit(X_scaled, y_all)
        mean_mse = lasso_cv.mse_path_.mean(axis=1)
        std_mse  = lasso_cv.mse_path_.std(axis=1)
        best_idx = np.argmin(mean_mse)
        threshold = mean_mse[best_idx] + std_mse[best_idx]
        valid = mean_mse <= threshold
        alpha_1se = lasso_cv.alphas_[valid].max()

        from sklearn.linear_model import Lasso
        lasso_1se = Lasso(alpha=alpha_1se, max_iter=10000)
        lasso_1se.fit(X_scaled, y_all)
        sel_mask = lasso_1se.coef_ != 0
        if sel_mask.sum() == 0:
            sel_mask = np.abs(lasso_cv.coef_).argsort()[-20:] != 0
        X_final = X_sel[:, sel_mask]
        final_names = [n for n, s in zip(names, sel_mask) if s]

    log.info(f"  Selected {len(final_names)} features for SHAP")

    # Train ExtraTrees on full dataset
    log.info("  Training ExtraTrees on full dataset ...")
    model = ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X_final, y_all)

    # SHAP values
    log.info("  Computing SHAP values (TreeExplainer) ...")
    explainer  = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_final)

    # Save
    np.save(out_dir / "shap_values.npy", shap_values)
    np.save(out_dir / "X_shap.npy", X_final)
    (out_dir / "shap_feature_names.json").write_text(json.dumps(final_names))
    log.info(f"  SHAP values saved to {out_dir}/")

    # Clean feature names for display
    def clean_name(n):
        n = n.replace("neutral_rdkit_", "neu_").replace("protonated_rdkit_", "prot_")
        n = n.replace("neutral_pm7_", "neu_pm7_").replace("protonated_pm7_", "prot_pm7_")
        n = n.replace("neutral_mordred_", "neu_").replace("protonated_mordred_", "prot_")
        n = n.replace("delta_rdkit_", "Δ_").replace("delta_pm7_", "Δpm7_")
        n = n.replace("delta_mordred_", "Δ_")
        n = n.replace("neutral_maccs_", "MACCS_").replace("protonated_maccs_", "prot_MACCS_")
        n = n.replace("neutral_morgan_", "Morgan_").replace("protonated_morgan_", "prot_Morgan_")
        return n

    display_names = [clean_name(n) for n in final_names]

    # ---- Plot 1: Beeswarm summary ----
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top20_idx = np.argsort(mean_abs_shap)[-20:][::-1]

    fig.subplots(figsize=(10, 8))
    plt.figure()
    shap.summary_plot(
        shap_values[:, top20_idx],
        X_final[:, top20_idx],
        feature_names=[display_names[i] for i in top20_idx],
        show=False, plot_size=None,
    )
    ax.set_xlabel("SHAP value (impact on PM7 correction, kcal/mol × 4.184)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{fig_stem}_beeswarm.{ext}")
    plt.close(fig)
    log.info(f"  Saved figures/{fig_stem}_beeswarm.pdf/.png")

    # ---- Plot 2: Bar chart mean |SHAP| ----
    fig.subplots(figsize=(9, 7))
    top_names = [display_names[i] for i in top20_idx]
    top_vals  = mean_abs_shap[top20_idx] * KJMOL_TO_KCAL  # convert to kcal/mol

    y = np.arange(len(top_names))
    ax.barh(y, top_vals[::-1], color="#2166AC", alpha=0.82, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(top_names[::-1], fontsize=TICK_SIZE - 4)
    ax.set_xlabel("Mean |SHAP value| (kcal/mol)")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="x", linewidth=0.4, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{fig_stem}_bar.{ext}")
    plt.close(fig)
    log.info(f"  Saved figures/{fig_stem}_bar.pdf/.png")

    # ---- Plot 3: Dependence plots for top 5 ----
    top5_idx = top20_idx[:5]
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, feat_idx in zip(axes, top5_idx):
        fname = display_names[feat_idx]
        x_vals = X_final[:, feat_idx]
        s_vals = shap_values[:, feat_idx] * KJMOL_TO_KCAL
        ax.scatter(x_vals, s_vals, s=8, alpha=0.5, color="#2166AC",
                   linewidths=0, rasterized=True)
        ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.5)
        ax.set_xlabel(fname, fontsize=11)
        ax.set_ylabel("SHAP (kcal/mol)" if ax == axes[0] else "")
        ax.tick_params(labelsize=11)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    fig.suptitle(f"SHAP dependence plots — top 5 features ({dataset_name.upper()})",
                 fontsize=LABEL_SIZE - 4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{fig_stem}_dependence.{ext}")
    plt.close(fig)
    log.info(f"  Saved figures/{fig_stem}_dependence.pdf/.png")

    log.info(f"  SHAP analysis complete for {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for ExtraTrees models.")
    parser.add_argument("--dataset", default="all", choices=["all", "nist", "kmeans"])
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("all", "nist"):
        run_shap_analysis("nist")
    if args.dataset in ("all", "kmeans"):
        run_shap_analysis("kmeans")

    print(f"\n  SHAP figures saved to: figures/")
    print(f"  shap_nist_beeswarm, shap_nist_bar, shap_nist_dependence")
    print(f"  shap_kmeans_beeswarm, shap_kmeans_bar, shap_kmeans_dependence")


if __name__ == "__main__":
    main()
