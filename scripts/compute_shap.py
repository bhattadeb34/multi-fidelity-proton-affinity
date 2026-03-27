"""
compute_shap.py
===============
SHAP analysis for ExtraTrees on NIST and k-means datasets.

Outputs (figures/shap/, PDF only):
  shap_beeswarm_nist.pdf         — beeswarm top 20 features
  shap_beeswarm_kmeans.pdf       — beeswarm top 20 features
  shap_beeswarm_combined.pdf     — 2-panel: NIST + k-means side by side
  shap_importance_nist.pdf       — mean |SHAP| bar chart
  shap_importance_kmeans.pdf     — mean |SHAP| bar chart
  shap_dependence_nist.pdf       — dependence plots top 5
  shap_dependence_kmeans.pdf     — dependence plots top 5

Raw data saved to results/{nist1155,kmeans251}/shap/
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
import shap
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR    = PROJECT_DIR / "data"
TARGET_DIR  = DATA_DIR / "targets"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR     = PROJECT_DIR / "figures" / "shap"

KJMOL_TO_KCAL = 1 / 4.184
TOP_N  = 20
TOP_DEP = 5

TICK_SIZE  = 13
LABEL_SIZE = 15
TITLE_SIZE = 15
SPINE_LW   = 1.2

NON_FEATURE_COLS = {
    "record_id","mol_id","source","dataset",
    "neutral_smiles","protonated_smiles",
    "site_idx","site_name","mordred_geom_source",
    "exp_pa_kjmol","exp_pa_kcalmol",
    "dft_pa_kjmol","dft_pa_kcalmol",
    "pm7_pa_kjmol","pm7_pa_kcalmol",
    "pm7_best_pa_kjmol","pm7_best_pa_kcalmol",
    "delta_dft_exp","delta_pm7_exp","dft_correction",
    "delta_dft_pm7","raw_pm7_error","correction_kcalmol",
    "pm7_pa_kcalmol","dft_pa_kcalmol",
}


def impute(X, medians=None):
    X = X.copy()
    if medians is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            medians = np.nanmedian(X, axis=0)
    nm = np.isnan(X)
    if nm.any():
        X[nm] = np.take(medians, np.where(nm)[1])
    return X, medians


def get_consensus_features(cv_path, min_folds=3):
    cv = json.loads(cv_path.read_text())
    counts = {}
    for fold in cv.get("selected_features_per_fold", []):
        for f in fold:
            counts[f] = counts.get(f, 0) + 1
    return [f for f, c in counts.items() if c >= min_folds]


def clean_name(n):
    state_label = ""
    for pre, label in [("neutral_","neut_"),("protonated_","prot_"),("delta_","delt_")]:
        if n.startswith(pre):
            state_label = label
            n = n[len(pre):]
            break
    for cat in ("rdkit_","mordred_","pm7_"):
        if n.startswith(cat):
            n = n[len(cat):]
            break
    PM7_PROPS = ("HOMO","LUMO","dipole","HOF","ionization","cosmo","total_energy","n_atoms")
    if any(n.startswith(p) for p in PM7_PROPS):
        return state_label + n
    return n


def rcparams():
    plt.rcParams.update({
        "axes.linewidth":   SPINE_LW,
        "xtick.labelsize":  TICK_SIZE,
        "ytick.labelsize":  TICK_SIZE,
        "figure.dpi":       300,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
    })


def top_indices(shap_values, n):
    return np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:n]


def plot_beeswarm(sv, X, names, title, out, top_n=TOP_N):
    rcparams()
    top_n = min(top_n, len(names))
    idx   = top_indices(sv, top_n)
    sv_t  = sv[:, idx]
    X_t   = X[:, idx]
    n_t   = [clean_name(names[i]) for i in idx]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
    fig.patch.set_facecolor("white")
    plt.sca(ax)
    shap.summary_plot(sv_t, X_t, feature_names=n_t,
                      plot_type="dot", max_display=top_n,
                      show=False, color_bar=True, plot_size=None)
    ax = plt.gca()
    ax.set_xlabel("SHAP value (kcal/mol)", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    log.info(f"  Saved {out}")
    plt.close("all")


def plot_importance(sv, names, title, out, top_n=TOP_N):
    rcparams()
    mean_abs = np.abs(sv).mean(axis=0)
    top_n = min(top_n, len(names))  # cap to available features
    idx  = np.argsort(mean_abs)[::-1][:top_n]
    vals = mean_abs[idx]
    nms  = [clean_name(names[i]) for i in idx]

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.4)))
    fig.patch.set_facecolor("white")
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, top_n))
    ax.barh(range(top_n), vals[::-1], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(nms[::-1], fontsize=TICK_SIZE)
    ax.set_xlabel("Mean |SHAP value| (kcal/mol)", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", lw=0.4, alpha=0.4, ls="--")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    log.info(f"  Saved {out}")
    plt.close("all")


def plot_dependence(sv, X, names, title, out, top_n=TOP_DEP):
    rcparams()
    top_n = min(top_n, len(names))
    idx = top_indices(sv, top_n)
    fig, axes = plt.subplots(1, top_n, figsize=(4.5*top_n, 4.5))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(wspace=0.55)
    if top_n == 1:
        axes = [axes]
    for ax, fi in zip(axes, idx):
        fname = clean_name(names[fi])
        fv = X[:, fi]
        sv_f = sv[:, fi]
        vmax = np.abs(sv_f).max()
        sc = ax.scatter(fv, sv_f, c=sv_f, cmap="coolwarm",
                        s=12, alpha=0.7, linewidths=0,
                        vmin=-vmax, vmax=vmax)
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel(fname, fontsize=LABEL_SIZE-2)
        ax.set_ylabel("SHAP value (kcal/mol)", fontsize=LABEL_SIZE-2)
        ax.spines[["top","right"]].set_visible(False)
        plt.colorbar(sc, ax=ax, pad=0.01)
    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    log.info(f"  Saved {out}")
    plt.close("all")


def plot_combined_beeswarm(sv_n, X_n, nm_n, sv_k, X_k, nm_k, out, top_n=TOP_N):
    """Side-by-side beeswarm for NIST and k-means."""
    rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, top_n*0.4)))
    fig.patch.set_facecolor("white")

    for ax, sv, X, names, title in [
        (axes[0], sv_n, X_n, nm_n, "NIST — Mol + PM7 features"),
        (axes[1], sv_k, X_k, nm_k, "k-means — Mol + PM7 features"),
    ]:
        plt.sca(ax)
        top_n_local = min(top_n, len(names))
        idx  = top_indices(sv, top_n_local)
        sv_t = sv[:, idx]
        X_t  = X[:, idx]
        n_t  = [clean_name(names[i]) for i in idx]
        shap.summary_plot(sv_t, X_t, feature_names=n_t,
                          plot_type="dot", max_display=top_n,
                          show=False, color_bar=True, plot_size=None)
        ax = plt.gca()
        ax.set_xlabel("SHAP value (kcal/mol)", fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    log.info(f"  Saved {out}")
    plt.close("all")



def plot_paper_figure(sv_n, X_n, nm_n, sv_k, X_k, nm_k, out,
                      top_n_nist=15, top_n_km=9):
    """4-panel paper figure: (a,b) beeswarms, (c,d) importance bars."""
    rcparams()
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.patch.set_facecolor("white")
    labels = ["(a)","(b)","(c)","(d)"]

    # (a) NIST beeswarm
    plt.sca(axes[0,0])
    tn = min(top_n_nist, len(nm_n))
    idx = top_indices(sv_n, tn)
    shap.summary_plot(sv_n[:,idx], X_n[:,idx],
                      feature_names=[clean_name(nm_n[i]) for i in idx],
                      plot_type="dot", max_display=tn, show=False,
                      color_bar=True, plot_size=None)
    ax = plt.gca(); ax.set_facecolor("white")
    ax.set_xlabel("SHAP value (kcal/mol)", fontsize=LABEL_SIZE-2)
    ax.set_title("NIST — Molecular + PM7 features", fontsize=TITLE_SIZE-1,
                 fontweight="bold", pad=8)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.spines[["top","right"]].set_visible(False)
    ax.text(-0.1, 1.02, labels[0], transform=ax.transAxes,
            fontsize=TITLE_SIZE, fontweight="bold")

    # (b) k-means beeswarm
    plt.sca(axes[0,1])
    tn_k = min(top_n_km, len(nm_k))
    idx_k = top_indices(sv_k, tn_k)
    shap.summary_plot(sv_k[:,idx_k], X_k[:,idx_k],
                      feature_names=[clean_name(nm_k[i]) for i in idx_k],
                      plot_type="dot", max_display=tn_k, show=False,
                      color_bar=True, plot_size=None)
    ax = plt.gca(); ax.set_facecolor("white")
    ax.set_xlabel("SHAP value (kcal/mol)", fontsize=LABEL_SIZE-2)
    ax.set_title("k-means — Molecular + PM7 features", fontsize=TITLE_SIZE-1,
                 fontweight="bold", pad=8)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.spines[["top","right"]].set_visible(False)
    ax.text(-0.1, 1.02, labels[1], transform=ax.transAxes,
            fontsize=TITLE_SIZE, fontweight="bold")

    # (c) NIST importance bar
    ax = axes[1,0]; ax.set_facecolor("white")
    ma = np.abs(sv_n).mean(axis=0)
    tn2 = min(top_n_nist, len(nm_n))
    idx2 = np.argsort(ma)[::-1][:tn2]
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, tn2))
    ax.barh(range(tn2), ma[idx2][::-1], color=colors)
    ax.set_yticks(range(tn2))
    ax.set_yticklabels([clean_name(nm_n[i]) for i in idx2][::-1], fontsize=TICK_SIZE-1)
    ax.set_xlabel("Mean |SHAP value| (kcal/mol)", fontsize=LABEL_SIZE-2)
    ax.set_title("Feature importance — NIST", fontsize=TITLE_SIZE-1,
                 fontweight="bold", pad=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", lw=0.4, alpha=0.4, ls="--")
    ax.text(-0.1, 1.02, labels[2], transform=ax.transAxes,
            fontsize=TITLE_SIZE, fontweight="bold")

    # (d) k-means importance bar
    ax = axes[1,1]; ax.set_facecolor("white")
    ma_k = np.abs(sv_k).mean(axis=0)
    tn_k2 = min(top_n_km, len(nm_k))
    idx3 = np.argsort(ma_k)[::-1][:tn_k2]
    colors_k = plt.cm.Blues(np.linspace(0.35, 0.85, tn_k2))
    ax.barh(range(tn_k2), ma_k[idx3][::-1], color=colors_k)
    ax.set_yticks(range(tn_k2))
    ax.set_yticklabels([clean_name(nm_k[i]) for i in idx3][::-1], fontsize=TICK_SIZE-1)
    ax.set_xlabel("Mean |SHAP value| (kcal/mol)", fontsize=LABEL_SIZE-2)
    ax.set_title("Feature importance — k-means", fontsize=TITLE_SIZE-1,
                 fontweight="bold", pad=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", lw=0.4, alpha=0.4, ls="--")
    ax.text(-0.1, 1.02, labels[3], transform=ax.transAxes,
            fontsize=TITLE_SIZE, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", facecolor="white", dpi=300)
    log.info(f"  Saved {out}")
    plt.close("all")


def run_shap(dataset):
    log.info(f"\n{'='*55}")
    log.info(f"  SHAP — {dataset.upper()}")
    log.info(f"{'='*55}")

    # Target parquets already contain all features — no merge needed
    if dataset == "nist":
        df = pd.read_parquet(TARGET_DIR / "nist1155_ml.parquet")
        if "correction_kcalmol" not in df.columns:
            df["correction_kcalmol"] = df["exp_pa_kcalmol"] - df["pm7_best_pa_kcalmol"]
        target_col  = "correction_kcalmol"
        results_sub = RESULTS_DIR / "nist1155"
    else:
        df = pd.read_parquet(TARGET_DIR / "kmeans251_ml.parquet")
        df["pm7_pa_kcalmol"] = df["pm7_pa_kjmol"] * KJMOL_TO_KCAL
        df["dft_pa_kcalmol"] = df["dft_pa_kjmol"] * KJMOL_TO_KCAL
        if "correction_kcalmol" not in df.columns:
            df["correction_kcalmol"] = df["dft_pa_kcalmol"] - df["pm7_pa_kcalmol"]
        target_col  = "correction_kcalmol"
        results_sub = RESULTS_DIR / "kmeans251"

    # Extract feature columns — same filter as train_models.py
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS
                 and not c.startswith("dft_") and not c.startswith("delta_")
                 and not c.startswith("raw_")]

    log.info(f"  {len(df)} samples, {len(feat_cols)} feature pool")

    # Get consensus features from CV
    consensus = get_consensus_features(results_sub / "cv_results.json")
    avail = [f for f in consensus if f in df.columns]
    log.info(f"  Consensus features: {len(avail)}")

    if len(avail) < 5:
        log.warning("  Too few consensus features — using full feature set with FS")
        X_full = df[feat_cols].values.astype(np.float64)
        y_full = df[target_col].values.astype(np.float64)
        X_full, _ = impute(X_full)
        from sklearn.feature_selection import VarianceThreshold
        vt = VarianceThreshold(0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vt.fit(X_full)
        avail = [c for c, k in zip(feat_cols, vt.get_support()) if k]

    X_raw = df[avail].values.astype(np.float64)
    y     = df[target_col].values.astype(np.float64)
    X, _  = impute(X_raw)

    log.info(f"  Training ExtraTrees (n=500) on {len(avail)} features ...")
    model = ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    log.info("  Computing SHAP values ...")
    explainer   = shap.TreeExplainer(model)
    sv          = explainer.shap_values(X)

    # Save
    shap_dir = results_sub / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sv,  columns=avail).to_csv(shap_dir/"shap_values.csv",         index=False)
    pd.DataFrame(X,   columns=avail).to_csv(shap_dir/"shap_feature_values.csv", index=False)
    (pd.Series(np.abs(sv).mean(axis=0), index=avail)
       .sort_values(ascending=False)
       .to_csv(shap_dir/"shap_importance.csv", header=["mean_abs_shap"]))
    log.info(f"  Data saved to {shap_dir}/")

    return sv, X, avail


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=["all","nist","kmeans"])
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sv_n = X_n = nm_n = None
    sv_k = X_k = nm_k = None

    if args.dataset in ("all","nist"):
        sv_n, X_n, nm_n = run_shap("nist")
        plot_beeswarm(sv_n, X_n, nm_n,
            "SHAP summary — NIST (Molecular + PM7 features)",
            FIG_DIR/"shap_beeswarm_nist.pdf")
        plot_importance(sv_n, nm_n,
            "Feature importance — NIST (Molecular + PM7 features)",
            FIG_DIR/"shap_importance_nist.pdf")
        plot_dependence(sv_n, X_n, nm_n,
            f"SHAP dependence — top {TOP_DEP} features (NIST)",
            FIG_DIR/"shap_dependence_nist.pdf")

    if args.dataset in ("all","kmeans"):
        sv_k, X_k, nm_k = run_shap("kmeans")
        plot_beeswarm(sv_k, X_k, nm_k,
            "SHAP summary — k-means (Molecular + PM7 features)",
            FIG_DIR/"shap_beeswarm_kmeans.pdf")
        plot_importance(sv_k, nm_k,
            "Feature importance — k-means (Molecular + PM7 features)",
            FIG_DIR/"shap_importance_kmeans.pdf")
        plot_dependence(sv_k, X_k, nm_k,
            f"SHAP dependence — top {TOP_DEP} features (k-means)",
            FIG_DIR/"shap_dependence_kmeans.pdf")

    if sv_n is not None and sv_k is not None:
        log.info("  Generating combined beeswarm ...")
        plot_combined_beeswarm(sv_n, X_n, nm_n, sv_k, X_k, nm_k,
                               FIG_DIR/"shap_beeswarm_combined.pdf")
        log.info("  Generating 4-panel paper figure ...")
        plot_paper_figure(sv_n, X_n, nm_n, sv_k, X_k, nm_k,
                          FIG_DIR/"shap_paper_figure.pdf")

    print(f"\n  Done. All SHAP figures in {FIG_DIR}/")
    for f in sorted(FIG_DIR.glob("*.pdf")):
        print(f"    {f.name}")