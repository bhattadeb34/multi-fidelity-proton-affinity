"""
plot_shap.py
============
Regenerates all SHAP figures from pre-saved CSV results.
Run this AFTER compute_shap.py has completed its computation.

Reads from:
  results/nist1155/shap/shap_values.csv
  results/nist1155/shap/shap_feature_values.csv
  results/kmeans251/shap/shap_values.csv
  results/kmeans251/shap/shap_feature_values.csv

Outputs (figures/shap/):
  shap_beeswarm_nist.pdf
  shap_beeswarm_kmeans.pdf
  shap_beeswarm_combined.pdf
  shap_importance_nist.pdf
  shap_importance_kmeans.pdf
  shap_dependence_nist.pdf
  shap_dependence_kmeans.pdf
  shap_paper_figure.pdf

Usage:
  python scripts/plot_shap.py
  python scripts/plot_shap.py --dataset nist
  python scripts/plot_shap.py --dataset kmeans
  python scripts/plot_shap.py --dataset all
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR     = PROJECT_DIR / "figures" / "shap"

TOP_N   = 20
TOP_DEP = 5

# Base style constants (journal target)
TICK_SIZE   = 9
LABEL_SIZE  = 10
LEGEND_SIZE = 9
PANEL_SIZE  = 11
SPINE_LW    = 1.2
DPI         = 600

# Journal style constants
J_TICK_SIZE = 9
J_LABEL_SIZE = 10
J_LEGEND_SIZE = 9
J_PANEL_SIZE = 11
J_SPINE_LW = 1.2
J_SINGLE_W = 3.4
J_DOUBLE_W = 7.0


# ---------------------------------------------------------------------------
# Helpers (identical to compute_shap.py)
# ---------------------------------------------------------------------------

def clean_name(n):
    state_label = ""
    for pre, label in [("neutral_", "neut_"), ("protonated_", "prot_"), ("delta_", "delt_")]:
        if n.startswith(pre):
            state_label = label
            n = n[len(pre):]
            break
    for cat in ("rdkit_", "mordred_", "pm7_"):
        if n.startswith(cat):
            n = n[len(cat):]
            break
    PM7_PROPS = ("HOMO", "LUMO", "dipole", "HOF", "ionization",
                 "cosmo", "total_energy", "n_atoms")
    if any(n.startswith(p) for p in PM7_PROPS):
        return state_label + n
    return n


def rcparams():
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          TICK_SIZE,
        "axes.linewidth":     SPINE_LW,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.labelsize":     LABEL_SIZE,
        "axes.titlesize":     LABEL_SIZE,
        "xtick.labelsize":    TICK_SIZE,
        "ytick.labelsize":    TICK_SIZE,
        "xtick.major.width":  SPINE_LW,
        "ytick.major.width":  SPINE_LW,
        "xtick.minor.width":  SPINE_LW * 0.5,
        "ytick.minor.width":  SPINE_LW * 0.5,
        "xtick.major.size":   7,
        "ytick.major.size":   7,
        "xtick.minor.size":   4,
        "ytick.minor.size":   4,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "legend.fontsize":    LEGEND_SIZE,
        "figure.dpi":         300,
        "savefig.dpi":        DPI,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })

def rcparams_journal():
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          J_TICK_SIZE,
        "axes.linewidth":     J_SPINE_LW,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.labelsize":     J_LABEL_SIZE,
        "axes.titlesize":     J_LABEL_SIZE,
        "xtick.labelsize":    J_TICK_SIZE,
        "ytick.labelsize":    J_TICK_SIZE,
        "xtick.major.width":  J_SPINE_LW,
        "ytick.major.width":  J_SPINE_LW,
        "xtick.minor.width":  J_SPINE_LW * 0.7,
        "ytick.minor.width":  J_SPINE_LW * 0.7,
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "xtick.minor.size":   2.0,
        "ytick.minor.size":   2.0,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "legend.fontsize":    J_LEGEND_SIZE,
        "figure.dpi":         300,
        "savefig.dpi":        DPI,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.04,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


def top_indices(shap_values, n):
    return np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:n]


def plot_beeswarm(sv, X, names, title, out, top_n=TOP_N):
    rcparams_journal()
    top_n = min(top_n, len(names))
    idx   = top_indices(sv, top_n)
    sv_t  = sv[:, idx]
    X_t   = X[:, idx]
    n_t   = [clean_name(names[i]) for i in idx]

    fig, ax = plt.subplots(figsize=(J_SINGLE_W, max(2.8, top_n * 0.22)))
    fig.patch.set_facecolor("white")
    plt.sca(ax)

    shap.summary_plot(sv_t, X_t, feature_names=n_t,
                      plot_type="dot", max_display=top_n,
                      show=False, color_bar=True, plot_size=None)

    ax = plt.gca()
    ax.set_xlabel("SHAP value\n(kcal/mol)", fontsize=J_LABEL_SIZE, labelpad=6)
    ax.set_title(title, fontsize=J_LABEL_SIZE, fontweight="bold", pad=8)
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=J_TICK_SIZE,
                   width=J_SPINE_LW, length=3.5)

    cb_ax = plt.gcf().axes[-1]
    if cb_ax != ax:
        cb_ax.tick_params(labelsize=J_TICK_SIZE)
        cb_ax.set_ylabel("Feature value", fontsize=J_LABEL_SIZE, labelpad=6)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    log.info(f"  Saved {out}")
    plt.close("all")


def plot_importance(sv, names, title, out, top_n=TOP_N):
    rcparams_journal()
    mean_abs = np.abs(sv).mean(axis=0)
    top_n = min(top_n, len(names))
    idx  = np.argsort(mean_abs)[::-1][:top_n]
    vals = mean_abs[idx]
    nms  = [clean_name(names[i]) for i in idx]

    fig, ax = plt.subplots(figsize=(J_SINGLE_W, max(2.8, top_n * 0.22)))
    fig.patch.set_facecolor("white")
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, top_n))

    ax.barh(range(top_n), vals[::-1], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(nms[::-1], fontsize=J_TICK_SIZE)
    ax.set_xlabel("Mean |SHAP value|\n(kcal/mol)", fontsize=J_LABEL_SIZE, labelpad=6)
    ax.set_title(title, fontsize=J_LABEL_SIZE, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=J_TICK_SIZE,
                   width=J_SPINE_LW, length=3.5)
    ax.grid(axis="x", lw=0.6, alpha=0.35, ls="--")

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    log.info(f"  Saved {out}")
    plt.close("all")


def plot_dependence(sv, X, names, title, out, top_n=TOP_DEP):
    rcparams_journal()
    top_n = min(top_n, len(names))
    idx = top_indices(sv, top_n)

    # Journal layout: taller canvas, single shared colorbar.
    fig, axes = plt.subplots(1, top_n, figsize=(J_DOUBLE_W, 2.2),
                             sharey=True, constrained_layout=False)
    fig.patch.set_facecolor("white")

    if top_n == 1:
        axes = [axes]

    # Compute a single shared SHAP color scale.
    vmax_global = np.max([np.abs(sv[:, fi]).max() for fi in idx])

    def _wrap_feature_name(name: str, max_chars: int = 14) -> str:
        if len(name) <= max_chars:
            return name
        # Prefer splitting on a separator near the midpoint.
        seps = ["_", "-", " ", ".", "/"]
        mid = len(name) // 2
        best = None
        for i, ch in enumerate(name):
            if ch in seps and abs(i - mid) < abs((best if best is not None else -1) - mid):
                best = i
        if best is not None and 0 < best < len(name) - 1:
            return name[:best + 1] + "\n" + name[best + 1:]
        return name[:mid] + "\n" + name[mid:]

    sc_last = None
    for i, (ax, fi) in enumerate(zip(axes, idx)):
        fname = _wrap_feature_name(clean_name(names[fi]))
        fv   = X[:, fi]
        sv_f = sv[:, fi]

        sc = ax.scatter(fv, sv_f, c=sv_f, cmap="coolwarm",
                        s=14, alpha=0.8, linewidths=0,
                        vmin=-vmax_global, vmax=vmax_global)
        sc_last = sc
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)

        ax.set_xlabel(fname, fontsize=J_LABEL_SIZE, labelpad=3)
        if i == 0:
            ax.set_ylabel("SHAP value\n(kcal/mol)",
                          fontsize=J_LABEL_SIZE, labelpad=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=J_TICK_SIZE,
                       width=J_SPINE_LW, length=3.0)
        ax.locator_params(axis="x", nbins=3)
        ax.locator_params(axis="y", nbins=5)

    fig.suptitle(title, fontsize=J_LABEL_SIZE, fontweight="bold", y=0.995)

    # Layout with room for suptitle and shared colorbar on the right.
    fig.tight_layout(rect=(0.0, 0.0, 0.93, 0.92), w_pad=1.6)
    cbar_ax = fig.add_axes([0.945, 0.18, 0.012, 0.66])
    cbar = fig.colorbar(sc_last, cax=cbar_ax)
    cbar.set_label("SHAP value\n(kcal/mol)",
                   fontsize=J_LABEL_SIZE, labelpad=4)
    cbar.ax.tick_params(labelsize=J_TICK_SIZE, width=J_SPINE_LW, length=3.0)

    plt.savefig(out, bbox_inches="tight", facecolor="white")
    log.info(f"  Saved {out}")
    plt.close("all")


def plot_dependence_combined(sv_n, X_n, nm_n, sv_k, X_k, nm_k, out,
                             top_n=TOP_DEP):
    """Two-row dependence figure: (a) NIST, (b) k-means. No per-plot titles."""
    rcparams_journal()

    def _wrap_feature_name(name: str, max_chars: int = 14) -> str:
        if len(name) <= max_chars:
            return name
        seps = ["_", "-", " ", ".", "/"]
        mid = len(name) // 2
        best = None
        for i, ch in enumerate(name):
            if ch in seps and abs(i - mid) < abs(
                (best if best is not None else -1) - mid
            ):
                best = i
        if best is not None and 0 < best < len(name) - 1:
            return name[:best + 1] + "\n" + name[best + 1:]
        return name[:mid] + "\n" + name[mid:]

    top_n_n = min(top_n, len(nm_n))
    top_n_k = min(top_n, len(nm_k))
    ncols = max(top_n_n, top_n_k)

    fig, axes = plt.subplots(2, ncols, figsize=(J_DOUBLE_W, 4.0))
    fig.patch.set_facecolor("white")

    # Global color scale across both rows so the shared colorbar is meaningful.
    idx_n = top_indices(sv_n, top_n_n)
    idx_k = top_indices(sv_k, top_n_k)
    vmax_global = max(
        np.max([np.abs(sv_n[:, fi]).max() for fi in idx_n]),
        np.max([np.abs(sv_k[:, fi]).max() for fi in idx_k]),
    )

    def _draw_row(row_axes, sv, X, names, idx, panel_letter):
        top_n_local = len(idx)
        sc_last = None
        for j in range(ncols):
            ax = row_axes[j]
            if j >= top_n_local:
                ax.set_visible(False)
                continue
            fi = idx[j]
            fname = _wrap_feature_name(clean_name(names[fi]))
            fv = X[:, fi]
            sv_f = sv[:, fi]

            sc = ax.scatter(fv, sv_f, c=sv_f, cmap="coolwarm",
                            s=14, alpha=0.8, linewidths=0,
                            vmin=-vmax_global, vmax=vmax_global)
            sc_last = sc
            ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)

            ax.set_xlabel(fname, fontsize=J_LABEL_SIZE, labelpad=3)
            if j == 0:
                ax.set_ylabel("SHAP value\n(kcal/mol)",
                              fontsize=J_LABEL_SIZE, labelpad=3)
                ax.text(-0.42, 1.05, panel_letter,
                        transform=ax.transAxes, fontsize=J_PANEL_SIZE,
                        fontweight="bold", va="top")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", labelsize=J_TICK_SIZE,
                           width=J_SPINE_LW, length=3.0)
            ax.locator_params(axis="x", nbins=3)
            ax.locator_params(axis="y", nbins=5)
        return sc_last

    _draw_row(axes[0], sv_n, X_n, nm_n, idx_n, "(a)")
    sc_last = _draw_row(axes[1], sv_k, X_k, nm_k, idx_k, "(b)")

    # Share y-axis limits within each row for consistent reading across panels.
    for row in axes:
        visible = [ax for ax in row if ax.get_visible()]
        if not visible:
            continue
        lo = min(ax.get_ylim()[0] for ax in visible)
        hi = max(ax.get_ylim()[1] for ax in visible)
        for ax in visible:
            ax.set_ylim(lo, hi)

    fig.tight_layout(rect=(0.0, 0.0, 0.93, 1.0), w_pad=1.6, h_pad=1.4)
    cbar_ax = fig.add_axes([0.945, 0.16, 0.012, 0.70])
    cbar = fig.colorbar(sc_last, cax=cbar_ax)
    cbar.set_label("SHAP value\n(kcal/mol)",
                   fontsize=J_LABEL_SIZE, labelpad=4)
    cbar.ax.tick_params(labelsize=J_TICK_SIZE,
                        width=J_SPINE_LW, length=3.0)

    plt.savefig(out, bbox_inches="tight", facecolor="white")
    log.info(f"  Saved {out}")
    plt.close("all")


def plot_combined_beeswarm(sv_n, X_n, nm_n, sv_k, X_k, nm_k, out, top_n=TOP_N):
    """Side-by-side beeswarm for NIST and k-means."""
    rcparams_journal()

    fig, axes = plt.subplots(1, 2, figsize=(J_DOUBLE_W, max(4.2, top_n * 0.23)))
    fig.patch.set_facecolor("white")

    panel_labels = ["(a)", "(b)"]

    for panel_idx, (ax, sv, X, names, title) in enumerate([
        (axes[0], sv_n, X_n, nm_n, "NIST dataset"),
        (axes[1], sv_k, X_k, nm_k, "k-means dataset"),
    ]):
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
        ax.set_xlabel("SHAP value\n(kcal/mol)", fontsize=J_LABEL_SIZE, labelpad=6)
        ax.set_title(f"{panel_labels[panel_idx]} {title}",
                     fontsize=J_LABEL_SIZE, fontweight="bold", pad=10)
        ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(J_SPINE_LW)
        ax.tick_params(axis="both", which="major", labelsize=J_TICK_SIZE,
                       width=J_SPINE_LW, length=3.5)
        ax.tick_params(axis="both", which="minor",
                       width=J_SPINE_LW * 0.7, length=2.0)

        # Style the colorbar
        cb_ax = plt.gcf().axes[-1]
        if cb_ax != ax:
            cb_ax.tick_params(labelsize=J_TICK_SIZE)
            cb_ax.set_ylabel("Feature value", fontsize=J_LABEL_SIZE, labelpad=6)

    plt.tight_layout(pad=1.0, w_pad=1.2)
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    log.info(f"  Saved {out}")
    plt.close("all")


def plot_paper_figure(sv_n, X_n, nm_n, sv_k, X_k, nm_k, out,
                      top_n_nist=15, top_n_km=9):
    """4-panel paper figure: (a,b) beeswarms, (c,d) importance bars."""
    rcparams_journal()

    fig, axes = plt.subplots(2, 2, figsize=(J_DOUBLE_W, 5.8))
    fig.patch.set_facecolor("white")
    labels = ["(a)", "(b)", "(c)", "(d)"]

    # (a) NIST beeswarm
    plt.sca(axes[0, 0])
    tn = min(top_n_nist, len(nm_n))
    idx = top_indices(sv_n, tn)
    shap.summary_plot(sv_n[:, idx], X_n[:, idx],
                      feature_names=[clean_name(nm_n[i]) for i in idx],
                      plot_type="dot", max_display=tn, show=False,
                      color_bar=True, plot_size=None)
    ax = plt.gca(); ax.set_facecolor("white")
    ax.set_xlabel("SHAP value (kcal/mol)", fontsize=LABEL_SIZE, labelpad=15)
    ax.set_title("NIST — Molecular + PM7 features", fontsize=LABEL_SIZE,
                 fontweight="bold", pad=25)
    ax.axvline(0, color="black", lw=2.5, ls="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.15, 1.08, labels[0], transform=ax.transAxes,
            fontsize=PANEL_SIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_SIZE,
                   width=SPINE_LW, length=7)

    # (b) k-means beeswarm
    plt.sca(axes[0, 1])
    tn_k = min(top_n_km, len(nm_k))
    idx_k = top_indices(sv_k, tn_k)
    shap.summary_plot(sv_k[:, idx_k], X_k[:, idx_k],
                      feature_names=[clean_name(nm_k[i]) for i in idx_k],
                      plot_type="dot", max_display=tn_k, show=False,
                      color_bar=True, plot_size=None)
    ax = plt.gca(); ax.set_facecolor("white")
    ax.set_xlabel("SHAP value (kcal/mol)", fontsize=LABEL_SIZE, labelpad=15)
    ax.set_title("k-means — Molecular + PM7 features", fontsize=LABEL_SIZE,
                 fontweight="bold", pad=25)
    ax.axvline(0, color="black", lw=2.5, ls="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.15, 1.08, labels[1], transform=ax.transAxes,
            fontsize=PANEL_SIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_SIZE,
                   width=SPINE_LW, length=7)

    # (c) NIST importance bar
    ax = axes[1, 0]; ax.set_facecolor("white")
    ma = np.abs(sv_n).mean(axis=0)
    tn2 = min(top_n_nist, len(nm_n))
    idx2 = np.argsort(ma)[::-1][:tn2]
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, tn2))
    ax.barh(range(tn2), ma[idx2][::-1], color=colors)
    ax.set_yticks(range(tn2))
    ax.set_yticklabels([clean_name(nm_n[i]) for i in idx2][::-1], fontsize=TICK_SIZE)
    ax.set_xlabel("Mean |SHAP value| (kcal/mol)", fontsize=LABEL_SIZE, labelpad=15)
    ax.set_title("Feature importance — NIST", fontsize=LABEL_SIZE,
                 fontweight="bold", pad=25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_SIZE,
                   width=SPINE_LW, length=7)
    ax.grid(axis="x", lw=0.6, alpha=0.35, ls="--")
    ax.text(-0.15, 1.08, labels[2], transform=ax.transAxes,
            fontsize=PANEL_SIZE, fontweight="bold")

    # (d) k-means importance bar
    ax = axes[1, 1]; ax.set_facecolor("white")
    ma_k = np.abs(sv_k).mean(axis=0)
    tn_k2 = min(top_n_km, len(nm_k))
    idx3 = np.argsort(ma_k)[::-1][:tn_k2]
    colors_k = plt.cm.Blues(np.linspace(0.35, 0.85, tn_k2))
    ax.barh(range(tn_k2), ma_k[idx3][::-1], color=colors_k)
    ax.set_yticks(range(tn_k2))
    ax.set_yticklabels([clean_name(nm_k[i]) for i in idx3][::-1], fontsize=TICK_SIZE)
    ax.set_xlabel("Mean |SHAP value| (kcal/mol)", fontsize=LABEL_SIZE, labelpad=15)
    ax.set_title("Feature importance — k-means", fontsize=LABEL_SIZE,
                 fontweight="bold", pad=25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_SIZE,
                   width=SPINE_LW, length=7)
    ax.grid(axis="x", lw=0.6, alpha=0.35, ls="--")
    ax.text(-0.15, 1.08, labels[3], transform=ax.transAxes,
            fontsize=PANEL_SIZE, fontweight="bold")

    plt.tight_layout(pad=2.0, w_pad=5.0, h_pad=4.0)
    plt.savefig(out, bbox_inches="tight", facecolor="white", dpi=DPI)
    log.info(f"  Saved {out}")
    plt.close("all")


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_shap_data(results_dir: Path, dataset: str):
    """
    Load pre-computed SHAP values and feature values from CSV.
    Returns (shap_values ndarray, feature_values ndarray, feature_names list).
    """
    shap_dir = results_dir / dataset / "shap"
    sv_path = shap_dir / "shap_values.csv"
    X_path  = shap_dir / "shap_feature_values.csv"
    if not sv_path.exists():
        raise FileNotFoundError(
            f"Not found: {sv_path}\n"
            "Run compute_shap.py first to generate the data."
        )
    sv_df = pd.read_csv(sv_path)
    X_df  = pd.read_csv(X_path)
    names = list(sv_df.columns)
    log.info(f"  Loaded {dataset}: {sv_df.shape[0]} samples, {len(names)} features")
    return sv_df.values, X_df.values, names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate SHAP figures from saved results."
    )
    parser.add_argument("--dataset", default="all",
                        choices=["all", "nist", "kmeans"])
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sv_n = X_n = nm_n = None
    sv_k = X_k = nm_k = None

    if args.dataset in ("all", "nist"):
        log.info("Loading NIST SHAP data ...")
        sv_n, X_n, nm_n = load_shap_data(RESULTS_DIR, "nist1155")
        plot_beeswarm(sv_n, X_n, nm_n,
                      "SHAP summary — NIST (Molecular + PM7 features)",
                      FIG_DIR / "shap_beeswarm_nist.pdf")
        plot_importance(sv_n, nm_n,
                        "Feature importance — NIST (Molecular + PM7 features)",
                        FIG_DIR / "shap_importance_nist.pdf")
        plot_dependence(sv_n, X_n, nm_n,
                        f"SHAP dependence — top {TOP_DEP} features (NIST)",
                        FIG_DIR / "shap_dependence_nist.pdf")

    if args.dataset in ("all", "kmeans"):
        log.info("Loading k-means SHAP data ...")
        sv_k, X_k, nm_k = load_shap_data(RESULTS_DIR, "kmeans251")
        plot_beeswarm(sv_k, X_k, nm_k,
                      "SHAP summary — k-means (Molecular + PM7 features)",
                      FIG_DIR / "shap_beeswarm_kmeans.pdf")
        plot_importance(sv_k, nm_k,
                        "Feature importance — k-means (Molecular + PM7 features)",
                        FIG_DIR / "shap_importance_kmeans.pdf")
        plot_dependence(sv_k, X_k, nm_k,
                        f"SHAP dependence — top {TOP_DEP} features (k-means)",
                        FIG_DIR / "shap_dependence_kmeans.pdf")

    if sv_n is not None and sv_k is not None:
        log.info("Generating combined beeswarm ...")
        plot_combined_beeswarm(sv_n, X_n, nm_n, sv_k, X_k, nm_k,
                               FIG_DIR / "shap_beeswarm_combined.pdf")
        log.info("Generating combined dependence figure ...")
        plot_dependence_combined(sv_n, X_n, nm_n, sv_k, X_k, nm_k,
                                 FIG_DIR / "shap_dependence_combined.pdf")
        log.info("Generating 4-panel paper figure ...")
        plot_paper_figure(sv_n, X_n, nm_n, sv_k, X_k, nm_k,
                          FIG_DIR / "shap_paper_figure.pdf")

    print(f"\n  Done. All SHAP figures in {FIG_DIR}/")
    for f in sorted(FIG_DIR.glob("*.pdf")):
        print(f"    {f.name}")
