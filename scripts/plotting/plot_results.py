"""
plot_results.py
===============
Model performance parity plots — combined layout.
Font sizes and style matched to exploration/parity_combined.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Directory setup
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR     = PROJECT_DIR / "figures"
FIG_PERF    = FIG_DIR / "model_performance"

KJMOL_TO_KCAL = 1 / 4.184

# ── Style (matched to exploration/parity_combined) ──────────────────────────
TICK_SIZE   = 30
LABEL_SIZE  = 34
LEGEND_SIZE = 24
PANEL_SIZE  = 40
SPINE_LW    = 3.8
DPI         = 600

# Colors by dataset + feature setting
COLOR_NIST_NO_DFT   = "#2166AC"
COLOR_NIST_WITH_DFT = "#1B9E77"
COLOR_KM_NO_DFT     = "#D01C8B"
COLOR_KM_WITH_DFT   = "#E67E22"

# Softer paired colors for model-comparison bars (e/f panels)
MC_NO_DFT_COLOR   = "#5DA5DA"
MC_WITH_DFT_COLOR = "#F28E2B"

# Typography for individual a-f exports
IND_TICK_SIZE = 10
IND_LABEL_SIZE = 11
IND_LEGEND_SIZE = 10
IND_PANEL_SIZE = 12

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth":     SPINE_LW,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
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
    "xtick.labelsize":    TICK_SIZE,
    "ytick.labelsize":    TICK_SIZE,
    "axes.labelsize":     LABEL_SIZE,
    "figure.dpi":         300,
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# ── Data helpers ──────────────────────────────────────────────────────────────

def load_predictions(dataset_name: str) -> pd.DataFrame:
    path = RESULTS_DIR / dataset_name / "predictions.csv"
    return pd.read_csv(path) if path.exists() else None

def load_mae_summary(dataset_name: str) -> pd.DataFrame:
    path = RESULTS_DIR / dataset_name / "mae_summary.csv"
    return pd.read_csv(path) if path.exists() else None

def get_best_model(dataset_name: str) -> str:
    return "ExtraTrees"

def compute_overall_mae(preds: pd.DataFrame, model: str) -> tuple:
    df = preds[preds["model"] == model].copy()
    df["pa_pred_kcal"] = df["pa_pred"] * KJMOL_TO_KCAL
    df["pa_true_kcal"] = df["pa_true"] * KJMOL_TO_KCAL
    fold_maes = df.groupby("fold").apply(
        lambda g: np.mean(np.abs(g["pa_pred_kcal"] - g["pa_true_kcal"]))
    ).values
    return float(np.mean(fold_maes)), float(np.std(fold_maes))

# ── Plotting logic ────────────────────────────────────────────────────────────

def make_combined_parity(has_dft: bool):
    configs = [
        ("nist1155",     "NIST (No DFT Features)",      "Exp. PA (kcal/mol)", "Pred. PA (kcal/mol)", COLOR_NIST_NO_DFT),
        ("nist1155_dft", "NIST (With DFT Features)",    "Exp. PA (kcal/mol)", "Pred. PA (kcal/mol)", COLOR_NIST_WITH_DFT),
        ("kmeans251",    "k-means (No DFT Features)",   "DFT PA (kcal/mol)",  "Pred. PA (kcal/mol)", COLOR_KM_NO_DFT),
        ("kmeans251_dft","k-means (With DFT Features)", "DFT PA (kcal/mol)",  "Pred. PA (kcal/mol)", COLOR_KM_WITH_DFT),
    ]
    if not has_dft:
        configs = [c for c in configs if "dft" not in c[0]]

    ncols = 3 if has_dft else len(configs)
    nrows = 2 if has_dft else 1
    panel_w, panel_h = (10.8 if has_dft else 10), (10.8 if has_dft else 10)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_w * ncols, panel_h * nrows),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]} if has_dft else None,
    )
    axes = np.array(axes)

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    parity_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]] if has_dft else list(np.array(axes).flatten())
    PANEL_Y_TOP = 1.25
    PANEL_Y_BOTTOM = 1.20

    for ax_idx, (dname, p_title, xlbl, ylbl, color) in enumerate(configs):
        ax = parity_axes[ax_idx]
        preds = load_predictions(dname)
        best_model = get_best_model(dname)
        if preds is None or best_model is None:
            continue

        df_p = preds[preds["model"] == best_model].copy()
        df_p["pa_pred_kcal"] = df_p["pa_pred"] * KJMOL_TO_KCAL
        df_p["pa_true_kcal"] = df_p["pa_true"] * KJMOL_TO_KCAL

        mae_m, mae_s = compute_overall_mae(preds, best_model)
        r2 = r2_score(df_p["pa_true_kcal"], df_p["pa_pred_kcal"])
        n_total = len(df_p)
        # Use molecule count for k-means (predictions are site-level there).
        if "kmeans" in dname and "neutral_smiles" in df_p.columns:
            n_display = int(df_p["neutral_smiles"].nunique())
            n_line = f"N = {n_display} | {n_total} Sites"
        else:
            n_display = n_total
            n_line = f"N = {n_display}"

        # Limits
        pad = (df_p["pa_true_kcal"].max() - df_p["pa_true_kcal"].min()) * 0.06
        lo = df_p["pa_true_kcal"].min() - pad
        hi = df_p["pa_true_kcal"].max() + pad
        lims = (lo, hi)

        # 1:1 line
        ax.plot(lims, lims, "k--", lw=1.8, zorder=1)

        # Scatter by fold with distinct markers.
        fold_markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
        unique_folds = sorted(df_p["fold"].dropna().unique())
        for i, fold in enumerate(unique_folds):
            sub = df_p[df_p["fold"] == fold]
            ax.scatter(
                sub["pa_true_kcal"], sub["pa_pred_kcal"],
                c=color, s=55, alpha=0.85,
                marker=fold_markers[i % len(fold_markers)],
                edgecolors="black", linewidths=0.4, zorder=2
            )

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel(xlbl, fontsize=LABEL_SIZE)
        ax.set_ylabel(ylbl, fontsize=LABEL_SIZE)

        # 5 consistent ticks
        ticks = np.linspace(lims[0], lims[1], 5)
        ticks = np.round(ticks).astype(int)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        ax.tick_params(axis="x", which="major", labelsize=TICK_SIZE,
                       width=SPINE_LW, length=7, pad=10)
        ax.tick_params(axis="y", which="major", labelsize=TICK_SIZE,
                       width=SPINE_LW, length=7, pad=10)
        ax.tick_params(axis="both", which="minor", width=SPINE_LW * 0.5,
                       length=4)

        # Stats box — bold text, dark border, bottom right
        stats_text = (
            f"MAE = {mae_m:.2f} \u00b1 {mae_s:.2f}\n"
            f"R\u00b2 = {r2:.3f}\n"
            f"{n_line}"
        )
        ax.text(0.985, 0.04, stats_text,
                transform=ax.transAxes, fontsize=LEGEND_SIZE,
                fontweight="bold", va="bottom", ha="right",
                multialignment="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="black", lw=2.0, alpha=0.95))

        # Panel label + title
        panel_y = PANEL_Y_TOP if (has_dft and ax_idx < 2) else PANEL_Y_BOTTOM
        ax.text(-0.22, panel_y, panel_labels[ax_idx],
                transform=ax.transAxes,
                fontsize=PANEL_SIZE, fontweight="bold", va="top")
        ax.set_title(p_title, fontsize=LABEL_SIZE, fontweight="bold", pad=25)

    if has_dft:
        def _draw_model_comparison(ax, dataset_base, title, panel_label, color_no, color_with, panel_y):
            no_dft = load_mae_summary(dataset_base)
            with_dft = load_mae_summary(f"{dataset_base}_dft")
            if no_dft is None or with_dft is None:
                ax.text(0.5, 0.5, "Model comparison data unavailable",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=LABEL_SIZE)
                return

            comp = no_dft[["model", "mae_pa_mean", "mae_pa_std"]].rename(
                columns={"mae_pa_mean": "mae_no", "mae_pa_std": "std_no"}
            )
            comp = comp.merge(
                with_dft[["model", "mae_pa_mean", "mae_pa_std"]].rename(
                    columns={"mae_pa_mean": "mae_with", "mae_pa_std": "std_with"}
                ),
                on="model", how="inner"
            )
            # Hide VotingEnsemble from the comparison panels.
            comp = comp[comp["model"] != "VotingEnsemble"]
            comp = comp.sort_values("mae_no", ascending=True).reset_index(drop=True)

            x = np.arange(len(comp))
            w = 0.34
            ax.bar(
                x - w / 2, comp["mae_no"], width=w,
                yerr=comp["std_no"], capsize=4,
                color=MC_NO_DFT_COLOR, edgecolor="black", linewidth=0.7, alpha=0.9,
                label="No DFT", zorder=2
            )
            ax.bar(
                x + w / 2, comp["mae_with"], width=w,
                yerr=comp["std_with"], capsize=4,
                color=MC_WITH_DFT_COLOR, edgecolor="black", linewidth=0.7, alpha=0.9,
                hatch="xx",
                label="With DFT", zorder=3
            )

            ax.set_xticks(x)
            ax.set_xticklabels(comp["model"], rotation=45, ha="right", fontsize=TICK_SIZE)
            ax.set_ylabel("MAE", fontsize=LABEL_SIZE)
            ax.set_title(title, fontsize=LABEL_SIZE, fontweight="bold", pad=25)
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            ax.tick_params(axis="x", which="major", labelsize=TICK_SIZE,
                           width=SPINE_LW, length=7, pad=8)
            ax.tick_params(axis="y", which="major", labelsize=TICK_SIZE,
                           width=SPINE_LW, length=7, pad=8)
            ax.tick_params(axis="both", which="minor", width=SPINE_LW * 0.5, length=4)
            ax.grid(axis="y", linewidth=0.6, alpha=0.35, linestyle="--")
            ax.legend(loc="upper left", fontsize=LEGEND_SIZE, frameon=True,
                      edgecolor="black", facecolor="white")
            ax.text(-0.22, panel_y, panel_label, transform=ax.transAxes,
                    fontsize=PANEL_SIZE, fontweight="bold", va="top")

        _draw_model_comparison(
            axes[0, 2], "nist1155", "NIST model comparison", panel_labels[4],
            COLOR_NIST_NO_DFT, COLOR_NIST_WITH_DFT, PANEL_Y_TOP
        )
        _draw_model_comparison(
            axes[1, 2], "kmeans251", "k-means model comparison", panel_labels[5],
            COLOR_KM_NO_DFT, COLOR_KM_WITH_DFT, PANEL_Y_BOTTOM
        )

    plt.subplots_adjust(hspace=0.72, wspace=0.50)

    stem = "parity_combined" if has_dft else "parity_combined_pm7only"
    FIG_PERF.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PERF / f"{stem}.pdf", bbox_inches="tight")
    log.info(f"  Saved {FIG_PERF / stem}.pdf")
    plt.close(fig)


def _draw_single_parity(
    ax, dname, p_title, xlbl, ylbl, color,
    show_title=False, title_fontsize=None, metrics_fontsize=None,
    scatter_size=None, metrics_box_pad=0.25
):
    preds = load_predictions(dname)
    best_model = get_best_model(dname)
    if preds is None or best_model is None:
        ax.text(0.5, 0.5, "Prediction data unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=IND_LABEL_SIZE)
        return

    df_p = preds[preds["model"] == best_model].copy()
    df_p["pa_pred_kcal"] = df_p["pa_pred"] * KJMOL_TO_KCAL
    df_p["pa_true_kcal"] = df_p["pa_true"] * KJMOL_TO_KCAL

    mae_m, mae_s = compute_overall_mae(preds, best_model)
    r2 = r2_score(df_p["pa_true_kcal"], df_p["pa_pred_kcal"])
    n_total = len(df_p)
    if "kmeans" in dname and "neutral_smiles" in df_p.columns:
        n_display = int(df_p["neutral_smiles"].nunique())
        n_line = f"N = {n_display} | {n_total} Sites"
    else:
        n_line = f"N = {n_total}"

    pad = (df_p["pa_true_kcal"].max() - df_p["pa_true_kcal"].min()) * 0.06
    lo = df_p["pa_true_kcal"].min() - pad
    hi = df_p["pa_true_kcal"].max() + pad
    lims = (lo, hi)

    ax.plot(lims, lims, "k--", lw=1.0, zorder=1)

    fold_markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    unique_folds = sorted(df_p["fold"].dropna().unique())
    for i, fold in enumerate(unique_folds):
        sub = df_p[df_p["fold"] == fold]
        ax.scatter(
            sub["pa_true_kcal"], sub["pa_pred_kcal"],
            c=color, s=(scatter_size or 14), alpha=0.88,
            marker=fold_markers[i % len(fold_markers)],
            edgecolors="black", linewidths=0.25, zorder=2
        )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    if show_title and p_title:
        ax.set_title(p_title, fontsize=(title_fontsize or IND_LABEL_SIZE),
                     fontweight="bold", pad=6)
    xlbl_wrapped = xlbl.replace(" (kcal/mol)", "\n(kcal/mol)")
    ylbl_wrapped = ylbl.replace(" (kcal/mol)", "\n(kcal/mol)")
    # Extra space between axis tick labels and axis titles (units on line 2).
    ax.set_xlabel(xlbl_wrapped, fontsize=IND_LABEL_SIZE, labelpad=6)
    ax.set_ylabel(ylbl_wrapped, fontsize=IND_LABEL_SIZE, labelpad=6)

    ticks = np.linspace(lims[0], lims[1], 5)
    ticks = np.round(ticks).astype(int)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, rotation=25, ha="right")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(axis="x", which="major", labelsize=IND_TICK_SIZE,
                   width=1.2, length=3.5, pad=4,
                   colors="black", labelcolor="black")
    ax.tick_params(axis="y", which="major", labelsize=IND_TICK_SIZE,
                   width=1.2, length=3.5, pad=4,
                   colors="black", labelcolor="black")
    ax.tick_params(axis="both", which="minor", width=0.8, length=2.0,
                   colors="black")

    stats_text = (
        f"MAE = {mae_m:.2f} ± {mae_s:.2f}\n"
        f"R² = {r2:.3f}\n"
        f"{n_line}"
    )
    # Put metrics inside plot (bottom-right) with boxed legend styling.
    ax.text(0.97, 0.03, stats_text,
            transform=ax.transAxes, fontsize=(metrics_fontsize or IND_LEGEND_SIZE),
            fontweight="bold", va="bottom", ha="right",
            multialignment="left",
            bbox=dict(boxstyle=f"round,pad={metrics_box_pad}", fc="white",
                      ec="black", lw=0.9, alpha=0.95))


def _draw_single_model_comparison(
    ax, dataset_base, title=None, show_title=False, title_fontsize=None
):
    no_dft = load_mae_summary(dataset_base)
    with_dft = load_mae_summary(f"{dataset_base}_dft")
    if no_dft is None or with_dft is None:
        ax.text(0.5, 0.5, "Model comparison data unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=IND_LABEL_SIZE)
        return

    comp = no_dft[["model", "mae_pa_mean", "mae_pa_std"]].rename(
        columns={"mae_pa_mean": "mae_no", "mae_pa_std": "std_no"}
    )
    comp = comp.merge(
        with_dft[["model", "mae_pa_mean", "mae_pa_std"]].rename(
            columns={"mae_pa_mean": "mae_with", "mae_pa_std": "std_with"}
        ),
        on="model", how="inner"
    )
    comp = comp[comp["model"] != "VotingEnsemble"]
    comp = comp.sort_values("mae_no", ascending=True).reset_index(drop=True)

    # Increase category spacing a bit so grouped bars and labels breathe more.
    spacing = 1.18
    x = np.arange(len(comp)) * spacing
    w = 0.30
    ax.bar(
        x - w / 2, comp["mae_no"], width=w,
        yerr=comp["std_no"], capsize=4,
        color=MC_NO_DFT_COLOR, edgecolor="black", linewidth=0.7, alpha=0.9,
        label="No DFT", zorder=2
    )
    ax.bar(
        x + w / 2, comp["mae_with"], width=w,
        yerr=comp["std_with"], capsize=4,
        color=MC_WITH_DFT_COLOR, edgecolor="black", linewidth=0.7, alpha=0.9,
        hatch="xx",
        label="With DFT", zorder=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels(comp["model"], rotation=45, ha="right", fontsize=IND_TICK_SIZE)
    ax.set_ylabel("MAE", fontsize=IND_LABEL_SIZE)
    if show_title and title:
        ax.set_title(title, fontsize=(title_fontsize or IND_LABEL_SIZE),
                     fontweight="bold", pad=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    # Keep several major ticks visible even when MAE range is narrow.
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, min_n_ticks=4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(axis="x", which="major", labelsize=IND_TICK_SIZE,
                   width=1.2, length=3.5, pad=4)
    ax.tick_params(axis="y", which="major", labelsize=IND_TICK_SIZE,
                   width=1.2, length=3.5, pad=4)
    ax.tick_params(axis="both", which="minor", width=0.8, length=2.0)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35, linestyle="--")
    leg = ax.legend(loc="upper left",
                    fontsize=IND_LEGEND_SIZE, frameon=True,
                    edgecolor="black", facecolor="white")
    leg.get_frame().set_linewidth(1.0)


def make_individual_a_to_f():
    out_dir = FIG_PERF / "individual_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    parity_cfg = [
        ("a", "nist1155", "NIST\n(No DFT Features)", "Exp. PA (kcal/mol)", "Pred. PA (kcal/mol)", COLOR_NIST_NO_DFT),
        ("b", "nist1155_dft", "NIST\n(With DFT Features)", "Exp. PA (kcal/mol)", "Pred. PA (kcal/mol)", COLOR_NIST_WITH_DFT),
        ("c", "kmeans251", "k-means\n(No DFT Features)", "DFT PA (kcal/mol)", "Pred. PA (kcal/mol)", COLOR_KM_NO_DFT),
        ("d", "kmeans251_dft", "k-means\n(With DFT Features)", "DFT PA (kcal/mol)", "Pred. PA (kcal/mol)", COLOR_KM_WITH_DFT),
    ]
    for letter, dname, title, xlbl, ylbl, color in parity_cfg:
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        _draw_single_parity(ax, dname, title, xlbl, ylbl, color)
        fig.subplots_adjust(right=0.78)
        fig.savefig(out_dir / f"parity_panel_{letter}.pdf", bbox_inches="tight", pad_inches=0.25)
        fig.savefig(out_dir / f"parity_panel_{letter}.png", bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)

    # e/f: wider and taller for label/bar readability.
    ef_cfg = [
        ("e", "nist1155"),
        ("f", "kmeans251"),
    ]
    for letter, base in ef_cfg:
        fig, ax = plt.subplots(figsize=(7.0, 3.4))
        _draw_single_model_comparison(ax, base)
        fig.tight_layout()
        fig.savefig(out_dir / f"parity_panel_{letter}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"parity_panel_{letter}.png", bbox_inches="tight")
        plt.close(fig)
    log.info(f"  Saved individual panels to {out_dir}")


def make_six_panel_journal():
    """Create 6-panel combined figure in order: a b e / c d f."""
    layout_cfg = [
        ("a", "parity", "nist1155", "Exp. PA (kcal/mol)", "Pred. PA (kcal/mol)", COLOR_NIST_NO_DFT,
         "(a) NIST no DFT"),
        ("b", "parity", "nist1155_dft", "Exp. PA (kcal/mol)", "Pred. PA (kcal/mol)", COLOR_NIST_WITH_DFT,
         "(b) NIST +DFT"),
        ("e", "model", "nist1155", None, None, None,
         "(e) NIST model comp"),
        ("c", "parity", "kmeans251", "DFT PA (kcal/mol)", "Pred. PA (kcal/mol)", COLOR_KM_NO_DFT,
         "(c) k-means no DFT"),
        ("d", "parity", "kmeans251_dft", "DFT PA (kcal/mol)", "Pred. PA (kcal/mol)", COLOR_KM_WITH_DFT,
         "(d) k-means +DFT"),
        ("f", "model", "kmeans251", None, None, None,
         "(f) k-means model comp"),
    ]

    fig, axes = plt.subplots(
        2, 3, figsize=(7.0, 4.2),
        gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]}
    )
    axes_flat = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]

    for ax, (_, kind, dname, xlbl, ylbl, color, title) in zip(axes_flat, layout_cfg):
        if kind == "parity":
            _draw_single_parity(
                ax, dname, title, xlbl, ylbl, color,
                show_title=True, title_fontsize=8, metrics_fontsize=8,
                scatter_size=18, metrics_box_pad=0.15
            )
        else:
            _draw_single_model_comparison(
                ax, dname, title=title, show_title=True, title_fontsize=8
            )

    fig.tight_layout(pad=0.45, w_pad=0.55, h_pad=0.75)
    FIG_PERF.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PERF / "parity_six_panel_journal.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {FIG_PERF / 'parity_six_panel_journal.pdf'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dft", action="store_true")
    args = parser.parse_args()
    has_dft = not args.no_dft
    FIG_PERF.mkdir(parents=True, exist_ok=True)
    make_individual_a_to_f()
    make_six_panel_journal()
    make_combined_parity(has_dft)
    print(f"\n  Saved to: {FIG_PERF}/")
