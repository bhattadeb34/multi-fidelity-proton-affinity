"""
plot_results.py
===============
Generates publication-quality figures from CV results.

Figures produced (saved to ../figures/):
  1. parity_nist_pm7only.pdf/.png     — NIST 1155, PM7-only features
  2. parity_nist_dft.pdf/.png         — NIST 1155, PM7+DFT features
  3. parity_kmeans_pm7only.pdf/.png   — k-means 251, PM7-only features
  4. parity_kmeans_dft.pdf/.png       — k-means 251, PM7+DFT features
  5. model_comparison_nist.pdf/.png   — bar chart, all models, NIST ± std
  6. model_comparison_kmeans.pdf/.png — bar chart, all models, k-means ± std
  7. model_comparison_all.pdf/.png    — combined 2×2 panel

Parity plot design:
  - x-axis: true PA (exp or DFT), y-axis: predicted PA = PM7 + delta_ML
  - Points colored by fold (5 colours)
  - Error bars not on individual points (each point appears in one fold only)
  - MAE ± std reported in legend
  - N data points shown in subtitle
  - For k-means: molecule-level PA — best DFT site per molecule, same site predicted

Style:
  - Tick font size  : 20
  - Axis label size : 24
  - Spine linewidth : 1.5 × default (1.5)
  - Legend font     : 18
  - Figure size     : 7×7 per panel
  - All axes square, 1:1 parity line
  - Units: kcal/mol

Usage
-----
  python scripts/plot_results.py
  python scripts/plot_results.py --no-dft   # skip dft panels if not yet available
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
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR     = PROJECT_DIR / "figures"
FIG_PERF    = FIG_DIR / "model_performance"

KJMOL_TO_KCAL = 1 / 4.184

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

TICK_SIZE   = 20
LABEL_SIZE  = 24
LEGEND_SIZE = 18
SPINE_LW    = 1.5
FIG_SIZE    = (7, 7)
DPI         = 300

# 5 fold colours — aesthetically distinct, print-safe
FOLD_COLORS = ["#2166AC", "#4DAC26", "#D01C8B", "#F1A340", "#7B3294"]

# Model bar chart colours
BAR_COLOR_PM7 = "#2166AC"   # blue  — PM7-only
BAR_COLOR_DFT = "#D01C8B"   # pink  — PM7+DFT

plt.rcParams.update({
    "font.family":        "sans-serif",
    "axes.linewidth":     SPINE_LW,
    "xtick.major.width":  SPINE_LW,
    "ytick.major.width":  SPINE_LW,
    "xtick.minor.width":  SPINE_LW * 0.6,
    "ytick.minor.width":  SPINE_LW * 0.6,
    "xtick.major.size":   6,
    "ytick.major.size":   6,
    "xtick.labelsize":    TICK_SIZE,
    "ytick.labelsize":    TICK_SIZE,
    "axes.labelsize":     LABEL_SIZE,
    "legend.fontsize":    LEGEND_SIZE,
    "figure.dpi":         DPI,
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.1,
})


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_predictions(dataset_name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / dataset_name / "predictions.csv"
    if not path.exists():
        log.warning(f"predictions.csv not found: {path}")
        return None
    return pd.read_csv(path)


def load_mae_summary(dataset_name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / dataset_name / "mae_summary.csv"
    if not path.exists():
        log.warning(f"mae_summary.csv not found: {path}")
        return None
    return pd.read_csv(path)


def load_cv_results(dataset_name: str) -> dict | None:
    path = RESULTS_DIR / dataset_name / "cv_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_best_model(dataset_name: str) -> str | None:
    mae_df = load_mae_summary(dataset_name)
    if mae_df is None:
        return None
    # Exclude VotingEnsemble — it is not a standalone model
    mae_df = mae_df[mae_df["model"] != "VotingEnsemble"]
    return mae_df.sort_values("mae_delta_mean").iloc[0]["model"]


def compute_overall_mae(preds: pd.DataFrame, model: str) -> tuple[float, float]:
    """MAE ± std across folds for a given model (in kcal/mol, already converted)."""
    df = preds[preds["model"] == model].copy()
    # pa_pred and pa_true are in kJ/mol (raw), convert for display
    df["pa_pred_kcal"] = df["pa_pred"] * KJMOL_TO_KCAL
    df["pa_true_kcal"] = df["pa_true"] * KJMOL_TO_KCAL
    fold_maes = (df.groupby("fold")
                 .apply(lambda g: np.mean(np.abs(g["pa_pred_kcal"] - g["pa_true_kcal"])))
                 .values)
    return float(np.mean(fold_maes)), float(np.std(fold_maes))


def molecule_level_parity(preds: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    For multi-site datasets (k-means): aggregate to molecule level.
    Strategy: find the site with the highest DFT PA per molecule per fold,
    use that site's predicted PA for comparison.
    This matches the molecule-level evaluation convention.
    """
    df = preds[preds["model"] == model].copy()
    df["pa_pred_kcal"] = df["pa_pred"] * KJMOL_TO_KCAL
    df["pa_true_kcal"] = df["pa_true"] * KJMOL_TO_KCAL

    # For each (neutral_smiles, fold): find the site with max true PA
    idx_best = (df.groupby(["neutral_smiles", "fold"])["pa_true_kcal"]
                .idxmax())
    return df.loc[idx_best].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Parity plot
# ---------------------------------------------------------------------------

def make_parity_plot(
    dataset_name: str,
    title: str,
    xlabel: str,
    ylabel: str,
    use_molecule_level: bool,
    output_stem: str,
):
    preds = load_predictions(dataset_name)
    if preds is None:
        log.warning(f"Skipping {output_stem} — no predictions found")
        return

    best_model = get_best_model(dataset_name)
    if best_model is None:
        log.warning(f"Skipping {output_stem} — no mae_summary found")
        return

    log.info(f"  Parity plot: {output_stem}  (best model: {best_model})")

    if use_molecule_level:
        df_plot = molecule_level_parity(preds, best_model)
    else:
        df_plot = preds[preds["model"] == best_model].copy()
        df_plot["pa_pred_kcal"] = df_plot["pa_pred"] * KJMOL_TO_KCAL
        df_plot["pa_true_kcal"] = df_plot["pa_true"] * KJMOL_TO_KCAL

    mae_mean, mae_std = compute_overall_mae(preds, best_model)
    n_points = len(df_plot)
    folds    = sorted(df_plot["fold"].unique())

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    all_vals = pd.concat([df_plot["pa_true_kcal"], df_plot["pa_pred_kcal"]])
    vmin = all_vals.min()
    vmax = all_vals.max()
    pad  = (vmax - vmin) * 0.05
    lims = (vmin - pad, vmax + pad)

    # Parity line
    ax.plot(lims, lims, color="black", linewidth=1.2, linestyle="--",
            zorder=1, label="_nolegend_")

    # Scatter by fold
    for fold_idx, fold in enumerate(folds):
        mask = df_plot["fold"] == fold
        ax.scatter(
            df_plot.loc[mask, "pa_true_kcal"],
            df_plot.loc[mask, "pa_pred_kcal"],
            color=FOLD_COLORS[fold_idx % len(FOLD_COLORS)],
            s=40, alpha=0.75, linewidths=0.3,
            edgecolors="white", zorder=2,
            label=f"Fold {fold}",
        )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # MAE annotation in top-left
    ax.text(0.04, 0.96,
            f"MAE = {mae_mean:.2f} ± {mae_std:.2f} kcal/mol\n"
            f"N = {n_points}  |  Model: {best_model}",
            transform=ax.transAxes,
            fontsize=LEGEND_SIZE - 1,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="lightgray", linewidth=0.8, alpha=0.9))

    ax.legend(loc="lower right", framealpha=0.9,
              edgecolor="lightgray", markerscale=1.3)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(left=0.15, bottom=0.25)

    for ext in ("pdf",):
        out = FIG_PERF / f"{output_stem}.{ext}"
        FIG_PERF.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        log.info(f"    Saved {out.relative_to(PROJECT_DIR)}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def make_model_comparison(
    dataset_pm7: str,
    dataset_dft: str | None,
    title: str,
    output_stem: str,
):
    cv_pm7 = load_cv_results(dataset_pm7)
    cv_dft = load_cv_results(dataset_dft) if dataset_dft else None

    if cv_pm7 is None:
        log.warning(f"Skipping {output_stem} — no cv_results for {dataset_pm7}")
        return

    log.info(f"  Model comparison: {output_stem}")

    # Gather models present in both runs
    models_pm7 = {m: v for m, v in cv_pm7["models"].items()
                  if v.get("mae_delta_mean") is not None}
    models_dft = ({m: v for m, v in cv_dft["models"].items()
                   if v.get("mae_delta_mean") is not None}
                  if cv_dft else {})

    # Exclude VotingEnsemble (used by Jin & Merz 2025 — avoid structural similarity)
    for d in [models_pm7, models_dft]:
        d.pop("VotingEnsemble", None)

    # Sort by PM7-only MAE
    sorted_models = sorted(models_pm7.keys(),
                           key=lambda m: models_pm7[m]["mae_delta_mean"])

    n = len(sorted_models)
    x = np.arange(n)
    width = 0.35 if cv_dft else 0.6

    fig_w = max(12, n * 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, 8))

    maes_pm7 = [models_pm7[m]["mae_delta_mean"] for m in sorted_models]
    stds_pm7 = [models_pm7[m]["mae_delta_std"]  for m in sorted_models]

    offset = -width / 2 if cv_dft else 0
    bars_pm7 = ax.bar(
        x + offset, maes_pm7, width,
        yerr=stds_pm7, capsize=4,
        color=BAR_COLOR_PM7, alpha=0.85,
        error_kw={"linewidth": 1.5, "capthick": 1.5, "ecolor": "black"},
        label="Without DFT features",
    )

    if cv_dft and models_dft:
        maes_dft = [models_dft.get(m, {}).get("mae_delta_mean", np.nan)
                    for m in sorted_models]
        stds_dft = [models_dft.get(m, {}).get("mae_delta_std",  np.nan)
                    for m in sorted_models]
        ax.bar(
            x + width / 2, maes_dft, width,
            yerr=stds_dft, capsize=4,
            color=BAR_COLOR_DFT, alpha=0.85,
            error_kw={"linewidth": 1.5, "capthick": 1.5, "ecolor": "black"},
            label="With DFT features",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=40, ha="right",
                       fontsize=TICK_SIZE - 2)
    ax.set_ylabel("Test MAE (kcal/mol)\n[5-fold CV]", labelpad=12)
    ax.set_xlabel("Model")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    if cv_dft:
        ax.legend(loc="upper left", framealpha=0.9, edgecolor="lightgray")

    ax.grid(axis="y", linewidth=0.5, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()

    for ext in ("pdf",):
        out = FIG_PERF / f"{output_stem}.{ext}"
        FIG_PERF.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        log.info(f"    Saved {out.relative_to(PROJECT_DIR)}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined 2×2 parity panel
# ---------------------------------------------------------------------------

def make_combined_parity(has_dft: bool):
    """2×2 panel: top row = NIST, bottom row = k-means; left = PM7, right = DFT."""
    configs = [
        ("nist1155",     False, "NIST (without DFT features)",     "Exp. PA (kcal/mol)", "Pred. PA (kcal/mol)"),
        ("nist1155_dft", False, "NIST (with DFT features)", "Exp. PA (kcal/mol)", "Pred. PA (kcal/mol)"),
        ("kmeans251",     True, "k-means (without DFT features)",  "DFT PA (kcal/mol)",  "Pred. PA (kcal/mol)"),
        ("kmeans251_dft", True, "k-means (with DFT features)","DFT PA (kcal/mol)","Pred. PA (kcal/mol)"),
    ]
    if not has_dft:
        configs = [c for c in configs if "dft" not in c[0]]
        ncols, nrows = len(configs), 1
    else:
        ncols, nrows = 2, 2

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(FIG_SIZE[0] * ncols, FIG_SIZE[1] * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, (dname, mol_level, panel_title, xlbl, ylbl) in enumerate(configs):
        ax = axes[ax_idx]

        preds = load_predictions(dname)
        if preds is None:
            ax.text(0.5, 0.5, f"No data\n({dname})", ha="center", va="center",
                    transform=ax.transAxes, fontsize=LABEL_SIZE - 4, color="gray")
            ax.set_title(panel_title, fontsize=LABEL_SIZE - 2, pad=10)
            continue

        best_model = get_best_model(dname)
        if best_model is None:
            continue

        if mol_level:
            df_plot = molecule_level_parity(preds, best_model)
        else:
            df_plot = preds[preds["model"] == best_model].copy()
            df_plot["pa_pred_kcal"] = df_plot["pa_pred"] * KJMOL_TO_KCAL
            df_plot["pa_true_kcal"] = df_plot["pa_true"] * KJMOL_TO_KCAL

        mae_mean, mae_std = compute_overall_mae(preds, best_model)
        n_points = len(df_plot)
        folds    = sorted(df_plot["fold"].unique())

        all_vals = pd.concat([df_plot["pa_true_kcal"], df_plot["pa_pred_kcal"]])
        vmin = all_vals.min()
        vmax = all_vals.max()
        pad  = (vmax - vmin) * 0.05
        lims = (vmin - pad, vmax + pad)

        ax.plot(lims, lims, color="black", linewidth=1.2, linestyle="--", zorder=1)

        for fold_idx, fold in enumerate(folds):
            mask = df_plot["fold"] == fold
            ax.scatter(
                df_plot.loc[mask, "pa_true_kcal"],
                df_plot.loc[mask, "pa_pred_kcal"],
                color=FOLD_COLORS[fold_idx % len(FOLD_COLORS)],
                s=30, alpha=0.7, linewidths=0.3,
                edgecolors="white", zorder=2,
                label=f"Fold {fold}",
            )

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.set_title(panel_title, fontsize=LABEL_SIZE - 2, pad=10)

        ax.text(0.04, 0.96,
                f"MAE = {mae_mean:.2f} ± {mae_std:.2f} kcal/mol\n"
                f"N = {n_points}  |  {best_model}",
                transform=ax.transAxes, fontsize=LEGEND_SIZE - 3,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="lightgray", linewidth=0.8, alpha=0.9))

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # shared fold legend on first panel
    legend_elements = [Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=FOLD_COLORS[i],
                               markersize=9, label=f"Fold {i+1}")
                       for i in range(5)]
    axes[0].legend(handles=legend_elements, loc="lower right",
                   framealpha=0.9, edgecolor="lightgray",
                   fontsize=LEGEND_SIZE - 4)

    fig.tight_layout(pad=1.5)

    stem = "parity_combined" if has_dft else "parity_combined_pm7only"
    for ext in ("pdf",):
        out = FIG_PERF / f"{stem}.{ext}"
        FIG_PERF.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        log.info(f"  Saved {out.relative_to(PROJECT_DIR)}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate publication figures.")
    parser.add_argument("--no-dft", action="store_true",
                        help="Skip DFT-augmented panels (if run not yet complete)")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    has_dft = not args.no_dft

    # ---- Individual parity plots ----
    log.info("Generating parity plots ...")

    make_parity_plot(
        dataset_name     = "nist1155",
        title            = "NIST 1155 — PM7 features",
        xlabel           = "Experimental PA (kcal/mol)",
        ylabel           = "Predicted PA (kcal/mol)",
        use_molecule_level = False,
        output_stem      = "parity_nist_pm7only",
    )

    make_parity_plot(
        dataset_name     = "kmeans251",
        title            = "k-means 251 — PM7 features",
        xlabel           = "DFT PA (kcal/mol)",
        ylabel           = "Predicted PA (kcal/mol)",
        use_molecule_level = True,
        output_stem      = "parity_kmeans_pm7only",
    )

    if has_dft:
        make_parity_plot(
            dataset_name     = "nist1155_dft",
            title            = "NIST 1155 — PM7+DFT features",
            xlabel           = "Experimental PA (kcal/mol)",
            ylabel           = "Predicted PA (kcal/mol)",
            use_molecule_level = False,
            output_stem      = "parity_nist_dft",
        )
        make_parity_plot(
            dataset_name     = "kmeans251_dft",
            title            = "k-means 251 — PM7+DFT features",
            xlabel           = "DFT PA (kcal/mol)",
            ylabel           = "Predicted PA (kcal/mol)",
            use_molecule_level = True,
            output_stem      = "parity_kmeans_dft",
        )

    # ---- Model comparison bar charts ----
    log.info("Generating model comparison charts ...")

    make_model_comparison(
        dataset_pm7  = "nist1155",
        dataset_dft  = "nist1155_dft" if has_dft else None,
        title        = "NIST 1155 — model comparison",
        output_stem  = "model_comparison_nist",
    )

    make_model_comparison(
        dataset_pm7  = "kmeans251",
        dataset_dft  = "kmeans251_dft" if has_dft else None,
        title        = "k-means 251 — model comparison",
        output_stem  = "model_comparison_kmeans",
    )

    # ---- Combined 2×2 parity panel ----
    log.info("Generating combined parity panel ...")
    make_combined_parity(has_dft=has_dft)

    print(f"\n  All figures saved to: {FIG_DIR.relative_to(PROJECT_DIR)}/")
    print(f"  Files: parity_*.pdf/png, model_comparison_*.pdf/png, parity_combined*.pdf/png")


if __name__ == "__main__":
    main()