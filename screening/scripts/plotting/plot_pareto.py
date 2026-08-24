"""
plot_pareto.py
==============
Regenerate the Pareto selection summary figure from pre-computed data.

Reads from:
    data/screening/iter{N}/llm_verdicts.parquet
    data/screening/iter{N}/pareto_selected.csv

Outputs:
    figures/screening/iter{N}_pareto.pdf

Usage:
    python screening/scripts/plotting/plot_pareto.py
    python screening/scripts/plotting/plot_pareto.py --iter 1
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
SCREENING  = SCRIPT_DIR.parent.parent
PROJECT    = SCREENING.parent
CONFIG_PATH = SCREENING / "config" / "pipeline_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text())

PA_TARGET_LOW  = CONFIG["pa_window_kcalmol"]["low"]
PA_TARGET_HIGH = CONFIG["pa_window_kcalmol"]["high"]
ELIGIBLE_VERDICTS = CONFIG["pareto"]["eligible_verdicts"]

# Journal-style defaults
DOUBLE_COL_W = 7.0
J_TICK = 9
J_LABEL = 10
J_TITLE = 10
J_LEGEND = 9
J_SPINE = 1.0

# Higher-contrast palette for small journal figures
COLOR_POOL = "#2166AC"      # darker blue
COLOR_SELECTED = "#B2182B"  # darker red


def main(iteration: int) -> None:
    data_root = next((p for p in (PROJECT / "data", PROJECT.parent / "data") if p.exists()), PROJECT / "data")
    iter_dir = data_root / "screening" / f"iter{iteration}"
    verdict_path = iter_dir / "llm_verdicts.parquet"
    pareto_csv   = iter_dir / "pareto_selected.csv"
    fig_dir    = PROJECT / "figures" / "screening"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not verdict_path.exists():
        log.error(f"Not found: {verdict_path}")
        return
    if not pareto_csv.exists():
        log.error(f"Not found: {pareto_csv}")
        return

    df = pd.read_parquet(verdict_path)
    mask = (
        df["final_verdict"].isin(ELIGIBLE_VERDICTS) &
        (df["pa_pred_kcalmol"] >= PA_TARGET_LOW) &
        (df["pa_pred_kcalmol"] <= PA_TARGET_HIGH)
    )
    pool = df[mask].copy().reset_index(drop=True)
    selected_df = pd.read_csv(pareto_csv)

    log.info(f"Pool: {len(pool):,}  |  Selected: {len(selected_df)}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": J_TICK,
        "axes.labelsize": J_LABEL,
        "axes.titlesize": J_TITLE,
        "axes.linewidth": J_SPINE,
        "xtick.labelsize": J_TICK,
        "ytick.labelsize": J_TICK,
        "legend.fontsize": J_LEGEND,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, 3.0))

    # Panel 1: PA distribution
    ax = axes[0]
    ax.hist(pool["pa_pred_kcalmol"], bins=40, color=COLOR_POOL, alpha=0.6)
    ax.hist(selected_df["pa_pred_kcalmol"], bins=20, color=COLOR_SELECTED, alpha=0.8)
    ax.axvline(PA_TARGET_LOW, color="k", ls="--", lw=1)
    ax.axvline(PA_TARGET_HIGH, color="k", ls="--", lw=1)
    ax.set_xlabel("PA_pred (kcal/mol)")
    ax.set_ylabel("Count")
    ax.text(-0.22, 1.05, "(a)", transform=ax.transAxes,
            fontsize=J_TITLE, fontweight="bold", va="top")

    # Panel 2: PA vs tree-ensemble dispersion
    ax = axes[1]
    ax.scatter(pool["pa_pred_kcalmol"], pool["uncertainty"],
               c=COLOR_POOL, alpha=0.45, s=14, edgecolors="none")
    ax.scatter(selected_df["pa_pred_kcalmol"], selected_df["uncertainty"],
               c=COLOR_SELECTED, s=36, marker="o",
               edgecolors="black", linewidths=0.35, zorder=5)
    ax.axvline(PA_TARGET_LOW, color="k", ls="--", lw=1)
    ax.axvline(PA_TARGET_HIGH, color="k", ls="--", lw=1)
    ax.set_xlabel("PA_pred (kcal/mol)")
    ax.set_ylabel("Ensemble dispersion (kcal/mol)")
    ax.text(-0.22, 1.05, "(b)", transform=ax.transAxes,
            fontsize=J_TITLE, fontweight="bold", va="top")

    # Panel 3: PA vs SA score
    ax = axes[2]
    ax.scatter(pool["pa_pred_kcalmol"], pool["sa_score"],
               c=COLOR_POOL, alpha=0.45, s=14, edgecolors="none")
    ax.scatter(selected_df["pa_pred_kcalmol"], selected_df["sa_score"],
               c=COLOR_SELECTED, s=36, marker="o",
               edgecolors="black", linewidths=0.35, zorder=5)
    ax.axvline(PA_TARGET_LOW, color="k", ls="--", lw=1)
    ax.axvline(PA_TARGET_HIGH, color="k", ls="--", lw=1)
    ax.set_xlabel("PA_pred (kcal/mol)")
    ax.set_ylabel("SA Score")
    ax.text(-0.22, 1.05, "(c)", transform=ax.transAxes,
            fontsize=J_TITLE, fontweight="bold", va="top")

    legend_handles = [
        mpatches.Patch(
            facecolor=COLOR_POOL, alpha=0.6, edgecolor="none",
            label=f"Pool ({'/'.join(ELIGIBLE_VERDICTS)}, in window)",
        ),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=COLOR_SELECTED, markeredgecolor="black",
               markeredgewidth=0.5, markersize=7, linestyle="None",
               label="Selected"),
        Line2D([0], [0], color="k", ls="--", lw=1,
               label=(f"PA target window "
                      f"({PA_TARGET_LOW:g}–{PA_TARGET_HIGH:g} kcal/mol)")),
    ]

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=True, edgecolor="black", facecolor="white",
        fontsize=J_LEGEND,
        handletextpad=0.6, columnspacing=1.6,
    )

    fig_path = fig_dir / f"iter{iteration}_pareto.pdf"
    png_path = fig_dir / f"iter{iteration}_pareto.png"
    plt.savefig(fig_path)
    plt.savefig(png_path, dpi=200)
    plt.close()
    log.info(f"Saved {fig_path}")
    log.info(f"Saved {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Pareto selection summary from saved results.")
    parser.add_argument("--iter", type=int, default=1)
    args = parser.parse_args()
    main(iteration=args.iter)
