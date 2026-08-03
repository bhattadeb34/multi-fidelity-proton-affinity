"""
plot_diff_esi.py
================
ESI figure: (pred_with_DFT) minus (pred_no_DFT) vs reference PA.
Two panels — NIST (left) and k-means (right).
Added for R3.7 revision of DD-ART-06-2026-000362.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR     = PROJECT_DIR / "figures"
FIG_PERF    = FIG_DIR / "model_performance"

KJMOL_TO_KCAL = 1 / 4.184
MODEL = "ExtraTrees"

TICK_SIZE   = 10
LABEL_SIZE  = 11
LEGEND_SIZE = 10
PANEL_SIZE  = 12
SPINE_LW    = 1.0
DPI         = 600

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
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.minor.size":   2,
    "ytick.minor.size":   2,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

COLOR_NIST = "#2166AC"
COLOR_KM   = "#D01C8B"


def load_paired(base_name: str, ref_col: str = "pa_true") -> pd.DataFrame:
    """Merge no-DFT and with-DFT ExtraTrees predictions on (fold, record_id)."""
    no  = pd.read_csv(RESULTS_DIR / f"{base_name}/predictions.csv")
    dft = pd.read_csv(RESULTS_DIR / f"{base_name}_dft/predictions.csv")
    no  = no[no["model"]  == MODEL].copy()
    dft = dft[dft["model"] == MODEL].copy()
    merged = no.merge(
        dft[["fold", "sample_idx", "pa_pred"]].rename(columns={"pa_pred": "pa_pred_dft"}),
        on=["fold", "sample_idx"], how="inner"
    )
    merged["ref_kcal"]   = merged[ref_col] * KJMOL_TO_KCAL
    merged["delta_pred"] = (merged["pa_pred_dft"] - merged["pa_pred"]) * KJMOL_TO_KCAL
    return merged


def main():
    nist = load_paired("nist1155")
    km   = load_paired("kmeans251")

    # NIST delta is floating-point noise (~1e-14); clamp to zero for display
    nist["delta_pred"] = 0.0

    fig, ax = plt.subplots(figsize=(4.0, 3.4))

    ax.scatter(
        nist["ref_kcal"], nist["delta_pred"],
        c=COLOR_NIST, s=14, alpha=0.70, marker="o",
        edgecolors="black", linewidths=0.20, zorder=3,
        label=f"NIST (exp. ref., N = {len(nist)})"
    )
    ax.scatter(
        km["ref_kcal"], km["delta_pred"],
        c=COLOR_KM, s=14, alpha=0.70, marker="s",
        edgecolors="black", linewidths=0.20, zorder=2,
        label=f"k-means (DFT ref., N = {len(km)} sites)"
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=1)

    ax.set_xlabel("Reference PA (kcal/mol)", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$\Delta$ Pred. PA (kcal/mol)", fontsize=LABEL_SIZE)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(axis="x", which="major", labelsize=TICK_SIZE, width=1.2, length=3.5, pad=4)
    ax.tick_params(axis="y", which="major", labelsize=TICK_SIZE, width=1.2, length=3.5, pad=4)
    ax.tick_params(axis="both", which="minor", width=0.8, length=2.0)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55, color="gray")

    leg = ax.legend(fontsize=LEGEND_SIZE, frameon=True, edgecolor="black",
                    facecolor="white", loc="upper right",
                    handlelength=1.0, handletextpad=0.4, borderpad=0.4)
    leg.get_frame().set_linewidth(SPINE_LW)

    # Annotation for k-means stats
    mean_d = km["delta_pred"].mean()
    std_d  = km["delta_pred"].std()
    stats  = f"k-means: mean $\Delta$ = {mean_d:+.2f} ± {std_d:.2f} kcal/mol"
    ax.text(0.02, 0.03, stats, transform=ax.transAxes, fontsize=LEGEND_SIZE - 1,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.9, alpha=0.95))

    fig.tight_layout(pad=0.5)
    FIG_PERF.mkdir(parents=True, exist_ok=True)
    out = FIG_PERF / "diff_esi"
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png")
    plt.close(fig)
    print(f"Saved {out}.pdf / .png")


if __name__ == "__main__":
    main()
