"""
plot_learning_curves.py
=======================
Regenerates all learning curve figures from pre-saved CSV results.
Run this AFTER learning_curve.py has completed its (4-8 hour) computation.

Reads from:
  results/learning_curve_nist_pm7/learning_curve_data.csv
  results/learning_curve_nist_dft/learning_curve_data.csv
  results/learning_curve_kmeans_pm7/learning_curve_data.csv
  results/learning_curve_kmeans_dft/learning_curve_data.csv
  results/{nist1155,nist1155_dft,kmeans251,kmeans251_dft}/cv_results.json

Outputs (figures/model_performance/):
  learning_curve_nist_pm7.pdf
  learning_curve_nist_dft.pdf
  learning_curve_kmeans_pm7.pdf
  learning_curve_kmeans_dft.pdf
  learning_curve_combined.pdf   — 2x2 combined panel

Usage:
  python scripts/plot_learning_curves.py
  python scripts/plot_learning_curves.py --datasets nist
  python scripts/plot_learning_curves.py --datasets kmeans
  python scripts/plot_learning_curves.py --datasets all
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR     = PROJECT_DIR / "figures"
FIG_PERF    = FIG_DIR / "model_performance"

# Style constants — identical to learning_curve.py
TICK_SIZE   = 22
LABEL_SIZE  = 26
LEGEND_SIZE = 18
TITLE_SIZE  = 22
SPINE_LW    = 1.5


# ---------------------------------------------------------------------------
# Helpers (identical to learning_curve.py)
# ---------------------------------------------------------------------------

def _rcparams():
    plt.rcParams.update({
        "axes.linewidth":    SPINE_LW,
        "xtick.major.width": SPINE_LW,
        "ytick.major.width": SPINE_LW,
        "xtick.labelsize":   TICK_SIZE,
        "ytick.labelsize":   TICK_SIZE,
        "axes.labelsize":    LABEL_SIZE,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })


def _aggregate(lc_df: pd.DataFrame) -> pd.DataFrame:
    test  = lc_df.groupby("fraction")["mae_test"].agg(
        mae_mean="mean", mae_std="std").reset_index()
    train = lc_df.groupby("fraction")["mae_train"].agg(
        train_mean="mean", train_std="std").reset_index()
    n_tr  = lc_df.groupby("fraction")["n_train"].median(
        ).round().astype(int).reset_index()
    n_sel = lc_df.groupby("fraction")["n_features_selected"].agg(
        nfeat_mean="mean", nfeat_std="std").reset_index()
    return (test.merge(train, on="fraction")
               .merge(n_tr, on="fraction")
               .merge(n_sel, on="fraction"))


def _add_secondary_xaxis(ax, summary):
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    pcts = summary["fraction"].values * 100
    ax2.set_xticks(pcts)
    ax2.set_xticklabels([str(n) for n in summary["n_train"].values],
                        rotation=45, ha="left", fontsize=TICK_SIZE - 5)
    ax2.set_xlabel("Number of training samples", fontsize=LABEL_SIZE - 6, labelpad=8)
    return ax2


def plot_single(
    lc_df: pd.DataFrame,
    title: str,
    output_stem: str,
    color: str,
    ref_mae: Optional[float] = None,
    ref_label: Optional[str] = None,
):
    """Single-dataset learning curve with legend outside and n_features annotation."""
    summary = _aggregate(lc_df)
    _rcparams()
    fig, ax = plt.subplots(figsize=(12, 6))
    pct = summary["fraction"] * 100

    ax.errorbar(
        pct, summary["mae_mean"], yerr=summary["mae_std"],
        fmt="o-", color=color, linewidth=2.2, markersize=8,
        capsize=5, capthick=1.8, elinewidth=1.8,
        markeredgecolor="white", markeredgewidth=0.8,
        zorder=3, label="Test MAE",
    )
    ax.errorbar(
        pct, summary["train_mean"], yerr=summary["train_std"],
        fmt="s--", color="#888888", linewidth=2.0, markersize=7,
        capsize=4, capthick=1.5, elinewidth=1.5,
        markeredgecolor="white", markeredgewidth=0.8,
        zorder=2, label="Train MAE",
    )

    if ref_mae is not None:
        ax.axhline(ref_mae, color="black", linewidth=1.5, linestyle=":",
                   zorder=1,
                   label=ref_label or f"5-fold CV MAE: {ref_mae:.2f} kcal/mol")

    for _, row in summary.iterrows():
        ax.annotate(
            f"n={row['nfeat_mean']:.0f}",
            xy=(row["fraction"] * 100, row["mae_mean"]),
            xytext=(0, 10), textcoords="offset points",
            fontsize=9, ha="center", color=color, alpha=0.75,
        )

    _add_secondary_xaxis(ax, summary)
    ax.set_xlabel("Training set size (%)", fontsize=LABEL_SIZE)
    ax.set_ylabel("MAE (kcal/mol)", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=28, fontweight="bold")
    ax.legend(framealpha=0.9, edgecolor="lightgray",
              bbox_to_anchor=(1.01, 1), loc="upper left",
              borderaxespad=0, fontsize=LEGEND_SIZE)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", linewidth=0.5, alpha=0.35, linestyle="--")
    fig.tight_layout()

    FIG_PERF.mkdir(parents=True, exist_ok=True)
    out = FIG_PERF / f"{output_stem}.pdf"
    fig.savefig(out)
    log.info(f"  Saved {out}")
    plt.close(fig)


def plot_combined(
    lc_nist_pm7: pd.DataFrame,
    lc_nist_dft: pd.DataFrame,
    lc_km_pm7:   pd.DataFrame,
    lc_km_dft:   pd.DataFrame,
    ref_nist_pm7: float,
    ref_nist_dft: float,
    ref_km_pm7:   float,
    ref_km_dft:   float,
    output_stem: str = "learning_curve_combined",
):
    """
    2x2 subplot figure:
      Top row:    NIST — PM7 only | PM7+DFT
      Bottom row: k-means — PM7 only | PM7+DFT
    """
    COLOR_PM7 = "#D01C8B"
    COLOR_DFT = "#2166AC"

    panels = [
        (lc_nist_pm7, ref_nist_pm7, COLOR_PM7,
         "NIST — Molecular + PM7 features",        "ExtraTrees"),
        (lc_nist_dft, ref_nist_dft, COLOR_DFT,
         "NIST — Molecular + PM7 + DFT features",  "ExtraTrees"),
        (lc_km_pm7,   ref_km_pm7,   COLOR_PM7,
         "k-means — Molecular + PM7 features",     "ExtraTrees"),
        (lc_km_dft,   ref_km_dft,   COLOR_DFT,
         "k-means — Molecular + PM7 + DFT features", "ExtraTrees"),
    ]

    _rcparams()
    fig, axes = plt.subplots(2, 2, figsize=(22, 12))
    axes = axes.flatten()

    for ax, (lc_df, ref_mae, color, title, _model) in zip(axes, panels):
        summary = _aggregate(lc_df)
        pct = summary["fraction"] * 100

        ax.errorbar(
            pct, summary["mae_mean"], yerr=summary["mae_std"],
            fmt="o-", color=color, linewidth=2.0, markersize=7,
            capsize=4, capthick=1.5, elinewidth=1.5,
            markeredgecolor="white", markeredgewidth=0.7,
            zorder=3, label="Test MAE",
        )
        ax.errorbar(
            pct, summary["train_mean"], yerr=summary["train_std"],
            fmt="s--", color="#888888", linewidth=1.8, markersize=6,
            capsize=3, capthick=1.2, elinewidth=1.2,
            markeredgecolor="white", markeredgewidth=0.7,
            zorder=2, label="Train MAE",
        )
        ax.axhline(ref_mae, color="black", linewidth=1.4, linestyle=":",
                   zorder=1, label=f"5-fold CV: {ref_mae:.2f} kcal/mol")

        for _, row in summary.iterrows():
            ax.annotate(
                f"n={row['nfeat_mean']:.0f}",
                xy=(row["fraction"] * 100, row["mae_mean"]),
                xytext=(0, 9), textcoords="offset points",
                fontsize=8, ha="center", color=color, alpha=0.8,
            )

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(pct)
        ax2.set_xticklabels([str(n) for n in summary["n_train"].values],
                            rotation=45, ha="left", fontsize=TICK_SIZE - 7)
        ax2.set_xlabel("Training samples", fontsize=LABEL_SIZE - 8, labelpad=6)

        ax.set_xlabel("Training set size (%)", fontsize=LABEL_SIZE - 4)
        ax.set_ylabel("MAE (kcal/mol)", fontsize=LABEL_SIZE - 4)
        ax.set_title(title, fontsize=TITLE_SIZE - 2, pad=26, fontweight="bold")
        ax.legend(framealpha=0.9, edgecolor="lightgray",
                  bbox_to_anchor=(1.01, 1), loc="upper left",
                  borderaxespad=0, fontsize=LEGEND_SIZE - 2)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.grid(axis="y", linewidth=0.4, alpha=0.35, linestyle="--")

    fig.suptitle("Learning curves: NIST and k-means datasets",
                 fontsize=TITLE_SIZE + 2, fontweight="bold", y=1.01)
    fig.tight_layout()

    FIG_PERF.mkdir(parents=True, exist_ok=True)
    out = FIG_PERF / f"{output_stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    log.info(f"  Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_lc(results_dir: Path) -> pd.DataFrame:
    csv_path = results_dir / "learning_curve_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Not found: {csv_path}\n"
            "Run learning_curve.py first to generate the data."
        )
    log.info(f"  Loaded {csv_path}")
    return pd.read_csv(csv_path)


def get_ref_mae(cv_results_dir: Path) -> float:
    """Return best model mae_pa_mean from cv_results.json, excluding VotingEnsemble."""
    cv_path = cv_results_dir / "cv_results.json"
    if not cv_path.exists():
        raise FileNotFoundError(
            f"Not found: {cv_path}\n"
            "Run train_models.py first."
        )
    cv = json.loads(cv_path.read_text())
    best = min(
        (m for m in cv["models"] if m != "VotingEnsemble"),
        key=lambda m: cv["models"][m].get("mae_pa_mean") or 999,
    )
    mae = cv["models"][best]["mae_pa_mean"]
    log.info(f"  Reference MAE from {cv_results_dir.name}: {best} = {mae:.3f} kcal/mol")
    return mae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate learning curve figures from saved results."
    )
    parser.add_argument("--datasets", default="all",
                        choices=["all", "nist", "kmeans"])
    args = parser.parse_args()

    run_nist   = args.datasets in ("all", "nist")
    run_kmeans = args.datasets in ("all", "kmeans")

    lc_nist_pm7 = lc_nist_dft = lc_km_pm7 = lc_km_dft = None
    ref_nist_pm7 = ref_nist_dft = ref_km_pm7 = ref_km_dft = None

    if run_nist:
        log.info("Loading NIST learning curve data ...")
        lc_nist_pm7  = load_lc(RESULTS_DIR / "learning_curve_nist_pm7")
        lc_nist_dft  = load_lc(RESULTS_DIR / "learning_curve_nist_dft")
        ref_nist_pm7 = get_ref_mae(RESULTS_DIR / "nist1155")
        ref_nist_dft = get_ref_mae(RESULTS_DIR / "nist1155_dft")

        plot_single(lc_nist_pm7,
                    title="NIST — Molecular + PM7 features",
                    output_stem="learning_curve_nist_pm7",
                    color="#D01C8B",
                    ref_mae=ref_nist_pm7,
                    ref_label=f"5-fold CV MAE: {ref_nist_pm7:.2f} kcal/mol (ExtraTrees)")
        plot_single(lc_nist_dft,
                    title="NIST — Molecular + PM7 + DFT features",
                    output_stem="learning_curve_nist_dft",
                    color="#2166AC",
                    ref_mae=ref_nist_dft,
                    ref_label=f"5-fold CV MAE: {ref_nist_dft:.2f} kcal/mol (ExtraTrees)")

    if run_kmeans:
        log.info("Loading k-means learning curve data ...")
        lc_km_pm7  = load_lc(RESULTS_DIR / "learning_curve_kmeans_pm7")
        lc_km_dft  = load_lc(RESULTS_DIR / "learning_curve_kmeans_dft")
        ref_km_pm7 = get_ref_mae(RESULTS_DIR / "kmeans251")
        ref_km_dft = get_ref_mae(RESULTS_DIR / "kmeans251_dft")

        plot_single(lc_km_pm7,
                    title="k-means — Molecular + PM7 features",
                    output_stem="learning_curve_kmeans_pm7",
                    color="#D01C8B",
                    ref_mae=ref_km_pm7,
                    ref_label=f"5-fold CV MAE: {ref_km_pm7:.2f} kcal/mol (ExtraTrees)")
        plot_single(lc_km_dft,
                    title="k-means — Molecular + PM7 + DFT features",
                    output_stem="learning_curve_kmeans_dft",
                    color="#2166AC",
                    ref_mae=ref_km_dft,
                    ref_label=f"5-fold CV MAE: {ref_km_dft:.2f} kcal/mol (ExtraTrees)")

    if all(x is not None for x in [lc_nist_pm7, lc_nist_dft, lc_km_pm7, lc_km_dft]):
        log.info("Generating combined 2x2 subplot ...")
        plot_combined(
            lc_nist_pm7=lc_nist_pm7, lc_nist_dft=lc_nist_dft,
            lc_km_pm7=lc_km_pm7,     lc_km_dft=lc_km_dft,
            ref_nist_pm7=ref_nist_pm7, ref_nist_dft=ref_nist_dft,
            ref_km_pm7=ref_km_pm7,     ref_km_dft=ref_km_dft,
        )

    print(f"\n  Done. Figures in {FIG_PERF}/")
    for f in sorted(FIG_PERF.glob("learning_curve_*.pdf")):
        print(f"    {f.name}")
