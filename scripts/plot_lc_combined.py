import pandas as pd, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
FIG_PERF    = Path("figures/model_performance")
TICK_SIZE, LABEL_SIZE, LEGEND_SIZE, TITLE_SIZE, SPINE_LW = 22, 26, 16, 22, 1.5

def get_best_model(d):
    cv = json.loads((d / "cv_results.json").read_text())
    best = min((m for m in cv["models"] if m != "VotingEnsemble"),
               key=lambda m: cv["models"][m].get("mae_pa_mean") or 999)
    return best, cv["models"][best]["mae_pa_mean"]

def aggregate(df):
    test  = df.groupby("fraction")["mae_test"].agg(
        mae_mean="mean", mae_std="std").reset_index()
    n_tr  = df.groupby("fraction")["n_train"].median(
        ).round().astype(int).reset_index()
    n_sel = df.groupby("fraction")["n_features_selected"].agg(
        nfeat_mean="mean").reset_index()
    return test.merge(n_tr, on="fraction").merge(n_sel, on="fraction")

lc_nist_pm7 = pd.read_csv("results/learning_curve_nist_pm7/learning_curve_data.csv")
lc_nist_dft = pd.read_csv("results/learning_curve_nist_dft/learning_curve_data.csv")
lc_km_pm7   = pd.read_csv("results/learning_curve_kmeans_pm7/learning_curve_data.csv")
lc_km_dft   = pd.read_csv("results/learning_curve_kmeans_dft/learning_curve_data.csv")

_, ref_nist_pm7 = get_best_model(RESULTS_DIR / "nist1155")
_, ref_nist_dft = get_best_model(RESULTS_DIR / "nist1155_dft")
_, ref_km_pm7   = get_best_model(RESULTS_DIR / "kmeans251")
_, ref_km_dft   = get_best_model(RESULTS_DIR / "kmeans251_dft")

COLOR_PM7 = "#D01C8B"
COLOR_DFT = "#2166AC"

panels = [
    # (lc_pm7, lc_dft, ref_pm7, ref_dft, title, ymin, ymax)
    (lc_nist_pm7, lc_nist_dft, ref_nist_pm7, ref_nist_dft,
     "NIST dataset", 1.5, 6.5),
    (lc_km_pm7,   lc_km_dft,   ref_km_pm7,   ref_km_dft,
     "k-means dataset", 4.0, 11.5),
]

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.linewidth": SPINE_LW,
    "xtick.labelsize": TICK_SIZE - 4, "ytick.labelsize": TICK_SIZE - 4,
})

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, (lc_pm7, lc_dft, ref_pm7, ref_dft, title, ymin, ymax) in zip(axes, panels):
    s_pm7 = aggregate(lc_pm7)
    s_dft = aggregate(lc_dft)
    pct   = s_pm7["fraction"] * 100

    # PM7-only curve
    ax.errorbar(pct, s_pm7["mae_mean"], yerr=s_pm7["mae_std"],
                fmt="o-", color=COLOR_PM7, lw=2.2, ms=8,
                capsize=4, capthick=1.5, elinewidth=1.5,
                markeredgecolor="white", markeredgewidth=0.8,
                zorder=3, label="Without DFT features")

    # DFT curve
    ax.errorbar(pct, s_dft["mae_mean"], yerr=s_dft["mae_std"],
                fmt="s-", color=COLOR_DFT, lw=2.2, ms=8,
                capsize=4, capthick=1.5, elinewidth=1.5,
                markeredgecolor="white", markeredgewidth=0.8,
                zorder=3, label="With DFT features")

    # CV reference lines removed — values reported in paper text

    # n_features annotations for both curves (skip n < 5 to reduce clutter)
    for s_cur, color, offset in [(s_pm7, COLOR_PM7, 10), (s_dft, COLOR_DFT, -16)]:
        for _, row in s_cur.iterrows():
            if row["nfeat_mean"] < 5:
                continue
            ax.annotate(f"n={row['nfeat_mean']:.0f}",
                        xy=(row["fraction"] * 100, row["mae_mean"]),
                        xytext=(0, offset), textcoords="offset points",
                        fontsize=13, ha="center", color=color,
                        alpha=0.85, fontweight="bold")

    # Secondary x-axis: absolute training samples
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(pct)
    ax2.set_xticklabels([str(n) for n in s_pm7["n_train"].values],
                        rotation=45, ha="left", fontsize=TICK_SIZE - 7)
    ax2.set_xlabel("Training samples", fontsize=LABEL_SIZE - 8, labelpad=6)

    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Training set size (%)", fontsize=LABEL_SIZE - 4)
    ax.set_ylabel("MAE (kcal/mol)", fontsize=LABEL_SIZE - 4)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=26, fontweight="bold")
    ax.legend(framealpha=0.9, edgecolor="lightgray",
              loc="upper right", fontsize=LEGEND_SIZE)
    ax.grid(axis="y", lw=0.4, alpha=0.35, ls="--")
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Learning curves: NIST and k-means datasets",
             fontsize=TITLE_SIZE + 2, fontweight="bold", y=1.02)
fig.tight_layout()
FIG_PERF.mkdir(parents=True, exist_ok=True)
out = FIG_PERF / "learning_curve_combined.pdf"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
