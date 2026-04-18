import pandas as pd, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
FIG_PERF    = PROJECT_DIR / "figures" / "model_performance"

# ── Journal-style defaults (readable at final print size) ───────────────────
TICK_SIZE   = 9
LABEL_SIZE  = 10
LEGEND_SIZE = 9
PANEL_SIZE  = 11
SPINE_LW    = 1.2
DPI         = 600

# ── Manual layout/typography controls (tune here) ───────────────────────────
X_TICK_SIZE = TICK_SIZE            # Bottom x-axis tick labels
Y_TICK_SIZE = TICK_SIZE            # Left y-axis tick labels
TOP_X_TICK_SIZE = TICK_SIZE        # Top axis ("n_train") tick labels

X_LABEL_SIZE = LABEL_SIZE          # Bottom x-axis label
Y_LABEL_SIZE = LABEL_SIZE          # Left y-axis label
TOP_X_LABEL_SIZE = LABEL_SIZE      # Top axis label ("Training samples")
TITLE_SIZE = LABEL_SIZE            # Subplot titles

PANEL_TEXT_SIZE = PANEL_SIZE       # (a), (b) text size
N_ANNOT_SIZE = 9                   # Font size for inline n=... annotations
SCATTER_SIZE = 22                  # Main learning-curve marker size
SCATTER_HIGHLIGHT_SIZE = 34        # Marker size for n=... highlighted points
N_ANNOT_TOP_YOFF = 10              # Vertical offset (points) for PM7 n=... labels
N_ANNOT_BOTTOM_YOFF = -12          # Vertical offset (points) for DFT n=... labels
N_ANNOT_FIRST_X_BONUS = 12         # Extra right shift for leftmost n=... labels

# Spacing between top-axis ticks/label and subplot title
TOP_X_TICK_PAD = 3                 # Tick-label offset from top spine
TOP_X_LABEL_PAD = 6                # "Training samples" offset from top ticks
TITLE_PAD = 10                     # Title offset from axes

# Legend placement inside each subplot
LEGEND_LOC = "upper right"         # e.g., "upper right", "upper left"
LEGEND_BBOX = (1.06, 1.00)         # x>1 moves legend right (away from data)

# Panel letter placement
PANEL_TEXT_X = -0.10
PANEL_TEXT_Y = 1.12

COLOR_PM7 = "#D01C8B"
COLOR_DFT = "#2166AC"

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth":     SPINE_LW,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
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
    "xtick.labelsize":    X_TICK_SIZE,  # Default x tick-label size
    "ytick.labelsize":    Y_TICK_SIZE,  # Default y tick-label size
    "axes.labelsize":     LABEL_SIZE,   # Default axis-label size
    "figure.dpi":         300,
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})


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


# ── Data ─────────────────────────────────────────────────────────────────────
lc_nist_pm7 = pd.read_csv(RESULTS_DIR / "learning_curve_nist_pm7" / "learning_curve_data.csv")
lc_nist_dft = pd.read_csv(RESULTS_DIR / "learning_curve_nist_dft" / "learning_curve_data.csv")
lc_km_pm7   = pd.read_csv(RESULTS_DIR / "learning_curve_kmeans_pm7" / "learning_curve_data.csv")
lc_km_dft   = pd.read_csv(RESULTS_DIR / "learning_curve_kmeans_dft" / "learning_curve_data.csv")

_, ref_nist_pm7 = get_best_model(RESULTS_DIR / "nist1155")
_, ref_nist_dft = get_best_model(RESULTS_DIR / "nist1155_dft")
_, ref_km_pm7   = get_best_model(RESULTS_DIR / "kmeans251")
_, ref_km_dft   = get_best_model(RESULTS_DIR / "kmeans251_dft")

panels = [
    (lc_nist_pm7, lc_nist_dft, ref_nist_pm7, ref_nist_dft,
     "NIST dataset", 1.5, 6.5),
    (lc_km_pm7,   lc_km_dft,   ref_km_pm7,   ref_km_dft,
     "k-means dataset", 4.0, 11.5),
]

panel_labels = ["(a)", "(b)"]

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))

for idx, (ax, (lc_pm7, lc_dft, ref_pm7, ref_dft, title, ymin, ymax)) in enumerate(
        zip(axes, panels)):

    s_pm7 = aggregate(lc_pm7)
    s_dft = aggregate(lc_dft)
    pct   = s_pm7["fraction"] * 100

    # ── Error bands (fill_between) + line + markers ──────────────────────
    # PM7-only
    ax.fill_between(pct, s_pm7["mae_mean"] - s_pm7["mae_std"],
                         s_pm7["mae_mean"] + s_pm7["mae_std"],
                    color=COLOR_PM7, alpha=0.15, zorder=1)
    ax.plot(pct, s_pm7["mae_mean"], "-", color=COLOR_PM7, lw=1.6, zorder=2)
    ax.scatter(pct, s_pm7["mae_mean"], color=COLOR_PM7, s=SCATTER_SIZE,
               edgecolors="black", linewidths=0.6, zorder=4,
               marker="o", label="Without DFT features")

    # DFT
    ax.fill_between(pct, s_dft["mae_mean"] - s_dft["mae_std"],
                         s_dft["mae_mean"] + s_dft["mae_std"],
                    color=COLOR_DFT, alpha=0.15, zorder=1)
    ax.plot(pct, s_dft["mae_mean"], "-", color=COLOR_DFT, lw=1.6, zorder=2)
    ax.scatter(pct, s_dft["mae_mean"], color=COLOR_DFT, s=SCATTER_SIZE,
               edgecolors="black", linewidths=0.6, zorder=4,
               marker="s", label="With DFT features")

    # ── n_features annotations ───────────────────────────────────────────
    # Show only first, middle, and last fractions to avoid overlap
    n_pts = len(s_pm7)
    show_idx = {0, n_pts // 2, n_pts - 1}

    # Highlight annotated points with a distinct marker for clarity.
    for s_cur, color in [(s_pm7, COLOR_PM7), (s_dft, COLOR_DFT)]:
        rows = s_cur.iloc[list(show_idx)]
        ax.scatter(rows["fraction"] * 100, rows["mae_mean"],
                   s=SCATTER_HIGHLIGHT_SIZE, marker="X", color=color, edgecolors="black",
                   linewidths=1.0, zorder=6)

    for s_cur, color, y_off in [
        (s_pm7, COLOR_PM7, N_ANNOT_TOP_YOFF),
        (s_dft, COLOR_DFT, N_ANNOT_BOTTOM_YOFF),
    ]:
        for i, (_, row) in enumerate(s_cur.iterrows()):
            if i not in show_idx:
                continue
            if row["nfeat_mean"] < 5:
                continue
            # Alternate horizontal offset to reduce crowding
            h_off = 12 if (i % 2 == 0) else -12
            if i == 0:
                # Push the first label right so it clears the left-axis corner.
                h_off += N_ANNOT_FIRST_X_BONUS
            ax.annotate(f"n={row['nfeat_mean']:.0f}",
                        xy=(row["fraction"] * 100, row["mae_mean"]),
                        xytext=(h_off, y_off), textcoords="offset points",
                        fontsize=N_ANNOT_SIZE, ha="center", color=color,  # n=... annotation size
                        alpha=0.90, fontweight="bold")

    # ── Axes formatting ──────────────────────────────────────────────────
    ax.set_ylim(ymin, ymax)
    y_ticks = np.linspace(ymin, ymax, 5)
    ax.set_yticks(np.round(y_ticks, 2))
    ax.set_xlabel("Training set size (%)", fontsize=X_LABEL_SIZE)  # Bottom x-axis label size
    ax.set_ylabel("MAE (kcal/mol)", fontsize=Y_LABEL_SIZE)          # Left y-axis label size
    ax.set_title(f"{panel_labels[idx]} {title}", fontsize=TITLE_SIZE,
                 fontweight="bold", pad=TITLE_PAD)  # Inline panel letter + title

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(axis="x", which="major", labelsize=X_TICK_SIZE,  # Bottom x tick-label size
                   width=SPINE_LW, length=7)
    ax.tick_params(axis="y", which="major", labelsize=Y_TICK_SIZE,  # Left y tick-label size
                   width=SPINE_LW, length=7)
    ax.tick_params(axis="both", which="minor", width=SPINE_LW * 0.5,
                   length=4)

    ax.grid(axis="y", lw=0.5, alpha=0.35, ls="--")
    ax.set_axisbelow(True)

    # ── Secondary x-axis: absolute training samples ──────────────────────
    # Match top ticks exactly to currently shown bottom ticks.
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    top_ticks = ax.get_xticks()
    # Interpolate n_train at the same x (percentage) positions as bottom ticks.
    top_n = np.interp(top_ticks, pct.values, s_pm7["n_train"].values)
    top_labels = [str(int(round(v))) for v in top_n]
    ax2.set_xticks(top_ticks)
    ax2.set_xticklabels(top_labels,
                        rotation=45, ha="left", fontsize=max(TOP_X_TICK_SIZE - 1, 8))
    ax2.set_xlabel("Training samples", fontsize=TOP_X_LABEL_SIZE, labelpad=TOP_X_LABEL_PAD)  # Top-axis label size/pad
    ax2.tick_params(axis="x", which="major", length=6, width=SPINE_LW, pad=TOP_X_TICK_PAD)
    ax2.spines["top"].set_linewidth(SPINE_LW)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)

fig.tight_layout(pad=0.8)
fig.subplots_adjust(top=0.84, bottom=0.28, wspace=0.35)

# ── Shared legend (below both subplots, centered) ───────────────────────────
legend_handles = [
    Line2D([0], [0], color=COLOR_PM7, lw=1.6, marker="o", markersize=6,
           markerfacecolor=COLOR_PM7, markeredgecolor="black", markeredgewidth=1.0,
           label="Without DFT features"),
    Line2D([0], [0], color=COLOR_DFT, lw=1.6, marker="s", markersize=6,
           markerfacecolor=COLOR_DFT, markeredgecolor="black", markeredgewidth=1.0,
           label="With DFT features"),
]
fig_leg = fig.legend(
    handles=legend_handles,
    loc="lower center", bbox_to_anchor=(0.5, 0.005),
    ncol=2, fontsize=LEGEND_SIZE,
    framealpha=0.95, edgecolor="black", fancybox=False
)
for text in fig_leg.get_texts():
    text.set_fontweight("bold")
fig_leg.get_frame().set_linewidth(1.0)

FIG_PERF.mkdir(parents=True, exist_ok=True)
out = FIG_PERF / "learning_curve_combined"
fig.savefig(f"{out}.pdf", bbox_inches="tight")
fig.savefig(f"{out}.png", bbox_inches="tight")
print(f"Saved {out}.pdf/.png")
