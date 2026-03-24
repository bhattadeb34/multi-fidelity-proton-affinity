"""
plot_chemical_analysis.py
=========================
Chemical class analysis of PM7 correction patterns.
Runs for BOTH NIST (experimental reference) and k-means (DFT reference).

Figures generated (saved to ../figures/):
  NIST:   correction_by_class_nist.pdf/.png
          representative_molecules_nist.pdf/.png
          correction_kde_by_class_nist.pdf/.png
          correction_vs_pa_nist.pdf/.png
          worst_best_molecules_nist.pdf/.png
          correction_summary_table_nist.pdf/.png

  KMEANS: correction_by_class_kmeans.pdf/.png
          ... (same set, site-level, DFT reference)

Usage
-----
  python scripts/plot_chemical_analysis.py              # both datasets
  python scripts/plot_chemical_analysis.py --dataset nist
  python scripts/plot_chemical_analysis.py --dataset kmeans
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from io import BytesIO
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR    = PROJECT_DIR / "data"
FIG_DIR     = PROJECT_DIR / "figures"
FIG_CHEM    = FIG_DIR / "chemical_analysis"

KJMOL_TO_KCAL = 1 / 4.184

# ── Typography ──────────────────────────────────────────────────────────────
TICK_SIZE   = 22
LABEL_SIZE  = 26
LEGEND_SIZE = 20
ANNOT_SIZE  = 18
TITLE_SIZE  = 24
SPINE_LW    = 1.8
DPI         = 300

plt.rcParams.update({
    "font.family":        "sans-serif",
    "axes.linewidth":     SPINE_LW,
    "xtick.major.width":  SPINE_LW, "ytick.major.width":  SPINE_LW,
    "xtick.minor.width":  1.2,      "ytick.minor.width":  1.2,
    "xtick.major.size":   7,        "ytick.major.size":   7,
    "xtick.minor.size":   4,        "ytick.minor.size":   4,
    "xtick.labelsize":    TICK_SIZE, "ytick.labelsize":    TICK_SIZE,
    "axes.labelsize":     LABEL_SIZE,
    "axes.labelweight":   "bold",
    "figure.dpi":         DPI,      "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",  "savefig.pad_inches": 0.20,
})

CLASS_COLORS = {
    "Nitrile":         "#E24B4A",
    "Primary amine":   "#85B7EB",
    "Carbonyl":        "#FAC775",
    "Carboxylic acid": "#97C459",
    "Secondary amine": "#5DCAA5",
    "Tertiary amine":  "#378ADD",
    "Amide":           "#AFA9EC",
    "Halogen":         "#F0997B",
    "Ether":           "#D4537E",
    "Aromatic N":      "#534AB7",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_nist_with_corrections() -> pd.DataFrame:
    """NIST: molecule-level, correction = PA_exp − PA_PM7."""
    log.info("Loading NIST data ...")
    nist_feat = pd.read_parquet(DATA_DIR / "features" / "nist1185_features.parquet")

    excel  = DATA_DIR / "Merz-and-hogni-dataset.xlsx"
    df_exp = pd.read_excel(excel)[["smiles", "EXP_PA"]].rename(
        columns={"smiles": "neutral_smiles", "EXP_PA": "exp_pa_kjmol"})
    df_exp["exp_pa_kcalmol"] = df_exp["exp_pa_kjmol"] * KJMOL_TO_KCAL

    nist_mol = (nist_feat.groupby("neutral_smiles", sort=False)
                .agg(pm7_pa_kjmol=("pm7_pa_kjmol", "max"),
                     mw=("neutral_rdkit_MolWt", "first"))
                .reset_index())
    nist_mol = nist_mol.merge(df_exp, on="neutral_smiles", how="left")
    nist_mol["correction_kcal"] = (
        (nist_mol["exp_pa_kjmol"] - nist_mol["pm7_pa_kjmol"]) * KJMOL_TO_KCAL)
    nist_mol["ref_pa_kcalmol"]  = nist_mol["exp_pa_kjmol"] * KJMOL_TO_KCAL
    nist_mol["pm7_pa_kcalmol"]  = nist_mol["pm7_pa_kjmol"] * KJMOL_TO_KCAL
    nist_mol = nist_mol.dropna(subset=["correction_kcal"])

    fg_cols  = [c for c in nist_feat.columns if c.startswith("neutral_rdkit_fr_")]
    fg_mol   = nist_feat.groupby("neutral_smiles")[fg_cols].max().reset_index()
    nist_mol = nist_mol.merge(fg_mol, on="neutral_smiles", how="left")

    log.info(f"  {len(nist_mol)} molecules  "
             f"correction mean={nist_mol.correction_kcal.mean():.2f} "
             f"std={nist_mol.correction_kcal.std():.2f} kcal/mol")
    return nist_mol


def load_kmeans_with_corrections() -> pd.DataFrame:
    """k-means: site-level, correction = PA_DFT − PA_PM7."""
    log.info("Loading k-means data (site-level) ...")
    km_feat = pd.read_parquet(DATA_DIR / "features" / "kmeans251_features.parquet")
    km_tgt  = pd.read_parquet(DATA_DIR / "targets"  / "kmeans251_ml.parquet")

    # Site-level PA columns — try multiple naming conventions
    def find_col(df, *keywords, exclude=None):
        for c in df.columns:
            cl = c.lower()
            if all(k in cl for k in keywords):
                if exclude and any(e in cl for e in exclude):
                    continue
                return c
        return None

    # Use kcalmol columns directly — present in kmeans251_ml.parquet
    km_tgt = km_tgt.copy()
    km_tgt["pm7_pa_kcalmol"]  = km_tgt["pm7_pa_kcalmol"]
    km_tgt["ref_pa_kcalmol"]  = km_tgt["dft_pa_kcalmol"]
    km_tgt["correction_kcal"] = km_tgt["delta_dft_pm7"] * KJMOL_TO_KCAL
    log.info(f"  PM7 PA mean: {km_tgt['pm7_pa_kcalmol'].mean():.1f} kcal/mol")
    log.info(f"  DFT PA mean: {km_tgt['ref_pa_kcalmol'].mean():.1f} kcal/mol")
    log.info(f"  Correction mean: {km_tgt['correction_kcal'].mean():.1f} kcal/mol")
    km_tgt = km_tgt.dropna(subset=["correction_kcal"])

    # FG columns already in kmeans251_ml.parquet — just add MW
    mw_col = next((c for c in km_feat.columns
                   if "MolWt" in c or "molwt" in c.lower()), None)
    if mw_col and "neutral_smiles" in km_feat.columns:
        mw_df  = (km_feat.groupby("neutral_smiles")[mw_col]
                  .first().reset_index()
                  .rename(columns={mw_col: "mw"}))
        km_tgt = km_tgt.merge(mw_df, on="neutral_smiles", how="left")
    else:
        km_tgt["mw"] = 150.0

    # Rename neutral_smiles -> neutral_smiles (keep as is)
    log.info(f"  {len(km_tgt)} sites from {km_tgt['neutral_smiles'].nunique()} molecules  "
             f"correction mean={km_tgt.correction_kcal.mean():.2f} "
             f"std={km_tgt.correction_kcal.std():.2f} kcal/mol")
    return km_tgt


def assign_chemical_classes(df: pd.DataFrame) -> pd.DataFrame:
    def get_col(df, *candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    priority = [
        ("Nitrile",         get_col(df, "neutral_rdkit_fr_nitrile")),
        ("Amide",           get_col(df, "neutral_rdkit_fr_amide")),
        ("Carboxylic acid", get_col(df, "neutral_rdkit_fr_COO", "neutral_rdkit_fr_Al_COO")),
        ("Aromatic N",      get_col(df, "neutral_rdkit_fr_Ar_N", "neutral_rdkit_fr_ArN")),
        ("Primary amine",   get_col(df, "neutral_rdkit_fr_NH2")),
        ("Secondary amine", get_col(df, "neutral_rdkit_fr_NH1")),
        ("Tertiary amine",  get_col(df, "neutral_rdkit_fr_NH0")),
        ("Carbonyl",        get_col(df, "neutral_rdkit_fr_C_O")),
        ("Ether",           get_col(df, "neutral_rdkit_fr_ether")),
        ("Halogen",         get_col(df, "neutral_rdkit_fr_halogen")),
    ]
    df = df.copy()
    df["chem_class"] = "Other"
    for name, col in reversed(priority):
        if col is not None:
            df.loc[df[col] > 0, "chem_class"] = name
    found = sum(1 for _, c in priority if c is not None)
    log.info(f"  FG columns matched: {found}/10")
    return df
def mol_to_image(smiles: str, size=(220, 170)):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.drawOptions().addStereoAnnotation = False
        drawer.drawOptions().padding = 0.15
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")
    except Exception:
        return None


    out_dir = FIG_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    log.info(f"  Saved figures/{subdir}/{stem}.pdf/.png")
    plt.close(fig)


def get_smiles_col(df):
    """Get the neutral SMILES column name."""
    for c in ["neutral_smiles", "smiles"]:
        if c in df.columns:
            return c
    return df.columns[0]


# ── Plotting functions (dataset-agnostic) ─────────────────────────────────────

def savefig(fig, stem, subdir="chemical_analysis"):
    out_dir = FIG_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf",):
        fig.savefig(out_dir / f"{stem}.{ext}")
    log.info(f"  Saved figures/{subdir}/{stem}.pdf")
    plt.close(fig)


def plot_correction_by_class(df: pd.DataFrame, tag: str, ref_label: str):
    log.info(f"Plot 1 [{tag}]: Correction by chemical class (violin)")

    class_data = {}
    for cls in CLASS_COLORS:
        sub = df[df["chem_class"] == cls]["correction_kcal"].values
        if len(sub) >= 8:
            class_data[cls] = sub

    if not class_data:
        log.warning("  No classes with N≥8 — skipping")
        return

    sorted_cls = sorted(class_data, key=lambda c: np.median(class_data[c]))
    data_list  = [class_data[c] for c in sorted_cls]
    colors     = [CLASS_COLORS[c] for c in sorted_cls]

    fig, ax = plt.subplots(figsize=(16, 8))

    parts = ax.violinplot(data_list, positions=range(len(sorted_cls)),
                          showmedians=False, showextrema=False, widths=0.72)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.65)
        body.set_edgecolor("white")
        body.set_linewidth(0.6)

    bp = ax.boxplot(data_list, positions=range(len(sorted_cls)),
                    widths=0.18, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2.8),
                    whiskerprops=dict(linewidth=1.4),
                    capprops=dict(linewidth=1.4),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                    markerfacecolor="gray", markeredgewidth=0))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    ax.axhline(0, color="black", lw=1.8, linestyle="--", alpha=0.5, zorder=1)

    y_top = max(d.max() for d in data_list) * 1.08
    for i, (cls, data) in enumerate(zip(sorted_cls, data_list)):
        ax.text(i, y_top,
                f"$\\bf{{N={len(data)}}}$\n{np.mean(data):+.1f}",
                ha="center", va="bottom",
                fontsize=ANNOT_SIZE - 2, color="black", linespacing=1.4)

    ax.set_xticks(range(len(sorted_cls)))
    ax.set_xticklabels(sorted_cls, rotation=35, ha="right",
                       fontsize=TICK_SIZE - 2)
    ax.set_ylabel(f"Signed PM7 correction (kcal/mol)\n"
                  f"[PA$_\\mathrm{{{ref_label}}}$ − PA$_\\mathrm{{PM7}}$]",
                  fontsize=LABEL_SIZE)
    ax.set_xlabel("Chemical class (primary functional group)",
                  fontsize=LABEL_SIZE)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", linewidth=0.5, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.text(0.02, 0.97,
            "Positive = PM7 underestimates PA\nNegative = PM7 overestimates PA",
            transform=ax.transAxes, fontsize=ANNOT_SIZE,
            va="top", style="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="lightgray", lw=1.0, alpha=0.92))
    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(bottom=0.22)
    savefig(fig, f"correction_by_class_{tag}")


def plot_representative_molecules(df: pd.DataFrame, tag: str):
    log.info(f"Plot 2 [{tag}]: Representative molecules per class")
    smiles_col = get_smiles_col(df)

    classes_of_interest = [
        "Nitrile", "Primary amine", "Aromatic N", "Tertiary amine",
        "Amide", "Ether", "Carbonyl", "Halogen",
    ]
    reps = []
    for cls in classes_of_interest:
        sub = df[df["chem_class"] == cls].copy()
        if len(sub) < 5:
            continue
        med  = sub["correction_kcal"].median()
        mean = sub["correction_kcal"].mean()
        std  = sub["correction_kcal"].std()
        pool = sub[(sub["mw"] >= 70) & (sub["mw"] <= 220)].copy() \
               if "mw" in sub.columns else sub.copy()
        if len(pool) < 3:
            pool = sub.copy()
        pool["dist"] = (pool["correction_kcal"] - med).abs()
        rep = pool.nsmallest(1, "dist").iloc[0]
        reps.append({"class": cls, "smiles": rep[smiles_col],
                     "correction": rep["correction_kcal"],
                     "mean": mean, "std": std, "n": len(sub)})

    if not reps:
        log.warning(f"  No representative molecules found for {tag}")
        return

    ncols = 4
    nrows = (len(reps) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.6, nrows * 4.2))
    axes = np.array(axes).flatten()

    for i, rep in enumerate(reps):
        ax  = axes[i]
        img = mol_to_image(rep["smiles"], size=(300, 210))
        if img:
            ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2.2)
            spine.set_edgecolor(CLASS_COLORS.get(rep["class"], "gray"))
        ax.set_title(f"$\\bf{{{rep['class']}}}$", fontsize=15, pad=5)
        ax.set_xlabel(
            f"Mean: {rep['mean']:+.1f} ± {rep['std']:.1f} kcal/mol\n"
            f"N = {rep['n']}",
            fontsize=11, labelpad=5)

    for j in range(len(reps), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Representative molecules by chemical class  [{tag.upper()}]\n"
                 "(structure closest to class median correction)",
                 fontsize=TITLE_SIZE, y=1.02, fontweight="bold")
    fig.tight_layout(pad=1.2)
    savefig(fig, f"representative_molecules_{tag}")


def plot_correction_kde_by_class(df: pd.DataFrame, tag: str, ref_label: str):
    log.info(f"Plot 3 [{tag}]: KDE correction distributions by class")

    highlight = {
        "Nitrile":        ("#E24B4A", "--"),
        "Aromatic N":     ("#534AB7", "-"),
        "Tertiary amine": ("#378ADD", "-"),
        "Primary amine":  ("#85B7EB", "-"),
        "Amide":          ("#AFA9EC", "-."),
        "Ether":          ("#D4537E", ":"),
    }

    fig, ax = plt.subplots(figsize=(12, 7))
    x_min = df["correction_kcal"].min() - 5
    x_max = df["correction_kcal"].max() + 5
    x_range = np.linspace(x_min, x_max, 500)

    plotted = False
    for cls, (color, ls) in highlight.items():
        sub = df[df["chem_class"] == cls]["correction_kcal"].values
        if len(sub) < 8:
            continue
        kde  = gaussian_kde(sub, bw_method=0.35)
        y    = kde(x_range)
        mean = np.mean(sub)
        ax.plot(x_range, y, color=color, lw=2.5, linestyle=ls,
                label=f"$\\bf{{{cls}}}$  (N={len(sub)}, mean={mean:+.1f})")
        ax.fill_between(x_range, y, alpha=0.10, color=color)
        ax.axvline(mean, color=color, lw=1.3, linestyle=":", alpha=0.75)
        plotted = True

    if not plotted:
        plt.close(fig)
        log.warning(f"  Not enough data for KDE [{tag}]")
        return

    ax.axvline(0, color="black", lw=2.0, linestyle="--", alpha=0.6,
               label="Zero (no correction needed)")
    ax.set_xlabel(f"Signed PM7 correction (kcal/mol)  "
                  f"[PA$_\\mathrm{{{ref_label}}}$ − PA$_\\mathrm{{PM7}}$]",
                  fontsize=LABEL_SIZE)
    ax.set_ylabel("Probability Density", fontsize=LABEL_SIZE)
    ax.legend(framealpha=0.92, edgecolor="lightgray",
              fontsize=LEGEND_SIZE, loc="upper left")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", linewidth=0.5, alpha=0.35, linestyle="--")

    # Nitrile annotation if present
    nit = df[df["chem_class"] == "Nitrile"]["correction_kcal"].values
    if len(nit) >= 8:
        ax.annotate("PM7 overestimates\nPA for nitriles",
                    xy=(np.mean(nit), 0.008),
                    xytext=(np.mean(nit) - 20, 0.025),
                    fontsize=ANNOT_SIZE, color="#E24B4A", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#E24B4A", lw=1.5))

    fig.tight_layout(pad=1.5)
    savefig(fig, f"correction_kde_by_class_{tag}")


def plot_correction_vs_pa(df: pd.DataFrame, tag: str, ref_label: str):
    log.info(f"Plot 4 [{tag}]: Signed correction vs reference PA")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── Left: binned mean ± std ──────────────────────────────────────────────
    ax_bin = axes[0]
    df2    = df.dropna(subset=["ref_pa_kcalmol", "correction_kcal"])
    lo     = np.floor(df2["ref_pa_kcalmol"].min() / 10) * 10
    hi     = np.ceil(df2["ref_pa_kcalmol"].max()  / 10) * 10
    bins   = np.arange(lo, hi + 10, 10)
    df2    = df2.copy()
    df2["pa_bin"] = pd.cut(df2["ref_pa_kcalmol"], bins=bins)
    bin_stats = (df2.groupby("pa_bin", observed=True)["correction_kcal"]
                 .agg(["mean", "std", "count"]).reset_index())
    bin_stats    = bin_stats[bin_stats["count"] >= 5]
    bin_centers  = [(b.left + b.right) / 2 for b in bin_stats["pa_bin"]]
    point_colors = ["#D73027" if m < 0 else "#2166AC" for m in bin_stats["mean"]]

    ax_bin.fill_between(bin_centers,
                        bin_stats["mean"] - bin_stats["std"],
                        bin_stats["mean"] + bin_stats["std"],
                        alpha=0.20, color="#888888")
    ax_bin.plot(bin_centers, bin_stats["mean"], "-",
                color="#333333", lw=2.2, zorder=2)
    ax_bin.scatter(bin_centers, bin_stats["mean"].values,
                   c=point_colors, s=90, zorder=3,
                   edgecolors="white", linewidths=0.8)
    ax_bin.axhline(0, color="black", lw=1.8, ls="--", alpha=0.6)
    ax_bin.set_xlabel(f"Reference PA — {ref_label}\n(kcal/mol)", fontsize=LABEL_SIZE)
    ax_bin.set_ylabel("Mean correction ± std\n(kcal/mol)", fontsize=LABEL_SIZE)
    ax_bin.text(0.97, 0.97, "Binned mean ± std",
                transform=ax_bin.transAxes, ha="right", va="top",
                fontsize=ANNOT_SIZE, style="italic", fontweight="bold")
    ax_bin.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax_bin.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # ── Right: scatter by class ──────────────────────────────────────────────
    ax_sc = axes[1]
    for cls in sorted(CLASS_COLORS,
                      key=lambda c: df[df["chem_class"]==c]["correction_kcal"].mean()
                      if len(df[df["chem_class"]==c]) >= 5 else 0):
        sub = df[df["chem_class"] == cls]
        if len(sub) < 5:
            continue
        mean_corr = sub["correction_kcal"].mean()
        color = "#D73027" if mean_corr < 0 else CLASS_COLORS.get(cls, "#2166AC")
        ax_sc.scatter(sub["ref_pa_kcalmol"], sub["correction_kcal"],
                      color=color, s=18, alpha=0.65,
                      linewidths=0, label=cls)
    ax_sc.axhline(0, color="black", lw=1.8, ls="--", alpha=0.6)
    ax_sc.set_xlabel(f"Reference PA — {ref_label}\n(kcal/mol)", fontsize=LABEL_SIZE)
    ax_sc.set_ylabel("Signed PM7 correction\n(kcal/mol)", fontsize=LABEL_SIZE)
    ax_sc.legend(fontsize=LEGEND_SIZE - 5, framealpha=0.9,
                 edgecolor="lightgray", markerscale=2.0,
                 ncol=1, loc="upper left")
    ax_sc.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax_sc.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    fig.tight_layout(pad=1.5)
    savefig(fig, f"correction_vs_pa_{tag}")

def plot_worst_best_molecules(df: pd.DataFrame, tag: str, ref_label: str):
    log.info(f"Plot 5 [{tag}]: Worst and best predicted molecules")
    smiles_col = get_smiles_col(df)

    worst = df.nlargest(6,  "correction_kcal")[
        [smiles_col, "correction_kcal", "ref_pa_kcalmol", "chem_class"]]
    best  = df.nsmallest(6, "correction_kcal")[
        [smiles_col, "correction_kcal", "ref_pa_kcalmol", "chem_class"]]

    fig, axes = plt.subplots(2, 6, figsize=(20, 8))

    for row_idx, (subset, row_label, color) in enumerate([
        (worst, f"Largest +ve correction\n(PM7 most underestimates)", "#2166AC"),
        (best,  f"Largest −ve correction\n(PM7 most overestimates)",  "#D01C8B"),
    ]):
        for col_idx, (_, row) in enumerate(subset.reset_index().iterrows()):
            ax  = axes[row_idx][col_idx]
            img = mol_to_image(row[smiles_col], size=(260, 190))
            if img:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(2.2)
                spine.set_edgecolor(color)
            ax.set_xlabel(
                f"$\\bf{{{row['correction_kcal']:+.1f}}}$ kcal/mol\n"
                f"PA={row['ref_pa_kcalmol']:.1f}  [{row['chem_class']}]",
                fontsize=10, labelpad=4)
        axes[row_idx][0].set_ylabel(row_label, fontsize=13,
                                     labelpad=10, color=color,
                                     fontweight="bold")

    fig.suptitle(f"Molecules with largest PM7 correction errors  [{tag.upper()}]",
                 fontsize=TITLE_SIZE, y=1.02, fontweight="bold")
    fig.tight_layout(pad=1.0)
    savefig(fig, f"worst_best_molecules_{tag}")


def plot_correction_summary_table(df: pd.DataFrame, tag: str):
    log.info(f"Plot 6 [{tag}]: Summary table by chemical class")

    rows = []
    for cls in CLASS_COLORS:
        sub = df[df["chem_class"] == cls]["correction_kcal"]
        if len(sub) < 8:
            continue
        rows.append({
            "Class":           cls,
            "N":               len(sub),
            "Mean (kcal/mol)": f"{sub.mean():+.2f}",
            "Std":             f"{sub.std():.2f}",
            "Median":          f"{sub.median():+.2f}",
            "% overestimate":  f"{100*(sub < 0).mean():.0f}%",
        })
    if not rows:
        log.warning(f"  No data for summary table [{tag}]")
        return

    df_table = pd.DataFrame(rows).sort_values("Mean (kcal/mol)")
    fig, ax  = plt.subplots(figsize=(13, 5.5))
    ax.axis("off")

    table = ax.table(
        cellText=df_table.values.tolist(),
        colLabels=list(df_table.columns),
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1, 2.1)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)
        cell.set_facecolor("white")
        if row == 0:
            cell.set_text_props(color="black", fontweight="bold", fontsize=16)
        else:
            cell.set_text_props(color="black", fontsize=15)

    fig.tight_layout()
    savefig(fig, f"correction_summary_table_{tag}")


# ── Per-dataset runner ────────────────────────────────────────────────────────

def run_dataset(dataset: str):
    log.info("=" * 60)
    log.info(f"  CHEMICAL CLASS ANALYSIS — {dataset.upper()}")
    log.info("=" * 60)

    if dataset == "nist":
        df        = load_nist_with_corrections()
        ref_label = "exp"
        title_ref = "Experimental PA reference (NIST)"
    else:
        df        = load_kmeans_with_corrections()
        ref_label = "DFT"
        title_ref = "DFT PA reference (B3LYP/def2-TZVP)"

    df = assign_chemical_classes(df)

    log.info("Class distribution:")
    for cls, n in df["chem_class"].value_counts().items():
        log.info(f"  {cls:<20}: {n:4d} ({100*n/len(df):.1f}%)")

    tag = dataset
    plot_correction_by_class(df,            tag, ref_label)
    plot_representative_molecules(df,       tag)
    plot_correction_vs_pa(df,               tag, ref_label)
    plot_worst_best_molecules(df,           tag, ref_label)
    plot_correction_summary_table(df,       tag)

    log.info(f"  All {dataset.upper()} figures saved to figures/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chemical class analysis of PM7 corrections.")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "nist", "kmeans"])
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("all", "nist"):
        run_dataset("nist")
    if args.dataset in ("all", "kmeans"):
        run_dataset("kmeans")

    print("\n  All figures saved to: figures/")
    print("  Suffixes: _nist  and  _kmeans")


if __name__ == "__main__":
    main()
