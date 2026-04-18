"""
plot_chemical_analysis.py
=========================
Chemical class analysis of PM7 correction patterns.
Runs for BOTH NIST (experimental reference) and k-means (DFT reference).
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
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR    = PROJECT_DIR / "data"
FIG_DIR     = PROJECT_DIR / "figures"
FIG_CHEM    = FIG_DIR / "chemical_analysis"

KJMOL_TO_KCAL = 1 / 4.184

# ── ACS double-column journal style ────────────────────────────────────────
# Reference: ACS (JCIM/JCTC) double-column width = 6.9 in (175 mm)
#   - 9 pt minimum font at final printed size
#   - Save PDF as vector (pdf.fonttype=42 embeds fonts as TrueType)
#   - 600 DPI for raster export
DOUBLE_COL  = 6.9   # ACS double-column width (inches)

TICK_SIZE   = 26
LABEL_SIZE  = 30
LEGEND_SIZE = 24
ANNOT_SIZE  = 22
TITLE_SIZE  = 30
SPINE_LW    = 1.6
DPI         = 600

plt.rcParams.update({
    # Font
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":             26,
    "axes.titlesize":        TITLE_SIZE,
    "axes.labelsize":        LABEL_SIZE,
    "xtick.labelsize":       TICK_SIZE,
    "ytick.labelsize":       TICK_SIZE,
    "legend.fontsize":       LEGEND_SIZE,
    "legend.title_fontsize": LEGEND_SIZE,
    # Lines & markers
    "lines.linewidth":       2.0,
    "lines.markersize":      8,
    "axes.linewidth":        SPINE_LW,
    "xtick.major.width":     SPINE_LW,     "ytick.major.width":  SPINE_LW,
    "xtick.minor.width":     1.0,          "ytick.minor.width":  1.0,
    "xtick.major.size":      6,            "ytick.major.size":   6,
    "xtick.minor.size":      4,            "ytick.minor.size":   4,
    "xtick.direction":       "in",         "ytick.direction":    "in",
    # Output
    "figure.dpi":            300,
    "savefig.dpi":           DPI,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.05,
    # Style
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "legend.frameon":        False,
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
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

    def find_col(df, *keywords, exclude=None):
        for c in df.columns:
            cl = c.lower()
            if all(k in cl for k in keywords):
                if exclude and any(e in cl for e in exclude):
                    continue
                return c
        return None

    km_tgt = km_tgt.copy()
    km_tgt["pm7_pa_kcalmol"]  = km_tgt["pm7_pa_kcalmol"]
    km_tgt["ref_pa_kcalmol"]  = km_tgt["dft_pa_kcalmol"]
    km_tgt["correction_kcal"] = km_tgt["delta_dft_pm7"] * KJMOL_TO_KCAL
    log.info(f"  PM7 PA mean: {km_tgt['pm7_pa_kcalmol'].mean():.1f} kcal/mol")
    log.info(f"  DFT PA mean: {km_tgt['ref_pa_kcalmol'].mean():.1f} kcal/mol")
    log.info(f"  Correction mean: {km_tgt['correction_kcal'].mean():.1f} kcal/mol")
    km_tgt = km_tgt.dropna(subset=["correction_kcal"])

    mw_col = next((c for c in km_feat.columns
                   if "MolWt" in c or "molwt" in c.lower()), None)
    if mw_col and "neutral_smiles" in km_feat.columns:
        mw_df  = (km_feat.groupby("neutral_smiles")[mw_col]
                  .first().reset_index()
                  .rename(columns={mw_col: "mw"}))
        km_tgt = km_tgt.merge(mw_df, on="neutral_smiles", how="left")
    else:
        km_tgt["mw"] = 150.0

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


def mol_to_image(smiles: str, size=(1200, 840)):
    """Draws a high-res RDKit image with thick bonds and legible text.
    White background guaranteed."""
    try:
        from PIL import ImageEnhance
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = False
        opts.padding = 0.12
        opts.bondLineWidth = 14
        opts.minFontSize = 78
        opts.multipleBondOffset = 0.22
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")
        img = ImageEnhance.Contrast(img).enhance(1.6)
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        white_bg.paste(img, mask=img)
        return white_bg
    except Exception:
        return None


def mol_to_image_dark(smiles: str, size=(1200, 840)):
    """Draws an RDKit image for the combined manuscript figure.

    Two-pass render: a probe pass measures the drawn content extent, then a
    final pass chooses ``fixedBondLength`` so each structure fills the target
    canvas up to a safety margin. Bond thickness and atom font sizes stay
    fixed in pixels, so appearance remains uniform across molecules.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        tw, th = int(size[0]), int(size[1])
        margin = 0.94

        def _render(bond_length: float):
            drawer = rdMolDraw2D.MolDraw2DCairo(tw, th)
            opts = drawer.drawOptions()
            opts.addStereoAnnotation = False
            opts.padding = 0.02
            opts.bondLineWidth = 14
            opts.minFontSize = 64
            opts.maxFontSize = 86
            opts.fixedBondLength = float(bond_length)
            opts.multipleBondOffset = 0.18
            opts.useDefaultAtomPalette()
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            return Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")

        def _content_bbox(img: Image.Image):
            arr = np.array(img.convert("RGB"))
            nonwhite = np.any(arr < 245, axis=2)
            if not np.any(nonwhite):
                return None
            ys, xs = np.where(nonwhite)
            return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        probe_bl = 80.0
        probe = _render(probe_bl)
        bbox = _content_bbox(probe)
        if bbox is None:
            final = probe
        else:
            cx0, cy0, cx1, cy1 = bbox
            cw = max(1, cx1 - cx0 + 1)
            ch = max(1, cy1 - cy0 + 1)
            scale = min((tw * margin) / cw, (th * margin) / ch)
            target_bl = max(10.0, min(probe_bl * scale, 240.0))
            final = _render(target_bl)

        white_bg = Image.new("RGBA", final.size, (255, 255, 255, 255))
        white_bg.paste(final, mask=final)
        return white_bg
    except Exception:
        return None


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
    fig.savefig(out_dir / f"{stem}.pdf")
    log.info(f"  Saved figures/{subdir}/{stem}.pdf")
    plt.close(fig)


def plot_correction_by_class(df: pd.DataFrame, tag: str, ref_label: str,
                             class_order=None):
    """
    Violin + box plot of PM7 correction by chemical class.

    class_order : optional list of class names defining the x-axis sequence.
        Classes present in the data but absent from class_order are appended
        at the end sorted by median.  Pass the NIST order to k-means so both
        plots share the same left-to-right sequence.

    Returns sorted_cls (list) so the caller can reuse the order.
    """
    log.info(f"Plot 1 [{tag}]: Correction by chemical class (violin)")

    class_data = {}
    for cls in CLASS_COLORS:
        sub = df[df["chem_class"] == cls]["correction_kcal"].values
        if len(sub) >= 8:
            class_data[cls] = sub

    if not class_data:
        log.warning("  No classes with N>=8 -- skipping")
        return []

    # ── Class ordering ────────────────────────────────────────────────────
    if class_order is not None:
        sorted_cls = [c for c in class_order if c in class_data]
        leftover   = sorted(
            (c for c in class_data if c not in sorted_cls),
            key=lambda c: np.median(class_data[c])
        )
        sorted_cls += leftover
    else:
        sorted_cls = sorted(class_data, key=lambda c: np.median(class_data[c]))

    data_list = [class_data[c] for c in sorted_cls]
    colors    = [CLASS_COLORS[c] for c in sorted_cls]
    n_cls     = len(sorted_cls)

    # Scale width so N= labels never crowd: ~1.8 in per violin, min 14 in
    fig_w = max(14, n_cls * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, 9))

    # ── Violin bodies ──────────────────────────────────────────────────────
    parts = ax.violinplot(data_list, positions=range(n_cls),
                          showmedians=False, showextrema=False, widths=0.72)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.80)
        body.set_edgecolor("white")
        body.set_linewidth(0.8)

    # ── Box plots ─────────────────────────────────────────────────────────
    bp = ax.boxplot(data_list, positions=range(n_cls),
                    widths=0.20, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2.2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=4, alpha=0.45,
                                    markerfacecolor="gray", markeredgewidth=0))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.95)

    ax.axhline(0, color="black", lw=1.2, linestyle="--", alpha=0.6, zorder=1)

    # ── N= (bold) and mean correction (italic) at y=-25 inside the plot ───
    for i, (cls, data) in enumerate(zip(sorted_cls, data_list)):
        ax.text(i, -25, f"N={len(data)}",
                ha="center", va="top",
                fontsize=ANNOT_SIZE, color="#222222", fontweight="bold")
        ax.text(i, -31, f"{np.mean(data):+.1f}",
                ha="center", va="top",
                fontsize=ANNOT_SIZE, color="#444444", style="italic")

    # ── Positive/Negative legend at tag-specific y (whitespace above violins)
    legend_y = 40 if tag == "nist" else 70
    ax.text(-0.4, legend_y,
            "Positive = PM7 underestimates PA\nNegative = PM7 overestimates PA",
            ha="left", va="top",
            fontsize=ANNOT_SIZE, style="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="lightgray", lw=0.7, alpha=0.92))

    # ── Y limits ──────────────────────────────────────────────────────────
    y_data_top = max(d.max() for d in data_list)
    ax.set_ylim(bottom=-38, top=max(y_data_top * 1.08, legend_y + 8))

    # ── X-tick labels ─────────────────────────────────────────────────────
    ax.set_xticks(range(n_cls))
    ax.set_xticklabels(sorted_cls, rotation=35, ha="right", fontsize=TICK_SIZE)

    # ── Axis labels ────────────────────────────────────────────────────────
    ax.set_ylabel(
        "Signed PM7 correction\n(kcal/mol)\n"
        f"[PA$_\\mathrm{{{ref_label}}}$ $-$ PA$_\\mathrm{{PM7}}$]",
        fontsize=LABEL_SIZE, labelpad=10,
    )
    ax.set_xlabel("Chemical class (primary functional group)",
                  fontsize=LABEL_SIZE, labelpad=12)

    # ── Grid ──────────────────────────────────────────────────────────────
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE,
                   width=SPINE_LW, length=6)
    ax.tick_params(axis="both", which="minor", width=1.0, length=4)
    ax.grid(axis="y", linewidth=0.6, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout(pad=1.8)
    fig.subplots_adjust(bottom=0.28, left=0.22)
    savefig(fig, f"correction_by_class_{tag}")
    return sorted_cls


def plot_correction_combined(nist_df: pd.DataFrame, km_df: pd.DataFrame,
                             class_order=None):
    """
    2-panel figure: top = NIST violin, bottom = k-means violin.
    Panel mapping:
      (a) top panel  -> NIST (PA_exp - PA_PM7)
      (b) bottom panel -> k-means (PA_DFT - PA_PM7)
    Sign convention:
      Positive correction -> PM7 underestimates PA
      Negative correction -> PM7 overestimates PA
    """
    log.info("Plot combined: Correction by class — NIST (top) + k-means (bottom)")

    def _build(df):
        return {cls: df[df["chem_class"] == cls]["correction_kcal"].values
                for cls in CLASS_COLORS
                if len(df[df["chem_class"] == cls]["correction_kcal"]) >= 8}

    nist_cd = _build(nist_df)
    km_cd   = _build(km_df)

    def _sort(cd, order):
        if order is not None:
            sc = [c for c in order if c in cd]
            sc += sorted((c for c in cd if c not in sc),
                         key=lambda c: np.median(cd[c]))
        else:
            sc = sorted(cd, key=lambda c: np.median(cd[c]))
        return sc

    # Top panel defines canonical order (NIST). Bottom panel follows
    # the same order while programmatically skipping classes absent in k-means.
    nist_cls = _sort(nist_cd, class_order)
    km_cls   = [c for c in nist_cls if c in km_cd]

    n_nist = len(nist_cls)
    n_km   = len(km_cls)
    fig_w  = DOUBLE_COL

    # Journal-style typography (final printed scale).
    FS_TICK  = 9
    FS_LABEL = 10
    FS_PANEL = 11
    FS_TOP   = 9           # top-axis annotation labels
    # Use standard high-contrast text colors distinct from class fills.
    N_TEXT_COLOR = "#1F4E79"       # dark blue
    SIGNED_TEXT_COLOR = "#B22222"  # firebrick red
    SPINE_W  = 1.0
    ANN_D_OFFSET_TOP = 3.0      # Δ̄ y = violin_max + offset (top panel)
    ANN_D_OFFSET_BOTTOM = 3.0   # Δ̄ y = violin_max + offset (bottom panel)
    ANN_N_ABOVE_D_TOP = 11.0    # N y = signed-value y + extra offset (top panel)
    ANN_N_ABOVE_D_BOTTOM = 7.8  # N y = Δ̄ y + extra offset (bottom panel)
    ANN_D_STAGGER_TOP = 4.0     # extra signed-value offset for alternating classes (top)
    ANN_D_STAGGER_BOTTOM = 1.2  # extra Δ̄ offset for alternating classes (bottom)

    fig, (ax_n, ax_k) = plt.subplots(2, 1, figsize=(fig_w, 6.2))

    def _draw(ax, cd, sorted_cls, ref_label, panel_label,
              ann_d_offset, ann_n_above_d, ann_d_stagger,
              alt_extra_lift=0.0, skip_last_alt=0):
        n_cls     = len(sorted_cls)
        data_list = [cd[c] for c in sorted_cls]
        colors    = [CLASS_COLORS[c] for c in sorted_cls]

        parts = ax.violinplot(data_list, positions=range(n_cls),
                              showmedians=False, showextrema=False, widths=0.75)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.80)
            body.set_edgecolor("white")
            body.set_linewidth(0.8)

        bp = ax.boxplot(data_list, positions=range(n_cls),
                        widths=0.20, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.2),
                        whiskerprops=dict(linewidth=1.0),
                        capprops=dict(linewidth=1.0),
                        flierprops=dict(marker="o", markersize=4, alpha=0.45,
                                        markerfacecolor="gray", markeredgewidth=0))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.95)

        ax.axhline(0, color="black", lw=1.0, linestyle="--", alpha=0.6, zorder=1)

        # ── Bottom x-axis: class names only ──────────────────────────────
        ax.set_xticks(range(n_cls))
        ax.set_xticklabels(sorted_cls, rotation=35, ha="right",
                           fontsize=FS_TICK)

        y_data_top = max(d.max() for d in data_list)
        y_data_bot = min(d.min() for d in data_list)
        y_min = min(y_data_bot * 1.10, -28)
        y_max = max(y_data_top * 1.15,
                    y_data_top + ann_d_offset + ann_d_stagger + ann_n_above_d + 1.0)
        ax.set_ylim(bottom=y_min, top=y_max)

        # Place signed corrections in a fixed row near the x-axis.
        signed_y = y_min + 2.0
        for i, data in enumerate(data_list):
            ann_d_y = data.max() + ann_d_offset + (i % 2) * ann_d_stagger
            # Optional extra lift for alternating labels (used for top panel).
            if (i % 2) == 1 and i < (n_cls - skip_last_alt):
                ann_d_y += alt_extra_lift
            ann_n_y = ann_d_y + ann_n_above_d
            ann_n_x = i
            # NIST-specific cleanup: avoid left-edge crowding and local overlap.
            if panel_label == "(a)" and i == 0:
                ann_n_x += 0.18
            if panel_label == "(a)" and i == 2:
                ann_n_y -= 3.2
            ax.text(
                ann_n_x, ann_n_y,
                f"N={len(data)}",
                ha="center", va="bottom", fontsize=FS_TOP,
                color=N_TEXT_COLOR
            )
            ax.text(
                i, signed_y,
                f"{np.mean(data):+.1f}",
                ha="center", va="bottom", fontsize=FS_TOP,
                color=SIGNED_TEXT_COLOR
            )

        # Y-axis label — wrap (kcal/mol) to its own line
        ax.set_ylabel(
            "Signed PM7 correction\n(kcal/mol)\n"
            f"[PA$_\\mathrm{{{ref_label}}}$ $-$ PA$_\\mathrm{{PM7}}$]",
            fontsize=FS_LABEL, labelpad=14,
        )

        # Thick spines
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_W)

        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(axis="both", which="major", labelsize=FS_TICK,
                       width=SPINE_W, length=4)
        ax.tick_params(axis="both", which="minor", width=0.8, length=2)
        ax.grid(axis="y", linewidth=0.6, alpha=0.4, linestyle="--")
        ax.set_axisbelow(True)

        # Panel label
        ax.text(-0.10, 1.08, panel_label, transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight="bold", va="top")

    _draw(ax_n, nist_cd, nist_cls, "exp", panel_label="(a)",
          ann_d_offset=ANN_D_OFFSET_TOP,
          ann_n_above_d=ANN_N_ABOVE_D_TOP,
          ann_d_stagger=ANN_D_STAGGER_TOP,
          alt_extra_lift=1.8, skip_last_alt=2)
    _draw(ax_k, km_cd,   km_cls,   "DFT", panel_label="(b)",
          ann_d_offset=ANN_D_OFFSET_BOTTOM,
          ann_n_above_d=ANN_N_ABOVE_D_BOTTOM,
          ann_d_stagger=ANN_D_STAGGER_BOTTOM)

    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(left=0.12, bottom=0.16, top=0.96, hspace=0.52)
    savefig(fig, "correction_by_class_combined")


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

    # Adaptive grid: 3 cols for ≤6 molecules (k-means), 4 cols otherwise
    ncols = 3 if len(reps) <= 6 else 4
    nrows = (len(reps) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 6.5, nrows * 7.5))
    axes = np.array(axes).flatten()

    REP_TITLE_FS = 34
    REP_XLABEL_FS = 28

    for i, rep in enumerate(reps):
        ax  = axes[i]
        img = mol_to_image(rep["smiles"], size=(1200, 840))
        if img:
            ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(3.0)
            spine.set_edgecolor(CLASS_COLORS.get(rep["class"], "gray"))
        ax.set_title(f"$\\bf{{{rep['class']}}}$", fontsize=REP_TITLE_FS, pad=10)
        ax.set_xlabel(
            f"Mean: {rep['mean']:+.1f} \u00b1 {rep['std']:.1f} kcal/mol\n"
            f"N = {rep['n']}",
            fontsize=REP_XLABEL_FS, labelpad=10)

    for j in range(len(reps), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Representative molecules by chemical class  [{tag.upper()}]\n"
                 "(structure closest to class median correction)",
                 fontsize=36, y=1.02, fontweight="bold")
    fig.tight_layout(pad=1.8)
    savefig(fig, f"representative_molecules_{tag}")


def _collect_representatives(df: pd.DataFrame):
    """Return a list of representative-molecule dicts for the given dataset.

    Mirrors the logic in `plot_representative_molecules` but returns the
    selection rather than rendering. Used by the combined figure.
    """
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
        med = sub["correction_kcal"].median()
        mean = sub["correction_kcal"].mean()
        std = sub["correction_kcal"].std()
        pool = sub[(sub["mw"] >= 70) & (sub["mw"] <= 220)].copy() \
               if "mw" in sub.columns else sub.copy()
        if len(pool) < 3:
            pool = sub.copy()
        pool["dist"] = (pool["correction_kcal"] - med).abs()
        rep = pool.nsmallest(1, "dist").iloc[0]
        reps.append({
            "class":      cls,
            "smiles":     rep[smiles_col],
            "correction": rep["correction_kcal"],
            "mean":       mean,
            "std":        std,
            "n":          len(sub),
        })
    return reps


def plot_representative_molecules_combined(nist_df: pd.DataFrame,
                                           km_df: pd.DataFrame):
    """Single journal-spec figure with four rows of representative molecules.

    Layout:
        Rows 0-1 : NIST classes  (panel label "(a)")
        Rows 2-3 : k-means class (panel label "(b)")
    Each row holds up to ``ncols`` panels; trailing slots are hidden if a
    dataset has fewer surviving classes than the grid can hold.
    """
    log.info("Plot: Representative molecules combined (NIST + k-means)")

    nist_reps = _collect_representatives(nist_df)
    km_reps   = _collect_representatives(km_df)
    if not nist_reps and not km_reps:
        log.warning("  No representatives found for either dataset; skipping.")
        return

    ncols = 4
    fig_w = 7.0
    fig_h = 7.6

    # Reserve a top band for the (a) panel label and a band between the two
    # blocks for the (b) panel label. Row spacing is reduced so the figure
    # stays the same height despite the new banner room.
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    gs = fig.add_gridspec(
        4, ncols,
        left=0.035, right=0.985,
        top=0.910,  bottom=0.035,
        hspace=0.55, wspace=0.20,
    )

    # Journal-spec typography (matches other paper figures)
    TITLE_FS  = 9
    XLABEL_FS = 8
    PANEL_FS  = 11

    # Vertical offset (figure-fraction) between the top of the first axis in
    # a block and the panel-label baseline. Title pad is small (4 pt) so the
    # title hugs the molecule, leaving the offset well above it.
    PANEL_LABEL_OFFSET = 0.038

    def _render_block(reps, row_offset: int, panel_label: str):
        if not reps:
            return
        for i, rep in enumerate(reps):
            r = row_offset + i // ncols
            c = i % ncols
            if r >= row_offset + 2:
                break
            ax = fig.add_subplot(gs[r, c])
            img = mol_to_image_dark(rep["smiles"], size=(900, 700))
            if img is not None:
                ax.imshow(img, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_edgecolor("0.3")
            ax.set_title(rep["class"], fontsize=TITLE_FS,
                         fontweight="bold", pad=4)
            ax.set_xlabel(
                f"{rep['mean']:+.1f} \u00b1 {rep['std']:.1f} kcal/mol\n"
                f"N = {rep['n']}",
                fontsize=XLABEL_FS, labelpad=2,
            )
        # Hide unused slots in this 2-row block.
        for j in range(len(reps), 2 * ncols):
            r = row_offset + j // ncols
            c = j % ncols
            if r >= row_offset + 2:
                break
            ax = fig.add_subplot(gs[r, c])
            ax.set_visible(False)

        # Panel label sits clearly above the row's molecule titles.
        first_ax = fig.add_subplot(gs[row_offset, 0])
        first_ax.set_visible(False)
        fig.text(
            0.005,
            first_ax.get_position().y1 + PANEL_LABEL_OFFSET,
            panel_label,
            fontsize=PANEL_FS, fontweight="bold",
            ha="left", va="bottom",
        )

    _render_block(nist_reps, row_offset=0, panel_label="(a) NIST")
    _render_block(km_reps,   row_offset=2, panel_label="(b) k-means")

    out_dir = FIG_DIR / "chemical_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "representative_molecules_combined.pdf"
    png_path = out_dir / "representative_molecules_combined.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.08, dpi=250)
    plt.close(fig)
    log.info(f"  Saved {pdf_path}")
    log.info(f"  Saved {png_path}")


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

    fig, ax = plt.subplots(figsize=(18, 10))
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
        ax.plot(x_range, y, color=color, lw=3.5, linestyle=ls,
                label=f"$\\bf{{{cls}}}$  (N={len(sub)}, mean={mean:+.1f})")
        ax.fill_between(x_range, y, alpha=0.10, color=color)
        ax.axvline(mean, color=color, lw=2.0, linestyle=":", alpha=0.75)
        plotted = True

    if not plotted:
        plt.close(fig)
        log.warning(f"  Not enough data for KDE [{tag}]")
        return

    ax.axvline(0, color="black", lw=2.5, linestyle="--", alpha=0.6,
               label="Zero (no correction needed)")
    ax.set_xlabel(f"Signed PM7 correction (kcal/mol)  "
                  f"[PA$_\\mathrm{{{ref_label}}}$ \u2212 PA$_\\mathrm{{PM7}}$]",
                  fontsize=LABEL_SIZE)
    ax.set_ylabel("Probability Density", fontsize=LABEL_SIZE)
    ax.legend(framealpha=0.92, edgecolor="lightgray",
              fontsize=LEGEND_SIZE, loc="upper left")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis="y", linewidth=0.6, alpha=0.4, linestyle="--")

    nit = df[df["chem_class"] == "Nitrile"]["correction_kcal"].values
    if len(nit) >= 8:
        ax.annotate("PM7 overestimates\nPA for nitriles",
                    xy=(np.mean(nit), 0.008),
                    xytext=(np.mean(nit) - 20, 0.025),
                    fontsize=ANNOT_SIZE, color="#E24B4A", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#E24B4A", lw=2.5))

    fig.tight_layout(pad=1.8)
    savefig(fig, f"correction_kde_by_class_{tag}")


def plot_correction_vs_pa(df: pd.DataFrame, tag: str, ref_label: str):
    log.info(f"Plot 4 [{tag}]: Signed correction vs reference PA")

    # Large font sizes for manuscript readability
    VS_TICK  = 30
    VS_LABEL = 34
    VS_ANNOT = 28
    VS_LEG   = 26

    fig, axes = plt.subplots(1, 2, figsize=(30, 12))

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
                color="#333333", lw=3.0, zorder=2)
    ax_bin.scatter(bin_centers, bin_stats["mean"].values,
                   c=point_colors, s=200, zorder=3,
                   edgecolors="white", linewidths=1.5)
    ax_bin.axhline(0, color="black", lw=2.2, ls="--", alpha=0.6)
    ax_bin.set_xlabel(f"Reference PA \u2014 {ref_label}\n(kcal/mol)", fontsize=VS_LABEL)
    ax_bin.set_ylabel("Mean correction \u00b1 std\n(kcal/mol)", fontsize=VS_LABEL)
    ax_bin.text(0.97, 0.97, "Binned mean \u00b1 std",
                transform=ax_bin.transAxes, ha="right", va="top",
                fontsize=VS_ANNOT, style="italic", fontweight="bold")
    ax_bin.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax_bin.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax_bin.tick_params(axis="both", which="major", labelsize=VS_TICK,
                       width=SPINE_LW, length=6)
    ax_bin.tick_params(axis="both", which="minor", width=1.0, length=4)

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
                      color=color, s=70, alpha=0.65,
                      linewidths=0, label=cls)
    ax_sc.axhline(0, color="black", lw=2.2, ls="--", alpha=0.6)
    ax_sc.set_xlabel(f"Reference PA \u2014 {ref_label}\n(kcal/mol)", fontsize=VS_LABEL)
    ax_sc.set_ylabel("Signed PM7 correction\n(kcal/mol)", fontsize=VS_LABEL)
    ax_sc.legend(fontsize=VS_LEG, framealpha=0.9,
                 edgecolor="lightgray", markerscale=2.5,
                 ncol=1, loc="upper left")
    ax_sc.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax_sc.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax_sc.tick_params(axis="both", which="major", labelsize=VS_TICK,
                      width=SPINE_LW, length=6)
    ax_sc.tick_params(axis="both", which="minor", width=1.0, length=4)

    fig.tight_layout(pad=2.0)
    savefig(fig, f"correction_vs_pa_{tag}")


def plot_worst_best_molecules(df: pd.DataFrame, tag: str, ref_label: str):
    log.info(f"Plot 5 [{tag}]: Worst and best predicted molecules")
    smiles_col = get_smiles_col(df)

    worst = df.nlargest(6,  "correction_kcal")[
        [smiles_col, "correction_kcal", "ref_pa_kcalmol", "chem_class"]]
    best  = df.nsmallest(6, "correction_kcal")[
        [smiles_col, "correction_kcal", "ref_pa_kcalmol", "chem_class"]]

    fig, axes = plt.subplots(2, 6, figsize=(30, 14))

    for row_idx, (subset, row_label, color) in enumerate([
        (worst, f"Largest +ve correction\n(PM7 most underestimates)", "#2166AC"),
        (best,  f"Largest \u2212ve correction\n(PM7 most overestimates)",  "#D01C8B"),
    ]):
        for col_idx, (_, row) in enumerate(subset.reset_index().iterrows()):
            ax  = axes[row_idx][col_idx]
            img = mol_to_image(row[smiles_col], size=(1200, 840))
            if img:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(3.5)
                spine.set_edgecolor(color)

            ax.set_xlabel(
                f"$\\bf{{{row['correction_kcal']:+.1f}}}$ kcal/mol\n"
                f"PA={row['ref_pa_kcalmol']:.1f}\n[{row['chem_class']}]",
                fontsize=32, labelpad=12)

        axes[row_idx][0].set_ylabel(row_label, fontsize=32,
                                     labelpad=18, color=color,
                                     fontweight="bold")

    fig.suptitle(f"Molecules with largest PM7 correction errors  [{tag.upper()}]",
                 fontsize=TITLE_SIZE, y=1.05, fontweight="bold")

    fig.tight_layout(pad=2.0, w_pad=2.0, h_pad=3.0)
    savefig(fig, f"worst_best_molecules_{tag}")


def plot_worst_best_combined(nist_df: pd.DataFrame, km_df: pd.DataFrame):
    """
    Combined 4-row x 6-col figure: NIST (a, top two rows) +
    k-means (b, bottom two rows).
    Panel mapping:
      (a) rows 0-1: NIST molecules
      (b) rows 2-3: k-means molecules
    Color convention:
      Blue border  -> largest positive correction (PM7 underestimates)
      Pink border  -> largest negative correction (PM7 overestimates)
    """
    log.info("Plot combined: Worst/best molecules — NIST (a) + k-means (b)")

    smiles_col_nist = get_smiles_col(nist_df)
    smiles_col_km   = get_smiles_col(km_df)

    nist_worst = nist_df.nlargest(6, "correction_kcal")[
        [smiles_col_nist, "correction_kcal", "ref_pa_kcalmol", "chem_class"]]
    nist_best = nist_df.nsmallest(6, "correction_kcal")[
        [smiles_col_nist, "correction_kcal", "ref_pa_kcalmol", "chem_class"]]
    km_worst = km_df.nlargest(6, "correction_kcal")[
        [smiles_col_km, "correction_kcal", "ref_pa_kcalmol", "chem_class"]]
    km_best = km_df.nsmallest(6, "correction_kcal")[
        [smiles_col_km, "correction_kcal", "ref_pa_kcalmol", "chem_class"]]

    # Font sizes for the combined figure — very large so text remains
    # readable when the figure is scaled down in a LaTeX manuscript
    COMB_XLABEL = 40 #36
    COMB_PANEL  = 46

    fig, axes = plt.subplots(4, 6, figsize=(30, 24))

    # Colour convention: blue border = largest +ve, pink = largest −ve
    POS_COLOR = "#2166AC"
    NEG_COLOR = "#D01C8B"

    def _draw_row(row_axes, subset, smiles_col, border_color):
        for col_idx, (_, row) in enumerate(subset.reset_index().iterrows()):
            ax = row_axes[col_idx]
            img = mol_to_image_dark(row[smiles_col], size=(1200, 840))
            if img:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(3.5)
                spine.set_edgecolor(border_color)
                spine.set_visible(True)
            ax.set_xlabel(
                f"$\\bf{{{row['correction_kcal']:+.1f}}}$ kcal/mol\n"
                f"PA={row['ref_pa_kcalmol']:.1f}\n[{row['chem_class']}]",
                fontsize=COMB_XLABEL, labelpad=12)

    # ── NIST rows (0, 1) ─────────────────────────────────────────────────
    _draw_row(axes[0], nist_worst, smiles_col_nist, POS_COLOR)
    _draw_row(axes[1], nist_best,  smiles_col_nist, NEG_COLOR)

    # ── k-means rows (2, 3) ──────────────────────────────────────────────
    _draw_row(axes[2], km_worst, smiles_col_km, POS_COLOR)
    _draw_row(axes[3], km_best,  smiles_col_km, NEG_COLOR)

    # ── Panel labels (a) and (b) ─────────────────────────────────────────
    axes[0][0].text(-0.20, 1.26, "(a)", transform=axes[0][0].transAxes,
                    fontsize=COMB_PANEL, fontweight="bold", va="top")
    axes[2][0].text(-0.20, 1.26, "(b)", transform=axes[2][0].transAxes,
                    fontsize=COMB_PANEL, fontweight="bold", va="top")

    fig.tight_layout(pad=2.0, w_pad=2.4, h_pad=4.0)
    savefig(fig, "worst_best_molecules_combined")


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
    fig, ax  = plt.subplots(figsize=(18, 8))
    ax.axis("off")

    table = ax.table(
        cellText=df_table.values.tolist(),
        colLabels=list(df_table.columns),
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(ANNOT_SIZE)
    table.scale(1, 2.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.2)
        cell.set_facecolor("white")
        if row == 0:
            cell.set_text_props(color="black", fontweight="bold", fontsize=LEGEND_SIZE)
        else:
            cell.set_text_props(color="black", fontsize=ANNOT_SIZE)

    fig.tight_layout()
    savefig(fig, f"correction_summary_table_{tag}")


# ── Per-dataset runner ────────────────────────────────────────────────────────

def run_dataset(dataset: str, class_order=None):
    log.info("=" * 60)
    log.info(f"  CHEMICAL CLASS ANALYSIS — {dataset.upper()}")
    log.info("=" * 60)

    if dataset == "nist":
        df        = load_nist_with_corrections()
        ref_label = "exp"
    else:
        df        = load_kmeans_with_corrections()
        ref_label = "DFT"

    df = assign_chemical_classes(df)

    log.info("Class distribution:")
    for cls, n in df["chem_class"].value_counts().items():
        log.info(f"  {cls:<20}: {n:4d} ({100*n/len(df):.1f}%)")

    tag = dataset
    plot_representative_molecules(df, tag)
    plot_correction_vs_pa(df, tag, ref_label)
    plot_correction_summary_table(df, tag)

    log.info(f"  All {dataset.upper()} figures saved to figures/")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Chemical class analysis of PM7 corrections.")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "nist", "kmeans"])
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    nist_df = km_df = None

    if args.dataset == "all":
        # Generate the combined correction-by-class figure first.
        # Panel mapping in the combined figure: (a) = NIST, (b) = k-means.
        nist_df = assign_chemical_classes(load_nist_with_corrections())
        km_df = assign_chemical_classes(load_kmeans_with_corrections())
        plot_correction_combined(nist_df, km_df, class_order=None)
        plot_representative_molecules_combined(nist_df, km_df)
        plot_correction_vs_pa(nist_df, "nist", "exp")
        plot_correction_summary_table(nist_df, "nist")
        plot_correction_vs_pa(km_df, "kmeans", "DFT")
        plot_correction_summary_table(km_df, "kmeans")
        plot_worst_best_combined(nist_df, km_df)
    else:
        if args.dataset == "nist":
            nist_df = run_dataset("nist")
        if args.dataset == "kmeans":
            km_df = run_dataset("kmeans")

    print("\n  All figures saved to: figures/")
    print("  Suffixes: _nist  and  _kmeans")


if __name__ == "__main__":
    main()
