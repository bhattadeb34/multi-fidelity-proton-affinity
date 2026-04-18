"""
plot_pa_parity_final.py
=======================
Regenerate the DFT validation parity plot from pre-computed data.

Reads from:
    data/screening/iter{N}/dft_files_summary.csv

Outputs:
    screening/figures/iter{N}_pa_parity_final.pdf

Usage:
    python screening/scripts/plotting/plot_pa_parity_final.py
    python screening/scripts/plotting/plot_pa_parity_final.py --iter 1
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageEnhance
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
SCREENING  = SCRIPT_DIR.parent.parent
PROJECT    = SCREENING.parent
DATA_DIR   = PROJECT / "data" / "screening"

PA_LOW, PA_HIGH = 210.0, 235.0

# Journal-style defaults
DOUBLE_COL_W = 7.0
J_TICK = 9
J_LABEL = 10
J_TITLE = 10
J_LEGEND = 9
J_SPINE = 1.0

TOP5_SMILES  = ['CCc1n[nH]c(-c2ccsc2)c1N', 'OCc1cc(-c2cccs2)n[nH]1',
                'Nc1ccn[nH]1', 'Oc1ccc(-c2ccn[nH]2)cc1', 'Cc1n[nH]c(C)c1-c1ccco1']
TOP5_LABELS  = ['Mol-1', 'Mol-2', 'Mol-3', 'Mol-4', 'Mol-5']
TOP5_MARKERS = ['D', 's', '^', 'P', 'X']


def mol_to_image(smiles: str, size=(1400, 1000), crop_pad: int = 24, trim: bool = False):
    """Render a high-contrast molecule image from SMILES using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.padding = 0.03
    opts.addStereoAnnotation = False
    opts.bondLineWidth = 3.0
    opts.fixedBondLength = 58
    opts.minFontSize = 22
    opts.maxFontSize = 38
    opts.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")
    # Boost visibility for print/PDF rendering.
    img = ImageEnhance.Contrast(img).enhance(1.45)
    img = ImageEnhance.Sharpness(img).enhance(1.35)
    # Optional trim step. Keep disabled for consistent panel scaling.
    if trim:
        rgb = np.array(img.convert("RGB"))
        nonwhite = np.any(rgb < 245, axis=2)
        if np.any(nonwhite):
            ys, xs = np.where(nonwhite)
            pad = crop_pad
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(rgb.shape[1], int(xs.max()) + pad + 1)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(rgb.shape[0], int(ys.max()) + pad + 1)
            img = img.crop((x0, y0, x1, y1))
    return img


def save_top5_marker_molecule_panels(fig_dir: Path, iteration: int) -> None:
    """Save Mol-1..Mol-5 as individual, consistently styled panels."""
    out_dir = fig_dir / f"iter{iteration}_top5_molecule_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    for smi, label, marker in zip(TOP5_SMILES, TOP5_LABELS, TOP5_MARKERS):
        fig, ax = plt.subplots(figsize=(2.2, 2.4))
        # Fixed axes box with extra top whitespace so box and title are easy to crop.
        ax.set_position([0.12, 0.14, 0.76, 0.62])
        img = mol_to_image(smi, crop_pad=24, trim=False)
        if img is not None:
            ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.5)

        # Marker centered on the top border midpoint of each black box.
        ax.scatter(
            0.5, 1.0, transform=ax.transAxes, marker=marker, s=70,
            facecolor="white", edgecolor="black", linewidth=1.0,
            clip_on=False, zorder=6
        )
        ax.set_title(label, fontsize=J_LABEL, pad=12)
        out_path = out_dir / f"{label.lower().replace('-', '_')}.pdf"
        fig.savefig(out_path, bbox_inches=None, pad_inches=0.0)
        plt.close(fig)
        log.info(f"Saved {out_path}")


def main(iteration: int) -> None:
    iter_dir = DATA_DIR / f"iter{iteration}"
    summary_csv = iter_dir / "dft_files_summary.csv"
    fig_dir = SCREENING / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not summary_csv.exists():
        log.error(f"Not found: {summary_csv}")
        return

    best_df = pd.read_csv(summary_csv)
    if "delta_pred_vs_dft" not in best_df.columns:
        best_df["delta_pred_vs_dft"] = best_df["pa_pred_kcalmol"] - best_df["pa_best_kcal"]

    mae_val  = best_df["delta_pred_vs_dft"].abs().mean()
    bias_val = best_df["delta_pred_vs_dft"].mean()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": J_SPINE,
        "xtick.labelsize": J_TICK,
        "ytick.labelsize": J_TICK,
        "axes.labelsize": J_LABEL,
        "axes.titlesize": J_TITLE,
        "legend.fontsize": J_LEGEND,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    plot_df = best_df.dropna(subset=["pa_pred_kcalmol", "pa_best_kcal"]).copy()
    lo = plot_df["pa_best_kcal"].min() - 4
    hi = plot_df["pa_best_kcal"].max() + 4
    lims = [lo, hi]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 3.4))

    ax.plot(lims, lims, color="black", ls="--", lw=1.5, alpha=0.6, label="$y=x$")
    ax.plot(lims, [l - bias_val for l in lims], color="#e07020", lw=1.5,
            ls="-.", label="Systematic Bias")
    ax.axhspan(PA_LOW, PA_HIGH, color="#55aa55", alpha=0.07, label="Target Window")
    ax.text(lo + 0.5, PA_HIGH + 0.5, "Target window",
            fontsize=J_LABEL, color="#338833", fontweight="bold")
    x_yx = lo + 1.8
    y_yx = x_yx
    ax.text(
        x_yx, y_yx, "y=x",
        fontsize=J_LEGEND, color="black", rotation=45,
        rotation_mode="anchor", ha="left", va="bottom"
    )
    x_bias = lo + 5.0
    y_bias = x_bias - bias_val + 1.0
    ax.text(
        x_bias, y_bias, "bias line",
        fontsize=J_LEGEND, color="#e07020", rotation=45,
        rotation_mode="anchor", ha="left", va="bottom"
    )

    mask_top5 = plot_df["smiles"].isin(TOP5_SMILES)
    sc = ax.scatter(
        plot_df.loc[~mask_top5, "pa_pred_kcalmol"],
        plot_df.loc[~mask_top5, "pa_best_kcal"],
        c=plot_df.loc[~mask_top5, "uncertainty"],
        cmap="YlOrRd", vmin=4, vmax=20, s=24,
        edgecolors="#333333", alpha=0.85, zorder=3,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label("ML Uncertainty\n(kcal/mol)", fontsize=J_LABEL)
    cbar.ax.tick_params(labelsize=J_TICK)

    for smi, marker in zip(TOP5_SMILES, TOP5_MARKERS):
        row = plot_df[plot_df["smiles"] == smi]
        if not row.empty:
            ax.scatter(
                row["pa_pred_kcalmol"], row["pa_best_kcal"],
                c=row["uncertainty"], cmap="YlOrRd", vmin=4, vmax=20,
                s=52, marker=marker, edgecolors="#111111",
                linewidths=0.8, zorder=5,
            )

    stats = f"MAE = {mae_val:.2f}\nBias = {bias_val:+.2f}\n$n$ = {len(plot_df)}"
    ax.text(0.05, 0.95, stats, transform=ax.transAxes, fontsize=J_LEGEND, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#aaaaaa"))

    ax.set_xlabel(
        "PA$_{\\mathrm{pred}}$ = PA$_{\\mathrm{PM7}}$ + "
        "$\\Delta_{\\mathrm{ML}}$\n(kcal/mol)", fontsize=J_LABEL)
    ax.set_ylabel("PA$_{\\mathrm{DFT}}$\n(kcal/mol)", fontsize=J_LABEL)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    # cand_handles = [
    #     mlines.Line2D([], [], color="#888888", marker=m, ls="None",
    #                   markersize=7, markeredgecolor="k", label=lbl)
    #     for lbl, m in zip(TOP5_LABELS, TOP5_MARKERS)
    # ]
    # Keep only Mol-1 ... Mol-5 marker legend (no reference legend box).
    # ax.legend(
    #     handles=cand_handles,
    #     loc="upper left", bbox_to_anchor=(1.42, 1.0),
    #     fontsize=J_LEGEND, title="Top-5 markers", title_fontsize=J_LABEL,
    #     frameon=False, borderaxespad=0.0,
    # )

    fig_path = fig_dir / f"iter{iteration}_pa_parity_final.pdf"
    fig.savefig(fig_path)
    plt.close()
    log.info(f"Saved {fig_path}")
    save_top5_marker_molecule_panels(fig_dir=fig_dir, iteration=iteration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot DFT validation parity from saved results.")
    parser.add_argument("--iter", type=int, default=1)
    args = parser.parse_args()
    main(iteration=args.iter)
