"""Compose the corrected prospective parity plot with the five priority leads.

The layout intentionally follows the submitted manuscript artwork: the parity
panel occupies the left half, while Mol-1 through Mol-5 are shown in bordered
structure panels on the right in a 2 + 2 + 1 arrangement. Marker shapes match
the highlighted points in the parity panel.
"""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D


RDLogger.DisableLog("rdApp.*")

SCRIPT_DIR = Path(__file__).resolve().parent
SCREENING = SCRIPT_DIR.parent.parent
PROJECT = SCREENING.parent
EXECUTION = SCREENING / "scripts" / "execution"
sys.path.insert(0, str(EXECUTION))

from pipeline_config import PATHS  # noqa: E402

CONFIG = json.loads((SCREENING / "config" / "pipeline_config.json").read_text())

PA_LOW = float(CONFIG["pa_window_kcalmol"]["low"])
PA_HIGH = float(CONFIG["pa_window_kcalmol"]["high"])
TOP5_MARKERS = ["D", "s", "^", "P", "X"]


def molecule_image(smiles: str, size: tuple[int, int] = (1200, 850)) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.padding = 0.02
    opts.addStereoAnnotation = False
    # Give the compact lead structures the same strong visual weight used for
    # the molecule renderings in Fig. 7, moderated for these smaller panels.
    opts.bondLineWidth = 10.0
    opts.fixedBondLength = 62
    opts.minFontSize = 52
    opts.maxFontSize = 72
    opts.multipleBondOffset = 0.18
    opts.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    image = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(1.45)
    image = ImageEnhance.Sharpness(image).enhance(1.35)

    array = np.asarray(image)
    nonwhite = np.any(array < 246, axis=2)
    if np.any(nonwhite):
        ys, xs = np.where(nonwhite)
        pad = 28
        image = image.crop((
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(array.shape[1], int(xs.max()) + pad + 1),
            min(array.shape[0], int(ys.max()) + pad + 1),
        ))
    return image


def draw_structure_panel(
    fig: plt.Figure,
    rect: list[float],
    smiles: str,
    label: str,
    marker: str,
    label_corner: str,
) -> None:
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.3)

    # Reserve the label corner without mapping the molecular image onto a
    # fixed data-space rectangle. The inset axis preserves the RDKit image's
    # native width-to-height ratio, preventing bonds and rings from stretching.
    image_boxes = {
        "top-left": (0.17, 0.06, 0.78, 0.76),
        "top-right": (0.05, 0.06, 0.78, 0.76),
        # Keep a dedicated lower label band; inset axes otherwise paint over
        # the upper portion of labels such as "Mol 5" during composition.
        "bottom-left": (0.15, 0.27, 0.80, 0.68),
        "bottom-right": (0.05, 0.27, 0.80, 0.68),
    }
    image_ax = ax.inset_axes(image_boxes[label_corner])
    image_ax.imshow(molecule_image(smiles), aspect="equal")
    image_ax.set_axis_off()

    corner_positions = {
        "top-left": (0.04, 0.92, "left", "top"),
        "top-right": (0.96, 0.92, "right", "top"),
        "bottom-left": (0.04, 0.07, "left", "bottom"),
        "bottom-right": (0.96, 0.07, "right", "bottom"),
    }
    x, y, ha, va = corner_positions[label_corner]
    ax.text(
        x, y, label.replace("-", " "), transform=ax.transAxes,
        ha=ha, va=va, fontsize=21, fontweight="bold", color="#8a8a8a",
        zorder=5,
    )
    ax.scatter(
        0.5, 1.0, transform=ax.transAxes, marker=marker, s=150,
        facecolor="white", edgecolor="black", linewidth=1.4,
        clip_on=False, zorder=8,
    )


def compose(iteration: int, output: Path, dft_dir: Path | None = None) -> None:
    archive_dir = PATHS.dft_validation_dir(
        iteration,
        explicit=dft_dir,
        required_files=("lead_selection_analysis.csv", "top5_leads.csv"),
    )
    analysis = pd.read_csv(archive_dir / "lead_selection_analysis.csv")
    leads = pd.read_csv(archive_dir / "top5_leads.csv")
    if len(leads) != 5:
        raise ValueError(f"Expected five priority leads, found {len(leads)}")

    analysis = analysis.rename(columns={"pa_dft_kcalmol": "pa_best_kcal"})
    analysis["delta_pred_vs_dft"] = (
        analysis["pa_pred_kcalmol"] - analysis["pa_best_kcal"]
    )
    plot_df = analysis.dropna(subset=["pa_pred_kcalmol", "pa_best_kcal"]).copy()
    mae = plot_df["delta_pred_vs_dft"].abs().mean()
    bias = plot_df["delta_pred_vs_dft"].mean()
    dispersion_vmin = float(np.floor(plot_df["uncertainty"].min()))
    dispersion_vmax = float(np.ceil(plot_df["uncertainty"].max()))
    if dispersion_vmax <= dispersion_vmin:
        dispersion_vmax = dispersion_vmin + 1.0
    lo = plot_df["pa_best_kcal"].min() - 4
    hi = plot_df["pa_best_kcal"].max() + 4
    lims = (lo, hi)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(14.2, 7.2), facecolor="white")
    ax = fig.add_axes([0.065, 0.105, 0.415, 0.80])

    ax.plot(lims, lims, color="black", ls="--", lw=2.0, alpha=0.6)
    ax.plot(lims, [v - bias for v in lims], color="#e07020", ls="-.", lw=2.0)
    ax.axhspan(PA_LOW, PA_HIGH, color="#55aa55", alpha=0.07)
    ax.set_axisbelow(True)
    ax.grid(
        True, which="major", linestyle=":", linewidth=0.7,
        color="#aaaaaa", alpha=0.65,
    )
    ax.text(lo + 0.5, PA_HIGH + 0.5, "Target window", fontsize=15,
            color="#338833", fontweight="bold")
    ax.text(lo + 1.8, lo + 1.8, "y=x", fontsize=12, rotation=45,
            rotation_mode="anchor", ha="left", va="bottom")
    x_bias = lo + 5.0
    ax.text(x_bias, x_bias - bias + 1.0, "bias line", fontsize=12,
            color="#e07020", rotation=45, rotation_mode="anchor",
            ha="left", va="bottom")

    top5_smiles = leads["smiles"].tolist()
    ordinary = ~plot_df["smiles"].isin(top5_smiles)
    scatter = ax.scatter(
        plot_df.loc[ordinary, "pa_pred_kcalmol"],
        plot_df.loc[ordinary, "pa_best_kcal"],
        c=plot_df.loc[ordinary, "uncertainty"], cmap="YlOrRd",
        vmin=dispersion_vmin, vmax=dispersion_vmax,
        s=56, edgecolors="#333333", linewidths=1.1,
        alpha=0.85, zorder=3,
    )
    for smiles, marker in zip(top5_smiles, TOP5_MARKERS):
        row = plot_df.loc[plot_df["smiles"] == smiles]
        if row.empty:
            raise ValueError(f"Priority lead absent from DFT table: {smiles}")
        ax.scatter(
            row["pa_pred_kcalmol"], row["pa_best_kcal"],
            c=row["uncertainty"], cmap="YlOrRd",
            vmin=dispersion_vmin, vmax=dispersion_vmax,
            s=125, marker=marker, edgecolors="#111111", linewidths=1.3,
            zorder=6,
        )

    stats = f"MAE = {mae:.2f}\nBias = {bias:+.2f}\n$n$ = {len(plot_df)}"
    ax.text(
        0.05, 0.80, stats, transform=ax.transAxes, fontsize=14, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92,
                  edgecolor="#aaaaaa", linewidth=1.2),
    )
    ax.set_xlabel(
        "PA$_{\\mathrm{pred}}$ = PA$_{\\mathrm{PM7}}$ + "
        "$\\Delta_{\\mathrm{ML}}$\n(kcal/mol)", fontsize=16,
    )
    ax.set_ylabel("PA$_{\\mathrm{DFT}}$\n(kcal/mol)", fontsize=16)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=14, width=1.2, length=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)

    cax = fig.add_axes([0.495, 0.225, 0.017, 0.59])
    colorbar = fig.colorbar(scatter, cax=cax)
    colorbar.set_label("RF ensemble dispersion\n(kcal/mol)", fontsize=15)
    colorbar.set_ticks(
        np.arange(dispersion_vmin, dispersion_vmax + 0.5, 1.0)
    )
    colorbar.ax.tick_params(labelsize=13, width=1.2)
    colorbar.outline.set_linewidth(1.2)

    panel_rects = [
        [0.585, 0.615, 0.175, 0.255],
        [0.790, 0.615, 0.175, 0.255],
        [0.585, 0.320, 0.175, 0.255],
        [0.790, 0.320, 0.175, 0.255],
        [0.6875, 0.025, 0.175, 0.255],
    ]
    label_corners = [
        "bottom-right", "top-right", "bottom-left", "top-left", "bottom-left"
    ]
    for rect, row, marker, corner in zip(
        panel_rects, leads.itertuples(index=False), TOP5_MARKERS, label_corners
    ):
        draw_structure_panel(
            fig, rect, row.smiles, row.lead_label, marker, corner
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--dft-dir", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "figures" / "screening" / "iter1_pa_parity_with_leads.pdf",
    )
    args = parser.parse_args()
    compose(args.iter, args.output, dft_dir=args.dft_dir)
