"""
plot_si_candidates.py
=====================
Renders all 30 DFT-validated Pareto-selected candidates as a grid figure
for the Supporting Information. Molecules are ranked 1-30 by PA_DFT
(descending). Mol-1 through Mol-5 correspond to the five priority leads.

Output: figures/si_all30_candidates.pdf

Usage:
    cd <repo_root>/screening
    python scripts/plotting/plot_si_candidates.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from io import BytesIO
from PIL import Image

# Directory setup relative to the script's location in screening/scripts/plotting/
SCRIPT_DIR   = Path(__file__).parent          # screening/scripts/plotting/
SCREENING_DIR = SCRIPT_DIR.parent.parent       # screening/
PROJECT_DIR  = SCREENING_DIR.parent           # proton-affinity-paper/
DATA_DIR     = PROJECT_DIR / "data" / "screening" / "iter1"
FIG_DIR      = PROJECT_DIR / "figures"

DPI = 600

# Journal-style typography
TICK_SIZE = 9
LABEL_SIZE = 10
TITLE_SIZE = 10
LEGEND_SIZE = 9
SUPTITLE_SIZE = 11
SPINE_LW = 1.0
DOUBLE_COL_W = 7.0

# Top 5 SMILES in order (to assign Mol-1 through Mol-5 labels)
TOP5_SMILES = [
    "CCc1n[nH]c(-c2ccsc2)c1N",    # Mol-1
    "OCc1cc(-c2cccs2)n[nH]1",     # Mol-2
    "Nc1ccn[nH]1",                # Mol-3
    "Oc1ccc(-c2ccn[nH]2)cc1",     # Mol-4
    "Cc1n[nH]c(C)c1-c1ccco1",     # Mol-5
]

TOP5_NAMES = {
    "CCc1n[nH]c(-c2ccsc2)c1N":  "Mol-1",
    "OCc1cc(-c2cccs2)n[nH]1":   "Mol-2",
    "Nc1ccn[nH]1":               "Mol-3",
    "Oc1ccc(-c2ccn[nH]2)cc1":   "Mol-4",
    "Cc1n[nH]c(C)c1-c1ccco1":   "Mol-5",
}

COLOR_TOP5     = "#2166AC"   # blue border for top 5
COLOR_INWINDOW = "#4DAC26"   # green border for in-window non-top-5
COLOR_OUTWINDOW = "#D73027"  # red border for out-of-window


def mol_to_image(smiles: str, size=(1200, 840)):
    """Two-pass RDKit render (adapted from plot_chemical_analysis.mol_to_image_dark).

    A probe render measures the drawn content extent, then a final render
    picks ``fixedBondLength`` so every molecule fills its panel with a uniform
    safety margin. Bond widths and atom font sizes stay fixed in pixels so
    appearance is uniform across all 30 panels.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        tw, th = int(size[0]), int(size[1])
        margin = 0.92

        def _render(bond_length: float):
            drawer = rdMolDraw2D.MolDraw2DCairo(tw, th)
            opts = drawer.drawOptions()
            opts.addStereoAnnotation = False
            opts.padding = 0.02
            opts.bondLineWidth = 10
            opts.minFontSize = 48
            opts.maxFontSize = 66
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


def wrap_smiles_two_lines(smiles: str, max_len: int = 18) -> str:
    """Force SMILES onto exactly two lines so xlabel heights stay uniform.

    Short strings get a trailing blank line; long strings are split near a
    natural separator close to the midpoint, falling back to midpoint split.
    """
    if len(smiles) <= max_len:
        return smiles + "\n "
    seps = ("_", "-", "(", ")", ".", "/")
    mid = len(smiles) // 2
    best = None
    for i, ch in enumerate(smiles):
        if ch in seps and (best is None or abs(i - mid) < abs(best - mid)):
            best = i
    if best is not None and 0 < best < len(smiles) - 1:
        return smiles[:best + 1] + "\n" + smiles[best + 1:]
    return smiles[:mid] + "\n" + smiles[mid:]


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": TICK_SIZE,
        "axes.linewidth": SPINE_LW,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.20,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    # Load DFT results
    df = pd.read_csv(DATA_DIR / "dft_files_summary.csv")
    best = (df[df["is_best_site"]]
            .sort_values("pa_best_kcal", ascending=False)
            .reset_index(drop=True))

    print(f"Loaded {len(best)} candidates")
    print(f"PA range: {best['pa_best_kcal'].min():.1f} - {best['pa_best_kcal'].max():.1f} kcal/mol")

    # Load SA scores from LLM verdicts or pareto files
    sa_scores = {}
    try:
        verdicts = pd.read_parquet(DATA_DIR / "llm_verdicts.parquet")
        if "sa_score" in verdicts.columns:
            sa_lookup = verdicts.set_index("smiles")["sa_score"].to_dict()
            sa_scores = sa_lookup
            print(f"Loaded SA scores for {len(sa_scores)} molecules")
    except Exception as e:
        print(f"Could not load SA scores: {e}")

    # Also load full pareto data for SA and uncertainty if available
    try:
        pareto_df = pd.read_parquet(DATA_DIR / "pareto_selected.parquet")
        has_pareto = True
        print("Loaded pareto data with SA and uncertainty")
    except FileNotFoundError:
        has_pareto = False
        print("No pareto parquet found — showing PA only")

    # Layout: 5 columns x 6 rows = 30 molecules (journal-width canvas)
    ncols = 5
    nrows = 6
    FALSE_POSITIVES = {9, 23, 30}

    fig = plt.figure(figsize=(DOUBLE_COL_W, 8.6), dpi=DPI)
    gs = fig.add_gridspec(
        nrows, ncols,
        left=0.045, right=0.955,
        top=0.985, bottom=0.120,
        hspace=0.35, wspace=0.32,
    )

    for idx, row in best.iterrows():
        rank = idx + 1  # 1-indexed
        smiles = row["smiles"]
        pa_dft = row["pa_best_kcal"]
        in_window = 210 <= pa_dft <= 235
        is_top5 = smiles in TOP5_SMILES
        mol_label = TOP5_NAMES.get(smiles, "")

        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])

        # Draw molecule with uniform scale
        img = mol_to_image(smiles)
        if img is not None:
            ax.imshow(img, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Border color / weight
        if is_top5:
            border_color = COLOR_TOP5
            border_lw = 1.4
        elif in_window:
            border_color = COLOR_INWINDOW
            border_lw = 1.0
        else:
            border_color = COLOR_OUTWINDOW
            border_lw = 1.0
        for spine in ax.spines.values():
            spine.set_linewidth(border_lw)
            spine.set_edgecolor(border_color)

        # Title: rank + optional Mol label (compact padding for row alignment)
        title = f"#{rank}"
        if mol_label:
            title += f"  ({mol_label})"
        title_color = COLOR_TOP5 if is_top5 else "black"
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold",
                     color=title_color, pad=3)

        # Mark false positives
        if rank in FALSE_POSITIVES:
            ax.text(0.97, 0.97, "†",
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=TITLE_SIZE + 1,
                    color=COLOR_OUTWINDOW,
                    fontweight="bold")

        sa_val = sa_scores.get(smiles, None)
        if sa_val is not None and not np.isnan(sa_val):
            stats_line = f"PA {pa_dft:.1f}   SA {sa_val:.2f}"
        else:
            stats_line = f"PA {pa_dft:.1f}"

        ax.set_xlabel(
            stats_line,
            fontsize=LABEL_SIZE - 1,
            fontweight="bold",
            labelpad=3,
            color="black",
        )

    dagger_handle = Line2D(
        [0], [0], marker=r"$\dagger$", color="none",
        markerfacecolor=COLOR_OUTWINDOW, markeredgecolor=COLOR_OUTWINDOW,
        markersize=10, linestyle="None",
        label="False positive",
    )
    legend_elements = [
        mpatches.Patch(facecolor="white", edgecolor=COLOR_TOP5,
                       linewidth=3, label="Top 5 priority leads (Mol-1 to Mol-5)"),
        mpatches.Patch(facecolor="white", edgecolor=COLOR_INWINDOW,
                       linewidth=2.5, label="In target PA window (210–235 kcal/mol)"),
        mpatches.Patch(facecolor="white", edgecolor=COLOR_OUTWINDOW,
                       linewidth=2.5, label="Outside target PA window"),
        dagger_handle,
    ]
    leg = fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        fontsize=LEGEND_SIZE,
        framealpha=0.95,
        edgecolor="black",
        handletextpad=0.6,
        columnspacing=2.2,
    )
    for text in leg.get_texts():
        text.set_fontweight("bold")

    # Save
    out_dir = FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "si_all30_candidates.pdf"
    out_png  = out_dir / "si_all30_candidates.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.20, dpi=DPI)
    fig.savefig(out_png,  bbox_inches="tight", pad_inches=0.20, dpi=200)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {out_png}")
    plt.close(fig)

    # Also print a summary table
    print("\nFull candidate table:")
    print(f"{'Rank':<5} {'Mol':>6} {'PA_DFT':>10} {'Window':>10}  SMILES")
    print("-" * 80)
    for idx, row in best.iterrows():
        rank = idx + 1
        smiles = row["smiles"]
        pa_dft = row["pa_best_kcal"]
        in_window = "YES" if 210 <= pa_dft <= 235 else "NO"
        mol_label = TOP5_NAMES.get(smiles, "---")
        print(f"{rank:<5} {mol_label:>6} {pa_dft:>10.2f} {in_window:>10}  {smiles}")


if __name__ == "__main__":
    main()