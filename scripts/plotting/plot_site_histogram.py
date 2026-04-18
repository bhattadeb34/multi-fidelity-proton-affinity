"""
plot_site_histogram.py
Generates a histogram of the number of protonation sites per molecule
for the k-means dataset (for the Supplementary Information).
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add scripts directory to path so we can import your existing loader
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
sys.path.append(str(SCRIPT_DIR))

from plot_chemical_analysis import load_kmeans_with_corrections

# Journal-style typography (matches other paper figures)
TICK_SIZE   = 9
LABEL_SIZE  = 10
ANNOT_SIZE  = 8
SPINE_LW    = 1.0
SINGLE_COL_W = 3.6   # inches


def main():
    print("Generating site distribution histogram...")

    df = load_kmeans_with_corrections()
    smiles_col = "neutral_smiles" if "neutral_smiles" in df.columns else df.columns[0]
    sites_per_molecule = df.groupby(smiles_col).size()
    counts = sites_per_molecule.value_counts().sort_index()

    min_sites = int(counts.index.min())
    max_sites = int(counts.index.max())

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": TICK_SIZE,
        "axes.linewidth": SPINE_LW,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": TICK_SIZE,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.4))

    bars = ax.bar(counts.index, counts.values,
                  color="#2166AC", edgecolor="white",
                  width=0.7, alpha=0.9, linewidth=0.6)

    ax.set_xlabel("N/O protonation sites per molecule")
    ax.set_ylabel("Number of molecules")

    ax.set_xticks(range(min_sites, max_sites + 1))
    ax.set_ylim(0, max(counts.values) * 1.18)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                height + (max(counts.values) * 0.02),
                f"{int(height)}", ha="center", va="bottom",
                fontsize=ANNOT_SIZE, fontweight="bold")

    ax.tick_params(axis="both", which="major",
                   length=3, width=SPINE_LW, pad=2)
    ax.grid(axis="y", linewidth=0.5, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)

    out_dir = PROJECT_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "si_site_distribution_kmeans.pdf"
    png_path = out_dir / "si_site_distribution_kmeans.png"

    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}  (range {min_sites}–{max_sites} sites)")

if __name__ == "__main__":
    main()


