"""
08_plot_results.py
==================
Generate publication-quality figures for the screening section of the paper.
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
SCREENING  = SCRIPT_DIR.parent.parent
PROJECT    = SCREENING.parent
DATA_BASE  = next((p for p in (PROJECT / "data", PROJECT.parent / "data") if p.exists()), PROJECT / "data")
DATA_ROOT  = DATA_BASE / "screening"
FIG_DIR    = PROJECT / "figures" / "screening"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "pool":     "#4878CF",
    "selected": "#D65F5F",
    "window":   "#6ACC65",
    "grey":     "#AAAAAA",
    "dark":     "#222222",
}
CONFIG = json.loads((SCREENING / "config" / "pipeline_config.json").read_text())
PA_LOW = float(CONFIG["pa_window_kcalmol"]["low"])
PA_HIGH = float(CONFIG["pa_window_kcalmol"]["high"])

# Journal spec targets
SINGLE_COL_W = 3.4
DOUBLE_COL_W = 7.0
J_TICK = 9
J_LABEL = 10
J_LEGEND = 9
J_TITLE = 10
J_SPINE = 1.0

def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         J_TICK,
        "axes.labelsize":    J_LABEL,
        "axes.titlesize":    J_TITLE,
        "axes.linewidth":    J_SPINE,
        "xtick.labelsize":   J_TICK,
        "ytick.labelsize":   J_TICK,
        "legend.fontsize":   J_LEGEND,
        "figure.dpi":        150,
        "savefig.dpi":       600,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.04,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })
    return plt

def plot_funnel(summary, plt):
    n_candidates = summary["retrieved_candidates"]
    n_pm7 = summary["pm7_successful_molecules"]
    n_predicted = summary["predicted_molecules"]
    n_predicted_sites = summary["predicted_sites"]
    n_in_window = summary["predicted_pa_window_molecules"]
    n_llm_accept = summary["llm_eligible_in_pa_window"]
    n_pareto = summary["pareto_selected"]

    labels = [
        f"ZINC library\n({summary['filtered_library_size']:,} filtered)",
        f"Similarity search\n({n_candidates:,} candidates)",
        f"PM7 calculation\n({n_pm7:,} molecules)",
        f"ML prediction\n({n_predicted:,} molecules; {n_predicted_sites:,} sites)",
        f"PA window {PA_LOW:g}–{PA_HIGH:g}\n({n_in_window:,} molecules)",
        f"LLM eligible in window\n({n_llm_accept:,} accept/flag)",
        f"Pareto selected\n({n_pareto} for DFT)",
    ]
    values = [summary["filtered_library_size"], n_candidates, n_pm7,
              n_predicted, n_in_window, n_llm_accept, n_pareto]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 3.0))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=COLORS["pool"], alpha=0.8, height=0.6)
    bars[-1].set_color(COLORS["selected"])

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", ha="left", fontsize=J_TICK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=J_TICK)
    ax.set_xscale("log")
    ax.set_xlim(1, 3e6)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "screening_funnel.pdf")
    plt.close()

def plot_pa_distribution(mol_pa, selected, plt):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 3.0))

    # Panel 1: PM7
    ax = axes[0]
    ax.hist(mol_pa["pa_pm7_kcalmol"], bins=50, color=COLORS["pool"],
            alpha=0.7, label="All candidates", density=True)
    ax.axvspan(PA_LOW, PA_HIGH, color=COLORS["window"], alpha=0.15,
               label="Target window")
    ax.axvline(PA_LOW, color=COLORS["window"], lw=2.0, ls="--")
    ax.axvline(PA_HIGH, color=COLORS["window"], lw=2.0, ls="--")
    for smi in selected["smiles"]:
        row = mol_pa[mol_pa["smiles"] == smi]
        if len(row):
            ax.axvline(row.iloc[0]["pa_pm7_kcalmol"],
                       color=COLORS["selected"], alpha=0.4, lw=0.8)
    ax.set_xlabel("PA$_\\mathrm{PM7}$ (kcal/mol)", fontsize=J_LABEL)
    ax.set_ylabel("Density", fontsize=J_LABEL)
    ax.set_title("PM7-Calculated PA", fontsize=J_TITLE, fontweight="bold")
    ax.tick_params(labelsize=J_TICK)
    ax.legend(frameon=False, fontsize=J_LEGEND)

    # Panel 2: ML-corrected
    ax = axes[1]
    ax.hist(mol_pa["pa_pred_kcalmol"], bins=50, color=COLORS["pool"],
            alpha=0.7, label="All candidates", density=True)
    ax.hist(selected["pa_pred_kcalmol"], bins=20, color=COLORS["selected"],
            alpha=0.85, label="Pareto selected", density=True)
    ax.axvspan(PA_LOW, PA_HIGH, color=COLORS["window"], alpha=0.15,
               label="Target window")
    ax.axvline(PA_LOW, color=COLORS["window"], lw=2.0, ls="--")
    ax.axvline(PA_HIGH, color=COLORS["window"], lw=2.0, ls="--")

    # Reference lines use distinct styles because the values are closely spaced.
    reference_lines = [
        ("Pyrazole", 213.7, ":"),
        ("Imidazole", 225.3, "--"),
        ("Benzimidazole", 228.0, "-."),
    ]
    for name, val, linestyle in reference_lines:
        ax.axvline(
            val, color="k", lw=1.2, ls=linestyle, alpha=0.8,
            label=f"{name} ({val:.1f})",
        )

    ax.set_xlabel("PA$_\\mathrm{pred}$ (kcal/mol)", fontsize=J_LABEL)
    ax.set_ylabel("Density", fontsize=J_LABEL)
    ax.set_title("ML-Corrected Predicted PA", fontsize=J_TITLE, fontweight="bold")
    ax.tick_params(labelsize=J_TICK)
    ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="upper right")

    plt.tight_layout()
    path = FIG_DIR / "pa_distribution.pdf"
    plt.savefig(path)
    plt.close()
    log.info(f"Saved {path}")

def plot_pareto_scatter(mol_pa, verdicts, selected, plt):
    accepted = verdicts[(verdicts["final_verdict"].isin(["accept", "flag"])) & (verdicts["pa_pred_kcalmol"] >= PA_LOW) & (verdicts["pa_pred_kcalmol"] <= PA_HIGH)]
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 3.0))
    sc = ax.scatter(accepted["pa_pred_kcalmol"], accepted["uncertainty"], c=accepted["sa_score"], cmap="YlOrRd", vmin=1, vmax=5, s=12, alpha=0.5)
    plt.colorbar(sc, ax=ax).set_label("SA Score", fontsize=J_LABEL)
    ax.scatter(selected["pa_pred_kcalmol"], selected["uncertainty"], c=COLORS["selected"], s=80, marker="*")
    ax.set_xlabel("PA$_{pred}$ (kcal mol$^{-1}$)")
    ax.set_ylabel("Ensemble dispersion (kcal mol$^{-1}$)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pareto_scatter.pdf")
    plt.close()

def plot_structures(selected, plt):
    try:
        from rdkit.Chem import Draw, MolFromSmiles
        top = selected.nsmallest(12, "uncertainty").reset_index(drop=True)
        mols, legends = [], []
        for _, row in top.iterrows():
            mol = MolFromSmiles(row["smiles"])
            if mol:
                mols.append(mol)
                legends.append(f"PA={row['pa_pred_kcalmol']:.1f} disp.={row['uncertainty']:.1f}")
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 220), legends=legends, returnPNG=False)
        img.save(str(FIG_DIR / "top_candidates.png"))
    except Exception as e: log.warning(f"Structure plot failed: {e}")

def plot_parity(selected, plt):
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 3.4))
    sc = ax.scatter(selected["pa_pm7_kcalmol"], selected["pa_pred_kcalmol"], c=selected["sa_score"], cmap="YlOrRd", s=70)
    lims = [min(selected["pa_pm7_kcalmol"].min(), selected["pa_pred_kcalmol"].min()) - 2, max(selected["pa_pm7_kcalmol"].max(), selected["pa_pred_kcalmol"].max()) + 2]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlabel("PA$_{PM7}$")
    ax.set_ylabel("PA$_{pred}$")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pa_parity_selected.pdf")
    plt.close()

def main(iteration: int) -> None:
    iter_dir = DATA_ROOT / f"iter{iteration}"
    candidates = pd.read_parquet(iter_dir / "candidates.parquet")
    mol_pa     = pd.read_parquet(iter_dir / "molecular_pa.parquet")
    verdicts   = pd.read_parquet(iter_dir / "llm_verdicts.parquet")
    selected   = pd.read_csv(iter_dir / "pareto_selected.csv")
    summary = json.loads((iter_dir / "pipeline_summary.json").read_text())["summary"]
    plt = setup_style()
    plot_funnel(summary, plt)
    plot_pa_distribution(mol_pa, selected, plt)
    plot_pareto_scatter(mol_pa, verdicts, selected, plt)
    plot_structures(selected, plt)
    plot_parity(selected, plt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=1)
    args = parser.parse_args()
    main(iteration=args.iter)
