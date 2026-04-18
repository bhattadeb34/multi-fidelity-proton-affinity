"""
plot_exploration.py
===================
Data exploration figures for the proton affinity delta-learning paper.

Figures (saved to ../figures/):
  1. parity_dft_vs_exp.pdf      B3LYP DFT vs experimental PA  (NIST)
  2. parity_pm7_vs_exp.pdf      PM7 vs experimental PA  (NIST)
  3. parity_pm7_vs_dft_kmeans.pdf   PM7 vs B3LYP DFT PA  (k-means, site-level)
  4. pa_distribution.pdf        PA distributions: Exp(NIST) & DFT(k-means)
  5. pa_vs_mw.pdf               PA vs molecular weight (both datasets)
  6. complexity_bottcher.pdf    Bottcher complexity: NIST vs k-means
  7. pa_dft_comparison.pdf      DFT PA: NIST vs k-means (apples to apples)
  8. functional_groups.pdf      Functional group prevalence: NIST vs k-means

Data sources
------------
  PM7 PA, MW, functional groups : data/features/nist1185_features.parquet
                                   data/features/kmeans251_features.parquet
  Experimental PA                : data/Merz-and-hogni-dataset.xlsx
  DFT PA (both datasets)         : data/processed/dataset.json

Usage
-----
  python scripts/plot_exploration.py
"""

import json
import logging
import warnings
from math import log
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit.Chem import Descriptors

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR    = Path(__file__).parent
PROJECT_DIR   = SCRIPT_DIR.parent.parent
DATA_DIR      = PROJECT_DIR / "data"
FIG_DIR       = PROJECT_DIR / "figures"
FIG_EXPLORE   = FIG_DIR / "exploration"
FIG_PERF      = FIG_DIR / "model_performance"
KJMOL_TO_KCAL = 1 / 4.184

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
TICK_SIZE   = 9
LABEL_SIZE  = 10
LEGEND_SIZE = 9
SPINE_LW    = 1.2
FIG_SQ      = (3.4, 3.4)
FIG_WIDE    = (7.0, 3.0)
DPI         = 600

COLOR_NIST = "#2166AC"
COLOR_KM   = "#D01C8B"
COLOR_PM7  = "#888780"

plt.rcParams.update({
    "font.family":        "sans-serif",
    "axes.linewidth":     SPINE_LW,
    "xtick.major.width":  SPINE_LW, "ytick.major.width":  SPINE_LW,
    "xtick.minor.width":  SPINE_LW * 0.6, "ytick.minor.width": SPINE_LW * 0.6,
    "xtick.major.size":   6, "ytick.major.size": 6,
    "xtick.labelsize":    TICK_SIZE, "ytick.labelsize": TICK_SIZE,
    "axes.labelsize":     LABEL_SIZE, "legend.fontsize": LEGEND_SIZE,
    "figure.dpi":         DPI, "savefig.dpi": DPI,
    "savefig.bbox":       "tight", "savefig.pad_inches": 0.15,
})


# ---------------------------------------------------------------------------
# Bottcher scorer (ported from user-provided bottchscore3.py using RDKit)
# ---------------------------------------------------------------------------
class BottcherScorer:
    """Bottcher molecular complexity scorer (RDKit port, version-robust)."""
    def __init__(self):
        self._mesomery = {
            "[$([#8;X1])]=*-[$([#8;X1])]": {"indices": [0, 2], "contribution": 1.5},
            "[$([#7;X2](=*))](=*)(-*=*)":   {"indices": [2, 1], "contribution": 1.5},
        }

    def _bo(self, bond) -> float:
        """Bond order from bond type enum, version-robust."""
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:   return 1.0
        if bt == Chem.BondType.DOUBLE:   return 2.0
        if bt == Chem.BondType.TRIPLE:   return 3.0
        if bt == Chem.BondType.AROMATIC: return 1.5
        return 1.0

    def score(self, mol) -> float:
        if mol is None:
            return np.nan
        try:
            Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        except Exception:
            pass
        mesomery_updates = {}
        for smarts, info in self._mesomery.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                continue
            for match in mol.GetSubstructMatches(pat):
                for t in info["indices"]:
                    mesomery_updates[match[t]] = info["contribution"]
        ranks     = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        atom_data = {}
        for atom in mol.GetAtoms():
            ai = atom.GetIdx()
            if atom.GetAtomicNum() == 1:
                continue
            heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
            d_i = len({ranks[n.GetIdx()] for n in heavy_nbrs})
            e_i = len({atom.GetAtomicNum()} | {n.GetAtomicNum() for n in heavy_nbrs})
            s_i = 2 if atom.GetChiralTag() != Chem.CHI_UNSPECIFIED else 1
            heavy_bo_sum = sum(self._bo(b) for b in atom.GetBonds()
                               if b.GetOtherAtom(atom).GetAtomicNum() != 1)
            V_i = 8 - int(round(heavy_bo_sum)) + atom.GetFormalCharge()
            b_i = sum(
                mesomery_updates.get(ai, self._bo(b))
                for b in atom.GetBonds()
                if b.GetOtherAtom(atom).GetAtomicNum() != 1
            )
            try:
                local_c = d_i * e_i * s_i * log(V_i * b_i, 2) if V_i > 0 and b_i > 0 else 0.0
            except Exception:
                local_c = 0.0
            atom_data[ai] = local_c
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                if bond.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
                    a1, a2 = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
                    if a1 in atom_data and a2 in atom_data:
                        if atom_data[a1] <= atom_data[a2]:
                            atom_data[a1] *= 2
                        else:
                            atom_data[a2] *= 2
        total = sum(atom_data.values())
        rank_groups = {}
        for ai in atom_data:
            rank_groups.setdefault(ranks[ai], []).append(ai)
        correction = sum(
            0.5 * atom_data[ai]
            for grp in rank_groups.values() if len(grp) > 1
            for ai in grp
        )
        return total - correction

    def score_smiles(self, smiles: str) -> float:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.nan
            mol = Chem.RemoveHs(mol)
            return self.score(mol)
        except Exception:
            return np.nan

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data() -> dict:
    log.info("Loading feature parquet files ...")
    nist_feat = pd.read_parquet(DATA_DIR / "features" / "nist1185_features.parquet")
    km_feat   = pd.read_parquet(DATA_DIR / "features" / "kmeans251_features.parquet")

    # Exp PA from Jin & Merz Excel
    excel = DATA_DIR / "Merz-and-hogni-dataset.xlsx"
    if not excel.exists():
        raise FileNotFoundError(f"Jin & Merz Excel not found: {excel}")
    df_exp = pd.read_excel(excel)[["smiles", "EXP_PA"]].rename(
        columns={"smiles": "neutral_smiles", "EXP_PA": "exp_pa_kjmol"})
    df_exp["exp_pa_kcalmol"] = df_exp["exp_pa_kjmol"] * KJMOL_TO_KCAL
    log.info(f"  Exp PA: {len(df_exp)} molecules")

    # DFT PA from dataset.json
    log.info("Loading dataset.json for DFT PA ...")
    dataset = json.loads((DATA_DIR / "processed" / "dataset.json").read_text())

    dft_nist_rows = []
    for rec in dataset.values():
        if rec["metadata"]["source"] != "json":
            continue
        smi   = rec["neutral"]["smiles"]
        dft_pa = rec["labels"].get("dft_pa_kjmol")
        if smi and dft_pa:
            dft_nist_rows.append({
                "neutral_smiles": smi,
                "dft_pa_kjmol":   dft_pa,
                "dft_pa_kcalmol": dft_pa * KJMOL_TO_KCAL,
            })
    dft_nist_mol = pd.DataFrame(dft_nist_rows)

    dft_km_rows = []
    for rec in dataset.values():
        if rec["metadata"]["source"] != "folder":
            continue
        smi = rec["neutral"]["smiles"]
        for site in rec.get("all_sites", []):
            pa = site.get("pa_kjmol")
            if pa:
                dft_km_rows.append({
                    "neutral_smiles":    smi,
                    "protonated_smiles": site.get("protonated_smiles", ""),
                    "site_idx":          site.get("site_idx"),
                    "dft_pa_kjmol":      pa,
                    "dft_pa_kcalmol":    pa * KJMOL_TO_KCAL,
                })
    dft_km_site = pd.DataFrame(dft_km_rows)

    nist_mol = (nist_feat
                .groupby("neutral_smiles", sort=False)
                .agg(pm7_pa_kjmol=("pm7_pa_kjmol", "max"),
                     mw=("neutral_rdkit_MolWt", "first"))
                .reset_index())
    nist_mol = nist_mol.merge(df_exp, on="neutral_smiles", how="left")
    nist_mol["pm7_pa_kcalmol"] = nist_mol["pm7_pa_kjmol"] * KJMOL_TO_KCAL

    km_mol = (dft_km_site
              .groupby("neutral_smiles", sort=False)
              .agg(dft_pa_kjmol=("dft_pa_kjmol", "max"),
                   dft_pa_kcalmol=("dft_pa_kcalmol", "max"))
              .reset_index())
    km_props = (km_feat
                .groupby("neutral_smiles", sort=False)
                .agg(pm7_pa_kjmol=("pm7_pa_kjmol", "max"),
                     mw=("neutral_rdkit_MolWt", "first"))
                .reset_index())
    km_mol = km_mol.merge(km_props, on="neutral_smiles", how="left")
    km_mol["pm7_pa_kcalmol"] = km_mol["pm7_pa_kjmol"] * KJMOL_TO_KCAL

    km_site = dft_km_site.merge(
        km_feat[["neutral_smiles", "protonated_smiles", "pm7_pa_kjmol"]],
        on=["neutral_smiles", "protonated_smiles"],
        how="left",
    )
    n_exact = km_site["pm7_pa_kjmol"].notna().sum()

    if n_exact < len(km_site) * 0.5:
        pm7_best = (km_feat.groupby("neutral_smiles")["pm7_pa_kjmol"]
                    .max().reset_index()
                    .rename(columns={"pm7_pa_kjmol": "pm7_pa_kjmol_fb"}))
        km_site = km_site.merge(pm7_best, on="neutral_smiles", how="left")
        mask = km_site["pm7_pa_kjmol"].isna() & km_site["pm7_pa_kjmol_fb"].notna()
        km_site.loc[mask, "pm7_pa_kjmol"] = km_site.loc[mask, "pm7_pa_kjmol_fb"]
        km_site.drop(columns=["pm7_pa_kjmol_fb"], inplace=True)

    km_site["pm7_pa_kcalmol"] = km_site["pm7_pa_kjmol"] * KJMOL_TO_KCAL

    return dict(
        nist_feat=nist_feat, km_feat=km_feat,
        nist_mol=nist_mol,   km_mol=km_mol,
        km_site=km_site,     dft_nist_mol=dft_nist_mol,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _setup_parity_ax(ax, x, y, color, xlabel, ylabel, n_label="molecules"):
    """Scatter on ax with 1:1 line; returns MAE."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        return np.nan
    mae  = np.mean(np.abs(x - y))
    r2   = np.corrcoef(x, y)[0, 1] ** 2

    pad  = (max(x.max(), y.max()) - min(x.min(), y.min())) * 0.04
    lims = (min(x.min(), y.min()) - pad, max(x.max(), y.max()) + pad)
    ax.plot(lims, lims, "k--", lw=1.5, zorder=1)
    ax.scatter(x, y, color=color, s=45, alpha=0.7,
               linewidths=0.3, edgecolors="white", zorder=2)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # Stats box shifted outside plot to the right
    stats_text = f"MAE = {mae:.2f} kcal/mol\nR² = {r2:.3f}\nN = {mask.sum()} {n_label}"
    ax.text(1.05, 0.5, stats_text,
            transform=ax.transAxes, fontsize=LEGEND_SIZE,
            va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="lightgray", lw=1.2, alpha=0.92))
    return mae


def savefig(fig, stem, subdir=None):
    out_dir = (FIG_DIR / subdir) if subdir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    # Only saving PDF now to save space/clutter
    for ext in ("pdf",):
        fig.savefig(out_dir / f"{stem}.{ext}")
    rel = f"{subdir}/{stem}" if subdir else stem
    log.info(f"  Saved figures/{rel}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1 — B3LYP DFT vs Exp PA (NIST)
# ---------------------------------------------------------------------------
def plot_parity_dft_vs_exp(data):
    log.info("Plot 1: DFT vs Exp parity (NIST)")
    df = data["nist_mol"].merge(data["dft_nist_mol"], on="neutral_smiles", how="inner")
    x  = df["exp_pa_kcalmol"].values
    y  = df["dft_pa_kcalmol"].values

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    _setup_parity_ax(ax, x, y, COLOR_NIST,
                     "Experimental PA\n(kcal/mol)",
                     "B3LYP/def2-TZVP PA\n(kcal/mol)")
    fig.tight_layout()
    fig.subplots_adjust(right=0.7) # Ensure space on right for text
    savefig(fig, "parity_dft_vs_exp", "exploration")


# ---------------------------------------------------------------------------
# Plot 2 — PM7 vs Exp PA (NIST)
# ---------------------------------------------------------------------------
def plot_parity_pm7_vs_exp(data):
    log.info("Plot 2: PM7 vs Exp parity (NIST)")
    df = data["nist_mol"].dropna(subset=["exp_pa_kcalmol", "pm7_pa_kcalmol"])
    x  = df["exp_pa_kcalmol"].values
    y  = df["pm7_pa_kcalmol"].values

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    _setup_parity_ax(ax, x, y, COLOR_PM7,
                     "Experimental PA\n(kcal/mol)",
                     "PM7 PA\n(kcal/mol)")
    fig.tight_layout()
    fig.subplots_adjust(right=0.7)
    savefig(fig, "parity_pm7_vs_exp", "exploration")


# ---------------------------------------------------------------------------
# Plot 3 — PM7 vs B3LYP DFT (k-means, site-level)
# ---------------------------------------------------------------------------
def plot_parity_pm7_vs_dft_kmeans(data):
    log.info("Plot 3: PM7 vs DFT parity (k-means site-level)")
    km_ml = pd.read_parquet(
        DATA_DIR / "targets" / "kmeans251_ml.parquet"
    )
    km_ml["pm7_pa_kcalmol"] = km_ml["pm7_pa_kjmol"] * KJMOL_TO_KCAL
    km_ml["dft_pa_kcalmol"] = km_ml["dft_pa_kjmol"] * KJMOL_TO_KCAL
    df = km_ml.dropna(subset=["pm7_pa_kcalmol", "dft_pa_kcalmol"])
    
    # PM7 on x, DFT on y
    x  = df["pm7_pa_kcalmol"].values
    y  = df["dft_pa_kcalmol"].values
    
    # Calculate unique molecules
    n_mol = df["neutral_smiles"].nunique() if "neutral_smiles" in df.columns else 251

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    
    _setup_parity_ax(ax, x, y, COLOR_KM,
                     "PM7 PA\n(kcal/mol)",
                     "B3LYP/def2-TZVP PA\n(kcal/mol)",
                     n_label=f"sites ({n_mol} molecules)") # <-- FIX IS HERE
                     
    fig.tight_layout()
    fig.subplots_adjust(right=0.7)
    savefig(fig, "parity_pm7_vs_dft_kmeans", "exploration")
# ---------------------------------------------------------------------------
# Plot 4 — PA distributions: Exp (NIST) and DFT (k-means)
# ---------------------------------------------------------------------------
def plot_pa_distribution(data):
    log.info("Plot 4: PA distributions")
    exp_pa = data["nist_mol"]["exp_pa_kcalmol"].dropna().values
    dft_pa = data["km_mol"]["dft_pa_kcalmol"].dropna().values

    # Resized from (20, 7) to (16, 8) to make individual subplots perfectly square (8x8)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))

    for ax, vals, color, xlabel, title_lbl in [
        (axes[0], exp_pa, COLOR_NIST,
         "Experimental PA (kcal/mol)", "NIST Experimental Dataset"),
        (axes[1], dft_pa, COLOR_KM,
         "B3LYP/def2-TZVP PA (kcal/mol)", "k-means DFT Dataset"),
    ]:
        n    = len(vals)
        mean = np.mean(vals)
        med  = np.median(vals)

        ax.hist(vals, bins=35, color=color, alpha=0.80,
                edgecolor="white", linewidth=0.4, density=True)
        ax.axvline(mean, color="black", lw=2.0, ls="--",
                   label=f"Mean: {mean:.1f} kcal/mol")
        ax.axvline(med,  color="black", lw=2.0, ls=":",
                   label=f"Median: {med:.1f} kcal/mol")
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        ax.set_ylabel("Probability Density", fontsize=LABEL_SIZE)
        ax.set_title(title_lbl, fontsize=LABEL_SIZE, pad=8)

        # Legend below the axes, moved much further down
        ax.legend(framealpha=0.9, edgecolor="lightgray",
                  fontsize=LEGEND_SIZE, ncol=1,
                  loc="upper center",
                  bbox_to_anchor=(0.5, -0.25),
                  borderaxespad=0)

        ax.text(0.96, 0.96, f"N = {n} molecules",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=LEGEND_SIZE, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="lightgray", lw=1.2, alpha=0.9))

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    fig.tight_layout()
    # Given the lower legend, we need more bottom padding
    fig.subplots_adjust(bottom=0.35)
    savefig(fig, "pa_distribution", "exploration")

# ---------------------------------------------------------------------------
# Plot 5 — PA vs Molecular Weight
# ---------------------------------------------------------------------------
def plot_pa_vs_mw(data):
    log.info("Plot 5: PA vs MW")
    nist = data["nist_mol"].dropna(subset=["exp_pa_kcalmol", "mw"])
    km   = data["km_mol"].dropna(subset=["dft_pa_kcalmol", "mw"])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for ax, df, x_col, y_col, xlabel, ylabel, color, n_str in [
        (axes[0], nist, "mw", "exp_pa_kcalmol",
         "Molecular Weight (Da)", "Experimental PA\n(kcal/mol)",
         COLOR_NIST, f"N = {len(nist)} molecules"),
        (axes[1], km,   "mw", "dft_pa_kcalmol",
         "Molecular Weight (Da)", "B3LYP/def2-TZVP PA\n(kcal/mol)",
         COLOR_KM, f"N = {len(km)} molecules"),
    ]:
        ax.scatter(df[x_col], df[y_col], color=color,
                   s=18, alpha=0.55, linewidths=0.2, edgecolors="white")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, pval = spearmanr(df[x_col], df[y_col])
        ax.text(0.96, 0.04,
                f"Spearman ρ = {r:.2f}\n{n_str}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=LEGEND_SIZE,
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="lightgray", lw=1.2, alpha=0.92))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    fig.tight_layout()
    savefig(fig, "pa_vs_mw", "exploration")


# ---------------------------------------------------------------------------
# Plot 6 — Bottcher complexity: NIST vs k-means
# ---------------------------------------------------------------------------
def _bottcher_score_inline(smiles: str) -> float:
    """Standalone Bottcher scorer bypassing class — immune to .pyc cache issues."""
    from math import log as _log
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        mol = Chem.RemoveHs(mol)
        try:
            Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        except Exception:
            pass
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        atom_data = {}
        for atom in mol.GetAtoms():
            ai = atom.GetIdx()
            if atom.GetAtomicNum() == 1:
                continue
            heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
            d_i = len({ranks[n.GetIdx()] for n in heavy_nbrs})
            e_i = len({atom.GetAtomicNum()} | {n.GetAtomicNum() for n in heavy_nbrs})
            s_i = 2 if atom.GetChiralTag() != Chem.CHI_UNSPECIFIED else 1
            hbs = 0.0
            for b in atom.GetBonds():
                if b.GetOtherAtom(atom).GetAtomicNum() == 1:
                    continue
                bt = b.GetBondType()
                if bt == Chem.BondType.SINGLE:   hbs += 1.0
                elif bt == Chem.BondType.DOUBLE:  hbs += 2.0
                elif bt == Chem.BondType.TRIPLE:  hbs += 3.0
                elif bt == Chem.BondType.AROMATIC: hbs += 1.5
                else: hbs += 1.0
            V_i = 8 - int(round(hbs)) + atom.GetFormalCharge()
            b_i = hbs
            try:
                c = d_i * e_i * s_i * _log(V_i * b_i, 2) if V_i > 0 and b_i > 0 else 0.0
            except Exception:
                c = 0.0
            atom_data[ai] = c
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                if bond.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
                    a1 = bond.GetBeginAtom().GetIdx()
                    a2 = bond.GetEndAtom().GetIdx()
                    if a1 in atom_data and a2 in atom_data:
                        if atom_data[a1] <= atom_data[a2]:
                            atom_data[a1] *= 2
                        else:
                            atom_data[a2] *= 2
        total = sum(atom_data.values())
        rank_groups = {}
        for ai in atom_data:
            rank_groups.setdefault(ranks[ai], []).append(ai)
        correction = sum(
            0.5 * atom_data[ai]
            for grp in rank_groups.values() if len(grp) > 1
            for ai in grp
        )
        return total - correction
    except Exception:
        return np.nan


def plot_complexity_bottcher(data):
    """Fig 6: Bottcher complexity score distributions."""
    log.info("Plot 6: Bottcher complexity (~2 min) ...")
    nist_smiles = data["nist_mol"]["neutral_smiles"].unique()
    km_smiles   = data["km_mol"]["neutral_smiles"].unique()
    log.info(f"  Scoring {len(nist_smiles)} NIST + {len(km_smiles)} k-means molecules ...")
    nist_scores = np.array([_bottcher_score_inline(s) for s in nist_smiles])
    km_scores   = np.array([_bottcher_score_inline(s) for s in km_smiles])
    nist_scores = nist_scores[np.isfinite(nist_scores) & (nist_scores > 0)]
    km_scores   = km_scores[np.isfinite(km_scores) & (km_scores > 0)]
    if len(nist_scores) == 0 or len(km_scores) == 0:
        log.error("  Bottcher scores still zero — skipping complexity plot")
        return
    log.info(f"  NIST mean: {np.mean(nist_scores):.1f}   k-means mean: {np.mean(km_scores):.1f}")

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    x_max = max(nist_scores.max(), km_scores.max())
    bins = np.linspace(0, x_max * 1.04, 55)

    ax.hist(nist_scores, bins=bins, color=COLOR_NIST, alpha=0.60,
            density=True, edgecolor="white", lw=0.3,
            label=f"NIST (N={len(nist_scores)})")
    ax.hist(km_scores, bins=bins, color=COLOR_KM, alpha=0.60,
            density=True, edgecolor="white", lw=0.3,
            label=f"k-means (N={len(km_scores)})")
    for vals, color in [
        (nist_scores, COLOR_NIST),
        (km_scores,   COLOR_KM),
    ]:
        ax.axvline(np.mean(vals), color=color, lw=2.2, ls="--")

    # Mean annotations inside plot — clamp to axis limits so text stays visible.
    # Manual controls (tune for best spacing):
    # - Offsets move the text relative to the dashed mean lines.
    # - Clamp fractions prevent labels from spilling outside the axis.
    MEAN_TEXT_BLUE_X_OFFSET = 3.0    # x text = nist_mean - offset
    MEAN_TEXT_PINK_X_OFFSET = 4.0    # x text = km_mean + offset
    MEAN_TEXT_BLUE_CLAMP_PAD_FRAC = 0.35  # min x = x_left + frac*x_span
    MEAN_TEXT_PINK_CLAMP_PAD_FRAC = 0.30  # max x = x_right - frac*x_span
    nist_mean = np.mean(nist_scores)
    km_mean   = np.mean(km_scores)
    y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.01
    x_left, x_right = ax.get_xlim()
    x_span = x_right - x_left
    left_pad = MEAN_TEXT_BLUE_CLAMP_PAD_FRAC * x_span
    right_pad = MEAN_TEXT_PINK_CLAMP_PAD_FRAC * x_span

    # NIST (blue): keep to the left of its dashed line, but never beyond y-axis.
    x_nist = max(nist_mean - MEAN_TEXT_BLUE_X_OFFSET, x_left + left_pad)
    ax.text(x_nist, y_top * 0.88,
            f"mean = {nist_mean:.1f}", color=COLOR_NIST,
            fontsize=LEGEND_SIZE, fontweight="bold", ha="right", va="top")

    # k-means (pink): move to the right, closer to its own dashed line.
    x_km = min(km_mean + MEAN_TEXT_PINK_X_OFFSET, x_right - right_pad)
    ax.text(x_km, y_top * 0.75,
            f"mean = {km_mean:.1f}", color=COLOR_KM,
            fontsize=LEGEND_SIZE, fontweight="bold", ha="left", va="top")

    # Wrapped x-axis label
    ax.set_xlabel("Böttcher Complexity Score\n(Structural & Electronic)",
                  fontsize=LABEL_SIZE)
    # Y-axis: scale ticks ×10³ so labels read 2, 4, 6 … instead of 0.002 …
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v * 1e3:.0f}"))
    ax.set_ylabel("Probability Density (×10$^{-3}$)", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # Legend inside, top-right, dark border
    ax.legend(framealpha=0.95, edgecolor="black", fancybox=False,
              loc="upper right", fontsize=LEGEND_SIZE,
              prop=dict(weight="bold"))
    fig.tight_layout()
    savefig(fig, "complexity_bottcher", "exploration")

def plot_parity_combined(data):
    """Combined 3-panel parity: (a) DFT vs Exp, (b) PM7 vs Exp, (c) PM7 vs DFT."""
    log.info("Plot combined: Parity — (a) DFT vs Exp, (b) PM7 vs Exp, (c) PM7 vs DFT")

    # ── Data prep ────────────────────────────────────────────────────────
    # (a) DFT vs Exp (NIST)
    df_a = data["nist_mol"].merge(data["dft_nist_mol"], on="neutral_smiles", how="inner")
    xa = df_a["exp_pa_kcalmol"].values
    ya = df_a["dft_pa_kcalmol"].values

    # (b) PM7 vs Exp (NIST)
    df_b = data["nist_mol"].dropna(subset=["exp_pa_kcalmol", "pm7_pa_kcalmol"])
    xb = df_b["exp_pa_kcalmol"].values
    yb = df_b["pm7_pa_kcalmol"].values

    # (c) PM7 vs DFT (k-means site-level)
    km_ml = pd.read_parquet(DATA_DIR / "targets" / "kmeans251_ml.parquet")
    km_ml["pm7_pa_kcalmol"] = km_ml["pm7_pa_kjmol"] * KJMOL_TO_KCAL
    km_ml["dft_pa_kcalmol"] = km_ml["dft_pa_kjmol"] * KJMOL_TO_KCAL
    df_c = km_ml.dropna(subset=["pm7_pa_kcalmol", "dft_pa_kcalmol"])
    xc = df_c["pm7_pa_kcalmol"].values
    yc = df_c["dft_pa_kcalmol"].values
    n_mol_c = df_c["neutral_smiles"].nunique() if "neutral_smiles" in df_c.columns else 251

    # Journal-style local settings for this combined parity figure.
    P_TICK  = 9
    P_LABEL = 10
    P_LEG   = 9
    P_PANEL = 11
    P_SPINE = 1.2

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))

    def _draw_panel(ax, x, y, color, xlabel, ylabel, stats_n_label,
                    legend_loc, panel_label, override_n=None):
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        mae = np.mean(np.abs(x - y))
        r2  = np.corrcoef(x, y)[0, 1] ** 2

        pad  = (max(x.max(), y.max()) - min(x.min(), y.min())) * 0.04
        lims = (min(x.min(), y.min()) - pad, max(x.max(), y.max()) + pad)
        ax.plot(lims, lims, "k--", lw=1.0, zorder=1)
        ax.scatter(x, y, color=color, s=12, alpha=0.80,
                   linewidths=0.4, edgecolors="black", zorder=2)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel, fontsize=P_LABEL)
        ax.set_ylabel(ylabel, fontsize=P_LABEL)

        # Exactly 5 major ticks, lowest and highest shown
        ticks = np.linspace(lims[0], lims[1], 5)
        # Round to nearest integer for clean labels
        ticks = np.round(ticks).astype(int)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        # Thicker spines
        for spine in ax.spines.values():
            spine.set_linewidth(P_SPINE)

        ax.tick_params(axis="both", which="major", labelsize=P_TICK,
                       width=P_SPINE, length=3.5)
        ax.tick_params(axis="both", which="minor", width=P_SPINE * 0.7,
                       length=2.0)

        # Stats legend inside the plot — bold text, dark border
        n_display = override_n if override_n is not None else mask.sum()
        stats_text = (f"MAE = {mae:.2f}\n"
                      f"R\u00b2 = {r2:.3f}\n"
                      f"N = {n_display}")
        ax.text(
            0.04 if "left" in legend_loc else 0.96,
            0.96 if "upper" in legend_loc else 0.04,
            stats_text,
            transform=ax.transAxes, fontsize=P_LEG,
            fontweight="bold", multialignment="left",
            va="top" if "upper" in legend_loc else "bottom",
            ha="left" if "left" in legend_loc else "right")

        # Panel label — above the axes, clear of y-tick labels
        ax.text(0.00, 1.12, panel_label, transform=ax.transAxes,
                fontsize=P_PANEL, fontweight="bold", va="top")

    _draw_panel(axes[0], xa, ya, COLOR_NIST,
                "Experimental PA\n(kcal/mol)",
                "B3LYP/def2-TZVP PA\n(kcal/mol)",
                "molecules", "upper left", "(a)")

    _draw_panel(axes[1], xb, yb, COLOR_PM7,
                "Experimental PA\n(kcal/mol)",
                "PM7 PA\n(kcal/mol)",
                "molecules", "upper left", "(b)")

    _draw_panel(axes[2], xc, yc, COLOR_KM,
                "PM7 PA\n(kcal/mol)",
                "B3LYP/def2-TZVP PA\n(kcal/mol)",
                "molecules", "lower right", "(c)",
                override_n=n_mol_c)

    fig.tight_layout(pad=0.8, w_pad=0.8)
    savefig(fig, "parity_combined", "exploration")


def plot_pa_dft_comparison(data):
    log.info("Plot 7: DFT PA comparison (apples-to-apples)")
    dft_nist = data["dft_nist_mol"]["dft_pa_kcalmol"].dropna().values
    dft_km   = data["km_mol"]["dft_pa_kcalmol"].dropna().values
    n_nist, n_km = len(dft_nist), len(dft_km)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    all_v = np.concatenate([dft_nist, dft_km])
    bins  = np.linspace(all_v.min() * 0.97, all_v.max() * 1.03, 45)

    ax.hist(dft_nist, bins=bins, color=COLOR_NIST, alpha=0.65,
            density=True, edgecolor="white", lw=0.3,
            label=f"NIST molecules  (N = {n_nist})")
    ax.hist(dft_km, bins=bins, color=COLOR_KM, alpha=0.65,
            density=True, edgecolor="white", lw=0.3,
            label=f"k-means molecules  (N = {n_km})")

    for vals, color, xoffset in [
        (dft_nist, COLOR_NIST, -15),
        (dft_km,   COLOR_KM,    8),
    ]:
        m = np.mean(vals)
        ax.axvline(m, color=color, lw=2.2, ls="--")
        ax.text(m + xoffset, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.01,
                f"{m:.1f}", color=color, fontsize=LEGEND_SIZE,
                ha="center", va="top", fontweight="bold")

    ax.set_xlabel("B3LYP/def2-TZVP PA\n(kcal/mol)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Probability Density", fontsize=LABEL_SIZE)
    ax.legend(framealpha=0.9, edgecolor="lightgray",
              fontsize=LEGEND_SIZE,
              loc="upper left", bbox_to_anchor=(1.08, 1),
              borderaxespad=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    fig.tight_layout()
    fig.subplots_adjust(right=0.72)
    savefig(fig, "pa_dft_comparison", "exploration")


# ---------------------------------------------------------------------------
# Plot 8 — Functional groups: NIST vs k-means
# ---------------------------------------------------------------------------
def plot_functional_groups(data):
    log.info("Plot 8: Functional group prevalence")

    # Chemically curated selection with readable labels
    FG_MAP = [
        ("fr_NH2",          "Primary amine (–NH₂)"),
        ("fr_NH1",          "Secondary amine (–NHR)"),
        ("fr_NH0",          "Tertiary amine (–NR₂)"),
        ("fr_Ar_N",         "Pyridine-type N"),
        ("fr_Ar_NH",        "Pyrrole-type NH"),
        ("fr_aniline",      "Aniline"),
        ("fr_amidine",      "Amidine"),
        ("fr_Imine",        "Imine (C=N)"),
        ("fr_nitrile",      "Nitrile (C≡N)"),
        ("fr_amide",        "Amide"),
        ("fr_COO",          "Carboxylic acid"),
        ("fr_ester",        "Ester"),
        ("fr_ketone",       "Ketone"),
        ("fr_Al_OH",        "Aliphatic OH"),
        ("fr_Ar_OH",        "Phenol"),
        ("fr_ether",        "Ether (C–O–C)"),
        ("fr_C_O",          "Carbonyl (C=O)"),
        ("fr_benzene",      "Benzene ring"),
        ("fr_halogen",      "Halogen"),
        ("fr_SH",           "Thiol (–SH)"),
    ]

    nist_feat = data["nist_feat"]
    km_feat   = data["km_feat"]

    # Molecule-level prevalence: group by neutral_smiles, check if any site > 0
    nist_grp = nist_feat.groupby("neutral_smiles")
    km_grp   = km_feat.groupby("neutral_smiles")

    rows = []
    for fg_key, fg_label in FG_MAP:
        col = f"neutral_rdkit_{fg_key}"
        if col not in nist_feat.columns:
            continue
        nist_pct = (nist_grp[col].max() > 0).mean() * 100
        km_pct   = (km_grp[col].max()   > 0).mean() * 100
        rows.append({"label": fg_label, "nist": nist_pct, "km": km_pct})

    df_fg = pd.DataFrame(rows).sort_values("nist", ascending=True)

    # Horizontal grouped bar chart
    n     = len(df_fg)
    y     = np.arange(n)
    h     = 0.36
    fig_h = max(8, n * 0.55)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, max(4.6, n * 0.25)),
                              sharey=True, gridspec_kw={"wspace": 0.04})

    for ax, col, color, title, n_mol in [
        (axes[0], "nist", COLOR_NIST, "NIST Experimental", 1155),
        (axes[1], "km",   COLOR_KM,   "k-means DFT",       251),
    ]:
        vals = df_fg[col].values
        bars = ax.barh(y, vals, height=h * 1.8, color=color,
                       alpha=0.82, edgecolor="white", linewidth=0.3)
        ax.set_xlim(0, max(df_fg["nist"].max(), df_fg["km"].max()) * 1.22)
        ax.set_xlabel("Molecules with group (%)", fontsize=LABEL_SIZE)
        ax.set_title(f"{title}\n(N = {n_mol} molecules)",
                     fontsize=LABEL_SIZE, pad=8)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.grid(axis="x", linewidth=0.5, alpha=0.4, ls="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=TICK_SIZE)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            if val >= 2.0:
                ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{val:.0f}%", va="center",
                        fontsize=TICK_SIZE, color="black")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df_fg["label"].values, fontsize=TICK_SIZE)
    axes[0].invert_xaxis()  # left panel reads right-to-left for back-to-back layout

    fig.tight_layout()
    savefig(fig, "functional_groups", "exploration")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("  DATA EXPLORATION PLOTS")
    log.info("=" * 60)

    data = load_all_data()

    plot_parity_dft_vs_exp(data)
    plot_parity_pm7_vs_exp(data)
    plot_parity_pm7_vs_dft_kmeans(data)
    plot_parity_combined(data)
    plot_pa_distribution(data)
    plot_pa_vs_mw(data)
    plot_complexity_bottcher(data)
    plot_pa_dft_comparison(data)
    plot_functional_groups(data)

    print(f"\n  All exploration figures saved to: figures/")
    print(f"  Files: parity_dft_vs_exp, parity_pm7_vs_exp,")
    print(f"         parity_pm7_vs_dft_kmeans, parity_combined,")
    print(f"         pa_distribution, pa_vs_mw, complexity_bottcher,")
    print(f"         pa_dft_comparison, functional_groups")


if __name__ == "__main__":
    main()