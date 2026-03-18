"""
plot_exploration.py
===================
Data exploration figures for the proton affinity delta-learning paper.

Figures (saved to ../figures/):
  1. parity_dft_vs_exp.pdf/.png      B3LYP DFT vs experimental PA  (NIST)
  2. parity_pm7_vs_exp.pdf/.png      PM7 vs experimental PA  (NIST)
  3. parity_pm7_vs_dft_kmeans.pdf/.png   PM7 vs B3LYP DFT PA  (k-means, site-level)
  4. pa_distribution.pdf/.png        PA distributions: Exp(NIST) & DFT(k-means)
  5. pa_vs_mw.pdf/.png               PA vs molecular weight (both datasets)
  6. complexity_bottcher.pdf/.png    Bottcher complexity: NIST vs k-means
  7. pa_dft_comparison.pdf/.png      DFT PA: NIST vs k-means (apples to apples)
  8. functional_groups.pdf/.png      Functional group prevalence: NIST vs k-means

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
PROJECT_DIR   = SCRIPT_DIR.parent
DATA_DIR      = PROJECT_DIR / "data"
FIG_DIR       = PROJECT_DIR / "figures"
KJMOL_TO_KCAL = 1 / 4.184

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
TICK_SIZE   = 20
LABEL_SIZE  = 24
LEGEND_SIZE = 18
SPINE_LW    = 1.5
FIG_SQ      = (7, 7)
FIG_WIDE    = (10, 6)
DPI         = 300

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
    """
    Returns dict with:
      nist_feat    — full NIST feature DataFrame (site-level, 1867 rows)
      km_feat      — full k-means feature DataFrame (site-level, 823 rows)
      nist_mol     — NIST molecule-level: neutral_smiles, exp_pa_kcalmol,
                     pm7_pa_kcalmol, mw  (1155 molecules)
      km_mol       — k-means molecule-level: neutral_smiles, dft_pa_kcalmol,
                     pm7_pa_kcalmol, mw  (251 molecules)
      km_site      — k-means site-level with both pm7_pa and dft_pa
      dft_nist_mol — NIST molecules with DFT PA from dataset.json json-source
    """
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

    # json-source records → NIST DFT
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
    log.info(f"  DFT PA (NIST json-source): {len(dft_nist_mol)} molecules")

    # folder-source records → k-means DFT (site-level then molecule-level)
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
    log.info(f"  DFT PA (k-means folder): {len(dft_km_site)} sites, "
             f"{dft_km_site['neutral_smiles'].nunique()} molecules")

    # NIST molecule-level (max PM7 PA per molecule)
    nist_mol = (nist_feat
                .groupby("neutral_smiles", sort=False)
                .agg(pm7_pa_kjmol=("pm7_pa_kjmol", "max"),
                     mw=("neutral_rdkit_MolWt", "first"))
                .reset_index())
    nist_mol = nist_mol.merge(df_exp, on="neutral_smiles", how="left")
    nist_mol["pm7_pa_kcalmol"] = nist_mol["pm7_pa_kjmol"] * KJMOL_TO_KCAL
    log.info(f"  NIST mol-level: {len(nist_mol)}, "
             f"exp PA: {nist_mol['exp_pa_kcalmol'].notna().sum()}")

    # k-means molecule-level (max DFT PA per molecule)
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
    log.info(f"  k-means mol-level: {len(km_mol)}")

    # k-means site-level with PM7 PA joined
    # Try exact join on both neutral + protonated SMILES
    km_site = dft_km_site.merge(
        km_feat[["neutral_smiles", "protonated_smiles", "pm7_pa_kjmol"]],
        on=["neutral_smiles", "protonated_smiles"],
        how="left",
    )
    n_exact = km_site["pm7_pa_kjmol"].notna().sum()
    log.info(f"  km_site exact join: {n_exact}/{len(km_site)} rows matched")

    # Fallback: if exact join mostly failed, join on neutral_smiles only (best PM7 site)
    if n_exact < len(km_site) * 0.5:
        log.info("  km_site falling back to neutral-only join ...")
        pm7_best = (km_feat.groupby("neutral_smiles")["pm7_pa_kjmol"]
                    .max().reset_index()
                    .rename(columns={"pm7_pa_kjmol": "pm7_pa_kjmol_fb"}))
        km_site = km_site.merge(pm7_best, on="neutral_smiles", how="left")
        mask = km_site["pm7_pa_kjmol"].isna() & km_site["pm7_pa_kjmol_fb"].notna()
        km_site.loc[mask, "pm7_pa_kjmol"] = km_site.loc[mask, "pm7_pa_kjmol_fb"]
        km_site.drop(columns=["pm7_pa_kjmol_fb"], inplace=True)
        log.info(f"  km_site after fallback: {km_site['pm7_pa_kjmol'].notna().sum()} matched")

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
    ax.plot(lims, lims, "k--", lw=1.2, zorder=1)
    ax.scatter(x, y, color=color, s=18, alpha=0.6,
               linewidths=0.2, edgecolors="white", zorder=2)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.text(0.04, 0.96,
            f"MAE = {mae:.2f} kcal/mol\nR² = {r2:.3f}\nN = {mask.sum()} {n_label}",
            transform=ax.transAxes, fontsize=LEGEND_SIZE - 2,
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="lightgray", lw=0.8, alpha=0.92))
    return mae


def savefig(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(p)
    log.info(f"  Saved figures/{stem}.pdf/.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1 — B3LYP DFT vs Exp PA (NIST)
# ---------------------------------------------------------------------------
def plot_parity_dft_vs_exp(data):
    log.info("Plot 1: DFT vs Exp parity (NIST)")
    df = data["nist_mol"].merge(data["dft_nist_mol"], on="neutral_smiles", how="inner")
    x  = df["exp_pa_kcalmol"].values
    y  = df["dft_pa_kcalmol"].values

    fig, ax = plt.subplots(figsize=FIG_SQ)
    _setup_parity_ax(ax, x, y, COLOR_NIST,
                     "Experimental PA (kcal/mol)",
                     "B3LYP/def2-TZVP PA (kcal/mol)")
    fig.tight_layout()
    savefig(fig, "parity_dft_vs_exp")


# ---------------------------------------------------------------------------
# Plot 2 — PM7 vs Exp PA (NIST)
# ---------------------------------------------------------------------------
def plot_parity_pm7_vs_exp(data):
    log.info("Plot 2: PM7 vs Exp parity (NIST)")
    df = data["nist_mol"].dropna(subset=["exp_pa_kcalmol", "pm7_pa_kcalmol"])
    x  = df["exp_pa_kcalmol"].values
    y  = df["pm7_pa_kcalmol"].values

    fig, ax = plt.subplots(figsize=FIG_SQ)
    _setup_parity_ax(ax, x, y, COLOR_PM7,
                     "Experimental PA (kcal/mol)",
                     "PM7 PA (kcal/mol)")
    fig.tight_layout()
    savefig(fig, "parity_pm7_vs_exp")


# ---------------------------------------------------------------------------
# Plot 3 — PM7 vs B3LYP DFT (k-means, site-level)
# ---------------------------------------------------------------------------
def plot_parity_pm7_vs_dft_kmeans(data):
    log.info("Plot 3: PM7 vs DFT parity (k-means site-level)")
    df = data["km_site"].dropna(subset=["pm7_pa_kcalmol", "dft_pa_kcalmol"])
    x  = df["dft_pa_kcalmol"].values
    y  = df["pm7_pa_kcalmol"].values

    fig, ax = plt.subplots(figsize=FIG_SQ)
    _setup_parity_ax(ax, x, y, COLOR_KM,
                     "B3LYP/def2-TZVP PA (kcal/mol)",
                     "PM7 PA (kcal/mol)",
                     n_label="sites")
    fig.tight_layout()
    savefig(fig, "parity_pm7_vs_dft_kmeans")


# ---------------------------------------------------------------------------
# Plot 4 — PA distributions: Exp (NIST) and DFT (k-means)
# ---------------------------------------------------------------------------
def plot_pa_distribution(data):
    log.info("Plot 4: PA distributions")
    exp_pa = data["nist_mol"]["exp_pa_kcalmol"].dropna().values
    dft_pa = data["km_mol"]["dft_pa_kcalmol"].dropna().values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, vals, color, xlabel, title_lbl in [
        (axes[0], exp_pa, COLOR_NIST,
         "Experimental PA (kcal/mol)", "NIST Experimental Dataset"),
        (axes[1], dft_pa, COLOR_KM,
         "B3LYP/def2-TZVP PA (kcal/mol)", "k-means DFT Dataset"),
    ]:
        n = len(vals)
        ax.hist(vals, bins=35, color=color, alpha=0.80,
                edgecolor="white", linewidth=0.4, density=True)
        ax.axvline(np.mean(vals), color="black", lw=2.0, ls="--",
                   label=f"Mean: {np.mean(vals):.1f} kcal/mol")
        ax.axvline(np.median(vals), color="black", lw=2.0, ls=":",
                   label=f"Median: {np.median(vals):.1f} kcal/mol")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability Density")
        ax.set_title(title_lbl, fontsize=LABEL_SIZE - 2, pad=8)
        ax.text(0.97, 0.97, f"N = {n} molecules",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=LEGEND_SIZE - 1,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="lightgray", lw=0.8, alpha=0.9))
        ax.legend(loc="upper left", framealpha=0.9, edgecolor="lightgray",
                  fontsize=LEGEND_SIZE - 3)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    fig.tight_layout()
    savefig(fig, "pa_distribution")


# ---------------------------------------------------------------------------
# Plot 5 — PA vs Molecular Weight
# ---------------------------------------------------------------------------
def plot_pa_vs_mw(data):
    log.info("Plot 5: PA vs MW")
    nist = data["nist_mol"].dropna(subset=["exp_pa_kcalmol", "mw"])
    km   = data["km_mol"].dropna(subset=["dft_pa_kcalmol", "mw"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, df, x_col, y_col, xlabel, ylabel, color, n_str in [
        (axes[0], nist, "mw", "exp_pa_kcalmol",
         "Molecular Weight (Da)", "Experimental PA (kcal/mol)",
         COLOR_NIST, f"N = {len(nist)} molecules"),
        (axes[1], km,   "mw", "dft_pa_kcalmol",
         "Molecular Weight (Da)", "B3LYP/def2-TZVP PA (kcal/mol)",
         COLOR_KM, f"N = {len(km)} molecules"),
    ]:
        ax.scatter(df[x_col], df[y_col], color=color,
                   s=18, alpha=0.55, linewidths=0.2, edgecolors="white")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, pval = spearmanr(df[x_col], df[y_col])
        ax.text(0.03, 0.97,
                f"Spearman ρ = {r:.2f}\n{n_str}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=LEGEND_SIZE - 2,
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="lightgray", lw=0.8, alpha=0.92))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    fig.tight_layout()
    savefig(fig, "pa_vs_mw")


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
    log.info(f"  NIST valid (>0): {len(nist_scores)}/{len(nist_smiles)}  "
             f"k-means valid (>0): {len(km_scores)}/{len(km_smiles)}")
    if len(nist_scores) == 0 or len(km_scores) == 0:
        log.error("  Bottcher scores still zero — skipping complexity plot")
        return
    log.info(f"  NIST mean: {np.mean(nist_scores):.1f}   k-means mean: {np.mean(km_scores):.1f}")
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    bins = np.linspace(0, max(nist_scores.max(), km_scores.max()) * 1.04, 55)
    ax.hist(nist_scores, bins=bins, color=COLOR_NIST, alpha=0.60,
            density=True, edgecolor="white", lw=0.3,
            label=f"NIST experimental  (N = {len(nist_scores)})")
    ax.hist(km_scores, bins=bins, color=COLOR_KM, alpha=0.60,
            density=True, edgecolor="white", lw=0.3,
            label=f"k-means DFT  (N = {len(km_scores)})")
    for vals, color, label in [
        (nist_scores, COLOR_NIST, f"NIST mean: {np.mean(nist_scores):.1f}"),
        (km_scores,   COLOR_KM,   f"k-means mean: {np.mean(km_scores):.1f}"),
    ]:
        ax.axvline(np.mean(vals), color=color, lw=2.2, ls="--", label=label)
    ax.set_xlabel("Böttcher Complexity Score (Structural & Electronic)")
    ax.set_ylabel("Probability Density")
    ax.legend(framealpha=0.9, edgecolor="lightgray", loc="upper right")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    fig.tight_layout()
    savefig(fig, "complexity_bottcher")

def plot_pa_dft_comparison(data):
    log.info("Plot 7: DFT PA comparison (apples-to-apples)")
    dft_nist = data["dft_nist_mol"]["dft_pa_kcalmol"].dropna().values
    dft_km   = data["km_mol"]["dft_pa_kcalmol"].dropna().values
    n_nist, n_km = len(dft_nist), len(dft_km)

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    all_v = np.concatenate([dft_nist, dft_km])
    bins  = np.linspace(all_v.min() * 0.97, all_v.max() * 1.03, 45)

    ax.hist(dft_nist, bins=bins, color=COLOR_NIST, alpha=0.65,
            density=True, edgecolor="white", lw=0.3,
            label=f"NIST molecules  (N = {n_nist})")
    ax.hist(dft_km, bins=bins, color=COLOR_KM, alpha=0.65,
            density=True, edgecolor="white", lw=0.3,
            label=f"k-means molecules  (N = {n_km})")

    # Mean lines with annotation
    for vals, color, xoffset in [
        (dft_nist, COLOR_NIST, -15),
        (dft_km,   COLOR_KM,    8),
    ]:
        m = np.mean(vals)
        ax.axvline(m, color=color, lw=2.2, ls="--")
        ax.text(m + xoffset, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.01,
                f"{m:.1f}", color=color, fontsize=LEGEND_SIZE - 3,
                ha="center", va="top", fontweight="bold")

    ax.set_xlabel("B3LYP/def2-TZVP PA (kcal/mol)")
    ax.set_ylabel("Probability Density")
    ax.legend(framealpha=0.9, edgecolor="lightgray")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    fig.tight_layout()
    savefig(fig, "pa_dft_comparison")


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

    fig, axes = plt.subplots(1, 2, figsize=(14, fig_h),
                              sharey=True, gridspec_kw={"wspace": 0.04})

    for ax, col, color, title, n_mol in [
        (axes[0], "nist", COLOR_NIST, "NIST Experimental", 1155),
        (axes[1], "km",   COLOR_KM,   "k-means DFT",       251),
    ]:
        vals = df_fg[col].values
        bars = ax.barh(y, vals, height=h * 1.8, color=color,
                       alpha=0.82, edgecolor="white", linewidth=0.3)
        ax.set_xlim(0, max(df_fg["nist"].max(), df_fg["km"].max()) * 1.18)
        ax.set_xlabel("Molecules with group (%)")
        ax.set_title(f"{title}\n(N = {n_mol} molecules)",
                     fontsize=LABEL_SIZE - 2, pad=8)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.grid(axis="x", linewidth=0.5, alpha=0.4, ls="--")
        ax.set_axisbelow(True)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            if val >= 2.0:
                ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{val:.0f}%", va="center",
                        fontsize=TICK_SIZE - 6, color="black")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df_fg["label"].values, fontsize=TICK_SIZE - 3)
    axes[0].invert_xaxis()  # left panel reads right-to-left for back-to-back layout

    fig.tight_layout()
    savefig(fig, "functional_groups")


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
    plot_pa_distribution(data)
    plot_pa_vs_mw(data)
    plot_complexity_bottcher(data)
    plot_pa_dft_comparison(data)
    plot_functional_groups(data)

    print(f"\n  All exploration figures saved to: figures/")
    print(f"  Files: parity_dft_vs_exp, parity_pm7_vs_exp,")
    print(f"         parity_pm7_vs_dft_kmeans, pa_distribution,")
    print(f"         pa_vs_mw, complexity_bottcher,")
    print(f"         pa_dft_comparison, functional_groups")


if __name__ == "__main__":
    main()
