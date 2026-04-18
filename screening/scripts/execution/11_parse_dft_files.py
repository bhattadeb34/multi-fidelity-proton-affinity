"""
11_parse_dft_files.py
=====================
Updated to include a comprehensive legend for reference lines (Identity, Bias, Window).
"""

import argparse
import logging
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR    = Path(__file__).parent
SCREENING     = SCRIPT_DIR.parent.parent
PROJECT       = SCREENING.parent
DATA_DIR      = PROJECT / "data" / "screening"
KJ_TO_KCAL    = 1 / 4.184
HARTREE_TO_KJ = 2625.4996
H_PROTON_KJ   = 6.197   
PA_LOW, PA_HIGH = 210.0, 235.0

def parse_log(log_path: Path) -> dict:
    props = {}
    if not log_path.exists(): return props
    text = log_path.read_text(errors='replace')
    m = re.search(r'SMILES:\s+(\S+)', text)
    if m: props['smiles'] = m.group(1)
    m = re.search(r'H\(total\)\s*=\s*([-\d.]+)\s*Ha', text)
    if m: props['H_total_ha'] = float(m.group(1))
    m = re.search(r'E\(elec\)\s*=\s*([-\d.]+)\s*Ha', text)
    if m: props['E_elec_ha'] = float(m.group(1))
    m = re.search(r'ZPE\s*=\s*([\d.]+)\s*kJ/mol', text)
    if m: props['ZPE_kjmol'] = float(m.group(1))
    m = re.search(r'HOMO\s*=\s*([-\d.]+)\s*eV', text)
    if m: props['HOMO_eV'] = float(m.group(1))
    m = re.search(r'LUMO\s*=\s*([-\d.]+)\s*eV', text)
    if m: props['LUMO_eV'] = float(m.group(1))
    m = re.search(r'HOMO-LUMO gap\s*=\s*([\d.]+)\s*eV', text)
    if m: props['gap_eV'] = float(m.group(1))
    m = re.search(r'\|mu\|\s*=\s*([\d.]+)\s*Debye', text)
    if m: props['dipole_debye'] = float(m.group(1))
    m = re.search(r'Imaginary frequencies:\s*(\d+)', text)
    if m: props['n_imag_freq'] = int(m.group(1))
    m = re.search(r'Charge:\s*(\d+)', text)
    if m: props['charge'] = int(m.group(1))
    m = re.search(r'N atoms:\s*(\d+)', text)
    if m: props['n_atoms'] = int(m.group(1))
    m = re.search(r'Wall time:\s*([\d.]+)\s*s', text)
    if m: props['wall_time_s'] = float(m.group(1))
    props['converged'] = 'Normal termination.' in text
    return props

def parse_freq(freq_path: Path) -> dict:
    props = {}
    if not freq_path.exists(): return props
    text = freq_path.read_text(errors='replace')
    m = re.search(r'Lowest:\s*([-\d.]+)\s*cm', text)
    if m: props['freq_lowest_cm'] = float(m.group(1))
    m = re.search(r'Highest:\s*([\d.]+)\s*cm', text)
    if m: props['freq_highest_cm'] = float(m.group(1))
    return props

def main(iteration: int, dft_dir_override: str = None) -> None:
    iter_dir    = DATA_DIR / f"iter{iteration}"
    dft_dir     = iter_dir / (dft_dir_override or "pareto_dft_files")
    pareto_csv  = iter_dir / "pareto_selected.csv"
    out_parquet = iter_dir / "dft_files_parsed.parquet"
    out_csv     = iter_dir / "dft_files_summary.csv"
    fig_dir     = SCREENING / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not dft_dir.exists():
        log.error(f"DFT files directory not found: {dft_dir}")
        sys.exit(1)

    pareto_df = pd.read_csv(pareto_csv)
    mol_dirs = sorted([d for d in dft_dir.iterdir() if d.is_dir()])

    all_records = []
    smiles_from_log = {}

    for mol_dir in mol_dirs:
        mol_idx = mol_dir.name
        log.info(f"Parsing molecule: {mol_idx}")
        
        neutral_dir = mol_dir / "neutral"
        neutral = parse_log(neutral_dir / "neutral.log")
        neutral.update(parse_freq(neutral_dir / "neutral_freq.txt"))
        
        if not neutral.get('converged') or 'H_total_ha' not in neutral:
            continue

        if 'smiles' in neutral: smiles_from_log[mol_idx] = neutral['smiles']

        H_neutral = neutral['H_total_ha']
        site_dirs = sorted([d for d in mol_dir.iterdir() if d.is_dir() and d.name.startswith('site_')])
        best_pa, best_site, site_records = None, None, []

        for site_dir in site_dirs:
            site_num = site_dir.name
            site_name = site_num.replace('_', '') 
            prot = parse_log(site_dir / f"protonated_{site_name}.log")
            prot.update(parse_freq(site_dir / f"protonated_{site_name}_freq.txt"))

            if not prot.get('converged') or 'H_total_ha' not in prot: continue

            pa_kcal = ((H_neutral - prot['H_total_ha']) * HARTREE_TO_KJ + H_PROTON_KJ) * KJ_TO_KCAL
            site_records.append({
                'mol_idx': mol_idx, 'site': site_num, 'pa_kcal': pa_kcal,
                'H_neutral_ha': H_neutral, 'H_prot_ha': prot['H_total_ha']
            })
            if best_pa is None or pa_kcal > best_pa:
                best_pa, best_site = pa_kcal, site_num

        for rec in site_records:
            rec['is_best_site'] = (rec['site'] == best_site)
            rec['pa_best_kcal'] = best_pa
        all_records.extend(site_records)

    df = pd.DataFrame(all_records)
    df['smiles'] = df['mol_idx'].map(smiles_from_log)
    df = df.merge(pareto_df[['smiles', 'pa_pred_kcalmol', 'uncertainty', 'sa_score']], on='smiles', how='left')
    df.to_parquet(out_parquet, index=False)

    best_df = df[df['is_best_site']].copy()
    best_df['delta_pred_vs_dft'] = best_df['pa_pred_kcalmol'] - best_df['pa_best_kcal']
    best_df.to_csv(out_csv, index=False)

    mae_val = best_df['delta_pred_vs_dft'].abs().mean()
    bias_val = best_df['delta_pred_vs_dft'].mean()

    # ── Final Publication-quality parity plot ──
    try:
        import matplotlib
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family": "sans-serif",
            "axes.spines.top": False, 
            "axes.spines.right": False, 
            "axes.linewidth": 1.5,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20
        })

        plot_df = best_df.dropna(subset=['pa_pred_kcalmol', 'pa_best_kcal']).copy()

        TOP5_SMILES = ['CCc1n[nH]c(-c2ccsc2)c1N', 'OCc1cc(-c2cccs2)n[nH]1', 'Nc1ccn[nH]1', 'Oc1ccc(-c2ccn[nH]2)cc1', 'Cc1n[nH]c(C)c1-c1ccco1']
        TOP5_LABELS  = ['Mol-1', 'Mol-2', 'Mol-3', 'Mol-4', 'Mol-5']
        TOP5_MARKERS = ['D', 's', '^', 'P', 'X']

        lo, hi = plot_df['pa_best_kcal'].min() - 4, plot_df['pa_best_kcal'].max() + 4
        lims = [lo, hi]
        
        fig, ax = plt.subplots(figsize=(14, 10)) 

        # 1. Identity Line
        h_yx = ax.plot(lims, lims, color='black', ls='--', lw=1.5, alpha=0.6, label='$y=x$')[0]
        
        # 2. Bias Line
        h_bias = ax.plot(lims, [l - bias_val for l in lims], color='#e07020', lw=1.5, ls='-.', label='Systematic Bias')[0]

        # 3. Target Window Shading
        h_win = ax.axhspan(PA_LOW, PA_HIGH, color='#55aa55', alpha=0.07, label='Target Window')
        ax.text(lo + 0.5, PA_HIGH + 0.5, 'Target window', fontsize=24, color='#338833', fontweight='bold')

        # Scatter points
        mask_top5 = plot_df['smiles'].isin(TOP5_SMILES)
        sc = ax.scatter(plot_df.loc[~mask_top5, 'pa_pred_kcalmol'], plot_df.loc[~mask_top5, 'pa_best_kcal'], 
                        c=plot_df.loc[~mask_top5, 'uncertainty'], cmap='YlOrRd', vmin=4, vmax=20, s=120, edgecolors='#333333', alpha=0.85, zorder=3)
        
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.8)
        cbar.set_label('ML Uncertainty (kcal mol$^{-1}$)', fontsize=26)
        cbar.ax.tick_params(labelsize=20)

        # Plot Top 5
        for smi, marker in zip(TOP5_SMILES, TOP5_MARKERS):
            row = plot_df[plot_df['smiles'] == smi]
            if not row.empty:
                ax.scatter(row['pa_pred_kcalmol'], row['pa_best_kcal'], c=row['uncertainty'], cmap='YlOrRd', vmin=4, vmax=20, s=250, marker=marker, edgecolors='#111111', linewidths=1.5, zorder=5)

        # Statistics Box
        stats = f'MAE = {mae_val:.2f}\nBias = {bias_val:+.2f}\n$n$ = {len(plot_df)}'
        ax.text(0.05, 0.95, stats, transform=ax.transAxes, fontsize=24, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#aaaaaa'))

        ax.set_xlabel('PA$_{\\mathrm{pred}}$ = PA$_{\\mathrm{PM7}}$ + $\\Delta_{\\mathrm{ML}}$ (kcal mol$^{-1}$)', fontsize=26)
        ax.set_ylabel('PA$_{\\mathrm{DFT}}$ (kcal mol$^{-1}$)', fontsize=26)
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
        
        # ── COMBINED EXTERNAL LEGEND ──
        # Reference Lines Handles
        ref_handles = [
            mlines.Line2D([], [], color='black', ls='--', lw=1.5, label='$y=x$'),
            mlines.Line2D([], [], color='#e07020', ls='-.', lw=1.5, label=f'Bias ({bias_val:+.2f})'),
            mpatches.Patch(color='#55aa55', alpha=0.3, label='Target Window')
        ]
        
        # Candidate Handles
        cand_handles = [mlines.Line2D([], [], color='#888888', marker=m, ls='None', markersize=14, markeredgecolor='k', label=lbl) 
                        for lbl, m in zip(TOP5_LABELS, TOP5_MARKERS)]
        
        all_handles = ref_handles + cand_handles
        
        ax.legend(handles=all_handles, 
                  loc='upper left', 
                  bbox_to_anchor=(1.35, 1.0), 
                  fontsize=20, 
                  title='Legend',
                  title_fontsize=22,
                  frameon=True)

        fig.savefig(fig_dir / f"iter1_pa_parity_final.pdf", dpi=300, bbox_inches='tight')
        log.info(f"Saved figure with detailed legend for Identity, Bias, and Window.")
    except Exception as e: log.warning(f"Plotting failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--dft-dir", default=None)
    args = parser.parse_args()
    main(iteration=args.iter, dft_dir_override=args.dft_dir)