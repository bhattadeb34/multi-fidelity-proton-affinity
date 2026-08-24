"""
tanimoto_bias_analysis.py
=========================
Plot model prediction bias as a function of maximum Tanimoto similarity
to the k-means training set, for the 30 DFT-validated Pareto candidates.

Writes:
  figures/tanimoto_bias_plot_final.pdf
  results/tanimoto_bias/tanimoto_bias_results.csv (if generated upstream)

Run from anywhere:
  python scripts/analysis/tanimoto_bias_analysis.py
"""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog('rdApp.*')

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR    = next((p for p in (PROJECT_DIR / "data", PROJECT_DIR.parent / "data") if p.exists()), PROJECT_DIR / "data")
FIG_DIR     = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results" / "tanimoto_bias"
CONFIG_PATH = PROJECT_DIR / "screening" / "config" / "pipeline_config.json"
EXECUTION_DIR = PROJECT_DIR / "screening" / "scripts" / "execution"
sys.path.insert(0, str(EXECUTION_DIR))

from pipeline_config import PATHS  # noqa: E402

FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--iter", type=int, default=1)
parser.add_argument("--dft-dir", type=Path, default=None)
parser.add_argument(
    "--top5-csv",
    type=Path,
    default=None,
    help="Generated priority-lead table from 12_select_dft_leads.py",
)
args = parser.parse_args()
archive_dir = PATHS.dft_validation_dir(
    args.iter,
    explicit=args.dft_dir,
    required_files=("lead_selection_analysis.csv", "top5_leads.csv"),
)
top5_csv = args.top5_csv or archive_dir / "top5_leads.csv"

config = json.loads(CONFIG_PATH.read_text())
PA_LOW = float(config["pa_window_kcalmol"]["low"])
PA_HIGH = float(config["pa_window_kcalmol"]["high"])
MORGAN_RADIUS = int(config["featurization"]["morgan_radius"])
MORGAN_FP_SIZE = int(config["featurization"]["morgan_fp_size"])

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading DFT prospective results...")
dft_best = pd.read_csv(archive_dir / "lead_selection_analysis.csv").rename(
    columns={"pa_dft_kcalmol": "pa_kcal"}
)

# Compute bias: Prediction - Ground Truth
dft_best['bias'] = dft_best['pa_pred_kcalmol'] - dft_best['pa_kcal']
mean_bias = dft_best['bias'].mean()

# ── Load k-means training SMILES & Compute Similarities ──────────────────
print("Computing Tanimoto similarities to training set...")
km = pd.read_parquet(DATA_DIR / "targets" / "kmeans251_ml.parquet", columns=['neutral_smiles'])
km_smiles = km['neutral_smiles'].unique().tolist()

def get_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return (
        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=MORGAN_RADIUS,
            nBits=MORGAN_FP_SIZE,
        )
        if mol
        else None
    )

km_fps = []
for smi in km_smiles:
    fp = get_fp(smi)
    if fp is not None:
        km_fps.append(fp)

tanimoto_max = []
for smi in dft_best['smiles']:
    fp = get_fp(smi)
    if fp:
        sims = DataStructs.BulkTanimotoSimilarity(fp, km_fps)
        tanimoto_max.append(max(sims))
    else:
        tanimoto_max.append(np.nan)

dft_best['tanimoto_max'] = tanimoto_max
threshold = np.nanmedian(tanimoto_max)
dft_best[[
    "mol_idx", "smiles", "pa_pred_kcalmol", "pa_kcal", "bias",
    "tanimoto_max", "lead_label", "lead_eligible",
]].to_csv(RESULTS_DIR / "tanimoto_bias_results.csv", index=False)
(RESULTS_DIR / "summary.json").write_text(json.dumps({
    "n_candidates": int(len(dft_best)),
    "mean_bias_kcalmol": float(mean_bias),
    "median_max_tanimoto": float(threshold),
    "pearson_similarity_vs_bias": float(
        dft_best[["tanimoto_max", "bias"]].corr(method="pearson").iloc[0, 1]
    ),
    "spearman_similarity_vs_bias": float(
        dft_best[["tanimoto_max", "bias"]].corr(method="spearman").iloc[0, 1]
    ),
    "fingerprint": (
        f"Morgan bit vector, radius={MORGAN_RADIUS}, nBits={MORGAN_FP_SIZE}"
    ),
    "pa_window_kcalmol": {"low": PA_LOW, "high": PA_HIGH},
}, indent=2))

# ── Journal-style defaults ────────────────────────────────────────────────
J_TICK = 9
J_LABEL = 10
J_LEGEND = 9
J_SPINE = 1.0
DOUBLE_COL_W = 7.0

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         J_TICK,
    'axes.linewidth':    J_SPINE,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.labelsize':   J_TICK,
    'ytick.labelsize':   J_TICK,
    'axes.labelsize':    J_LABEL,
    'legend.fontsize':   J_LEGEND,
    'savefig.dpi':       600,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.04,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 3.4))

# Load the generated Top 5 mapping (marker shape for identity; color = window).
if not top5_csv.exists():
    raise FileNotFoundError(
        f"Priority-lead table not found: {top5_csv}. "
        "Run 12_select_dft_leads.py first."
    )
lead_df = pd.read_csv(top5_csv)
required = {'lead_label', 'smiles'}
missing = required - set(lead_df.columns)
if missing:
    raise ValueError(f"Priority-lead table is missing columns: {sorted(missing)}")
markers = ['D', 's', '^', 'P', 'X']
if len(lead_df) != len(markers):
    raise ValueError(f"Expected five priority leads, found {len(lead_df)}")
top5_data = {
    row.smiles: {'label': row.lead_label, 'marker': marker}
    for row, marker in zip(lead_df.itertuples(index=False), markers)
}

COLOR_IN_WINDOW  = '#2166AC'
COLOR_OUT_WINDOW = '#d95f02'

top5_smiles = list(top5_data.keys())
background_df = dft_best[~dft_best['smiles'].isin(top5_smiles)]

# Plot Background
in_window = (background_df['pa_kcal'] >= PA_LOW) & (background_df['pa_kcal'] <= PA_HIGH)
ax.scatter(background_df.loc[in_window, 'tanimoto_max'], background_df.loc[in_window, 'bias'],
           c=COLOR_IN_WINDOW, s=24, alpha=0.7, edgecolors='none',
           label=f'In PA Window ({PA_LOW:g}-{PA_HIGH:g})')
ax.scatter(background_df.loc[~in_window, 'tanimoto_max'], background_df.loc[~in_window, 'bias'],
           c=COLOR_OUT_WINDOW, s=24, alpha=0.7, edgecolors='none',
           label='Outside Window')

# Plot Top 5 with Markers — colored by PA window status
for smi, info in top5_data.items():
    row = dft_best[dft_best['smiles'] == smi]
    if not row.empty:
        in_win = ((row['pa_kcal'] >= PA_LOW) & (row['pa_kcal'] <= PA_HIGH)).iloc[0]
        face_color = COLOR_IN_WINDOW if in_win else COLOR_OUT_WINDOW
        ax.scatter(row['tanimoto_max'], row['bias'],
                   marker=info['marker'], s=60, c=face_color,
                   edgecolors='black', linewidths=0.8, zorder=5)

# ── Reference Lines and Window ──
ax.axhline(mean_bias, color='#e07020', lw=1.2, ls='-.', alpha=0.85)
ax.axvline(threshold, color='gray', lw=1.2, ls=':', alpha=0.85)
# Horizontal line at 0
ax.axhline(0, color='black', lw=1.0, ls='--', alpha=0.35)

# ── Formatting ──
ax.set_xlabel('Max Tanimoto Similarity to Training Set',
              fontsize=J_LABEL, labelpad=4)
ax.set_ylabel(r'Bias: PA$_{\mathrm{pred}}$ - PA$_{\mathrm{DFT}}$' + '\n(kcal/mol)',
              fontsize=J_LABEL, labelpad=4)
ax.tick_params(axis='both', which='major', labelsize=J_TICK,
               width=J_SPINE, length=3.5)

# Right-anchored annotations for reference lines, so they use empty right space.
x_left, x_right = ax.get_xlim()
y_lower, y_upper = ax.get_ylim()
# "Mean Bias" text at right side, slightly above the bias line.
ax.text(x_right - 0.005, mean_bias + 0.15,
        f'Mean Bias: {mean_bias:+.2f} kcal/mol',
        fontsize=J_LEGEND, color='#e07020', ha='right', va='bottom')
# "Median Similarity" rotated text on the right side of the dotted line.
ax.text(threshold + 0.006, y_upper - (y_upper - y_lower) * 0.03,
        f'Median Similarity: {threshold:.2f}',
        fontsize=J_LEGEND, color='gray', rotation=90, va='top', ha='left')

# Legend Configuration (remove Mean Bias + Median Similarity — shown inline)
# Top-5 legend markers: filled with "in-window" color (their expected location),
# marker shape communicates identity.
marker_handles = [mlines.Line2D([], [], color=COLOR_IN_WINDOW, marker=info['marker'],
                                ls='None', markersize=7,
                                markeredgecolor='black', markeredgewidth=0.6,
                                label=info['label'])
                  for info in top5_data.values()]

ref_handles = [
    mpatches.Patch(
        color='#2166AC',
        alpha=0.7,
        label=f'In PA Window ({PA_LOW:g}-{PA_HIGH:g})',
    ),
    mpatches.Patch(color='#d95f02', alpha=0.7, label='Outside Window'),
]

all_handles = ref_handles + marker_handles

ax.legend(handles=all_handles,
          loc='upper left',
          bbox_to_anchor=(1.02, 1.0),
          fontsize=J_LEGEND,
          title='Legend',
          title_fontsize=J_LABEL,
          frameon=True,
          edgecolor='black',
          facecolor='white',
          borderaxespad=0.0)

plt.tight_layout()
plt.savefig(FIG_DIR / 'tanimoto_bias_plot_final.pdf', bbox_inches='tight')
print(f"Saved: {FIG_DIR}/tanimoto_bias_plot_final.pdf")
