import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
import os
from pathlib import Path

# ── Paths (relative to project root) ──
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR    = PROJECT_DIR / "data"
FIG_DIR     = PROJECT_DIR / "figures"

DFT_PATH = DATA_DIR / 'DFT_enhanced_256molecules_20250904_140333.csv'
BG_PATH  = DATA_DIR / 'StructureEmbeddingMany-TransformMorganFingerprints-WriteAllMorganFingerprints-ConcatCSV-2D-AK-AKEC-009.csv'
OUT_DIR  = FIG_DIR

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")
dft = pd.read_csv(DFT_PATH)
bg  = pd.read_csv(BG_PATH)
bg_s = bg.sample(80000, random_state=42)

# ── Journal visual settings (double-column) ──
TICK_SIZE   = 9
LABEL_SIZE  = 10
LEGEND_SIZE = 9
SPINE_LW    = 1.0
DPI         = 600

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         LABEL_SIZE,
    'axes.linewidth':    SPINE_LW,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.major.width': SPINE_LW,
    'ytick.major.width': SPINE_LW,
    'xtick.minor.width': SPINE_LW * 0.5,
    'ytick.minor.width': SPINE_LW * 0.5,
    'xtick.major.size':  4,
    'ytick.major.size':  4,
    'xtick.minor.size':  2,
    'ytick.minor.size':  2,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'axes.labelsize':    LABEL_SIZE,
    'xtick.labelsize':   TICK_SIZE,
    'ytick.labelsize':   TICK_SIZE,
    'legend.fontsize':   LEGEND_SIZE,
    'savefig.dpi':       DPI,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

DFT_COLOR = '#d95f02'
DFT_EDGE  = '#7f2700'
CEN_COLOR = '#1B9E77'       # distinct teal to separate centroids from reps
BINS      = 110

# ── Layout ──
fig = plt.figure(figsize=(7.0, 2.8))
gs  = fig.add_gridspec(1, 3, left=0.08, right=0.985, bottom=0.30, top=0.92, wspace=0.36)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

projections = [
    ('latent_1', 'latent_2', 'Latent dimension 1', 'Latent dimension 2'),
    ('latent_1', 'latent_3', 'Latent dimension 1', 'Latent dimension 3'),
    ('latent_2', 'latent_3', 'Latent dimension 2', 'Latent dimension 3'),
]

def get_range(col):
    return [-8.8, 10.5] if '1' in col else [-9.0, 9.0]

im_ref = None
for ax, (xc, yc, xl, yl) in zip(axes, projections):
    x_bg, y_bg = bg_s[xc].values, bg_s[yc].values
    x_dft, y_dft = dft[xc].values, dft[yc].values
    x_cen, y_cen = dft['centroid_'+xc].values, dft['centroid_'+yc].values
    xr, yr = get_range(xc), get_range(yc)

    H, xe, ye = np.histogram2d(x_bg, y_bg, bins=BINS, range=[xr, yr])
    H = gaussian_filter(H.T.astype(float), sigma=1.8)
    # Keep low-density tails to avoid an artificial soft boundary contour.
    # H[H < H.max() * 0.005] = np.nan

    im = ax.imshow(H, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   cmap='Blues', aspect='auto', vmin=0, vmax=np.nanpercentile(H, 96),
                   alpha=0.90, zorder=1)
    if im_ref is None: im_ref = im

    # Background scatter
    ax.scatter(x_bg, y_bg, s=1.8, c='#9ab3d4', alpha=0.12,
               linewidths=0, rasterized=True, zorder=2)
    # Cluster centroids
    ax.scatter(x_cen, y_cen, s=26, marker='+', c=CEN_COLOR,
               alpha=0.85, linewidths=1.0, zorder=3)
    # DFT representatives
    ax.scatter(x_dft, y_dft, s=24, c=DFT_COLOR, alpha=0.90,
               edgecolors=DFT_EDGE, linewidths=0.5, zorder=5)

    ax.set_xlim(xr)
    ax.set_ylim(yr)
    ax.set_xlabel(xl, fontsize=LABEL_SIZE, labelpad=4)
    ax.set_ylabel(yl, fontsize=LABEL_SIZE, labelpad=4)

    # 5 ticks on each axis, lowest and highest shown
    xticks = np.linspace(xr[0], xr[1], 5)
    yticks = np.linspace(yr[0], yr[1], 5)
    ax.set_xticks(np.round(xticks, 1))
    ax.set_yticks(np.round(yticks, 1))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    # Extra tick-label padding prevents lower-left x/y tick labels from colliding.
    ax.tick_params(axis='x', which='major', labelsize=TICK_SIZE,
                   width=SPINE_LW, length=4, pad=3)
    ax.tick_params(axis='y', which='major', labelsize=TICK_SIZE,
                   width=SPINE_LW, length=4, pad=3)
    ax.tick_params(axis='both', which='minor', width=SPINE_LW * 0.5,
                   length=2)

# ── Colorbar ──
cbar_ax = fig.add_axes([0.17, 0.14, 0.66, 0.03])
cb = fig.colorbar(im_ref, cax=cbar_ax, orientation='horizontal')
cb.set_label('Point density', fontsize=LEGEND_SIZE, labelpad=4)
cb.set_ticks([])
cb.ax.text(-0.01, 0.5, 'Low', transform=cb.ax.transAxes,
           ha='right', va='center', fontsize=LEGEND_SIZE)
cb.ax.text(1.01, 0.5, 'High', transform=cb.ax.transAxes,
           ha='left', va='center', fontsize=LEGEND_SIZE)

# ── Legend ──
legend_elements = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor='#9ab3d4',
           markeredgecolor='#8899bb', markeredgewidth=0.8, markersize=5,
           label='Chemical space'),
    Line2D([0], [0], marker='P', color='none', markerfacecolor=CEN_COLOR,
           markeredgecolor=CEN_COLOR, markersize=5,
           label='Cluster centroids (256)'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=DFT_COLOR,
           markeredgecolor=DFT_EDGE, markeredgewidth=0.8, markersize=5,
           label='Selected representatives (256)'),
]

fig.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.02),
    ncol=3,
    fontsize=LEGEND_SIZE,
    frameon=False,
    handlelength=2.0,
    columnspacing=1.5,
)

# Save
output_base = str(OUT_DIR / 'figure1_kmeans_latent_space_clean')
plt.savefig(f'{output_base}.pdf', dpi=DPI, bbox_inches='tight', format='pdf')
plt.savefig(f'{output_base}.png', dpi=DPI, bbox_inches='tight', format='png')

print(f"Saved: {output_base}")
