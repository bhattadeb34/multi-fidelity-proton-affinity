"""
Shared journal plotting style for manuscript figures.

Targets:
- minimum readable text around 9 pt
- single-column width ~3.4 in
- double-column width ~7.0 in
"""

import matplotlib.pyplot as plt

# Journal typography
TICK_SIZE = 9
LABEL_SIZE = 10
LEGEND_SIZE = 9
ANNOT_SIZE = 9
PANEL_SIZE = 11
TITLE_SIZE = 10

# Journal linework (avoid heavy poster-like styling)
SPINE_LW = 1.0
MAJOR_TICK_LEN = 3.5
MINOR_TICK_LEN = 2.0

# Common figure widths
SINGLE_COL_W = 3.4
DOUBLE_COL_W = 7.0

# Export settings
DPI = 600


def apply_journal_style(*, show_top_right_spines: bool = False):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": TICK_SIZE,
        "axes.linewidth": SPINE_LW,
        "axes.spines.top": show_top_right_spines,
        "axes.spines.right": show_top_right_spines,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "xtick.major.width": SPINE_LW,
        "ytick.major.width": SPINE_LW,
        "xtick.minor.width": SPINE_LW * 0.7,
        "ytick.minor.width": SPINE_LW * 0.7,
        "xtick.major.size": MAJOR_TICK_LEN,
        "ytick.major.size": MAJOR_TICK_LEN,
        "xtick.minor.size": MINOR_TICK_LEN,
        "ytick.minor.size": MINOR_TICK_LEN,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": LEGEND_SIZE,
        "figure.dpi": 300,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
