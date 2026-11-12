"""
fig_style.py — Shared figure style for all manuscript figures.
─────────────────────────────────────────────────────────────────
Import at the top of every plotting script:

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fig_style
    fig_style.apply()

Then use fig_style constants for figsize, font sizes, band colors, etc.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Target page width (NeuroImage / standard journal) ─────────────────────────
# NeuroImage: 1-col = 3.5", 1.5-col = 5.3", 2-col = 7.2" (text width)
COL1  = 3.50   # single column
COL15 = 5.30   # 1.5 column
COL2  = 7.20   # full text width — use for 3-column panel figures

# Standard figure heights (choose based on number of rows)
H1 = 2.6   # 1 row
H2 = 4.8   # 2 rows
H3 = 6.8   # 3 rows
H4 = 8.5   # 4 rows (rare)

# ── Font sizes ─────────────────────────────────────────────────────────────────
FS_PANEL  = 11   # panel letter (A, B, C …)
FS_TITLE  = 10   # subplot/column title (band name)
FS_LABEL  = 9    # axis labels (x, y)
FS_TICK   = 8    # tick labels
FS_LEGEND = 8    # legend text
FS_ANNOT  = 7    # in-panel annotations (ICC, p-values, Cohen's d)

# ── Line / marker sizes ────────────────────────────────────────────────────────
LW_MAIN   = 1.6   # main data lines
LW_THIN   = 0.5   # spine / grid lines
LW_TRACE  = 1.0   # secondary lines
MS_MAIN   = 5     # marker size (circle)

# ── Band colors (CANONICAL — use these everywhere) ─────────────────────────────
BAND_COLORS = {
    'broad':      '#1B4F8A',
    'neurogenic': '#8B1A8B',
    'vasogenic':  '#C45E00',
}

BAND_LABELS = {
    'broad':      'Broad\n0.01–0.10 Hz',
    'neurogenic': 'Neurogenic\n0.02–0.04 Hz',
    'vasogenic':  'Vasogenic\n0.06–0.10 Hz',
}

BAND_ORDER = ['broad', 'neurogenic', 'vasogenic']

BAND_RANGES = {
    'broad':      (0.01, 0.10),
    'neurogenic': (0.02, 0.04),
    'vasogenic':  (0.06, 0.10),
}

BAND_MAX_LAG = {
    'broad':      15.0,
    'neurogenic': 20.0,
    'vasogenic':  10.0,
}

# ── ROI colors and style ───────────────────────────────────────────────────────
ROI_ORDER = ['z0', 'z1', 'z2', 'z3', 'fv-lower', 'fv-upper']

ROI_COLORS = {
    'z0':       '#D32F2F',
    'z1':       '#F57C00',
    'z2':       '#388E3C',
    'z3':       '#0288D1',
    'fv-lower': '#6A1B9A',
    'fv-upper': '#00796B',
}

ROI_LS = {
    'z0': '-', 'z1': '-',
    'z2': '--', 'z3': '--',
    'fv-lower': '-.', 'fv-upper': '-.',
}

ROI_LW = {
    'z0': 1.8, 'z1': 1.8,
    'z2': 1.4, 'z3': 1.4,
    'fv-lower': 1.2, 'fv-upper': 1.2,
}

ROI_LABELS = {
    'z0': 'z0', 'z1': 'z1', 'z2': 'z2', 'z3': 'z3',
    'fv-lower': 'fv-lower', 'fv-upper': 'fv-upper',
}

# AAFT figures use a slightly different key format (underscores)
ROI_ORDER_AAFT  = ['z0', 'z1', 'z2', 'z3', 'fv_lower', 'fv_upper']
ROI_LABELS_AAFT = {
    'z0': 'Slice z0', 'z1': 'Slice z1',
    'z2': 'Slice z2', 'z3': 'Slice z3',
    'fv_lower': '4V lower', 'fv_upper': '4V upper',
}

# ── Nuisance palette (Fig 4) ───────────────────────────────────────────────────
# Muted, ordered: motion = neutral steel, WM = warm rose, aCompCor = teal
NUISANCE_COLORS = {
    'r2_motion':        '#5B7FA6',   # steel blue — baseline regressor
    'delta_r2_wm':      '#B85C5C',   # muted rose — WM mean
    'delta_r2_acompcor':'#4A9B8E',   # teal — aCompCor
}
NUISANCE_LABELS = {
    'r2_motion':        'Motion $R^2$',
    'delta_r2_wm':      'WM $\\Delta R^2$',
    'delta_r2_acompcor':'aCompCor $\\Delta R^2$',
}

# ── Pipeline palette (Fig 5) ───────────────────────────────────────────────────
# P0 = light neutral (baseline, not highlighted)
# P1–P3 = sequential single-hue from light to saturated blue
PIPELINE_COLORS = {
    'P0': '#BDBDBD',   # light grey — no regression baseline
    'P1': '#90B8D4',   # light blue — motion only
    'P2': '#3D7FB5',   # mid blue — motion + WM
    'P3': '#1A4E80',   # dark blue — motion + aCompCor
}
PIPELINE_LABELS = {
    'P0': 'P0: no regression',
    'P1': 'P1: motion',
    'P2': 'P2: motion + WM mean',
    'P3': 'P3: motion + aCompCor',
}
PIPELINE_ORDER = ['P0', 'P1', 'P2', 'P3']

# ── GridSpec defaults ──────────────────────────────────────────────────────────
GS_LEFT   = 0.10
GS_RIGHT  = 0.97
GS_TOP    = 0.93
GS_BOTTOM = 0.10

# ── Helpers ────────────────────────────────────────────────────────────────────

def apply():
    """Apply shared rcParams. Call once at top of every script."""
    mpl.rcParams.update({
        # Font
        'font.family':          'sans-serif',
        'font.sans-serif':      ['Arial', 'Helvetica Neue', 'Helvetica',
                                 'DejaVu Sans'],
        'font.size':            FS_TICK,
        'axes.titlesize':       FS_TITLE,
        'axes.labelsize':       FS_LABEL,
        'xtick.labelsize':      FS_TICK,
        'ytick.labelsize':      FS_TICK,
        'legend.fontsize':      FS_LEGEND,
        # Lines
        'lines.linewidth':      LW_MAIN,
        'axes.linewidth':       LW_THIN,
        'xtick.major.width':    LW_THIN,
        'ytick.major.width':    LW_THIN,
        'xtick.major.size':     2.5,
        'ytick.major.size':     2.5,
        # Figure
        'figure.facecolor':     'white',
        'figure.dpi':           150,
        'savefig.dpi':          600,
        'savefig.bbox':         'tight',
        'savefig.pad_inches':   0.02,
        'savefig.facecolor':    'white',
        # Axes
        'axes.spines.top':      False,
        'axes.spines.right':    False,
        'axes.facecolor':       'white',
        # PDF text as real text (not paths) — important for journal submission
        'pdf.fonttype':         42,
        'svg.fonttype':         'none',
    })


def clean_ax(ax):
    """Standard axis cleanup — call on every subplot."""
    ax.set_facecolor('white')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_linewidth(LW_THIN)
    ax.spines['bottom'].set_linewidth(LW_THIN)
    ax.tick_params(labelsize=FS_TICK, length=3, width=LW_THIN)
    ax.yaxis.grid(True, color='#EBEBEB', lw=0.4, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, letter, x=-0.18, y=1.15):
    """Add bold panel letter (A, B, C …) in axes coordinates."""
    ax.text(x, y, letter,
            transform=ax.transAxes,
            fontsize=FS_PANEL + 1, fontweight='bold', va='top')


def roi_to_filename(roi):
    """Convert ROI key to filename-safe form: 'fv-lower' → 'fv_lower'."""
    return roi.replace("-", "_")


def save(fig, stem, out_dir):
    """
    Save figure as both PDF and PNG at 600 dpi.

    Parameters
    ----------
    fig     : matplotlib Figure
    stem    : filename stem without extension, e.g. 'Fig3_spatial_gradient'
    out_dir : output directory (must exist)
    """
    import os
    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'{stem}.{ext}')
        fig.savefig(path, dpi=600, bbox_inches='tight',
                    facecolor='white', pad_inches=0.02)
        print(f'  ✓ {path}')
    plt.close(fig)
