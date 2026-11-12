"""
figures_g2.py — Group 2 (25/26-subject) unified figure runner
════════════════════════════════════════════════════════════════════════════════
Generates all figures derived from the 25/26-subject HCP-YA subsample.

Figures
───────
  Fig S4  Window surrogate vs observed (4 windows, z0, broad band)
  Fig S8  ICC hierarchy (within-run → split-half → cross-run, z0)
           Also exports: TableS_icc_hierarchy.csv / .tex

Usage
─────
  python code/figures_g2.py --all
  python code/figures_g2.py --fig S4 S8

Global options
──────────────
  --root   ROOT     Dataset root (default: /Volumes/BrainLab01/Datasets/BALSA/HCP-YA-gBOLD-CSF)
  --outdir DIR      Output directory (default: <root>/results/figures)

Fig S4 extra options
─────────────────────
  --windows CSV     surrogate_windows.csv
                    (default: <root>/results/surrogate/surrogate_windows.csv)
  --roi    ROI      CSF ROI for S4 (default: z0)
  --mock           Generate synthetic data for preview (no real data needed)

Fig S8 extra options
─────────────────────
  --ca_dir DIR      coupling_analysis directory
                    (default: <root>/results/coupling_analysis_26)
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import fig_style
fig_style.apply()

TR     = 0.72
N_VOLS = 1190


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ci95(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    n    = len(vals)
    if n < 2:
        return 0.0
    return float(stats.sem(vals) * stats.t.ppf(0.975, n - 1))


def _resolve_out(args):
    out = args.outdir or os.path.join(args.root, "results/figures")
    os.makedirs(out, exist_ok=True)
    return out


def _find_csv_prefix(directory, prefix):
    matches = sorted(f for f in os.listdir(directory)
                     if f.startswith(prefix) and f.endswith(".csv"))
    if not matches:
        raise FileNotFoundError(f"No CSV starting with '{prefix}' in {directory}")
    return os.path.join(directory, matches[0])


def _fmt_p(p):
    if p < 0.0001: return "p < 0.0001"
    elif p < 0.001: return f"p = {p:.4f}"
    elif p < 0.01:  return f"p = {p:.3f}"
    return f"p = {p:.2f}"


# ══════════════════════════════════════════════════════════════════════════════
# Fig S4 — Window surrogate vs observed
# Source: figS_surrogate_windows.py
# BUG FIXES:
#   • stem renamed FigS_surrogate_windows → FigS4_surrogate_windows
#   • error message no longer references coupling_subjects_104.csv
#     (this is a 25/26-subject Group 2 figure)
# ══════════════════════════════════════════════════════════════════════════════

def _mock_data():
    rng  = np.random.default_rng(42)
    rows = []
    for w in range(1, 5):
        base_obs  = 0.30 + w * 0.07
        base_null = 0.15 + w * 0.03
        for sub in range(26):
            obs  = abs(rng.normal(base_obs,  0.08))
            null_m = abs(rng.normal(base_null, 0.05))
            null_s = abs(rng.normal(0.04, 0.01))
            rows.append(dict(
                window=w, subject=f"sub_{sub:03d}",
                band="broad", roi="z0",
                obs_absr=obs, null_mean=null_m, null_sd=null_s,
            ))
    return pd.DataFrame(rows)


def _violin_half(ax, data, x_center, side, color, alpha=0.45, width=0.35):
    """Half-violin for paired null | observed plots."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 3:
        return
    from scipy.stats import gaussian_kde
    kd   = gaussian_kde(data, bw_method=0.25)
    ymin = max(0, data.min() - 0.05)
    ymax = data.max() + 0.05
    ygrid = np.linspace(ymin, ymax, 300)
    density = kd(ygrid)
    density = density / density.max() * width
    if side == "left":
        ax.fill_betweenx(ygrid, x_center - density, x_center, alpha=alpha, color=color)
        ax.plot(x_center - density, ygrid, lw=0.8, color=color, alpha=0.75)
    else:
        ax.fill_betweenx(ygrid, x_center, x_center + density, alpha=alpha, color=color)
        ax.plot(x_center + density, ygrid, lw=0.8, color=color, alpha=0.75)
    median = float(np.median(data))
    ax.hlines(median, x_center - (width if side == "left" else 0),
              x_center + (0 if side == "left" else width),
              colors=color, linewidths=1.6, zorder=5)


NULL_COLOR = "#BBBBBB"   # grey for AAFT surrogate null across all bands


def _full_violin(ax, data, x_center, color, alpha=0.55, width=0.38):
    """Full violin with white median line and subject scatter dots."""
    from scipy.stats import gaussian_kde
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 3:
        return
    kd      = gaussian_kde(data, bw_method=0.20)
    ymin    = max(0.0, data.min() - 0.02)
    ymax    = min(1.02, data.max() + 0.02)
    ygrid   = np.linspace(ymin, ymax, 300)
    density = kd(ygrid)
    density = density / density.max() * width
    ax.fill_betweenx(ygrid, x_center - density, x_center + density,
                     alpha=alpha, color=color, linewidth=0, zorder=3)
    ax.plot(x_center - density, ygrid, lw=0.6, color=color, alpha=0.75, zorder=3)
    ax.plot(x_center + density, ygrid, lw=0.6, color=color, alpha=0.75, zorder=3)
    median = float(np.median(data))
    hw = float(density[np.argmin(np.abs(ygrid - median))]) * 0.85
    ax.hlines(median, x_center - hw, x_center + hw,
              colors="white", linewidths=2.0, zorder=5)


def _make_s4_figure(df, out, roi="z0"):
    """
    Original style: 1 row × 3 band columns.
    Each column: 4 windows; each window has paired violins — grey null (left) + coloured observed (right).
    Stats annotation (Δ|r|, p) in 2×2 grid at top of each panel.
    Legend at bottom. Y-axis: Coupling |r| (ROI: z0).
    """
    from scipy.stats import ttest_rel

    windows    = sorted(df["window"].unique())
    band_order = fig_style.BAND_ORDER
    rng_jit    = np.random.default_rng(0)

    # Layout constants
    WIN_SPACING = 1.0      # centre-to-centre between windows
    NULL_OFFSET = -0.22    # null violin left of window centre
    OBS_OFFSET  = +0.22    # observed violin right of window centre
    VIO_WIDTH   = 0.19     # half-width of each violin

    fig = plt.figure(figsize=(fig_style.COL2, fig_style.H2 - 1.2), facecolor="white")
    gs  = gridspec.GridSpec(1, len(band_order), figure=fig,
                             hspace=0.0, wspace=0.28,
                             left=0.09, right=0.97,
                             top=0.82, bottom=0.20)

    for bi, band in enumerate(band_order):
        ax   = fig.add_subplot(gs[0, bi])
        fig_style.clean_ax(ax)
        bcol = fig_style.BAND_COLORS[band]
        bname, bhz = fig_style.BAND_LABELS[band].split("\n")
        bdf  = df[(df["band"] == band) & (df["roi"] == roi)]

        win_centres = []
        stat_lines  = []          # collect per-window stat strings for top annotation
        y_all       = []

        for wi, w in enumerate(windows):
            wdf      = bdf[bdf["window"] == w]
            obs_vals = wdf["obs_absr"].dropna().values
            null_m   = wdf["null_mean"].dropna().values
            null_sd  = wdf["null_sd"].dropna().values

            x_ctr = wi * WIN_SPACING
            win_centres.append(x_ctr)

            # Use null_mean directly — these are the actual per-subject surrogate |r| means.
            # Avoids reconstructing Gaussian pools which produce phantom tails beyond real range.
            null_pool = null_m[np.isfinite(null_m)]

            # Grey null violin (left)
            _full_violin(ax, null_pool, x_ctr + NULL_OFFSET, NULL_COLOR, alpha=0.65, width=VIO_WIDTH)
            # Coloured observed violin (right)
            _full_violin(ax, obs_vals,  x_ctr + OBS_OFFSET,  bcol,       alpha=0.70, width=VIO_WIDTH)

            # Scatter dots — same data as violins
            jn = rng_jit.uniform(-VIO_WIDTH * 0.6, VIO_WIDTH * 0.6, len(null_pool))
            ax.scatter(x_ctr + NULL_OFFSET + jn, null_pool,
                       s=4, color=NULL_COLOR, alpha=0.70, zorder=4, linewidths=0)
            jo = rng_jit.uniform(-VIO_WIDTH * 0.6, VIO_WIDTH * 0.6, len(obs_vals))
            ax.scatter(x_ctr + OBS_OFFSET + jo, obs_vals,
                       s=4, color=bcol, alpha=0.75, zorder=4, linewidths=0)

            # Per-window stats (paired: obs vs null_mean per subject)
            if len(obs_vals) > 1 and len(null_m) > 1:
                n    = min(len(obs_vals), len(null_m))
                diff = obs_vals[:n] - null_m[:n]
                t, p = ttest_rel(obs_vals[:n], null_m[:n])
                p_one = p / 2 if t > 0 else 1.0 - p / 2
                delta = float(np.mean(diff))
                p_str = "p<.0001" if p_one < 0.0001 else f"p={p_one:.4f}".rstrip("0")
                stat_lines.append(f"W{w}: Δ|r|={delta:.2f},  {p_str}")

            y_all.extend(null_pool.tolist())
            y_all.extend(obs_vals.tolist())

        # Axes formatting
        ax.set_xticks(win_centres)
        ax.set_xticklabels([f"Win {w}" for w in windows], fontsize=fig_style.FS_TICK)
        ax.set_xlabel(f"Time window (~3.6 min)", fontsize=fig_style.FS_LABEL)
        ax.set_xlim(win_centres[0] - 0.55, win_centres[-1] + 0.55)
        y_finite = [y for y in y_all if np.isfinite(y) and y >= 0]
        ax.set_ylim(0, min(1.02, np.percentile(y_finite, 99.5) * 1.06) if y_finite else 1.02)
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
        ax.tick_params(axis="y", labelsize=fig_style.FS_TICK)
        if bi == 0:
            ax.set_ylabel(f"Coupling |r|\n(ROI: {roi})", fontsize=fig_style.FS_LABEL)

        # Stats annotation — 2×2 grid above plot (no band title; described in caption)
        for si, stxt in enumerate(stat_lines):
            col_i = si % 2; row_i = si // 2
            ax.text(0.02 + col_i * 0.50, 1.10 - row_i * 0.06, stxt,
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=fig_style.FS_ANNOT - 1, color=bcol, fontweight="normal")

    # Shared legend at bottom
    null_patch = mpatches.Patch(facecolor=NULL_COLOR, alpha=0.80, label="AAFT surrogate null")
    obs_handles = [
        mpatches.Patch(facecolor=fig_style.BAND_COLORS[b], alpha=0.80,
                       label=f"Observed ({b})")
        for b in band_order
    ]
    fig.legend(handles=[null_patch] + obs_handles,
               loc="lower center", ncol=4,
               fontsize=fig_style.FS_LEGEND + 0.5, frameon=False,
               bbox_to_anchor=(0.54, 0.02))

    fig_style.save(fig, "FigS4_surrogate_windows", out)
    print(f"  ✓  FigS4_surrogate_windows  →  {out}")


def figS4_surrogate_windows(args):
    print("\n── Fig S4: Window surrogate vs observed ─────────────────────")
    out = _resolve_out(args)
    roi = getattr(args, "roi", "z0") or "z0"

    if getattr(args, "mock", False):
        print("  Using synthetic (mock) data for preview")
        df = _mock_data()
        _make_s4_figure(df, out, roi=roi)
        return

    windows_csv = getattr(args, "windows", None) or \
                  os.path.join(args.root, "results/surrogate/surrogate_windows.csv")

    if not os.path.exists(windows_csv):
        # BUG FIXED: error message no longer references coupling_subjects_104.csv
        print(f"  ERROR: surrogate_windows.csv not found at:\n    {windows_csv}")
        print("  For Group 2 (25/26-subject subsample), ensure the surrogate_windows.csv")
        print("  from the Group 2 analysis is present, or run with --mock for preview.")
        return

    df = pd.read_csv(windows_csv)
    if "roi" not in df.columns:
        df["roi"] = roi
    if "band" not in df.columns:
        df["band"] = "broad"

    # Optionally merge time-window coupling data for obs_absr
    win_coupling = os.path.join(args.root, f"results/{roi}_timewindow_coupling.csv")
    if "obs_absr" not in df.columns and os.path.exists(win_coupling):
        wc = pd.read_csv(win_coupling)
        merge_on = [c for c in ["subject", "band", "roi", "window"] if c in df.columns and c in wc.columns]
        df = df.merge(wc, on=merge_on, how="left", suffixes=("", "_wc"))
        if "obs_absr_wc" in df.columns and "obs_absr" not in df.columns:
            df["obs_absr"] = df["obs_absr_wc"]

    if "obs_absr" not in df.columns:
        print("  ERROR: 'obs_absr' column missing from surrogate_windows.csv")
        print("  Expected columns: window, band, roi, obs_absr, null_mean, null_sd")
        return

    _make_s4_figure(df, out, roi=roi)


# ══════════════════════════════════════════════════════════════════════════════
# Fig S8 — ICC hierarchy
# Source: figS_icc_hierarchy_full.py
# BUG FIX: stem renamed FigS_icc_hierarchy_full → FigS8_icc_hierarchy
# ══════════════════════════════════════════════════════════════════════════════

# ICC(2,1) for 2 raters — Shrout & Fleiss 1979
def _icc21(x, y):
    mat  = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    mask = np.all(np.isfinite(mat), axis=1)
    mat  = mat[mask]
    n, k = mat.shape
    if n < 4:
        return np.nan, np.nan, np.nan
    grand     = mat.mean()
    subj_mean = mat.mean(axis=1)
    rate_mean = mat.mean(axis=0)
    SSs = k * np.sum((subj_mean - grand) ** 2)
    SSr = n * np.sum((rate_mean - grand) ** 2)
    SSe = np.sum((mat - grand) ** 2) - SSs - SSr
    MSs = SSs / (n - 1)
    MSe = SSe / ((n - 1) * (k - 1))
    MSr = SSr / (k - 1)
    icc = (MSs - MSe) / (MSs + (k - 1) * MSe + k * (MSr - MSe) / n)
    icc = float(np.clip(icc, -1, 1))
    alpha = 0.05
    F    = MSs / MSe
    df1, df2 = n - 1, (n - 1) * (k - 1)
    FL   = F / stats.f.ppf(1 - alpha / 2, df1, df2)
    FU   = F * stats.f.ppf(1 - alpha / 2, df2, df1)
    ci_l = (FL - 1) / (FL + k - 1)
    ci_h = (FU - 1) / (FU + k - 1)
    return icc, float(ci_l), float(ci_h)


def _load_run_csv(ca_dir, prefix, run):
    """Load a per-run CSV, e.g. stage3_splithalf_26s_LR.csv"""
    # Try exact pattern: prefix_26s_RUN.csv
    for tag in [f"26s_{run}", run]:
        fname = f"{prefix}_{tag}.csv"
        path  = os.path.join(ca_dir, fname)
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


def _filter(df, band, roi):
    if df is None:
        return None
    if "band" in df.columns:
        df = df[df["band"] == band]
    if "roi" in df.columns:
        df = df[df["roi"] == roi]
    return df


ORDERED_LABELS   = ["Win1 — Win2", "Win2 — Win3", "Win3 — Win4",
                    "Half1 — Half2",
                    "LR — RL1", "LR1 — LR2", "LR1 — RL2"]
SECTION_LABELS   = ["Within-run\n(3.6-min windows)", "Split-half\n(7.2 min)", "Cross-run\n(≥ 1 hr)"]
SECTION_COLORS   = ["#DBEAFE", "#D1FAE5", "#FEE2E2"]
SECTION_X_RANGES = [(-0.5, 2.5), (2.5, 3.5), (3.5, 6.5)]
DIVIDERS         = [2.5, 3.5]
XPOS             = np.arange(7)


def _compute_icc_table(ca_dir, roi="z0"):
    """
    7 ICC values per band for the specified ROI:
      W1-W2, W1-W3, W2-W3   — within-run windows  (from stage3_timewindow_26s_<run>.csv)
      H1-H2                  — split-half           (from stage3_splithalf_26s_<run>.csv, run=LR)
      LR-RL, LR-LR2, RL-RL2 — cross-run            (pairing stage3_coupling_26s_<run>.csv files)
    """
    rows = []
    for band in fig_style.BAND_ORDER:

        # ── Within-run windows (consecutive pairs) ──────────────────────────
        win_pairs = [("Win1 — Win2", 1, 2), ("Win2 — Win3", 2, 3), ("Win3 — Win4", 3, 4)]
        for lbl, w1, w2 in win_pairs:
            icc_v = ci_l = ci_h = np.nan
            for run in ["LR", "LR2", "RL", "RL2"]:
                wdf = _load_run_csv(ca_dir, "stage3_timewindow", run)
                wdf = _filter(wdf, band, roi)
                if wdf is None or wdf.empty:
                    continue
                win_col = next((c for c in ["window", "win", "half"] if c in wdf.columns), None)
                if win_col is None:
                    continue
                x = wdf[wdf[win_col] == w1]["abs_r"].dropna().values
                y = wdf[wdf[win_col] == w2]["abs_r"].dropna().values
                n = min(len(x), len(y))
                if n >= 4:
                    icc_v, ci_l, ci_h = _icc21(x[:n], y[:n])
                    break   # got data from this run, done
            rows.append(dict(band=band, label=lbl, icc=icc_v, ci_l=ci_l, ci_h=ci_h,
                             section="within"))

        # ── Split-half (LR primary, fallback to others) ─────────────────────
        icc_v = ci_l = ci_h = np.nan
        for run in ["LR", "LR2", "RL", "RL2"]:
            shdf = _load_run_csv(ca_dir, "stage3_splithalf", run)
            shdf = _filter(shdf, band, roi)
            if shdf is None or shdf.empty:
                continue
            half_col = next((c for c in ["half", "window"] if c in shdf.columns), None)
            if half_col is None:
                continue
            sub_col = next((c for c in ["subject", "sub"] if c in shdf.columns), "subject")
            h1 = shdf[shdf[half_col] == 1].set_index(sub_col)["abs_r"]
            h2 = shdf[shdf[half_col] == 2].set_index(sub_col)["abs_r"]
            shared = h1.index.intersection(h2.index)
            if len(shared) >= 4:
                icc_v, ci_l, ci_h = _icc21(h1[shared].values, h2[shared].values)
                break
        rows.append(dict(band=band, label="Half1 — Half2", icc=icc_v, ci_l=ci_l, ci_h=ci_h,
                         section="splithalf"))

        # ── Cross-run (pair stage3_coupling files by subject) ───────────────
        def _load_coupling(run):
            df = _load_run_csv(ca_dir, "stage3_coupling", run)
            return _filter(df, band, roi)

        cr_pairs = [("LR — RL1", "LR", "RL"), ("LR1 — LR2", "LR", "LR2"), ("LR1 — RL2", "LR2", "RL2")]
        for lbl, r1, r2 in cr_pairs:
            icc_v = ci_l = ci_h = np.nan
            d1 = _load_coupling(r1)
            d2 = _load_coupling(r2)
            if d1 is not None and d2 is not None and not d1.empty and not d2.empty:
                sub_col = next((c for c in ["subject", "sub"] if c in d1.columns), None)
                if sub_col:
                    d1i = d1.set_index(sub_col)["abs_r"].dropna()
                    d2i = d2.set_index(sub_col)["abs_r"].dropna()
                    shared = d1i.index.intersection(d2i.index)
                    if len(shared) >= 4:
                        icc_v, ci_l, ci_h = _icc21(d1i[shared].values, d2i[shared].values)
            rows.append(dict(band=band, label=lbl, icc=icc_v, ci_l=ci_l, ci_h=ci_h,
                             section="crossrun"))

    return pd.DataFrame(rows)


def _export_icc_table(icc_df, out):
    """Export ICC hierarchy as CSV and LaTeX."""
    pivot = icc_df.pivot_table(index="label", columns="band", values="icc")
    ordered_labels = ORDERED_LABELS
    pivot = pivot.reindex([l for l in ordered_labels if l in pivot.index])
    csv_path = os.path.join(out, "TableS_icc_hierarchy.csv")
    pivot.to_csv(csv_path)
    print(f"  ✓  TableS_icc_hierarchy.csv  →  {out}")

    tex_path = os.path.join(out, "TableS_icc_hierarchy.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{ICC(2,1) hierarchy for CSF z0 coupling |r|}\n")
        f.write("\\label{tab:icc_hierarchy}\n")
        col_spec = "l" + "c" * len(fig_style.BAND_ORDER)
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")
        col_header = " & ".join(["Comparison"] + [b.capitalize() for b in fig_style.BAND_ORDER])
        f.write(col_header + " \\\\\n\\midrule\n")
        section_names = {"within": "Within-run windows", "splithalf": "Split-half",
                         "crossrun": "Cross-run"}
        seen_sections = set()
        for lbl in ordered_labels:
            row = icc_df[icc_df["label"] == lbl]
            if row.empty:
                continue
            section = row["section"].iloc[0]
            if section not in seen_sections:
                f.write(f"\\multicolumn{{{1 + len(fig_style.BAND_ORDER)}}}{{l}}"
                        f"{{\\textit{{{section_names.get(section, section)}}}}}\\\\\n")
                seen_sections.add(section)
            vals = []
            for band in fig_style.BAND_ORDER:
                brow = row[row["band"] == band]
                if brow.empty or not np.isfinite(brow["icc"].iloc[0]):
                    vals.append("--")
                else:
                    icc_v = float(brow["icc"].iloc[0])
                    ci_l  = float(brow["ci_l"].iloc[0])
                    ci_h  = float(brow["ci_h"].iloc[0])
                    vals.append(f"{icc_v:.2f} [{ci_l:.2f}, {ci_h:.2f}]")
            f.write(f"  {lbl} & " + " & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  ✓  TableS_icc_hierarchy.tex  →  {out}")


def _make_s8_figure(icc_df, out):
    """
    3 rows × 1 col: one band per row.
    7 ICC values connected by a line, ordered by temporal separation:
      Within-run (Win1-Win2, Win2-Win3, Win3-Win4) | Split-half (Half1-Half2) |
      Cross-run (LR-RL1, LR1-LR2, LR1-RL2)
    Matches original figure style: connected line + filled circles + capped error bars +
    bold section headers with duration labels inside shaded regions.
    """
    fig = plt.figure(figsize=(fig_style.COL2, fig_style.H3), facecolor="white")
    gs  = gridspec.GridSpec(len(fig_style.BAND_ORDER), 1, figure=fig,
                             hspace=0.18, wspace=0.0,
                             left=0.12, right=0.97,
                             top=0.90, bottom=0.14)

    for bi, band in enumerate(fig_style.BAND_ORDER):
        ax   = fig.add_subplot(gs[bi, 0])
        fig_style.clean_ax(ax)
        bcol = fig_style.BAND_COLORS[band]
        bname, bhz = fig_style.BAND_LABELS[band].split("\n")
        bdf  = icc_df[icc_df["band"] == band]
        is_last = (bi == len(fig_style.BAND_ORDER) - 1)

        # Section shading
        for (x0, x1), sc in zip(SECTION_X_RANGES, SECTION_COLORS):
            ax.axvspan(x0, x1, color=sc, alpha=0.50, zorder=0)

        # Dividers
        for xd in DIVIDERS:
            ax.axvline(xd, color="#aaaaaa", lw=0.9, ls="--", zorder=1)

        # Reference lines
        ax.axhline(0,    color="#CCCCCC", lw=0.6, zorder=1)
        ax.axhline(0.75, color="#888888", lw=0.9, ls="--", zorder=2)

        # Collect points for the connecting line
        xs_line = []; ys_line = []
        for xi, lbl in enumerate(ORDERED_LABELS):
            row = bdf[bdf["label"] == lbl]
            if row.empty or not np.isfinite(float(row["icc"].iloc[0])):
                xs_line.append(xi); ys_line.append(np.nan)
                continue
            icc_v = float(row["icc"].iloc[0])
            ci_l  = float(row["ci_l"].iloc[0])
            ci_h  = float(row["ci_h"].iloc[0])
            xs_line.append(xi); ys_line.append(icc_v)
            ylo = max(icc_v - ci_l, 0)
            yhi = max(ci_h - icc_v, 0)
            ax.errorbar(xi, icc_v, yerr=[[ylo], [yhi]],
                        fmt="none", color=bcol,
                        lw=1.6, capsize=5, capthick=1.5, zorder=4)

        # Draw connecting line first, then dots on top
        ax.plot(xs_line, ys_line, color=bcol, lw=1.8, zorder=3, alpha=0.85)
        for xi, yv in enumerate(ys_line):
            if np.isfinite(yv):
                ax.scatter(xi, yv, s=52, color=bcol, zorder=5,
                           edgecolors="white", linewidths=1.0)

        # Section header labels — top row only
        if bi == 0:
            for (x0, x1), slbl in zip(SECTION_X_RANGES, SECTION_LABELS):
                ax.text((x0 + x1) / 2, 1.04, slbl,
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom",
                        fontsize=fig_style.FS_ANNOT + 0.5,
                        color="#333333", fontweight="bold")

        ax.set_xlim(-0.6, 6.6)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
        ax.tick_params(axis="y", labelsize=fig_style.FS_TICK)
        ax.set_ylabel("ICC(2,1)", fontsize=fig_style.FS_LABEL)
        ax.text(6.62, 0.755, "0.75", fontsize=fig_style.FS_ANNOT,
                va="bottom", ha="left", color="#777777")

        # X-axis tick labels — bottom row only
        ax.set_xticks(XPOS)
        if is_last:
            ax.set_xticklabels(ORDERED_LABELS, fontsize=fig_style.FS_TICK - 0.5,
                               rotation=15, ha="right")
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)

        # Band label left-aligned next to panel letter
        panel_letter = ["A", "B", "C"][bi]
        ax.text(-0.10, 1.02, panel_letter,
                transform=ax.transAxes,
                fontsize=fig_style.FS_PANEL, fontweight="bold", va="bottom", ha="right")
        ax.text(-0.09, 1.02, f"{bname}  {bhz}",
                transform=ax.transAxes,
                fontsize=fig_style.FS_TITLE, fontweight="bold",
                color=bcol, va="bottom", ha="left")

    fig.text(0.54, 0.005,
             "Temporal separation (increasing →)",
             ha="center", va="bottom",
             fontsize=fig_style.FS_LABEL, color="#555555")

    fig_style.save(fig, "FigS8_icc_hierarchy", out)
    print(f"  ✓  FigS8_icc_hierarchy  →  {out}")


def figS8_icc_hierarchy(args):
    print("\n── Fig S8: ICC hierarchy ────────────────────────────────────")
    out    = _resolve_out(args)
    ca_dir = getattr(args, "ca_dir", None) or \
             os.path.join(args.root, "results/coupling_g2")

    if not os.path.isdir(ca_dir):
        print(f"  ERROR: coupling_analysis_26 directory not found at:\n    {ca_dir}")
        print("  Set --ca_dir to the correct path.")
        return

    print(f"  Using ca_dir: {ca_dir}")
    icc_df = _compute_icc_table(ca_dir, roi="z0")

    _export_icc_table(icc_df, out)
    _make_s8_figure(icc_df, out)


# ══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ══════════════════════════════════════════════════════════════════════════════

FIG_MAP = {
    "s4":  figS4_surrogate_windows,
    "s8":  figS8_icc_hierarchy,
    # aliases
    "surrogate": figS4_surrogate_windows,
    "windows":   figS4_surrogate_windows,
    "icc":       figS8_icc_hierarchy,
    "hierarchy": figS8_icc_hierarchy,
}

ALL_KEYS = ["s4", "s8"]


def main():
    ap = argparse.ArgumentParser(
        description="Group 2 (25/26-subject) unified figure runner — NeuroImage gBOLD–CSF paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python code/figures_g2.py --all\n"
            "  python code/figures_g2.py --fig S4 --mock\n"
            "  python code/figures_g2.py --fig S8 --ca_dir results/coupling_analysis_26\n"
        )
    )
    ap.add_argument("--all",    action="store_true", help="Generate all Group 2 figures")
    ap.add_argument("--fig",    nargs="+", metavar="FIG",
                    help="Figure(s) to generate: S4 S8")
    ap.add_argument("--root",   default="/Volumes/BrainLab01/Datasets/BALSA/HCP-YA-gBOLD-CSF",
                    help="Dataset root directory")
    ap.add_argument("--outdir", default=None, help="Output directory")
    # S4 extra
    ap.add_argument("--windows", default=None, help="surrogate_windows.csv path")
    ap.add_argument("--roi",     default="z0", help="CSF ROI for Fig S4 (default: z0)")
    ap.add_argument("--mock",    action="store_true",
                    help="Use synthetic data for preview (Fig S4)")
    # S8 extra
    ap.add_argument("--ca_dir",  default=None,
                    help="Group 2 coupling_analysis directory (default: <root>/results/coupling_analysis_26)")
    args = ap.parse_args()

    if not args.all and not args.fig:
        ap.print_help(); sys.exit(0)

    targets = ALL_KEYS if (args.all or (args.fig and any(k.lower() == "all" for k in args.fig))) else args.fig

    for key in targets:
        if key.lower() == "all":
            continue
        fn = FIG_MAP.get(key.lower())
        if fn is None:
            print(f"  ⚠  Unknown figure key '{key}' — valid: {list(FIG_MAP.keys())}"); continue
        try:
            fn(args)
        except Exception as e:
            import traceback
            print(f"  ERROR in fig '{key}': {e}")
            traceback.print_exc()

    print("\nAll done.")


if __name__ == "__main__":
    main()
