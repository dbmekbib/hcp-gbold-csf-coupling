"""
figures_g1.py — Group 1 (104-subject) unified figure runner
════════════════════════════════════════════════════════════════════════════════
Generates all figures derived from the 104-subject HCP-YA cohort.

Figures
───────
  Fig 1   Spatial gradient of gBOLD–CSF coupling       (requires utils_hcp)
  Fig 2   Within-run reliability (ICC split-half)
  Fig 3   Time-window stability                         (requires utils_hcp)
  Fig 4   Nuisance variance (frequency-specific)
  Fig 5   Pipeline sensitivity
  Fig S3  AAFT surrogate validation — PSD + amplitude   (requires timeseries)
  Fig S5  AAFT surrogate coupling (null vs observed)
  Fig S6  CSF-sampling ICC across all 6 ROIs
  Fig S7  Lag instability vs coupling scatter           (requires utils_hcp)
  Fig S9  ROI nuisance sensitivity

Usage
─────
  python code/figures_g1.py --all
  python code/figures_g1.py --fig 1 2 3 4 5
  python code/figures_g1.py --fig S3 S5 S6 S7 S9
  python code/figures_g1.py --fig 1 S3 --ts_root results/timeseries --run LR

Global options
──────────────
  --root   ROOT     Dataset root (default:)
  --outdir DIR      Output directory (default: <root>/results/figures)
  --run    RUN      Scan run for time-series figures (default: LR)

Fig S3 / S5 extra options
──────────────────────────
  --ts_root DIR     Timeseries root for power/amplitude panels
  --summary CSV     surrogate_summary.csv (default: <root>/results/surrogate_analysis/surrogate_summary.csv)
  --n_surr  N       AAFT surrogates per subject (default: 1000)

Fig S7 extra options
─────────────────────
  --subjects_csv CSV  Subject list CSV (default: <root>/coupling_subjects_104.csv)
  --ts_dir       DIR  Timeseries root (falls back to --root/results/timeseries)
"""

import argparse
import json
import os
import sys
import csv
import glob
import traceback
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.signal import welch
from scipy.stats import gaussian_kde, ttest_rel

warnings.filterwarnings("ignore", category=RuntimeWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import fig_style
fig_style.apply()

# Optional heavy dependency — only needed for Figs 1, 3, S7
try:
    from config_hcp import xcorr as _xcorr
    HAS_XCORR = True
except ImportError:
    HAS_XCORR = False

TR     = 0.72
N_VOLS = 1190
HALF   = N_VOLS // 2   # 595 volumes

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

def _load_subjects(path):
    with open(path) as f:
        return [r["subject"] for r in csv.DictReader(f)]

def _resolve_paths(args):
    """Return (ts_dir, csv_s, out) from args."""
    ts_dir = os.path.join(args.root, "data/timeseries_primary")
    csv_s  = os.path.join(args.root, "subjects/coupling_subjects_104.csv")
    out    = args.outdir or os.path.join(args.root, "results/figures")
    os.makedirs(out, exist_ok=True)
    return ts_dir, csv_s, out

def _fmt_p(p):
    if p < 0.0001: return "p < 0.0001"
    elif p < 0.001: return f"p = {p:.4f}"
    elif p < 0.01:  return f"p = {p:.3f}"
    return f"p = {p:.2f}"


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Spatial gradient of gBOLD–CSF coupling
# ══════════════════════════════════════════════════════════════════════════════

def _group_xcorr(subjects, ts_dir, run, band_tag, max_lag):
    ROIS = fig_style.ROI_ORDER
    roi_xcs = {r: [] for r in ROIS}
    for sub in subjects:
        sd         = os.path.join(ts_dir, f"{sub}_{run}")
        gbold_path = os.path.join(sd, f"gbold_{band_tag}.csv")
        if not os.path.exists(gbold_path):
            continue
        gbold = pd.read_csv(gbold_path).iloc[:, 0].values.astype(float)
        for roi in ROIS:
            csf_path = os.path.join(sd, f"csf_{fig_style.roi_to_filename(roi)}_{band_tag}.csv")
            if not os.path.exists(csf_path):
                continue
            csf = pd.read_csv(csf_path).iloc[:, 0].values.astype(float)
            n   = min(len(gbold), len(csf))
            cc  = _xcorr(gbold[:n], csf[:n], tr=TR, max_lag_sec=max_lag)
            roi_xcs[roi].append(cc["xc"])
    result = {}
    for roi, xcs in roi_xcs.items():
        if not xcs:
            continue
        min_len = min(len(x) for x in xcs)
        arr     = np.array([x[:min_len] for x in xcs])
        ml      = min_len // 2
        lags    = np.arange(-ml, ml + 1) * TR
        if len(lags) > min_len:
            lags = lags[:min_len]
        result[roi] = (lags, arr.mean(0), arr.std(0), len(xcs))
    return result


def fig1_spatial_gradient(args):
    print("\n── Fig 1: Spatial gradient ─────────────────────────────────")
    if not HAS_XCORR:
        print("  SKIP: config_hcp.xcorr not available"); return
    ts_dir, csv_s, out = _resolve_paths(args)
    s3_csv   = os.path.join(args.root, "results/coupling_g1/stage3_coupling.csv")
    subjects = _load_subjects(csv_s)
    run      = args.run
    ROIS     = fig_style.ROI_ORDER

    band_curves = {}
    for band in fig_style.BAND_ORDER:
        max_lag = fig_style.BAND_MAX_LAG[band]
        print(f"  xcorr [{band}] …", end=" ", flush=True)
        band_curves[band] = _group_xcorr(subjects, ts_dir, run, band, max_lag)
        print(f"{len(band_curves[band])} ROIs")

    df3 = pd.read_csv(s3_csv)
    if "band" not in df3.columns and "band_tag" in df3.columns:
        df3 = df3.rename(columns={"band_tag": "band"})
    if "roi" in df3.columns:
        df3["roi"] = df3["roi"].str.replace("_", "-", regex=False)

    fig = plt.figure(figsize=(fig_style.COL2, fig_style.H3), facecolor="white")
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                             height_ratios=[2.2, 1.2, 1.2],
                             hspace=0.52, wspace=0.30,
                             left=0.10, right=0.97,
                             top=0.94, bottom=0.09)

    x_bar = np.arange(len(ROIS)); w_bar = 0.65

    for bi, band in enumerate(fig_style.BAND_ORDER):
        bcol    = fig_style.BAND_COLORS[band]
        blabel  = fig_style.BAND_LABELS[band]
        max_lag = fig_style.BAND_MAX_LAG[band]
        curves  = band_curves.get(band, {})
        bdf     = df3[df3["band"] == band]
        band_name, band_hz = blabel.split("\n")

        ax0 = fig.add_subplot(gs[0, bi]); fig_style.clean_ax(ax0)
        ax0.axhline(0, color="#ccc", lw=0.6, zorder=1)
        ax0.axvspan(0, max_lag, alpha=0.05, color=bcol, zorder=0)
        ax0.axvline(0, color="#aaa", lw=0.8, ls="--", alpha=0.5)
        for roi in ROIS:
            if roi not in curves:
                continue
            lags, mean_xc, std_xc, _ = curves[roi]
            c = fig_style.ROI_COLORS[roi]; ls = fig_style.ROI_LS[roi]; lw = fig_style.ROI_LW[roi]
            ax0.fill_between(lags, mean_xc - std_xc, mean_xc + std_xc,
                             color=c, alpha=0.10, zorder=2)
            ax0.plot(lags, mean_xc, color=c, lw=lw, ls=ls, zorder=4,
                     label=fig_style.ROI_LABELS[roi])
            pk = np.argmin(mean_xc)
            ax0.scatter(lags[pk], mean_xc[pk], color=c, s=25,
                        zorder=6, edgecolors="white", linewidths=0.8)
        ax0.set_ylim(-1.05, 1.05); ax0.set_xlim(-max_lag, max_lag)
        ax0.set_xlabel("Lag (s)", fontsize=fig_style.FS_LABEL, labelpad=3)
        if bi == 0:
            ax0.set_ylabel("Coupling (gBOLD, CSF) r", fontsize=fig_style.FS_LABEL)
        ax0.set_title(f"{band_name}\n{band_hz}", fontsize=fig_style.FS_TITLE,
                      fontweight="bold", color=bcol, pad=6)
        fig_style.panel_label(ax0, ["A", "B", "C"][bi], y=1.15)

        ax1 = fig.add_subplot(gs[1, bi]); fig_style.clean_ax(ax1)
        for ki, roi in enumerate(ROIS):
            vals = bdf[bdf["roi"] == roi]["abs_r"].dropna().values
            if not len(vals):
                continue
            m = vals.mean(); ci = _ci95(vals)
            ax1.bar(x_bar[ki], m, width=w_bar, color=fig_style.ROI_COLORS[roi],
                    alpha=0.85, edgecolor="white", lw=0.3, zorder=3)
            ax1.errorbar(x_bar[ki], m, yerr=ci, fmt="none", color="#333",
                         capsize=3, lw=1.0, zorder=5)
        ax1.set_ylim(0, 1.0); ax1.set_yticks([0, .25, .50, .75, 1.0])
        ax1.set_xticks(x_bar)
        ax1.set_xticklabels([fig_style.ROI_LABELS[r] for r in ROIS],
                             fontsize=fig_style.FS_TICK - 0.5, rotation=30, ha="right")
        if bi == 0:
            ax1.set_ylabel("Peak |r| ± 95% CI", fontsize=fig_style.FS_LABEL)
        fig_style.panel_label(ax1, ["D", "E", "F"][bi], y=1.20)

        ax2 = fig.add_subplot(gs[2, bi]); fig_style.clean_ax(ax2)
        lag_vals = []
        for ki, roi in enumerate(ROIS):
            vals = bdf[bdf["roi"] == roi]["peak_lag"].dropna().values
            if not len(vals):
                continue
            m = vals.mean(); ci = _ci95(vals); lag_vals.append(m)
            ax2.bar(x_bar[ki], m, width=w_bar, color=fig_style.ROI_COLORS[roi],
                    alpha=0.85, edgecolor="white", lw=0.3, zorder=3)
            ax2.errorbar(x_bar[ki], m, yerr=ci, fmt="none", color="#333",
                         capsize=3, lw=1.0, zorder=5)
        if lag_vals:
            ymax = max(abs(min(lag_vals)), max(lag_vals)) * 1.4 + 1
            ax2.set_ylim(-1, ymax)
        ax2.set_xticks(x_bar)
        ax2.set_xticklabels([fig_style.ROI_LABELS[r] for r in ROIS],
                             fontsize=fig_style.FS_TICK - 0.5, rotation=30, ha="right")
        if bi == 0:
            ax2.set_ylabel("Peak lag (s) ± 95% CI", fontsize=fig_style.FS_LABEL)
        fig_style.panel_label(ax2, ["G", "H", "I"][bi], y=1.10)

    handles = [Line2D([0], [0], color=fig_style.ROI_COLORS[r],
                      lw=fig_style.ROI_LW[r], ls=fig_style.ROI_LS[r],
                      label=fig_style.ROI_LABELS[r])
               for r in ROIS]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               fontsize=fig_style.FS_LEGEND + 1, frameon=False,
               bbox_to_anchor=(0.53, -0.04))
    fig_style.save(fig, "Fig1_spatial_gradient", out)
    print(f"  ✓  Fig1_spatial_gradient  →  {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Within-run reliability (ICC)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_icc(args):
    print("\n── Fig 2: Within-run reliability (ICC) ─────────────────────")
    _, _, out = _resolve_paths(args)
    ca_dir = os.path.join(args.root, "results/coupling_g1")
    sh_csv = _find_csv_prefix(ca_dir, "stage3_splithalf")
    ic_csv = _find_csv_prefix(ca_dir, "stage3_icc")
    shdf = pd.read_csv(sh_csv); icdf = pd.read_csv(ic_csv)
    zicc = icdf[icdf["roi"] == "z0"].copy()
    PRIMARY = "z0"
    METRICS  = ["amplitude_gbold", "amplitude_csf", "abs_r", "peak_lag"]
    ML = {"amplitude_gbold": "gBOLD", "amplitude_csf": "CSF",
          "abs_r": "|r|", "peak_lag": "Lag"}
    MC = {"amplitude_gbold": "#455A64", "amplitude_csf": "#607D8B",
          "abs_r": "#1565C0", "peak_lag": "#B71C1C"}

    fig = plt.figure(figsize=(fig_style.COL2, fig_style.H2), facecolor="white")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                             height_ratios=[1.25, 1.0],
                             hspace=0.48, wspace=0.28,
                             left=0.09, right=0.98,
                             top=0.93, bottom=0.12)

    for bi, band in enumerate(fig_style.BAND_ORDER):
        bc = fig_style.BAND_COLORS[band]
        bn, bh = fig_style.BAND_LABELS[band].split("\n")

        ax = fig.add_subplot(gs[0, bi]); fig_style.clean_ax(ax)
        h1 = shdf[(shdf["band"] == band) & (shdf["roi"] == PRIMARY) & (shdf["half"] == 1)]["abs_r"].values
        h2 = shdf[(shdf["band"] == band) & (shdf["roi"] == PRIMARY) & (shdf["half"] == 2)]["abs_r"].values
        n  = min(len(h1), len(h2)); h1 = h1[:n]; h2 = h2[:n]
        ax.scatter(h1, h2, s=22, color=bc, alpha=0.68, edgecolors="none", zorder=3)
        ax.plot([0, 1], [0, 1], color="#888", lw=0.9, ls="--", zorder=2)
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.set_xticks([0, .25, .50, .75, 1]); ax.set_yticks([0, .25, .50, .75, 1])
        ax.set_xlabel("Half 1  |r|", fontsize=fig_style.FS_LABEL)
        if bi == 0:
            ax.set_ylabel("Half 2  |r|", fontsize=fig_style.FS_LABEL)
        ax.set_title(f"{bn}\n{bh}", fontsize=fig_style.FS_TITLE,
                     fontweight="bold", color=bc, pad=3)
        fig_style.panel_label(ax, ["A", "B", "C"][bi], y=1.16)
        row = zicc[(zicc["band"] == band) & (zicc["metric"] == "abs_r")]
        if len(row):
            icc = row["icc"].iloc[0]; cl = row["ci_low"].iloc[0]; ch = row["ci_high"].iloc[0]
            ax.text(0.05, 0.95, f"ICC = {icc:.2f}\n95% CI [{cl:.2f}, {ch:.2f}]",
                    transform=ax.transAxes, va="top",
                    fontsize=fig_style.FS_ANNOT + 1, color=bc, fontweight="bold")

        ax = fig.add_subplot(gs[1, bi]); fig_style.clean_ax(ax)
        for mi, metric in enumerate(METRICS):
            row = zicc[(zicc["band"] == band) & (zicc["metric"] == metric)]
            if not len(row):
                continue
            icc = row["icc"].iloc[0]; cl = row["ci_low"].iloc[0]; ch = row["ci_high"].iloc[0]
            yerr = [[max(icc - cl, 0)], [max(ch - icc, 0)]]
            ax.bar(mi, icc, width=0.62, color=MC[metric], alpha=0.88,
                   edgecolor="white", lw=0.4, zorder=3)
            ax.errorbar(mi, icc, yerr=yerr, fmt="none", color="#333",
                        capsize=3, lw=1.0, zorder=4)
        ax.axhline(0, color="#ccc", lw=0.6)
        ax.set_ylim(-0.05, 1.05); ax.set_yticks([0, .25, .50, .75, 1])
        ax.set_xticks(np.arange(len(METRICS)))
        ax.set_xticklabels([ML[m] for m in METRICS], fontsize=fig_style.FS_TICK)
        if bi == 0:
            ax.set_ylabel("ICC(2,1)", fontsize=fig_style.FS_LABEL)
        fig_style.panel_label(ax, ["D", "E", "F"][bi], y=1.14)

    fig_style.save(fig, "Fig2_withinrun_reliability_z0", out)
    print(f"  ✓  Fig2_withinrun_reliability_z0  →  {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Time-window stability
# ══════════════════════════════════════════════════════════════════════════════

def fig3_timewindow(args):
    print("\n── Fig 3: Time-window stability ────────────────────────────")
    if not HAS_XCORR:
        print("  SKIP: config_hcp.xcorr not available"); return
    ts_dir, csv_s, out = _resolve_paths(args)
    subjects = _load_subjects(csv_s); run = args.run
    N_WIN = 4; WIN_LEN = 1190 // N_WIN; WIN_DUR = WIN_LEN * TR / 60

    def compute_windowed(band, max_lag):
        rows = []
        for sub in subjects:
            sd = os.path.join(ts_dir, f"{sub}_{run}")
            gp = os.path.join(sd, f"gbold_{band}.csv")
            cp = os.path.join(sd, f"csf_z0_{band}.csv")
            if not os.path.exists(gp) or not os.path.exists(cp):
                continue
            gbold = pd.read_csv(gp).iloc[:, 0].values.astype(float)
            csf   = pd.read_csv(cp).iloc[:, 0].values.astype(float)
            n     = min(len(gbold), len(csf))
            for wi in range(N_WIN):
                s = wi * WIN_LEN; e = s + WIN_LEN
                if e > n:
                    continue
                sl = min(max_lag, (WIN_LEN * TR) * 0.25)
                cc = _xcorr(gbold[s:e], csf[s:e], tr=TR, max_lag_sec=sl)
                rows.append({"subject": sub, "band": band, "window": wi + 1,
                             "abs_r": abs(cc["neg_peak_r"])})
        return rows

    all_rows = []
    for band in fig_style.BAND_ORDER:
        print(f"  windowed [{band}] …", end=" ", flush=True)
        rows = compute_windowed(band, fig_style.BAND_MAX_LAG[band])
        all_rows.extend(rows); print(len(rows))
    wdf = pd.DataFrame(all_rows)
    windows = list(range(1, N_WIN + 1))

    fig = plt.figure(figsize=(fig_style.COL2, fig_style.H1 + 0.6), facecolor="white")
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                             hspace=0.46, wspace=0.30,
                             left=0.10, right=0.97,
                             top=0.88, bottom=0.18)

    for bi, band in enumerate(fig_style.BAND_ORDER):
        bcol = fig_style.BAND_COLORS[band]
        bn, bh = fig_style.BAND_LABELS[band].split("\n")
        bdf  = wdf[wdf["band"] == band]

        ax = fig.add_subplot(gs[0, bi]); fig_style.clean_ax(ax)
        means = []; cis = []
        for w in windows:
            v = bdf[bdf["window"] == w]["abs_r"].dropna().values.astype(float)
            means.append(np.mean(v) if len(v) else np.nan)
            cis.append(_ci95(v) if len(v) > 1 else 0.0)
        means = np.array(means); cis = np.array(cis)
        ax.fill_between(windows, means - cis, means + cis,
                        color=bcol, alpha=0.15, zorder=2)
        ax.plot(windows, means, color=bcol, lw=fig_style.LW_MAIN,
                marker="o", markersize=fig_style.MS_MAIN, zorder=4)
        valid = means[np.isfinite(means)]
        ax.set_ylim((max(0, valid.min() - 0.06), min(1.0, valid.max() + 0.08))
                    if len(valid) else (0, 1))
        ax.set_xlim(windows[0] - .3, windows[-1] + .3); ax.set_xticks(windows)
        ax.set_xticklabels([f"Win {w}" for w in windows],
                           fontsize=fig_style.FS_TICK - 1)
        ax.set_xlabel(f"Time window (~{WIN_DUR:.1f} min)", fontsize=fig_style.FS_LABEL)
        if bi == 0:
            ax.set_ylabel("Coupling |r|", fontsize=fig_style.FS_LABEL)
        ax.set_title(f"{bn}\n{bh}", fontsize=fig_style.FS_TITLE,
                     fontweight="bold", color=bcol, pad=6)
        fig_style.panel_label(ax, ["A", "B", "C"][bi], y=1.22)

    fig_style.save(fig, "Fig3_timewindow_stability_z0", out)
    print(f"  ✓  Fig3_timewindow_stability_z0  →  {out}")


def fig4_nuisance(args):
    print("\n── Fig 4: Nuisance variance ────────────────────────────────")
    _, _, out = _resolve_paths(args)
    s2 = os.path.join(args.root, "results/coupling_g1/stage2_freq_incremental_r2.csv")
    if not os.path.exists(s2):
        print(f"  ERROR: {s2}"); return
    df = pd.read_csv(s2)
    TARGETS    = ["gbold", "csf_z0"]
    INC_ORDER  = ["r2_motion", "delta_r2_wm", "delta_r2_acompcor"]
    INC_LABELS = {"r2_motion":        "Motion $R^2$",
                  "delta_r2_wm":       "WM $\\Delta R^2$",
                  "delta_r2_acompcor": "aCompCor $\\Delta R^2$"}
    # BUG FIXED: use canonical fig_style.NUISANCE_COLORS instead of wrong local dict
    INC_COLORS = fig_style.NUISANCE_COLORS

    fig = plt.figure(figsize=(fig_style.COL2, fig_style.H2), facecolor="white")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                             left=0.12, right=0.97, top=0.88, bottom=0.17,
                             hspace=0.30, wspace=0.30)
    x = np.arange(len(INC_ORDER))

    for ri, target in enumerate(TARGETS):
        for ci, band in enumerate(fig_style.BAND_ORDER):
            ax = fig.add_subplot(gs[ri, ci]); fig_style.clean_ax(ax)
            sub = df[(df["band"] == band) & (df["target"] == target)]
            means = [sub[m].dropna().values for m in INC_ORDER]
            ms = [v.mean() if len(v) else np.nan for v in means]
            cs = [_ci95(v) if len(v) > 1 else 0.0 for v in means]
            ax.bar(x, ms, width=0.68, color=[INC_COLORS[m] for m in INC_ORDER],
                   alpha=0.88, edgecolor="white", lw=0.5, zorder=3)
            ax.errorbar(x, ms, yerr=cs, fmt="none", color="#333",
                        capsize=3, lw=1.0, zorder=5)
            ax.set_ylim(0, 0.48); ax.set_yticks([0, .10, .20, .30, .40])
            ax.set_xticks([])
            if ri == 0:
                bn, bh = fig_style.BAND_LABELS[band].split("\n")
                ax.set_title(f"{bn}\n{bh}", fontsize=fig_style.FS_TITLE,
                             fontweight="bold", color=fig_style.BAND_COLORS[band], pad=5)
            if ci == 0:
                ax.set_ylabel(
                    "gBOLD variance" if target == "gbold" else "CSF z0 variance",
                    fontsize=fig_style.FS_LABEL, labelpad=8)

    fig.text(0.075, 0.90, "A", fontsize=fig_style.FS_PANEL, fontweight="bold")
    fig.text(0.075, 0.45, "B", fontsize=fig_style.FS_PANEL, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=INC_COLORS[m], alpha=0.88,
                              label=INC_LABELS[m]) for m in INC_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=fig_style.FS_LEGEND + 1, frameon=False,
               bbox_to_anchor=(0.54, 0.04))
    fig_style.save(fig, "Fig4_frequency_specific_nuisance_variance", out)
    print(f"  ✓  Fig4_frequency_specific_nuisance_variance  →  {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Pipeline sensitivity
# BUG FIX: replaced local PL_COLORS (wrong values) with fig_style.PIPELINE_COLORS
# ══════════════════════════════════════════════════════════════════════════════

def fig5_pipeline(args):
    print("\n── Fig 5: Pipeline sensitivity ─────────────────────────────")
    _, _, out = _resolve_paths(args)
    rat_csv  = os.path.join(args.root, "results/coupling_g1/stage45_matched_ratios_summary.csv")
    full_csv = os.path.join(args.root, "results/coupling_g1/stage45_matched_ratios.csv")
    if not os.path.exists(full_csv):
        print(f"  ERROR: {full_csv}"); return
    ROI = "z0"
    rat  = pd.read_csv(rat_csv)
    rat  = rat[(rat["roi"] == ROI) & (rat["pipeline"] != "P0")]
    full = pd.read_csv(full_csv); full = full[full["roi"] == ROI]
    PIPELINES = ["P0", "P1", "P2", "P3"]
    # BUG FIXED: use canonical fig_style.PIPELINE_COLORS (grey→blue gradient)
    PL_COLORS = fig_style.PIPELINE_COLORS
    PL_LABELS = {"P0": "P0: no regression", "P1": "P1: motion",
                 "P2": "P2: motion + WM mean", "P3": "P3: motion + aCompCor"}

    fig = plt.figure(figsize=(fig_style.COL2, fig_style.H2), facecolor="white")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                             height_ratios=[1, 1],
                             hspace=0.48, wspace=0.30,
                             left=0.11, right=0.97,
                             top=0.88, bottom=0.18)

    for bi, band in enumerate(fig_style.BAND_ORDER):
        bcol = fig_style.BAND_COLORS[band]
        bn, bh = fig_style.BAND_LABELS[band].split("\n")
        bfull = full[full["band"] == band]

        ax1 = fig.add_subplot(gs[0, bi]); fig_style.clean_ax(ax1)
        for ki, pl in enumerate(PIPELINES):
            vals = bfull[bfull["pipeline"] == pl]["abs_r"].dropna().values
            if not len(vals):
                continue
            m = vals.mean(); ci = _ci95(vals)
            ax1.bar(ki, m, width=0.62, color=PL_COLORS[pl], alpha=0.88,
                    edgecolor="white", lw=0.3, zorder=3)
            ax1.errorbar(ki, m, yerr=ci, fmt="none", color="#333",
                         capsize=3, lw=1.0, zorder=5)
        ax1.set_xticks(range(4))
        ax1.set_xticklabels(PIPELINES, fontsize=fig_style.FS_TICK, fontweight="bold")
        ax1.set_ylim(0, 0.85)
        if bi == 0:
            ax1.set_ylabel("Coupling |r|", fontsize=fig_style.FS_LABEL)
        ax1.set_title(f"{bn}\n{bh}", fontsize=fig_style.FS_TITLE,
                      fontweight="bold", color=bcol, pad=6)
        fig_style.panel_label(ax1, ["A", "B", "C"][bi], y=1.12)

        ax2 = fig.add_subplot(gs[1, bi]); fig_style.clean_ax(ax2)
        ax2.axhline(1.0, color="#333", lw=1.0, ls="--", zorder=2)
        for ki, pl in enumerate(["P1", "P2", "P3"]):
            vals = (bfull[bfull["pipeline"] == pl]["ratio_r"].dropna().values
                    if "ratio_r" in bfull.columns else np.array([]))
            if len(vals):
                m = vals.mean(); ci = _ci95(vals)
            else:
                row = rat[(rat["band"] == band) & (rat["pipeline"] == pl)]
                if not len(row):
                    continue
                m = row["mean_ratio_r"].iloc[0]
                n = int(row["n"].iloc[0]) if "n" in row.columns and row["n"].iloc[0] > 1 else 2
                ci = float(row["sd_ratio_r"].iloc[0] / np.sqrt(n) * stats.t.ppf(0.975, n - 1))
            ax2.bar(ki, m, width=0.58, color=PL_COLORS[pl], alpha=0.88,
                    edgecolor="white", lw=0.3, zorder=3)
            ax2.errorbar(ki, m, yerr=ci, fmt="none", color="#333",
                         capsize=4, lw=1.0, zorder=5)
        ax2.set_ylim(0.4, 1.15); ax2.set_yticks([0.5, 0.75, 1.0])
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(["P1", "P2", "P3"], fontsize=fig_style.FS_TICK, fontweight="bold")
        if bi == 0:
            ax2.set_ylabel("Coupling |r|\nretained", fontsize=fig_style.FS_LABEL)
        fig_style.panel_label(ax2, ["D", "E", "F"][bi], y=1.12)

    handles = [Patch(facecolor=PL_COLORS[p], alpha=0.88,
                     edgecolor="white", label=PL_LABELS[p]) for p in PIPELINES]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=fig_style.FS_LEGEND, frameon=False,
               bbox_to_anchor=(0.53, 0.035))
    fig_style.save(fig, "Fig5_pipeline_sensitivity_nwz0", out)
    print(f"  ✓  Fig5_pipeline_sensitivity_nwz0  →  {out}")



# AAFT helpers (from plot_aaft_final.py)
def _aaft_surrogate(x, rng):
    n      = len(x)
    rank_x = np.argsort(np.argsort(x))
    gauss  = rng.standard_normal(n)
    gauss  = (gauss - gauss.mean()) / (gauss.std() + 1e-12)
    x_g    = np.sort(gauss)[rank_x]
    fft_g  = np.fft.rfft(x_g)
    n_freq = len(fft_g)
    phases = rng.uniform(0, 2 * np.pi, n_freq)
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    fft_r = np.abs(fft_g) * np.exp(1j * phases)
    x_rg  = np.fft.irfft(fft_r, n=n)
    return np.sort(x)[np.argsort(np.argsort(x_rg))].astype(np.float64)

def _load_gbold(ts_root, subject, run, band):
    p = os.path.join(ts_root, f"{subject}_{run}", f"gbold_{band}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p).iloc[:, 0].to_numpy(dtype=np.float64)

def _get_tr(ts_root, run):
    ms = glob.glob(os.path.join(ts_root, f"*_{run}", "meta.json"))
    if ms:
        with open(ms[0]) as fh:
            return float(json.load(fh).get("TR", 0.72))
    return 0.72

def _style_violin_aaft(vp, fc, alpha_body=0.35, lw=0.8):
    for body in vp["bodies"]:
        body.set_facecolor(fc); body.set_edgecolor(fc)
        body.set_alpha(alpha_body); body.set_linewidth(lw)
    if "cmedians" in vp:
        vp["cmedians"].set_color(fc)
        vp["cmedians"].set_linewidth(1.8)
        vp["cmedians"].set_alpha(1.0)

def _make_aaft_grid(n_rows, n_cols, fig_w, fig_h, hspace=0.55, wspace=0.35):
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(n_rows, n_cols,
                            hspace=hspace, wspace=wspace,
                            left=0.11, right=0.97,
                            top=0.93, bottom=0.07)
    axes = np.array([[fig.add_subplot(gs[r, c])
                      for c in range(n_cols)]
                     for r in range(n_rows)])
    return fig, axes

def _set_row_col_labels_aaft(axes, col_titles, col_colors=None):
    """Band labels as column headers (top row), ROI labels as row y-labels.
    BUG FIXED: fontsize uses fig_style.FS_TITLE / fig_style.FS_LABEL (was 8.5 / 8).
    """
    ROI_ORDER_A = fig_style.ROI_ORDER_AAFT
    ROI_LABELS_A = fig_style.ROI_LABELS_AAFT
    for ci, ttl in enumerate(col_titles):
        color = col_colors[ci] if col_colors else "black"
        # BUG FIXED: was fontsize=8.5; now uses FS_TITLE=10
        axes[0, ci].set_title(ttl, fontsize=fig_style.FS_TITLE, fontweight="bold",
                              pad=5, color=color)
    for ri, roi in enumerate(ROI_ORDER_A):
        # BUG FIXED: was fontsize=8; now uses FS_LABEL=9
        axes[ri, 0].set_ylabel(ROI_LABELS_A[roi], fontsize=fig_style.FS_LABEL,
                               fontweight="bold", labelpad=4)

def _aaft_placeholder(ax):
    ax.text(0.5, 0.5, "Timeseries\nnot provided",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=fig_style.FS_ANNOT, color="#aaaaaa")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


POWER_XLIM_AAFT = {
    "broad":      (0.00, 0.11),
    "neurogenic": (0.01, 0.05),
    "vasogenic":  (0.05, 0.11),
}
AMP_XLIM_AAFT = {
    "broad":      (-50,  50),
    "neurogenic": (-40,  40),
    "vasogenic":  (-25,  25),
}
ORIG_COLOR_AAFT = "#222222"


def figS3_aaft_validation_combined(band_data, out, has_ts):
    """Fig S3 — gBOLD AAFT validation: 2 rows × 3 cols.
    Row 0: Normalised PSD (original gBOLD vs AAFT surrogate).
    Row 1: Amplitude distribution (original vs AAFT surrogate).
    One panel per frequency band; gBOLD signal only (not CSF ROIs).
    Outputs a single figure: FigS3_psd_amplitude.
    """
    BAND_ORDER_A  = fig_style.BAND_ORDER
    BAND_COLORS_A = fig_style.BAND_COLORS
    BAND_LABELS_A = fig_style.BAND_LABELS

    fig = plt.figure(figsize=(fig_style.COL2, fig_style.H1 + 0.6), facecolor="white")
    gs  = gridspec.GridSpec(2, len(BAND_ORDER_A), figure=fig,
                             hspace=0.50, wspace=0.30,
                             left=0.10, right=0.97,
                             top=0.90, bottom=0.18)
    axes = np.array([[fig.add_subplot(gs[r, c])
                      for c in range(len(BAND_ORDER_A))]
                     for r in range(2)])

    # Column headers (band names)
    for ci, band in enumerate(BAND_ORDER_A):
        bname, bhz = BAND_LABELS_A[band].split("\n")
        axes[0, ci].set_title(f"{bname}\n{bhz}",
                              fontsize=fig_style.FS_TITLE, fontweight="bold",
                              color=BAND_COLORS_A[band], pad=4)

    # Row labels (amplitude y-label set per-panel inside loop)
    axes[0, 0].set_ylabel("Norm. PSD", fontsize=fig_style.FS_LABEL, labelpad=4)

    for ci, band in enumerate(BAND_ORDER_A):
        color = BAND_COLORS_A[band]
        lo, hi = fig_style.BAND_RANGES[band]
        bd         = band_data.get(band, {})
        signals    = bd.get("signals", [])
        surrogates = bd.get("surrogates", [])
        fs         = bd.get("fs", 1 / 0.72)

        ax_psd = axes[0, ci]
        ax_amp = axes[1, ci]
        fig_style.clean_ax(ax_psd)
        fig_style.clean_ax(ax_amp)

        if has_ts and signals:
            # ── PSD row ──────────────────────────────────────────────
            nperseg   = min(256, min(len(s) for s in signals) // 4)
            psds_orig = np.array([welch(s, fs=fs, nperseg=nperseg)[1] for s in signals])
            psds_surr = np.array([welch(s, fs=fs, nperseg=nperseg)[1] for s in surrogates])
            freqs     = welch(signals[0], fs=fs, nperseg=nperseg)[0]
            psd_om = psds_orig.mean(0); psd_os = psds_orig.std(0) / np.sqrt(len(signals))
            psd_sm = psds_surr.mean(0); psd_ss = psds_surr.std(0) / np.sqrt(len(surrogates))
            xlo, xhi = POWER_XLIM_AAFT[band]
            mask = (freqs >= xlo) & (freqs <= xhi)
            fp   = freqs[mask]
            g_mx = psd_om[mask].max() or 1.0
            gm = psd_om[mask] / g_mx; ge = psd_os[mask] / g_mx
            sm = psd_sm[mask] / g_mx; se = psd_ss[mask] / g_mx

            SURR_COLOR = "#E87722"   # orange dash-dot — consistent across PSD and amplitude
            # PSD: grey shading around both curves (combined SEM envelope)
            ax_psd.fill_between(fp, gm - ge, gm + ge, color="#888888", alpha=0.25, zorder=1)
            ax_psd.plot(fp, gm, color=ORIG_COLOR_AAFT, lw=1.3, zorder=3)
            ax_psd.plot(fp, sm, color=SURR_COLOR, lw=1.3, ls="-.", zorder=3)
            # axvspan covers full x-range with band colour tint
            ax_psd.axvspan(xlo, xhi, alpha=0.09, color=color, zorder=0)
            ax_psd.set_xlim(xlo, xhi)
            ax_psd.set_ylim(0, 1.32); ax_psd.set_yticks([0, 0.5, 1.0])
            ax_psd.tick_params(labelsize=fig_style.FS_TICK)
            ax_psd.set_xlabel("Frequency (Hz)", fontsize=fig_style.FS_LABEL)

            # ── Amplitude row — smooth KDE, grey fill under original only
            orig_pool = np.concatenate(signals)
            surr_pool = np.concatenate(surrogates)
            amp_xlo, amp_xhi = AMP_XLIM_AAFT[band]
            amp_x = np.linspace(amp_xlo, amp_xhi, 500)
            ok = gaussian_kde(orig_pool, bw_method=0.08)(amp_x)
            sk = gaussian_kde(surr_pool, bw_method=0.08)(amp_x)
            # Grey fill under original only; orange dash-dot on top for surrogate
            ax_amp.fill_between(amp_x, ok, alpha=0.50, color="#888888", zorder=2)
            ax_amp.plot(amp_x, ok, color=ORIG_COLOR_AAFT, lw=1.3, zorder=3)
            ax_amp.plot(amp_x, sk, color=SURR_COLOR,      lw=1.3, ls="-.", zorder=4)
            ax_amp.set_xlim(amp_xlo, amp_xhi)
            ax_amp.set_ylim(bottom=0)
            ax_amp.tick_params(labelsize=fig_style.FS_TICK)
            ax_amp.set_xlabel("Amplitude (a.u.)", fontsize=fig_style.FS_LABEL)
        else:
            _aaft_placeholder(ax_psd)
            _aaft_placeholder(ax_amp)

    # Row-0 y-label only on leftmost panel; row-1 y-label shared
    axes[1, 0].set_ylabel("Density", fontsize=fig_style.FS_LABEL, labelpad=4)

    # Shared legend — inside figure, bottom-centre below amplitude panels
    leg_handles = [
        plt.Line2D([0], [0], color=ORIG_COLOR_AAFT, lw=1.3,           label="Original gBOLD"),
        plt.Line2D([0], [0], color="#E87722",        lw=1.3, ls="-.",  label="AAFT surrogate"),
    ]
    fig.legend(handles=leg_handles, loc="lower center", ncol=2,
               fontsize=fig_style.FS_LABEL, frameon=False,
               bbox_to_anchor=(0.54, 0.01), bbox_transform=fig.transFigure)

    fig_style.save(fig, "FigS3_psd_amplitude", out)
    print(f"  ✓  FigS3_psd_amplitude  →  {out}")


def _figS5_coupling(band_dfs, out):
    """Fig S5 — AAFT coupling (null violin vs observed violin), 6 ROIs × 3 bands.
    BUG FIXES:
      • BAND_COLORS from fig_style
      • dpi via fig_style.save()
      • removed duplicate style_violin / jo lines
      • stem is now FigS5_aaft_coupling
    """
    ROI_ORDER_A  = fig_style.ROI_ORDER_AAFT
    ROI_LABELS_A = fig_style.ROI_LABELS_AAFT
    BAND_ORDER_A = fig_style.BAND_ORDER
    BAND_COLORS_A = fig_style.BAND_COLORS
    BAND_LABELS_A = fig_style.BAND_LABELS

    n_rows, n_cols = len(ROI_ORDER_A), len(BAND_ORDER_A)
    rng_jit = np.random.default_rng(0)

    fig, axes = _make_aaft_grid(n_rows, n_cols, fig_w=5.0, fig_h=n_rows,
                                 hspace=0.60, wspace=0.35)
    _set_row_col_labels_aaft(axes,
                              [BAND_LABELS_A[b] for b in BAND_ORDER_A],
                              [BAND_COLORS_A[b] for b in BAND_ORDER_A])

    # Pre-compute stats & shared y-max per band column
    cell_stats = {}
    y_r_max    = {}
    for band in BAND_ORDER_A:
        df_band = band_dfs.get(band)
        if df_band is None:
            y_r_max[band] = 1.0; continue
        all_obs = []
        for roi in ROI_ORDER_A:
            df_c     = df_band[df_band["roi"] == roi]
            obs_vals = df_c["obs_absr"].dropna().values
            null_m   = df_c["null_mean"].dropna().values
            null_sd  = df_c["null_sd"].dropna().values
            n = min(len(obs_vals), len(null_m), len(null_sd))
            obs_vals = obs_vals[:n]; null_m = null_m[:n]; null_sd = null_sd[:n]
            rng_s = np.random.default_rng(42)
            null_pool = np.abs(
                rng_s.normal(null_m[:, None],
                             np.where(null_sd < 1e-6, 1e-6, null_sd)[:, None],
                             (n, 100))
            ).ravel()
            t_stat, p_val = ttest_rel(obs_vals, null_m)
            p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
            diff     = obs_vals - null_m
            cohens_d = diff.mean() / (diff.std() + 1e-9)
            cell_stats[(band, roi)] = dict(
                obs_vals=obs_vals, null_pool=null_pool,
                null_sub_median=null_m,
                t_stat=t_stat, p_val=p_one, cohens_d=cohens_d)
            all_obs.extend(obs_vals.tolist())
        y_r_max[band] = np.percentile(all_obs, 99) * 1.15 if all_obs else 1.0

    for ci, band in enumerate(BAND_ORDER_A):
        color = BAND_COLORS_A[band]
        for ri, roi in enumerate(ROI_ORDER_A):
            ax = axes[ri, ci]
            if (band, roi) not in cell_stats:
                _aaft_placeholder(ax); continue
            cs             = cell_stats[(band, roi)]
            obs_vals       = cs["obs_vals"]
            null_pool      = cs["null_pool"]
            null_sub_med   = cs["null_sub_median"]
            p_val          = cs["p_val"]
            cohens_d       = cs["cohens_d"]

            # Null violin (position 0)
            vp_n = ax.violinplot([null_pool], positions=[0], widths=0.50,
                                 showmedians=True, showextrema=False)
            _style_violin_aaft(vp_n, color, alpha_body=0.28)
            jn = rng_jit.uniform(-0.09, 0.09, len(null_sub_med))
            ax.scatter(jn, null_sub_med, s=4, color=color, alpha=0.55,
                       zorder=4, linewidths=0)
            ax.hlines(null_sub_med.mean(), -0.22, 0.22,
                      colors=color, linewidths=1.8, zorder=5)

            # Observed violin (position 1)
            # BUG FIXED: removed duplicate style_violin + jo lines
            vp_o = ax.violinplot([obs_vals], positions=[1], widths=0.50,
                                 showmedians=True, showextrema=False)
            _style_violin_aaft(vp_o, color, alpha_body=0.55)
            jo = rng_jit.uniform(-0.09, 0.09, len(obs_vals))
            ax.scatter(jo + 1, obs_vals, s=4, color=color, alpha=0.55,
                       zorder=4, linewidths=0)
            ax.hlines(obs_vals.mean(), 0.78, 1.22,
                      colors=color, linewidths=1.8, zorder=5)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Null", "Obs."], fontsize=6.5)
            ax.set_xlim(-0.52, 1.62)
            ax.set_ylim(0, y_r_max[band])
            ax.tick_params(labelsize=6.5)
            ax.spines[["top", "right"]].set_visible(False)

            if ci == 0:
                ax.set_ylabel(ROI_LABELS_A[roi] + "\n|r|",
                              fontsize=fig_style.FS_LABEL, fontweight="bold", labelpad=3)
            if ri == n_rows - 1:
                ax.set_xlabel("Distribution", fontsize=fig_style.FS_LABEL)

            ax.text(0.04, 0.97,
                    f"d = {cohens_d:.2f}\n{_fmt_p(p_val)}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=5, fontweight="bold", color="#111111",
                    bbox=dict(boxstyle="round,pad=0.20", fc="white",
                              ec="#cccccc", alpha=0.88, lw=0.4))

    # BUG FIXED: stem was "fig_coupling.pdf"; now "FigS5_aaft_coupling"
    fig_style.save(fig, "FigS5_aaft_coupling", out)
    print(f"  ✓  FigS5_aaft_coupling  →  {out}")


def _collect_aaft_data(args, out):
    """Load timeseries and generate AAFT surrogates for Figs S3 & S5."""
    summary = args.summary or os.path.join(
        args.root, "results/surrogate_analysis/surrogate_summary.csv")
    if not os.path.exists(summary):
        print(f"  ERROR: {summary}"); return None, None, None

    df_all = pd.read_csv(summary)
    df_all["significant"] = df_all["significant"].astype(bool)
    run = args.run

    ts_root = getattr(args, "ts_root", None)
    if ts_root is None:
        for _cand in [
            os.path.join(args.root, "data", "timeseries_primary"),
            os.path.join(args.root, "results", "timeseries"),
            os.path.join(args.root, "timeseries"),
        ]:
            if os.path.isdir(_cand) and len(glob.glob(os.path.join(_cand, f"*_{run}"))) > 0:
                ts_root = _cand
                print(f"  Auto-detected ts_root: {ts_root}")
                break

    ts_available = (ts_root is not None and os.path.isdir(ts_root) and
                    len(glob.glob(os.path.join(ts_root, f"*_{run}"))) > 0)
    TR_local = _get_tr(ts_root, run) if ts_available else 0.72
    fs       = 1.0 / TR_local

    band_data = {}
    band_dfs  = {}
    n_sub     = int(df_all[df_all["run"] == run]["subject"].nunique())

    for band in fig_style.BAND_ORDER:
        df_band = df_all[(df_all["band"] == band) & (df_all["run"] == run)].copy()
        if df_band.empty:
            continue
        band_dfs[band] = df_band
        signals = []; surrogates = []

        if ts_available:
            subjects = sorted(df_band["subject"].astype(str).unique())
            for sub in subjects:
                x = _load_gbold(ts_root, sub, run, band)
                if x is not None and len(x) > 100:
                    signals.append(x)
            if signals:
                rng = np.random.default_rng(42)
                n_surr = getattr(args, "n_surr", 1000)
                print(f"  AAFT surrogates [{band}]: {len(signals)} subs × {n_surr} …")
                for i, sig in enumerate(signals):
                    for _ in range(n_surr):
                        surrogates.append(_aaft_surrogate(sig, rng))
                    if (i + 1) % 25 == 0:
                        print(f"    {i+1}/{len(signals)}")
            else:
                ts_available = False

        band_data[band] = {"signals": signals, "surrogates": surrogates, "fs": fs}

    has_ts = ts_available and any(len(band_data[b]["signals"]) > 0
                                   for b in fig_style.BAND_ORDER if b in band_data)
    return band_data, band_dfs, has_ts


def figS3_aaft_validation(args):
    """Fig S3 — AAFT PSD + amplitude (requires timeseries).
    Single combined figure: 2 rows × 3 cols (PSD top, amplitude bottom), gBOLD only.
    """
    print("\n── Fig S3: AAFT PSD + amplitude ────────────────────────────")
    _, _, out = _resolve_paths(args)
    band_data, band_dfs, has_ts = _collect_aaft_data(args, out)
    if band_data is None:
        return
    figS3_aaft_validation_combined(band_data, out, has_ts)


def figS5_aaft_coupling(args):
    """Fig S5 — AAFT coupling (null vs observed)."""
    print("\n── Fig S5: AAFT coupling ────────────────────────────────────")
    _, _, out = _resolve_paths(args)
    band_data, band_dfs, has_ts = _collect_aaft_data(args, out)
    if band_dfs is None:
        return
    _figS5_coupling(band_dfs, out)



def _icc21_2raters(x, y):
    """ICC(2,1) for 2 raters — Shrout & Fleiss 1979."""
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


def _compute_roi_icc_table(sh_df):
    rows = []
    for band in fig_style.BAND_ORDER:
        for roi in fig_style.ROI_ORDER:
            sub = sh_df[(sh_df["band"] == band) & (sh_df["roi"] == roi)]
            h1 = sub[sub["half"] == 1].set_index("subject")
            h2 = sub[sub["half"] == 2].set_index("subject")
            shared = h1.index.intersection(h2.index)
            if len(shared) < 4:
                continue
            x_r = h1.loc[shared, "abs_r"].values.astype(float)
            y_r = h2.loc[shared, "abs_r"].values.astype(float)
            icc_r, lo_r, hi_r = _icc21_2raters(x_r, y_r)
            x_l = h1.loc[shared, "peak_lag"].values.astype(float)
            y_l = h2.loc[shared, "peak_lag"].values.astype(float)
            icc_l, lo_l, hi_l = _icc21_2raters(x_l, y_l)
            rows.append(dict(band=band, roi=roi,
                             icc_r=icc_r, lo_r=lo_r, hi_r=hi_r,
                             icc_l=icc_l, lo_l=lo_l, hi_l=hi_l,
                             n=len(shared)))
    return pd.DataFrame(rows)


def _find_csv_prefix(ca_dir, prefix):
    matches = sorted(f for f in os.listdir(ca_dir)
                     if f.startswith(prefix) and f.endswith(".csv"))
    if not matches:
        raise FileNotFoundError(f"No CSV starting with '{prefix}' in {ca_dir}")
    return os.path.join(ca_dir, matches[0])


def figS6_csf_sampling_icc(args):
    print("\n── Fig S6: CSF-sampling ICC ─────────────────────────────────")
    _, _, out = _resolve_paths(args)
    ca_dir = os.path.join(args.root, "results/coupling_g1")

    try:
        sh_csv = _find_csv_prefix(ca_dir, "stage3_splithalf")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}"); return

    sh_df = pd.read_csv(sh_csv)
    if "roi" in sh_df.columns:
        sh_df["roi"] = sh_df["roi"].str.replace("_", "-", regex=False)
    sh_df["half"] = sh_df["half"].astype(int)

    icc_tbl = _compute_roi_icc_table(sh_df)

    ROW_METRICS = [
        ("icc_r", "lo_r", "hi_r", "ICC(2,1)  |r|"),
        ("icc_l", "lo_l", "hi_l", "ICC(2,1)  lag"),
    ]
    ROI_ORDER = fig_style.ROI_ORDER
    BAR_W = 0.62; GAP = 0.38
    X_CENTERS = np.arange(len(ROI_ORDER)) * (BAR_W + GAP)
    n_rows = len(ROW_METRICS)
    fig_h  = 1.9 * n_rows + 0.9

    fig = plt.figure(figsize=(fig_style.COL2, fig_h), facecolor="white")
    gs  = gridspec.GridSpec(n_rows, len(fig_style.BAND_ORDER), figure=fig,
                             hspace=0.55, wspace=0.30,
                             left=0.12, right=0.97,
                             top=0.93, bottom=0.14)

    for ri, (icc_col, lo_col, hi_col, ylabel) in enumerate(ROW_METRICS):
        for bi, band in enumerate(fig_style.BAND_ORDER):
            ax = fig.add_subplot(gs[ri, bi])
            fig_style.clean_ax(ax)
            band_color = fig_style.BAND_COLORS[band]
            sub = icc_tbl[icc_tbl["band"] == band]

            for xi, roi in enumerate(ROI_ORDER):
                roi_color = fig_style.ROI_COLORS[roi]
                cx = X_CENTERS[xi]
                row = sub[sub["roi"] == roi]
                if not len(row):
                    continue
                icc = float(row[icc_col].iloc[0])
                lo  = float(row[lo_col].iloc[0])
                hi  = float(row[hi_col].iloc[0])
                if not np.isfinite(icc):
                    continue
                ax.bar(cx, max(icc, 0), width=BAR_W, color=roi_color, alpha=0.88,
                       linewidth=0.8, edgecolor="white", zorder=3)
                ax.errorbar(cx, icc,
                            yerr=[[max(icc - lo, 0)], [max(hi - icc, 0)]],
                            fmt="none", color="#333333",
                            lw=1.3, capsize=3.5, capthick=1.3, zorder=4)

            ax.axhline(0,    color="#CCCCCC", lw=0.6, zorder=1)
            ax.axhline(0.75, color="#888888", lw=0.9, ls="--", zorder=2)

            if ri == 0 and bi == len(fig_style.BAND_ORDER) - 1:
                ax.text(X_CENTERS[-1] + BAR_W * 0.6, 0.76, "0.75",
                        fontsize=fig_style.FS_ANNOT, va="bottom", ha="left",
                        color="#777777")

            ax.set_xticks(X_CENTERS)
            ax.set_xticklabels([fig_style.ROI_LABELS[r] for r in ROI_ORDER],
                                fontsize=fig_style.FS_TICK, rotation=35, ha="right")
            ax.set_xlim(X_CENTERS[0] - BAR_W, X_CENTERS[-1] + BAR_W)
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
            ax.tick_params(axis="y", labelsize=fig_style.FS_TICK)

            if bi == 0:
                ax.set_ylabel(ylabel, fontsize=fig_style.FS_LABEL + 1)
            if ri == 0:
                bname, bhz = fig_style.BAND_LABELS[band].split("\n")
                ax.set_title(f"{bname}\n{bhz}", fontsize=fig_style.FS_TITLE,
                             fontweight="bold", color=band_color, pad=4)
            if bi == 0:
                ax.text(-0.28, 1.18, ["A", "B"][ri],
                        transform=ax.transAxes,
                        fontsize=fig_style.FS_PANEL, fontweight="bold", va="top")

    handles = [mpatches.Patch(facecolor=fig_style.ROI_COLORS[r], alpha=0.88,
                               edgecolor="white", linewidth=0.8,
                               label=fig_style.ROI_LABELS[r])
               for r in ROI_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(ROI_ORDER),
               fontsize=fig_style.FS_LEGEND, frameon=False,
               bbox_to_anchor=(0.54, 0.02), bbox_transform=fig.transFigure)

    # BUG FIXED: stem was "FigS_roi_icc"; now "FigS6_csf_sampling_icc"
    fig_style.save(fig, "FigS6_csf_sampling_icc", out)
    print(f"  ✓  FigS6_csf_sampling_icc  →  {out}")



# S7 uses config_hcp and utils_hcp — optional imports handled below
try:
    import config_hcp as _cfg_hcp
    _HAS_CFG_HCP = True
except ImportError:
    _HAS_CFG_HCP = False

_S7_BANDS = {
    "broad":      (0.010, 0.100),
    "neurogenic": (0.020, 0.040),
    "vasogenic":  (0.060, 0.100),
}
_S7_ALL_ROIS = ["z0", "z1", "z2", "z3", "fv-lower", "fv-upper"]
_S7_N_BINS   = 4


def _s7_load_subjects(csv_path):
    with open(csv_path) as f:
        return [row["subject"] for row in csv.DictReader(f)]


def _s7_read_ts(ts_root, sub, run_id, fname):
    path = os.path.join(ts_root, f"{sub}_{run_id}", fname)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).iloc[:, 0].to_numpy(dtype=np.float64)


def _s7_coupling(gbold_seg, csf_seg, max_lag):
    n = len(gbold_seg)
    if n < 30 or np.std(gbold_seg) < 1e-10 or np.std(csf_seg) < 1e-10:
        return np.nan, np.nan, np.nan
    res = _xcorr(gbold_seg, csf_seg, tr=TR, max_lag_sec=max_lag)
    return float(res["neg_peak_r"]), float(abs(res["neg_peak_r"])), float(res["neg_peak_lag"])


def _s7_run_subject(sub, run_id, ts_root, rois, bands, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    rows = []
    for band_key, (lo, hi) in bands.items():
        # Determine max_lag: if config_hcp available, use BAND_LAG_SEC; else 20% of full run
        if _HAS_CFG_HCP and hasattr(_cfg_hcp, "BAND_LAG_SEC"):
            lag_key = {"broad": "standard_rs", "neurogenic": "neurogenic",
                       "vasogenic": "vasogenic"}.get(band_key, "standard_rs")
            max_lag = _cfg_hcp.BAND_LAG_SEC.get(lag_key, (hi - lo) * 5)
        else:
            max_lag = 20.0
        for roi in rois:
            roi_file = roi.replace("-", "_")
            gb = _s7_read_ts(ts_root, sub, run_id, f"gbold_{band_key}.csv")
            cs = _s7_read_ts(ts_root, sub, run_id, f"csf_{roi_file}_{band_key}.csv")
            if gb is None or cs is None:
                continue
            n = min(len(gb), len(cs))
            if n < 2 * HALF:
                log(f"  ✗ {sub} {band_key} {roi}: {n} vols")
                continue
            r_h1, a_h1, l_h1 = _s7_coupling(gb[:HALF],      cs[:HALF],      max_lag)
            r_h2, a_h2, l_h2 = _s7_coupling(gb[HALF:2*HALF], cs[HALF:2*HALF], max_lag)
            r_full, a_full, l_full = _s7_coupling(gb[:n], cs[:n], max_lag)
            lag_diff  = abs(l_h1 - l_h2) if np.isfinite(l_h1) and np.isfinite(l_h2) else np.nan
            absr_mean = np.nanmean([a_h1, a_h2])
            rows.append(dict(
                subject=sub, run=run_id, band=band_key, roi=roi,
                r_h1=r_h1, r_h2=r_h2, absr_h1=a_h1, absr_h2=a_h2,
                lag_h1=l_h1, lag_h2=l_h2,
                lag_instability=lag_diff, absr_mean=absr_mean,
                r_full=r_full, absr_full=a_full, lag_full=l_full,
            ))
    return rows


def _s7_p_str(p):
    if p < 0.001: return "p<0.001"
    elif p < 0.01: return f"p={p:.3f}"
    return f"p={p:.2f}"


def _s7_compute_correlations(df):
    rows = []
    group_cols = ["band", "roi"] if "roi" in df.columns else ["band"]
    for keys, grp in df.groupby(group_cols):
        if isinstance(keys, str): keys = (keys,)
        sub = grp.dropna(subset=["lag_instability", "absr_mean"])
        if len(sub) < 5: continue
        x = sub["absr_mean"].to_numpy()
        y = sub["lag_instability"].to_numpy()
        r_s, p_s = stats.spearmanr(x, y)
        row_dict = dict(zip(group_cols, keys))
        row_dict.update(spearman_r=float(r_s), spearman_p=float(p_s), n=len(sub))
        rows.append(row_dict)
    return pd.DataFrame(rows)


def _s7_compute_bins(df):
    rows = []
    group_cols = ["band", "roi"] if "roi" in df.columns else ["band"]
    for keys, grp in df.groupby(group_cols):
        if isinstance(keys, str): keys = (keys,)
        sub = grp.dropna(subset=["lag_instability", "absr_mean"]).copy()
        if len(sub) < _S7_N_BINS * 3: continue
        sub["absr_quartile"] = pd.qcut(sub["absr_mean"], q=_S7_N_BINS,
                                       labels=[f"Q{i+1}" for i in range(_S7_N_BINS)])
        for q, qgrp in sub.groupby("absr_quartile", observed=True):
            row_dict = dict(zip(group_cols, keys))
            row_dict.update(
                quartile=str(q), n=len(qgrp),
                mean_absr=float(qgrp["absr_mean"].mean()),
                mean_lag_instab=float(qgrp["lag_instability"].mean()),
                sem_lag_instab=float(qgrp["lag_instability"].sem()),
            )
            rows.append(row_dict)
    return pd.DataFrame(rows)


def _s7_make_scatter(df, df_corr, out):
    """6 ROI × 3 band scatter: lag_instability vs absr_mean."""
    # BUG FIXED: all constants from fig_style (no fallback blocks)
    roi_order   = fig_style.ROI_ORDER
    band_order  = fig_style.BAND_ORDER
    band_colors = fig_style.BAND_COLORS
    band_labels = fig_style.BAND_LABELS
    roi_colors  = fig_style.ROI_COLORS
    roi_labels  = fig_style.ROI_LABELS

    def clean(ax):
        ax.set_facecolor("white")
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        ax.spines["left"].set_linewidth(fig_style.LW_THIN)
        ax.spines["bottom"].set_linewidth(fig_style.LW_THIN)
        ax.tick_params(labelsize=fig_style.FS_TICK, length=3, width=fig_style.LW_THIN)

    fig = plt.figure(figsize=(fig_style.COL2, 1.65 * len(roi_order)), facecolor="white")
    gs  = gridspec.GridSpec(len(roi_order), len(band_order), figure=fig,
                             hspace=0.55, wspace=0.32,
                             left=0.10, right=0.97,
                             top=0.95, bottom=0.06)

    for ri, roi in enumerate(roi_order):
        for bi, band_key in enumerate(band_order):
            ax = fig.add_subplot(gs[ri, bi]); clean(ax)
            mask = (df["band"] == band_key)
            if "roi" in df.columns:
                mask = mask & (df["roi"] == roi)
            sub = df[mask].dropna(subset=["lag_instability", "absr_mean"])

            if len(sub) == 0:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=fig_style.FS_ANNOT,
                        color="#999999")
            else:
                x = sub["absr_mean"].to_numpy()
                y = sub["lag_instability"].to_numpy()
                color = band_colors.get(band_key, "steelblue")
                ax.scatter(x, y, s=fig_style.MS_MAIN ** 2, alpha=0.55,
                           color=color, linewidths=0, zorder=3)
                try:
                    m, b_i, _, _, _ = stats.linregress(x, y)
                    xfit = np.linspace(x.min(), x.max(), 100)
                    ax.plot(xfit, m * xfit + b_i, color="#333333",
                            linewidth=fig_style.LW_MAIN * 0.7, zorder=4)
                except Exception:
                    pass
                corr_mask = (df_corr["band"] == band_key)
                if "roi" in df_corr.columns:
                    corr_mask = corr_mask & (df_corr["roi"] == roi)
                crow = df_corr[corr_mask]
                if len(crow) > 0:
                    rho = float(crow["spearman_r"].iloc[0])
                    p_s = float(crow["spearman_p"].iloc[0])
                    ax.text(0.97, 0.97, f"ρ={rho:.2f}\n{_s7_p_str(p_s)}",
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=fig_style.FS_ANNOT, color=color, fontweight="bold")

            if bi == 0:
                ax.set_ylabel("|Δlag| (s)", fontsize=fig_style.FS_LABEL)
            if ri == len(roi_order) - 1:
                ax.set_xlabel("Mean |r|", fontsize=fig_style.FS_LABEL)
            if ri == 0:
                bname, bhz = band_labels.get(band_key, band_key).split("\n")
                ax.set_title(f"{bname}\n{bhz}", fontsize=fig_style.FS_TITLE,
                             fontweight="bold", color=band_colors.get(band_key, "k"), pad=3)
            if bi == 0:
                ax.text(-0.38, 0.5, roi_labels.get(roi, roi),
                        transform=ax.transAxes, ha="center", va="center", rotation=90,
                        fontsize=fig_style.FS_LABEL, fontweight="bold",
                        color=roi_colors.get(roi, "k"))

    # BUG FIXED: stem was "fig_lag_coupling_scatter"; now "FigS7_lag_scatter"
    fig_style.save(fig, "FigS7_lag_scatter", out)
    print(f"  ✓  FigS7_lag_scatter  →  {out}")


def _s7_make_bins(df_bins, out):
    roi_order   = fig_style.ROI_ORDER
    band_order  = fig_style.BAND_ORDER
    band_colors = fig_style.BAND_COLORS
    band_labels = fig_style.BAND_LABELS
    roi_colors  = fig_style.ROI_COLORS
    roi_labels  = fig_style.ROI_LABELS

    def clean(ax):
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=fig_style.FS_TICK)

    has_roi = "roi" in df_bins.columns
    n_roi   = len(roi_order) if has_roi else 1

    fig = plt.figure(figsize=(fig_style.COL2, 1.65 * n_roi), facecolor="white")
    gs  = gridspec.GridSpec(n_roi, len(band_order), figure=fig,
                             hspace=0.60, wspace=0.35,
                             left=0.12, right=0.97, top=0.93, bottom=0.08)
    roi_iter = roi_order if has_roi else [None]

    for ri, roi in enumerate(roi_iter):
        for bi, band_key in enumerate(band_order):
            ax = fig.add_subplot(gs[ri, bi]); clean(ax)
            color = band_colors.get(band_key, "steelblue")
            mask = df_bins["band"] == band_key
            if has_roi and roi is not None:
                mask = mask & (df_bins["roi"] == roi)
            sub = df_bins[mask]
            if len(sub) > 0:
                x = np.arange(len(sub))
                ax.bar(x, sub["mean_lag_instab"], yerr=sub["sem_lag_instab"],
                       color=color, alpha=0.75, capsize=3,
                       edgecolor="white", lw=0.4, zorder=3)
                ax.set_xticks(x)
                ax.set_xticklabels(sub["quartile"], fontsize=fig_style.FS_TICK)
            if ri == n_roi - 1:
                ax.set_xlabel("Coupling quartile", fontsize=fig_style.FS_LABEL)
            if bi == 0:
                ax.set_ylabel("|Δlag| mean ± SEM (s)", fontsize=fig_style.FS_LABEL)
            if ri == 0:
                bname, bhz = band_labels.get(band_key, band_key).split("\n")
                ax.set_title(f"{bname}\n{bhz}", fontsize=fig_style.FS_TITLE,
                             fontweight="bold", color=color, pad=3)
            if has_roi and roi is not None and bi == 0:
                ax.text(-0.45, 0.5, roi_labels.get(roi, roi),
                        transform=ax.transAxes, ha="center", va="center", rotation=90,
                        fontsize=fig_style.FS_LABEL, fontweight="bold",
                        color=roi_colors.get(roi, "k"))

    # BUG FIXED: stem was "fig_lag_coupling_bins"; now "FigS7_lag_bins"
    fig_style.save(fig, "FigS7_lag_bins", out)
    print(f"  ✓  FigS7_lag_bins  →  {out}")


def figS7_lag_scatter(args):
    print("\n── Fig S7: Lag instability vs coupling scatter ──────────────")
    if not HAS_XCORR:
        print("  SKIP: config_hcp.xcorr not available"); return
    _, _, out = _resolve_paths(args)

    subjects_csv = getattr(args, "subjects_csv", None) or \
                   os.path.join(args.root, "subjects/coupling_subjects_104.csv")
    ts_dir = getattr(args, "ts_dir", None) or \
             os.path.join(args.root, "data", "timeseries_primary")

    subjects = _s7_load_subjects(subjects_csv)
    rois     = _S7_ALL_ROIS
    bands    = _S7_BANDS

    print(f"  Subjects: {len(subjects)}, ROIs: {rois}, ts_dir: {ts_dir}")

    all_rows = []
    for i, sub in enumerate(subjects, 1):
        rows = _s7_run_subject(sub, args.run, ts_dir, rois, bands, verbose=False)
        all_rows.extend(rows)
        if i % 20 == 0:
            print(f"  [{i}/{len(subjects)}]")

    if not all_rows:
        print("  ✗ No data collected. Check timeseries directory."); return

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(out, "FigS7_lag_coupling_data.csv"), index=False)

    df_corr = _s7_compute_correlations(df)
    df_corr.to_csv(os.path.join(out, "FigS7_lag_coupling_correlations.csv"), index=False)

    df_bins = _s7_compute_bins(df)
    df_bins.to_csv(os.path.join(out, "FigS7_lag_coupling_bins.csv"), index=False)

    _s7_make_scatter(df, df_corr, out)
    _s7_make_bins(df_bins, out)



def figS9_roi_nuisance(args):
    print("\n── Fig S9: ROI nuisance sensitivity ────────────────────────")
    _, _, out = _resolve_paths(args)
    ca_dir = os.path.join(args.root, "results/coupling_g1")

    try:
        ratio_csv = _find_csv_prefix(ca_dir, "stage45_matched_ratios")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}"); return

    df = pd.read_csv(ratio_csv)
    if "roi" in df.columns:
        df["roi"] = df["roi"].str.replace("_", "-", regex=False)

    PIPELINES = [
        ("P1", "P1: Motion regression"),
        ("P2", "P2: Motion + WM mean"),
        ("P3", "P3: Motion + aCompCor"),
    ]
    ROW_LABELS = ["A", "B", "C"]
    ROI_ORDER  = fig_style.ROI_ORDER
    BAR_W = 0.62; GAP = 0.38
    X_CENTERS = np.arange(len(ROI_ORDER)) * (BAR_W + GAP)
    n_rows = len(PIPELINES)
    fig_h  = 1.9 * n_rows + 0.9

    fig = plt.figure(figsize=(fig_style.COL2, fig_h), facecolor="white")
    gs  = gridspec.GridSpec(n_rows, len(fig_style.BAND_ORDER), figure=fig,
                             hspace=0.55, wspace=0.30,
                             left=0.12, right=0.97,
                             top=0.93, bottom=0.14)

    for ri, (pl_key, pl_label) in enumerate(PIPELINES):
        pl_df  = df[df["pipeline"] == pl_key]
        all_df = df

        for bi, band in enumerate(fig_style.BAND_ORDER):
            ax = fig.add_subplot(gs[ri, bi])
            fig_style.clean_ax(ax)
            band_color = fig_style.BAND_COLORS[band]
            band_df    = pl_df[pl_df["band"] == band]

            for xi, roi in enumerate(ROI_ORDER):
                roi_color = fig_style.ROI_COLORS[roi]
                cx = X_CENTERS[xi]
                roi_df = band_df[band_df["roi"] == roi]
                p0_df  = all_df[(all_df["band"] == band) & (all_df["roi"] == roi)
                                & (df["pipeline"] == "P0")]
                shared_subs = roi_df["subject"].values
                p0_sub = p0_df[p0_df["subject"].isin(shared_subs)]
                px_r = roi_df["abs_r"].dropna().values
                p0_r = p0_sub["abs_r"].dropna().values

                if len(px_r) == 0 or len(p0_r) == 0 or np.mean(p0_r) == 0:
                    continue

                m = float(np.mean(px_r) / np.mean(p0_r))

                # Bootstrap 95% CI for group-level ratio
                rng = np.random.default_rng(42)
                n   = min(len(px_r), len(p0_r))
                boot = []
                for _ in range(2000):
                    idx  = rng.integers(0, n, n)
                    px_b = px_r[idx] if len(px_r) >= n else px_r
                    p0_b = p0_r[idx] if len(p0_r) >= n else p0_r
                    if np.mean(p0_b) > 0:
                        boot.append(np.mean(px_b) / np.mean(p0_b))
                # BUG FIXED: removed duplicate y_vals.append(m) (dead code)
                e = float(np.percentile(boot, 97.5) - np.percentile(boot, 2.5)) / 2

                ax.bar(cx, m, width=BAR_W, color=roi_color, alpha=0.88,
                       linewidth=0.8, edgecolor="white", zorder=3)
                ax.errorbar(cx, m, yerr=e, fmt="none", color="#333333",
                            lw=1.3, capsize=3.5, capthick=1.3, zorder=4)

            ax.axhline(1.0, color="#555555", lw=1.0, ls="--", zorder=2)
            ax.set_xticks(X_CENTERS)
            ax.set_xticklabels([fig_style.ROI_LABELS[r] for r in ROI_ORDER],
                                fontsize=fig_style.FS_TICK, rotation=35, ha="right")
            ax.set_xlim(X_CENTERS[0] - BAR_W, X_CENTERS[-1] + BAR_W)
            ax.set_ylim(0.45, 1.09)
            ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax.tick_params(axis="y", labelsize=fig_style.FS_TICK)

            if bi == 0:
                ax.set_ylabel("Coupling |r| retained", fontsize=fig_style.FS_LABEL + 1)
            if bi == 0:
                ax.text(0.02, 0.97, pl_label, transform=ax.transAxes,
                        fontsize=fig_style.FS_ANNOT, va="top", ha="left",
                        color="#333333", fontweight="bold")
            if ri == 0:
                bname, bhz = fig_style.BAND_LABELS[band].split("\n")
                ax.set_title(f"{bname}\n{bhz}", fontsize=fig_style.FS_TITLE,
                             fontweight="bold", color=band_color, pad=4)
            if bi == 0:
                ax.text(-0.28, 1.18, ROW_LABELS[ri], transform=ax.transAxes,
                        fontsize=fig_style.FS_PANEL, fontweight="bold", va="top")

    handles = [mpatches.Patch(facecolor=fig_style.ROI_COLORS[r], alpha=0.88,
                               edgecolor="white", linewidth=0.8,
                               label=fig_style.ROI_LABELS[r])
               for r in ROI_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(ROI_ORDER),
               fontsize=fig_style.FS_LEGEND, frameon=False,
               bbox_to_anchor=(0.54, 0.02), bbox_transform=fig.transFigure)

    # BUG FIXED: stem was "FigS_roi_nuisance"; now "FigS9_roi_nuisance"
    fig_style.save(fig, "FigS9_roi_nuisance", out)
    print(f"  ✓  FigS9_roi_nuisance  →  {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ══════════════════════════════════════════════════════════════════════════════

FIG_MAP = {
    "1":   fig1_spatial_gradient,
    "2":   fig2_icc,
    "3":   fig3_timewindow,
    "4":   fig4_nuisance,
    "5":   fig5_pipeline,
    "s3":  figS3_aaft_validation,
    "s5":  figS5_aaft_coupling,
    "s6":  figS6_csf_sampling_icc,
    "s7":  figS7_lag_scatter,
    "s9":  figS9_roi_nuisance,
    # aliases
    "spatial":    fig1_spatial_gradient,
    "icc":        fig2_icc,
    "timewindow": fig3_timewindow,
    "nuisance":   fig4_nuisance,
    "pipeline":   fig5_pipeline,
    "aaft":       figS3_aaft_validation,
    "coupling":   figS5_aaft_coupling,
    "roi_icc":    figS6_csf_sampling_icc,
    "lag":        figS7_lag_scatter,
    "roi_nuis":   figS9_roi_nuisance,
}

ALL_KEYS = ["1", "2", "3", "4", "5", "s3", "s5", "s6", "s7", "s9"]


def main():
    ap = argparse.ArgumentParser(
        description="Group 1 (104-subject) unified figure runner — NeuroImage gBOLD–CSF paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python code/figures_g1.py --all\n"
            "  python code/figures_g1.py --fig 1 2 3 4 5\n"
            "  python code/figures_g1.py --fig S3 S5 --ts_root results/timeseries\n"
            "  python code/figures_g1.py --fig S7 --subjects_csv coupling_subjects_104.csv\n"
        )
    )
    ap.add_argument("--all",    action="store_true", help="Generate all Group 1 figures")
    ap.add_argument("--fig",    nargs="+", metavar="FIG",
                    help="Figure(s): 1 2 3 4 5 S3 S5 S6 S7 S9")
    ap.add_argument("--root",   default=".",
                    help="Dataset root directory (default: current directory)")
    ap.add_argument("--outdir", default=None, help="Output directory")
    ap.add_argument("--run",    default="LR",  help="Scan run (default: LR)")
    # S3/S5 extra
    ap.add_argument("--ts_root", default=None, help="Timeseries root for AAFT figures")
    ap.add_argument("--summary", default=None, help="surrogate_summary.csv path")
    ap.add_argument("--n_surr",  type=int, default=1000,
                    help="AAFT surrogates per subject (default: 1000)")
    # S7 extra
    ap.add_argument("--subjects_csv", default=None,
                    help="Subject list CSV (default: <root>/coupling_subjects_104.csv)")
    ap.add_argument("--ts_dir",  default=None,
                    help="Timeseries root for S7 (default: <root>/results/timeseries)")
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
            print(f"  ERROR in fig '{key}': {e}")
            traceback.print_exc()

    print("\nAll done.")


if __name__ == "__main__":
    main()
