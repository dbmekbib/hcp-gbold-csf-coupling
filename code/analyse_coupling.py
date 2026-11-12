"""
analyse_coupling.py
═══════════════════════════════════════════════════════════════════════════════
Five-stage gBOLD–CSF coupling analysis.
Reads pre-extracted CSV timeseries from extract_timeseries.py.

  Stage 1  Unfiltered nuisance variance (R², detrended data)
  Stage 2  Frequency-specific nuisance variance (R² per band)
  Stage 3  Coupling strength, PSD, and split-half reproducibility (P0)
  Stage 4  Motion regression sensitivity (P1 vs P0 ratios)
  Stage 5  WM regression sensitivity (P2, P3 vs P0 ratios)

Data flow
─────────
All signals read from results/timeseries/{sub}_{run}/
  gbold_detrended.csv          gBOLD detrended, unfiltered
  wm_mean.csv                  WM mean, detrended, unfiltered
  acompcor.csv                 5 aCompCor PCs, detrended, unfiltered
  motion.csv                   6 motion parameters (mc.par)
  csf_{roi}_detrended.csv      CSF detrended, unfiltered   (roi = z0..z3, fv_lower, fv_upper)
  gbold_{band}.csv             gBOLD bandpassed per band
  csf_{roi}_{band}.csv         CSF bandpassed per band per ROI

Regression is done on-the-fly from detrended signals for Stages 4–5:
  P0  no regression (use pre-extracted bandpassed signals directly)
  P1  regress 6 motion from gbold_detrended → bandpass residuals
  P2  regress 6 motion + WM mean from gbold_detrended → bandpass residuals
  P3  regress 6 motion + 5 aCompCor from gbold_detrended → bandpass residuals

Results saved to results/coupling_analysis/
  stage1_nuisance_r2.csv
  stage2_freq_r2.csv
  stage3_coupling.csv
  stage3_psd.csv
  stage3_icc.csv
  stage3_timewindow.csv
  stage45_ratios.csv

Usage
─────
  # Full cohort, all stages
  python code/analyse_coupling.py \\
      --subjects_csv coupling_subjects_104.csv \\
      --ts_dir       results/timeseries \\
      --out          results/coupling_analysis

  # Single subject test
  python code/analyse_coupling.py \\
      --subjects_csv coupling_subjects_104.csv \\
      --subjects     100206 \\
      --stages       1 2 3 \\
      --ts_dir       results/timeseries \\
      --out          results/coupling_analysis
"""

import os, sys, argparse, csv, warnings
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import signal as spsig
from scipy import stats
from sklearn.linear_model import LinearRegression
from nilearn.signal import clean as nilearn_clean

warnings.filterwarnings("ignore", category=RuntimeWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import config_hcp as cfg
from utils_hcp import xcorr, deriv_xcorr

TR      = cfg.TR
FS      = 1.0 / TR   # sampling frequency (Hz)

# ── Constants ─────────────────────────────────────────────────────────────────

BANDS = {
    "broad":      (0.010, 0.100, cfg.BAND_LAG_SEC["standard_rs"]),
    "neurogenic": (0.020, 0.040, cfg.BAND_LAG_SEC["neurogenic"]),
    "endogenic":  (0.001, 0.020, cfg.BAND_LAG_SEC["endogenic"]),
    "vasogenic":  (0.060, 0.100, cfg.BAND_LAG_SEC["vasogenic"]),
}

ROIS = ["z0", "z1", "z2", "z3", "fv_lower", "fv_upper"]

# Primary CSF ROI for Stages 4–5
PRIMARY_ROI = "z0"

# Split-half: first half vs second half
N_VOLS      = 1190   # after dummy removal
HALF        = N_VOLS // 2   # 595

# Time windows for stability analysis (Stage 3C)
N_WINDOWS   = 4
WIN_LEN     = N_VOLS // N_WINDOWS   # ~297 volumes per window

# ICC benchmark thresholds (Zuo et al. 2010 NeuroImage)
ICC_POOR      = 0.40
ICC_MODERATE  = 0.60
ICC_GOOD      = 0.75


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_subjects(csv_path, subjects_arg):
    with open(csv_path) as f:
        all_subs = [row["subject"] for row in csv.DictReader(f)]
    return [s for s in subjects_arg
            if s in all_subs] if subjects_arg else all_subs


def _sub_dir(ts_root, sub, run):
    return os.path.join(ts_root, f"{sub}_{run}")


def _read(sub_dir, fname):
    """Load a single-column or multi-column CSV, return numpy array."""
    path = os.path.join(sub_dir, fname)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).values   # (T, ncols)


def _col(sub_dir, fname):
    """Load CSV and return first column as 1-D array."""
    arr = _read(sub_dir, fname)
    return arr[:, 0] if arr is not None else None


def _save(df, out_dir, fname):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, fname), index=False)
    print(f"  ✓ {fname}  ({len(df)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# Signal processing helpers
# ══════════════════════════════════════════════════════════════════════════════

# ============================================================
# Signal processing helpers
# ============================================================

def _r2(y, X):
    """
    OLS R² of multivariate regression X → y.
    """
    if X.ndim == 1:
        X = X[:, None]

    T, k = X.shape

    if T <= k + 1:
        return np.nan

    Xc = np.hstack([
        np.ones((T, 1)),
        X
    ])

    b = np.linalg.lstsq(
        Xc,
        y,
        rcond=None
    )[0]

    yh = Xc @ b

    ss_res = np.sum(
        (y - yh) ** 2
    )

    ss_tot = np.sum(
        (y - y.mean()) ** 2
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    return (
        float(np.clip(r2, 0.0, 1.0))
        if not np.isnan(r2)
        else np.nan
    )


# ============================================================
# NEW: nested / incremental nuisance R²
# ============================================================

def _incremental_r2(y, motion, wm, acompcor):
    """
    Compute nested nuisance-model R².

    Models
    ------
    M1:
        y ~ motion

    M2:
        y ~ motion + WM mean

    M3:
        y ~ motion + 5 aCompCor components

    Returns
    -------
    dict containing:

        r2_motion
        r2_motion_wm
        delta_r2_wm
        r2_motion_acompcor
        delta_r2_acompcor

    delta_r2_wm:
        additional variance associated with WM mean
        after accounting for motion.

    delta_r2_acompcor:
        additional variance associated with aCompCor
        after accounting for motion.
    """

    # Make sure WM is T × 1
    if wm.ndim == 1:
        wm = wm[:, None]

    # Motion-only model
    X_motion = motion

    # Motion + WM
    X_motion_wm = np.hstack([
        motion,
        wm
    ])

    # Motion + aCompCor
    X_motion_acomp = np.hstack([
        motion,
        acompcor
    ])

    # Fit nested models
    r2_motion = _r2(
        y,
        X_motion
    )

    r2_motion_wm = _r2(
        y,
        X_motion_wm
    )

    r2_motion_acomp = _r2(
        y,
        X_motion_acomp
    )

    # Incremental R²
    delta_wm = (
        r2_motion_wm
        - r2_motion
    )

    delta_acomp = (
        r2_motion_acomp
        - r2_motion
    )

    return {
        "r2_motion":
            r2_motion,

        "r2_motion_wm":
            r2_motion_wm,

        "delta_r2_wm":
            delta_wm,

        "r2_motion_acompcor":
            r2_motion_acomp,

        "delta_r2_acompcor":
            delta_acomp,
    }




def _regress_out(y, X):
    """
    Remove variance of X from y via OLS.
    Returns residuals (T,) with same mean as y.
    """
    if X.ndim == 1:
        X = X[:, None]
    mu  = y.mean()
    Xc  = np.hstack([np.ones((len(y), 1)), X])
    b   = np.linalg.lstsq(Xc, y, rcond=None)[0]
    return y - (Xc @ b) + mu


def _bandpass(ts, lo, hi):
    """
    Bandpass a 1-D timeseries using nilearn (matches pipeline bandpass).
    Returns filtered array same length as ts.
    """
    ts2d = ts.reshape(-1, 1).astype(float)
    out  = nilearn_clean(ts2d, detrend=False, standardize=None,
                         high_pass=lo, low_pass=hi, t_r=TR)
    return out[:, 0]


def _psd_welch(ts, nperseg=None):
    """
    Welch PSD of a 1-D timeseries.
    Returns (freqs, power) both 1-D arrays.

    nperseg = full timeseries length (no averaging across segments).
    This maximises frequency resolution — critical for the narrow
    endogenic (0.001–0.020 Hz) and neurogenic (0.020–0.040 Hz) bands.

    At TR=0.72s, N=1190:
      resolution = 1/(1190 × 0.72) = 0.00117 Hz
      endogenic band → 17 frequency points
      neurogenic band → 17 frequency points
      vasogenic band  → 34 frequency points
    """
    if nperseg is None:
        nperseg = len(ts)          # full length = maximum resolution
    # noverlap=0 + nperseg=len(ts) → single-segment periodogram.
    # Maximises frequency resolution at cost of spectral variance.
    # Variance is reduced by averaging across subjects at group level.
    f, p = spsig.welch(ts, fs=FS, nperseg=nperseg,
                       noverlap=0, scaling="density")
    return f, p


def _band_power(freqs, power, lo, hi):
    """
    Integrate PSD within [lo, hi] Hz band.
    Uses trapezoid rule when ≥2 points exist.
    Falls back to power × band_width when only 1 point exists
    (can happen at coarse frequency resolution).
    """
    mask = (freqs >= lo) & (freqs <= hi)
    n    = mask.sum()
    if n == 0:
        return np.nan
    if n == 1:
        # Single bin: approximate as power × bin width
        df = freqs[1] - freqs[0] if len(freqs) > 1 else (hi - lo)
        return float(power[mask][0] * df)
    return float(np.trapz(power[mask], freqs[mask]))


def _icc21(half1, half2):
    """
    ICC(2,1) — two-way random effects, single measurement, absolute agreement.
    half1, half2: 1-D arrays of length N (one value per subject).
    Returns (icc, ci_low, ci_high) at 95% CI.
    Formula: Shrout & Fleiss (1979).
    """
    n    = len(half1)
    data = np.column_stack([half1, half2])
    grand_mean = data.mean()
    row_means  = data.mean(axis=1)
    col_means  = data.mean(axis=0)

    ss_r = 2 * np.sum((row_means - grand_mean) ** 2)
    ss_c = n * np.sum((col_means - grand_mean) ** 2)
    ss_e = np.sum((data - row_means[:, None] -
                   col_means[None, :] + grand_mean) ** 2)
    ss_t = np.sum((data - grand_mean) ** 2)

    ms_r = ss_r / (n - 1)
    ms_c = ss_c / 1           # k - 1 = 1
    ms_e = ss_e / (n - 1)

    icc  = (ms_r - ms_e) / (ms_r + ms_e + (2 / n) * (ms_c - ms_e))

    # 95% CI via F-distribution
    k    = 2
    F    = ms_r / ms_e
    df1, df2 = n - 1, (n - 1) * (k - 1)
    fl   = F / stats.f.ppf(0.975, df1, df2)
    fu   = F * stats.f.ppf(0.975, df2, df1)
    ci_l = (fl - 1) / (fl + k - 1)
    ci_h = (fu - 1) / (fu + k - 1)
    return float(icc), float(ci_l), float(ci_h)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Unfiltered nuisance variance
# ══════════════════════════════════════════════════════════════════════════════

def stage1_one(sd):
    """
    Compute R²(nuisance → target) for one subject.
    Returns list of dicts, one per (target, nuisance_model).
    """
    gbold   = _col(sd, "gbold_detrended.csv")
    wm      = _col(sd, "wm_mean.csv")
    motion  = _read(sd, "motion.csv")        # (T, 6)
    acompcor = _read(sd, "acompcor.csv")     # (T, 5)

    if any(x is None for x in [gbold, wm, motion, acompcor]):
        return None

    targets = {"gbold": gbold}
    for roi in ROIS:
        ts = _col(sd, f"csf_{roi}_detrended.csv")
        if ts is not None:
            targets[f"csf_{roi}"] = ts

    rows = []
    for tname, ty in targets.items():
        rows.append(dict(target=tname, nuisance="6_motion",
                         r2=_r2(ty, motion)))
        rows.append(dict(target=tname, nuisance="wm_mean",
                         r2=_r2(ty, wm)))
        rows.append(dict(target=tname, nuisance="5_acompcor",
                         r2=_r2(ty, acompcor)))
    return rows


def run_stage1(subjects, run, ts_root, out_dir):
    print("\n" + "="*60)
    print("Stage 1 — Unfiltered nuisance R²")
    print("="*60)

    all_rows = []
    for sub in subjects:
        sd   = _sub_dir(ts_root, sub, run)
        rows = stage1_one(sd)
        if rows is None:
            print(f"  ✗ {sub} — missing files, skip")
            continue
        for r in rows:
            r["subject"] = sub
        all_rows.extend(rows)
        print(f"  ✓ {sub}")

    df = pd.DataFrame(all_rows)
    _save(df, out_dir, "stage1_nuisance_r2.csv")

    # Group summary
    summary = (df.groupby(["target", "nuisance"])["r2"]
                 .agg(n="count", mean_r2="mean", sd_r2="std")
                 .reset_index())
    _save(summary, out_dir, "stage1_nuisance_r2_summary.csv")
    print("\n  Group summary (mean R²):")
    print(summary.to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Frequency-specific nuisance variance
# ══════════════════════════════════════════════════════════════════════════════

def stage2_one(sd):
    """
    R²(raw nuisance → bandpassed target) per band per ROI.
    Nuisance regressors are NOT bandpassed — they stay raw.
    """
    wm       = _col(sd, "wm_mean.csv")
    motion   = _read(sd, "motion.csv")
    acompcor = _read(sd, "acompcor.csv")

    if any(x is None for x in [wm, motion, acompcor]):
        return None

    rows = []
    rows_incremental = []
    for band, (lo, hi, _) in BANDS.items():
        # bandpassed targets
        gbold_bp = _col(sd, f"gbold_{band}.csv")
        if gbold_bp is None:
            continue

        targets = {"gbold": gbold_bp}
        for roi in ROIS:
            ts = _col(sd, f"csf_{roi}_{band}.csv")
            if ts is not None:
                targets[f"csf_{roi}"] = ts

        for tname, ty in targets.items():
            rows.append(dict(band=band, target=tname,
                             nuisance="6_motion",   r2=_r2(ty, motion)))
            rows.append(dict(band=band, target=tname,
                             nuisance="wm_mean",    r2=_r2(ty, wm)))
            rows.append(dict(band=band, target=tname,
                             nuisance="5_acompcor", r2=_r2(ty, acompcor)))
    return rows

# ── NEW: frequency-specific incremental R² (original Stage 2 untouched) ────────
def stage2_incremental_one(sd):
    wm=_col(sd,"wm_mean.csv"); motion=_read(sd,"motion.csv"); acompcor=_read(sd,"acompcor.csv")
    if any(x is None for x in [wm,motion,acompcor]): return None
    rows=[]
    for band,(lo,hi,_) in BANDS.items():
        gbold=_col(sd,f"gbold_{band}.csv")
        if gbold is None: continue
        targets={"gbold":gbold}
        for roi in ROIS:
            ts=_col(sd,f"csf_{roi}_{band}.csv")
            if ts is not None: targets[f"csf_{roi}"]=ts

        # Match nuisance regressors to the same frequency band as the target
        motion_bp=np.column_stack([_bandpass(motion[:,j],lo,hi) for j in range(motion.shape[1])])
        wm_bp=_bandpass(wm,lo,hi)
        acompcor_bp=np.column_stack([_bandpass(acompcor[:,j],lo,hi) for j in range(acompcor.shape[1])])

        for target,y in targets.items():
            r=_incremental_r2(y,motion_bp,wm_bp,acompcor_bp)
            rows.append(dict(band=band,target=target,**r))
    return rows
# ── NEW: run/save incremental Stage 2 without altering original outputs ────────
def run_stage2_incremental(subjects,run,ts_root,out_dir):
    print("\n"+"="*60); print("Stage 2B — Frequency-specific incremental nuisance R²"); print("="*60)
    rows=[]
    for sub in subjects:
        sd=_sub_dir(ts_root,sub,run); rr=stage2_incremental_one(sd)
        if rr is None: print(f"  ✗ {sub} — missing files, skip"); continue
        for r in rr: r["subject"]=sub
        rows.extend(rr); print(f"  ✓ {sub}")

    df=pd.DataFrame(rows)
    _save(df,out_dir,"stage2_freq_incremental_r2.csv")

    summary=(df.groupby(["band","target"]).agg(
        n=("subject","count"),
        mean_r2_motion=("r2_motion","mean"),sd_r2_motion=("r2_motion","std"),
        mean_r2_motion_wm=("r2_motion_wm","mean"),
        mean_delta_r2_wm=("delta_r2_wm","mean"),sd_delta_r2_wm=("delta_r2_wm","std"),
        mean_r2_motion_acompcor=("r2_motion_acompcor","mean"),
        mean_delta_r2_acompcor=("delta_r2_acompcor","mean"),sd_delta_r2_acompcor=("delta_r2_acompcor","std")
    ).reset_index())

    _save(summary,out_dir,"stage2_freq_incremental_r2_summary.csv")
    print("\n  Incremental R² summary:"); print(summary.to_string(index=False))
    return df

def run_stage2(subjects, run, ts_root, out_dir):
    print("\n" + "="*60)
    print("Stage 2 — Frequency-specific nuisance R²")
    print("="*60)

    all_rows = []
    for sub in subjects:
        sd   = _sub_dir(ts_root, sub, run)
        rows = stage2_one(sd)
        if rows is None:
            print(f"  ✗ {sub} — missing files, skip")
            continue
        for r in rows:
            r["subject"] = sub
        all_rows.extend(rows)
        print(f"  ✓ {sub}")

    df = pd.DataFrame(all_rows)
    _save(df, out_dir, "stage2_freq_r2.csv")

    summary = (df.groupby(["band", "target", "nuisance"])["r2"]
                 .agg(n="count", mean_r2="mean", sd_r2="std")
                 .reset_index())
    _save(summary, out_dir, "stage2_freq_r2_summary.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Coupling, PSD, reproducibility (P0)
# ══════════════════════════════════════════════════════════════════════════════

def stage3_coupling_one(sd):
    """
    3A: coupling strength (|r|, lag) per band per ROI.
    3B: split-half coupling for each half.
    Returns (coupling_rows, splithalf_rows).
    """
    coupling_rows   = []
    splithalf_rows  = []

    for band, (lo, hi, max_lag) in BANDS.items():
        gbold_bp = _col(sd, f"gbold_{band}.csv")
        if gbold_bp is None:
            continue

        for roi in ROIS:
            csf_bp = _col(sd, f"csf_{roi}_{band}.csv")
            if csf_bp is None:
                continue

            # ── Full timeseries coupling (xcorr + deriv_xcorr in parallel) ──
            cc  = xcorr(gbold_bp, csf_bp, tr=TR, max_lag_sec=max_lag)
            dcc = deriv_xcorr(gbold_bp, csf_bp, tr=TR, max_lag_sec=max_lag)
            coupling_rows.append(dict(
                band=band, roi=roi,
                # xcorr (gBOLD, CSF) — negative peak (primary: gBOLD leads CSF)
                peak_r        = cc["neg_peak_r"],
                abs_r         = abs(cc["neg_peak_r"]),
                peak_lag      = cc["neg_peak_lag"],
                # xcorr (gBOLD, CSF) — positive peak (secondary: CSF leads gBOLD)
                pos_peak_r    = cc["pos_peak_r"],
                pos_abs_r     = abs(cc["pos_peak_r"]),
                pos_peak_lag  = cc["pos_peak_lag"],
                # deriv xcorr (-d/dt gBOLD, CSF) — positive peak (primary)
                deriv_peak_r  = dcc["pos_peak_r"],
                deriv_abs_r   = abs(dcc["pos_peak_r"]),
                deriv_peak_lag= dcc["pos_peak_lag"],
                # deriv xcorr — negative peak (secondary)
                deriv_neg_r   = dcc["neg_peak_r"],
                deriv_neg_lag = dcc["neg_peak_lag"],
                amplitude_gbold = float(np.std(gbold_bp)),
                amplitude_csf   = float(np.std(csf_bp)),
            ))

            # ── Split-half coupling ───────────────────────────────────────────
            for half_id, (g, c) in enumerate([
                (gbold_bp[:HALF],  csf_bp[:HALF]),
                (gbold_bp[HALF:],  csf_bp[HALF:]),
            ]):
                cc_h  = xcorr(g, c, tr=TR, max_lag_sec=max_lag)
                dcc_h = deriv_xcorr(g, c, tr=TR, max_lag_sec=max_lag)
                splithalf_rows.append(dict(
                    band=band, roi=roi, half=half_id + 1,
                    # xcorr — negative peak (primary)
                    peak_r        = cc_h["neg_peak_r"],
                    abs_r         = abs(cc_h["neg_peak_r"]),
                    peak_lag      = cc_h["neg_peak_lag"],
                    # xcorr — positive peak (secondary)
                    pos_peak_r    = cc_h["pos_peak_r"],
                    pos_abs_r     = abs(cc_h["pos_peak_r"]),
                    pos_peak_lag  = cc_h["pos_peak_lag"],
                    # deriv xcorr — positive peak (primary)
                    deriv_peak_r  = dcc_h["pos_peak_r"],
                    deriv_abs_r   = abs(dcc_h["pos_peak_r"]),
                    deriv_peak_lag= dcc_h["pos_peak_lag"],
                    # deriv xcorr — negative peak (secondary)
                    deriv_neg_r   = dcc_h["neg_peak_r"],
                    deriv_neg_lag = dcc_h["neg_peak_lag"],
                    amplitude_gbold = float(np.std(g)),
                    amplitude_csf   = float(np.std(c)),
                ))

    return coupling_rows, splithalf_rows


def stage3_psd_one(sd):
    """
    3B: PSD per signal (gBOLD, each CSF ROI, WM) from detrended unfiltered.
    Returns list of dicts with band power per ROI.
    """
    rows = []
    signals = {"gbold": _col(sd, "gbold_detrended.csv"),
               "wm":    _col(sd, "wm_mean.csv")}
    for roi in ROIS:
        ts = _col(sd, f"csf_{roi}_detrended.csv")
        if ts is not None:
            signals[f"csf_{roi}"] = ts

    for sig_name, ts in signals.items():
        if ts is None:
            continue
        freqs, power = _psd_welch(ts)
        row = dict(signal=sig_name)
        for band, (lo, hi, _) in BANDS.items():
            row[f"power_{band}"] = _band_power(freqs, power, lo, hi)
        rows.append(row)
    return rows


def stage3_timewindow_one(sd):
    """
    3C: coupling per non-overlapping time window (scan duration stability).
    Returns list of dicts.
    """
    rows = []
    for band, (lo, hi, max_lag) in BANDS.items():
        gbold_bp = _col(sd, f"gbold_{band}.csv")
        if gbold_bp is None:
            continue
        csf_bp = _col(sd, f"csf_{PRIMARY_ROI}_{band}.csv")
        if csf_bp is None:
            continue
        # Endogenic (0.001–0.020 Hz) has period up to 1000s.
        # A 3.6-min window cannot reliably capture this — flag in output.
        win_too_short = (lo < 0.005 and WIN_LEN * TR < 300)

        for wi in range(N_WINDOWS):
            s = wi * WIN_LEN
            e = s + WIN_LEN
            if e > len(gbold_bp):
                break
            g, c = gbold_bp[s:e], csf_bp[s:e]
            # Cap max_lag at 25% of window duration to avoid unreliable lags
            win_dur  = WIN_LEN * TR
            safe_lag = min(max_lag, win_dur * 0.25)
            cc   = xcorr(g, c, tr=TR, max_lag_sec=safe_lag)
            dcc  = deriv_xcorr(g, c, tr=TR, max_lag_sec=safe_lag)
            rows.append(dict(
                band=band, roi=PRIMARY_ROI,
                window=wi + 1,
                t_start_s    = s * TR,
                t_end_s      = e * TR,
                # xcorr — negative peak (primary)
                peak_r        = cc["neg_peak_r"],
                abs_r         = abs(cc["neg_peak_r"]),
                peak_lag      = cc["neg_peak_lag"],
                # xcorr — positive peak (secondary)
                pos_peak_r    = cc["pos_peak_r"],
                pos_abs_r     = abs(cc["pos_peak_r"]),
                pos_peak_lag  = cc["pos_peak_lag"],
                # deriv xcorr — positive peak (primary)
                deriv_peak_r  = dcc["pos_peak_r"],
                deriv_abs_r   = abs(dcc["pos_peak_r"]),
                deriv_peak_lag= dcc["pos_peak_lag"],
                # deriv xcorr — negative peak (secondary)
                deriv_neg_r   = dcc["neg_peak_r"],
                deriv_neg_lag = dcc["neg_peak_lag"],
                window_ok    = not win_too_short,
            ))
    return rows


def run_stage3(subjects, run, ts_root, out_dir):
    print("\n" + "="*60)
    print("Stage 3 — Coupling strength, PSD, reproducibility (P0)")
    print("="*60)

    coup_rows, sh_rows, psd_rows, tw_rows = [], [], [], []

    for sub in subjects:
        sd = _sub_dir(ts_root, sub, run)

        cr, shr = stage3_coupling_one(sd)
        for r in cr + shr:
            r["subject"] = sub
        coup_rows.extend(cr)
        sh_rows.extend(shr)

        for r in stage3_psd_one(sd):
            r["subject"] = sub
            psd_rows.append(r)

        for r in stage3_timewindow_one(sd):
            r["subject"] = sub
            tw_rows.append(r)

        print(f"  ✓ {sub}")

    # ── Save per-subject results ──────────────────────────────────────────────
    _save(pd.DataFrame(coup_rows), out_dir, "stage3_coupling.csv")
    _save(pd.DataFrame(sh_rows),   out_dir, "stage3_splithalf.csv")
    _save(pd.DataFrame(psd_rows),  out_dir, "stage3_psd.csv")
    _save(pd.DataFrame(tw_rows),   out_dir, "stage3_timewindow.csv")

    # ── ICC(2,1): split-half reproducibility ─────────────────────────────────
    print("\n  Computing ICC(2,1) ...")
    icc_rows = []
    shdf = pd.DataFrame(sh_rows)

    for band in BANDS:
        for roi in ROIS:
            for metric in ["abs_r", "peak_lag",
                           "pos_abs_r", "pos_peak_lag",
                           "deriv_abs_r", "deriv_peak_lag",
                           "deriv_neg_r", "deriv_neg_lag",
                           "amplitude_gbold", "amplitude_csf"]:
                h1 = (shdf[(shdf["band"] == band) &
                           (shdf["roi"]  == roi)  &
                           (shdf["half"] == 1)][metric]
                       .values)
                h2 = (shdf[(shdf["band"] == band) &
                           (shdf["roi"]  == roi)  &
                           (shdf["half"] == 2)][metric]
                       .values)
                if len(h1) < 2 or len(h2) < 2:
                    continue
                # Align subjects (same order in both halves)
                min_n = min(len(h1), len(h2))
                icc, ci_l, ci_h = _icc21(h1[:min_n], h2[:min_n])
                icc_rows.append(dict(
                    band=band, roi=roi, metric=metric,
                    n=min_n, icc=icc, ci_low=ci_l, ci_high=ci_h,
                    benchmark_poor=ICC_POOR,
                    benchmark_good=ICC_GOOD,
                    meets_good=(icc >= ICC_GOOD),
                ))

    icc_df = pd.DataFrame(icc_rows)
    _save(icc_df, out_dir, "stage3_icc.csv")

    print("\n  ICC summary (coupling |r|):")
    if icc_df.empty or "metric" not in icc_df.columns:
        print("  (not enough subjects for ICC — need N≥2)")
    else:
        sub_icc = icc_df[icc_df["metric"] == "abs_r"][
            ["band", "roi", "n", "icc", "ci_low", "ci_high", "meets_good"]]
        print(sub_icc.to_string(index=False))

    return pd.DataFrame(coup_rows)


# ══════════════════════════════════════════════════════════════════════════════
# Stages 4–5 — Pipeline sensitivity (P1, P2, P3 vs P0)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_pipeline_coupling(sd, pipeline_label, band, lo, hi, max_lag,
                               roi=PRIMARY_ROI):
    """
    Matched nuisance processing of gBOLD and CSF.

    P0: original bandpassed gBOLD + CSF
    P1: motion regression from both signals
    P2: motion + WM mean regression from both signals
    P3: motion + 5 aCompCor regression from both signals

    Regression is performed on detrended unfiltered signals,
    followed by identical bandpass filtering.
    """

    if pipeline_label == "P0":

        gbold_bp = _col(sd, f"gbold_{band}.csv")
        csf_bp   = _col(sd, f"csf_{roi}_{band}.csv")

        if gbold_bp is None or csf_bp is None:
            return None

    else:

        gbold_det = _col(sd, "gbold_detrended.csv")
        csf_det   = _col(sd, f"csf_{roi}_detrended.csv")
        motion    = _read(sd, "motion.csv")

        if gbold_det is None or csf_det is None or motion is None:
            return None

        if pipeline_label == "P1":
            confounds = motion

        elif pipeline_label == "P2":
            wm = _col(sd, "wm_mean.csv")
            if wm is None:
                return None
            confounds = np.hstack([motion, wm[:, None]])

        elif pipeline_label == "P3":
            acompcor = _read(sd, "acompcor.csv")
            if acompcor is None:
                return None
            confounds = np.hstack([motion, acompcor])

        else:
            return None

        gbold_resid = _regress_out(gbold_det, confounds)
        csf_resid   = _regress_out(csf_det,   confounds)

        gbold_bp = _bandpass(gbold_resid, lo, hi)
        csf_bp   = _bandpass(csf_resid,   lo, hi)

    n = min(len(gbold_bp), len(csf_bp))
    gbold_bp = gbold_bp[:n]
    csf_bp   = csf_bp[:n]

    cc  = xcorr(gbold_bp, csf_bp, tr=TR, max_lag_sec=max_lag)
    dcc = deriv_xcorr(gbold_bp, csf_bp, tr=TR, max_lag_sec=max_lag)

    return dict(
        peak_r         = cc["neg_peak_r"],
        abs_r          = abs(cc["neg_peak_r"]),
        peak_lag       = cc["neg_peak_lag"],

        pos_peak_r     = cc["pos_peak_r"],
        pos_abs_r      = abs(cc["pos_peak_r"]),
        pos_peak_lag   = cc["pos_peak_lag"],

        deriv_peak_r   = dcc["pos_peak_r"],
        deriv_abs_r    = abs(dcc["pos_peak_r"]),
        deriv_peak_lag = dcc["pos_peak_lag"],

        deriv_neg_r    = dcc["neg_peak_r"],
        deriv_neg_lag  = dcc["neg_peak_lag"],

        gbold_var      = float(np.var(gbold_bp)),
        csf_var        = float(np.var(csf_bp)),
    )


def run_stage45(subjects, run, ts_root, out_dir):
    print("\n" + "="*60)
    print("Stages 4–5 — Pipeline sensitivity (P1, P2, P3 vs P0)")
    print("="*60)

    pipelines = ["P0", "P1", "P2", "P3"]
    all_rows  = []

    for sub in subjects:
        sd  = _sub_dir(ts_root, sub, run)
        sub_rows = {}   # (band, roi, pipeline) → result dict

        for band, (lo, hi, max_lag) in BANDS.items():
            for roi in ROIS:
                for pl in pipelines:
                    res = _compute_pipeline_coupling(
                        sd, pl, band, lo, hi, max_lag, roi=roi)
                    if res is not None:
                        sub_rows[(band, roi, pl)] = res

        # ── Compute ratios relative to P0 ────────────────────────────────────
        for (band, roi, pl), res in sub_rows.items():
            p0 = sub_rows.get((band, roi, "P0"))
            if p0 is None:
                continue

            # Avoid division by zero
            r0   = p0["abs_r"]    if abs(p0["abs_r"])    > 1e-6 else np.nan
            lag0 = p0["peak_lag"] if abs(p0["peak_lag"]) > 1e-6 else np.nan
            gvar0 = p0["gbold_var"] if p0["gbold_var"] > 1e-10 else np.nan
            cvar0 = p0["csf_var"]   if p0["csf_var"]   > 1e-10 else np.nan

            # Positive xcorr P0 denominators
            pr0   = p0["pos_abs_r"]   if abs(p0.get("pos_abs_r",0))   > 1e-6 else np.nan
            plag0 = p0["pos_peak_lag"] if abs(p0.get("pos_peak_lag",0)) > 1e-6 else np.nan
            # Deriv P0 denominators
            dr0   = p0["deriv_abs_r"]    if abs(p0.get("deriv_abs_r",0))    > 1e-6 else np.nan
            dlag0 = p0["deriv_peak_lag"] if abs(p0.get("deriv_peak_lag",0)) > 1e-6 else np.nan

            all_rows.append(dict(
                subject        = sub,
                band           = band,
                roi            = roi,
                pipeline       = pl,
                # xcorr — negative peak (primary)
                abs_r          = res["abs_r"],
                peak_r         = res["peak_r"],
                peak_lag       = res["peak_lag"],
                # xcorr — positive peak (secondary)
                pos_abs_r      = res.get("pos_abs_r", np.nan),
                pos_peak_r     = res.get("pos_peak_r", np.nan),
                pos_peak_lag   = res.get("pos_peak_lag", np.nan),
                # deriv xcorr — positive peak (primary)
                deriv_abs_r    = res.get("deriv_abs_r", np.nan),
                deriv_peak_r   = res.get("deriv_peak_r", np.nan),
                deriv_peak_lag = res.get("deriv_peak_lag", np.nan),
                # deriv xcorr — negative peak (secondary)
                deriv_neg_r    = res.get("deriv_neg_r", np.nan),
                deriv_neg_lag  = res.get("deriv_neg_lag", np.nan),
                gbold_var      = res["gbold_var"],
                csf_var        = res["csf_var"],
                # Ratios vs P0 — xcorr negative peak
                ratio_r        = res["abs_r"]    / r0   if not np.isnan(r0)   else np.nan,
                ratio_lag      = abs(res["peak_lag"]) / abs(lag0) if not np.isnan(lag0) else np.nan,
                ratio_gbold_var = res["gbold_var"] / gvar0 if not np.isnan(gvar0) else np.nan,
                ratio_csf_var   = res["csf_var"]   / cvar0 if not np.isnan(cvar0) else np.nan,
                # Ratios vs P0 — xcorr positive peak
                pos_ratio_r    = res.get("pos_abs_r", np.nan) / pr0   if not np.isnan(pr0)   else np.nan,
                pos_ratio_lag  = abs(res.get("pos_peak_lag", np.nan)) / abs(plag0) if not np.isnan(plag0) else np.nan,
                # Ratios vs P0 — deriv positive peak
                deriv_ratio_r  = res.get("deriv_abs_r", np.nan) / dr0   if not np.isnan(dr0)   else np.nan,
                deriv_ratio_lag= abs(res.get("deriv_peak_lag", np.nan)) / abs(dlag0) if not np.isnan(dlag0) else np.nan,
            ))

        print(f"  ✓ {sub}")

    df = pd.DataFrame(all_rows)
    _save(df, out_dir, "stage45_matched_ratios.csv")

    # ── Group summary of ratios ───────────────────────────────────────────────
    summary = (df[df["pipeline"] != "P0"]
               .groupby(["band", "roi", "pipeline"])
               .agg(
                   n                   = ("subject",         "count"),
                   # xcorr — negative peak (primary)
                   mean_abs_r          = ("abs_r",           "mean"),
                   sd_abs_r            = ("abs_r",           "std"),
                   mean_ratio_r        = ("ratio_r",         "mean"),
                   sd_ratio_r          = ("ratio_r",         "std"),
                   mean_ratio_lag      = ("ratio_lag",       "mean"),
                   sd_ratio_lag        = ("ratio_lag",       "std"),
                   mean_ratio_gbold_var = ("ratio_gbold_var", "mean"),
                   sd_ratio_gbold_var   = ("ratio_gbold_var", "std"),

                   mean_ratio_csf_var   = ("ratio_csf_var", "mean"),
                   sd_ratio_csf_var     = ("ratio_csf_var", "std"),
                   # xcorr — positive peak (secondary)
                   mean_pos_abs_r      = ("pos_abs_r",       "mean"),
                   sd_pos_abs_r        = ("pos_abs_r",       "std"),
                   mean_pos_ratio_r    = ("pos_ratio_r",     "mean"),
                   sd_pos_ratio_r      = ("pos_ratio_r",     "std"),
                   mean_pos_ratio_lag  = ("pos_ratio_lag",   "mean"),
                   sd_pos_ratio_lag    = ("pos_ratio_lag",   "std"),
                   # deriv xcorr — positive peak (primary)
                   mean_deriv_abs_r    = ("deriv_abs_r",     "mean"),
                   sd_deriv_abs_r      = ("deriv_abs_r",     "std"),
                   mean_deriv_ratio_r  = ("deriv_ratio_r",   "mean"),
                   sd_deriv_ratio_r    = ("deriv_ratio_r",   "std"),
                   mean_deriv_ratio_lag= ("deriv_ratio_lag", "mean"),
                   sd_deriv_ratio_lag  = ("deriv_ratio_lag", "std"),
               )
               .reset_index())

    # Paired t-test vs P0 for each band × roi × pipeline
    ttest_rows = []
    for (band, roi), grp in df.groupby(["band", "roi"]):
        p0_r = grp[grp["pipeline"] == "P0"]["abs_r"].values
        for pl in ["P1", "P2", "P3"]:
            pn_r = grp[grp["pipeline"] == pl]["abs_r"].values
            n    = min(len(p0_r), len(pn_r))
            if n < 4:
                continue
            t, p = stats.ttest_rel(p0_r[:n], pn_r[:n])
            ttest_rows.append(dict(
                band=band, roi=roi, pipeline=pl,
                n=n, t_stat=round(t, 3), p_val=round(p, 4),
                significant=(p < 0.05),
            ))

    _save(summary,                  out_dir, "stage45_matched_ratios_summary.csv")
    _save(pd.DataFrame(ttest_rows), out_dir, "stage45_matched_ttest.csv")

    print(f"\n  Matched-pipeline ratio summary (primary ROI: {PRIMARY_ROI}, coupling |r|):")

    pri = summary[(summary["roi"] == PRIMARY_ROI)][
        ["band", "pipeline", "n",
        "mean_ratio_r", "sd_ratio_r",
        "mean_ratio_lag", "sd_ratio_lag",
        "mean_ratio_gbold_var", "sd_ratio_gbold_var",
        "mean_ratio_csf_var", "sd_ratio_csf_var",
        "mean_deriv_abs_r", "sd_deriv_abs_r",
        "mean_deriv_ratio_r", "sd_deriv_ratio_r"]]

    print(pri.to_string(index=False))

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="5-stage gBOLD–CSF coupling analysis")
    ap.add_argument("--subjects_csv", required=True)
    ap.add_argument("--subjects",     nargs="+", default=None)
    ap.add_argument("--run",          default="LR")
    ap.add_argument("--ts_dir",       default="results/timeseries",
                    help="Root directory of extracted timeseries CSVs")
    ap.add_argument("--out",          default="results/coupling_analysis")
    ap.add_argument("--stages",       nargs="+", type=int,
                    default=[1, 2, 3, 4, 5],
                    choices=[1, 2, 3, 4, 5],
                    help="Which stages to run (default: all)")
    args = ap.parse_args()

    subjects = _load_subjects(args.subjects_csv, args.subjects)
    os.makedirs(args.out, exist_ok=True)

    print(f"\nanalyse_coupling.py")
    print(f"  Subjects : {len(subjects)}")
    print(f"  Run      : {args.run}")
    print(f"  TS dir   : {args.ts_dir}")
    print(f"  Out      : {args.out}")
    print(f"  Stages   : {args.stages}")
    print(f"  Primary ROI: {PRIMARY_ROI}")
    print(f"  Bands    : {list(BANDS.keys())}")

    if 1 in args.stages:
        run_stage1(subjects, args.run, args.ts_dir, args.out)

    if 2 in args.stages:
        run_stage2(subjects, args.run, args.ts_dir, args.out)
        run_stage2_incremental(subjects,args.run,args.ts_dir,args.out)

    if 3 in args.stages:
        run_stage3(subjects, args.run, args.ts_dir, args.out)

    if 4 in args.stages or 5 in args.stages:
        run_stage45(subjects, args.run, args.ts_dir, args.out)

    print(f"\n{'='*60}")
    print(f"All done. Results in: {args.out}/")
    print(f"\nOutputs:")
    print(f"  stage1_nuisance_r2.csv         per-subject R² (unfiltered)")
    print(f"  stage1_nuisance_r2_summary.csv group mean R²")
    print(f"  stage2_freq_r2.csv             per-subject R² per band")
    print(f"  stage2_freq_r2_summary.csv     group mean R² per band")
    print(f"  stage3_coupling.csv            per-subject coupling P0")
    print(f"  stage3_splithalf.csv           per-subject half-1 / half-2")
    print(f"  stage3_psd.csv                 per-subject band power")
    print(f"  stage3_timewindow.csv          per-subject windowed coupling")
    print(f"  stage3_icc.csv                 ICC(2,1) split-half")
    print(f"  stage45_matched_ratios.csv         per-subject matched P0–P3 coupling")
    print(f"  stage45_matched_ratios_summary.csv group mean matched ratios")
    print(f"  stage45_matched_ttest.csv          paired t-test P{{N}} vs P0")


if __name__ == "__main__":
    main()
