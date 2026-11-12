#!/usr/bin/env python3
"""
surrogate_null.py
═══════════════════════════════════════════════════════════════════════════════
Surrogate null distribution for gBOLD–CSF coupling (Reviewer #2, Comment 1).

Method: Amplitude-Adjusted Fourier Transform (AAFT) phase randomisation.
  - Preserves the amplitude spectrum and marginal distribution of gBOLD.
  - Destroys temporal structure (phase coupling with CSF).
  - 1000 surrogates per subject per band → empirical null distribution of |r|.

For each subject × band × ROI:
  1. Load bandpassed gBOLD and CSF from results/timeseries/{sub}_{run}/
  2. Generate N_SURR=1000 AAFT surrogates of gBOLD
  3. Run xcorr(surrogate, csf) → collect neg_peak_r values
  4. Compare observed |r| against null distribution
  5. Report: z-score, percentile, p-value (proportion of surrogates ≥ observed |r|)

Also runs the full-timeseries window analysis (R2-3):
  For temporal-stability windows (4 × WIN_LEN volumes), repeats the
  surrogate test within each window so per-window coupling can be compared
  against its own window-specific null.

Outputs (to OUT_DIR):
  surrogate_summary.csv          — subject × band × ROI summary stats
  surrogate_windows.csv          — same, per time window
  surrogate_null_distributions/  — per-subject per-band |r| arrays (.npy)

Usage
─────
  python code/surrogate_null.py \\
      --subjects_csv coupling_subjects_104.csv \\
      --run LR \\
      --roi z0 \\
      --n_surr 1000 \\
      --out results/surrogate_analysis

  # Fast test with 100 surrogates and 5 subjects:
  python code/surrogate_null.py \\
      --subjects_csv coupling_subjects_104.csv \\
      --subjects 100206 100307 100408 100610 101006 \\
      --n_surr 100 \\
      --out results/surrogate_analysis_test
"""

import os
import sys
import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config_hcp as cfg
from utils_hcp import xcorr

TR      = cfg.TR
N_VOLS  = 1190              # volumes after dummy removal
N_WIN   = 4
WIN_LEN = N_VOLS // N_WIN   # 297 volumes = ~3.6 min

# Band definitions matching analyse_coupling.py / config_hcp.py
BANDS = {
    "broad":      (0.010, 0.100, cfg.BAND_LAG_SEC["standard_rs"]),
    "neurogenic": (0.020, 0.040, cfg.BAND_LAG_SEC["neurogenic"]),
    "endogenic":  (0.001, 0.020, cfg.BAND_LAG_SEC["endogenic"]),
    "vasogenic":  (0.060, 0.100, cfg.BAND_LAG_SEC["vasogenic"]),
}

ROIS = ["z0", "z1", "z2", "z3", "fv_lower", "fv_upper"]


# ══════════════════════════════════════════════════════════════════════════════
# AAFT surrogate generation
# ══════════════════════════════════════════════════════════════════════════════

def aaft_surrogate(x, rng):
    """
    Generate one Amplitude-Adjusted Fourier Transform (AAFT) surrogate.

    Algorithm (Theiler et al. 1992; Schreiber & Schmitz 1996):
      1. Rank-order x to produce a Gaussian rescaled series x_g.
      2. Phase-randomise x_g in the Fourier domain.
       3. Rank-order x_g_random to match the amplitude distribution of x.

    Preserves: amplitude spectrum (power spectral density) and marginal
               distribution (histogram) of x.
    Destroys:  phase relationships between x and any other signal.

    Parameters
    ----------
    x   : 1-D array, zero-mean preferred (already bandpassed)
    rng : numpy.random.Generator

    Returns
    -------
    surrogate : 1-D array, same length as x
    """
    n = len(x)

    # 1. Sort x to get rank ordering
    rank_x = np.argsort(np.argsort(x))   # rank of each element of x

    # 2. Produce Gaussian series with same ranks
    gauss   = rng.standard_normal(n)
    gauss   = (gauss - gauss.mean()) / (gauss.std() + 1e-12)
    x_gauss = np.sort(gauss)[rank_x]     # rescale Gaussian to match x's ranks

    # 3. Phase-randomise the Gaussian series
    fft_g = np.fft.rfft(x_gauss)
    # Random phases, but keep DC and (if n even) Nyquist real
    n_freq  = len(fft_g)
    phases  = rng.uniform(0, 2 * np.pi, n_freq)
    phases[0]  = 0.0                     # DC stays real
    if n % 2 == 0:
        phases[-1] = 0.0                 # Nyquist stays real
    fft_rand    = np.abs(fft_g) * np.exp(1j * phases)
    x_rand_gauss = np.fft.irfft(fft_rand, n=n)

    # 4. Rank-order x back to match amplitude distribution of x
    rank_rand = np.argsort(np.argsort(x_rand_gauss))
    surrogate = np.sort(x)[rank_rand]

    return surrogate.astype(np.float64)


def aaft_batch(x, n_surr, seed=42):
    """
    Generate n_surr AAFT surrogates of x.

    Returns
    -------
    surrogates : (n_surr, n) array
    """
    rng        = np.random.default_rng(seed)
    surrogates = np.zeros((n_surr, len(x)), dtype=np.float64)
    for i in range(n_surr):
        surrogates[i] = aaft_surrogate(x, rng)
    return surrogates


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_subjects(csv_path, subjects_arg):
    with open(csv_path) as f:
        all_subs = [row["subject"] for row in csv.DictReader(f)]
    if subjects_arg:
        return [s for s in subjects_arg if s in all_subs]
    return all_subs


def _sub_dir(ts_root, sub, run):
    return os.path.join(ts_root, f"{sub}_{run}")


def _read_ts(sub_dir, fname):
    """Return first column of a CSV as float64 1-D array, or None."""
    path = os.path.join(sub_dir, fname)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).iloc[:, 0].to_numpy(dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Per-segment surrogate test
# ══════════════════════════════════════════════════════════════════════════════

def surrogate_test(gbold_seg, csf_seg, n_surr, max_lag_sec, seed=42):
    """
    Run surrogate null test for one segment.

    Parameters
    ----------
    gbold_seg  : 1-D array, bandpassed gBOLD (already trimmed to segment)
    csf_seg    : 1-D array, bandpassed CSF (same length)
    n_surr     : number of AAFT surrogates
    max_lag_sec: max lag for xcorr (band-specific)
    seed       : random seed

    Returns
    -------
    dict with observed |r| and null distribution statistics
    """
    n = len(gbold_seg)
    if n < 30:
        return None

    # Observed coupling
    obs = xcorr(gbold_seg, csf_seg, tr=TR, max_lag_sec=max_lag_sec)
    obs_r  = obs["neg_peak_r"]       # negative = gBOLD leads CSF
    obs_absr = abs(obs_r)

    # Generate surrogates and compute null |r| values
    surrogates = aaft_batch(gbold_seg, n_surr, seed=seed)
    null_absr  = np.zeros(n_surr, dtype=np.float64)

    for i, surr in enumerate(surrogates):
        res = xcorr(surr, csf_seg, tr=TR, max_lag_sec=max_lag_sec)
        null_absr[i] = abs(res["neg_peak_r"])

    # Statistics
    null_mean = float(null_absr.mean())
    null_sd   = float(null_absr.std())
    z_score   = (obs_absr - null_mean) / (null_sd + 1e-12)
    pct_rank  = float(np.mean(null_absr < obs_absr)) * 100.0  # percentile of obs in null
    p_surr    = float(np.mean(null_absr >= obs_absr))          # proportion surrogates ≥ observed

    return dict(
        obs_r         = float(obs_r),
        obs_absr      = float(obs_absr),
        obs_lag       = float(obs["neg_peak_lag"]),
        null_mean     = null_mean,
        null_sd       = null_sd,
        null_p05      = float(np.percentile(null_absr, 5)),
        null_p50      = float(np.percentile(null_absr, 50)),
        null_p95      = float(np.percentile(null_absr, 95)),
        null_p99      = float(np.percentile(null_absr, 99)),
        z_score       = float(z_score),
        pct_rank      = float(pct_rank),
        p_surrogate   = float(p_surr),
        significant   = bool(p_surr < 0.05),
        n_surr        = n_surr,
        n_vols        = n,
        null_dist     = null_absr,    # kept for saving to .npy
    )


# ══════════════════════════════════════════════════════════════════════════════
# Per-subject analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_subject(sub, run, ts_root, rois, bands, n_surr,
                npy_dir, verbose=True):
    """
    Run surrogate analysis for one subject.

    Returns
    -------
    full_rows    : list of dicts for surrogate_summary.csv
    window_rows  : list of dicts for surrogate_windows.csv
    """
    log = print if verbose else (lambda *a, **k: None)
    sd  = _sub_dir(ts_root, sub, run)

    if not os.path.isdir(sd):
        log(f"  ✗ timeseries dir missing: {sd}")
        return [], []

    full_rows    = []
    window_rows  = []

    for band_key, (lo, hi, max_lag) in bands.items():
        # Load bandpassed gBOLD
        gbold = _read_ts(sd, f"gbold_{band_key}.csv")
        if gbold is None:
            log(f"  ✗ {sub} {band_key}: gbold missing")
            continue

        for roi in rois:
            csf = _read_ts(sd, f"csf_{roi}_{band_key}.csv")
            if csf is None:
                continue

            # Trim to minimum length (should both be N_VOLS)
            n   = min(len(gbold), len(csf))
            gb  = gbold[:n]
            cs  = csf[:n]

            # ── Full timeseries surrogate test ─────────────────────────────
            seed = abs(hash(f"{sub}_{band_key}_{roi}")) % (2**31)
            res  = surrogate_test(gb, cs, n_surr, max_lag, seed=seed)

            if res is None:
                continue

            # Optionally save null distribution as .npy
            npy_path = os.path.join(
                npy_dir, f"{sub}_{run}_{band_key}_{roi}_null.npy")
            np.save(npy_path, res.pop("null_dist"))

            full_rows.append(dict(
                subject   = sub,
                run       = run,
                band      = band_key,
                roi       = roi,
                **{k: v for k, v in res.items()},
            ))

            log(f"  {sub} {band_key} {roi}: "
                f"|r|={res['obs_absr']:.3f}  z={res['z_score']:.2f}  "
                f"p_surr={res['p_surrogate']:.3f}")

            # ── Window-specific surrogate tests (R2-3) ─────────────────────
            for win_idx in range(N_WIN):
                w_start = win_idx * WIN_LEN
                w_end   = w_start + WIN_LEN
                if w_end > n:
                    continue

                gb_w = gb[w_start:w_end]
                cs_w = cs[w_start:w_end]

                w_seed = abs(hash(f"{sub}_{band_key}_{roi}_win{win_idx}")) % (2**31)
                w_res  = surrogate_test(gb_w, cs_w, n_surr, max_lag, seed=w_seed)

                if w_res is None:
                    continue

                w_res.pop("null_dist", None)   # don't store per-window arrays

                window_rows.append(dict(
                    subject    = sub,
                    run        = run,
                    band       = band_key,
                    roi        = roi,
                    window     = win_idx + 1,
                    win_start  = w_start,
                    win_end    = w_end,
                    **w_res,
                ))

    return full_rows, window_rows


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AAFT surrogate null distribution for gBOLD-CSF coupling")
    parser.add_argument("--subjects_csv", required=True,
                        help="CSV with 'subject' column (coupling_subjects_104.csv)")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Optional subset of subject IDs")
    parser.add_argument("--run",    default="LR",
                        help="Pipeline run (default: LR)")
    parser.add_argument("--ts_dir", default=None,
                        help="Root of timeseries directory "
                             "(default: ROOT/results/timeseries)")
    parser.add_argument("--out",    default=None,
                        help="Output directory "
                             "(default: ROOT/results/surrogate_analysis)")
    parser.add_argument("--roi",    nargs="*", default=None,
                        help="ROIs to run (default: all 6)")
    parser.add_argument("--bands",  nargs="*", default=None,
                        help="Bands to run: broad neurogenic endogenic vasogenic")
    parser.add_argument("--n_surr", type=int, default=1000,
                        help="Number of AAFT surrogates (default: 1000)")
    parser.add_argument("--windows", action="store_true",
                        help="Also run window-specific surrogate tests (R2-3)")
    parser.add_argument("--no_windows", dest="windows", action="store_false")
    parser.set_defaults(windows=True)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    ts_root  = args.ts_dir  or os.path.join(cfg.ROOT, "results", "timeseries")
    out_dir  = args.out     or os.path.join(cfg.ROOT, "results", "surrogate_analysis")
    npy_dir  = os.path.join(out_dir, "surrogate_null_distributions")
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(npy_dir,  exist_ok=True)

    # ── Resolve subjects / bands / ROIs ───────────────────────────────────────
    subjects = _load_subjects(args.subjects_csv, args.subjects)
    rois     = args.roi   or ROIS
    bands    = {k: v for k, v in BANDS.items()
                if args.bands is None or k in args.bands}

    print(f"Surrogate null analysis")
    print(f"  Subjects : {len(subjects)}")
    print(f"  Run      : {args.run}")
    print(f"  ROIs     : {rois}")
    print(f"  Bands    : {list(bands)}")
    print(f"  N surr   : {args.n_surr}")
    print(f"  Windows  : {args.windows}")
    print(f"  ts_root  : {ts_root}")
    print(f"  out_dir  : {out_dir}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    all_full   = []
    all_windows = []

    for i, sub in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] {sub}")
        f_rows, w_rows = run_subject(
            sub, args.run, ts_root, rois, bands, args.n_surr,
            npy_dir, verbose=args.verbose)
        all_full.extend(f_rows)
        if args.windows:
            all_windows.extend(w_rows)

    # ── Save results ──────────────────────────────────────────────────────────
    if all_full:
        df_full = pd.DataFrame(all_full)
        out_path = os.path.join(out_dir, "surrogate_summary.csv")
        df_full.to_csv(out_path, index=False)
        print(f"\n✓ surrogate_summary.csv  ({len(df_full)} rows)  → {out_path}")

        # Print group-level summary
        print("\nGroup summary (mean obs |r|, z-score, % significant):")
        grp = df_full.groupby(["band", "roi"]).agg(
            n            = ("subject", "count"),
            mean_obs_absr = ("obs_absr", "mean"),
            mean_null_p95 = ("null_p95", "mean"),
            mean_z        = ("z_score",  "mean"),
            pct_sig       = ("significant", "mean"),
        ).reset_index()
        grp["pct_sig"] = (grp["pct_sig"] * 100).round(1)
        print(grp.to_string(index=False))

    if args.windows and all_windows:
        df_win = pd.DataFrame(all_windows)
        win_path = os.path.join(out_dir, "surrogate_windows.csv")
        df_win.to_csv(win_path, index=False)
        print(f"\n✓ surrogate_windows.csv  ({len(df_win)} rows)  → {win_path}")

    print(f"\n✓ Null distribution arrays (.npy) → {npy_dir}")


if __name__ == "__main__":
    main()
