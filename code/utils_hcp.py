"""
utils_hcp.py
════════════════════════════════════════════════════════════════════════
Shared utilities for the HCP-YA gBOLD-CSF coupling postprocessing package.

This is a stripped-down version for reproducibility. It contains only
the cross-correlation functions used by the coupling analysis and figure
scripts. Upstream NIfTI processing functions have been removed.

Contents
────────
  xcorr        — normalised gBOLD-CSF cross-correlation
  deriv_xcorr  — derivative cross-correlation (−d/dt gBOLD × CSF)
"""

import numpy as np

TR = 0.72   # HCP-YA repetition time (seconds)


def xcorr(gbold, csf, tr=None, max_lag_sec=30.0):
    """
    Normalised cross-correlation xcorr(gBOLD, CSF).

    neg_peak: lag > 0  — gBOLD leads CSF (Fultz 2019 hydraulic coupling)
    pos_peak: lag < 0  — CSF leads gBOLD

    Returns dict: lags, xc, neg_peak_r, neg_peak_lag,
                  pos_peak_r, pos_peak_lag, ci95
    """
    if tr is None:
        tr = TR
    n   = len(gbold)
    ml  = min(int(np.round(max_lag_sec / tr)), n // 4)
    xn  = lambda s: (s - s.mean()) / (s.std() + 1e-10)
    xc  = np.correlate(xn(gbold), xn(csf), mode="full") / n
    mid = len(xc) // 2
    xc  = xc[mid - ml: mid + ml + 1]
    lgs = np.arange(-ml, ml + 1) * tr

    pm    = lgs > 0
    neg_i = int(np.argmin(np.where(pm, xc, np.inf))) \
            if pm.any() else int(np.argmin(xc))
    neg_r, neg_lag = float(xc[neg_i]), float(lgs[neg_i])

    nm    = lgs < 0
    pos_i = int(np.argmax(np.where(nm, xc, -np.inf))) \
            if nm.any() else int(np.argmax(xc))
    pos_r, pos_lag = float(xc[pos_i]), float(lgs[pos_i])

    dom  = neg_r if abs(neg_r) > abs(pos_r) else pos_r
    ci95 = np.nan
    if abs(dom) < 0.9999:
        zf   = np.arctanh(abs(dom))
        se   = 1.0 / np.sqrt(max(n - 3, 1))
        ci95 = float(np.tanh(zf + 1.96 * se) - np.tanh(zf - 1.96 * se))

    return dict(lags=lgs, xc=xc,
                neg_peak_r=neg_r, neg_peak_lag=neg_lag,
                pos_peak_r=pos_r, pos_peak_lag=pos_lag, ci95=ci95)


def deriv_xcorr(gbold, csf, tr=None, max_lag_sec=30.0):
    """
    Cross-correlation of −d/dt(gBOLD) with CSF.
    Positive peak searched within |lag| ≤ 3 s of zero.
    Returns dict: lags, xc, pos_peak_r, pos_peak_lag, neg_peak_r, neg_peak_lag
    """
    if tr is None:
        tr = TR
    xn     = lambda s: (s - s.mean()) / (s.std() + 1e-10)
    neg_dg = -np.gradient(gbold, tr)
    n      = len(gbold)
    ml     = min(int(np.round(max_lag_sec / tr)), n // 4)
    xc     = np.correlate(xn(neg_dg), xn(csf), mode="full") / n
    mid    = len(xc) // 2
    xc     = xc[mid - ml: mid + ml + 1]
    lgs    = np.arange(-ml, ml + 1) * tr

    near  = np.abs(lgs) <= 3.0
    pos_i = int(np.argmax(np.where(near, xc, -np.inf))) \
            if (near.any() and xc[near].max() > 0) \
            else int(np.argmax(xc))

    return dict(lags=lgs, xc=xc,
                pos_peak_r=float(xc[pos_i]),   pos_peak_lag=float(lgs[pos_i]),
                neg_peak_r=float(xc[np.argmin(xc)]),
                neg_peak_lag=float(lgs[np.argmin(xc)]))
