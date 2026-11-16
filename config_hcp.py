"""
config_hcp.py
════════════════════════════════════════════════════════════════════════
Shared configuration, path routing, and utilities for the HCP-YA
gBOLD–CSF coupling pipeline.

Sections
────────
  A. Dataset constants          — ROOT, TR, scan params, frequency bands
  B. Run / session mapping      — 4-run support (LR, RL, LR2, RL2)
  C. subject_dirs()             — all path routing for one subject/run
  D. Subject list helpers       — load_subjects, load_csf_masks
  E. Signal processing          — compute_fd, bandpass_volume, xcorr, deriv_xcorr
  F. I/O helpers                — save_nii

Run labels
──────────
  LR   REST1 LR phase-encode  (primary cohort, N=104)
  RL   REST1 RL phase-encode
  LR2  REST2 LR phase-encode  (test-retest subsample, N=25)
  RL2  REST2 RL phase-encode

Topup convention
────────────────
  Each REST session has its own paired spin-echo field maps:
    REST1 → rfMRI_REST1_LR_SBRef + rfMRI_REST1_RL_SBRef
    REST2 → rfMRI_REST2_LR_SBRef + rfMRI_REST2_RL_SBRef
  Topup is computed once per session and shared between both runs of
  that session. A lock-file in pipeline_steps.py prevents double
  computation when both runs are processed in parallel.
"""

import os
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# A. Dataset constants
# ══════════════════════════════════════════════════════════════════════════════

ROOT = "/Volumes/BrainLab01/Datasets/BALSA/HCP-YA-gBOLD-CSF"

# Acquisition parameters
TR            = 0.72    # repetition time (s)
N_DUMMY_VOLS  = 10      # volumes dropped from start of each run
N_VOLUMES     = 1190    # volumes after dummy removal (1200 acquired − 10 dummies)
N_VOLS_KEEP   = N_VOLUMES                  # alias kept for backward compat
N_SLICES      = 72
MB_FACTOR     = 8

# Slice timing (interleaved multiband; from DICOM, NOT TR/N_SLICES)
# dt_MB = 77.5 ms (measured; not 72/9 = 80 ms)
SLICE_TIME_DT   = 0.0775   # seconds between multiband groups
HCP_SLICE_TIMING = [
    round(((s % MB_FACTOR) + (s // MB_FACTOR) * SLICE_TIME_DT), 6)
    for s in range(N_SLICES)
]

# Topup
TOPUP_READOUT_TIME = 0.0742   # total readout time for HCP-YA SBRef (s)
TOPUP_CONFIG       = "b02b0.cnf"

# Spatial smoothing
SMOOTH_FWHM_MM = 4.0

# Frequency bands: key → (low_hz, high_hz, label)
BAND_CONFIGS = {
    "standard_rs": (0.010, 0.100, "broad"),
    "neurogenic":  (0.020, 0.040, "neurogenic"),
    "vasogenic":   (0.060, 0.100, "vasogenic"),
    # endogenic (0.001–0.02 Hz) excluded — insufficient SNR at HCP TR=0.72s
}

# Maximum cross-correlation lag (s) per band
BAND_LAG_SEC = {
    "standard_rs": 15.0,
    "neurogenic":  20.0,
    "vasogenic":   10.0,
}

# Nuisance pipelines
PIPELINES_TO_RUN = ["P0", "P1", "P2", "P3"]

# FreeSurfer
USE_FREESURFER = True
FS_DIR = os.path.join(ROOT, "FreeSurfer")   # <ROOT>/FreeSurfer/<sub>/mri/

# Manual CSF mask directory (parallel to FINAL/, independent of preprocessing)
CSF_MASK_DIR = os.path.join(ROOT, "manualCSF-MASKS")


# ══════════════════════════════════════════════════════════════════════════════
# B. Run / session mapping
# ══════════════════════════════════════════════════════════════════════════════

# Maps run label → (HCP results directory, REST session tag)
_RUN_INFO = {
    "LR":  ("rfMRI_REST1_LR", "REST1"),
    "RL":  ("rfMRI_REST1_RL", "REST1"),
    "LR2": ("rfMRI_REST2_LR", "REST2"),
    "RL2": ("rfMRI_REST2_RL", "REST2"),
}

VALID_RUNS = list(_RUN_INFO.keys())   # ["LR", "RL", "LR2", "RL2"]


# ══════════════════════════════════════════════════════════════════════════════
# C. Path routing
# ══════════════════════════════════════════════════════════════════════════════

def subject_dirs(sub: str, run: str) -> dict:
    """
    Return a dict of all relevant paths for one subject / run.

    Parameters
    ----------
    sub : str   HCP-YA subject ID, e.g. "100307"
    run : str   One of LR | RL | LR2 | RL2

    Key groups
    ----------
    HCP source      bold_raw, sbref_raw, se_lr, se_rl,
                    bias_bc, bias_32ch,
                    fs_dir, fs_mri, aseg, t1
    Topup           topup_dir, topup_base, topup_acqp, topup_apply_index
    Preprocessing   out_root, reg_dir, masks_dir,
                    bold_nodummy, bold_biascorr, sbref_biascorr,
                    bold_unwarped, sbref_unwarped,
                    bold_stc, bold_mc, mc_par, fd_file, mean_bold
    Masks           wm_mask, wm_mask_ero1_novent, wm_mask_ero2_novent,
                    gm_mask, brain_mask, fourthvent_mask, aseg_epi
    Cleaned volumes bold_mc_smooth, bold_mc_smooth_clean_P0–P3
    Timeseries      ts_dir
    """
    if run not in _RUN_INFO:
        raise ValueError(f"Unknown run '{run}'. Valid: {VALID_RUNS}")

    hcp_dir, session = _RUN_INFO[run]

    # ── HCP unprocessed source tree ───────────────────────────────────────────
    # Layout on disk:
    #   rsfMRI1/<sub>_3T_rfMRI_REST1_unproc/<sub>/unprocessed/3T/rfMRI_REST1_LR/
    #   rsfMRI2/<sub>_3T_rfMRI_REST2_unproc/<sub>/unprocessed/3T/rfMRI_REST2_LR/
    rsfmri_root = os.path.join(ROOT, "rsfMRI1" if session == "REST1" else "rsfMRI2")
    sess_tag    = "REST1" if session == "REST1" else "REST2"
    unproc_dir  = os.path.join(rsfmri_root,
                               f"{sub}_3T_rfMRI_{sess_tag}_unproc",
                               str(sub), "unprocessed", "3T")
    func_dir    = os.path.join(unproc_dir, hcp_dir)

    # FreeSurfer outputs live under <ROOT>/FreeSurfer/<sub>/
    fs_dir = os.path.join(ROOT, "FreeSurfer", str(sub))

    # Spin-echo field map pair for topup: use the SBRef volumes (one per PE dir)
    # LR SBRef lives in the LR run folder, RL SBRef in the RL run folder
    se_lr = os.path.join(unproc_dir, f"rfMRI_{sess_tag}_LR",
                         f"{sub}_3T_rfMRI_{sess_tag}_LR_SBRef.nii.gz")
    se_rl = os.path.join(unproc_dir, f"rfMRI_{sess_tag}_RL",
                         f"{sub}_3T_rfMRI_{sess_tag}_RL_SBRef.nii.gz")

    # Output roots
    out_root  = os.path.join(ROOT, "results", "preprocessing", f"{sub}_{run}")
    reg_dir   = os.path.join(out_root, "reg")
    masks_dir = os.path.join(out_root, "masks")

    # Topup is shared within a session (REST1 or REST2)
    topup_dir  = os.path.join(ROOT, "results", "preprocessing",
                               f"{sub}_{session}_topup")
    topup_base = os.path.join(topup_dir, "topup")
    topup_acqp = os.path.join(topup_dir, "acqparams.txt")

    return {
        # HCP source files
        "bold_raw":        os.path.join(func_dir, f"{sub}_3T_{hcp_dir}.nii.gz"),
        "sbref_raw":       os.path.join(func_dir, f"{sub}_3T_{hcp_dir}_SBRef.nii.gz"),
        "se_lr":           se_lr,
        "se_rl":           se_rl,
        "bias_bc":         os.path.join(func_dir, f"{sub}_3T_BIAS_BC.nii.gz"),
        "bias_32ch":       os.path.join(func_dir, f"{sub}_3T_BIAS_32CH.nii.gz"),
        "bias_bc_sb":      os.path.join(func_dir, f"{sub}_3T_BIAS_BC.nii.gz"),
        "bias_32ch_sb":    os.path.join(func_dir, f"{sub}_3T_BIAS_32CH.nii.gz"),
        "fs_dir":          fs_dir,
        "fs_mri":          os.path.join(fs_dir, "mri"),
        "aseg":            os.path.join(fs_dir, "mri", "aseg.mgz"),
        "t1":              os.path.join(fs_dir, "mri", "T1.mgz"),
        # Topup (session-shared)
        "topup_dir":         topup_dir,
        "topup_base":        topup_base,
        "topup_acqp":        topup_acqp,
        "topup_apply_index": 1 if run in ("LR", "LR2") else 2,
        # Preprocessing outputs
        "out_root":        out_root,   # alias kept for backward compat
        "outdir":          out_root,   # used by pipeline_steps.py
        "reg_dir":         reg_dir,
        "masks_dir":       masks_dir,
        "bold_nodummy":    os.path.join(out_root, "bold_nodummy.nii.gz"),
        "bold_biascorr":   os.path.join(out_root, "bold_biascorr.nii.gz"),
        "sbref_biascorr":  os.path.join(out_root, "sbref_biascorr.nii.gz"),
        "bold_unwarped":   os.path.join(out_root, "bold_unwarped.nii.gz"),
        "sbref_unwarped":  os.path.join(out_root, "sbref_unwarped.nii.gz"),
        "bold_stc":        os.path.join(out_root, "bold_stc.nii.gz"),
        "bold_mc":         os.path.join(out_root, "bold_mc.nii.gz"),
        "mc_par":          os.path.join(out_root, "bold_mc.par"),
        "fd_file":         os.path.join(out_root, "fd.txt"),
        "mean_bold":       os.path.join(out_root, "mean_bold.nii.gz"),
        # Tissue masks
        "wm_mask":              os.path.join(masks_dir, "wm_mask.nii.gz"),
        "wm_mask_ero1_novent":  os.path.join(masks_dir, "wm_mask_ero1_novent.nii.gz"),
        "wm_mask_ero2_novent":  os.path.join(masks_dir, "wm_mask_ero2_novent.nii.gz"),
        "gm_mask":              os.path.join(masks_dir, "gm_mask.nii.gz"),
        "brain_mask":           os.path.join(masks_dir, "brain_mask.nii.gz"),
        "fourthvent_mask":      os.path.join(masks_dir, "fourthvent_mask.nii.gz"),
        "fourthvent_lower":     os.path.join(masks_dir, "fourthvent_lower.nii.gz"),
        "fourthvent_upper":     os.path.join(masks_dir, "fourthvent_upper.nii.gz"),
        "aseg_epi":             os.path.join(masks_dir, "aseg_epi.nii.gz"),
        # Cleaned volumes
        "bold_mc_smooth":              os.path.join(out_root, "bold_mc_smooth.nii.gz"),
        "bold_mc_smooth_clean_P0":     os.path.join(out_root, "bold_mc_smooth_clean_P0.nii.gz"),
        "bold_mc_smooth_clean_P1":     os.path.join(out_root, "bold_mc_smooth_clean_P1.nii.gz"),
        "bold_mc_smooth_clean_P2":     os.path.join(out_root, "bold_mc_smooth_clean_P2.nii.gz"),
        "bold_mc_smooth_clean_P3":     os.path.join(out_root, "bold_mc_smooth_clean_P3.nii.gz"),
        # Timeseries output
        "ts_dir":          os.path.join(ROOT, "results", "timeseries",
                                        f"{sub}_{run}"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# D. Subject list helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_subjects(csv_path=None, subjects_list=None) -> list:
    """
    Load subject IDs from a CSV/text file or a direct list.

    Parameters
    ----------
    csv_path      : str or None   Path to a CSV or plain-text file.
                                  First column used if CSV; one ID per line
                                  if plain text. Lines starting with '#'
                                  are ignored.
    subjects_list : list or None  Direct list of subject ID strings.
                                  Takes priority over csv_path.

    Returns
    -------
    list of str
    """
    if subjects_list:
        return [str(s) for s in subjects_list]
    if not csv_path:
        raise ValueError("Provide csv_path or subjects_list")
    subjects = []
    with open(csv_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Accept CSV (take first column) or plain text
            subjects.append(line.split(",")[0].strip())
    return subjects


def load_csf_masks(sub: str, log=None) -> dict:
    """
    Load manually drawn CSF ROI masks for a subject.

    Expects masks at:
        {ROOT}/manualCSF-MASKS/{sub}-slice_{zz}.nii.gz
    where zz is a zero-padded slice index (01, 02, …).
    Directory is parallel to FINAL/ and independent of preprocessing outputs.

    Returns
    -------
    dict with keys:
        by_z     : {z_index: mask_array (x,y,z bool)}
        z_sorted : list of z indices in ascending order
    """
    if log is None:
        log = lambda msg: None   # noqa: E731

    mask_dir = CSF_MASK_DIR
    result   = {"by_z": {}, "z_sorted": []}

    if not os.path.isdir(mask_dir):
        log(f"  csf_masks dir not found: {mask_dir}")
        return result

    import nibabel as nib

    for fname in sorted(os.listdir(mask_dir)):
        if not fname.startswith(f"{sub}-slice_") or not fname.endswith(".nii.gz"):
            continue
        # Parse z index from filename: {sub}-slice_{zz}.nii.gz
        try:
            z = int(fname.replace(f"{sub}-slice_", "").replace(".nii.gz", ""))
        except ValueError:
            continue
        path = os.path.join(mask_dir, fname)
        try:
            img  = nib.load(path)
            data = img.get_fdata(dtype="float32") > 0.5
            result["by_z"][z] = data
        except Exception as e:
            log(f"  ⚠ could not load CSF mask {fname}: {e}")

    result["z_sorted"] = sorted(result["by_z"].keys())
    return result


# ══════════════════════════════════════════════════════════════════════════════
# E. Signal processing
# ══════════════════════════════════════════════════════════════════════════════

def compute_fd(par_file: str, head_radius_mm: float = 50.0) -> np.ndarray:
    """
    Compute framewise displacement from an mcflirt .par file.

    Columns: Rx Ry Rz Tx Ty Tz  (radians, mm).
    Rotations are converted to mm using head_radius_mm.
    FD = sum of absolute first differences across all 6 parameters.

    Returns array of shape (N-1,).
    """
    params = np.loadtxt(par_file)        # (N, 6)
    params_mm = params.copy()
    params_mm[:, :3] *= head_radius_mm   # radians → mm
    diff = np.abs(np.diff(params_mm, axis=0))
    return diff.sum(axis=1)


def bandpass_volume(data: np.ndarray, tr: float,
                    low_hz: float, high_hz: float) -> np.ndarray:
    """
    Bandpass filter a volume (or 1-D timeseries reshaped as 1×1×1×T)
    using nilearn clean_img.

    Parameters
    ----------
    data    : np.ndarray  Shape (..., T) or (1,1,1,T)
    tr      : float       Repetition time in seconds
    low_hz  : float       High-pass cutoff
    high_hz : float       Low-pass cutoff

    Returns np.ndarray of same shape as data.
    """
    import nibabel as nib
    from nilearn.image import clean_img

    scalar = data.ndim == 1
    if scalar:
        data = data.reshape(1, 1, 1, -1)

    img     = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    cleaned = clean_img(img, t_r=tr,
                        high_pass=low_hz, low_pass=high_hz,
                        detrend=False, standardize=None)
    out = np.asarray(cleaned.dataobj)
    return out.ravel() if scalar else out


def extract_signals(vol_data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract mean timeseries from vol_data at mask voxels.

    Parameters
    ----------
    vol_data : np.ndarray  Shape (x, y, z, T)
    mask     : np.ndarray  Shape (x, y, z), boolean

    Returns np.ndarray of shape (T,).
    """
    return vol_data[mask > 0.5, :].mean(axis=0)


def detrend_timeseries(ts: np.ndarray) -> np.ndarray:
    """Remove linear trend from a 1-D timeseries."""
    n = len(ts)
    t = np.arange(n)
    return ts - np.polyval(np.polyfit(t, ts, 1), t)


def xcorr(gbold: np.ndarray, csf: np.ndarray,
          tr: float = None, max_lag_sec: float = 30.0) -> dict:
    """
    Normalised cross-correlation xcorr(gBOLD, CSF).

    neg_peak: lag > 0  — gBOLD leads CSF (Fultz 2019 gBOLD-CSF coupling)
    pos_peak: lag < 0  — CSF leads gBOLD

    Returns dict: lags, xc, neg_peak_r, neg_peak_lag,
                  pos_peak_r, pos_peak_lag, ci95
    """
    if tr is None:
        tr = TR
    n  = len(gbold)
    ml = min(int(np.round(max_lag_sec / tr)), n // 4)
    xn = lambda s: (s - s.mean()) / (s.std() + 1e-10)   # noqa: E731
    xc = np.correlate(xn(gbold), xn(csf), mode="full") / n
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


def deriv_xcorr(gbold: np.ndarray, csf: np.ndarray,
                tr: float = None, max_lag_sec: float = 30.0) -> dict:
    """
    Cross-correlation of −d/dt(gBOLD) with CSF.

    Returns dict: lags, xc, pos_peak_r, pos_peak_lag,
                  neg_peak_r, neg_peak_lag
    """
    if tr is None:
        tr = TR
    xn     = lambda s: (s - s.mean()) / (s.std() + 1e-10)   # noqa: E731
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


# ══════════════════════════════════════════════════════════════════════════════
# F. I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_nii(data: np.ndarray, affine: np.ndarray,
             header, path: str) -> None:
    """Save a NIfTI image, creating parent directories as needed."""
    import nibabel as nib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.Nifti1Image(data, affine, header).to_filename(path)


def erode_mask(mask: np.ndarray, n_vox: int) -> np.ndarray:
    """
    Erode a binary 3-D mask by n_vox voxels using a 3×3×3 ball, iterated.

    Parameters
    ----------
    mask  : np.ndarray  Boolean or 0/1 array, shape (x, y, z)
    n_vox : int         Number of erosion iterations

    Returns np.ndarray of same shape, dtype bool.
    """
    from scipy.ndimage import binary_erosion
    struct = np.ones((3, 3, 3), dtype=bool)
    out = mask.astype(bool)
    for _ in range(n_vox):
        out = binary_erosion(out, structure=struct)
    return out
