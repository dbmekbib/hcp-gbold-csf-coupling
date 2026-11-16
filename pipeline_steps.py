"""
pipeline_steps.py — HCP-YA gBOLD–CSF Coupling Pipeline Steps
══════════════════════════════════════════════════════════════════════════════
Pipeline overview
─────────────────
Step 1  │ Input QC + dummy volume removal (bold → bold_nodummy)
Step 2  │ Bias correction + FSL topup (bold_nodummy → bold_biascorr,
        │                              bold_unwarped, sbref_unwarped)
Step 3  │ Slice-timing correction (bold_unwarped → bold_stc)
Step 4  │ Motion correction + FD (bold_stc → bold_mc, mean_bold)
Step 4b │ Mean + robust variance of bold_unwarped (before motion correction)
Step 5  │ FreeSurfer registration + tissue masks
        │   (aseg.mgz → wm_mask, gm_mask_epi, fourthvent_csf_roi)
Step 5b │ WM mask refinement + QC (ero1/ero2 + ventricle exclusion)
Step 6a │ Nuisance regression — P0/P1/P2/P3
        │   (bold_mc → bold_mc_smooth_clean_P{N})
Step 6b │ Bandpass filtering across four frequency bands
        │   (bold_mc_smooth_clean_P{N} → bold_mc_smooth_filtered_P{N}_{tag})

Configuration
─────────────

HCP-YA acquisition parameters (from config_hcp.py)
    TR         = 0.72 s
    MB_FACTOR  = 8          (multiband)
    N_SLICES   = 72
    N_DUMMY    = 10         (dropped at start of each run)
    N_VOLUMES  = 1200       (after dummy removal)
    dt_MB      = 77.5 ms    (from DICOM, NOT TR/9 = 80 ms)

Frequency bands (BAND_CONFIGS)
    standard_rs : 0.010–0.100 Hz  (broad)
    neurogenic  : 0.020–0.040 Hz
    vasogenic   : 0.060–0.100 Hz

Regression pipelines
    P0 : detrend only                   
    P1 : detrend + 6 motion params
    P2 : detrend + motion + WM mean
    P3 : detrend + motion + WM aCompCor 5 PCs

WM mask strategy (applied in step5b, used here in step 6a)
    Primary regressor mask: wm_mask_ero2_novent
    (2-vox erosion; lateral + 3rd ventricle excluded with 1-vox dilation buffer)

Data sharing note
─────────────────
Raw fMRI volumes are NOT shared per HCP Data Use Terms.
Derived CSVs (timeseries, coupling metrics) and subject IDs are shareable.
HCP-YA subject IDs are public (open-access release).
"""

import json
import os
import sys
import glob
import time
import types
import traceback
import subprocess
import shutil
from datetime import datetime

import numpy as np
from numpy.linalg import svd
import nibabel as nib
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.stats import trim_mean
from scipy.signal import butter, filtfilt

from nilearn.image import clean_img
from nilearn.maskers import NiftiMasker

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

import config_hcp as cfg
from config_hcp import erode_mask, save_nii

TR = cfg.TR


# ══════════════════════════════════════════════════════════════════════════════
# Internal shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _run_cmd(cmd, log):
    """Run a shell command; raise RuntimeError on non-zero exit."""
    log(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:500])
    return r.stdout.strip()


def _run_safe(cmd, log):
    """Run a shell command; log warning and return False on failure."""
    try:
        _run_cmd(cmd, log)
        return True
    except RuntimeError as e:
        log(f"  ⚠ {e}")
        return False


def _ensure_dirs(p):
    """Create output directories for a subject/run if they don't exist."""
    for key in ("outdir", "out_root", "reg_dir", "masks_dir", "topup_dir", "ts_dir"):
        d = p.get(key)
        if d:
            os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Input QC + dummy volume removal
# ══════════════════════════════════════════════════════════════════════════════

class step1:
    """
    Step 1: Input QC and dummy volume removal.

    Checks all required inputs exist and have correct dimensions, verifies
    BOLD header (4-D, correct TR, correct volume count), then drops the
    first N_DUMMY_VOLS volumes from the BOLD to produce bold_nodummy.nii.gz.

    Returns
    -------
    dict with keys: subject, run, step (=1), status, missing, warnings
    """

    @staticmethod
    def run(sub: str, run: str = "LR", logger=None, force: bool = False) -> dict:
        """
        Parameters
        ----------
        sub    : HCP-YA subject ID (e.g. "100307")
        run    : "LR" or "RL"
        logger : callable(str) for progress messages; defaults to print
        force  : re-run even if output already exists
        """
        p   = cfg.subject_dirs(sub, run)
        log = logger or print
        _ensure_dirs(p)
        report = {
            "subject": sub, "run": run, "step": 1,
            "status": "ok", "missing": [], "warnings": [],
        }

        log(f"\n  ━━ Step 1: Input QC + Dummy Removal ━━")

        # ── 1.1 Check required inputs ─────────────────────────────────────
        required = [
            ("bold_raw",   p["bold_raw"]),
            ("sbref_raw",  p["sbref_raw"]),
            ("se_lr",      p["se_lr"]),
            ("se_rl",      p["se_rl"]),
        ]
        for label, path in required:
            if not os.path.exists(path):
                log(f"  ✗ MISSING: {label} → {path}")
                report["missing"].append(label)

        if report["missing"]:
            report["status"] = "fail"
            return report

        # ── 1.2 BOLD header check ─────────────────────────────────────────
        bold_img = nib.load(p["bold_raw"])
        hdr      = bold_img.header
        shape    = bold_img.shape

        if len(shape) != 4:
            log(f"  ✗ BOLD is not 4-D: shape={shape}")
            report["status"] = "fail"
            return report

        tr_hdr = round(float(hdr.get_zooms()[3]), 4)
        if abs(tr_hdr - cfg.TR) > 0.01:
            msg = f"TR in header={tr_hdr}, expected {cfg.TR}"
            log(f"  ⚠ {msg}")
            report["warnings"].append(msg)

        n_vols_raw = shape[3]
        n_vols_expected = cfg.N_VOLUMES + cfg.N_DUMMY_VOLS
        if n_vols_raw != n_vols_expected:
            msg = (f"volume count={n_vols_raw}, "
                   f"expected {n_vols_expected} "
                   f"({cfg.N_VOLUMES} + {cfg.N_DUMMY_VOLS} dummy)")
            log(f"  ⚠ {msg}")
            report["warnings"].append(msg)

        log(f"  BOLD: shape={shape}, TR={tr_hdr}s, {n_vols_raw} vols")

        # ── 1.3 Drop dummy volumes ────────────────────────────────────────
        out = p["bold_nodummy"]
        if os.path.exists(out) and not force:
            log(f"  ⏭ bold_nodummy exists — skipping")
            return report

        log(f"  Dropping first {cfg.N_DUMMY_VOLS} dummy volumes …")
        data   = bold_img.get_fdata(dtype=np.float32)
        clipped = data[..., cfg.N_DUMMY_VOLS:]
        nib.Nifti1Image(clipped, bold_img.affine, hdr).to_filename(out)
        log(f"  ✓ bold_nodummy: {clipped.shape[3]} volumes → {out}")

        return report


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Bias correction + topup
# ══════════════════════════════════════════════════════════════════════════════

class step2:
    """
    Step 2: Bias-field correction + FSL topup distortion correction.

    2a  Multiply bold_nodummy by (BIAS_BC / BIAS_32CH) resampled to EPI space
        → bold_biascorr.nii.gz
    2b  Run topup on the spin-echo field maps (shared between LR and RL;
        a lock-file mechanism ensures only one process computes it)
    2c  Apply topup to bold_biascorr → bold_unwarped.nii.gz
    2d  Apply topup to sbref_biascorr → sbref_unwarped.nii.gz

    Returns
    -------
    dict with keys: subject, run, step (=2), status, warnings
    """

    @staticmethod
    def run(sub: str, run: str = "LR", logger=None) -> dict:
        p   = cfg.subject_dirs(sub, run)
        log = logger or print
        _ensure_dirs(p)
        report = {
            "subject": sub, "run": run, "step": 2,
            "status": "ok", "warnings": [],
        }

        log(f"\n  ━━ Step 2: Bias Correction + Topup ━━")

        # ── 2a Bias correction ────────────────────────────────────────────
        step2._bias_correct(p, log, report)
        if report["status"] == "fail":
            return report

        # ── 2b Run topup (shared for LR + RL) ────────────────────────────
        step2._run_topup(p, sub, log, report)
        if report["status"] == "fail":
            return report

        # ── 2c Apply topup to fMRI ────────────────────────────────────────
        # Index comes from subject_dirs(): LR/LR2 = 1, RL/RL2 = 2
        apply_index = p["topup_apply_index"]
        step2._apply_topup(
            p["bold_biascorr"], p["bold_unwarped"],
            p["topup_base"], apply_index, p, log, report,
            label="bold_unwarped")
        if report["status"] == "fail":
            return report

        # ── 2d Apply topup to SBRef ───────────────────────────────────────
        step2._apply_topup(
            p["sbref_biascorr"], p["sbref_unwarped"],
            p["topup_base"], apply_index, p, log, report,
            label="sbref_unwarped")

        return report

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _bias_correct(p, log, report):
        from nilearn.image import resample_to_img  # narrow import: used only here

        for src_key, bias_bc_key, bias_32ch_key, out_key, label in [
            ("bold_nodummy",  "bias_bc",     "bias_32ch",     "bold_biascorr",  "bold"),
            ("sbref_raw",     "bias_bc_sb",  "bias_32ch_sb",  "sbref_biascorr", "sbref"),
        ]:
            out = p.get(out_key, "")
            if out and os.path.exists(out):
                log(f"  ⏭ {label} bias-corrected exists")
                continue

            src_path = p.get(src_key, "")
            bc_path  = p.get(bias_bc_key, "")
            ch_path  = p.get(bias_32ch_key, "")

            if not all(os.path.exists(x) for x in [src_path, bc_path, ch_path]):
                # Bias fields optional; skip gracefully
                msg = f"{label}: bias fields not found — skipping bias correction"
                log(f"  ⚠ {msg}")
                report["warnings"].append(msg)
                # Copy raw as "corrected" so downstream steps work
                if src_path and os.path.exists(src_path) and out:
                    shutil.copy2(src_path, out)
                continue

            src_img = nib.load(src_path)
            bc_img  = resample_to_img(bc_path,  src_img, interpolation="linear")
            ch_img  = resample_to_img(ch_path,  src_img, interpolation="linear")

            bc_data = bc_img.get_fdata(dtype=np.float32)
            ch_data = ch_img.get_fdata(dtype=np.float32)
            ratio   = np.where(ch_data > 0, bc_data / ch_data, 1.0)

            src_data = src_img.get_fdata(dtype=np.float32)
            # Broadcast bias ratio over time dimension if 4-D
            if src_data.ndim == 4:
                ratio = ratio[..., np.newaxis]
            corrected = src_data * ratio
            nib.Nifti1Image(corrected, src_img.affine,
                            src_img.header).to_filename(out)
            log(f"  ✓ {label} bias-corrected → {os.path.basename(out)}")

    @staticmethod
    def _write_acqparams(p, log):
        """Write acqparams.txt for topup (LR then RL phase-encode directions)."""
        acq = p.get("topup_acqp", p.get("acqparams", ""))
        if acq and os.path.exists(acq):
            return acq
        # HCP-YA: LR = [1 0 0 TRT], RL = [-1 0 0 TRT]
        trt = getattr(cfg, "TOPUP_READOUT_TIME", 0.0684)
        lines = [f" 1  0  0  {trt:.4f}\n",
                 f"-1  0  0  {trt:.4f}\n"]
        with open(acq, "w") as fh:
            fh.writelines(lines)
        log(f"  wrote acqparams: {acq}")
        return acq

    @staticmethod
    def _run_topup(p, sub, log, report):
        topup_base = p.get("topup_base", "")
        if not topup_base:
            report["warnings"].append("topup_base path not configured")
            return

        fieldcoef = topup_base + "_fieldcoef.nii.gz"
        if os.path.exists(fieldcoef):
            log(f"  ⏭ topup already done (shared)")
            return

        # Lock-file: prevent duplicate topup in parallel runs
        lock = topup_base + ".lock"
        if os.path.exists(lock):
            # Another process is running topup — poll briefly
            log(f"  ⏳ Waiting for topup lock …")
            for _ in range(120):
                time.sleep(5)
                if not os.path.exists(lock):
                    break
            if os.path.exists(fieldcoef):
                log(f"  ⏭ topup finished by parallel process")
                return
            msg = "topup lock timed out"
            log(f"  ⚠ {msg}")
            report["warnings"].append(msg)
            return

        try:
            open(lock, "w").close()

            # Merge SE field maps: LR first, RL second
            se_lr  = p["se_lr"]
            se_rl  = p["se_rl"]
            merged = topup_base + "_merged.nii.gz"
            if not os.path.exists(merged):
                _run_safe(
                    f"fslmerge -t {merged} {se_lr} {se_rl}", log)

            acq = step2._write_acqparams(p, log)
            topup_cfg = getattr(cfg, "TOPUP_CONFIG",
                                "b02b0.cnf")  # FSL default
            os.makedirs(os.path.dirname(topup_base), exist_ok=True)
            ok = _run_safe(
                f"topup --imain={merged} --datain={acq} "
                f"--config={topup_cfg} --out={topup_base} "
                f"--iout={topup_base}_iout --fout={topup_base}_fout",
                log)
            if not ok or not os.path.exists(fieldcoef):
                report["status"] = "fail"
                report["warnings"].append("topup failed")
        finally:
            if os.path.exists(lock):
                os.remove(lock)

    @staticmethod
    def _apply_topup(src, dst, topup_base, index, p, log, report, label=""):
        if os.path.exists(dst):
            log(f"  ⏭ {label} already unwarped")
            return
        fieldcoef = topup_base + "_fieldcoef.nii.gz"
        if not os.path.exists(fieldcoef):
            msg = f"topup fieldcoef not found — cannot unwarp {label}"
            log(f"  ✗ {msg}")
            report["status"] = "fail"
            report["warnings"].append(msg)
            return
        acq = p.get("topup_acqp", p.get("acqparams", ""))
        ok = _run_safe(
            f"applytopup --imain={src} --inindex={index} "
            f"--datain={acq} --topup={topup_base} "
            f"--out={dst} --method=jac",
            log)
        if not ok or not os.path.exists(dst):
            report["status"] = "fail"
            report["warnings"].append(f"applytopup failed for {label}")
            return
        log(f"  ✓ {label} → {os.path.basename(dst)}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Slice-timing correction
# ══════════════════════════════════════════════════════════════════════════════

class step3:
    """
    Step 3: Slice-timing correction.

    Writes the HCP-YA MB=8 interleaved slice-timing vector to a text file,
    then runs FSL slicetimer using that custom timing file.

    HCP-YA specifics (from config_hcp.HCP_SLICE_TIMING):
        MB_FACTOR = 8, 72 slices → 9 slice groups
        dt = 77.5 ms (from DICOM header; NOT TR/9 = 80.0 ms)
        acquisition order: [0,2,4,6,8,1,3,5,7] × 8 simultaneous slices

    Returns
    -------
    dict with keys: subject, run, step (=3), status, source
    """

    @staticmethod
    def run(sub: str, run: str = "LR", logger=None) -> dict:
        p   = cfg.subject_dirs(sub, run)
        log = logger or print
        _ensure_dirs(p)
        report = {
            "subject": sub, "run": run, "step": 3,
            "status": "ok", "source": "bold_unwarped",
        }

        log(f"\n  ━━ Step 3: Slice-Timing Correction ━━")

        out = p["bold_stc"]
        if os.path.exists(out):
            log(f"  ⏭ bold_stc exists — skipping")
            return report

        # Write slice timing file
        st_file = os.path.join(p["outdir"], "slice_timing.txt")
        timing  = cfg.HCP_SLICE_TIMING
        with open(st_file, "w") as fh:
            fh.write("\n".join(str(t) for t in timing) + "\n")
        log(f"  Slice timing: {len(timing)} entries, dt={timing[1]-timing[0]:.4f}s")

        in_path = p["bold_unwarped"]
        if not os.path.exists(in_path):
            log(f"  ✗ bold_unwarped not found")
            report["status"] = "fail"
            return report

        ok = _run_safe(
            f"slicetimer -i {in_path} -o {out} "
            f"-r {cfg.TR} --ocustom={st_file}",
            log)

        if not ok or not os.path.exists(out):
            report["status"] = "fail"
            log(f"  ✗ slicetimer failed")
            return report

        log(f"  ✓ bold_stc → {os.path.basename(out)}")
        return report


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Motion correction + framewise displacement
# ══════════════════════════════════════════════════════════════════════════════

class step4:
    """
    Step 4: Motion correction (FSL mcflirt) + framewise displacement.

    Runs mcflirt on bold_stc using sbref_unwarped as the reference volume,
    computes FD from the motion parameter file (50 mm head radius for
    rotations), and saves mean_bold via fslmaths -Tmean.

    Returns
    -------
    dict with keys: subject, run, step (=4), status,
                    mean_fd, max_fd, n_high_motion
    """

    @staticmethod
    def run(sub: str, run: str = "LR", logger=None) -> dict:
        p   = cfg.subject_dirs(sub, run)
        log = logger or print
        _ensure_dirs(p)
        report = {
            "subject": sub, "run": run, "step": 4,
            "status": "ok", "mean_fd": None, "max_fd": None,
            "n_high_motion": None,
        }

        log(f"\n  ━━ Step 4: Motion Correction ━━")

        # Reference volume: sbref_unwarped preferred (high SNR single-band)
        ref = (p["sbref_unwarped"] if os.path.exists(p["sbref_unwarped"])
               else p["mean_bold"])
        log(f"  Reference: {os.path.basename(ref)}")

        mc_out = p["bold_mc"]
        par    = p["mc_par"]

        if not os.path.exists(mc_out):
            if not os.path.exists(p["bold_stc"]):
                log(f"  ✗ bold_stc not found")
                report["status"] = "fail"
                return report
            # Strip .nii.gz for mcflirt -out
            mc_prefix = mc_out.replace(".nii.gz", "")
            ok = _run_safe(
                f"mcflirt -in {p['bold_stc']} -out {mc_prefix} "
                f"-reffile {ref} -plots -report -mats",
                log)
            if not ok or not os.path.exists(mc_out):
                report["status"] = "fail"
                log(f"  ✗ mcflirt failed")
                return report
            log(f"  ✓ bold_mc → {os.path.basename(mc_out)}")

        # Mean BOLD
        mean = p["mean_bold"]
        if not os.path.exists(mean):
            _run_safe(f"fslmaths {mc_out} -Tmean {mean}", log)
            log(f"  ✓ mean_bold → {os.path.basename(mean)}")

        # Framewise displacement
        if not os.path.exists(par):
            log(f"  ⚠ mc.par not found — cannot compute FD")
            report["warnings"] = ["mc.par missing"]
            return report

        fd = step4._compute_fd(par, p["fd_file"], log)
        if fd is not None:
            report["mean_fd"]       = float(np.mean(fd))
            report["max_fd"]        = float(np.max(fd))
            report["n_high_motion"] = int((fd > 0.5).sum())
            log(f"  FD: mean={report['mean_fd']:.3f} mm  "
                f"max={report['max_fd']:.3f} mm  "
                f">0.5mm: {report['n_high_motion']}")

        return report

    @staticmethod
    def _compute_fd(par_file, fd_file, log) -> np.ndarray:
        """
        Compute framewise displacement from mcflirt .par file.

        Columns: Rx Ry Rz Tx Ty Tz (radians, mm)
        Rotations converted to mm using 50 mm head radius.
        FD = sum of absolute first differences across all 6 params.
        """
        HEAD_RADIUS_MM = 50.0
        try:
            params = np.loadtxt(par_file)   # shape (N, 6)
        except Exception as e:
            log(f"  ⚠ Could not load mc.par: {e}")
            return None

        params_mm = params.copy()
        params_mm[:, :3] *= HEAD_RADIUS_MM  # rotations → mm

        diff = np.abs(np.diff(params_mm, axis=0))  # shape (N-1, 6)
        fd   = diff.sum(axis=1)                     # shape (N-1,)

        np.savetxt(fd_file, fd, fmt="%.6f")
        log(f"  FD saved → {os.path.basename(fd_file)}")
        return fd


# ══════════════════════════════════════════════════════════════════════════════
# Step 4b — Mean + robust variance (from bold_unwarped, before motion corr.)
# ══════════════════════════════════════════════════════════════════════════════

class step4b:
    """
    Step 4b: Mean and robust variance maps from bold_unwarped.

    Computed from bold_unwarped (after topup, before motion correction) so
    the variance map captures signal variance uncorrupted by interpolation
    artefacts from rigid-body realignment.

    4b.1  FSL fslmaths -Tmean → mean_bold_nomc.nii.gz
    4b.2  Trimmed-mean variance (5% each tail) computed slice-by-slice
          in Python to stay under the ~6.5 GB peak RAM that would result
          from loading the full 4-D HCP-YA volume at once.
          → variance_nomc.nii.gz

    Returns
    -------
    dict with keys: subject, run, step (="4b"), status
    """

    @staticmethod
    def run(sub: str, run: str = "LR", logger=None) -> dict:
        p   = cfg.subject_dirs(sub, run)
        log = logger or print
        _ensure_dirs(p)
        report = {
            "subject": sub, "run": run, "step": "4b",
            "status": "ok",
        }

        log(f"\n  ━━ Step 4b: Mean + Variance (pre-MC) ━━")

        in_path = p["bold_unwarped"]
        if not os.path.exists(in_path):
            log(f"  ✗ bold_unwarped not found")
            report["status"] = "fail"
            return report

        # 4b.1 Mean
        mean_out = p.get("mean_bold_nomc",
                         os.path.join(p["outdir"], "mean_bold_nomc.nii.gz"))
        if not os.path.exists(mean_out):
            ok = _run_safe(f"fslmaths {in_path} -Tmean {mean_out}", log)
            if ok:
                log(f"  ✓ mean_bold_nomc")
            else:
                report["status"] = "fail"
                return report

        # 4b.2 Robust variance (slice-by-slice to cap RAM)
        var_out = p.get("variance_nomc",
                        os.path.join(p["outdir"], "variance_nomc.nii.gz"))
        if not os.path.exists(var_out):
            step4b._compute_variance(in_path, var_out, log)
            if not os.path.exists(var_out):
                report["status"] = "fail"

        return report

    @staticmethod
    def _compute_variance(bold_path, var_out, log):
        """
        Compute trimmed (5% each tail) temporal variance slice by slice.

        HCP-YA volumes are large (91×109×91 × 1200 @ float32 ≈ 4.3 GB).
        Loading slice by slice keeps peak RAM well under 1 GB.
        """

        log(f"  Computing robust variance (5% trim, slice-by-slice) …")
        ref_img = nib.load(bold_path)
        nx, ny, nz = ref_img.shape[:3]
        var_map = np.zeros((nx, ny, nz), dtype=np.float32)

        # Read the raw gzip stream slice-by-slice via nibabel proxy
        proxy = ref_img.dataobj
        for z in range(nz):
            sl = np.array(proxy[:, :, z, :], dtype=np.float32)  # (nx, ny, T)
            sl_flat = sl.reshape(-1, sl.shape[-1])
            trim_frac = 0.05
            n = sl_flat.shape[1]
            k = max(1, int(np.floor(n * trim_frac)))
            sl_sorted = np.sort(sl_flat, axis=1)
            sl_trim   = sl_sorted[:, k: n - k]
            var_map[:, :, z] = sl_trim.var(axis=1).reshape(nx, ny)

        nib.Nifti1Image(var_map, ref_img.affine,
                        ref_img.header).to_filename(var_out)
        log(f"  ✓ variance_nomc → {os.path.basename(var_out)}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Registration + tissue masks + WM refinement (consolidated)
# ══════════════════════════════════════════════════════════════════════════════

# QC thresholds (EPI space, 2 mm iso)
_WM_ERO1_MIN  =  3_000
_WM_ERO1_MAX  = 20_000
_GM_MIN_VOX   = 30_000
_GM_MAX_VOX   = 180_000
_FV_MIN_VOX   = 20

# aseg label groups
_WM_LABELS        = [2, 41, 7, 46]    # cerebral WM L/R + cerebellar WM L/R
_LATVENT_LABELS   = [4, 43, 14]       # lateral + 3rd ventricle (exclusion buffer)
_FOURTHVENT_LABEL = 15                # 4th ventricle (CSF ROI — NOT excluded from WM)
_VENT_DILATE_VOX  = 1                 # dilation buffer around exclusion zone

# Overlay label map for FSLeyes (single-volume integer labels)
_OVERLAY_LABELS = {"WM_raw": 1, "WM_final": 2, "excl_zone": 3, "4thvent_roi": 4}

# Montage colours (RGBA)
_COL_WM_RAW   = np.array([1.0, 0.8, 0.0, 0.35])
_COL_WM_FINAL = np.array([0.1, 0.8, 0.2, 0.55])
_COL_EXCL     = np.array([0.9, 0.1, 0.1, 0.50])
_COL_4THVENT  = np.array([0.0, 0.8, 1.0, 0.65])


def _5_dilate(mask, n):
    s = np.ones((3, 3, 3), dtype=bool)
    out = mask.copy()
    for _ in range(n):
        out = binary_dilation(out, structure=s)
    return out


def _5_sample_slices(n_total, n_slices):
    margin = max(1, n_total // (n_slices + 1))
    return list(range(margin, n_total - margin,
                      max(1, (n_total - 2 * margin) // (n_slices - 1))))[:n_slices]


def _5_overlay_rgb(bg, masks_colours):
    lo, hi = bg.min(), bg.max()
    bg_n = (bg - lo) / (hi - lo) if hi > lo else np.zeros_like(bg, float)
    rgb  = np.stack([bg_n, bg_n, bg_n], axis=-1)
    for mask, colour in masks_colours:
        a = colour[3]
        for c in range(3):
            rgb[:, :, c] = np.where(mask,
                                    rgb[:, :, c] * (1 - a) + colour[c] * a,
                                    rgb[:, :, c])
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _5_make_montage(sub, run, mean_bold, wm_raw, wm_final,
                    excl_zone, fourthvent, out_path):
    nx, ny, nz = mean_bold.shape
    ax_sl  = _5_sample_slices(nz, 9)
    cor_sl = _5_sample_slices(ny, 5)
    sag_sl = _5_sample_slices(nx, 5)
    n_cols = max(len(ax_sl), len(cor_sl), len(sag_sl))

    fig = plt.figure(figsize=(2.2 * n_cols, 7.0), facecolor="black")
    gs  = gridspec.GridSpec(3, n_cols, figure=fig,
                            hspace=0.04, wspace=0.02,
                            left=0.01, right=0.99,
                            top=0.88, bottom=0.01)
    mc = [(wm_raw, _COL_WM_RAW), (excl_zone, _COL_EXCL),
          (wm_final, _COL_WM_FINAL), (fourthvent, _COL_4THVENT)]
    row_defs = [
        (ax_sl,  "Axial",    lambda z: mean_bold[:, :, z].T,
         lambda z, m: m[:, :, z].T),
        (cor_sl, "Coronal",  lambda y: mean_bold[:, y, :].T,
         lambda y, m: m[:, y, :].T),
        (sag_sl, "Sagittal", lambda x: mean_bold[x, :, :].T,
         lambda x, m: m[x, :, :].T),
    ]
    for row_i, (slices, label, bg_fn, mask_fn) in enumerate(row_defs):
        for ci, idx in enumerate(slices):
            ax = fig.add_subplot(gs[row_i, ci])
            ax.set_facecolor("black")
            rgb = _5_overlay_rgb(bg_fn(idx),
                                 [(mask_fn(idx, m), c) for m, c in mc])
            ax.imshow(rgb, origin="lower", aspect="equal",
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(label, color="white", fontsize=8, fontweight="bold")
            ax.text(0.5, 0.02, f"{['z','y','x'][row_i]}={idx}",
                    transform=ax.transAxes, ha="center",
                    fontsize=6.5, color="white", alpha=0.8)
        for ci in range(len(slices), n_cols):
            ax = fig.add_subplot(gs[row_i, ci])
            ax.set_facecolor("black"); ax.axis("off")

    n_raw = int(wm_raw.sum()); n_fin = int(wm_final.sum())
    n_excl = int(excl_zone.sum()); n_4v = int(fourthvent.sum())
    pct = n_fin / n_raw * 100 if n_raw else 0
    fig.suptitle(
        f"WM Mask QC — {sub} / {run}\n"
        f"WM raw={n_raw:,}   WM ero1_novent={n_fin:,} ({pct:.0f}%)   "
        f"Excl={n_excl:,}   4th-vent={n_4v:,}",
        color="white", fontsize=10, fontweight="bold", y=0.97)
    fig.legend(handles=[
        Patch(facecolor=_COL_WM_RAW[:3],   alpha=0.8,
              label=f"WM raw ({n_raw:,} vox)"),
        Patch(facecolor=_COL_WM_FINAL[:3], alpha=0.8,
              label=f"WM ero1_novent ({n_fin:,} vox) ← PRIMARY"),
        Patch(facecolor=_COL_EXCL[:3],     alpha=0.8,
              label=f"Excl zone ({n_excl:,} vox) — lat+3rd vent ±1 vox"),
        Patch(facecolor=_COL_4THVENT[:3],  alpha=0.8,
              label=f"4th vent ({n_4v:,} vox) — CSF ROI"),
    ], loc="lower center", ncol=2, fontsize=7.5, frameon=False,
       facecolor="black", labelcolor="white",
       bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)

class step5:
    """
    Step 5: EPI-to-T1 registration, tissue mask extraction, WM refinement, 4th-ventricle splitting, and QC montage.
    Registration strategy
    ─────────────────────
    PRIMARY (FreeSurfer available):
      5.1  Convert brain.mgz + aseg.mgz → NIfTI
      5.2  bbregister: EPI SBRef → T1 (surface-constrained BBR, ~0.5 mm accuracy)
           Falls back to FLIRT 6-DOF if bbregister fails or is unavailable.
      5.3  lta_convert: VOX-to-VOX LTA → RAS-to-RAS (type 1)
           Critical: bbregister writes VOX-to-VOX (type 0); applying it to a
           NIfTI without conversion causes silent coordinate-frame mismatch.
      5.4  mri_vol2vol: project aseg.mgz directly (preserves FS coords, --nearest)
      5.5  Extract raw tissue masks from aseg_epi
      5.6  Erode WM mask: 1-vox (primary) and 2-vox (sensitivity)
           Subtract ventricle buffer (lat+3rd vent dilated 1 vox)
           → wm_mask_ero1_novent (PRIMARY for regression P2/P3)
           → wm_mask_ero2_novent (sensitivity check)
           Note: ero2_novent too aggressive for HCP-YA 2×2×2 mm; ero1 used.
      5.7  Split 4th ventricle into upper/lower halves (z-median split)
           → fourthvent_lower.nii.gz, fourthvent_upper.nii.gz
      5.8  QC overlay (FSLeyes-ready) + montage PNG

    Returns
    -------
    dict with keys: subject, run, step (=5), status, warnings, method,
                    voxel counts, qc_flag
    """

    @staticmethod
    def run(sub: str, run: str = "LR", logger=None, force: bool = False) -> dict:
        p   = cfg.subject_dirs(sub, run)
        log = logger or print
        _ensure_dirs(p)
        report = {
            "subject": sub, "run": run, "step": 5,
            "status": "ok", "warnings": [], "method": "freesurfer",
            "qc_flag": "",
            # voxel counts filled in below
            "n_wm_raw": 0, "n_wm_ero1_novent": 0, "n_wm_ero2_novent": 0,
            "n_brain": 0, "n_gm": 0, "n_4thvent": 0,
            "n_4thvent_lower": 0, "n_4thvent_upper": 0,
            "pct_retained_ero1": 0.0,
        }

        log(f"\n  ━━ Step 5: Registration + Tissue Masks + WM Refinement ━━")

        use_fs = getattr(cfg, "USE_FREESURFER", True) and step5._has_freesurfer(sub)
        log(f"  Method: {'FreeSurfer + bbregister' if use_fs else 'FSL fallback'}")

        # EPI reference — sbref_unwarped preferred; fallback to mean_bold
        epi_ref = (p["sbref_unwarped"] if os.path.exists(p["sbref_unwarped"])
                   else p["mean_bold"])
        log(f"  EPI ref: {os.path.basename(epi_ref)}")

        outdir     = p["outdir"]
        masks_dir  = p["masks_dir"]
        brain_nii  = os.path.join(outdir, "T1w_brain.nii.gz")
        aseg_nii   = os.path.join(outdir, "aseg.nii.gz")
        lta_file   = os.path.join(outdir, "epi2t1.lta")
        lta_ras    = os.path.join(outdir, "epi2t1_ras.lta")
        fsl_mat    = os.path.join(outdir, "epi2t1.mat")
        inv_mat    = os.path.join(outdir, "t1_to_epi.mat")
        aseg_epi   = os.path.join(outdir, "aseg_epi.nii.gz")
        wm_mask    = p["wm_mask"]
        brain_mask = p["brain_mask"]
        gm_mask    = p.get("gm_mask_epi",
                           os.path.join(masks_dir, "gm_mask_epi.nii.gz"))
        fv_mask    = p["fourthvent_mask"]

        # ── 5.1  Convert MGZ → NIfTI ─────────────────────────────────────────
        if use_fs:
            log(f"\n  [5.1] Convert MGZ → NIfTI")
            step5._mgz_to_nifti(step5._fs_mri(sub, "brain.mgz"),    brain_nii, log)
            step5._mgz_to_nifti(step5._fs_mri(sub, "aseg.mgz"),     aseg_nii,  log)
            if not os.path.exists(brain_nii) or not os.path.exists(aseg_nii):
                report["status"] = "fail"; return report

            # ── 5.2  bbregister EPI → T1 ─────────────────────────────────────
            # bbregister uses FreeSurfer surface geometry for boundary-based
            # registration — far more accurate than pure intensity (FLIRT 6-DOF).
            # Output is an LTA file (VOX-to-VOX by default from bbregister).
            log(f"\n  [5.2] bbregister: EPI → T1 (surface-constrained BBR)")
            if os.path.exists(lta_file):
                # bbregister succeeded on a prior run
                use_lta = True
                log(f"  skip exists: epi2t1.lta")
            elif os.path.exists(fsl_mat) and os.path.exists(inv_mat):
                # FLIRT fallback already ran on a prior run — don't redo
                use_lta = False
                report["method"] = "flirt_6dof"
                log(f"  skip exists: epi2t1.mat (FLIRT fallback from prior run)")
            else:
                bbr_ok = _run_safe(
                    f"bbregister --s {sub} --mov {epi_ref} "
                    f"--reg {lta_file} --bold --init-fsl", log)
                if not bbr_ok or not os.path.exists(lta_file):
                    # Fallback: FLIRT 6-DOF (less accurate but always available)
                    log(f"  ⚠ bbregister failed — falling back to FLIRT 6-DOF")
                    report["warnings"].append("bbregister failed; used FLIRT 6-DOF")
                    report["method"] = "flirt_6dof"
                    _run_safe(
                        f"flirt -in {epi_ref} -ref {brain_nii} "
                        f"-omat {fsl_mat} -dof 6", log)
                    if not os.path.exists(fsl_mat):
                        report["status"] = "fail"; return report
                    _run_safe(f"convert_xfm -omat {inv_mat} -inverse {fsl_mat}", log)
                    use_lta = False
                else:
                    use_lta = True
                    log(f"  ✓ bbregister LTA written")

            # If bbregister succeeded, derive FSL .mat for any FSL tools that
            # need it, and produce the inverse for T1→EPI projection.
            if use_lta:
                # ── 5.3  Convert LTA VOX→VOX to RAS→RAS ─────────────────────
                # bbregister often writes type-0 (VOX-to-VOX) transforms.
                # Applying a VOX-to-VOX LTA to a NIfTI (which has a RAS affine)
                # causes a silent coordinate-frame mismatch — masks land in the
                # wrong place.  Converting to type-1 (RAS-to-RAS) first fixes this.
                log(f"\n  [5.3] lta_convert: VOX-to-VOX → RAS-to-RAS")
                if not os.path.exists(lta_ras):
                    _run_safe(
                        f"lta_convert --inlta {lta_file} --outlta {lta_ras} "
                        f"--src {step5._fs_mri(sub, 'aseg.mgz')} "
                        f"--trg {epi_ref} --ltavox2vox", log)
                lta_use = lta_ras if os.path.exists(lta_ras) else lta_file

                # Also derive FSL .mat + inverse (needed for brain mask projection)
                if not os.path.exists(fsl_mat):
                    _run_safe(
                        f"lta_convert --inlta {lta_use} --outfsl {fsl_mat} "
                        f"--src {epi_ref} --trg {brain_nii}", log)
                if not os.path.exists(inv_mat) and os.path.exists(fsl_mat):
                    _run_safe(f"convert_xfm -omat {inv_mat} -inverse {fsl_mat}", log)
            else:
                lta_use = None

            log(f"  ✓ Registration matrices ready")

            # ── 5.4  Project aseg.mgz → EPI space ────────────────────────────
            # Pass aseg.mgz DIRECTLY (not the NIfTI) to mri_vol2vol so the
            # FreeSurfer coordinate frame is preserved.  Uses --nearest to keep
            # integer labels exact.
            log(f"\n  [5.4] Project aseg → EPI space")
            if not os.path.exists(aseg_epi):
                if lta_use and _run_safe(
                    f"mri_vol2vol "
                    f"--mov {step5._fs_mri(sub, 'aseg.mgz')} "
                    f"--targ {epi_ref} "
                    f"--lta {lta_use} "
                    f"--o {aseg_epi} --nearest", log):
                    # Round and cast to int32 to guarantee exact labels
                    img  = nib.load(aseg_epi)
                    data = np.round(img.get_fdata()).astype(np.int32)
                    nib.Nifti1Image(data, img.affine,
                                    img.header).to_filename(aseg_epi)
                    log(f"  aseg_epi: {len(np.unique(data))-1} labels")
                else:
                    # Last resort: FLIRT applyxfm on the NIfTI aseg
                    log(f"  ⚠ mri_vol2vol failed — using FLIRT applyxfm on aseg NIfTI")
                    report["warnings"].append("mri_vol2vol failed; used FLIRT applyxfm")
                    _run_safe(
                        f"flirt -in {aseg_nii} -ref {epi_ref} "
                        f"-applyxfm -init {inv_mat} "
                        f"-out {aseg_epi} -interp nearestneighbour", log)
            if not os.path.exists(aseg_epi):
                report["status"] = "fail"; return report

            # ── 5.5  Extract raw tissue masks from aseg_epi ───────────────────
            log(f"\n  [5.5] Extract tissue masks")
            gm_labels = [3, 42, 8, 47, 17, 53, 18, 54,
                         11, 50, 12, 51, 13, 52, 26, 58, 10, 49]
            step5._extract_mask(aseg_epi, _WM_LABELS,  wm_mask, log)
            step5._extract_mask(aseg_epi, [_FOURTHVENT_LABEL], fv_mask, log)
            step5._extract_mask(aseg_epi, gm_labels,   gm_mask, log)

            # Brain mask from brainmask.mgz projected via same transform
            brainmask_nii = os.path.join(outdir, "brainmask.nii.gz")
            brainmask_epi = os.path.join(outdir, "brainmask_epi.nii.gz")
            step5._mgz_to_nifti(step5._fs_mri(sub, "brainmask.mgz"),
                                brainmask_nii, log)
            if os.path.exists(brainmask_nii) and not os.path.exists(brainmask_epi):
                _run_safe(
                    f"flirt -in {brainmask_nii} -ref {epi_ref} "
                    f"-applyxfm -init {inv_mat} "
                    f"-out {brainmask_epi} -interp nearestneighbour", log)
            if os.path.exists(brainmask_epi):
                img = nib.load(brainmask_epi)
                bm  = (img.get_fdata() > 0).astype(np.uint8)
                nib.Nifti1Image(bm, img.affine, img.header).to_filename(brain_mask)
                report["n_brain"] = int(bm.sum())
                log(f"  brain_mask: {report['n_brain']:,} voxels")

        else:
            # FSL fallback (no FreeSurfer)
            report["method"] = "fsl_fallback"
            log(f"\n  [5.1-5.5] FSL fallback (no FreeSurfer)")
            step5._fsl_fallback(p, epi_ref, fsl_mat, inv_mat,
                                brain_mask, wm_mask, outdir, log, report)
            aseg_epi = os.path.join(outdir, "aseg_epi.nii.gz")

        # Abort if critical masks are missing
        for label, path in [("brain_mask", brain_mask), ("wm_mask", wm_mask)]:
            if not os.path.exists(path):
                log(f"  ✗ MISSING: {label}")
                report["status"] = "fail"
        if report["status"] == "fail":
            return report

        # Record raw voxel counts
        report["n_wm_raw"] = int(nib.load(wm_mask).get_fdata().sum())
        if os.path.exists(fv_mask):
            report["n_4thvent"] = int(nib.load(fv_mask).get_fdata().sum())
        if os.path.exists(gm_mask):
            report["n_gm"] = int(nib.load(gm_mask).get_fdata().sum())

        # ── 5.6  WM erosion + ventricle exclusion ────────────────────────────
        log(f"\n  [5.6] WM mask refinement (erosion + ventricle exclusion)")
        try:
            wm_img   = nib.load(wm_mask)
            affine   = wm_img.affine
            header   = wm_img.header
            wm_raw   = wm_img.get_fdata(dtype=np.float32) > 0.5
            n_raw    = int(wm_raw.sum())
            log(f"  WM raw = {n_raw:,} vox")

            # Build ventricle exclusion zone from aseg_epi
            latvent_bin    = np.zeros(wm_raw.shape, dtype=bool)
            fourthvent_bin = np.zeros(wm_raw.shape, dtype=bool)
            if os.path.exists(aseg_epi):
                aseg = np.round(nib.load(aseg_epi).get_fdata()).astype(int)
                for lbl in _LATVENT_LABELS:
                    latvent_bin |= (aseg == lbl)
                fourthvent_bin = (aseg == _FOURTHVENT_LABEL)
            else:
                log(f"  ⚠ aseg_epi not found — ventricle exclusion skipped")
                report["warnings"].append("aseg_epi missing; no ventricle exclusion")

            n_latvent = int(latvent_bin.sum())
            n_4thvent = int(fourthvent_bin.sum())
            report["n_4thvent"] = n_4thvent
            log(f"  lat+3rd vent={n_latvent:,}  4th vent={n_4thvent:,}")

            # Dilate exclusion zone by 1 vox buffer
            latvent_dil = _5_dilate(latvent_bin, _VENT_DILATE_VOX)

            # Erode WM
            wm_ero1 = erode_mask(wm_raw, 1)   # PRIMARY for HCP-YA 2×2×2 mm
            wm_ero2 = erode_mask(wm_raw, 2)   # sensitivity
            wm_ero1_nv = wm_ero1 & ~latvent_dil
            wm_ero2_nv = wm_ero2 & ~latvent_dil

            report["n_wm_ero1_novent"] = int(wm_ero1_nv.sum())
            report["n_wm_ero2_novent"] = int(wm_ero2_nv.sum())
            pct = report["n_wm_ero1_novent"] / n_raw * 100 if n_raw else 0
            report["pct_retained_ero1"] = round(pct, 1)
            log(f"  ero1_novent={report['n_wm_ero1_novent']:,} ({pct:.1f}%)  "
                f"ero2_novent={report['n_wm_ero2_novent']:,}")

            # QC flags
            flags = []
            if report["n_wm_ero1_novent"] < _WM_ERO1_MIN:
                flags.append(f"ERO1_NOVENT_TOO_SMALL(<{_WM_ERO1_MIN})")
            if report["n_wm_ero1_novent"] > _WM_ERO1_MAX:
                flags.append(f"ERO1_NOVENT_TOO_LARGE(>{_WM_ERO1_MAX})")
            if n_latvent == 0:
                flags.append("NO_LATVENT_MASK")
            if n_4thvent == 0:
                flags.append("NO_4THVENT_MASK")
            if n_4thvent < _FV_MIN_VOX:
                flags.append(f"FV_TOO_SMALL(<{_FV_MIN_VOX})")
            report["qc_flag"] = " | ".join(flags) or "OK"

            # Save refined WM masks
            ero1_path    = os.path.join(masks_dir, "wm_mask_ero1.nii.gz")
            ero2_path    = os.path.join(masks_dir, "wm_mask_ero2.nii.gz")
            latvent_path = os.path.join(masks_dir, "latvent_excl_mask.nii.gz")
            latdil_path  = os.path.join(masks_dir, "latvent_excl_dil1.nii.gz")
            save_nii(wm_ero1,     affine, header, ero1_path)
            save_nii(wm_ero1_nv,  affine, header, p["wm_mask_ero1_novent"])
            save_nii(wm_ero2,     affine, header, ero2_path)
            save_nii(wm_ero2_nv,  affine, header, p["wm_mask_ero2_novent"])
            save_nii(latvent_bin, affine, header, latvent_path)
            save_nii(latvent_dil, affine, header, latdil_path)
            save_nii(fourthvent_bin, affine, header, fv_mask)

        except Exception as e:
            log(f"  ✗ WM refinement error: {e}\n{traceback.format_exc()}")
            report["warnings"].append(f"WM refinement failed: {e}")
            wm_ero1_nv     = None
            latvent_dil    = None
            fourthvent_bin = None

        # ── 5.7  Split 4th ventricle upper / lower ────────────────────────────
        log(f"\n  [5.7] 4th ventricle upper/lower split")
        fv_low_path = os.path.join(masks_dir, "fourthvent_lower.nii.gz")
        fv_up_path  = os.path.join(masks_dir, "fourthvent_upper.nii.gz")
        try:
            if fourthvent_bin is not None and fourthvent_bin.any():
                fv_zs = np.where(fourthvent_bin.sum(axis=(0, 1)) > 0)[0]
                if len(fv_zs) >= 2:
                    fv_zmed = int(np.median(fv_zs))
                    fv_lower = fourthvent_bin.copy()
                    fv_lower[:, :, fv_zmed + 1:] = False
                    fv_upper = fourthvent_bin.copy()
                    fv_upper[:, :, :fv_zmed + 1] = False
                    report["n_4thvent_lower"] = int(fv_lower.sum())
                    report["n_4thvent_upper"] = int(fv_upper.sum())
                    log(f"  4th vent z={int(fv_zs.min())}–{int(fv_zs.max())} "
                        f"split at z={fv_zmed}  "
                        f"lower={report['n_4thvent_lower']}  "
                        f"upper={report['n_4thvent_upper']}")
                    save_nii(fv_lower, affine, header, fv_low_path)
                    save_nii(fv_upper, affine, header, fv_up_path)
                else:
                    log(f"  ⚠ 4th vent only {len(fv_zs)} z-slice — split skipped")
                    report["warnings"].append("FV_TOO_THIN_TO_SPLIT")
                    report["qc_flag"] += " | FV_TOO_THIN_TO_SPLIT"
        except Exception as e:
            log(f"  ⚠ 4th vent split error: {e}")
            report["warnings"].append(f"4th vent split failed: {e}")

        # ── 5.8  QC overlay + montage PNG ─────────────────────────────────────
        log(f"\n  [5.8] QC overlay + montage")
        try:
            if wm_ero1_nv is not None:
                overlay = np.zeros(wm_raw.shape, dtype=np.uint8)
                overlay[wm_raw]            = _OVERLAY_LABELS["WM_raw"]
                overlay[wm_ero1_nv]        = _OVERLAY_LABELS["WM_final"]
                if latvent_dil is not None:
                    overlay[latvent_dil]   = _OVERLAY_LABELS["excl_zone"]
                if fourthvent_bin is not None:
                    overlay[fourthvent_bin] = _OVERLAY_LABELS["4thvent_roi"]
                ov_img = nib.Nifti1Image(overlay, affine, header.copy())
                ov_img.header.set_intent(1002)  # NIFTI_INTENT_LABEL
                nib.save(ov_img, os.path.join(masks_dir, "wm_mask_qc_overlay.nii.gz"))

                mean_bold_path = os.path.join(outdir, "mean_bold.nii.gz")
                bg = (nib.load(mean_bold_path).get_fdata(dtype=np.float32)
                      if os.path.exists(mean_bold_path)
                      else wm_raw.astype(np.float32))
                montage_path = os.path.join(masks_dir,
                                            f"wm_mask_montage_{sub}_{run}.png")
                _5_make_montage(sub, run, bg, wm_raw, wm_ero1_nv,
                                latvent_dil if latvent_dil is not None
                                else np.zeros_like(wm_raw),
                                fourthvent_bin if fourthvent_bin is not None
                                else np.zeros_like(wm_raw),
                                montage_path)
                log(f"  ✓ QC montage → {os.path.basename(montage_path)}")
        except Exception as e:
            log(f"  ⚠ QC overlay error: {e}")
            report["warnings"].append(f"QC overlay failed: {e}")

        # ── Final verify ───────────────────────────────────────────────────────
        log(f"\n  [5.9] Verify outputs")
        for label, path in [
            ("brain_mask",       brain_mask),
            ("wm_mask",          wm_mask),
            ("wm_mask_ero1_nov", p["wm_mask_ero1_novent"]),
            ("4thvent",          fv_mask),
            ("gm_mask",          gm_mask),
        ]:
            if os.path.exists(path):
                n = int(nib.load(path).get_fdata().sum())
                log(f"  {label}: {n:,} vox")
            else:
                log(f"  {label}: MISSING (optional)" if label in ("4thvent", "gm_mask")
                    else f"  ✗ MISSING: {label}")
                if label not in ("4thvent", "gm_mask"):
                    report["status"] = "fail"

        if report["status"] == "ok":
            log(f"\n  ✓ Step 5 complete ({report['method']})  "
                f"qc_flag={report['qc_flag']}")
        return report

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _fs_mri(sub, fname):
        return os.path.join(cfg.FS_DIR, sub, "mri", fname)

    @staticmethod
    def _has_freesurfer(sub):
        return os.path.exists(step5._fs_mri(sub, "brain.mgz"))

    @staticmethod
    def _mgz_to_nifti(mgz, nii, log):
        if os.path.exists(nii):
            log(f"  skip exists: {os.path.basename(nii)}")
            return True
        return _run_safe(f"mri_convert {mgz} {nii}", log)

    @staticmethod
    def _extract_mask(aseg_epi, labels, out_path, log):
        """Binary mask for given aseg label(s) in EPI space."""
        if os.path.exists(out_path):
            log(f"  skip exists: {os.path.basename(out_path)}")
            return True
        img  = nib.load(aseg_epi)
        data = np.round(img.get_fdata()).astype(np.int32)
        mask = np.zeros(data.shape, dtype=np.uint8)
        for lbl in labels:
            mask[data == lbl] = 1
        nib.Nifti1Image(mask, img.affine, img.header).to_filename(out_path)
        log(f"  {os.path.basename(out_path)}: {int(mask.sum())} vox "
            f"(labels {labels})")
        return True

    @staticmethod
    def _fsl_fallback(p, epi_ref, fsl_mat, inv_mat,
                      brain_mask, wm_mask, outdir, log, report):
        """FSL BET + FAST + FLIRT BBR fallback when FreeSurfer is unavailable."""
        t1_raw   = p.get("t1w_raw", "")
        t1_brain = os.path.join(outdir, "T1w_brain.nii.gz")
        wm_pve   = os.path.join(outdir, "T1w_fast_pve_2.nii.gz")
        wm_t1    = os.path.join(outdir, "wm_mask_t1.nii.gz")

        if not t1_raw or not os.path.exists(t1_raw):
            log(f"  ✗ No T1 for FSL fallback")
            report["status"] = "fail"; return

        # BET
        if not os.path.exists(t1_brain):
            prefix = t1_brain.replace(".nii.gz", "")
            _run_safe(f"bet {t1_raw} {prefix} -R -f 0.35 -g 0 -m", log)

        # FAST tissue segmentation
        fast_pre = os.path.join(outdir, "T1w_fast")
        if not os.path.exists(wm_pve):
            _run_safe(
                f"fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 "
                f"-o {fast_pre} {t1_brain}", log)
        if os.path.exists(wm_pve):
            _run_safe(f"fslmaths {wm_pve} -thr 0.9 -bin {wm_t1}", log)

        # FLIRT (6-DOF, then BBR)
        if not os.path.exists(fsl_mat):
            init_mat = fsl_mat.replace(".mat", "_init.mat")
            _run_safe(
                f"flirt -in {epi_ref} -ref {t1_brain} "
                f"-omat {init_mat} -dof 6", log)
            if os.path.exists(wm_t1) and os.path.exists(init_mat):
                ok = _run_safe(
                    f"flirt -in {epi_ref} -ref {t1_brain} "
                    f"-omat {fsl_mat} -dof 6 -cost bbr "
                    f"-wmseg {wm_t1} -init {init_mat} "
                    f"-schedule $FSLDIR/etc/flirtsch/bbr.sch", log)
                if not ok:
                    shutil.copy2(init_mat, fsl_mat)
            elif os.path.exists(init_mat):
                shutil.copy2(init_mat, fsl_mat)

        if not os.path.exists(fsl_mat):
            report["status"] = "fail"; return

        # Invert
        _run_safe(f"convert_xfm -omat {inv_mat} -inverse {fsl_mat}", log)

        # Project masks to EPI
        t1_mask = t1_brain.replace(".nii.gz", "_mask.nii.gz")
        for src, dst in [(t1_mask, brain_mask), (wm_t1, wm_mask)]:
            if os.path.exists(src) and os.path.exists(inv_mat):
                _run_safe(
                    f"flirt -in {src} -ref {epi_ref} "
                    f"-applyxfm -init {inv_mat} "
                    f"-out {dst} -interp nearestneighbour", log)
                _run_safe(f"fslmaths {dst} -bin {dst}", log)

    @staticmethod
    def inventory(sub: str, run: str = "LR") -> dict:
        p   = cfg.subject_dirs(sub, run)
        row = dict(subject=sub, run=run)

        def _chk(path, label, mn=None, mx=None):
            if not path or not os.path.exists(path):
                return "MISSING", 0, f"{label} not found"
            try:
                d = nib.load(path).get_fdata(dtype=np.float32)
            except Exception as e:
                return "FLAG", 0, f"load error: {e}"
            n = int((d > 0.5).sum())
            notes = []
            if not all(v in (0.0, 1.0) for v in np.unique(d)):
                notes.append("not binary")
            if mn is not None and n < mn:
                notes.append(f"n={n}<{mn}")
            if mx is not None and n > mx:
                notes.append(f"n={n}>{mx}")
            return ("FLAG" if notes else "OK"), n, " | ".join(notes)

        for label, key, mn, mx in [
            ("wm_raw",         "wm_mask",             None,         None),
            ("wm_ero1_novent", "wm_mask_ero1_novent", _WM_ERO1_MIN, _WM_ERO1_MAX),
            ("wm_ero2_novent", "wm_mask_ero2_novent", None,         None),
        ]:
            path = p.get(key, "")
            st, n, note = _chk(path, label, mn, mx)
            row[f"{label}_status"] = st; row[f"{label}_n"] = n
            if note: row[f"{label}_note"] = note

        for label, key, mn in [
            ("fv_roi",   "fourthvent_mask",  _FV_MIN_VOX),
            ("fv_lower", "fourthvent_lower", 10),
            ("fv_upper", "fourthvent_upper", 10),
        ]:
            path = p.get(key, "")
            st, n, note = _chk(path, label, mn, None)
            row[f"{label}_status"] = st; row[f"{label}_n"] = n
            if note: row[f"{label}_note"] = note

        for label, key in [
            ("bold_mc",    "bold_mc"),
            ("bold_stc",   "bold_stc"),
            ("brain_mask", "brain_mask"),
        ]:
            row[f"{label}_status"] = (
                "OK" if os.path.exists(p.get(key, "")) else "MISSING")

        statuses = [v for k, v in row.items() if k.endswith("_status")]
        row["overall"] = ("MISSING" if "MISSING" in statuses
                          else "FLAG" if "FLAG" in statuses else "OK")
        return row


# step5b is now a compatibility shim — step5.run() does everything.
class step5b:
    """
    Compatibility shim: step5b logic is now part of step5 (substeps 5.6–5.8).
    Calling step5b.run() simply delegates to step5.run() so existing
    --steps 5b invocations continue to work.
    """
    @staticmethod
    def run(sub: str, run: str = "LR", force: bool = False,
            logger=None) -> dict:
        log = logger or print
        log(f"\n  ━━ Step 5b (shim) → delegating to step5 ━━")
        result = step5.run(sub, run, logger=logger, force=force)
        result["step"] = "5b"
        return result

    @staticmethod
    def inventory(sub: str, run: str = "LR") -> dict:
        return step5.inventory(sub, run)


# ══════════════════════════════════════════════════════════════════════════════
# Step 6a — Nuisance regression
# ══════════════════════════════════════════════════════════════════════════════

class step6a:
    """
    Step 6a: Nuisance regression across four pipelines (P0–P3).

    Applies a 3 mm FWHM spatial smooth before regression, then uses
    nilearn.image.clean_img for each pipeline:

        P0 : detrend only               (primary — no confound removal)
        P1 : detrend + 6 motion params
        P2 : detrend + motion + WM mean
        P3 : detrend + motion + WM aCompCor 5 PCs

    Output per pipeline: bold_mc_smooth_clean_P{N}.nii.gz

    WM mask used for P2/P3: wm_mask_ero1_novent (1-vox erosion + lateral/3rd
    ventricle exclusion; produced by step5b).
    ero2_novent (2-vox) was found to be too aggressive for HCP-YA 2×2×2 mm data.
    Falls back to wm_mask_raw if the eroded mask is not yet available.

    Returns
    -------
    dict with keys: subject, run, step (="6a"), status, warnings
    """

    @staticmethod
    def run(sub: str, run: str = "LR",
            pipelines: list = None, logger=None) -> dict:
        p   = cfg.subject_dirs(sub, run)
        log = logger or print
        _ensure_dirs(p)
        report = {
            "subject": sub, "run": run, "step": "6a",
            "status": "ok", "warnings": [],
        }

        log(f"\n  ━━ Step 6a: Nuisance Regression ━━")

        if pipelines is None:
            pipelines = cfg.PIPELINES_TO_RUN

        # Smooth bold_mc → bold_mc_smooth
        mc_in     = p["bold_mc"]
        mc_smooth = os.path.join(p["outdir"], "bold_mc_smooth.nii.gz")
        if not os.path.exists(mc_in):
            log(f"  ✗ bold_mc not found")
            report["status"] = "fail"
            return report

        if not os.path.exists(mc_smooth):
            fwhm_mm = getattr(cfg, "SMOOTH_FWHM_MM", 3.0)
            sigma   = fwhm_mm / 2.3548
            ok = _run_safe(
                f"fslmaths {mc_in} -s {sigma:.4f} {mc_smooth}", log)
            if not ok:
                report["status"] = "fail"; return report
            log(f"  ✓ smoothed ({fwhm_mm} mm FWHM) → bold_mc_smooth")

        # WM mask: prefer ero1_novent (1-vox erosion + ventricle exclusion)
        # ero2_novent (2-vox) is too aggressive for HCP-YA 2×2×2mm data.
        # wm_mask_ero1_novent is produced by step5b.
        wm_mask_path = (p.get("wm_mask_ero1_novent", "")
                        or p.get("wm_mask_corrected", "")
                        or p.get("wm_mask", ""))
        if not os.path.exists(wm_mask_path):
            # Try fallback key
            wm_mask_path = p.get("wm_mask", "")
        if not os.path.exists(wm_mask_path):
            msg = "WM mask not found — P2/P3 will be skipped"
            log(f"  ⚠ {msg}")
            report["warnings"].append(msg)

        # Run each pipeline
        bold_img = nib.load(mc_smooth)

        for pl in pipelines:
            out = os.path.join(p["outdir"],
                               f"bold_mc_smooth_clean_{pl}.nii.gz")
            if os.path.exists(out):
                log(f"  ⏭ {pl} cleaned exists")
                continue

            confounds = step6a._build_confounds(
                p["mc_par"], wm_mask_path, bold_img, pl, log)

            if confounds is None and pl in ("P2", "P3"):
                msg = f"{pl}: WM confounds unavailable — skipping"
                log(f"  ⚠ {msg}")
                report["warnings"].append(msg)
                continue

            log(f"  → {pl}: detrend"
                + (f" + confounds {confounds.shape[1]}" if confounds is not None else ""))
            try:
                cleaned = clean_img(
                    bold_img,
                    t_r            = cfg.TR,
                    detrend        = True,
                    standardize    = None,
                    confounds      = confounds,
                )
                cleaned.to_filename(out)
                log(f"  ✓ {pl} → {os.path.basename(out)}")
            except Exception as e:
                msg = f"{pl} clean_img failed: {e}"
                log(f"  ✗ {msg}")
                report["warnings"].append(msg)
                report["status"] = "fail"

        return report

    @staticmethod
    def _build_confounds(mc_par, wm_mask_path, bold_img, pipeline, log):
        """
        Build confound matrix for the requested pipeline.

        Returns np.ndarray of shape (T, n_confounds), or None for P0.
        """
        # Load motion parameters
        try:
            params = np.loadtxt(mc_par)   # (T, 6): Rx Ry Rz Tx Ty Tz
        except Exception as e:
            log(f"  ⚠ Could not load mc.par: {e}")
            params = None

        if pipeline == "P0":
            return None

        if pipeline == "P1":
            if params is None:
                return None
            return params  # (T, 6)

        # P2 or P3: need WM mask
        if not os.path.exists(wm_mask_path):
            return None

        try:
            masker = NiftiMasker(
                mask_img    = wm_mask_path,
                standardize = False,
                detrend     = False,
            ).fit()
            wm_signals = masker.transform(bold_img)   # (T, n_wm_vox)
        except Exception as e:
            log(f"  ⚠ WM masking failed: {e}")
            return None

        if pipeline == "P2":
            wm_mean = wm_signals.mean(axis=1, keepdims=True)   # (T, 1)
            if params is not None:
                return np.hstack([params, wm_mean])
            return wm_mean

        if pipeline == "P3":
            # aCompCor: 5 PCs from WM signal
            wm_demeaned = wm_signals - wm_signals.mean(axis=0)
            U, s, _ = svd(wm_demeaned, full_matrices=False)
            wm_pcs  = U[:, :5] * s[:5]   # (T, 5)
            if params is not None:
                return np.hstack([params, wm_pcs])
            return wm_pcs

        return None


# ══════════════════════════════════════════════════════════════════════════════
# Step 6b — Bandpass filtering
# ══════════════════════════════════════════════════════════════════════════════

class step6b:
    """
    Step 6b: Bandpass filtering across four frequency bands.

    Input: bold_mc_smooth_clean_P{N}.nii.gz (from step 6a)
    Both gBOLD and CSF signals are extracted from the same unified volume,
    ensuring identical preprocessing for the coupling analysis.

    Four bands (from cfg.BAND_CONFIGS):
        standard_rs : 0.010–0.100 Hz  (broad, primary analysis)
        neurogenic  : 0.020–0.040 Hz
        vasogenic   : 0.060–0.100 Hz

    csf_source parameter
    ─────────────────────
        "mc"  (default) : CSF from bold_mc_smooth_clean (same as gBOLD)
                          → copied from mc_out rather than re-filtered
        "stc"           : CSF from bold_stc_clean (no motion correction)
                          → sensitivity analysis only

    detrend=False because step 6a already detrended the volume.

    Returns
    -------
    dict with keys: subject, run, step (="6b"), status, warnings, files_created
    """

    @staticmethod
    def run(sub: str, run: str = "LR",
            pipelines: list = None,
            csf_source: str = "mc",
            logger=None) -> dict:

        p   = cfg.subject_dirs(sub, run)
        log = logger or print
        _ensure_dirs(p)
        report = {
            "subject": sub, "run": run, "step": "6b",
            "status": "ok", "warnings": [], "files_created": [],
        }

        if pipelines is None:
            pipelines = cfg.PIPELINES_TO_RUN

        log(f"\n  ━━ Step 6b: Bandpass Filter ━━")

        if csf_source not in ("mc", "stc"):
            log(f"  ✗ Invalid csf_source='{csf_source}' — must be 'mc' or 'stc'")
            report["status"] = "fail"
            return report

        # Fast completion check
        _last_band = list(cfg.BAND_CONFIGS.values())[-1][2]
        _last_pl   = pipelines[-1]
        _last_mc   = os.path.join(
            p["outdir"],
            f"bold_mc_smooth_filtered_{_last_pl}_{_last_band}.nii.gz")
        _last_csf  = os.path.join(
            p["outdir"],
            f"bold_csf_filtered_{_last_pl}_{_last_band}_{csf_source}.nii.gz")
        if os.path.exists(_last_mc) and os.path.exists(_last_csf):
            log(f"  ⏭ Step 6b: all filtered volumes exist — skipping")
            return report

        log(f"  csf_source={csf_source}")

        for pl in pipelines:

            # gBOLD input
            mc_clean = os.path.join(
                p["outdir"], f"bold_mc_smooth_clean_{pl}.nii.gz")
            if not os.path.exists(mc_clean):
                log(f"  ✗ bold_mc_smooth_clean_{pl} not found — run step6a first")
                report["warnings"].append(f"{pl}: mc_smooth_clean not found")
                continue

            # CSF input
            if csf_source == "mc":
                csf_clean = mc_clean
            else:
                csf_clean = os.path.join(
                    p["outdir"], f"bold_stc_clean_{pl}.nii.gz")
                if not os.path.exists(csf_clean):
                    log(f"  ✗ bold_stc_clean_{pl} not found")
                    report["warnings"].append(f"{pl}: stc_clean not found")
                    continue

            # Volume count check
            n_vols_mc  = nib.load(mc_clean).shape[3]
            n_vols_csf = (n_vols_mc if csf_source == "mc"
                          else nib.load(csf_clean).shape[3])
            if n_vols_mc != n_vols_csf:
                msg = (f"{pl}: volume count mismatch "
                       f"mc={n_vols_mc} csf={n_vols_csf}")
                log(f"  ⚠ {msg}")
                report["warnings"].append(msg)
            n_vols = min(n_vols_mc, n_vols_csf)
            if n_vols < 50:
                log(f"  ⚠ {pl}: too few volumes ({n_vols}) — skipping")
                report["warnings"].append(f"{pl}: insufficient volumes ({n_vols})")
                continue

            for band_key, (low, high, tag) in cfg.BAND_CONFIGS.items():
                mc_out  = os.path.join(
                    p["outdir"],
                    f"bold_mc_smooth_filtered_{pl}_{tag}.nii.gz")
                csf_out = os.path.join(
                    p["outdir"],
                    f"bold_csf_filtered_{pl}_{tag}_{csf_source}.nii.gz")

                mc_done  = os.path.exists(mc_out)
                csf_done = os.path.exists(csf_out)

                if mc_done and csf_done:
                    log(f"  ⏭ {pl}_{tag}_{csf_source}: both exist")
                    continue

                log(f"  → {pl} [{band_key}] {low:.4f}–{high:.4f} Hz")

                # Filter gBOLD
                if not mc_done:
                    try:
                        clean_img(
                            mc_clean,
                            t_r         = cfg.TR,
                            high_pass   = low,
                            low_pass    = high,
                            detrend     = False,   # step 6a already detrended
                            standardize = None,
                        ).to_filename(mc_out)
                        report["files_created"].append(
                            f"bold_mc_smooth_filtered_{pl}_{tag}.nii.gz")
                        log(f"    ✓ gBOLD: bold_mc_smooth_filtered_{pl}_{tag}")
                    except Exception as e:
                        msg = f"{pl}_{tag} gBOLD filter failed: {e}"
                        log(f"    ✗ {msg}")
                        report["warnings"].append(msg)
                        report["status"] = "fail"
                        continue

                # Filter CSF
                if not csf_done:
                    if csf_source == "mc" and os.path.exists(mc_out):
                        # Identical preprocessing → copy instead of re-filter
                        shutil.copy2(mc_out, csf_out)
                        report["files_created"].append(
                            f"bold_csf_filtered_{pl}_{tag}_{csf_source}.nii.gz")
                        log(f"    ✓ CSF (mc): copied from mc_out")
                    else:
                        try:
                            clean_img(
                                csf_clean,
                                t_r         = cfg.TR,
                                high_pass   = low,
                                low_pass    = high,
                                detrend     = False,
                                standardize = None,
                            ).to_filename(csf_out)
                            report["files_created"].append(
                                f"bold_csf_filtered_{pl}_{tag}_{csf_source}.nii.gz")
                            log(f"    ✓ CSF: bold_csf_filtered_{pl}_{tag}_{csf_source}")
                        except Exception as e:
                            msg = f"{pl}_{tag}_{csf_source} CSF filter failed: {e}"
                            log(f"    ✗ {msg}")
                            report["warnings"].append(msg)
                            report["status"] = "fail"
                            continue

        n_done  = len(report["files_created"])
        n_total = len(pipelines) * len(cfg.BAND_CONFIGS) * 2
        log(f"\n  ✓ Step 6b: {n_done}/{n_total} files created "
            f"[csf_source={csf_source}]")
        return report




# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Time-series extraction
# ══════════════════════════════════════════════════════════════════════════════

class step7:
    """
    Extract 1-D timeseries from bold_mc_smooth_clean_P0.

    Outputs (to ROOT/FINAL/timeseries/{sub}_{run}/):
      gbold_detrended.csv        — global BOLD mean across brain mask
      wm_mean.csv                — WM mean (ero1_novent mask)
      acompcor.csv               — top-5 WM PCs
      motion.csv                 — 6 mcflirt motion parameters
      fd.txt                     — per-volume framewise displacement
      csf_z{i}_detrended.csv     — per-slice manual CSF ROI signals
      csf_combined_detrended.csv — mean across all CSF slices
      fv_lower_detrended.csv     — 4th ventricle lower half
      fv_upper_detrended.csv     — 4th ventricle upper half
      gbold_{band}.csv           — bandpassed gBOLD (4 bands)
      csf_{roi}_{band}.csv       — bandpassed CSF (4 bands × ROI)
      meta.json                  — provenance / done-flag

    CSF source
      csf_source="mc"  (default): uses bold_mc_smooth_clean_P0 as CSF volume
      csf_source="stc" (sensitivity): uses bold_stc (before motion correction)
    """

    N_PCS_WM = 5

    FREQ_BANDS = {
        "standard_rs": (0.01, 0.10),
        "neurogenic":  (0.02, 0.04),
        "vasogenic":   (0.06, 0.10),
        # endogenic excluded — irrelevant for gBOLD-CSF coupling analysis
    }

    @staticmethod
    def run(sub: str, run: str = "LR", csf_source: str = "mc",
            force: bool = False, logger=None) -> dict:
        log = logger or print
        log(f"\n{'━'*60}")
        log(f"  Step 7 — timeseries extraction  [{sub} / {run}]")
        log(f"{'━'*60}")

        p       = cfg.subject_dirs(sub, run)
        out_dir = os.path.join(cfg.ROOT, "FINAL", "timeseries", f"{sub}_{run}")
        os.makedirs(out_dir, exist_ok=True)

        done_flag = os.path.join(out_dir, "meta.json")
        report = dict(
            step="7", subject=sub, run=run, csf_source=csf_source,
            status="unknown", n_vols=0, n_brain=0, n_wm=0,
            n_csf_slices=0, n_fv_lower=0, n_fv_upper=0,
            mean_fd=float("nan"), warnings=[], files_created=[],
        )

        if not force and os.path.exists(done_flag):
            log(f"  skip: meta.json exists (--force to redo)")
            report["status"] = "skip"
            return report

        # ── Load P0 volume ───────────────────────────────────────────────────
        bold_path = p.get("bold_mc_smooth_clean_P0", "")
        if not bold_path or not os.path.exists(bold_path):
            report["status"] = "fail"
            report["warnings"].append(
                f"bold_mc_smooth_clean_P0 not found: {bold_path}  "
                f"→ run steps 1-6a first")
            log(f"  ✗ bold_mc_smooth_clean_P0 not found — aborting step 7")
            return report

        log(f"  loading P0 volume …")
        bold_img  = nib.load(bold_path)
        bold_data = bold_img.get_fdata(dtype=np.float32)   # (x,y,z,t)
        n_vols    = bold_data.shape[3]
        report["n_vols"] = n_vols
        t = np.arange(n_vols)

        if n_vols != cfg.N_VOLUMES:
            msg = f"expected {cfg.N_VOLUMES} volumes, got {n_vols}"
            log(f"  ⚠ {msg}")
            report["warnings"].append(msg)

        # ── Brain mask ───────────────────────────────────────────────────────
        brain_mask_path = p.get("brain_mask", "")
        if not brain_mask_path or not os.path.exists(brain_mask_path):
            report["status"] = "fail"
            report["warnings"].append("brain_mask not found — run step5 first")
            log(f"  ✗ brain_mask not found — aborting")
            return report

        brain_mask = nib.load(brain_mask_path).get_fdata(dtype=np.float32) > 0.5
        n_brain = int(brain_mask.sum())
        report["n_brain"] = n_brain
        log(f"  brain mask: {n_brain:,} voxels")

        # ── Global BOLD ──────────────────────────────────────────────────────
        log(f"  extracting gBOLD …")
        gbold_ts = bold_data[brain_mask, :].mean(axis=0)
        gbold_ts -= np.polyval(np.polyfit(t, gbold_ts, 1), t)   # detrend
        pd.DataFrame({"gbold": gbold_ts}).to_csv(
            os.path.join(out_dir, "gbold_detrended.csv"), index=False)
        report["files_created"].append("gbold_detrended.csv")

        # ── WM signals ───────────────────────────────────────────────────────
        wm_mask_path = (p.get("wm_mask_ero1_novent")
                        or p.get("wm_mask_corrected")
                        or p.get("wm_mask", ""))
        if wm_mask_path and os.path.exists(wm_mask_path):
            wm_mask = nib.load(wm_mask_path).get_fdata(dtype=np.float32) > 0.5
            n_wm    = int(wm_mask.sum())
            report["n_wm"] = n_wm
            log(f"  WM mask (ero1_novent): {n_wm:,} voxels")
            wm_ts   = bold_data[wm_mask, :]
            wm_mean = wm_ts.mean(axis=0)
            pd.DataFrame({"wm_mean": wm_mean}).to_csv(
                os.path.join(out_dir, "wm_mean.csv"), index=False)
            report["files_created"].append("wm_mean.csv")

            # aCompCor: top-5 PCs
            wm_demeaned = wm_ts - wm_ts.mean(axis=1, keepdims=True)
            try:
                _, _, Vt = np.linalg.svd(wm_demeaned, full_matrices=False)
                n_pcs     = min(step7.N_PCS_WM, Vt.shape[0])
                acompcor  = Vt[:n_pcs, :].T          # (t, n_pcs)
            except np.linalg.LinAlgError as e:
                msg = f"SVD for aCompCor failed ({e}) — using WM mean only"
                log(f"  ⚠ {msg}")
                report["warnings"].append(msg)
                acompcor = wm_mean.reshape(-1, 1)
            pd.DataFrame(acompcor,
                         columns=[f"PC{i+1}" for i in range(acompcor.shape[1])]
                         ).to_csv(os.path.join(out_dir, "acompcor.csv"),
                                  index=False)
            report["files_created"].append("acompcor.csv")
        else:
            msg = f"WM mask not found ({wm_mask_path}) — skipping WM/aCompCor"
            log(f"  ⚠ {msg}")
            report["warnings"].append(msg)

        # ── Motion parameters ────────────────────────────────────────────────
        mc_par_path = p.get("mc_par", "")
        if mc_par_path and os.path.exists(mc_par_path):
            motion = np.loadtxt(mc_par_path)
            if motion.ndim == 1:
                motion = motion.reshape(1, -1)
            pd.DataFrame(motion,
                         columns=["rot_x","rot_y","rot_z","tx","ty","tz"]
                         ).to_csv(os.path.join(out_dir, "motion.csv"),
                                  index=False)
            report["files_created"].append("motion.csv")

            # Framewise displacement (Power 2012 formula)
            rot_rad  = motion[:, :3]
            trans_mm = motion[:, 3:]
            rot_mm   = np.abs(np.diff(rot_rad,  axis=0)) * 50  # ~50 mm radius
            tr_mm    = np.abs(np.diff(trans_mm, axis=0))
            fd_vals  = np.concatenate([[0],
                                       rot_mm.sum(axis=1) + tr_mm.sum(axis=1)])
            report["mean_fd"] = float(np.nanmean(fd_vals))
            log(f"  mean FD = {report['mean_fd']:.3f} mm")
            np.savetxt(os.path.join(out_dir, "fd.txt"), fd_vals, fmt="%.6f")
            report["files_created"].append("fd.txt")
        else:
            msg = f"mc_par not found ({mc_par_path}) — skipping motion"
            log(f"  ⚠ {msg}")
            report["warnings"].append(msg)

        # ── CSF volume selection ─────────────────────────────────────────────
        if csf_source == "mc":
            csf_vol = bold_data
            log(f"  CSF source: mc (P0 volume)")
        elif csf_source == "stc":
            stc_path = p.get("bold_stc", "")
            if not (stc_path and os.path.exists(stc_path)):
                msg = f"bold_stc not found for csf_source='stc': {stc_path}"
                report["status"] = "fail"
                report["warnings"].append(msg)
                log(f"  ✗ {msg}")
                return report
            log(f"  CSF source: stc (before motion correction)")
            csf_vol = nib.load(stc_path).get_fdata(dtype=np.float32)
        else:
            raise ValueError(f"Unknown csf_source: {csf_source!r} "
                             "(must be 'mc' or 'stc')")

        # ── Manual CSF masks (per-slice) ─────────────────────────────────────
        try:
            csf_info = cfg.load_csf_masks(sub, log)
        except Exception as e:
            csf_info = {"by_z": {}, "z_sorted": []}
            msg = f"load_csf_masks failed ({e}) — skipping manual CSF"
            log(f"  ⚠ {msg}")
            report["warnings"].append(msg)

        z_sorted    = csf_info.get("z_sorted", [])
        csf_by_z    = csf_info.get("by_z", {})
        csf_ts_dict = {}   # z → (t,) detrended array

        for zi, z in enumerate(z_sorted):
            slice_mask = csf_by_z[z]
            if slice_mask.ndim == 2:
                full_mask = np.zeros(csf_vol.shape[:3], dtype=bool)
                full_mask[:, :, z] = slice_mask > 0.5
            else:
                full_mask = slice_mask > 0.5
            if full_mask.sum() == 0:
                log(f"  ⚠ CSF z={z}: empty mask — skipped")
                continue
            ts = csf_vol[full_mask, :].mean(axis=0)
            ts -= np.polyval(np.polyfit(t, ts, 1), t)
            csf_ts_dict[z] = ts
            fname = f"csf_z{zi}_detrended.csv"
            pd.DataFrame({"csf": ts}).to_csv(
                os.path.join(out_dir, fname), index=False)
            report["files_created"].append(fname)

        report["n_csf_slices"] = len(csf_ts_dict)
        log(f"  manual CSF slices: {len(csf_ts_dict)}")

        # Combined CSF mean
        csf_combined = None
        if csf_ts_dict:
            csf_combined = np.stack(list(csf_ts_dict.values()),
                                    axis=0).mean(axis=0)
            pd.DataFrame({"csf": csf_combined}).to_csv(
                os.path.join(out_dir, "csf_combined_detrended.csv"), index=False)
            report["files_created"].append("csf_combined_detrended.csv")

        # ── 4th ventricle lower / upper ──────────────────────────────────────
        for label, key, fname in [
            ("fv_lower", "fourthvent_lower", "fv_lower_detrended.csv"),
            ("fv_upper", "fourthvent_upper", "fv_upper_detrended.csv"),
        ]:
            fp = p.get(key, "")
            if fp and os.path.exists(fp):
                mask = nib.load(fp).get_fdata(dtype=np.float32) > 0.5
                n_v  = int(mask.sum())
                report[f"n_{label}"] = n_v
                if n_v > 0:
                    ts = csf_vol[mask, :].mean(axis=0)
                    ts -= np.polyval(np.polyfit(t, ts, 1), t)
                    pd.DataFrame({"csf": ts}).to_csv(
                        os.path.join(out_dir, fname), index=False)
                    report["files_created"].append(fname)
                    log(f"  {label}: {n_v:,} voxels")
                else:
                    log(f"  ⚠ {label} mask is empty")
            else:
                msg = f"{label} mask not found ({fp})"
                log(f"  ⚠ {msg}")
                report["warnings"].append(msg)

        # ── Bandpass ─────────────────────────────────────────────────────────
        log(f"  bandpassing {len(step7.FREQ_BANDS)} bands …")

        def _bp(ts_1d):
            """Bandpass a 1-D array by wrapping nilearn clean on a fake 4-D."""
            try:
                fake = nib.Nifti1Image(
                    ts_1d.reshape(1, 1, 1, -1).astype(np.float32),
                    np.eye(4))
                return clean_img(fake, t_r=cfg.TR,
                                 high_pass=lo, low_pass=hi,
                                 detrend=False,
                                 standardize=False).get_fdata().ravel()
            except Exception:
                # scipy fallback
                nyq = 0.5 / cfg.TR
                b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
                return filtfilt(b, a, ts_1d)

        for band_name, (lo, hi) in step7.FREQ_BANDS.items():
            # gBOLD
            gbold_bp = _bp(gbold_ts)
            pd.DataFrame({band_name: gbold_bp}).to_csv(
                os.path.join(out_dir, f"gbold_{band_name}.csv"), index=False)
            report["files_created"].append(f"gbold_{band_name}.csv")

            # Per-slice CSF
            for zi, (z, ts) in enumerate(csf_ts_dict.items()):
                ts_bp = _bp(ts)
                pd.DataFrame({band_name: ts_bp}).to_csv(
                    os.path.join(out_dir, f"csf_z{zi}_{band_name}.csv"),
                    index=False)
                report["files_created"].append(f"csf_z{zi}_{band_name}.csv")

            # Combined CSF
            if csf_combined is not None:
                ts_bp = _bp(csf_combined)
                pd.DataFrame({band_name: ts_bp}).to_csv(
                    os.path.join(out_dir, f"csf_combined_{band_name}.csv"),
                    index=False)
                report["files_created"].append(f"csf_combined_{band_name}.csv")

            # 4th ventricle halves
            for roi in ["fv_lower", "fv_upper"]:
                fp = os.path.join(out_dir, f"{roi}_detrended.csv")
                if os.path.exists(fp):
                    ts = pd.read_csv(fp)["csf"].values
                    ts_bp = _bp(ts)
                    pd.DataFrame({band_name: ts_bp}).to_csv(
                        os.path.join(out_dir, f"{roi}_{band_name}.csv"),
                        index=False)
                    report["files_created"].append(f"{roi}_{band_name}.csv")

        # ── Save meta / done-flag ────────────────────────────────────────────
        meta = dict(
            subject      = sub,
            run          = run,
            csf_source   = csf_source,
            bold_source  = bold_path,
            n_vols       = n_vols,
            n_brain      = n_brain,
            n_wm         = report.get("n_wm", 0),
            n_csf_slices = report["n_csf_slices"],
            n_fv_lower   = report.get("n_fv_lower", 0),
            n_fv_upper   = report.get("n_fv_upper", 0),
            mean_fd      = report.get("mean_fd", float("nan")),
            freq_bands   = step7.FREQ_BANDS,
            pipeline     = "P0",
            timestamp    = datetime.now().isoformat(),
            version      = "1.0",
            warnings     = report["warnings"],
        )
        with open(done_flag, "w") as f:
            json.dump(meta, f, indent=2)
        report["files_created"].append("meta.json")

        n_files = len(report["files_created"])
        log(f"\n  ✓ Step 7 done: {n_files} files → {out_dir}")
        report["status"] = "done"
        return report

    @staticmethod
    def inventory(sub: str, run: str = "LR") -> dict:
        out_dir   = os.path.join(cfg.ROOT, "FINAL", "timeseries", f"{sub}_{run}")
        done_flag = os.path.join(out_dir, "meta.json")
        return {
            "done":      os.path.exists(done_flag),
            "out_dir":   out_dir,
            "meta_json": done_flag,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Top-level convenience wrappers
# ══════════════════════════════════════════════════════════════════════════════

def run_step1(sub, run="LR", logger=None, force=False):
    return step1.run(sub, run, logger=logger, force=force)

def run_step2(sub, run="LR", logger=None):
    return step2.run(sub, run, logger=logger)

def run_step3(sub, run="LR", logger=None):
    return step3.run(sub, run, logger=logger)

def run_step4(sub, run="LR", logger=None):
    return step4.run(sub, run, logger=logger)

def run_step4b(sub, run="LR", logger=None):
    return step4b.run(sub, run, logger=logger)

def run_step5(sub, run="LR", logger=None):
    return step5.run(sub, run, logger=logger)

def run_step5b(sub, run="LR", force=False, logger=None):
    return step5b.run(sub, run, force=force, logger=logger)

def run_step6a(sub, run="LR", pipelines=None, logger=None):
    return step6a.run(sub, run, pipelines=pipelines, logger=logger)

def run_step6b(sub, run="LR", pipelines=None, csf_source="mc", logger=None):
    return step6b.run(sub, run, pipelines=pipelines,
                      csf_source=csf_source, logger=logger)

def run_step7(sub, run="LR", csf_source="mc", force=False, logger=None):
    return step7.run(sub, run, csf_source=csf_source,
                     force=force, logger=logger)
