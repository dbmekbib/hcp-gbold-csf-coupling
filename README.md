# README — HCP-YA gBOLD–CSF Coupling Pipeline

This repository contains the analysis code for a resting-state fMRI study of global BOLD (gBOLD) and CSF signal coupling in the Human Connectome Project Young Adult (HCP-YA) dataset. The pipeline covers all stages from raw HCP preprocessing through bandpass-filtered timeseries extraction, cross-correlation coupling analysis, surrogate null testing, linear mixed-effects (LME) modelling, and figure generation. Only code is submitted; the underlying HCP-YA imaging data are not included.

---

## Repository Structure

All scripts reside in `code/`.

| File | Purpose |
|---|---|
| `config_hcp.py` | Single configuration source — paths, constants, helper functions |
| `pipeline_steps.py` | Preprocessing steps 1–7 (class namespaces); step 7 = timeseries extraction |
| `run_pipeline_hcp.py` | Runs any subset of steps 1–7 via command-line flags |
| `analyse_coupling.py` | Five-stage gBOLD–CSF coupling analysis, Group 1 (104 subjects) |
| `analyse_coupling_25sub.py` | Same pipeline applied to Group 2 (25-subject replication) |
| `surrogate_null.py` | AAFT surrogate generation and null-distribution summary |
| `run_lme.py` | LME models: ROI × Band and Window × Band (two subcommands) |
| `figures_g1.py` | All figures for Group 1 results |
| `figures_g2.py` | All figures for Group 2 / replication results |

---

## Software Requirements

Python ≥ 3.9 is required. The following packages must be installed:

| Package | Tested version | Notes |
|---|---|---|
| `numpy` | ≥ 1.24 | Core numerics |
| `scipy` | ≥ 1.10 | Filtering, stats, surrogates |
| `nibabel` | ≥ 5.0 | NIfTI I/O |
| `nilearn` | ≥ 0.10 | `clean_img`, `NiftiMasker` |
| `pandas` | ≥ 2.0 | CSV I/O, LME tables |
| `matplotlib` | ≥ 3.7 | Figure rendering |
| `statsmodels` | ≥ 0.14 | LME (`MixedLM`), GEE |
| `pingouin` | ≥ 0.5 | ICC computation |

Install with:

```
pip install numpy scipy nibabel nilearn pandas matplotlib statsmodels pingouin
```

---

## Configuration

`config_hcp.py` is the single source of truth for all study constants. Key parameters:

| Constant | Value | Description |
|---|---|---|
| `TR` | `0.72` s | Repetition time |
| `N_VOLUMES` | `1190` | Volumes after dummy removal (1200 − 10) |
| `RUNS` | `['LR', 'RL', 'LR2', 'RL2']` | Four HCP-YA rest runs |
| `FREQ_BANDS` | broad (0.01–0.10 Hz), neurogenic (0.02–0.04 Hz), vasogenic (0.06–0.10 Hz) | Three analysis bands |
| `BAND_LAG_SEC` | broad: 15 s, neurogenic: 20 s, vasogenic: 10 s | Maximum cross-correlation lag |
| `N_SURROGATES` | `1000` | AAFT surrogates per subject/band/ROI |

Edit the `ROOT` path in `config_hcp.py` to point to your local HCP-YA directory before running any step.

---

## Pipeline Overview

The preprocessing pipeline is implemented as numbered steps inside `pipeline_steps.py` and orchestrated by `run_pipeline_hcp.py`.

| Step | Description |
|---|---|
| 1 | Dummy-volume removal (first 10 volumes discarded) |
| 2 | Bias-field correction + topup distortion correction |
| 3 | Slice-timing correction (STC) |
| 4 | Motion correction (MC) |
| 4b | QC variance check |
| 5 | EPI-to-T1 registration (bbregister, boundary-based) + tissue mask extraction (WM, GM, CSF, 4th ventricle) in functional space |
| 5b | WM mask refinement (erosion + ventricle exclusion) |
| 6a | Spatial smoothing (4 mm FWHM) + detrending of the 4D volume. Produces `bold_mc_smooth_clean_P0` (primary) and P1–P3 variants (for nuisance sensitivity analysis) |
| 7 | Extract 1-D signals (gBOLD, CSF, WM, motion) from the cleaned volume; detrend signals; bandpass filter per frequency band at the timeseries level |

The primary nuisance pipeline is **P0** (smooth + detrend only). Its output (`bold_mc_smooth_clean_P0`) is the source volume for step 7. Bandpass filtering and all further signal processing are performed in 1-D within step 7, not at the volume level. P1–P3 regression for nuisance sensitivity analysis is applied on-the-fly inside `analyse_coupling.py` (Stages 4–5) from the detrended unfiltered 1-D signals.

---

## Running the Full Preprocessing Pipeline

`run_pipeline_hcp.py` accepts `--steps` (space-separated list), `--subjects_csv`, `--run`, `--workers`, and `--csf_source`.

**Run all preprocessing steps (1–6a) for a subject list:**

```bash
python code/run_pipeline_hcp.py \
    --steps 1 2 3 4 4b 5 5b 6a \
    --subjects_csv coupling_subjects_109.csv \
    --run LR RL LR2 RL2 \
    --workers 4
```

**Extract timeseries (step 7):**

```bash
python code/run_pipeline_hcp.py \
    --steps 7 \
    --subjects_csv coupling_subjects_109.csv \
    --run LR RL LR2 RL2 \
    --csf_source mc \
    --workers 4
```

`--csf_source mc` (default) extracts CSF from the motion-corrected P0 volume, matching the gBOLD source.

**Run a single step for one subject (debugging):**

```bash
python code/run_pipeline_hcp.py \
    --steps 6a \
    --subjects_csv single_subject.csv \
    --run LR
```

---

## Coupling Analysis

Coupling is computed in five sequential stages (Stages 1–5): windowed cross-correlation between gBOLD and CSF timeseries, lag-peak extraction, surrogate normalisation, ROI aggregation, and time-window analysis.

**Group 1 (104 subjects, `coupling_g1/`):**

```bash
python code/analyse_coupling.py \
    --ts_dir  results/timeseries \
    --out_dir results/coupling_g1
```

**Group 2 (25-subject replication, `coupling_g2/`):**

```bash
python code/analyse_coupling_25sub.py \
    --ts_dir  results/timeseries \
    --out_dir results/coupling_g2
```

Both scripts write per-stage CSV files consumed by the LME and figure scripts.

---

## Surrogate Null Distribution

`surrogate_null.py` generates 1000 Amplitude-Adjusted Fourier Transform (AAFT) surrogates per subject/band/ROI to form the null distribution. The null 50th percentile (median) is subtracted from observed |r| values to yield Δ|r|, the primary coupling metric.

```bash
python code/surrogate_null.py \
    --ts_dir  results/timeseries \
    --out_dir results/surrogate_analysis
```

Outputs:
- `surrogate_summary.csv` — per-subject/band/ROI null summaries (used by `run_lme.py roi_band`)
- `surrogate_windows.csv` — per-window null summaries (used by `run_lme.py window_band`)

---

## Statistical Analysis (LME)

`run_lme.py` fits linear mixed-effects models with subject as a random intercept. It provides two subcommands.

**ROI × Band model** (`Δ|r| ~ ROI × Band + (1|subject)`):

```bash
python code/run_lme.py roi_band \
    --coupling_csv  results/coupling_g1/stage3_coupling.csv \
    --surrogate_csv results/surrogate_analysis/surrogate_summary.csv \
    --out           results/statistics/roi_band_delta_r
```

**Window × Band model** (`Δ|r| ~ Window × Band + (1|subject)`):

```bash
python code/run_lme.py window_band \
    --coupling_csv  results/coupling_g1/stage3_timewindow.csv \
    --surrogate_csv results/surrogate_analysis/surrogate_windows.csv \
    --out           results/statistics/z0_window_band_delta_r
```

Both subcommands apply FDR correction (Benjamini–Hochberg) to pairwise contrasts. A GEE sensitivity check (exchangeable working correlation) is also run for the Window × Band model.

---

## Figure Generation

**Group 1 figures:**

```bash
python code/figures_g1.py \
    --root results/coupling_g1 \
    --out  results/figures/g1
```

**Group 2 / replication figures:**

```bash
python code/figures_g2.py \
    --root results/coupling_g2 \
    --out  results/figures/g2
```

Both scripts read the stage CSV files produced by `analyse_coupling.py` / `analyse_coupling_25sub.py` and write publication-quality PDF/PNG figures to the specified output directory. Figure style and band ordering (`broad`, `neurogenic`, `vasogenic`) are controlled by `fig_style.py` (imported automatically).

---

## Data Availability Note

The HCP-YA resting-state fMRI data used in this study are available from the Human Connectome Project (https://www.humanconnectome.org) under a data-use agreement. **The preprocessed data are not included in this code submission.** Only the analysis code is provided. Researchers wishing to reproduce these results should obtain access to the HCP-YA dataset independently and configure the `ROOT` path in `config_hcp.py` accordingly.

Group assignments (subject IDs for Group 1 and Group 2) are provided in the accompanying CSV files (`coupling_subjects_109.csv` and the 25-subject list).

---

## Citation

If you use this code in your research, please cite the associated manuscript:

> Destaw Mekbib, Ph.D., Xi Zhu, Ph.D., Spatial and Spectral Dependence of gBOLD–CSF Coupling in Resting-State fMRI: Reproducibility, Temporal Stability, and Nuisance Sensitivity. (manuscript under review)

Full citation details including journal, volume, and DOI will be updated upon publication.

---

## Contact

For questions about the analysis pipeline or code, please contact:

Destaw Bayabil Mekbib
Email: destawb37@gmail.com; destaw.mekbib@uta.edu
University of Texas at Arlington

