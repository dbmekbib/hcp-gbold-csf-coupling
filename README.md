# README — HCP-YA gBOLD–CSF Coupling Pipeline


This repository contains Python code for investigating the measurement properties of global blood-oxygen-level-dependent (gBOLD) and cerebrospinal fluid (CSF) coupling using resting-state functional MRI from the Human Connectome Project Young Adult (HCP-YA) S1200 dataset.

The pipeline includes preprocessing, signal extraction, frequency-specific coupling estimation, temporal stability and reproducibility analyses, surrogate-based validation, nuisance-regression sensitivity analyses, statistical modeling, and figure generation.

The study examines three frequency bands:

    Broad: 0.01–0.10 Hz
    Neurogenic: 0.02–0.04 Hz
    Vasogenic: 0.06–0.10 Hz

The primary analyses include 104 participants, with an additional 25-participant subsample used to evaluate reliability across independent resting-state acquisitions.
Only analysis code and documentation are provided. HCP-YA imaging data are not redistributed.

## Repository Structure

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

## Data Access and Configuration
The analyses use the HCP-YA S1200 Data Release, available through the Human Connectome Project, subject to applicable data-access requirements.
To reproduce the analyses, obtain the HCP-YA dataset and update the ROOT directory in config_hcp.py to match your local data location.

Key analysis parameters include:

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

## Preprocessing Pipeline

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
| 6a | Spatial smoothing (4 mm FWHM) + detrending |
| 7 | Extraction and frequency-specific filtering of gBOLD and CSF time series |

The primary preprocessing approach (P0) applies spatial smoothing and detrending without nuisance regression. Additional nuisance-regression strategies are evaluated separately to assess sensitivity to motion and WM-derived signals.

---

## Running Preprocessing Pipeline

python code/run_pipeline_hcp.py \
    --steps 1 2 3 4 4b 5 5b 6a \
    --subjects_csv coupling_subjects.csv \
    --run LR RL LR2 RL2 \
    --workers 4

**Extract timeseries (step 7):**

python code/run_pipeline_hcp.py \
    --steps 7 \
    --subjects_csv coupling_subjects.csv \
    --run LR RL LR2 RL2 \
    --csf_source mc \
    --workers 4

---
These commands assume the scripts are located in code/. If the scripts are in the repository root, remove the code/ prefix.

## gBOLD–CSF Coupling Analysis
Coupling is estimated using cross-correlation between the global gray matter BOLD signal and CSF signals extracted from six anatomical sampling locations.
The analyses evaluate:

    1. Coupling magnitude and peak lag across CSF locations and frequency bands
    2. Temporal variation across four consecutive 3.6-minute windows
    3. Split-half reliability across two 7.2-minute segments
    4. Independent-acquisition reliability within and across sessions
    5. Sensitivity to motion and WM-derived nuisance regression

** Primary Cohort:**
python code/analyse_coupling.py \
    --ts_dir  results/timeseries \
    --out_dir results/coupling_g1

** Reliability Subsample:**
python code/analyse_coupling_25sub.py \
    --ts_dir  results/timeseries \
    --out_dir results/coupling_g2


## Surrogate Analysis

Amplitude-Adjusted Fourier Transform (AAFT) surrogates are used to evaluate whether observed coupling exceeds expectations under a surrogate null model.

For each participant and frequency band, 1,000 surrogate time series are generated from the filtered gBOLD signal while retaining the original CSF signal. The same cross-correlation and peak-selection procedures are applied to the surrogate data.

Surrogate-adjusted coupling is calculated by subtracting the median surrogate coupling magnitude from the observed coupling magnitude.

python code/surrogate_null.py \
  --ts_dir results/timeseries \
  --out_dir results/surrogate_analysis
  

## Statistical Analysis
Linear mixed-effects models evaluate coupling differences across CSF sampling locations, frequency bands, and temporal windows.

**ROI × Band model** (`Δ|r| ~ ROI × Band + (1|subject)`):

python code/run_lme.py roi_band \
    --coupling_csv  results/coupling_g1/stage3_coupling.csv \
    --surrogate_csv results/surrogate_analysis/surrogate_summary.csv \
    --out           results/statistics/roi_band_delta_r

**Window × Band model** (`Δ|r| ~ Window × Band + (1|subject)`):

python code/run_lme.py window_band \
    --coupling_csv  results/coupling_g1/stage3_timewindow.csv \
    --surrogate_csv results/surrogate_analysis/surrogate_windows.csv \
    --out           results/statistics/z0_window_band_delta_r


## Figure Generation

** Primary cohort:**

python code/figures_g1.py \
    --root results/coupling_g1 \
    --out  results/figures/g1

** Reliability subsample:**

python code/figures_g2.py \
    --root results/coupling_g2 \
    --out  results/figures/g2


## Data Code Availability
The HCP-YA imaging data are available through the Human Connectome Project. Imaging data and participant-level datasets are not redistributed through this repository.

The analysis scripts are available at:



## Citation

If you use this code, please cite the associated manuscript:

> Destaw Mekbib, Ph.D., Xi Zhu, Ph.D., Spatial and Spectral Dependence of gBOLD–CSF Coupling in Resting-State fMRI: Reproducibility, Temporal Stability, and Nuisance Sensitivity. (manuscript under review)

Full citation details will be updated upon publication.

---

## Contact
Destaw Mekbib, Ph.D.
University of Texas at Arlington
Email: destaw.mekbib@uta.edu


