# Reproducibility Package — gBOLD–CSF Coupling (HCP-YA)

This package contains all code and preprocessed timeseries data needed to
reproduce the figures in:

> Paper title:  Spatial and spectral dependence of gBOLD–CSF coupling in resting-state fMRI: reproducibility, temporal stability, and nuisance sensitivity
> Authors: Destaw Mekbib (Ph.D.) Xi Zhu (Ph.D.)  
> 

---

## Contents

```
hcp_gbold_csf_repro/
├── README.md                         ← this file
├── code/
│   ├── config_hcp.py                 ← shared constants (TR, bands, lag windows)
│   ├── utils_hcp.py                  ← cross-correlation utilities
│   ├── fig_style.py                  ← figure style constants
│   ├── analyse_coupling.py           ← Step 1a: Group 1 coupling analysis
│   ├── analyse_coupling_25sub.py     ← Step 1b: Group 2 coupling analysis
│   ├── surrogate_null.py             ← Step 2: AAFT surrogate null distributions
│   ├── figures_g1.py                 ← Step 3: Group 1 figures (Figs 1–5, S3, S5–S7, S9)
│   └── figures_g2.py                 ← Step 4: Group 2 figures (Figs S4, S8)
├── subjects/
│   ├── coupling_subjects_104.csv     ← Group 1 subject list (104 subjects)
│   └── coupling_subjects_25.csv      ← Group 2 subject list (25 subjects)
└── data/
    ├── timeseries_primary/           ← Group 1 preprocessed timeseries
    │   └── {subject}_{run}/          ← one folder per subject × run
    │       ├── gbold_detrended.csv
    │       ├── gbold_{band}.csv      ← band = broad | neurogenic | endogenic | vasogenic
    │       ├── csf_{roi}_{band}.csv  ← roi = z0 | z1 | z2 | z3 | fv_lower | fv_upper
    │       ├── csf_{roi}_detrended.csv
    │       ├── wm_mean.csv
    │       ├── acompcor.csv
    │       └── motion.csv
    └── timeseries_subsample/         ← Group 2 preprocessed timeseries (same structure)
```

---

## Requirements

Python 3.8 or later. Install dependencies with:

```bash
pip install numpy scipy pandas scikit-learn nilearn matplotlib
```

> **Note**: `nilearn` is used only for bandpass filtering in the timeseries
> extraction stage (upstream preprocessing). The postprocessing scripts in
> this package (`analyse_coupling.py`, `figures_g1.py`, `figures_g2.py`)
> do **not** require `nibabel` — all inputs are pre-extracted CSV timeseries.

Tested with:
- Python 3.9
- numpy 1.24, scipy 1.10, pandas 1.5, scikit-learn 1.2, nilearn 0.10, matplotlib 3.7

---

## Workflow

Run the four steps below **in order** from the package root directory
(`hcp_gbold_csf_repro/`).

### Step 1a — Group 1 coupling analysis (104 subjects)

Computes 5-stage gBOLD–CSF coupling for the primary cohort:
- Stage 1: nuisance variance (unfiltered R²)
- Stage 2: frequency-specific nuisance variance
- Stage 3: coupling strength, PSD, split-half reproducibility, ICC(2,1)
- Stages 4–5: pipeline sensitivity (motion and WM regression)

```bash
python code/analyse_coupling.py \
    --subjects_csv subjects/coupling_subjects_104.csv \
    --ts_dir       data/timeseries_primary \
    --out          results/coupling_g1
```

**Outputs** → `results/coupling_g1/`

| File | Contents |
|------|----------|
| `stage1_nuisance_r2.csv` | Per-subject unfiltered nuisance R² |
| `stage2_freq_incremental_r2.csv` | Per-subject frequency-specific R² |
| `stage3_coupling.csv` | Per-subject coupling strength and peak lag |
| `stage3_splithalf.csv` | Split-half coupling (half-1 / half-2) |
| `stage3_psd.csv` | Per-subject band power |
| `stage3_timewindow.csv` | Windowed coupling (4 × ~297 volumes) |
| `stage3_icc.csv` | ICC(2,1) split-half reliability |
| `stage45_matched_ratios.csv` | Per-subject P0–P3 coupling ratios |

---

### Step 1b — Group 2 coupling analysis (25 subjects, 4 runs)

Same 5-stage analysis for the subsample cohort across four scan runs.

```bash
python code/analyse_coupling_26sub.py \
    --subjects_csv subjects/coupling_subjects_25.csv \
    --ts_dir       data/timeseries_subsample \
    --runs LR LR2 RL RL2 --suffix 26s \
    --out          results/coupling_g2
```

**Outputs** → `results/coupling_g2/` (same files as above, suffixed `_26s_{run}`)

---

### Step 2 — AAFT surrogate null distributions

Generates 1000 amplitude-adjusted Fourier transform (AAFT) surrogates per
subject × band × ROI to build empirical null distributions of coupling strength.

```bash
python code/surrogate_null.py \
    --subjects_csv subjects/coupling_subjects_104.csv \
    --ts_dir       data/timeseries_primary \
    --out          results/surrogate
```

**Outputs** → `results/surrogate/`

| File | Contents |
|------|----------|
| `surrogate_summary.csv` | Per-subject z-score and p-value vs null |
| `surrogate_windows.csv` | Same, per time window (4 windows) |
| `surrogate_null_distributions/` | Per-subject per-band |r| arrays (.npy) |

> ⏱ **Runtime**: approximately 2–4 hours for 104 subjects × 1000 surrogates.
> To run a quick test with 100 surrogates on 1 subject:
> ```bash
> python code/surrogate_null.py \
>     --subjects_csv subjects/coupling_subjects_104.csv \
>     --subjects 100206 --n_surr 100 \
>     --ts_dir data/timeseries_primary \
>     --out results/surrogate_test
> ```

---

### Steps 3 & 4 — Generate all figures

```bash
# Group 1 figures: Figs 1, 2, 3, 4, 5, S3, S5, S6, S7, S9
python code/figures_g1.py --fig all \
    --root . \
    --ts_root data/timeseries_primary \
    --summary results/surrogate/surrogate_summary.csv

# Group 2 figures: Figs S4, S8
python code/figures_g2.py --fig all --root .
```

**Outputs** → `results/figures/` (PDF + PNG at 600 DPI)

---

## Figure index

| Figure | Script | Key inputs |
|--------|--------|-----------|
| Fig 1 — Spatial gradient | `figures_g1.py` | `stage3_coupling.csv`, timeseries |
| Fig 2 — Within-run reliability | `figures_g1.py` | `stage3_splithalf.csv`, `stage3_icc.csv` |
| Fig 3 — Time-window stability | `figures_g1.py` | `stage3_timewindow.csv`, timeseries |
| Fig 4 — Nuisance variance | `figures_g1.py` | `stage2_freq_incremental_r2.csv` |
| Fig 5 — Pipeline sensitivity | `figures_g1.py` | `stage45_matched_ratios.csv` |
| Fig S3 — AAFT PSD + amplitude | `figures_g1.py` | timeseries, `surrogate_summary.csv` |
| Fig S5 — AAFT coupling | `figures_g1.py` | `stage3_coupling.csv`, `surrogate_summary.csv` |
| Fig S6 — CSF-sampling ICC | `figures_g1.py` | `stage3_icc.csv` |
| Fig S7 — Lag scatter | `figures_g1.py` | `stage3_coupling.csv`, timeseries |
| Fig S9 — ROI nuisance sensitivity | `figures_g1.py` | `stage2_freq_incremental_r2.csv` |
| Fig S4 — Window surrogate | `figures_g2.py` | `surrogate_windows.csv` |
| Fig S8 — ICC hierarchy | `figures_g2.py` | `results/coupling_g2/` |

---

## Single-subject validation

To verify the pipeline runs correctly before the full cohort run:

```bash
python code/analyse_coupling.py \
    --subjects_csv subjects/coupling_subjects_104.csv \
    --subjects 100206 --stages 1 2 3 \
    --ts_dir data/timeseries_primary \
    --out results/coupling_g1

python code/surrogate_null.py \
    --subjects_csv subjects/coupling_subjects_104.csv \
    --subjects 100206 \
    --ts_dir data/timeseries_primary \
    --out results/surrogate
```

Expected output for subject 100206:
- `stage3_coupling.csv`: 24 rows (6 ROIs × 4 bands)
- `surrogate_summary.csv`: z > 3 for z0–z3 in broad and neurogenic bands

---

## Data provenance

Timeseries were extracted from HCP Young Adult (HCP-YA) resting-state fMRI
(TR = 0.72 s, 1190 volumes, 3T, multiband factor 8).

- **Group 1**: 104 subjects, LR phase-encoding run
- **Group 2**: 25 subjects, LR + LR2 + RL + RL2 runs

Preprocessing: motion correction, spatial smoothing (4 mm),
detrending. CSF signals extracted from 6 ROIs along the paravascular spaces at the base of cerebellum
(z0–z3, fv-lower, fv-upper). gBOLD = gray-matter mean BOLD signal.
Bandpass filtering via `nilearn.signal.clean` (Butterworth).

---

## Contact

Corresponding author: Destaw Mekbib (Ph.D.)  
Email:  destaw.mekbib@uta.edu; destawb37@gmail.com
The University of Texas at Arlington
