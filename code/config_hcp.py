"""
config_hcp.py
════════════════════════════════════════════════════════════════════════
Shared constants for the HCP-YA gBOLD-CSF coupling postprocessing package.

This is a portable version for reproducibility. All absolute laboratory
paths and upstream NIfTI processing constants have been removed.
Only the constants required by the coupling analysis and figure scripts
are retained.
"""

# ── Acquisition ──────────────────────────────────────────────────────
TR = 0.72          # HCP-YA repetition time (seconds)

# ── Frequency band definitions ────────────────────────────────────────
# key → (low_hz, high_hz, label)
BAND_CONFIGS = {
    "standard_rs": (0.010, 0.100, "broad"),
    "endogenic":   (0.001, 0.020, "endogenic"),
    "neurogenic":  (0.020, 0.040, "neurogenic"),
    "vasogenic":   (0.060, 0.100, "vasogenic"),
}

# Maximum cross-correlation lag (seconds) used per band
BAND_LAG_SEC = {
    "standard_rs": 15.0,
    "endogenic":   30.0,
    "neurogenic":  20.0,
    "vasogenic":   10.0,
}
