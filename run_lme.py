#!/usr/bin/env python3
"""
run_lme.py
═══════════════════════════════════════════════════════════════════════════════
Linear mixed-effects models for gBOLD–CSF coupling using surrogate-corrected
Δ|r| = observed |r| − null_p50  (HCP-YA, NIMG-26-2222).

Two subcommands
───────────────
  roi_band    Δ|r| ~ ROI × Band + (1|subject)
              6 ROIs × 3 bands, all 15 pairwise ROI contrasts per band

  window_band Δ|r| ~ Window × Band + (1|subject)
              4 time-windows × 3 bands at z0, 5 planned window pairs per band
              + GEE sensitivity check (exchangeable working correlation)

Usage
─────
  python code/run_lme.py roi_band \\
      --coupling_csv  results/coupling_g1/stage3_coupling.csv \\
      --surrogate_csv results/surrogate_analysis/surrogate_summary.csv \\
      --out           results/statistics/roi_band_delta_r

  python code/run_lme.py window_band \\
      --coupling_csv  results/coupling_g1/stage3_timewindow.csv \\
      --surrogate_csv results/surrogate_analysis/surrogate_windows.csv \\
      --out           results/statistics/z0_window_band_delta_r

  # filter to a specific run when surrogate CSVs contain multiple runs
  python code/run_lme.py roi_band    ... --run LR
  python code/run_lme.py window_band ... --run LR

Outputs (roi_band)
──────────────────
  delta_r_input.csv                    merged data with Δ|r|
  roi_band_descriptive_statistics.csv  group mean Δ|r| per ROI × Band
  mixed_model_full.txt                 full model summary
  omnibus_likelihood_ratio_tests.csv   ROI, Band, ROI × Band LRTs
  variance_components.csv              between-subject and residual variance
  all_pairwise_roi_contrasts.csv       15 ROI contrasts per band (FDR-corrected)
  z0_roi_contrasts_from_all_pairs.csv  z0 vs other ROIs (subset)

Outputs (window_band)
─────────────────────
  delta_r_window_input.csv                merged data with Δ|r|
  window_band_descriptive_statistics.csv  group mean Δ|r| per Window × Band
  mixed_model_window_full.txt             full LME model summary
  omnibus_lrt_window_band.csv             Window, Band, Window × Band LRTs
  pairwise_window_contrasts.csv           5 pairs × 3 bands (FDR-corrected)
  gee_window_band_summary.txt             GEE sensitivity model summary

Dependencies
────────────
  statsmodels, scipy, numpy, pandas

References
──────────
  Benjamini & Hochberg (1995) JRSS-B — FDR correction
  Liang & Zeger (1986) Biometrika — GEE
  Shrout & Fleiss (1979) Psychol Bull — variance components
"""

import os
import argparse
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

ROI_ORDER    = ["z0", "z1", "z2", "z3", "fv_lower", "fv_upper"]
BAND_ORDER   = ["broad", "neurogenic", "vasogenic"]
WINDOW_ORDER = ["Win1", "Win2", "Win3", "Win4"]

PLANNED_PAIRS = [
    ("Win1", "Win2"),
    ("Win1", "Win3"),
    ("Win1", "Win4"),
    ("Win2", "Win3"),
    ("Win3", "Win4"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def fit_lme(formula, data, reml=False):
    """Random-intercept LME with three-optimizer fallback."""
    model      = smf.mixedlm(formula, data=data,
                              groups=data["subject"], re_formula="1")
    last_error = None
    for method in ["lbfgs", "powell", "cg"]:
        try:
            result = model.fit(reml=reml, method=method,
                               maxiter=2000, disp=False)
            print(f"    optimizer={method:<7s} converged={result.converged}")
            if result.converged:
                return result
        except Exception as exc:
            last_error = exc
            print(f"    {method} failed: {exc}")
    if last_error is not None:
        raise RuntimeError(f"Mixed model failed. Last error: {last_error}")
    raise RuntimeError("Mixed model did not converge.")


def likelihood_ratio_test(full, reduced):
    """LRT for nested ML-fitted models. Returns dict with LR_chi2, df, p."""
    lr      = 2.0 * (full.llf - reduced.llf)
    df_diff = len(full.fe_params) - len(reduced.fe_params)
    return {"LR_chi2": lr, "df": df_diff, "p": chi2.sf(lr, df_diff)}


def load_and_merge(coupling_csv, surrogate_csv, run, merge_keys):
    """
    Load coupling and surrogate CSVs, filter to run, compute Δ|r|.
    merge_keys: list of columns to join on besides 'subject' and 'band',
                e.g. ['roi'] or ['window'].
    """
    df = pd.read_csv(coupling_csv)
    print(f"\nObserved coupling : {coupling_csv}  ({len(df)} rows)")

    # Tolerate alternate subject column names
    if "subject" not in df.columns:
        for c in ["sub", "Subject", "subject_id"]:
            if c in df.columns:
                df = df.rename(columns={c: "subject"})
                break

    df["subject"] = df["subject"].astype(str)

    surr = pd.read_csv(surrogate_csv)
    print(f"Surrogate summary : {surrogate_csv}  ({len(surr)} rows)")

    if "run" in surr.columns:
        surr = surr[surr["run"] == run].copy()
        print(f"  Filtered to run={run}: {len(surr)} rows")

    surr["subject"] = surr["subject"].astype(str)
    join_cols       = ["subject", "band"] + merge_keys
    surr = (surr[join_cols + ["null_p50"]]
            .rename(columns={"null_p50": "surrogate_median"})
            .drop_duplicates(subset=join_cols))

    merged = df.merge(surr, on=join_cols, how="left")

    n_missing = merged["surrogate_median"].isna().sum()
    if n_missing > 0:
        print(f"\nWARNING: {n_missing} rows have no surrogate median — dropped")
        merged = merged.dropna(subset=["surrogate_median"])

    merged["delta_r"] = merged["abs_r"] - merged["surrogate_median"]
    print(f"\nMerged rows : {len(merged)}")
    print(f"Subjects    : {merged['subject'].nunique()}")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Subcommand: roi_band
# ══════════════════════════════════════════════════════════════════════════════

def roi_effect_vector(roi, fe_names):
    """Fixed-effect contrast vector for one ROI (z0-reference model)."""
    v = np.zeros(len(fe_names))
    v[fe_names.index("Intercept")] = 1.0
    if roi != "z0":
        name = f"C(roi, Treatment(reference='z0'))[T.{roi}]"
        v[fe_names.index(name)] = 1.0
    return v


def run_roi_band(args):
    out = args.out
    os.makedirs(out, exist_ok=True)

    # ── Load & merge ──────────────────────────────────────────────────────────
    merged = load_and_merge(args.coupling_csv, args.surrogate_csv,
                            args.run, merge_keys=["roi"])
    merged = merged[merged["roi"].isin(ROI_ORDER) &
                    merged["band"].isin(BAND_ORDER)].dropna(
                        subset=["subject", "roi", "band", "abs_r"])

    print("\nΔ|r| summary (observed |r| − null_p50):")
    print(merged.groupby(["band", "roi"])["delta_r"].describe().round(4))
    merged.to_csv(os.path.join(out, "delta_r_input.csv"), index=False)

    df = merged.copy()
    df["roi"]  = pd.Categorical(df["roi"],  categories=ROI_ORDER,  ordered=True)
    df["band"] = pd.Categorical(df["band"], categories=BAND_ORDER, ordered=True)

    # ── Descriptive statistics ────────────────────────────────────────────────
    desc = (df.groupby(["band", "roi"], observed=True)["delta_r"]
              .agg(n="count", mean="mean", sd="std", median="median")
              .reset_index())
    desc["sem"]       = desc["sd"] / np.sqrt(desc["n"])
    desc["ci95_low"]  = desc["mean"] - 1.96 * desc["sem"]
    desc["ci95_high"] = desc["mean"] + 1.96 * desc["sem"]
    desc.to_csv(os.path.join(out, "roi_band_descriptive_statistics.csv"),
                index=False)
    print("\nDescriptive statistics (Δ|r|):")
    print(desc.to_string(index=False))

    # ── Full ROI × Band model ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FULL MODEL: Δ|r| ~ ROI × Band + (1|Subject)")
    print("=" * 60)

    full_formula = (
        "delta_r ~ C(roi, Treatment(reference='z0')) * "
        "C(band, Treatment(reference='broad'))"
    )
    full = fit_lme(full_formula, df, reml=False)
    print(full.summary())
    with open(os.path.join(out, "mixed_model_full.txt"), "w") as f:
        f.write(full.summary().as_text())

    # ── Omnibus likelihood-ratio tests ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OMNIBUS LIKELIHOOD-RATIO TESTS")
    print("=" * 60)

    additive  = fit_lme(
        "delta_r ~ C(roi, Treatment(reference='z0')) + "
        "C(band, Treatment(reference='broad'))", df, reml=False)
    band_only = fit_lme(
        "delta_r ~ C(band, Treatment(reference='broad'))", df, reml=False)
    roi_only  = fit_lme(
        "delta_r ~ C(roi, Treatment(reference='z0'))", df, reml=False)

    lrt_roi  = likelihood_ratio_test(additive, band_only)
    lrt_band = likelihood_ratio_test(additive, roi_only)
    lrt_int  = likelihood_ratio_test(full,     additive)

    for name, res in [("ROI main effect",        lrt_roi),
                      ("Band main effect",       lrt_band),
                      ("ROI × Band interaction", lrt_int)]:
        print(f"  {name:<28s} χ²={res['LR_chi2']:.3f}, "
              f"df={res['df']}, p={res['p']:.6g}")

    pd.DataFrame([
        {"effect": "ROI",        **lrt_roi},
        {"effect": "Band",       **lrt_band},
        {"effect": "ROI_x_Band", **lrt_int},
    ]).to_csv(os.path.join(out, "omnibus_likelihood_ratio_tests.csv"),
              index=False)

    # ── Variance components ───────────────────────────────────────────────────
    subject_var = float(full.cov_re.iloc[0, 0])
    resid_var   = float(full.scale)
    icc_subj    = subject_var / (subject_var + resid_var)

    pd.DataFrame({
        "component": ["between_subject_variance", "residual_variance",
                      "conditional_subject_ICC"],
        "value":     [subject_var, resid_var, icc_subj],
    }).to_csv(os.path.join(out, "variance_components.csv"), index=False)

    print(f"\n  Subject variance : {subject_var:.6f}")
    print(f"  Residual variance: {resid_var:.6f}")
    print(f"  Subject ICC      : {icc_subj:.3f}")

    # ── All 15 pairwise ROI contrasts within each band ────────────────────────
    print("\n" + "=" * 60)
    print("ALL PAIRWISE ROI CONTRASTS WITHIN EACH BAND")
    print("15 contrasts per band, FDR-corrected (Benjamini-Hochberg)")
    print("=" * 60)

    contrast_rows = []
    roi_pairs     = list(combinations(ROI_ORDER, 2))

    for band in BAND_ORDER:
        print(f"\n--- {band} ---")
        bdf   = df[df["band"] == band].copy()
        model = fit_lme(
            "delta_r ~ C(roi, Treatment(reference='z0'))", bdf, reml=True)

        fe_names = list(model.fe_params.index)
        beta     = model.fe_params.to_numpy()
        cov_fe   = model.cov_params().loc[fe_names, fe_names].to_numpy()

        for roi1, roi2 in roi_pairs:
            c        = (roi_effect_vector(roi1, fe_names)
                        - roi_effect_vector(roi2, fe_names))
            estimate = float(c @ beta)
            se       = float(np.sqrt(c @ cov_fe @ c))
            z        = estimate / se if se > 0 else np.nan
            p        = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan

            contrast_rows.append({
                "band":                  band,
                "contrast":              f"{roi1} - {roi2}",
                "roi1":                  roi1,
                "roi2":                  roi2,
                "estimate_delta_r_diff": estimate,
                "SE":                    se,
                "ci95_low":              estimate - 1.96 * se,
                "ci95_high":             estimate + 1.96 * se,
                "z":                     z,
                "p_raw":                 p,
            })

    contrasts = pd.DataFrame(contrast_rows)
    contrasts["p_FDR"]           = np.nan
    contrasts["significant_FDR"] = False

    for band in BAND_ORDER:
        idx = contrasts.index[contrasts["band"] == band]
        reject, p_fdr, _, _ = multipletests(
            contrasts.loc[idx, "p_raw"], alpha=0.05, method="fdr_bh")
        contrasts.loc[idx, "p_FDR"]           = p_fdr
        contrasts.loc[idx, "significant_FDR"] = reject

    contrasts.to_csv(os.path.join(out, "all_pairwise_roi_contrasts.csv"),
                     index=False)
    print("\nAll pairwise ROI contrasts (Δ|r|, FDR-corrected within band):")
    print(contrasts[["band", "contrast", "estimate_delta_r_diff",
                      "ci95_low", "ci95_high", "z",
                      "p_raw", "p_FDR", "significant_FDR"]]
          .to_string(index=False))

    # z0-focused subset
    contrasts[contrasts["roi1"] == "z0"].to_csv(
        os.path.join(out, "z0_roi_contrasts_from_all_pairs.csv"), index=False)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Outputs → {out}/")


# ══════════════════════════════════════════════════════════════════════════════
# Subcommand: window_band
# ══════════════════════════════════════════════════════════════════════════════

def run_window_band(args):
    out = args.out
    os.makedirs(out, exist_ok=True)

    # ── Load & merge ──────────────────────────────────────────────────────────
    merged = load_and_merge(args.coupling_csv, args.surrogate_csv,
                            args.run, merge_keys=["window"])

    # Normalise window column to "Win1" format if stored as integers
    if not merged["window"].astype(str).str.startswith("Win").all():
        merged["window"] = "Win" + merged["window"].astype(int).astype(str)

    merged = merged[merged["window"].isin(WINDOW_ORDER) &
                    merged["band"].isin(BAND_ORDER)].dropna(
                        subset=["subject", "window", "band", "abs_r"])

    print("\nΔ|r| summary (observed |r| − null_p50):")
    print(merged.groupby(["band", "window"])["delta_r"].describe().round(4))
    merged.to_csv(os.path.join(out, "delta_r_window_input.csv"), index=False)

    df = merged.copy()
    df["window"] = pd.Categorical(df["window"], categories=WINDOW_ORDER, ordered=True)
    df["band"]   = pd.Categorical(df["band"],   categories=BAND_ORDER,   ordered=True)

    # ── Descriptive statistics ────────────────────────────────────────────────
    desc = (df.groupby(["band", "window"], observed=True)["delta_r"]
              .agg(n="count", mean="mean", sd="std", median="median")
              .reset_index())
    desc["sem"]       = desc["sd"] / np.sqrt(desc["n"])
    desc["ci95_low"]  = desc["mean"] - 1.96 * desc["sem"]
    desc["ci95_high"] = desc["mean"] + 1.96 * desc["sem"]
    desc.to_csv(os.path.join(out, "window_band_descriptive_statistics.csv"),
                index=False)
    print("\nDescriptive statistics (Δ|r|):")
    print(desc.to_string(index=False))

    # ── Full Window × Band model ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FULL LME MODEL: Δ|r| ~ Window × Band + (1|Subject)")
    print("=" * 60)

    full_formula = (
        "delta_r ~ C(window, Treatment(reference='Win1')) * "
        "C(band, Treatment(reference='broad'))"
    )
    print("Fitting full model (ML)...")
    full = fit_lme(full_formula, df, reml=False)
    print(full.summary())
    with open(os.path.join(out, "mixed_model_window_full.txt"), "w") as f:
        f.write(full.summary().as_text())

    # ── Omnibus likelihood-ratio tests ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OMNIBUS LIKELIHOOD-RATIO TESTS")
    print("=" * 60)

    print("Fitting additive model (ML)...")
    additive    = fit_lme(
        "delta_r ~ C(window, Treatment(reference='Win1')) + "
        "C(band, Treatment(reference='broad'))", df, reml=False)
    print("Fitting band-only model (ML)...")
    band_only   = fit_lme(
        "delta_r ~ C(band, Treatment(reference='broad'))", df, reml=False)
    print("Fitting window-only model (ML)...")
    window_only = fit_lme(
        "delta_r ~ C(window, Treatment(reference='Win1'))", df, reml=False)

    lrt_window = likelihood_ratio_test(additive, band_only)
    lrt_band   = likelihood_ratio_test(additive, window_only)
    lrt_int    = likelihood_ratio_test(full,     additive)

    for name, res in [("Window main effect",        lrt_window),
                      ("Band main effect",          lrt_band),
                      ("Window × Band interaction", lrt_int)]:
        print(f"  {name:<30s} χ²={res['LR_chi2']:.3f}, "
              f"df={res['df']}, p={res['p']:.6g}")

    pd.DataFrame([
        {"effect": "Window",        **lrt_window},
        {"effect": "Band",          **lrt_band},
        {"effect": "Window_x_Band", **lrt_int},
    ]).to_csv(os.path.join(out, "omnibus_lrt_window_band.csv"), index=False)

    # ── Planned pairwise window contrasts — paired t-tests ───────────────────
    print("\n" + "=" * 60)
    print("PAIRWISE WINDOW CONTRASTS (5 planned pairs × 3 bands)")
    print("FDR-corrected across all 15 tests (Benjamini-Hochberg)")
    print("=" * 60)

    contrast_rows = []
    subject_means = (df.groupby(["subject", "window", "band"], observed=True)
                     ["delta_r"].mean().reset_index())

    for band in BAND_ORDER:
        bdf = subject_means[subject_means["band"] == band]
        for w1, w2 in PLANNED_PAIRS:
            w1_vals = bdf[bdf["window"] == w1].set_index("subject")["delta_r"]
            w2_vals = bdf[bdf["window"] == w2].set_index("subject")["delta_r"]
            common  = w1_vals.index.intersection(w2_vals.index)
            diff    = w1_vals.loc[common] - w2_vals.loc[common]

            if len(diff) < 2:
                continue

            t_stat, p_raw = stats.ttest_1samp(diff, 0.0)
            se = diff.std(ddof=1) / np.sqrt(len(diff))

            contrast_rows.append({
                "band":       band,
                "contrast":   f"{w1} - {w2}",
                "window1":    w1,
                "window2":    w2,
                "n_subjects": len(diff),
                "mean_diff":  diff.mean(),
                "sd_diff":    diff.std(ddof=1),
                "SE":         se,
                "ci95_low":   diff.mean() - 1.96 * se,
                "ci95_high":  diff.mean() + 1.96 * se,
                "t":          t_stat,
                "df":         len(diff) - 1,
                "p_raw":      p_raw,
            })

    contrasts = pd.DataFrame(contrast_rows)
    reject, p_fdr, _, _ = multipletests(contrasts["p_raw"], alpha=0.05,
                                         method="fdr_bh")
    contrasts["p_FDR"]           = p_fdr
    contrasts["significant_FDR"] = reject

    contrasts.to_csv(os.path.join(out, "pairwise_window_contrasts.csv"),
                     index=False)
    print("\nPlanned pairwise window contrasts (Δ|r|, FDR-corrected):")
    print(contrasts[["band", "contrast", "mean_diff", "ci95_low", "ci95_high",
                      "t", "df", "p_raw", "p_FDR", "significant_FDR"]]
          .to_string(index=False))

    # ── GEE sensitivity check ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GEE SENSITIVITY CHECK (exchangeable working correlation)")
    print("=" * 60)

    try:
        gee_result = smf.gee(
            "delta_r ~ C(window, Treatment(reference='Win1')) * "
            "C(band, Treatment(reference='broad'))",
            groups     = "subject",
            data       = df,
            family     = sm.families.Gaussian(),
            cov_struct = sm.cov_struct.Exchangeable(),
        ).fit()
        print(gee_result.summary())
        with open(os.path.join(out, "gee_window_band_summary.txt"), "w") as f:
            f.write(gee_result.summary().as_text())
        print(f"  GEE summary written to {out}/gee_window_band_summary.txt")
    except Exception as exc:
        print(f"  WARNING: GEE fitting failed — {exc}")
        print("  Proceeding without GEE output.")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Outputs → {out}/")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="LME models for surrogate-corrected Δ|r| (HCP-YA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # ── shared arguments ──────────────────────────────────────────────────────
    def add_shared(p, default_out):
        p.add_argument("--coupling_csv",   required=True,
                       help="Coupling CSV from analyse_coupling.py "
                            "(must contain: subject, band, abs_r, + roi or window)")
        p.add_argument("--surrogate_csv",  required=True,
                       help="Surrogate summary CSV from surrogate_null.py "
                            "(must contain: subject, band, null_p50, + roi or window)")
        p.add_argument("--run",            default="LR",
                       help="Run label to filter from surrogate CSV "
                            "(default: LR)")
        p.add_argument("--out",            default=default_out,
                       help=f"Output directory (default: {default_out})")

    # ── roi_band subcommand ───────────────────────────────────────────────────
    p_roi = sub.add_parser(
        "roi_band",
        help="Δ|r| ~ ROI × Band + (1|subject)  —  6 ROIs × 3 bands",
    )
    add_shared(p_roi, "results/statistics/roi_band_delta_r")

    # ── window_band subcommand ────────────────────────────────────────────────
    p_win = sub.add_parser(
        "window_band",
        help="Δ|r| ~ Window × Band + (1|subject)  —  4 windows × 3 bands at z0",
    )
    add_shared(p_win, "results/statistics/z0_window_band_delta_r")

    args = ap.parse_args()

    if args.command == "roi_band":
        run_roi_band(args)
    elif args.command == "window_band":
        run_window_band(args)


if __name__ == "__main__":
    main()
