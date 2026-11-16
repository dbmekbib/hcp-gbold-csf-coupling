"""
run_pipeline_hcp.py — HCP-YA gBOLD-CSF pipeline runner
════════════════════════════════════════════════════════════════════════════════
Preprocessing and timeseries extraction steps 1–7 for HCP-YA resting-state fMRI.
    1   Dummy volume removal + QC
    2   Bias correction + topup distortion correction
    3   Slice timing correction
    4   Motion correction (mcflirt, sbref_unwarped reference)
    42  Variance from bold_unwarped (no MC — QC branch)
    5   Registration (FLIRT 6-DOF, aseg projection, tissue masks)
    5b  WM mask refinement + QC  (step5b in pipeline_steps.py)
    61  Nuisance regression P0–P3 (detrend + optional confounds)
    62  Bandpass filter — 4 bands × 2 CSF sources
    7   Timeseries extraction — coupling-ready CSVs (gBOLD, CSF, WM, motion)


Run labels
──────────
    LR    REST1 LR (primary cohort, N=104)
    RL    REST1 RL (primary cohort, N=104)
    LR2   REST2 LR (test-retest subsample, N=25)
    RL2   REST2 RL (test-retest subsample, N=25)
    both  LR + RL   (REST1 only)
    all4  LR + RL + LR2 + RL2

Topup convention
────────────────
    LR / RL   → REST1 session SE field maps (rfMRI_REST1_LR_SBRef + rfMRI_REST1_RL_SBRef)
    LR2 / RL2 → REST2 session SE field maps (rfMRI_REST2_LR_SBRef + rfMRI_REST2_RL_SBRef)
    Within each session, topup is computed once and shared between both runs.
    A per-session lock file prevents double computation in parallel runs.

Usage
─────
    # Single-subject test run (explicit ID, no CSV needed):
    python run_pipeline_hcp.py \\
        --steps 1 2 3 4 42 5 5b 61 62 7 --subject-ids 100307 --run LR --workers 1

    # Full pipeline, first 2 subjects, REST1 only (needs subject CSV):
    python run_pipeline_hcp.py \\
        --steps 1 2 3 4 42 5 61 62 7 --csv coupling_subjects_109.csv \\
        --subjects 2 --run both --csf_source mc

    # REST2 runs (test-retest subsample):
    python run_pipeline_hcp.py \\
        --steps 1 2 3 4 5 61 62 7 --csv retest_subjects_25.csv \\
        --subjects 25 --run all4 --workers 4
"""

import os
import sys
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config_hcp as cfg
from pipeline_steps import (step1, step2, step3, step4, step4b,
                            step5, step5b, step6a, step6b, step7)


# ── Step registry ─────────────────────────────────────────────────────────────
STEP_MODS = {
    1:  step1,
    2:  step2,
    3:  step3,
    4:  step4,
    42: step4b,
    5:  step5,
    61: step6a,
    62: step6b,
    7:  step7,
}

STEP_DESCRIPTIONS = {
    1:  "Dummy volume removal + QC",
    2:  "Bias correction + topup distortion correction",
    3:  "Slice timing correction",
    4:  "Motion correction (mcflirt, sbref_unwarped reference)",
    42: "Variance from bold_unwarped (no MC — QC branch)",
    5:  "Registration (FLIRT 6-DOF) + tissue masks from aseg",
    61: "Nuisance regression P0–P3 (detrend + optional confounds)",
    62: "Bandpass filter — 4 bands (gBOLD + CSF from csf_source)",
    7:  "Timeseries extraction — gBOLD, WM, CSF, motion, bandpass",
}

# Steps that do NOT use csf_source
STEPS_NO_SOURCE = {1, 2, 3, 4, 42, 5, 61}


# ── Per-subject runner ────────────────────────────────────────────────────────
def run_subject(sub: str, run: str,
                steps: list,
                csf_source: str = "mc",
                force: bool = False) -> dict:
    """Run pipeline steps sequentially for one subject/run."""
    result = {"subject": sub, "run": run, "steps": {}, "status": "ok"}

    prefix = f"[{sub}/{run}]"

    def log(msg):
        print(f"{prefix} {msg}", flush=True)

    log(f"\n{'─'*55}")
    log(f"  HCP  {sub}/{run}  steps={steps}  csf_source={csf_source}")
    log(f"  TR={cfg.TR}s  N_VOLS={cfg.N_VOLUMES}  "
        f"N_SLICES={cfg.N_SLICES}  MB={cfg.MB_FACTOR}")
    log(f"{'─'*55}")

    for step_n in steps:

        # Step 5b: WM mask refinement
        if step_n == "5b":
            log(f"\n  ━━ Step 5b: WM mask refinement ━━")
            try:
                r = step5b.run(sub, run, force=force, logger=log)
                st = r.get("status", "unknown")
                result["steps"]["5b"] = st
                if st in ("ok", "skip"):
                    log(f"  ✓ Step 5b complete  "
                        f"ero1_novent={r.get('n_wm_ero1_novent', 0):,}  "
                        f"flag={r.get('qc_flag', '')}")
                else:
                    log(f"  ✗ Step 5b {st}  {r.get('error', '')}")
                    result["status"] = "fail"
                    break
            except Exception as e:
                log(f"  ✗ Step 5b EXCEPTION: {e}")
                traceback.print_exc()
                result["steps"]["5b"] = "exception"
                result["status"] = "fail"
                break
            continue

        mod = STEP_MODS.get(step_n)
        if mod is None:
            log(f"  ⚠ Step {step_n} not in registry — skipping")
            continue

        log(f"\n  ━━ Step {step_n}: {STEP_DESCRIPTIONS.get(step_n, '')} ━━")

        try:
            if step_n == 7:
                r = mod.run(sub, run, csf_source=csf_source,
                            force=force, logger=log)
            elif step_n in STEPS_NO_SOURCE:
                r = mod.run(sub, run, logger=log)
            elif step_n == 62:
                r = mod.run(sub, run,
                            pipelines=cfg.PIPELINES_TO_RUN,
                            csf_source=csf_source,
                            logger=log)
            else:
                r = mod.run(sub, run, logger=log)

            status = r.get("status", "ok")
            result["steps"][step_n] = status

            if status == "fail":
                log(f"  ✗ Step {step_n} FAILED")
                if r.get("missing"):
                    log(f"    Missing: {r['missing']}")
                if r.get("warnings"):
                    for w in r["warnings"]:
                        log(f"    ⚠ {w}")
                result["status"] = "fail"
                break
            else:
                if r.get("warnings"):
                    for w in r["warnings"]:
                        log(f"    ⚠ {w}")
                log(f"  ✓ Step {step_n} complete")

        except Exception as e:
            log(f"  ✗ Step {step_n} EXCEPTION: {e}")
            traceback.print_exc()
            result["steps"][step_n] = "exception"
            result["status"] = "fail"
            break

    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="HCP-YA gBOLD-CSF pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    ap.add_argument("--steps", nargs="+", default=["1", "2", "3", "4", "5"],
                    help="Steps to run: 1 2 3 4 42 5 5b 61 62 7  "
                         "(5b = WM mask refinement; 7 = timeseries extraction)")
    ap.add_argument("--run",
                    choices=["LR", "RL", "LR2", "RL2", "both", "all4"],
                    default="both",
                    help=(
                        "Run IDs to process:\n"
                        "  LR    REST1 LR (primary)\n"
                        "  RL    REST1 RL (primary)\n"
                        "  LR2   REST2 LR (test-retest)\n"
                        "  RL2   REST2 RL (test-retest)\n"
                        "  both  LR + RL  (REST1 only, default)\n"
                        "  all4  LR + RL + LR2 + RL2"))
    ap.add_argument("--subjects", type=int, default=None,
                    help="Process first N subjects from the CSV (requires --csv)")
    ap.add_argument("--skip", type=int, default=0,
                    help="Skip first N subjects in the CSV (resume from N+1)")
    ap.add_argument("--subject-list", type=str, default=None,
                    help="Text file with one subject ID per line")
    ap.add_argument("--subject-ids", nargs="+", default=None,
                    metavar="ID",
                    help="Explicit subject IDs, e.g. --subject-ids 100307 100408")
    ap.add_argument("--csv", type=str, default=None,
                    help="Subject CSV for --subjects / --skip slicing "
                         "(required when neither --subject-list nor --subject-ids is given)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel workers (1 = sequential)")
    ap.add_argument("--force", action="store_true",
                    help="Rerun even if outputs exist")
    ap.add_argument("--csf_source", choices=["mc", "stc"], default="mc",
                    help=(
                        "CSF volume source for step 62:\n"
                        "  'mc'  = bold_mc_smooth_clean (same as gBOLD) — PRIMARY\n"
                        "  'stc' = bold_stc (before motion correction) — sensitivity\n"
                        "Step 61 is source-agnostic — run once only."))
    args = ap.parse_args()

    # Convert step args to int where possible, keep "5b" as string
    steps = []
    for s in args.steps:
        try:
            steps.append(int(s))
        except ValueError:
            steps.append(s)   # "5b"

    # ── Run expansion ─────────────────────────────────────────────────────────
    if args.run == "both":
        runs = ["LR", "RL"]
    elif args.run == "all4":
        runs = ["LR", "RL", "LR2", "RL2"]
    else:
        runs = [args.run]

    # ── Subject selection ─────────────────────────────────────────────────────
    if args.subject_list:
        with open(args.subject_list) as f:
            subjects = [l.strip() for l in f
                        if l.strip() and not l.startswith("#")]
        print(f"  Subjects: {len(subjects)} from {args.subject_list}")
    elif args.subject_ids:
        subjects = [str(s) for s in args.subject_ids]
        print(f"  Subjects: {len(subjects)} explicit IDs")
    else:
        # Need a CSV to slice from
        if args.csv is None:
            ap.error("Provide --subject-list, --subject-ids, or --csv "
                     "(required for --subjects / --skip slicing)")
        all_subjects = cfg.load_subjects(args.csv)
        skip = args.skip or 0
        if args.subjects:
            subjects = all_subjects[skip: skip + args.subjects]
        elif args.skip:
            subjects = all_subjects[skip:]
        else:
            subjects = all_subjects
        print(f"  Subjects: {len(subjects)} from {args.csv}"
              + (f" [skip={skip}]" if skip else ""))

    work = [(sub, run) for sub in subjects for run in runs]

    # ── Print plan ────────────────────────────────────────────────────────────
    source_relevant = 62 in steps
    print(f"\n{'='*60}")
    print(f"  HCP-YA gBOLD-CSF Pipeline")
    print(f"  Subjects   : {len(subjects)}")
    print(f"  Runs       : {runs}")
    print(f"  Steps      : {steps}")
    print(f"  Workers    : {args.workers}")
    if source_relevant:
        src_label = ("bold_mc_smooth_clean (same as gBOLD)"
                     if args.csf_source == "mc"
                     else "bold_stc (before motion correction)")
        print(f"  csf_source : {args.csf_source}  ({src_label})")
    print(f"  ROOT       : {cfg.ROOT}")
    print()
    for s in steps:
        if s == "5b":
            desc = "WM mask refinement + QC"
        else:
            desc = STEP_DESCRIPTIONS.get(s, "unknown")
        src_note = f"  [csf_source={args.csf_source}]" if s == 62 else ""
        print(f"    Step {str(s):>3s}: {desc}{src_note}")
    print(f"{'='*60}")

    out_root = os.path.join(cfg.ROOT, "results", "preprocessing")
    os.makedirs(out_root, exist_ok=True)

    # ── Run ───────────────────────────────────────────────────────────────────
    n_ok = n_fail = 0

    if args.workers == 1:
        for sub, run in work:
            r = run_subject(sub, run, steps,
                            csf_source=args.csf_source,
                            force=args.force)
            if r["status"] == "ok":
                n_ok += 1
                print(f"  [{sub}/{run}] ✓")
            else:
                n_fail += 1
                print(f"  [{sub}/{run}] ✗  steps={r['steps']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(run_subject, sub, run, steps,
                          args.csf_source, args.force): (sub, run)
                for sub, run in work
            }
            for fut in as_completed(futures):
                sub, run = futures[fut]
                try:
                    r = fut.result()
                    if r["status"] == "ok":
                        n_ok += 1
                        print(f"  [{sub}/{run}] ✓")
                    else:
                        n_fail += 1
                        print(f"  [{sub}/{run}] ✗  steps={r['steps']}")
                except Exception as e:
                    n_fail += 1
                    print(f"  [{sub}/{run}] EXCEPTION: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Done: {n_ok}/{len(work)} OK  |  {n_fail} failed")
    if source_relevant:
        print(f"  csf_source={args.csf_source}")
    print(f"{'='*60}")

    # ── Next step hints ───────────────────────────────────────────────────────
    if 5 in steps and n_ok > 0:
        print(f"\n  NEXT: Run step 5b (WM mask refinement):")
        print(f"    python run_pipeline_hcp.py --steps 5b "
              f"--subjects {len(subjects)} --run {args.run} --workers {args.workers}")
        print(f"    (step5b is built into pipeline_steps.py — "
              f"no separate registration_qc.py needed)")

    if 61 in steps and n_ok > 0:
        print(f"\n  NEXT: Bandpass (run once per csf_source):")
        for src in ["mc", "stc"]:
            print(f"    python run_pipeline_hcp.py --steps 62 "
                  f"--subjects {len(subjects)} --run {args.run} "
                  f"--workers {args.workers} --csf_source {src}")

    if 62 in steps and n_ok > 0:
        other = "stc" if args.csf_source == "mc" else "mc"
        print(f"\n  NEXT: Extract timeseries (step 7):")
        print(f"    python run_pipeline_hcp.py --steps 7 "
              f"--subjects_csv coupling_subjects_109.csv "
              f"--run {' '.join(runs)} --csf_source {args.csf_source} "
              f"--workers {args.workers}")
        print(f"\n  Also run the other csf_source condition (step 62):")
        print(f"    python run_pipeline_hcp.py --steps 62 "
              f"--subjects {len(subjects)} --run {args.run} "
              f"--workers {args.workers} --csf_source {other}")

    if 2 in steps and n_ok > 0:
        print(f"\n  NEXT: Draw CSF masks on sbref_unwarped.nii.gz")
        for sub in subjects[:3]:
            p = cfg.subject_dirs(sub, runs[0])
            print(f"    fsleyes {p['sbref_unwarped']}")
        if len(subjects) > 3:
            print(f"    … and {len(subjects)-3} more")


if __name__ == "__main__":
    main()
