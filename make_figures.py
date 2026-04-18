#!/usr/bin/env python3
"""
make_figures.py
===============
Single-entry-point orchestrator that regenerates every figure in the paper
from the scripts under ``scripts/plotting/`` and ``screening/scripts/plotting/``.

Intended for reviewers who want to verify that every figure in the manuscript
and SI is reproducible from the shipped code and data.

Usage
-----
    # regenerate everything (main-paper + screening figures)
    python make_figures.py

    # list the ordered steps without running anything
    python make_figures.py --list

    # only main-paper figures (scripts/plotting/)
    python make_figures.py --only main

    # only the prospective-screening figures (screening/scripts/plotting/)
    python make_figures.py --only screening

    # stop at the first failing script instead of continuing
    python make_figures.py --stop-on-error

Exit status
-----------
    0 - every step completed successfully
    1 - at least one step failed (see the per-step summary at the end)

Data dependencies
-----------------
All scripts assume:
  - ``results/`` and ``figures/`` exist (shipped in-repo).
  - ``data/`` contains the ~6.9 GB raw + processed data set (not in-repo;
    download from the external host listed in the top-level README, or
    regenerate with ``scripts/calculations/`` + ``pm7_scripts/`` /
    ``dft_scripts/``).
Scripts that need ``data/`` will fail with a self-explanatory
FileNotFoundError if it is missing; this orchestrator reports that
cleanly instead of crashing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class Step:
    """One plotting script invocation."""
    name: str
    script: Path
    group: str
    args: Sequence[str] = field(default_factory=tuple)
    produces: str = ""

    @property
    def cmd(self) -> list[str]:
        return [sys.executable, str(self.script), *self.args]


# ---------------------------------------------------------------------------
# Step registry. Order matters: scripts earlier in the list should not depend
# on outputs of later scripts. Within a group, ordering is roughly "cheap first
# (uses results/) -> expensive last (uses raw data/)".
# ---------------------------------------------------------------------------
STEPS: list[Step] = [
    # --- main-paper figures (scripts/plotting/) ----------------------------
    Step(
        name="parity plots + model comparison",
        script=REPO_ROOT / "scripts" / "plotting" / "plot_results.py",
        group="main",
        produces="figures/model_performance/parity_*.pdf",
    ),
    Step(
        name="learning curves (per-task)",
        script=REPO_ROOT / "scripts" / "plotting" / "plot_learning_curves.py",
        group="main",
        produces="figures/model_performance/learning_curve_*.pdf",
    ),
    Step(
        name="learning curves (combined 2x2 panel)",
        script=REPO_ROOT / "scripts" / "plotting" / "plot_lc_combined.py",
        group="main",
        produces="figures/model_performance/learning_curve_combined.pdf",
    ),
    Step(
        name="SHAP beeswarm / dependence / importance",
        script=REPO_ROOT / "scripts" / "plotting" / "plot_shap.py",
        group="main",
        produces="figures/shap/*.pdf",
    ),
    Step(
        name="chemical analysis (FG classes, corrections)",
        script=REPO_ROOT / "scripts" / "plotting" / "plot_chemical_analysis.py",
        group="main",
        produces="figures/chemical_analysis/*.pdf",
    ),
    Step(
        name="data exploration (PA dist, Bottcher, FG prevalence)",
        script=REPO_ROOT / "scripts" / "plotting" / "plot_exploration.py",
        group="main",
        produces="figures/exploration/*.pdf",
    ),
    Step(
        name="k-means latent space vs ZINC",
        script=REPO_ROOT / "scripts" / "plotting" / "plot_chemical_space_zinc.py",
        group="main",
        produces="figures/figure1_kmeans_latent_space_clean.pdf",
    ),
    Step(
        name="SI: protonation-site histogram (k-means)",
        script=REPO_ROOT / "scripts" / "plotting" / "plot_site_histogram.py",
        group="main",
        produces="figures/si_site_distribution_kmeans.pdf",
    ),

    # --- prospective-screening figures (screening/scripts/plotting/) -------
    Step(
        name="Pareto front figure (iter 1)",
        script=REPO_ROOT / "screening" / "scripts" / "plotting" / "plot_pareto.py",
        group="screening",
        args=("--iter", "1"),
        produces="screening/figures/iter1/pareto_*.pdf",
    ),
    Step(
        name="PA parity (DFT vs PM7 vs predicted)",
        script=REPO_ROOT / "screening" / "scripts" / "plotting" / "plot_pa_parity_final.py",
        group="screening",
        args=("--iter", "1"),
        produces="screening/figures/iter1/pa_parity_*.pdf",
    ),
    Step(
        name="iter-1 summary panel",
        script=REPO_ROOT / "screening" / "scripts" / "plotting" / "08_plot_results.py",
        group="screening",
        args=("--iter", "1"),
        produces="screening/figures/iter1/*.pdf",
    ),
    Step(
        name="SI: all 30 DFT-validated candidates",
        script=REPO_ROOT / "screening" / "scripts" / "plotting" / "plot_si_candidates.py",
        group="screening",
        produces="figures/si_all30_candidates.pdf",
    ),
]


def format_header(text: str, char: str = "=") -> str:
    line = char * 72
    return f"\n{line}\n{text}\n{line}"


def run_step(step: Step, verbose: bool) -> tuple[bool, float, str]:
    """Run one step. Returns (ok, elapsed_seconds, tail_of_output)."""
    t0 = time.perf_counter()
    proc = subprocess.run(
        step.cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    ok = proc.returncode == 0

    if verbose or not ok:
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)

    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    last_line = tail[-1] if tail else ""
    return ok, elapsed, last_line


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate all paper figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--only",
        choices=("main", "screening", "all"),
        default="all",
        help="Restrict to a group (default: all).",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print the ordered step list and exit.",
    )
    ap.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort at the first failing step instead of continuing.",
    )
    ap.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Stream each step's stdout/stderr live instead of only on failure.",
    )
    args = ap.parse_args()

    steps = [s for s in STEPS if args.only == "all" or s.group == args.only]

    if args.list:
        print(format_header(f"{len(steps)} plotting steps (group: {args.only})"))
        for i, s in enumerate(steps, 1):
            marker = "[main]" if s.group == "main" else "[scrn]"
            rel = s.script.relative_to(REPO_ROOT)
            print(f"  {i:2d}. {marker} {s.name}")
            print(f"      script:   {rel}")
            if s.args:
                print(f"      args:     {' '.join(s.args)}")
            if s.produces:
                print(f"      produces: {s.produces}")
        return 0

    missing = [s for s in steps if not s.script.exists()]
    if missing:
        print("ERROR: the following plot scripts are missing from the repo:",
              file=sys.stderr)
        for s in missing:
            print(f"  - {s.script.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1

    print(format_header(
        f"Regenerating {len(steps)} figures (group: {args.only})"
    ))

    results: list[tuple[Step, bool, float, str]] = []
    t_global = time.perf_counter()
    for i, step in enumerate(steps, 1):
        rel = step.script.relative_to(REPO_ROOT)
        print(f"\n[{i:2d}/{len(steps)}] {step.name}")
        print(f"         python {rel} {' '.join(step.args)}".rstrip())
        ok, elapsed, last = run_step(step, verbose=args.verbose)
        status = "OK  " if ok else "FAIL"
        print(f"         {status}  ({elapsed:5.1f}s)")
        if not ok:
            print(f"         last:  {last[:200]}")
        results.append((step, ok, elapsed, last))
        if not ok and args.stop_on_error:
            print("\nAborting due to --stop-on-error.")
            break

    total = time.perf_counter() - t_global
    n_ok = sum(1 for _, ok, _, _ in results if ok)
    n_fail = len(results) - n_ok

    print(format_header(
        f"Summary: {n_ok} ok, {n_fail} failed, total {total:.1f}s"
    ))
    for step, ok, elapsed, last in results:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {elapsed:5.1f}s  {step.name}")
        if not ok:
            print(f"           -> {last[:160]}")

    if n_fail:
        print("\nHint: rerun a single failing step with -v to see the full"
              " traceback, or check that data/ is present (see top-level"
              " README for the external data link).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
