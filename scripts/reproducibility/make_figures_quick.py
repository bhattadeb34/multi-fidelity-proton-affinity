#!/usr/bin/env python3
"""Regenerate core quantitative figures from saved artifact tables only.

This path does not read raw DFT outputs, calculate descriptors, select
features, fit estimators, or call an external service.  It writes to
``artifacts/quick_figures/`` unless another output directory is requested.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from artifact_store import ReleaseLayout


SCRIPTS = (
    "scripts/plotting/plot_results.py",
    "scripts/plotting/plot_learning_curves.py",
    "scripts/plotting/plot_shap.py",
    "scripts/plotting/plot_diff_esi.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP plots when the optional shap package is unavailable.",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    layout = ReleaseLayout.discover(args.code_root, args.artifact_root)
    results = layout.artifacts / "plot_inputs" / "results"
    output = (args.output or layout.artifacts / "quick_figures").resolve()
    if args.list:
        for script in SCRIPTS:
            print(script)
        return
    if not results.exists():
        raise FileNotFoundError(
            f"Saved plot inputs are absent at {results}. "
            "Run build_quick_artifacts.py first."
        )
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MFPA_RESULTS_DIR"] = str(results)
    environment["MFPA_FIGURES_DIR"] = str(output)
    scripts = [script for script in SCRIPTS if not (args.skip_shap and script.endswith("plot_shap.py"))]
    with tempfile.TemporaryDirectory(prefix="mfpa-matplotlib-") as cache:
        environment["MPLCONFIGDIR"] = cache
        for relative in scripts:
            script = layout.code / relative
            print(f"Running {relative}")
            subprocess.run(
                [sys.executable, str(script)],
                cwd=layout.code,
                env=environment,
                check=True,
            )
    print(f"Quick figures written to {output}")


if __name__ == "__main__":
    main()
