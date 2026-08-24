#!/usr/bin/env python3
"""Reproduce the NIST PM7-descriptor sensitivity analysis.

The low-fidelity PM7 proton affinity remains the baseline. This analysis
removes only the 13 neutral and 13 protonated PM7 property columns before
running the production fold-local selector and ExtraTrees evaluation. The
default command writes ``results/nist1155_no_pm7_descriptors/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from train_models import run_cv


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = next((p for p in (PROJECT_DIR / "data", PROJECT_DIR.parent / "data") if p.exists()), PROJECT_DIR / "data")
TARGET_PATH = DATA_DIR / "targets" / "nist1155_ml.parquet"


def main() -> None:
    """Load the corrected NIST target table and run the fixed sensitivity."""
    data = pd.read_parquet(TARGET_PATH)
    pm7_descriptor_columns = [
        column
        for column in data.columns
        if column.startswith(("neutral_pm7_", "protonated_pm7_"))
    ]
    if len(pm7_descriptor_columns) != 26:
        raise ValueError(
            "Expected 26 neutral/protonated PM7 descriptor columns, found "
            f"{len(pm7_descriptor_columns)}"
        )

    data = data.drop(columns=pm7_descriptor_columns)
    run_cv(
        df=data,
        target_col="delta_pm7_exp",
        pa_pm7_col="pm7_best_pa_kjmol",
        pa_true_col="exp_pa_kjmol",
        dataset_name="nist1155_no_pm7_descriptors",
        model_names={"ExtraTrees"},
    )


if __name__ == "__main__":
    main()
