#!/usr/bin/env python3
"""Run the corrected k-means workflow with row-wise KFold as an ESI audit.

This intentionally non-production analysis quantifies the effect of allowing
sites from one molecule to cross folds. Production results remain molecule-
grouped in ``results/kmeans251``.
"""

from pathlib import Path

import pandas as pd

from train_models import KJMOL_TO_KCAL, TARGET_DIR, run_cv


def main() -> None:
    df = pd.read_parquet(TARGET_DIR / "kmeans251_ml.parquet")
    run_cv(
        df=df,
        target_col="delta_dft_pm7",
        pa_pm7_col="pm7_pa_kjmol",
        pa_true_col="dft_pa_kjmol",
        dataset_name="kmeans251_rowkfold_audit",
        n_folds=5,
        seed=42,
        variance_threshold=0.01,
        correlation_threshold=0.95,
        unit_scale=KJMOL_TO_KCAL,
        group_col=None,
    )


if __name__ == "__main__":
    main()
