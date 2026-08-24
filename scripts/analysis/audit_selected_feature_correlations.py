#!/usr/bin/env python3
"""Measure residual correlation among fold-selected production features.

For each outer fold, correlations are evaluated only on that fold's training
rows after training-median imputation. The selected feature names and split
definitions come from the validated production result bundles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = next((p for p in (PROJECT_DIR / "data", PROJECT_DIR.parent / "data") if p.exists()), PROJECT_DIR / "data")
DATASETS = {
    "nist1155": {
        "target": DATA_DIR / "targets" / "nist1155_ml.parquet",
        "results": PROJECT_DIR / "results" / "nist1155" / "cv_results.json",
        "group_col": None,
    },
    "kmeans251": {
        "target": DATA_DIR / "targets" / "kmeans251_ml.parquet",
        "results": PROJECT_DIR / "results" / "kmeans251" / "cv_results.json",
        "group_col": "record_id",
    },
}


def correlation_summary(values: np.ndarray) -> dict[str, float | int]:
    """Return summary statistics for one imputed training-fold matrix."""
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.abs(np.corrcoef(values, rowvar=False))
    pairs = correlation[np.triu_indices(correlation.shape[0], k=1)]
    pairs = pairs[np.isfinite(pairs)]
    return {
        "n_pairs": int(len(pairs)),
        "median_abs_r": float(np.median(pairs)),
        "p95_abs_r": float(np.quantile(pairs, 0.95)),
        "max_abs_r": float(np.max(pairs)),
        "fraction_abs_r_gt_0_5": float(np.mean(pairs > 0.5)),
        "fraction_abs_r_gt_0_8": float(np.mean(pairs > 0.8)),
        "fraction_abs_r_gt_0_95": float(np.mean(pairs > 0.95)),
    }


def audit_dataset(name: str, specification: dict) -> dict:
    """Reconstruct outer folds and summarize their selected features."""
    data = pd.read_parquet(specification["target"])
    cv = json.loads(specification["results"].read_text())
    selected_by_fold = cv["selected_features_per_fold"]
    group_col = specification["group_col"]

    if group_col:
        splitter = GroupKFold(n_splits=5).split(
            data,
            groups=data[group_col].to_numpy(),
        )
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42).split(data)

    folds = []
    for fold_number, ((train_index, _), selected) in enumerate(
        zip(splitter, selected_by_fold),
        start=1,
    ):
        missing = sorted(set(selected) - set(data.columns))
        if missing:
            raise ValueError(f"{name} fold {fold_number} missing features: {missing}")

        train = data.iloc[train_index][selected].astype(float)
        train = train.fillna(train.median(axis=0))
        if train.isna().any().any():
            raise ValueError(f"{name} fold {fold_number} has all-NaN selected features")

        summary = correlation_summary(train.to_numpy())
        folds.append(
            {
                "fold": fold_number,
                "n_features": len(selected),
                **summary,
            }
        )

    aggregate_keys = [
        "median_abs_r",
        "p95_abs_r",
        "max_abs_r",
        "fraction_abs_r_gt_0_5",
        "fraction_abs_r_gt_0_8",
        "fraction_abs_r_gt_0_95",
    ]
    return {
        "dataset": name,
        "definition": (
            "Absolute Pearson correlations among fold-selected features, "
            "evaluated on that fold's training rows after training-median imputation."
        ),
        "folds": folds,
        "aggregate": {
            key: float(np.mean([fold[key] for fold in folds]))
            for key in aggregate_keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "results" / "selected_feature_correlation_audit.json",
    )
    args = parser.parse_args()

    report = {
        name: audit_dataset(name, specification)
        for name, specification in DATASETS.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
