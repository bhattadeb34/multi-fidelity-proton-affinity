#!/usr/bin/env python3
"""Verify warning-free preprocessing reproduces stored production features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from train_models import NON_FEATURE_COLS, select_features


PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = next((p for p in (PROJECT / "data", PROJECT.parent / "data") if p.exists()), PROJECT / "data")
TARGETS = DATA_ROOT / "targets"
RESULTS = PROJECT / "results"
LABEL_COLS = {
    "exp_pa_kjmol", "exp_pa_kcalmol", "dft_pa_kjmol", "dft_pa_kcalmol",
    "pm7_pa_kjmol", "pm7_pa_kcalmol", "pm7_best_pa_kjmol",
    "pm7_best_pa_kcalmol",
}


def audit_one(name: str) -> dict:
    if name == "nist1155":
        df = pd.read_parquet(TARGETS / "nist1155_ml.parquet")
        target, baseline, truth, group = (
            "delta_pm7_exp", "pm7_best_pa_kjmol", "exp_pa_kjmol", None
        )
    else:
        df = pd.read_parquet(TARGETS / "kmeans251_ml.parquet")
        target, baseline, truth, group = (
            "delta_dft_pm7", "pm7_pa_kjmol", "dft_pa_kjmol", "record_id"
        )

    features = [
        col for col in df.columns
        if col not in NON_FEATURE_COLS
        and col not in {target, baseline, truth}
        and col not in LABEL_COLS
        and not col.startswith("delta_")
        and not col.startswith("raw_")
    ]
    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    if group:
        splits = GroupKFold(5).split(X, y, df[group].to_numpy())
        strategy = "GroupKFold(record_id)"
    else:
        splits = KFold(5, shuffle=True, random_state=42).split(X)
        strategy = "KFold(shuffle=True, random_state=42)"

    stored = json.loads((RESULTS / name / "cv_results.json").read_text())
    folds = []
    for fold, (train_idx, _) in enumerate(splits, start=1):
        _, selected = select_features(X[train_idx], y[train_idx], features)
        expected = stored["selected_features_per_fold"][fold - 1]
        folds.append({
            "fold": fold,
            "stored_feature_count": len(expected),
            "recomputed_feature_count": len(selected),
            "ordered_feature_lists_identical": selected == expected,
        })
    return {
        "dataset": name,
        "cv_strategy": strategy,
        "feature_pool_size": len(features),
        "folds": folds,
        "all_folds_identical": all(x["ordered_feature_lists_identical"] for x in folds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "nist1155", "kmeans251"],
                        default="all")
    parser.add_argument("--output", type=Path,
                        default=RESULTS / "preprocessing_equivalence_audit.json")
    args = parser.parse_args()
    names = ["nist1155", "kmeans251"] if args.dataset == "all" else [args.dataset]
    report = {
        "change_tested": (
            "explicit removal of all-missing/non-finite columns before median "
            "imputation and variance filtering"
        ),
        "datasets": [audit_one(name) for name in names],
    }
    report["valid"] = all(x["all_folds_identical"] for x in report["datasets"])
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
