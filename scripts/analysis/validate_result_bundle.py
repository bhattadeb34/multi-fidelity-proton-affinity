#!/usr/bin/env python3
"""Validate that CV JSON, predictions, and summary CSV form one result run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold


KJMOL_TO_KCAL = 1 / 4.184
PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = next((p for p in (PROJECT / "data", PROJECT.parent / "data") if p.exists()), PROJECT / "data")


def close(a: float, b: float, tol: float = 5e-4) -> bool:
    return bool(np.isclose(float(a), float(b), atol=tol, rtol=0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("--write-report", action="store_true")
    ap.add_argument("--group-col", default=None,
                    help="Validation override for legacy JSON lacking split metadata")
    args = ap.parse_args()
    result_dir = args.result_dir.resolve()

    cv = json.loads((result_dir / "cv_results.json").read_text())
    predictions = pd.read_csv(result_dir / "predictions.csv")
    summary = pd.read_csv(result_dir / "mae_summary.csv")
    failures: list[str] = []
    checks: dict[str, object] = {}

    json_models = set(cv["models"])
    csv_models = set(summary["model"])
    pred_models = set(predictions["model"])
    expected_pred_models = json_models - {"PM7+bias"}
    if json_models != csv_models or expected_pred_models != pred_models:
        failures.append("model sets differ across JSON/summary/predictions")

    n_samples_by_model = predictions.groupby("model")["sample_idx"].nunique()
    checks["n_samples_by_model"] = n_samples_by_model.to_dict()
    if n_samples_by_model.nunique() != 1:
        failures.append("models do not contain the same number of held-out samples")
    if predictions.duplicated(["model", "sample_idx"]).any():
        failures.append("a model predicts at least one sample more than once")

    recomputed = {}
    for model in sorted(json_models & pred_models):
        model_df = predictions[predictions["model"] == model]
        fold_delta = []
        fold_pa = []
        for fold in sorted(model_df["fold"].unique()):
            f = model_df[model_df["fold"] == fold]
            fold_delta.append(
                mean_absolute_error(f["y_true_delta"], f["y_pred_delta"])
                * KJMOL_TO_KCAL
            )
            fold_pa.append(
                mean_absolute_error(f["pa_true"], f["pa_pred"])
                * KJMOL_TO_KCAL
            )
        stored = cv["models"][model]
        if not np.allclose(fold_delta, stored["mae_delta_per_fold"], atol=5e-10):
            failures.append(f"{model}: per-fold delta MAEs disagree")
        if not np.allclose(fold_pa, stored["mae_pa_per_fold"], atol=5e-10):
            failures.append(f"{model}: per-fold PA MAEs disagree")
        row = summary[summary["model"] == model].iloc[0]
        for col, value in [
            ("mae_delta_mean", np.mean(fold_delta)),
            ("mae_delta_std", np.std(fold_delta)),
            ("mae_pa_mean", np.mean(fold_pa)),
            ("mae_pa_std", np.std(fold_pa)),
        ]:
            if not close(row[col], value):
                failures.append(f"{model}: {col} disagrees with predictions")
        recomputed[model] = {
            "mae_delta_per_fold": fold_delta,
            "mae_pa_per_fold": fold_pa,
            "mae_pa_mean": float(np.mean(fold_pa)),
        }

    feature_counts = [len(x) for x in cv["selected_features_per_fold"]]
    checks["selected_feature_counts"] = feature_counts
    for model, stored in cv["models"].items():
        if model == "PM7+bias":
            continue
        if stored.get("n_features_mean") is not None and not close(
            stored["n_features_mean"], np.mean(feature_counts), tol=1e-8
        ):
            failures.append(f"{model}: feature-count mean disagrees with fold lists")

    # Verify molecule isolation directly from the stored held-out predictions.
    # GroupKFold's exact assignment of equally sized groups can differ across
    # scikit-learn versions, so reconstructed fold identity is informative but
    # is not a portable correctness condition. The recorded clean environment
    # is used when exact fold reproduction is required.
    group_col = cv.get("group_col") or args.group_col
    if group_col:
        target = pd.read_parquet(
            DATA_ROOT / "targets" / "kmeans251_ml.parquet",
            columns=[group_col],
        )
        groups = target[group_col].to_numpy()
        expected_fold = np.empty(len(target), dtype=int)
        for fold, (_, test_idx) in enumerate(
            GroupKFold(5).split(np.zeros(len(target)), groups=groups), start=1
        ):
            expected_fold[test_idx] = fold
        first_model = sorted(pred_models)[0]
        first = predictions[predictions["model"] == first_model].copy()
        if group_col not in first.columns:
            failures.append(f"predictions are missing declared group column {group_col!r}")
        else:
            folds_per_group = first.groupby(group_col)["fold"].nunique()
            checks["groups_spanning_multiple_folds"] = int((folds_per_group > 1).sum())
            if (folds_per_group > 1).any():
                failures.append("at least one molecule occurs in multiple held-out folds")
            target_groups = set(target[group_col].astype(str))
            predicted_groups = set(first[group_col].astype(str))
            checks["group_coverage_matches_target"] = predicted_groups == target_groups
            if predicted_groups != target_groups:
                failures.append("held-out prediction groups do not match target groups")

        observed = (first.set_index("sample_idx")["fold"].sort_index().to_numpy())
        checks["reconstructed_groupkfold_exact_match"] = bool(
            np.array_equal(observed, expected_fold)
        )
        checks["group_count"] = int(pd.Series(groups).nunique())

    try:
        report_path = str(result_dir.relative_to(PROJECT))
    except ValueError:
        report_path = str(result_dir)

    report = {
        "result_dir": report_path,
        "dataset": cv.get("dataset"),
        "cv_strategy": cv.get("cv_strategy", "legacy metadata absent"),
        "group_col": group_col,
        "checks": checks,
        "recomputed": recomputed,
        "failures": failures,
        "valid": not failures,
    }
    if args.write_report:
        (result_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
