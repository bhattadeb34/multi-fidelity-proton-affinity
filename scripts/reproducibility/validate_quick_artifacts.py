#!/usr/bin/env python3
"""Validate quick artifacts, archived CV metrics, and optional model checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from artifact_store import ReleaseLayout, sha256


KJMOL_TO_KCAL = 1 / 4.184


def resolve_manifest_path(value: str, layout: ReleaseLayout) -> Path:
    if value.startswith("external:"):
        return Path(value.removeprefix("external:"))
    return layout.release / value


def validate_manifest(layout: ReleaseLayout) -> list[str]:
    manifest_path = layout.artifacts / "artifact_manifest.json"
    if not manifest_path.exists():
        return [f"Missing {manifest_path}"]
    manifest = json.loads(manifest_path.read_text())
    failures: list[str] = []
    for record in manifest.get("files", []):
        path = resolve_manifest_path(record["path"], layout)
        if not path.exists():
            failures.append(f"Missing artifact: {path}")
        elif sha256(path) != record["sha256"]:
            failures.append(f"Artifact hash mismatch: {path}")
        if "source_path" in record:
            source = resolve_manifest_path(record["source_path"], layout)
            if not source.exists():
                failures.append(f"Missing indexed source: {source}")
            elif sha256(source) != record["source_sha256"]:
                failures.append(f"Source changed after artifact export: {source}")
    return failures


def validate_cv_bundles(layout: ReleaseLayout, tolerance: float = 1e-10) -> list[str]:
    root = layout.artifacts / "plot_inputs" / "results"
    failures: list[str] = []
    for cv_path in sorted(root.glob("*/cv_results.json")):
        result_dir = cv_path.parent
        prediction_path = result_dir / "predictions.csv"
        summary_path = result_dir / "mae_summary.csv"
        if not prediction_path.exists() or not summary_path.exists():
            continue
        cv = json.loads(cv_path.read_text())
        predictions = pd.read_csv(prediction_path)
        summary = pd.read_csv(summary_path).set_index("model")
        for model, metrics in cv.get("models", {}).items():
            subset = predictions[predictions["model"] == model]
            if subset.empty:
                if model != "PM7+bias":
                    failures.append(f"{result_dir.name}/{model}: predictions absent")
                continue
            evaluated = subset.assign(
                delta_absolute_error=np.abs(subset["y_true_delta"] - subset["y_pred_delta"]),
                pa_absolute_error=np.abs(subset["pa_true"] - subset["pa_pred"]),
            )
            delta_fold = (
                evaluated.groupby("fold")["delta_absolute_error"].mean().to_numpy()
                * KJMOL_TO_KCAL
            )
            pa_fold = (
                evaluated.groupby("fold")["pa_absolute_error"].mean().to_numpy()
                * KJMOL_TO_KCAL
            )
            expected_delta = np.asarray(metrics["mae_delta_per_fold"], dtype=float)
            expected_pa = np.asarray(metrics["mae_pa_per_fold"], dtype=float)
            if not np.allclose(delta_fold, expected_delta, rtol=0, atol=tolerance):
                failures.append(f"{result_dir.name}/{model}: delta fold MAE mismatch")
            if not np.allclose(pa_fold, expected_pa, rtol=0, atol=tolerance):
                failures.append(f"{result_dir.name}/{model}: PA fold MAE mismatch")
            if model in summary.index:
                row = summary.loc[model]
                if not np.isclose(delta_fold.mean(), row["mae_delta_mean"], atol=5e-5):
                    failures.append(f"{result_dir.name}/{model}: summary delta MAE mismatch")
                if not np.isclose(pa_fold.mean(), row["mae_pa_mean"], atol=5e-5):
                    failures.append(f"{result_dir.name}/{model}: summary PA MAE mismatch")
    return failures


def transformed_rows(
    table: pd.DataFrame,
    features: list[str],
    rows: np.ndarray,
    medians: np.ndarray,
) -> np.ndarray:
    values = table.iloc[rows][features].to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    missing_rows, missing_cols = np.where(np.isnan(values))
    values[missing_rows, missing_cols] = medians[missing_cols]
    return values


def validate_checkpoints(layout: ReleaseLayout, tolerance: float = 1e-10) -> list[str]:
    root = layout.artifacts / "model_checkpoints"
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        print("Model checkpoints are not present. Their validation was skipped.")
        return []
    failures: list[str] = []
    manifest = json.loads(manifest_path.read_text())
    for dataset_record in manifest["datasets"]:
        dataset = dataset_record["dataset"]
        if dataset == "screening":
            iteration = int(dataset_record["iteration"])
            bundle = root / "screening" / "RandomForest"
            provenance = json.loads((bundle / "provenance.json").read_text())
            model_path = bundle / "model.joblib"
            preprocessing_path = bundle / "preprocessing.npz"
            feature_path = (
                layout.artifacts / "features" / f"screening_iter{iteration}_consensus.parquet"
            )
            prediction_path = (
                layout.artifacts
                / "plot_inputs"
                / "screening"
                / f"iter{iteration}"
                / "predictions.parquet"
            )
            if sha256(model_path) != provenance["model_sha256"]:
                failures.append(f"Checkpoint hash mismatch: {model_path}")
                continue
            if sha256(preprocessing_path) != provenance["preprocessing_sha256"]:
                failures.append(f"Preprocessing hash mismatch: {preprocessing_path}")
                continue
            table = pd.read_parquet(feature_path)
            archived = pd.read_parquet(prediction_path)
            state = np.load(preprocessing_path)
            features = state["feature_names"].astype(str).tolist()
            medians = state["training_medians"]
            estimator = joblib.load(model_path)
            rows = np.arange(len(table), dtype=int)
            observed = estimator.predict(transformed_rows(table, features, rows, medians))
            if not np.allclose(
                observed,
                archived["delta_pred_kcalmol"].to_numpy(dtype=float),
                rtol=0,
                atol=tolerance,
            ):
                failures.append("screening/RandomForest: prediction mismatch")
            continue
        model_name = dataset_record["model"]
        feature_path = layout.artifacts / "features" / f"{dataset}_selected_union.parquet"
        prediction_path = layout.artifacts / "plot_inputs" / "results" / dataset / "predictions.csv"
        table = pd.read_parquet(feature_path)
        predictions = pd.read_csv(prediction_path)
        expected_model = predictions[predictions["model"] == model_name]
        bundle = root / dataset / model_name
        for fold_dir in sorted(bundle.glob("fold_*")):
            report_path = fold_dir / "validation.json"
            report = json.loads(report_path.read_text())
            model_path = fold_dir / "model.joblib"
            preprocessing_path = fold_dir / "preprocessing.npz"
            if sha256(model_path) != report["model_sha256"]:
                failures.append(f"Checkpoint hash mismatch: {model_path}")
                continue
            if sha256(preprocessing_path) != report["preprocessing_sha256"]:
                failures.append(f"Preprocessing hash mismatch: {preprocessing_path}")
                continue
            state = np.load(preprocessing_path)
            features = state["feature_names"].astype(str).tolist()
            medians = state["training_medians"]
            test_idx = state["test_indices"].astype(int)
            estimator = joblib.load(model_path)
            observed = estimator.predict(transformed_rows(table, features, test_idx, medians))
            fold = int(report["fold"])
            wanted = expected_model[expected_model["fold"] == fold].sort_values("sample_idx")
            if not np.array_equal(test_idx, wanted["sample_idx"].to_numpy(dtype=int)):
                failures.append(f"{dataset}/{model_name}/fold {fold}: test indices changed")
                continue
            if not np.allclose(observed, wanted["y_pred_delta"], rtol=0, atol=tolerance):
                failures.append(f"{dataset}/{model_name}/fold {fold}: prediction mismatch")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    layout = ReleaseLayout.discover(args.code_root, args.artifact_root, args.data_root)
    failures = []
    failures.extend(validate_manifest(layout))
    failures.extend(validate_cv_bundles(layout, args.tolerance))
    failures.extend(validate_checkpoints(layout, args.tolerance))
    if failures:
        print("Artifact validation failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Artifact hashes and archived CV result bundles are internally consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
