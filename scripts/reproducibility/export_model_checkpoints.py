#!/usr/bin/env python3
"""Export verified fold models without rerunning feature selection.

The selected feature names and held-out membership are read from the archived
CV result bundle.  Each estimator is fitted on the complementary training rows
from the compact selected-feature table.  Export succeeds only if its held-out
predictions reproduce ``predictions.csv`` within the requested tolerance.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from artifact_store import ReleaseLayout, file_record, runtime_record, sha256


SCRIPT_DIR = Path(__file__).resolve().parent
CALCULATION_DIR = SCRIPT_DIR.parent / "calculations"
if str(CALCULATION_DIR) not in sys.path:
    sys.path.insert(0, str(CALCULATION_DIR))

from model_factory import ModelFactory  # noqa: E402


DATASET_TARGETS = {
    "nist1155": "delta_pm7_exp",
    "kmeans251": "delta_dft_pm7",
}

EXCLUDED_MODELS = {"PM7+bias", "VotingEnsemble"}
FACTORY = ModelFactory(random_state=42, n_estimators=200, gpr_max_samples=500)


def best_model(cv: dict[str, object]) -> str:
    models = cv["models"]
    eligible = {
        name: metrics
        for name, metrics in models.items()
        if name not in EXCLUDED_MODELS and metrics.get("mae_pa_mean") is not None
    }
    if not eligible:
        raise ValueError("CV bundle has no eligible fitted model")
    return min(eligible, key=lambda name: eligible[name]["mae_pa_mean"])


def training_medians(table: pd.DataFrame, features: list[str], train_idx: np.ndarray) -> np.ndarray:
    values = table.iloc[train_idx][features].to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    medians = np.nanmedian(values, axis=0)
    if np.isnan(medians).any():
        bad = [name for name, value in zip(features, medians) if np.isnan(value)]
        raise ValueError(f"Selected features are all-missing in training rows: {bad}")
    return medians


def transform(table: pd.DataFrame, features: list[str], rows: np.ndarray, medians: np.ndarray) -> np.ndarray:
    values = table.iloc[rows][features].to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    missing_rows, missing_cols = np.where(np.isnan(values))
    values[missing_rows, missing_cols] = medians[missing_cols]
    return values


def export_dataset(
    layout: ReleaseLayout,
    dataset: str,
    tolerance: float,
    force: bool,
) -> dict[str, object]:
    result_dir = layout.code / "results" / dataset
    cv_path = result_dir / "cv_results.json"
    prediction_path = result_dir / "predictions.csv"
    feature_path = layout.artifacts / "features" / f"{dataset}_selected_union.parquet"
    required = [cv_path, prediction_path, feature_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint input. Run build_quick_artifacts.py first. "
            + ", ".join(missing)
        )

    cv = json.loads(cv_path.read_text())
    model_name = best_model(cv)
    predictions = pd.read_csv(prediction_path)
    archived = predictions[predictions["model"] == model_name].copy()
    table = pd.read_parquet(feature_path)
    if len(table) != archived["sample_idx"].nunique():
        raise ValueError(
            f"{dataset}: feature rows ({len(table)}) do not match unique archived "
            f"sample indices ({archived['sample_idx'].nunique()})"
        )

    destination = layout.artifacts / "model_checkpoints" / dataset / model_name
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to replace {destination}. Pass --force.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{dataset}-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        fold_reports: list[dict[str, object]] = []
        all_indices = np.arange(len(table), dtype=int)
        selected_by_fold = cv["selected_features_per_fold"]

        for fold_number, features in enumerate(selected_by_fold, start=1):
            expected = archived[archived["fold"] == fold_number].sort_values("sample_idx")
            test_idx = expected["sample_idx"].to_numpy(dtype=int)
            train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=True)
            medians = training_medians(table, features, train_idx)
            X_train = transform(table, features, train_idx, medians)
            X_test = transform(table, features, test_idx, medians)
            y_train = table.iloc[train_idx][DATASET_TARGETS[dataset]].to_numpy(dtype=float)

            models = FACTORY.build(n_train=len(train_idx))
            if model_name not in models:
                raise RuntimeError(f"Required model {model_name!r} is unavailable")
            estimator = models[model_name]
            estimator.fit(X_train, y_train)
            observed = np.asarray(estimator.predict(X_test), dtype=float)
            wanted = expected["y_pred_delta"].to_numpy(dtype=float)
            absolute_error = np.abs(observed - wanted)
            maximum_error = float(absolute_error.max(initial=0.0))
            mean_error = float(absolute_error.mean()) if len(absolute_error) else 0.0
            if not np.allclose(observed, wanted, rtol=0.0, atol=tolerance):
                raise ValueError(
                    f"{dataset} {model_name} fold {fold_number} does not reproduce "
                    f"archived predictions. max_abs_error={maximum_error:.12g}, "
                    f"tolerance={tolerance:.12g}"
                )

            fold_dir = staging / f"fold_{fold_number:02d}"
            fold_dir.mkdir()
            model_path = fold_dir / "model.joblib"
            preprocessing_path = fold_dir / "preprocessing.npz"
            joblib.dump(estimator, model_path, compress=3)
            np.savez_compressed(
                preprocessing_path,
                feature_names=np.asarray(features, dtype=str),
                training_medians=medians,
                train_indices=train_idx,
                test_indices=test_idx,
            )
            fold_report = {
                "fold": fold_number,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_features": len(features),
                "max_abs_prediction_difference": maximum_error,
                "mean_abs_prediction_difference": mean_error,
                "absolute_tolerance": tolerance,
                "verified": True,
                "model_sha256": sha256(model_path),
                "preprocessing_sha256": sha256(preprocessing_path),
            }
            (fold_dir / "validation.json").write_text(json.dumps(fold_report, indent=2) + "\n")
            fold_reports.append(fold_report)

        provenance = {
            "dataset": dataset,
            "model": model_name,
            "target_column": DATASET_TARGETS[dataset],
            "fold_source": "archived predictions.csv sample_idx and fold columns",
            "feature_source": "archived cv_results.json selected_features_per_fold",
            "feature_table_sha256": sha256(feature_path),
            "cv_results_sha256": sha256(cv_path),
            "predictions_sha256": sha256(prediction_path),
            "runtime": runtime_record(
                ["numpy", "pandas", "pyarrow", "scikit-learn", "joblib"]
            ),
            "folds": fold_reports,
            "all_folds_verified": True,
        }
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(staging, destination)

    return {
        "dataset": dataset,
        "model": model_name,
        "path": str(destination.relative_to(layout.release)),
        "all_folds_verified": True,
        "folds": fold_reports,
    }


def export_screening(
    layout: ReleaseLayout,
    iteration: int,
    tolerance: float,
    force: bool,
) -> dict[str, object]:
    """Export the full-data screening model and verify every archived site prediction."""
    training_path = layout.artifacts / "features" / "kmeans251_selected_union.parquet"
    screening_path = layout.artifacts / "features" / f"screening_iter{iteration}_consensus.parquet"
    feature_metadata_path = screening_path.with_suffix(".json")
    prediction_path = (
        layout.artifacts
        / "plot_inputs"
        / "screening"
        / f"iter{iteration}"
        / "predictions.parquet"
    )
    config_path = layout.code / "screening" / "config" / "pipeline_config.json"
    required = [training_path, screening_path, feature_metadata_path, prediction_path, config_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing screening checkpoint input. Rebuild quick artifacts first. "
            + ", ".join(missing)
        )
    feature_metadata = json.loads(feature_metadata_path.read_text())
    features = feature_metadata["consensus_features_minimum_3_folds"]
    config = json.loads(config_path.read_text())
    model_config = config["random_forest"]
    training = pd.read_parquet(training_path)
    screening = pd.read_parquet(screening_path)
    archived = pd.read_parquet(prediction_path)
    if len(screening) != len(archived):
        raise ValueError(
            f"Screening feature rows ({len(screening)}) do not match archived "
            f"prediction rows ({len(archived)})"
        )
    identity_columns = [
        name
        for name in ["mol_id", "smiles", "protonated_smiles", "site_idx"]
        if name in screening.columns and name in archived.columns
    ]
    for name in identity_columns:
        if not screening[name].fillna("").astype(str).equals(
            archived[name].fillna("").astype(str)
        ):
            raise ValueError(f"Screening row identity differs in column {name!r}")

    train_idx = np.arange(len(training), dtype=int)
    medians = training_medians(training, features, train_idx)
    X_train = transform(training, features, train_idx, medians)
    X_screen = transform(screening, features, np.arange(len(screening)), medians)
    # Match the deployed screening script's operation order exactly. Although
    # algebraically equivalent, subtracting in kJ before conversion can differ
    # by a few floating-point ulps and may change tree split tie-breaking.
    conversion = 1 / 4.184
    y_train = (
        training["dft_pa_kjmol"].to_numpy(dtype=float) * conversion
        - training["pm7_pa_kjmol"].to_numpy(dtype=float) * conversion
    )
    estimator = RandomForestRegressor(
        n_estimators=model_config["n_estimators"],
        random_state=model_config["random_state"],
        n_jobs=model_config["n_jobs"],
    )
    estimator.fit(X_train, y_train)
    observed = estimator.predict(X_screen)
    wanted = archived["delta_pred_kcalmol"].to_numpy(dtype=float)
    difference = np.abs(observed - wanted)
    maximum_error = float(difference.max(initial=0.0))
    mean_error = float(difference.mean()) if len(difference) else 0.0
    if not np.allclose(observed, wanted, rtol=0.0, atol=tolerance):
        raise ValueError(
            "Screening RandomForest does not reproduce archived site predictions. "
            f"max_abs_error={maximum_error:.12g}, tolerance={tolerance:.12g}"
        )

    destination = layout.artifacts / "model_checkpoints" / "screening" / "RandomForest"
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to replace {destination}. Pass --force.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".screening-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        model_path = staging / "model.joblib"
        preprocessing_path = staging / "preprocessing.npz"
        joblib.dump(estimator, model_path, compress=3)
        np.savez_compressed(
            preprocessing_path,
            feature_names=np.asarray(features, dtype=str),
            training_medians=medians,
        )
        report = {
            "dataset": "screening",
            "iteration": iteration,
            "model": "RandomForest",
            "n_train": len(training),
            "n_screening_sites": len(screening),
            "n_features": len(features),
            "max_abs_prediction_difference": maximum_error,
            "mean_abs_prediction_difference": mean_error,
            "absolute_tolerance": tolerance,
            "verified": True,
            "training_feature_table_sha256": sha256(training_path),
            "screening_feature_table_sha256": sha256(screening_path),
            "predictions_sha256": sha256(prediction_path),
            "config_sha256": sha256(config_path),
            "model_sha256": sha256(model_path),
            "preprocessing_sha256": sha256(preprocessing_path),
            "runtime": runtime_record(
                ["numpy", "pandas", "pyarrow", "scikit-learn", "joblib"]
            ),
        }
        (staging / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(staging, destination)
    return report | {"path": str(destination.relative_to(layout.release))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASET_TARGETS, "screening"],
        default="all",
    )
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    layout = ReleaseLayout.discover(args.code_root, args.artifact_root, args.data_root)
    datasets = list(DATASET_TARGETS) if args.dataset == "all" else [args.dataset]
    reports = [
        export_dataset(layout, dataset, args.tolerance, args.force)
        for dataset in datasets
        if dataset in DATASET_TARGETS
    ]
    if args.dataset in {"all", "screening"}:
        reports.append(
            export_screening(layout, args.iteration, args.tolerance, args.force)
        )
    manifest = layout.artifacts / "model_checkpoints" / "manifest.json"
    existing = json.loads(manifest.read_text()).get("datasets", []) if manifest.exists() else []
    report_by_dataset = {record["dataset"]: record for record in existing}
    report_by_dataset.update({record["dataset"]: record for record in reports})
    manifest.write_text(
        json.dumps(
            {"schema_version": "1.0", "datasets": list(report_by_dataset.values())},
            indent=2,
        )
        + "\n"
    )
    print(f"Exported and verified {len(reports)} model checkpoint bundles")


if __name__ == "__main__":
    main()
