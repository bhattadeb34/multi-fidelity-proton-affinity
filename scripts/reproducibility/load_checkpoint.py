#!/usr/bin/env python3
"""Load a verified model checkpoint and predict from a feature table.

The input table may contain additional columns.  Features are selected and
ordered from ``preprocessing.npz``.  Missing values are filled with the saved
training-fold medians before prediction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from artifact_store import sha256


@dataclass
class LoadedCheckpoint:
    """A fitted estimator paired with its immutable preprocessing state."""

    estimator: object
    feature_names: list[str]
    training_medians: np.ndarray

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(self.feature_names) - set(table.columns))
        if missing:
            raise ValueError(f"Input table is missing checkpoint features: {missing[:10]}")
        values = table[self.feature_names].to_numpy(dtype=float, copy=True)
        values[~np.isfinite(values)] = np.nan
        missing_rows, missing_cols = np.where(np.isnan(values))
        values[missing_rows, missing_cols] = self.training_medians[missing_cols]
        return np.asarray(self.estimator.predict(values), dtype=float)


def load_checkpoint(directory: Path, verify_hashes: bool = True) -> LoadedCheckpoint:
    """Load one fold or screening checkpoint and optionally verify its hashes."""
    directory = directory.expanduser().resolve()
    model_path = directory / "model.joblib"
    preprocessing_path = directory / "preprocessing.npz"
    provenance_candidates = [directory / "validation.json", directory / "provenance.json"]
    provenance_path = next((path for path in provenance_candidates if path.exists()), None)
    if not model_path.exists() or not preprocessing_path.exists():
        raise FileNotFoundError(f"Incomplete checkpoint directory: {directory}")
    if verify_hashes:
        if provenance_path is None:
            raise FileNotFoundError(f"Checkpoint provenance is absent: {directory}")
        provenance = json.loads(provenance_path.read_text())
        if sha256(model_path) != provenance["model_sha256"]:
            raise ValueError(f"Model checksum does not match provenance: {model_path}")
        if sha256(preprocessing_path) != provenance["preprocessing_sha256"]:
            raise ValueError(
                f"Preprocessing checksum does not match provenance: {preprocessing_path}"
            )
    state = np.load(preprocessing_path)
    return LoadedCheckpoint(
        estimator=joblib.load(model_path),
        feature_names=state["feature_names"].astype(str).tolist(),
        training_medians=state["training_medians"].astype(float),
    )


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError("Input must be a Parquet or CSV table")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prediction-column", default="checkpoint_prediction")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to replace {args.output}. Pass --force.")
    table = read_table(args.input)
    checkpoint = load_checkpoint(args.checkpoint)
    output = table.copy()
    output[args.prediction_column] = checkpoint.predict(table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".parquet":
        output.to_parquet(args.output, index=False)
    elif args.output.suffix.lower() == ".csv":
        output.to_csv(args.output, index=False)
    else:
        raise ValueError("Output must use .parquet or .csv")
    print(f"Wrote {len(output)} predictions to {args.output}")


if __name__ == "__main__":
    main()
