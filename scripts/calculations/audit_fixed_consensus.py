#!/usr/bin/env python3
"""Post-hoc grouped-CV audit of the fixed deployment consensus feature set."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent.parent
KJMOL_TO_KCAL = 1 / 4.184


def main() -> None:
    data_root = next((p for p in (PROJECT / "data", PROJECT.parent / "data") if p.exists()), PROJECT / "data")
    target_path = data_root / "targets" / "kmeans251_ml.parquet"
    cv_path = PROJECT / "results" / "kmeans251" / "cv_results.json"
    output_path = PROJECT / "results" / "kmeans251" / "fixed_consensus_cv_audit.json"

    df = pd.read_parquet(target_path)
    cv_results = json.loads(cv_path.read_text())
    counts = Counter(
        feature
        for fold_features in cv_results["selected_features_per_fold"]
        for feature in fold_features
    )
    features = sorted(feature for feature, count in counts.items() if count >= 3)

    x = df[features].to_numpy(dtype=float)
    y = df["delta_dft_pm7"].to_numpy(dtype=float)
    pm7 = df["pm7_pa_kjmol"].to_numpy(dtype=float)
    dft = df["dft_pa_kjmol"].to_numpy(dtype=float)
    groups = df["record_id"].to_numpy()

    prediction_rows = []
    fold_mae = []
    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(5).split(x, y, groups), start=1
    ):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise AssertionError(f"Molecule leakage detected in fold {fold}")

        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(x[train_idx])
        x_test = imputer.transform(x[test_idx])
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y[train_idx])
        predicted_delta = model.predict(x_test)
        predicted_pa = pm7[test_idx] + predicted_delta
        fold_mae.append(
            mean_absolute_error(dft[test_idx], predicted_pa) * KJMOL_TO_KCAL
        )

        for position, row_idx in enumerate(test_idx):
            prediction_rows.append({
                "record_id": groups[row_idx],
                "pa_true": dft[row_idx],
                "pa_pred": predicted_pa[position],
                "pa_pm7": pm7[row_idx],
            })

    predictions = pd.DataFrame(prediction_rows)
    molecule_true = predictions.groupby("record_id")["pa_true"].max()
    molecule_pred = predictions.groupby("record_id")["pa_pred"].max()
    molecule_pm7 = predictions.groupby("record_id")["pa_pm7"].max()

    result = {
        "feature_specification": (
            "fixed global consensus selected in >=3/5 prior GroupKFold folds"
        ),
        "n_features": len(features),
        "features": features,
        "site_mae_folds_kcalmol": fold_mae,
        "site_mae_mean_kcalmol": float(np.mean(fold_mae)),
        "site_mae_std_kcalmol": float(np.std(fold_mae)),
        "molecule_max_mae_kcalmol": float(
            (molecule_pred - molecule_true).abs().mean() * KJMOL_TO_KCAL
        ),
        "molecule_max_bias_kcalmol": float(
            (molecule_pred - molecule_true).mean() * KJMOL_TO_KCAL
        ),
        "pm7_molecule_max_mae_kcalmol": float(
            (molecule_pm7 - molecule_true).abs().mean() * KJMOL_TO_KCAL
        ),
        "caveat": (
            "Post-hoc diagnostic: the consensus feature list was derived using "
            "all five outer folds, so this is not an unbiased nested performance estimate."
        ),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
