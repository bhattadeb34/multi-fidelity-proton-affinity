"""
05_predict_pa.py
================
Apply the trained RandomForest k-means model to predict PA corrections,
compute uncertainty, and select the best protonation site per molecule.

Consensus features are loaded programmatically from cv_results.json
(same source as 04_featurize.py) rather than hardcoded.

For each molecule:
  - PA_pred per site = PA_PM7 + Delta_ML
  - Uncertainty = std of predictions across 200 trees (heuristic ensemble dispersion, not calibrated)
  - Molecular PA_pred = max(PA_pred across sites)
  - PA_spread = PA_pred_max - PA_pred_second (amphoteric character)

Reads from:
    results/kmeans251/cv_results.json          (consensus feature source)
    data/screening/iter{N}/features.parquet
    data/targets/kmeans251_ml.parquet          (training data)

Writes to:
    data/screening/iter{N}/predictions.parquet  -- per-site predictions
    data/screening/iter{N}/molecular_pa.parquet -- per-molecule summary

Usage:
    python screening/scripts/execution/05_predict_pa.py --iter 1
"""

import argparse
import json
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from pipeline_config import DEFAULT_CONFIG_PATH, load_pipeline_config

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
SCREENING  = SCRIPT_DIR.parent.parent
PROJECT    = SCREENING.parent
DATA_ROOT  = next((p for p in (PROJECT / "data", PROJECT.parent / "data") if p.exists()), PROJECT / "data")
DATA_DIR   = DATA_ROOT / "screening"
RESULTS    = PROJECT / "results" / "kmeans251"
CONFIG     = load_pipeline_config()

KJMOL_TO_KCAL = 1 / 4.184


def load_consensus_features(threshold: int = 3) -> list[str]:
    cv_path = RESULTS / "cv_results.json"
    with open(cv_path) as f:
        data = json.load(f)
    counts = Counter()
    n_folds = len(data["selected_features_per_fold"])
    for fold_feats in data["selected_features_per_fold"]:
        counts.update(fold_feats)
    consensus = sorted([f for f, c in counts.items() if c >= threshold])
    log.info(f"Loaded {len(consensus)} consensus features "
             f"(>= {threshold}/{n_folds} folds) from {cv_path.name}")
    return consensus


# ---------------------------------------------------------------------------
# Load training data and retrain RandomForest on full k-means dataset
# ---------------------------------------------------------------------------

def load_training_data(consensus_features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load k-means training data and return X, y for the consensus features."""
    target_path = DATA_ROOT / "targets" / "kmeans251_ml.parquet"
    df = pd.read_parquet(target_path)

    df["pm7_pa_kcalmol"] = df["pm7_pa_kjmol"] * KJMOL_TO_KCAL
    df["dft_pa_kcalmol"] = df["dft_pa_kjmol"] * KJMOL_TO_KCAL
    df["correction_kcalmol"] = df["dft_pa_kcalmol"] - df["pm7_pa_kcalmol"]

    available = [f for f in consensus_features if f in df.columns]
    missing = [f for f in consensus_features if f not in df.columns]
    if missing:
        raise ValueError(
            f"Consensus features missing from training table: {missing}"
        )

    X = df[available].values.astype(np.float64)
    y = df["correction_kcalmol"].values.astype(np.float64)

    log.info(f"  Training data: {len(df):,} sites, {len(available)} features")
    return X, y, available


def train_model(X: np.ndarray, y: np.ndarray, model_config: dict) -> tuple:
    """Train RandomForest on full k-means dataset with imputation."""
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=model_config["n_estimators"],
        random_state=model_config["random_state"],
        n_jobs=model_config["n_jobs"],
    )
    log.info(
        "  Training RandomForest "
        f"(n={model_config['n_estimators']}) on full k-means dataset ..."
    )
    model.fit(X_imp, y)
    log.info("  Done.")
    return model, imputer


# ---------------------------------------------------------------------------
# Prediction with uncertainty
# ---------------------------------------------------------------------------

def predict_with_uncertainty(
    model: RandomForestRegressor,
    imputer: SimpleImputer,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict delta correction and uncertainty (std across trees).
    Returns (delta_pred, delta_std).
    """
    X_imp = imputer.transform(X)

    # Per-tree predictions for uncertainty
    tree_preds = np.array([
        tree.predict(X_imp) for tree in model.estimators_
    ])  # shape: (n_trees, n_samples)

    delta_pred = tree_preds.mean(axis=0)
    delta_std  = tree_preds.std(axis=0)
    return delta_pred, delta_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(iteration: int, threshold: int) -> None:
    model_config = CONFIG["random_forest"]
    pa_low = CONFIG["pa_window_kcalmol"]["low"]
    pa_high = CONFIG["pa_window_kcalmol"]["high"]
    log.info(f"Screening config: {DEFAULT_CONFIG_PATH}")
    iter_dir   = DATA_DIR / f"iter{iteration}"
    feat_path  = iter_dir / "features.parquet"
    pred_path  = iter_dir / "predictions.parquet"
    mol_path   = iter_dir / "molecular_pa.parquet"

    if not feat_path.exists():
        log.error(f"Features not found: {feat_path}")
        sys.exit(1)

    consensus_features = load_consensus_features(threshold)

    feat_df = pd.read_parquet(feat_path)
    log.info(f"Loaded {len(feat_df):,} site records")

    log.info("Loading k-means training data ...")
    X_train, y_train, train_features = load_training_data(consensus_features)
    model, imputer = train_model(X_train, y_train, model_config)

    # ── Prepare screening features ───────────────────────────────────────────
    # Use only features that were available in training
    avail = [f for f in train_features if f in feat_df.columns]
    missing = [f for f in train_features if f not in feat_df.columns]
    if missing:
        raise ValueError(
            f"Consensus features missing from screening table: {missing}"
        )

    X_screen = feat_df[train_features].values.astype(np.float64)
    log.info(f"  Screening features: {len(train_features)}, "
             f"available: {len(avail)}, missing: {len(missing)}")

    # ── Predict ──────────────────────────────────────────────────────────────
    log.info("Predicting PA corrections ...")
    delta_pred, delta_std = predict_with_uncertainty(model, imputer, X_screen)

    pa_pm7  = feat_df["pa_pm7_kcalmol"].values
    pa_pred = pa_pm7 + delta_pred

    # ── Build per-site predictions dataframe ─────────────────────────────────
    pred_df = feat_df[["mol_id", "smiles", "protonated_smiles",
                        "site_idx", "site_element", "site_index",
                        "site_normalized_index", "site_n_sites",
                        "pa_pm7_kcalmol"]].copy()
    pred_df["delta_pred_kcalmol"] = delta_pred
    pred_df["delta_std_kcalmol"]  = delta_std
    pred_df["pa_pred_kcalmol"]    = pa_pred

    pred_df.to_parquet(pred_path, index=False)
    log.info(f"Saved per-site predictions → {pred_path}")

    # ── Per-molecule summary ─────────────────────────────────────────────────
    log.info("Computing per-molecule PA summary ...")
    mol_records = []

    for smiles, group in pred_df.groupby("smiles"):
        group = group.sort_values("pa_pred_kcalmol", ascending=False)

        best_site     = group.iloc[0]
        pa_pred_max   = best_site["pa_pred_kcalmol"]
        uncertainty   = best_site["delta_std_kcalmol"]

        # PA spread -- difference between best and second best site
        if len(group) >= 2:
            pa_spread = best_site["pa_pred_kcalmol"] - group.iloc[1]["pa_pred_kcalmol"]
        else:
            pa_spread = 0.0

        mol_records.append({
            "smiles":           smiles,
            "mol_id":           best_site["mol_id"],
            # Keep the PM7 baseline and ML correction from the same site so
            # pa_pred_kcalmol == pa_pm7_kcalmol + delta_pred remains true.
            "pa_pm7_kcalmol":   best_site["pa_pm7_kcalmol"],
            "pa_pred_kcalmol":  pa_pred_max,
            "delta_pred":       best_site["delta_pred_kcalmol"],
            "uncertainty":      uncertainty,
            "pa_spread":        pa_spread,
            "n_sites":          len(group),
            "best_site_element": best_site["site_element"],
        })

    mol_df = pd.DataFrame(mol_records)

    # Load SA score from metadata
    meta = pd.read_parquet(
        DATA_DIR / "processed" / "zinc_metadata.parquet",
        columns=["smiles", "sa_score", "MW"]
    )
    mol_df = mol_df.merge(meta, on="smiles", how="left")

    mol_df.to_parquet(mol_path, index=False)
    log.info(f"Saved molecular PA summary → {mol_path}")

    # ── Summary statistics ───────────────────────────────────────────────────
    log.info(f"\n=== Prediction Summary — Iteration {iteration} ===")
    log.info(f"  Molecules predicted:     {len(mol_df):,}")
    log.info(f"  PA_pred range:           "
             f"{mol_df['pa_pred_kcalmol'].min():.1f} – "
             f"{mol_df['pa_pred_kcalmol'].max():.1f} kcal/mol")
    log.info(f"  PA_pred mean:            {mol_df['pa_pred_kcalmol'].mean():.1f} kcal/mol")

    in_window = ((mol_df["pa_pred_kcalmol"] >= pa_low) &
                 (mol_df["pa_pred_kcalmol"] <= pa_high))
    log.info(
        f"  In {pa_low:g}-{pa_high:g} window:       "
        f"{in_window.sum():,} molecules"
    )

    log.info(f"  Ensemble dispersion mean:{mol_df['uncertainty'].mean():.2f} kcal/mol")
    log.info(f"  PA spread mean:          {mol_df['pa_spread'].mean():.2f} kcal/mol")

    # Known seed molecules sanity check
    log.info(f"\n  Sanity check:")
    for smi, name, exp_pa in [
        ("c1c[nH]cn1",        "imidazole",     223.0),
        ("c1ccc2[nH]cnc2c1",  "benzimidazole", 230.0),
        ("c1cc[nH]n1",        "pyrazole",      213.0),
    ]:
        row = mol_df[mol_df["smiles"] == smi]
        if len(row) > 0:
            pred = row.iloc[0]["pa_pred_kcalmol"]
            pm7  = row.iloc[0]["pa_pm7_kcalmol"]
            log.info(f"    {name}: PM7={pm7:.1f}, pred={pred:.1f}, exp={exp_pa:.1f} kcal/mol")

    log.info(f"\n  Top 10 candidates by predicted PA:")
    top10 = mol_df.nlargest(10, "pa_pred_kcalmol")[
        ["smiles", "pa_pred_kcalmol", "pa_pm7_kcalmol",
         "uncertainty", "pa_spread", "sa_score", "MW"]]
    log.info(top10.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict PA for screening candidates")
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument(
        "--threshold", type=int, default=CONFIG["consensus_min_folds"],
        help="Min folds for consensus (default: pipeline config)")
    args = parser.parse_args()
    main(iteration=args.iter, threshold=args.threshold)
