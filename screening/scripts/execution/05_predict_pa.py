"""
05_predict_pa.py
================
Apply the trained ExtraTrees k-means model to predict PA corrections,
compute uncertainty, and select the best protonation site per molecule.

For each molecule:
  - PA_pred per site = PA_PM7 + Delta_ML
  - Uncertainty = std of predictions across 500 trees
  - Molecular PA_pred = max(PA_pred across sites)
  - PA_spread = PA_pred_max - PA_pred_second (amphoteric character)

Reads from:
    data/screening/iter{N}/features.parquet
    results/kmeans251/cv_results.json          (to rebuild model)
    data/targets/kmeans251_ml.parquet          (training data)

Writes to:
    data/screening/iter{N}/predictions.parquet  -- per-site predictions
    data/screening/iter{N}/molecular_pa.parquet -- per-molecule summary

Usage:
    python screening/scripts/05_predict_pa.py --iter 1
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
SCREENING  = SCRIPT_DIR.parent.parent
PROJECT    = SCREENING.parent
DATA_DIR   = PROJECT / "data" / "screening"

KJMOL_TO_KCAL = 1 / 4.184

# The 23 consensus features in the exact order the model expects
CONSENSUS_FEATURES = [
    # 5/5 folds
    "protonated_pm7_LUMO_eV",
    "protonated_pm7_HOMO_LUMO_gap_eV",
    "protonated_mordred_nBase",
    "maccs_86",
    # 4/5 folds
    "maccs_88",
    "morgan_969",
    # 3/5 folds
    "maccs_58",
    "morgan_360",
    "neutral_rdkit_fr_guanido",
    # 2/5 folds
    "maccs_25",
    "maccs_95",
    "morgan_338",
    "morgan_886",
    "neutral_mordred_ATS0Z",
    "neutral_mordred_FPSA2",
    "neutral_mordred_GATS8s",
    "neutral_mordred_PNSA1",
    "neutral_rdkit_PEOE_VSA13",
    "protonated_mordred_GATS1c",
    "protonated_mordred_NdNH",
    "protonated_rdkit_EState_VSA9",
    "protonated_rdkit_SlogP_VSA2",
    "site_normalized_index",
]


# ---------------------------------------------------------------------------
# Load training data and retrain ExtraTrees on full k-means dataset
# ---------------------------------------------------------------------------

def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Load k-means training data and return X, y for the consensus features."""
    target_path = PROJECT / "data" / "targets" / "kmeans251_ml.parquet"
    df = pd.read_parquet(target_path)

    # Compute target
    df["pm7_pa_kcalmol"] = df["pm7_pa_kjmol"] * KJMOL_TO_KCAL
    df["dft_pa_kcalmol"] = df["dft_pa_kjmol"] * KJMOL_TO_KCAL
    df["correction_kcalmol"] = df["dft_pa_kcalmol"] - df["pm7_pa_kcalmol"]

    # Map feature names from parquet columns
    # The parquet uses slightly different column names than our consensus list
    col_map = {
        "protonated_pm7_LUMO_eV":          "protonated_pm7_LUMO_eV",
        "protonated_pm7_HOMO_LUMO_gap_eV": "protonated_pm7_HOMO_LUMO_gap_eV",
        "protonated_mordred_nBase":        "protonated_mordred_nBase",
        "maccs_86":                        "maccs_86",
        "maccs_88":                        "maccs_88",
        "morgan_969":                      "morgan_969",
        "maccs_58":                        "maccs_58",
        "morgan_360":                      "morgan_360",
        "neutral_rdkit_fr_guanido":        "neutral_rdkit_fr_guanido",
        "maccs_25":                        "maccs_25",
        "maccs_95":                        "maccs_95",
        "morgan_338":                      "morgan_338",
        "morgan_886":                      "morgan_886",
        "neutral_mordred_ATS0Z":           "neutral_mordred_ATS0Z",
        "neutral_mordred_FPSA2":           "neutral_mordred_FPSA2",
        "neutral_mordred_GATS8s":          "neutral_mordred_GATS8s",
        "neutral_mordred_PNSA1":           "neutral_mordred_PNSA1",
        "neutral_rdkit_PEOE_VSA13":        "neutral_rdkit_PEOE_VSA13",
        "protonated_mordred_GATS1c":       "protonated_mordred_GATS1c",
        "protonated_mordred_NdNH":         "protonated_mordred_NdNH",
        "protonated_rdkit_EState_VSA9":    "protonated_rdkit_EState_VSA9",
        "protonated_rdkit_SlogP_VSA2":     "protonated_rdkit_SlogP_VSA2",
        "site_normalized_index":           "site_normalized_index",
    }

    # Get available features
    available = []
    for feat in CONSENSUS_FEATURES:
        parquet_col = col_map.get(feat, feat)
        if parquet_col in df.columns:
            available.append((feat, parquet_col))
        else:
            log.warning(f"  Training feature not found: {feat} (mapped to {parquet_col})")

    X = df[[col for _, col in available]].values.astype(np.float64)
    y = df["correction_kcalmol"].values.astype(np.float64)

    log.info(f"  Training data: {len(df):,} sites, {len(available)} features")
    return X, y, [feat for feat, _ in available]


def train_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """Train ExtraTrees on full k-means dataset with imputation."""
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    model = ExtraTreesRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
    )
    log.info("  Training ExtraTrees (n=500) on full k-means dataset ...")
    model.fit(X_imp, y)
    log.info("  Done.")
    return model, imputer


# ---------------------------------------------------------------------------
# Prediction with uncertainty
# ---------------------------------------------------------------------------

def predict_with_uncertainty(
    model: ExtraTreesRegressor,
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

def main(iteration: int) -> None:
    iter_dir   = DATA_DIR / f"iter{iteration}"
    feat_path  = iter_dir / "features.parquet"
    pred_path  = iter_dir / "predictions.parquet"
    mol_path   = iter_dir / "molecular_pa.parquet"

    if not feat_path.exists():
        log.error(f"Features not found: {feat_path}")
        sys.exit(1)

    # ── Load features ────────────────────────────────────────────────────────
    feat_df = pd.read_parquet(feat_path)
    log.info(f"Loaded {len(feat_df):,} site records")

    # ── Train model on full k-means dataset ─────────────────────────────────
    log.info("Loading k-means training data ...")
    X_train, y_train, train_features = load_training_data()
    model, imputer = train_model(X_train, y_train)

    # ── Prepare screening features ───────────────────────────────────────────
    # Use only features that were available in training
    avail = [f for f in train_features if f in feat_df.columns]
    missing = [f for f in train_features if f not in feat_df.columns]
    if missing:
        log.warning(f"  Missing features in screening data: {missing}")
        # Fill missing with NaN — imputer will handle
        for f in missing:
            feat_df[f] = np.nan

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

        best_site    = group.iloc[0]
        pa_pred_max  = best_site["pa_pred_kcalmol"]
        pa_pm7_max   = group["pa_pm7_kcalmol"].max()
        uncertainty  = best_site["delta_std_kcalmol"]

        # PA spread — difference between best and second best site
        if len(group) >= 2:
            pa_spread = pa_pred_max - group.iloc[1]["pa_pred_kcalmol"]
        else:
            pa_spread = 0.0

        mol_records.append({
            "smiles":           smiles,
            "mol_id":           best_site["mol_id"],
            "pa_pm7_kcalmol":   pa_pm7_max,
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

    in_window = ((mol_df["pa_pred_kcalmol"] >= 210) &
                 (mol_df["pa_pred_kcalmol"] <= 235))
    log.info(f"  In 210-235 window:       {in_window.sum():,} molecules")

    log.info(f"  Uncertainty mean:        {mol_df['uncertainty'].mean():.2f} kcal/mol")
    log.info(f"  PA spread mean:          {mol_df['pa_spread'].mean():.2f} kcal/mol")

    # Known seed molecules sanity check
    log.info(f"\n  Sanity check:")
    for smi, name, exp_pa in [
        ("c1cn[nH]c1",        "imidazole",     223.0),
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
    args = parser.parse_args()
    main(iteration=args.iter)
