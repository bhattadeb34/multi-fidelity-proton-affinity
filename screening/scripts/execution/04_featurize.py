"""
04_featurize.py
===============
Compute consensus features for each site record in pm7_results.parquet.

Features are derived programmatically from cv_results.json rather than
hardcoded. The script reads the selected features per fold, computes the
consensus set (>= threshold folds), parses each feature name to determine
the computation method, and computes only what is needed.

Reads from:
    results/kmeans251/cv_results.json          (consensus feature source)
    data/screening/iter{N}/pm7_results.parquet

Writes to:
    data/screening/iter{N}/features.parquet

Usage:
    python screening/scripts/execution/04_featurize.py --iter 1
    python screening/scripts/execution/04_featurize.py --iter 1 --resume
    python screening/scripts/execution/04_featurize.py --iter 1 --threshold 2
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
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, Descriptors, rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors

from pipeline_config import DEFAULT_CONFIG_PATH, load_pipeline_config

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
SCREENING  = SCRIPT_DIR.parent.parent
PROJECT    = SCREENING.parent
DATA_ROOT  = next((p for p in (PROJECT / "data", PROJECT.parent / "data") if p.exists()), PROJECT / "data")
DATA_DIR   = DATA_ROOT / "screening"
RESULTS    = PROJECT / "results" / "kmeans251"
CONFIG     = load_pipeline_config()

# Reuse the training implementation for Mordred, including deterministic
# ETKDGv3 embedding (seed 42), MMFF optimization, descriptor ordering, and
# conversion of Mordred Missing/Error values to NaN.
FEATURIZE_DIR = PROJECT / "scripts" / "calculations" / "featurize"
if str(FEATURIZE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURIZE_DIR))
from desc_mordred import MORDRED_3D_NAMES, mordred_three_states

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consensus feature extraction from cv_results.json
# ---------------------------------------------------------------------------

def load_consensus_features(cv_results_path: Path, threshold: int = 3):
    with open(cv_results_path) as f:
        data = json.load(f)

    counts = Counter()
    n_folds = len(data["selected_features_per_fold"])
    for fold_feats in data["selected_features_per_fold"]:
        counts.update(fold_feats)

    consensus = sorted([f for f, c in counts.items() if c >= threshold])
    log.info(f"CV results: {n_folds} folds, threshold >= {threshold}")
    log.info(f"Consensus features: {len(consensus)}")
    return consensus


def parse_feature_names(features: list[str]):
    """Parse feature names into computation groups."""
    groups = {
        "maccs_bits": [],
        "morgan_cols": [],
        "neutral_rdkit": [],
        "protonated_rdkit": [],
        "neutral_mordred": [],
        "protonated_mordred": [],
        "pm7_neutral": [],
        "pm7_protonated": [],
        "site": [],
    }

    for f in features:
        if f.startswith("maccs_"):
            groups["maccs_bits"].append(int(f.split("_")[1]))
        elif f.startswith("morgan_"):
            groups["morgan_cols"].append(int(f.split("_")[1]))
        elif f.startswith("neutral_rdkit_"):
            groups["neutral_rdkit"].append(f.replace("neutral_rdkit_", ""))
        elif f.startswith("protonated_rdkit_"):
            groups["protonated_rdkit"].append(f.replace("protonated_rdkit_", ""))
        elif f.startswith("neutral_mordred_"):
            groups["neutral_mordred"].append(f.replace("neutral_mordred_", ""))
        elif f.startswith("protonated_mordred_"):
            groups["protonated_mordred"].append(f.replace("protonated_mordred_", ""))
        elif f.startswith("neutral_pm7_"):
            groups["pm7_neutral"].append(f)
        elif f.startswith("protonated_pm7_"):
            groups["pm7_protonated"].append(f)
        elif f.startswith("site_"):
            groups["site"].append(f)
        else:
            raise ValueError(
                f"Consensus feature {f!r} has no screening implementation"
            )

    for cat, items in groups.items():
        if items:
            log.info(f"  {cat}: {len(items)} features")

    return groups


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def get_maccs(mol, bits: list[int]) -> dict:
    if mol is None:
        return {f"maccs_{b}": 0 for b in bits}
    fp = MACCSkeys.GenMACCSKeys(mol)
    return {f"maccs_{b}": int(fp[b]) for b in bits}


def get_morgan_cols(
    mol,
    col_indices: list[int],
    radius: int,
    fp_size: int,
) -> dict:
    """Return the same fixed-width hashed count bins used in training."""
    result = {f"morgan_{i}": 0 for i in col_indices}
    if mol is None:
        return result
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size,
    )
    counts = generator.GetCountFingerprintAsNumPy(mol)
    return {f"morgan_{i}": int(counts[i]) for i in col_indices}


def get_rdkit_desc(mol, desc_names: list[str], prefix: str) -> dict:
    result = {f"{prefix}_{d}": np.nan for d in desc_names}
    if mol is None:
        return result
    for d in desc_names:
        try:
            fn = getattr(Descriptors, d, None)
            if fn is None:
                fn = getattr(rdMolDescriptors, f"Calc{d}", None)
            if fn is not None:
                result[f"{prefix}_{d}"] = float(fn(mol))
        except Exception:
            pass
    return result


def get_mordred_descs(
    neutral_smiles: str,
    protonated_smiles: str,
    neutral_names: list[str],
    protonated_names: list[str],
) -> dict:
    """Compute the exact all-states 3D Mordred representation used in training."""
    requested = {
        **{f"neutral_mordred_{name}": ("neutral", name) for name in neutral_names},
        **{f"protonated_mordred_{name}": ("protonated", name) for name in protonated_names},
    }
    result = {column: np.nan for column in requested}
    if not requested:
        return result

    values = mordred_three_states(
        neutral_smiles=neutral_smiles,
        protonated_smiles=protonated_smiles,
        compute_3d=True,
        state_strategy="all_states",
    )
    name_to_index = {name: idx for idx, name in enumerate(MORDRED_3D_NAMES)}
    for column, (state, name) in requested.items():
        idx = name_to_index.get(name)
        if idx is not None:
            result[column] = float(values[state][idx])
    return result


def get_site_features(row, site_names: list[str]) -> dict:
    result = {}
    for sf in site_names:
        if sf in ("site_is_N", "site_is_O", "site_is_S"):
            elem = str(row.get("site_element", "")).strip().upper()
            target = sf.split("_")[-1]
            result[sf] = 1.0 if elem == target else 0.0
        elif sf == "site_normalized_index":
            result[sf] = row.get("site_normalized_index", np.nan)
        elif sf == "site_index":
            result[sf] = row.get("site_index", np.nan)
        elif sf == "site_n_sites":
            result[sf] = row.get("site_n_sites", np.nan)
        else:
            result[sf] = row.get(sf, np.nan)
    return result


def get_pm7_features(row, pm7_features: list[str]) -> dict:
    """Map pm7 features from parquet column names.

    Training data uses protonated_pm7_LUMO_eV, but pm7_results.parquet
    stores it as protonated_LUMO_eV (no pm7_ infix). This function
    handles the mapping.
    """
    result = {}
    for f in pm7_features:
        parquet_col = f.replace("_pm7_", "_")
        result[f] = row.get(parquet_col, np.nan)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(iteration: int, threshold: int, resume: bool = False) -> None:
    feature_config = CONFIG["featurization"]
    morgan_radius = feature_config["morgan_radius"]
    morgan_fp_size = feature_config["morgan_fp_size"]
    log.info(f"Screening config: {DEFAULT_CONFIG_PATH}")
    cv_path = RESULTS / "cv_results.json"
    if not cv_path.exists():
        log.error(f"cv_results.json not found: {cv_path}")
        sys.exit(1)

    consensus = load_consensus_features(cv_path, threshold)
    groups = parse_feature_names(consensus)

    iter_dir   = DATA_DIR / f"iter{iteration}"
    pm7_path   = iter_dir / "pm7_results.parquet"
    out_path   = iter_dir / "features.parquet"

    if not pm7_path.exists():
        log.error(f"PM7 results not found: {pm7_path}")
        sys.exit(1)

    df = pd.read_parquet(pm7_path)
    log.info(f"Loaded {len(df):,} site records")

    keep_cols = ["mol_id", "smiles", "protonated_smiles",
                 "site_idx", "site_element", "site_index",
                 "site_normalized_index", "site_n_sites",
                 "pa_pm7_kcalmol"]
    key_cols = ["smiles", "protonated_smiles", "site_idx"]
    base = df[keep_cols].copy()
    base["_row_order"] = np.arange(len(base))

    reused = pd.DataFrame()
    pending = df
    if resume and out_path.exists():
        existing = pd.read_parquet(out_path)
        missing_existing = set(consensus) - set(existing.columns)
        if missing_existing:
            raise ValueError(
                "Existing feature artifact is incompatible with current consensus: "
                f"{sorted(missing_existing)}"
            )
        if existing.duplicated(key_cols).any() or df.duplicated(key_cols).any():
            raise ValueError("Duplicate site keys prevent safe incremental featurization")
        reused = base.merge(
            existing[key_cols + consensus],
            on=key_cols,
            how="inner",
            validate="one_to_one",
        )
        reused_keys = pd.MultiIndex.from_frame(reused[key_cols])
        current_keys = pd.MultiIndex.from_frame(df[key_cols])
        pending = df[~current_keys.isin(reused_keys)].copy()
        log.info(
            f"Resume: reusing {len(reused):,} verified site rows; "
            f"computing {len(pending):,} new rows"
        )

    if groups["morgan_cols"]:
        invalid = [i for i in groups["morgan_cols"] if not 0 <= i < morgan_fp_size]
        if invalid:
            raise ValueError(f"Morgan columns outside fpSize={morgan_fp_size}: {invalid}")
        log.info(
            "Morgan representation: fixed-width hashed count fingerprint "
            f"(radius={morgan_radius}, fpSize={morgan_fp_size})"
        )

    has_mordred = False
    if groups["neutral_mordred"] or groups["protonated_mordred"]:
        try:
            import mordred
            has_mordred = True
            log.info("Mordred available")
        except ImportError:
            raise RuntimeError(
                "The current consensus requires Mordred descriptors, but "
                "Mordred is unavailable; refusing a silently degraded run."
            )

    log.info("Computing features ...")
    all_features = []
    from tqdm import tqdm
    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="Featurizing"):
        neutral_smi = row["smiles"]
        prot_smi    = row["protonated_smiles"]
        neutral_mol = Chem.MolFromSmiles(neutral_smi)
        prot_mol    = Chem.MolFromSmiles(prot_smi)

        features = {}

        if groups["maccs_bits"]:
            features.update(get_maccs(neutral_mol, groups["maccs_bits"]))

        if groups["morgan_cols"]:
            features.update(get_morgan_cols(
                neutral_mol,
                groups["morgan_cols"],
                morgan_radius,
                morgan_fp_size,
            ))

        if groups["neutral_rdkit"]:
            features.update(get_rdkit_desc(neutral_mol, groups["neutral_rdkit"], "neutral_rdkit"))

        if groups["protonated_rdkit"]:
            features.update(get_rdkit_desc(prot_mol, groups["protonated_rdkit"], "protonated_rdkit"))

        if groups["neutral_mordred"] or groups["protonated_mordred"]:
            features.update(get_mordred_descs(
                neutral_smi,
                prot_smi,
                groups["neutral_mordred"],
                groups["protonated_mordred"],
            ))

        if groups["pm7_neutral"] or groups["pm7_protonated"]:
            features.update(get_pm7_features(row, groups["pm7_neutral"] + groups["pm7_protonated"]))

        if groups["site"]:
            features.update(get_site_features(row, groups["site"]))

        all_features.append(features)

    feat_df = pd.DataFrame(all_features, index=pending.index)
    computed = pd.concat([base.loc[pending.index], feat_df], axis=1)
    out_df = pd.concat([reused, computed], ignore_index=True)
    out_df = (out_df.sort_values("_row_order")
              .drop(columns="_row_order")
              .reset_index(drop=True))
    out_df = out_df.loc[:, ~out_df.columns.duplicated()]

    out_df.to_parquet(out_path, index=False)
    log.info(f"Saved features -> {out_path}")
    log.info(f"  Shape: {out_df.shape}")
    log.info(f"  Feature columns: {len(consensus)}")

    missing = set(consensus) - set(out_df.columns)
    if missing:
        raise ValueError(f"Consensus features absent from output: {sorted(missing)}")
    output_features = out_df[consensus]
    all_nan = output_features.columns[output_features.isna().all()].tolist()
    if all_nan:
        raise ValueError(
            f"Consensus features are entirely NaN in screening output: {all_nan}"
        )
    nan_frac = output_features.isna().mean()
    problem_cols = nan_frac[nan_frac > 0.1]
    if len(problem_cols) > 0:
        log.warning(f"  High NaN features (>10%):")
        for col, frac in problem_cols.items():
            log.warning(f"    {col}: {frac*100:.1f}% NaN")
    else:
        log.info("  All features have <10% NaN")

    matched = set(output_features.columns) & set(consensus)
    log.info(f"  Consensus features computed: {len(matched)}/{len(consensus)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize PM7 results")
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument(
        "--threshold", type=int, default=CONFIG["consensus_min_folds"],
        help="Min folds for consensus (default: pipeline config)")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse exact site-key matches from an existing features.parquet",
    )
    args = parser.parse_args()
    main(iteration=args.iter, threshold=args.threshold, resume=args.resume)
