"""
04_featurize.py
===============
Compute the 23 consensus features for each site record in pm7_results.parquet.

The PM7 electronic properties (LUMO, gap, etc.) are already in the parquet.
This script adds the remaining fingerprint and Mordred features.

Consensus feature set (appeared in >=2 of 5 CV folds on k-means dataset):
  5/5 folds: protonated_pm7_LUMO_eV, protonated_pm7_HOMO_LUMO_gap_eV,
             protonated_mordred_nBase, maccs_86
  4/5 folds: maccs_88, morgan_969
  3/5 folds: maccs_58, morgan_360, neutral_rdkit_fr_guanido
  2/5 folds: maccs_25, maccs_95, morgan_338, morgan_886,
             neutral_mordred_ATS0Z, neutral_mordred_FPSA2,
             neutral_mordred_GATS8s, neutral_mordred_PNSA1,
             neutral_rdkit_PEOE_VSA13, protonated_mordred_GATS1c,
             protonated_mordred_NdNH, protonated_rdkit_EState_VSA9,
             protonated_rdkit_SlogP_VSA2, site_normalized_index

Reads from:
    data/screening/iter{N}/pm7_results.parquet

Writes to:
    data/screening/iter{N}/features.parquet

Usage:
    python screening/scripts/04_featurize.py --iter 1
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
SCREENING  = SCRIPT_DIR.parent.parent
PROJECT    = SCREENING.parent
DATA_DIR   = PROJECT / "data" / "screening"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Consensus feature set — exactly as selected by the k-means CV pipeline
# ---------------------------------------------------------------------------

# MACCS bit indices needed
MACCS_BITS = [25, 58, 86, 88, 95]

# Morgan bit keys needed (from zinc_fp_keys.npy — these are column indices)
# We'll match by the actual key values stored during 01_build_index.py
MORGAN_COL_INDICES = [338, 360, 886, 969]  # column positions in fp_matrix

# RDKit descriptor names
RDKIT_NEUTRAL_DESCS    = ["fr_guanido", "PEOE_VSA13"]
RDKIT_PROT_DESCS       = ["EState_VSA9", "SlogP_VSA2"]

# Mordred descriptor names
MORDRED_NEUTRAL_DESCS  = ["ATS0Z", "FPSA2", "GATS8s", "PNSA1"]
MORDRED_PROT_DESCS     = ["GATS1c", "NdNH", "nBase"]

# PM7 features — already in pm7_results.parquet, just rename
PM7_MAP = {
    "protonated_LUMO_eV":             "protonated_pm7_LUMO_eV",
    "protonated_HOMO_LUMO_gap_eV":    "protonated_pm7_HOMO_LUMO_gap_eV",
}

# Site features — already in pm7_results.parquet
SITE_FEATURES = ["site_normalized_index"]


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def get_maccs(mol) -> dict:
    """Compute MACCS keys — returns dict of needed bits."""
    if mol is None:
        return {f"maccs_{b}": 0 for b in MACCS_BITS}
    fp = MACCSkeys.GenMACCSKeys(mol)
    return {f"maccs_{b}": int(fp[b]) for b in MACCS_BITS}


def get_morgan_cols(mol, fp_keys: np.ndarray) -> dict:
    """Compute Morgan count FP and extract needed column values."""
    result = {f"morgan_{i}": 0 for i in MORGAN_COL_INDICES}
    if mol is None:
        return result
    fp = AllChem.GetMorganFingerprint(mol, radius=2)
    counts = fp.GetNonzeroElements()
    key_to_col = {int(k): i for i, k in enumerate(fp_keys)}
    for k, v in counts.items():
        col = key_to_col.get(int(k))
        if col in MORGAN_COL_INDICES:
            result[f"morgan_{col}"] = int(v)
    return result


def get_rdkit_desc(mol, desc_names: list, prefix: str) -> dict:
    """Compute specific RDKit descriptors."""
    result = {f"{prefix}_{d}": np.nan for d in desc_names}
    if mol is None:
        return result
    for d in desc_names:
        try:
            fn = getattr(Descriptors, d, None)
            if fn is None:
                # Try rdMolDescriptors
                fn = getattr(rdMolDescriptors, f"Calc{d}", None)
            if fn is not None:
                result[f"{prefix}_{d}"] = float(fn(mol))
        except Exception:
            pass
    return result


def get_mordred_desc(mol, desc_names: list, prefix: str) -> dict:
    """Compute specific Mordred descriptors."""
    result = {f"{prefix}_mordred_{d}": np.nan for d in desc_names}
    if mol is None:
        return result
    try:
        from mordred import Calculator, descriptors as mordred_descs
        # Build a minimal calculator with only needed descriptors
        calc = Calculator()
        # Map descriptor names to mordred descriptor classes
        desc_map = {}
        for desc_cls in mordred_descs.__all__:
            cls = getattr(mordred_descs, desc_cls, None)
            if cls is None:
                continue
            try:
                instance = cls()
                for name in desc_names:
                    if name in str(instance):
                        desc_map[name] = instance
            except Exception:
                continue

        # Fallback: use full calculator and filter
        full_calc = Calculator(mordred_descs, ignore_3D=True)
        res = full_calc(mol)
        for d in desc_names:
            val = res[d] if d in res.asdict() else np.nan
            try:
                result[f"{prefix}_mordred_{d}"] = float(val)
            except Exception:
                result[f"{prefix}_mordred_{d}"] = np.nan
    except Exception:
        pass
    return result


def featurize_row(row: pd.Series, fp_keys: np.ndarray) -> dict:
    """Compute all 23 consensus features for one site record."""
    neutral_smi = row["smiles"]
    prot_smi    = row["protonated_smiles"]

    neutral_mol = Chem.MolFromSmiles(neutral_smi)
    prot_mol    = Chem.MolFromSmiles(prot_smi)

    features = {}

    # 1. MACCS — molecule level (use neutral)
    features.update(get_maccs(neutral_mol))

    # 2. Morgan — molecule level (use neutral)
    features.update(get_morgan_cols(neutral_mol, fp_keys))

    # 3. RDKit descriptors — neutral
    features.update(get_rdkit_desc(neutral_mol, RDKIT_NEUTRAL_DESCS, "neutral_rdkit"))

    # 4. RDKit descriptors — protonated
    features.update(get_rdkit_desc(prot_mol, RDKIT_PROT_DESCS, "protonated_rdkit"))

    # 5. Mordred — neutral
    features.update(get_mordred_desc(neutral_mol, MORDRED_NEUTRAL_DESCS, "neutral"))

    # 6. Mordred — protonated
    features.update(get_mordred_desc(prot_mol, MORDRED_PROT_DESCS, "protonated"))

    # 7. PM7 features — already in parquet, just copy with correct names
    for parquet_col, feature_name in PM7_MAP.items():
        features[feature_name] = row.get(parquet_col, np.nan)

    # 8. Site features
    for sf in SITE_FEATURES:
        features[sf] = row.get(sf, np.nan)

    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(iteration: int) -> None:
    iter_dir   = DATA_DIR / f"iter{iteration}"
    pm7_path   = iter_dir / "pm7_results.parquet"
    out_path   = iter_dir / "features.parquet"
    fp_keys_path = DATA_DIR / "processed" / "zinc_fp_keys.npy"

    if not pm7_path.exists():
        log.error(f"PM7 results not found: {pm7_path}")
        sys.exit(1)

    df = pd.read_parquet(pm7_path)
    log.info(f"Loaded {len(df):,} site records")

    fp_keys = np.load(fp_keys_path)
    log.info(f"Loaded {len(fp_keys):,} Morgan FP keys")

    # Check mordred available
    try:
        import mordred
        log.info("Mordred available")
    except ImportError:
        log.warning("Mordred not installed — mordred features will be NaN")
        log.warning("Install with: pip install mordred")

    log.info("Computing features ...")
    all_features = []
    from tqdm import tqdm
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing"):
        all_features.append(featurize_row(row, fp_keys))

    feat_df = pd.DataFrame(all_features)

    # Combine with essential columns from PM7 results
    keep_cols = ["mol_id", "smiles", "protonated_smiles",
                 "site_idx", "site_element", "site_index",
                 "site_normalized_index", "site_n_sites",
                 "pa_pm7_kcalmol"]
    out_df = pd.concat([
        df[keep_cols].reset_index(drop=True),
        feat_df.reset_index(drop=True)
    ], axis=1)

    # Drop duplicate site_normalized_index (already in keep_cols)
    if "site_normalized_index" in feat_df.columns:
        out_df = out_df.loc[:, ~out_df.columns.duplicated()]

    out_df.to_parquet(out_path, index=False)
    log.info(f"Saved features → {out_path}")
    log.info(f"  Shape: {out_df.shape}")
    log.info(f"  Feature columns: {len(feat_df.columns)}")

    # Check coverage
    nan_frac = feat_df.isna().mean()
    problem_cols = nan_frac[nan_frac > 0.1]
    if len(problem_cols) > 0:
        log.warning(f"  High NaN features (>10%):")
        for col, frac in problem_cols.items():
            log.warning(f"    {col}: {frac*100:.1f}% NaN")
    else:
        log.info("  All features have <10% NaN — good coverage")

    log.info(f"\nSample feature values:")
    log.info(feat_df.head(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize PM7 results")
    parser.add_argument("--iter", type=int, default=1)
    args = parser.parse_args()
    main(iteration=args.iter)
