"""
02_retrieve_candidates.py
=========================
Retrieve candidate molecules for screening via similarity search.

For each seed molecule (imidazole, benzimidazole, pyrazole + any
top candidates from previous iterations), queries the FAISS index
to find the nearest neighbors in PCA space, then filters by
Tanimoto similarity on the actual Morgan fingerprints.

Reads from:
    data/screening/processed/zinc_metadata.parquet
    data/screening/processed/zinc_fingerprints.npy
    data/screening/processed/zinc_fp_keys.npy
    data/screening/processed/zinc_index.faiss
    data/screening/iter{N-1}/pareto_selected.csv   (if N > 1)

Writes to:
    data/screening/iter{N}/candidates.parquet

Usage:
    python screening/scripts/02_retrieve_candidates.py --iter 1
    python screening/scripts/02_retrieve_candidates.py --iter 2
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm

from pipeline_config import DEFAULT_CONFIG_PATH, load_pipeline_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
SCREENING  = SCRIPT_DIR.parent.parent
PROJECT    = SCREENING.parent
DATA_ROOT  = next((p for p in (PROJECT / "data", PROJECT.parent / "data") if p.exists()), PROJECT / "data")
DATA_DIR   = DATA_ROOT / "screening"
PROCESSED  = DATA_DIR / "processed"
CONFIG     = load_pipeline_config()
RETRIEVAL_CONFIG = CONFIG["retrieval"]
MORGAN_RADIUS = CONFIG["featurization"]["morgan_radius"]

# ---------------------------------------------------------------------------
# Seed molecules — known proton conductors
# ---------------------------------------------------------------------------
SEED_MOLECULES = {
    "imidazole":     "c1ncc[nH]1",   # canonical: c1c[nH]cn1
    "benzimidazole": "c1ccc2[nH]cnc2c1",
    "pyrazole":      "c1cc[nH]n1",
    "1H-1,2,4-triazole": "c1ncn[nH]1",
    "benzotriazole": "c1ccc2[nH]nnc2c1",
}

# Similarity bounds for candidate retrieval
TANIMOTO_MIN = RETRIEVAL_CONFIG["tanimoto_min"]
TANIMOTO_MAX = RETRIEVAL_CONFIG["tanimoto_max"]
FAISS_K = RETRIEVAL_CONFIG["faiss_k"]
MAX_CANDIDATES = RETRIEVAL_CONFIG["max_candidates"]

# ---------------------------------------------------------------------------
# Tanimoto similarity using Morgan fingerprints
# ---------------------------------------------------------------------------

def smiles_to_fp_vector(smiles: str, fp_keys: np.ndarray, n_bits: int = 1024) -> np.ndarray:
    """Convert SMILES to dense count FP vector using the same key set as the index."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprint(mol, radius=MORGAN_RADIUS)
    counts = fp.GetNonzeroElements()
    vec = np.zeros(n_bits, dtype=np.float32)
    key_to_col = {int(k): i for i, k in enumerate(fp_keys)}
    for k, v in counts.items():
        col = key_to_col.get(int(k))
        if col is not None:
            vec[col] = float(v)
    return vec


def tanimoto_batch(query: np.ndarray, db: np.ndarray) -> np.ndarray:
    """
    Tanimoto similarity between one query vector and a batch of db vectors.
    Uses the count-based Tanimoto: sum(min) / sum(max) per pair.
    """
    min_sum = np.minimum(query, db).sum(axis=1)
    max_sum = np.maximum(query, db).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tanimoto = np.where(max_sum > 0, min_sum / max_sum, 0.0)
    return tanimoto


def get_seed_pca(smiles: str, metadata: pd.DataFrame) -> np.ndarray | None:
    """Get PCA coordinates for a seed molecule if it exists in the index."""
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) if Chem.MolFromSmiles(smiles) else None
    if canonical is None:
        return None
    # Try exact SMILES match
    match = metadata[metadata["smiles"] == smiles]
    if len(match) == 0:
        match = metadata[metadata["smiles"] == canonical]
    if len(match) > 0:
        return match[["latent_1", "latent_2", "latent_3"]].values[0].astype(np.float32)
    return None


def compute_seed_pca_from_neighbors(
    seed_smiles: str,
    fp_keys: np.ndarray,
    fp_matrix: np.ndarray,
    metadata: pd.DataFrame,
    index: faiss.Index,
    k: int = 50,
) -> np.ndarray:
    """
    For seeds not in the ZINC index, estimate PCA coords as the
    weighted average of the k most similar molecules in the index.
    """
    seed_fp = smiles_to_fp_vector(seed_smiles, fp_keys)
    # Search broadly in PCA space (center of mass as starting point)
    center = metadata[["latent_1","latent_2","latent_3"]].mean().values.astype(np.float32)
    D, I = index.search(center.reshape(1,-1), k * 10)
    neighbor_fps = fp_matrix[I[0]].astype(np.float32)
    sims = tanimoto_batch(seed_fp, neighbor_fps)
    top_idx = np.argsort(sims)[::-1][:k]
    top_sims = sims[top_idx]
    top_global_idx = I[0][top_idx]
    if top_sims.sum() == 0:
        return center
    weights = top_sims / top_sims.sum()
    pca_coords = metadata[["latent_1","latent_2","latent_3"]].values[top_global_idx]
    return (weights[:, None] * pca_coords).sum(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(iteration: int) -> None:
    log.info(f"Screening config: {DEFAULT_CONFIG_PATH}")
    iter_dir = DATA_DIR / f"iter{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    out_path = iter_dir / "candidates.parquet"

    # ── Load processed data ─────────────────────────────────────────────────
    log.info("Loading processed index data ...")
    metadata  = pd.read_parquet(PROCESSED / "zinc_metadata.parquet")
    fp_matrix = np.load(PROCESSED / "zinc_fingerprints.npy").astype(np.float32)
    fp_keys   = np.load(PROCESSED / "zinc_fp_keys.npy")
    index     = faiss.read_index(str(PROCESSED / "zinc_index.faiss"))
    if not (len(metadata) == len(fp_matrix) == index.ntotal):
        raise ValueError(
            "Screening index artifacts are row-misaligned: "
            f"metadata={len(metadata)}, fingerprints={len(fp_matrix)}, "
            f"faiss={index.ntotal}"
        )
    if len(fp_keys) != fp_matrix.shape[1]:
        raise ValueError("Fingerprint key list does not match matrix width")
    log.info(f"  {len(metadata):,} molecules in index")

    # ── Build seed list ──────────────────────────────────────────────────────
    seeds = dict(SEED_MOLECULES)  # base seeds always included

    if iteration > 1:
        prev_pareto = DATA_DIR / f"iter{iteration-1}" / "pareto_selected.csv"
        if prev_pareto.exists():
            prev_df = pd.read_csv(prev_pareto)
            top_prev = prev_df.head(
                RETRIEVAL_CONFIG["previous_iteration_seeds"]
            )
            for _, row in top_prev.iterrows():
                seeds[f"iter{iteration-1}_{row.name}"] = row["smiles"]
            log.info(f"  Added {len(top_prev)} seeds from iteration {iteration-1}")
        else:
            log.warning(f"  No previous Pareto results found at {prev_pareto}")

    log.info(f"Total seeds: {len(seeds)}")

    # ── Retrieve candidates via FAISS + Tanimoto filter ─────────────────────
    all_candidate_indices = set()

    for seed_name, seed_smiles in seeds.items():
        log.info(f"  Processing seed: {seed_name} ({seed_smiles})")

        # Get or estimate PCA coordinates for seed
        pca = get_seed_pca(seed_smiles, metadata)
        if pca is None:
            log.info(f"    Seed not in index — estimating PCA from neighbors ...")
            pca = compute_seed_pca_from_neighbors(
                seed_smiles, fp_keys, fp_matrix, metadata, index)
        log.info(f"    PCA coords: {pca.round(3)}")

        # FAISS search
        D, I = index.search(pca.reshape(1, -1), FAISS_K)
        neighbor_idx = I[0]
        log.info(f"    FAISS retrieved {len(neighbor_idx)} neighbors")

        # Tanimoto filter
        seed_fp = smiles_to_fp_vector(seed_smiles, fp_keys)
        neighbor_fps = fp_matrix[neighbor_idx]
        sims = tanimoto_batch(seed_fp, neighbor_fps)
        mask = (sims >= TANIMOTO_MIN) & (sims <= TANIMOTO_MAX)
        filtered_idx = neighbor_idx[mask]
        filtered_sims = sims[mask]
        log.info(f"    After Tanimoto [{TANIMOTO_MIN}-{TANIMOTO_MAX}]: "
                 f"{mask.sum()} candidates")

        # Store with similarity score
        for idx, sim in zip(filtered_idx, filtered_sims):
            all_candidate_indices.add(int(idx))

        log.info(f"    Running total candidates: {len(all_candidate_indices)}")

    log.info(f"\nTotal unique candidates: {len(all_candidate_indices):,}")

    # ── Build candidate dataframe ────────────────────────────────────────────
    candidate_idx = sorted(all_candidate_indices)

    # Exclude molecules already in training set (251 k-means molecules)
    # Load training SMILES to avoid re-screening known molecules
    train_path = DATA_ROOT / "targets" / "kmeans251_ml.parquet"
    train_df = pd.read_parquet(train_path, columns=["neutral_smiles"])
    training_smiles = set(train_df["neutral_smiles"].tolist())
    log.info(f"  Loaded {len(training_smiles)} training SMILES to exclude")

    cand_df = metadata.iloc[candidate_idx].copy().reset_index(drop=True)
    cand_df["zinc_index"] = candidate_idx

    # Exclude training molecules
    before = len(cand_df)
    cand_df = cand_df[~cand_df["smiles"].isin(training_smiles)].reset_index(drop=True)
    log.info(f"  Excluded {before - len(cand_df)} training molecules")

    # Apply configured SA score filter.
    if "sa_score" not in cand_df or cand_df["sa_score"].isna().any() or (
        cand_df["sa_score"] <= 0
    ).any():
        raise ValueError("Valid positive SA scores are required for candidate filtering")
    before = len(cand_df)
    sa_max = RETRIEVAL_CONFIG["sa_score_max"]
    cand_df = cand_df[cand_df["sa_score"] <= sa_max].reset_index(drop=True)
    log.info(f"  SA score filter (≤{sa_max:g}): {before} → {len(cand_df)}")

    # Compute Tanimoto to all seeds combined (max similarity across seeds)
    log.info("Computing max Tanimoto similarity to seeds ...")
    seed_fps = np.stack([
        smiles_to_fp_vector(smi, fp_keys) for smi in seeds.values()
    ])
    cand_fps = fp_matrix[cand_df["zinc_index"].values]
    max_sims = np.zeros(len(cand_df))
    for i, seed_fp in enumerate(seed_fps):
        sims = tanimoto_batch(seed_fp, cand_fps)
        max_sims = np.maximum(max_sims, sims)
    cand_df["max_tanimoto_to_seeds"] = max_sims

    # Sort by similarity and cap
    cand_df = cand_df.sort_values("max_tanimoto_to_seeds", ascending=False)
    if len(cand_df) > MAX_CANDIDATES:
        log.info(f"  Capping to {MAX_CANDIDATES} candidates (was {len(cand_df)})")
        cand_df = cand_df.head(MAX_CANDIDATES).reset_index(drop=True)
    if cand_df["smiles"].duplicated().any():
        raise ValueError("Retrieved candidate table contains duplicate SMILES")

    # ── Save ─────────────────────────────────────────────────────────────────
    cand_df.to_parquet(out_path, index=False)
    log.info(f"\nSaved {len(cand_df):,} candidates → {out_path}")

    # Summary
    log.info(f"\n=== Iteration {iteration} Candidate Pool ===")
    log.info(f"  Total candidates:          {len(cand_df):,}")
    log.info(f"  Tanimoto range:            "
             f"{cand_df['max_tanimoto_to_seeds'].min():.3f} – "
             f"{cand_df['max_tanimoto_to_seeds'].max():.3f}")
    if "sa_score" in cand_df.columns:
        log.info(f"  SA score mean:             {cand_df['sa_score'].mean():.2f}")
    if "MW" in cand_df.columns:
        log.info(f"  MW range:                  "
                 f"{cand_df['MW'].min():.0f} – {cand_df['MW'].max():.0f} Da")
    log.info(f"  Sample SMILES:")
    for smi in cand_df["smiles"].head(5):
        log.info(f"    {smi}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve candidates for screening iteration")
    parser.add_argument("--iter", type=int, default=1,
                        help="Iteration number (1, 2, or 3)")
    args = parser.parse_args()
    main(iteration=args.iter)
