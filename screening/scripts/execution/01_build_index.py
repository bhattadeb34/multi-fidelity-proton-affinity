"""
01_build_index.py
=================
Build the FAISS index and processed metadata for the 821K filtered ZINC molecules.

Reads from:
    data/screening/zinc_raw/filtered_821k.csv
    data/screening/zinc_raw/WriteAllMorganFingerprints-ConcatCSV-2D-AK-AKEC-*.json

Writes to:
    data/screening/processed/zinc_metadata.parquet   -- SMILES + PCA + MW + SA score
    data/screening/processed/zinc_fingerprints.npy   -- (N, 1024) uint8 Morgan FP matrix
    data/screening/processed/zinc_fp_keys.npy        -- (1024,) int64 array of FP bit keys
    data/screening/processed/zinc_index.faiss         -- FAISS index on PCA coordinates
    data/screening/processed/build_report.json        -- summary stats

Usage:
    python screening/scripts/01_build_index.py
    python screening/scripts/01_build_index.py --dry-run   # process first 5000 rows only
"""

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD
import sys
# SA score — use local sascorer.py (Ertl & Schuffenhauer, 2009)
# sascorer.py and fpscores.pkl.gz must be in the same directory as this script
sys.path.insert(0, str(Path(__file__).parent))
try:
    import sascorer
except ImportError:
    sascorer = None
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
SCREENING    = SCRIPT_DIR.parent.parent
PROJECT      = SCREENING.parent
DATA_DIR     = PROJECT / "data" / "screening"
RAW_DIR      = DATA_DIR / "zinc_raw"
PROCESSED    = DATA_DIR / "processed"
FILTERED_CSV = RAW_DIR / "filtered_821k.csv"
JSON_GLOB    = "WriteAllMorganFingerprints-ConcatCSV-2D-AK-AKEC-*.json"

# ---------------------------------------------------------------------------
# SA score helper
# ---------------------------------------------------------------------------

def compute_sa_score(smiles: str) -> float:
    """Compute SA score. Returns -1.0 if sascorer not available."""
    if sascorer is None:
        return -1.0
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        return float(sascorer.calculateScore(mol))
    except Exception:
        return 10.0


# ---------------------------------------------------------------------------
# Morgan fingerprint helpers
# ---------------------------------------------------------------------------

def load_all_json_fps(raw_dir: Path) -> dict[str, dict[str, int]]:
    """Load all 10 JSON files into one SMILES -> {key: count} dict."""
    json_files = sorted(raw_dir.glob(JSON_GLOB))
    log.info(f"Found {len(json_files)} Morgan fingerprint JSON files")
    all_fps: dict[str, dict[str, int]] = {}
    for jf in json_files:
        log.info(f"  Loading {jf.name} ...")
        with open(jf) as f:
            chunk = json.load(f)
        all_fps.update(chunk)
        log.info(f"    → {len(chunk):,} molecules (total: {len(all_fps):,})")
    return all_fps


def build_fp_matrix(
    smiles_list: list[str],
    all_fps: dict[str, dict[str, int]],
    n_bits: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert sparse Morgan FP dicts to dense (N, n_bits) matrix.

    The JSON stores arbitrary integer keys (hashed Morgan bit indices).
    We map them to a fixed 1024-column space by taking the top-1024
    most frequent keys across the dataset, consistent with how the
    original pipeline built the Morgan features.

    Returns:
        fp_matrix  -- (N, n_bits) uint8 array
        fp_keys    -- (n_bits,) int64 array of the selected bit keys
    """
    log.info("Counting Morgan key frequencies across dataset ...")
    key_counts: dict[int, int] = {}
    found = 0
    missing = 0
    for smi in tqdm(smiles_list, desc="Counting keys"):
        fps = all_fps.get(smi)
        if fps is None:
            missing += 1
            continue
        found += 1
        for k in fps:
            ik = int(k)
            key_counts[ik] = key_counts.get(ik, 0) + 1

    log.info(f"  Found FPs for {found:,} / {len(smiles_list):,} molecules")
    log.info(f"  Missing FPs: {missing:,}")
    log.info(f"  Total unique Morgan keys: {len(key_counts):,}")

    # Select top n_bits keys by frequency
    top_keys = sorted(key_counts, key=lambda k: -key_counts[k])[:n_bits]
    top_keys = np.array(sorted(top_keys), dtype=np.int64)
    key_to_col = {int(k): i for i, k in enumerate(top_keys)}
    log.info(f"  Selected top {n_bits} keys for fingerprint matrix")

    log.info("Building fingerprint matrix ...")
    fp_matrix = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for row_idx, smi in enumerate(tqdm(smiles_list, desc="Building FP matrix")):
        fps = all_fps.get(smi)
        if fps is None:
            continue
        for k, v in fps.items():
            col = key_to_col.get(int(k))
            if col is not None:
                fp_matrix[row_idx, col] = min(v, 255)

    return fp_matrix, top_keys


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(pca_coords: np.ndarray) -> faiss.Index:
    """Build a flat L2 FAISS index on PCA coordinates."""
    coords = pca_coords.astype(np.float32)
    d = coords.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(coords)
    log.info(f"  FAISS index built: {index.ntotal:,} vectors, dim={d}")
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    t0 = time.time()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # ── Load filtered molecule CSV ──────────────────────────────────────────
    log.info(f"Loading filtered molecule CSV: {FILTERED_CSV}")
    df = pd.read_csv(FILTERED_CSV)
    if dry_run:
        log.warning("DRY RUN — using first 5,000 molecules only")
        df = df.head(5000)
    log.info(f"  {len(df):,} molecules loaded")

    smiles_list = df["smiles"].tolist()

    # ── Compute SA scores ───────────────────────────────────────────────────
    log.info("Computing SA scores ...")
    sa_scores = [compute_sa_score(smi) for smi in tqdm(smiles_list, desc="SA score")]
    df["sa_score"] = sa_scores
    if sascorer is not None:
        log.info(f"  SA score range: {min(sa_scores):.2f} – {max(sa_scores):.2f}")
        log.info(f"  SA ≤ 3: {sum(s<=3 for s in sa_scores):,}  "
                 f"SA ≤ 5: {sum(s<=5 for s in sa_scores):,}")
    else:
        log.warning("  SA score unavailable — install with: pip install sascorer")

    # ── Save metadata parquet ───────────────────────────────────────────────
    meta_cols = ["smiles", "latent_1", "latent_2", "latent_3",
                 "MW", "HBD", "HBA", "heavy_atoms", "protonatable_sites", "sa_score"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta_df = df[meta_cols].copy()
    meta_path = PROCESSED / "zinc_metadata.parquet"
    meta_df.to_parquet(meta_path, index=False)
    log.info(f"Saved metadata: {meta_path}  ({len(meta_df):,} rows)")

    # ── Load Morgan fingerprint JSONs ───────────────────────────────────────
    log.info("Loading Morgan fingerprint JSON files ...")
    all_fps = load_all_json_fps(RAW_DIR)

    # ── Build fingerprint matrix ────────────────────────────────────────────
    fp_matrix, fp_keys = build_fp_matrix(smiles_list, all_fps, n_bits=1024)

    fp_matrix_path = PROCESSED / "zinc_fingerprints.npy"
    fp_keys_path   = PROCESSED / "zinc_fp_keys.npy"
    np.save(fp_matrix_path, fp_matrix)
    np.save(fp_keys_path,   fp_keys)
    log.info(f"Saved FP matrix:  {fp_matrix_path}  shape={fp_matrix.shape}")
    log.info(f"Saved FP keys:    {fp_keys_path}     shape={fp_keys.shape}")

    # ── Build FAISS index ───────────────────────────────────────────────────
    log.info("Building FAISS index on PCA coordinates ...")
    pca_cols = ["latent_1", "latent_2", "latent_3"]
    pca_coords = df[pca_cols].values.astype(np.float32)
    index = build_faiss_index(pca_coords)
    index_path = PROCESSED / "zinc_index.faiss"
    faiss.write_index(index, str(index_path))
    log.info(f"Saved FAISS index: {index_path}")

    # ── Save build report ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    report = {
        "n_molecules":       len(df),
        "n_fp_bits":         int(fp_matrix.shape[1]),
        "n_fp_keys":         int(len(fp_keys)),
        "fp_coverage":       float((fp_matrix.sum(axis=1) > 0).mean()),
        "sa_le3":            int(sum(s <= 3 for s in sa_scores)),
        "sa_le5":            int(sum(s <= 5 for s in sa_scores)),
        "sa_mean":           float(np.mean(sa_scores)),
        "faiss_ntotal":      int(index.ntotal),
        "faiss_dim":         int(pca_coords.shape[1]),
        "dry_run":           dry_run,
        "elapsed_seconds":   round(elapsed, 1),
    }
    report_path = PROCESSED / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info(f"Saved report: {report_path}")

    log.info(f"\nDone in {elapsed/60:.1f} min")
    log.info(f"  Molecules:    {report['n_molecules']:,}")
    log.info(f"  FP coverage:  {report['fp_coverage']*100:.1f}%")
    log.info(f"  SA ≤ 5:       {report['sa_le5']:,} ({report['sa_le5']/report['n_molecules']*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index for ZINC screening")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process first 5,000 molecules only for testing")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
