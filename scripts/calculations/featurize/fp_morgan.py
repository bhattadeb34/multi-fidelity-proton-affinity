"""
fp_morgan.py
============
Morgan circular fingerprints, radius=2, 1024 features, as COUNT vectors.

Count vectors (not bit vectors) are used because they encode how many times
each circular environment appears in the molecule, providing richer information
than binary presence/absence — important for distinguishing e.g. two vs four
NH2 groups.

Output columns: morgan_0 ... morgan_1023
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

MORGAN_RADIUS = 2
MORGAN_NBITS  = 1024
MORGAN_COLS   = [f"morgan_{i}" for i in range(MORGAN_NBITS)]

# Build generator once at module level (avoids repeated construction)
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_NBITS,
)


def morgan_from_smiles(smiles: str) -> np.ndarray | None:
    """
    Compute Morgan count fingerprint for a SMILES string.

    Returns
    -------
    np.ndarray of shape (1024,), dtype int32, or None if SMILES is invalid.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # GetCountFingerprintAsNumPy returns a dense array of counts
    return _MORGAN_GEN.GetCountFingerprintAsNumPy(mol).astype(np.int32)


def morgan_batch(smiles_list: list[str]) -> np.ndarray:
    """
    Compute Morgan count fingerprints for a list of SMILES.

    Returns
    -------
    np.ndarray of shape (N, 1024). Rows for invalid SMILES are filled with NaN.
    """
    results = []
    for smi in smiles_list:
        fp = morgan_from_smiles(smi)
        if fp is None:
            results.append(np.full(MORGAN_NBITS, np.nan))
        else:
            results.append(fp.astype(float))
    return np.vstack(results)
