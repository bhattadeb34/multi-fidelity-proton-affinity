"""
fp_maccs.py
===========
MACCS structural key fingerprints (167 bits).

Encodes presence/absence of 166 common chemical substructures
(bit 0 is unused by convention, so 167 total bits, 166 informative).
Captures functional groups and structural motifs known to influence
proton affinity.

Output columns: maccs_1 ... maccs_166  (bit index, 1-based)
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys

N_MACCS = 167
MACCS_COLS = [f"maccs_{i}" for i in range(N_MACCS)]


def maccs_from_smiles(smiles: str) -> np.ndarray | None:
    """
    Compute MACCS keys for a SMILES string.

    Returns
    -------
    np.ndarray of shape (167,), dtype int8, or None if SMILES is invalid.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.int8)


def maccs_batch(smiles_list: list[str]) -> np.ndarray:
    """
    Compute MACCS keys for a list of SMILES.

    Returns
    -------
    np.ndarray of shape (N, 167). Rows for invalid SMILES are filled with NaN.
    """
    results = []
    for smi in smiles_list:
        fp = maccs_from_smiles(smi)
        if fp is None:
            results.append(np.full(N_MACCS, np.nan))
        else:
            results.append(fp.astype(float))
    return np.vstack(results)
