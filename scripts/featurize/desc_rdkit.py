"""
desc_rdkit.py
=============
210 RDKit molecular descriptors per molecular state.

Computed for neutral, protonated, and delta (protonated - neutral),
yielding 630 total RDKit features per molecule.

The 210 descriptors include:
  - Topological indices: Chi0-4 (n/v), Kappa1-3, BalabanJ, BertzCT
  - EState indices: MaxAbsEStateIndex, MinEStateIndex, EState_VSA1-11, etc.
  - Electrotopological: PEOE_VSA, SMR_VSA, SlogP_VSA
  - Partial charge: MaxPartialCharge, MinPartialCharge, MaxAbsPartialCharge
  - BCUT2D: BCUT2D_MWHI/LOW, BCUT2D_CHGHI/LOW, BCUT2D_LOGPHI/LOW, BCUT2D_MRHI/LOW
  - Physicochemical: MolLogP, MolMR, TPSA, LabuteASA, MolWt, HeavyAtomMolWt
  - Fragment counts: fr_* (amines, carbonyls, etc.)
  - Ring descriptors: RingCount, NumAromaticRings, etc.

Excluded from the full 217-descriptor RDKit set (7 descriptors):
  qed, SPS               — newer additions not in original feature set
  Ipc, AvgIpc            — numerical overflow on larger molecules
  FpDensityMorgan1/2/3   — fingerprint density, not traditional descriptors

Output columns (per state):
  neutral_rdkit_{name}, protonated_rdkit_{name}, delta_rdkit_{name}
"""

import numpy as np
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors

# ---------------------------------------------------------------------------
# Define the 210 descriptors
# ---------------------------------------------------------------------------

_EXCLUDE = {
    "qed", "SPS",                              # newer, not in original set
    "Ipc", "AvgIpc",                           # overflow on large molecules
    "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3",  # non-standard
}

# Ordered list of (name, function) pairs — order is fixed for reproducibility
RDKIT_DESC_LIST = [(name, fn) for name, fn in Descriptors.descList
                   if name not in _EXCLUDE]

assert len(RDKIT_DESC_LIST) == 210, (
    f"Expected 210 descriptors, got {len(RDKIT_DESC_LIST)}. "
    f"RDKit version may have changed."
)

RDKIT_DESC_NAMES  = [name for name, _ in RDKIT_DESC_LIST]
N_RDKIT           = len(RDKIT_DESC_NAMES)

# Column names for each state
NEUTRAL_RDKIT_COLS    = [f"neutral_rdkit_{n}"    for n in RDKIT_DESC_NAMES]
PROTONATED_RDKIT_COLS = [f"protonated_rdkit_{n}" for n in RDKIT_DESC_NAMES]
DELTA_RDKIT_COLS      = [f"delta_rdkit_{n}"      for n in RDKIT_DESC_NAMES]
ALL_RDKIT_COLS        = NEUTRAL_RDKIT_COLS + PROTONATED_RDKIT_COLS + DELTA_RDKIT_COLS  # 630


def _compute_single(mol) -> np.ndarray:
    """Compute all 210 descriptors for one RDKit mol object."""
    vals = np.empty(N_RDKIT, dtype=np.float64)
    for i, (name, fn) in enumerate(RDKIT_DESC_LIST):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v = fn(mol)
            vals[i] = float(v) if v is not None else np.nan
        except Exception:
            vals[i] = np.nan
    return vals


def rdkit_descs_from_smiles(smiles: str) -> np.ndarray | None:
    """
    Compute 210 RDKit descriptors for a single SMILES.

    Returns
    -------
    np.ndarray of shape (210,) or None if SMILES is invalid.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _compute_single(mol)


def rdkit_descs_batch(smiles_list: list[str]) -> np.ndarray:
    """
    Compute 210 RDKit descriptors for a list of SMILES.

    Returns
    -------
    np.ndarray of shape (N, 210). Invalid SMILES rows are NaN.
    """
    results = []
    for smi in smiles_list:
        vals = rdkit_descs_from_smiles(smi)
        results.append(vals if vals is not None else np.full(N_RDKIT, np.nan))
    return np.vstack(results)


def rdkit_descs_three_states(
    neutral_smiles: str,
    protonated_smiles: str,
) -> dict[str, np.ndarray]:
    """
    Compute descriptors for neutral, protonated, and delta states.

    Returns
    -------
    dict with keys 'neutral', 'protonated', 'delta', each (210,) array.
    Delta = protonated - neutral (NaN where either is NaN).
    """
    neu  = rdkit_descs_from_smiles(neutral_smiles)
    prot = rdkit_descs_from_smiles(protonated_smiles)

    neu_arr  = neu  if neu  is not None else np.full(N_RDKIT, np.nan)
    prot_arr = prot if prot is not None else np.full(N_RDKIT, np.nan)
    delta    = prot_arr - neu_arr   # NaN propagates correctly

    return {"neutral": neu_arr, "protonated": prot_arr, "delta": delta}
