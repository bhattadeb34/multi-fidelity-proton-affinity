"""
desc_mordred.py
===============
Mordred molecular descriptors — 2D (1613) and optionally 3D (213 additional).

State strategy
--------------
Two strategies are supported, controlled by the `state_strategy` parameter:

  'all_states'  (default):
      Compute all descriptors for neutral, protonated, and delta states.
      Total: 1613×3 = 4839 (2D only) or 1826×3 = 5478 (with 3D).
      Recommended when dataset size allows feature selection to prune.

  'neutral_full_delta_2d':
      2D descriptors for all 3 states, but 3D descriptors for neutral only.
      Rationale: 3D geometry changes upon protonation are already captured
      by RDKit and PM7 delta features. This halves the 3D-only column count.
      Total: 1613×3 + 213×1 = 5052.

3D geometry source
------------------
  Priority 1: DFT-optimized B3LYP/def2-TZVP coordinates (from folder records)
  Priority 2: ETKDG conformer generated on-the-fly from SMILES

Known NaN sources (handled downstream by feature selection)
------------------------------------------------------------
  ~383 descriptors (ABC, ABCGG, AATS*dv) are consistently NaN due to a
  numpy 1.x / mordred 1.2.0 compatibility issue (np.float removed in 1.24).
  These will have zero variance and are removed in the feature selection step.

Output columns
--------------
  neutral_mordred_{name}
  protonated_mordred_{name}   (all strategies)
  delta_mordred_{name}        (all strategies)
  mordred_geom_source         (companion metadata column, not a feature)
"""

import warnings
import logging
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

log = logging.getLogger(__name__)

try:
    from mordred import Calculator, descriptors as mordred_descriptors
    _MORDRED_AVAILABLE = True
except ImportError:
    _MORDRED_AVAILABLE = False
    log.warning("mordred not installed — all Mordred features will be NaN")

# ---------------------------------------------------------------------------
# Build calculators once at module load
# ---------------------------------------------------------------------------

if _MORDRED_AVAILABLE:
    _CALC_2D = Calculator(mordred_descriptors, ignore_3D=True)
    _CALC_3D = Calculator(mordred_descriptors, ignore_3D=False)

    MORDRED_2D_NAMES   = [str(d) for d in _CALC_2D.descriptors]   # 1613
    MORDRED_3D_NAMES   = [str(d) for d in _CALC_3D.descriptors]   # 1826

    _2D_SET = set(MORDRED_2D_NAMES)
    MORDRED_3D_ONLY_NAMES = [n for n in MORDRED_3D_NAMES if n not in _2D_SET]  # 213

    N_MORDRED_2D      = len(MORDRED_2D_NAMES)   # 1613
    N_MORDRED_3D      = len(MORDRED_3D_NAMES)   # 1826
    N_MORDRED_3D_ONLY = len(MORDRED_3D_ONLY_NAMES)  # 213
else:
    MORDRED_2D_NAMES      = []
    MORDRED_3D_NAMES      = []
    MORDRED_3D_ONLY_NAMES = []
    N_MORDRED_2D          = 0
    N_MORDRED_3D          = 0
    N_MORDRED_3D_ONLY     = 0

# Column name sets — always defined against the full 3D name list so CSV
# columns are consistent regardless of which states actually have 3D computed.
NEUTRAL_MORDRED_COLS    = [f"neutral_mordred_{n}"    for n in MORDRED_3D_NAMES]
PROTONATED_MORDRED_COLS = [f"protonated_mordred_{n}" for n in MORDRED_3D_NAMES]
DELTA_MORDRED_COLS      = [f"delta_mordred_{n}"      for n in MORDRED_3D_NAMES]
ALL_MORDRED_COLS        = NEUTRAL_MORDRED_COLS + PROTONATED_MORDRED_COLS + DELTA_MORDRED_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_array(result, n: int) -> np.ndarray:
    """
    Convert mordred Result to float array.
    mordred uses its own Error/Missing types (not Python exceptions).
    float() on them returns NaN — this is the intended behaviour.
    """
    arr = np.empty(n, dtype=np.float64)
    for i, v in enumerate(result.values()):
        try:
            arr[i] = float(v)
        except (TypeError, ValueError):
            arr[i] = np.nan
    return arr


def _mol_from_smiles(smiles: str):
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def _inject_dft_coords(mol, symbols: list, coords: list):
    """
    Inject DFT-optimized coordinates into mol (with Hs).
    Returns mol_with_H on success, None on atom-count mismatch.
    """
    if not coords or not symbols:
        return None
    mol_h = Chem.AddHs(mol)
    if len(coords) != mol_h.GetNumAtoms():
        return None
    conf = Chem.Conformer(mol_h.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol_h.RemoveAllConformers()
    mol_h.AddConformer(conf, assignId=True)
    return mol_h


def _etkdg_mol(mol):
    """Generate ETKDG conformer. Returns mol_with_H or None."""
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol_h, params) == -1:
        # fallback to basic ETKDG
        if AllChem.EmbedMolecule(mol_h, AllChem.ETKDG()) == -1:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        pass
    return mol_h


# ---------------------------------------------------------------------------
# Single-molecule calculators
# ---------------------------------------------------------------------------

def _compute_2d(smiles: str) -> np.ndarray | None:
    """1613 Mordred 2D descriptors for one SMILES."""
    if not _MORDRED_AVAILABLE:
        return None
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _to_array(_CALC_2D(mol), N_MORDRED_2D)


def _compute_3d(
    smiles: str,
    dft_symbols: list | None = None,
    dft_coords: list | None  = None,
) -> tuple[np.ndarray | None, str]:
    """
    1826 Mordred 3D descriptors for one SMILES.

    Returns (array, geom_source) where geom_source is 'dft', 'etkdg', or 'failed'.
    """
    if not _MORDRED_AVAILABLE:
        return None, "mordred_unavailable"
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None, "invalid_smiles"

    mol_3d   = None
    geom_src = "failed"

    if dft_symbols and dft_coords:
        mol_3d = _inject_dft_coords(mol, dft_symbols, dft_coords)
        if mol_3d is not None:
            geom_src = "dft"

    if mol_3d is None:
        mol_3d = _etkdg_mol(mol)
        if mol_3d is not None:
            geom_src = "etkdg"

    if mol_3d is None:
        return None, "failed"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arr = _to_array(_CALC_3D(mol_3d), N_MORDRED_3D)
    return arr, geom_src


def _pad_2d_to_3d(arr_2d: np.ndarray) -> np.ndarray:
    """
    Pad a 2D descriptor array (1613,) to the full 3D layout (1826,)
    by appending NaN for the 213 3D-only positions.
    Ensures consistent column count across all rows regardless of
    whether 3D was computed.
    """
    arr_3d = np.full(N_MORDRED_3D, np.nan)
    # 2D descriptors occupy the first N_MORDRED_2D positions in the 3D calc
    # (mordred preserves ordering — 3D-only descriptors are appended at the end)
    arr_3d[:N_MORDRED_2D] = arr_2d
    return arr_3d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mordred_three_states(
    neutral_smiles: str,
    protonated_smiles: str,
    neutral_dft_symbols: list | None  = None,
    neutral_dft_coords: list | None   = None,
    prot_dft_symbols: list | None     = None,
    prot_dft_coords: list | None      = None,
    compute_3d: bool                  = True,
    state_strategy: str               = "all_states",
) -> dict:
    """
    Compute Mordred descriptors for neutral, protonated, and delta states.

    Parameters
    ----------
    neutral_smiles, protonated_smiles : SMILES strings
    neutral_dft_symbols/coords        : DFT-optimized geometry for neutral mol
    prot_dft_symbols/coords           : DFT-optimized geometry for protonated mol
    compute_3d                        : include 3D descriptors (requires geometry)
    state_strategy                    : 'all_states' or 'neutral_full_delta_2d'
        'all_states'          — 3D for neutral, protonated, and delta
        'neutral_full_delta_2d' — 3D for neutral only; protonated/delta use
                                  2D padded to 3D layout (213 3D cols = NaN)

    Returns
    -------
    dict:
      'neutral'        : np.ndarray (N_MORDRED_3D,)
      'protonated'     : np.ndarray (N_MORDRED_3D,)
      'delta'          : np.ndarray (N_MORDRED_3D,)
      'geom_source'    : str  — geometry source for neutral
      'n_nan_neutral'  : int
      'n_nan_prot'     : int
    """
    N = N_MORDRED_3D  # always output full 3D-layout arrays for consistent columns
    nan_arr = np.full(N, np.nan)

    if not _MORDRED_AVAILABLE:
        return {"neutral": nan_arr, "protonated": nan_arr, "delta": nan_arr,
                "geom_source": "mordred_unavailable", "n_nan_neutral": N, "n_nan_prot": N}

    if not compute_3d:
        # 2D only — pad to 3D layout so column count is consistent
        neu_2d  = _compute_2d(neutral_smiles)
        prot_2d = _compute_2d(protonated_smiles)
        neu_arr  = _pad_2d_to_3d(neu_2d)  if neu_2d  is not None else nan_arr
        prot_arr = _pad_2d_to_3d(prot_2d) if prot_2d is not None else nan_arr
        delta    = prot_arr - neu_arr
        return {"neutral": neu_arr, "protonated": prot_arr, "delta": delta,
                "geom_source": "2d_only",
                "n_nan_neutral": int(np.isnan(neu_arr).sum()),
                "n_nan_prot":    int(np.isnan(prot_arr).sum())}

    # 3D for neutral always
    neu_arr, neu_src = _compute_3d(neutral_smiles, neutral_dft_symbols, neutral_dft_coords)
    neu_arr = neu_arr if neu_arr is not None else nan_arr

    if state_strategy == "neutral_full_delta_2d":
        # Protonated: 2D only, padded — 213 3D-only cols will be NaN
        prot_2d  = _compute_2d(protonated_smiles)
        prot_arr = _pad_2d_to_3d(prot_2d) if prot_2d is not None else nan_arr
        geom_src = f"{neu_src}+2d_prot"
    else:
        # all_states: full 3D for protonated too
        prot_arr, _ = _compute_3d(protonated_smiles, prot_dft_symbols, prot_dft_coords)
        prot_arr = prot_arr if prot_arr is not None else nan_arr
        geom_src = neu_src

    delta = prot_arr - neu_arr

    return {
        "neutral":       neu_arr,
        "protonated":    prot_arr,
        "delta":         delta,
        "geom_source":   geom_src,
        "n_nan_neutral": int(np.isnan(neu_arr).sum()),
        "n_nan_prot":    int(np.isnan(prot_arr).sum()),
    }
