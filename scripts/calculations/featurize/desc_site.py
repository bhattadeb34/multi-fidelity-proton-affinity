"""
desc_site.py
============
Site-specific descriptors encoding protonation site characteristics.

6 features total:
  1-3. One-hot encoded element type: is_N, is_O, is_S
       (covers the three most common protonation sites in organic molecules)
  4.   site_index (raw integer index of the protonated atom)
  5.   normalized_site_index = site_index / n_atoms
       (encodes relative position in the molecule)
  6.   n_sites (total number of protonation sites per molecule)

These features are site-level, not molecule-level — each (mol, site) row
gets its own values.

Output columns:
  site_is_N, site_is_O, site_is_S,
  site_index, site_normalized_index, site_n_sites
"""

import numpy as np

SITE_COLS = [
    "site_is_N",
    "site_is_O",
    "site_is_S",
    "site_index",
    "site_normalized_index",
    "site_n_sites",
]
N_SITE = len(SITE_COLS)   # 6

# Elements covered by one-hot encoding
_ONEHOT_ELEMENTS = ["N", "O", "S"]


def site_features(
    site_element: str,
    site_index: int | None,
    n_atoms: int | None,
    n_sites: int,
) -> np.ndarray:
    """
    Compute 6 site-specific features for one (mol, site) row.

    Parameters
    ----------
    site_element : element symbol of the protonated atom (e.g. 'N', 'O', 'S')
    site_index   : integer index of the protonated atom in the molecule
    n_atoms      : total number of heavy atoms + H in the neutral molecule
    n_sites      : total number of protonation sites for this molecule

    Returns
    -------
    np.ndarray of shape (6,)
    """
    elem = str(site_element).strip().upper() if site_element else ""

    # one-hot
    is_N = 1.0 if elem == "N" else 0.0
    is_O = 1.0 if elem == "O" else 0.0
    is_S = 1.0 if elem == "S" else 0.0

    # site index (use NaN if not available)
    idx = float(site_index) if site_index is not None else np.nan

    # normalized index
    if site_index is not None and n_atoms and n_atoms > 0:
        norm_idx = float(site_index) / float(n_atoms)
    else:
        norm_idx = np.nan

    return np.array([is_N, is_O, is_S, idx, norm_idx, float(n_sites)],
                    dtype=np.float64)


def site_features_from_record(
    site_dict: dict,
    neutral_dict: dict,
    n_sites: int,
) -> np.ndarray:
    """
    Convenience wrapper: extract site features directly from record dicts.

    Parameters
    ----------
    site_dict    : one element of record['all_sites']
    neutral_dict : record['neutral']
    n_sites      : record['metadata']['n_sites']
    """
    element   = site_dict.get("site_name", "")
    # site_name in folder records is e.g. "site_1"; element is the atom letter
    # in JSON/PM7 records site_name is the element letter directly
    # Normalise: if it looks like 'site_1' extract element from protonated SMILES
    if element.lower().startswith("site_"):
        element = _element_from_smiles(site_dict.get("protonated_smiles", ""))

    site_idx = site_dict.get("site_idx")
    n_atoms  = (neutral_dict.get("n_atoms")
                or site_dict.get("n_atoms"))

    return site_features(element, site_idx, n_atoms, n_sites)


def _element_from_smiles(protonated_smiles: str) -> str:
    """
    Infer the protonated atom's element from the protonated SMILES by
    looking for [NH*+], [OH*+], [SH*+] patterns.
    Falls back to '' if not found.
    """
    if not protonated_smiles:
        return ""
    import re
    # Common protonated patterns
    if re.search(r'\[NH?\d*\+\]|\[NH\d*\+\d*\]', protonated_smiles):
        return "N"
    if re.search(r'\[OH?\d*\+\]|\[OH\d*\+\d*\]', protonated_smiles):
        return "O"
    if re.search(r'\[SH?\d*\+\]|\[SH\d*\+\d*\]', protonated_smiles):
        return "S"
    return ""
