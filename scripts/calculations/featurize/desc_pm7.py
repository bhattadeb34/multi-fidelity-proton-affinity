"""
desc_pm7.py
===========
PM7 quantum chemical descriptors extracted from MOPAC calculations.

13 properties per molecular state (neutral, protonated, delta) = 39 features.

Properties
----------
  1.  HOMO_eV                  — highest occupied MO energy
  2.  LUMO_eV                  — lowest unoccupied MO energy
  3.  HOMO_LUMO_gap_eV         — HOMO-LUMO gap (chemical hardness proxy)
  4.  dipole_debye              — total dipole moment magnitude
  5.  dipole_x                 — x-component of dipole
  6.  dipole_y                 — y-component of dipole
  7.  dipole_z                 — z-component of dipole
  8.  HOF_kcalmol              — heat of formation (PM7 reference energy)
  9.  ionization_potential_eV  — Koopmans' theorem IP (= -HOMO_eV)
  10. cosmo_area               — COSMO solvation surface area
  11. cosmo_volume             — COSMO solvation volume
  12. total_energy_eV          — total electronic energy
  13. n_atoms                  — atom count (structural size descriptor)

Delta = protonated - neutral for all 13 properties.

Sources
-------
  PM7 dataset:  from pm7_dataset.json  (neutral + per-site values)
  DFT dataset:  partial overlap — DFT has HOMO/LUMO/gap/dipole but not
                HOF/IP/COSMO; those columns will be NaN for DFT records.

Output column naming:
  neutral_pm7_{prop}, protonated_pm7_{prop}, delta_pm7_{prop}
"""

import numpy as np

# The 13 PM7 properties, in fixed order
PM7_PROPERTIES = [
    "HOMO_eV",
    "LUMO_eV",
    "HOMO_LUMO_gap_eV",
    "dipole_debye",
    "dipole_x",
    "dipole_y",
    "dipole_z",
    "HOF_kcalmol",
    "ionization_potential_eV",
    "cosmo_area",
    "cosmo_volume",
    "total_energy_eV",
    "n_atoms",
]
N_PM7 = len(PM7_PROPERTIES)   # 13

NEUTRAL_PM7_COLS    = [f"neutral_pm7_{p}"    for p in PM7_PROPERTIES]
PROTONATED_PM7_COLS = [f"protonated_pm7_{p}" for p in PM7_PROPERTIES]
DELTA_PM7_COLS      = [f"delta_pm7_{p}"      for p in PM7_PROPERTIES]
ALL_PM7_COLS        = NEUTRAL_PM7_COLS + PROTONATED_PM7_COLS + DELTA_PM7_COLS  # 39


def _extract_state(state_dict: dict) -> np.ndarray:
    """
    Extract the 13 PM7 properties from a neutral or site dict.
    Missing values become NaN.
    """
    vals = np.empty(N_PM7, dtype=np.float64)
    for i, prop in enumerate(PM7_PROPERTIES):
        v = state_dict.get(prop)
        vals[i] = float(v) if v is not None else np.nan
    return vals


def pm7_features_from_record(
    neutral_dict: dict,
    site_dict: dict,
) -> dict[str, np.ndarray]:
    """
    Extract PM7 features for neutral, protonated, and delta states
    from a molecule record's neutral and site dicts.

    Parameters
    ----------
    neutral_dict : record['neutral']
    site_dict    : one element of record['all_sites']

    Returns
    -------
    dict with keys 'neutral', 'protonated', 'delta', each (13,) array.
    """
    neu_arr  = _extract_state(neutral_dict)
    prot_arr = _extract_state(site_dict)
    delta    = prot_arr - neu_arr

    return {"neutral": neu_arr, "protonated": prot_arr, "delta": delta}


def pm7_features_from_dft_record(
    neutral_dict: dict,
    site_dict: dict,
) -> dict[str, np.ndarray]:
    """
    Extract the PM7-compatible subset from a DFT record.
    DFT records have HOMO/LUMO/gap/dipole but not HOF/IP/COSMO/total_energy_eV
    (those will be NaN, indicating DFT-source rows in the combined dataset).

    Parameters
    ----------
    neutral_dict : record['neutral']  from DFT dataset
    site_dict    : one element of record['all_sites'] from DFT dataset

    Returns
    -------
    dict with keys 'neutral', 'protonated', 'delta', each (13,) array.
    """
    # DFT neutral key mapping -> PM7 property names
    DFT_NEUTRAL_MAP = {
        "HOMO_eV":          "HOMO_eV",
        "LUMO_eV":          "LUMO_eV",
        "HOMO_LUMO_gap_eV": "HOMO_LUMO_gap_eV",
        "dipole_debye":     "dipole_debye",
        "dipole_x":         "dipole_x",
        "dipole_y":         "dipole_y",
        "dipole_z":         "dipole_z",
        "n_atoms":          "n_atoms",
    }
    DFT_SITE_MAP = {
        "HOMO_eV":          "HOMO_eV",
        "LUMO_eV":          "LUMO_eV",
        "HOMO_LUMO_gap_eV": "HOMO_LUMO_gap_eV",
        "dipole_debye":     "dipole_debye",
        "n_atoms":          "n_atoms",
    }

    def _from_dft(source_dict, field_map):
        d = {prop: source_dict.get(dft_key) for dft_key, prop in field_map.items()}
        return _extract_state(d)

    neu_arr  = _from_dft(neutral_dict, DFT_NEUTRAL_MAP)
    prot_arr = _from_dft(site_dict,    DFT_SITE_MAP)
    delta    = prot_arr - neu_arr

    return {"neutral": neu_arr, "protonated": prot_arr, "delta": delta}
