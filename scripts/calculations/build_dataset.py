"""
build_dataset.py
================
Parses B3LYP/def2-TZVP DFT data from two sources and builds a unified dataset.

Sources
-------
  folder  : ../data/kmeans251/b3lyp_dft/molecule_folders/mol_XXXXX/
  json    : ../data/nist1185/b3lyp_dft/results/mol_XXXXX.json

Outputs  (written to ../data/processed/)
-------
  dataset.json     — full archive keyed by unique record_id
  features.csv     — flat table, one row per (record_id, site_idx)
  parse_report.json — skipped files, warnings, summary stats

PA formula
----------
  PA (kJ/mol) = (H_neutral - H_protonated) * 2625.5 + H_Hplus_kjmol
  H(H+)       = 5/2 * R * T = 5/2 * 8.314e-3 * 298.15 = 6.197 kJ/mol
  1 kJ/mol    = 0.239006 kcal/mol

Units stored: kJ/mol (native). kcal/mol provided as derived fields.
"""

import json
import re
import csv
import logging
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).parent
DATA_DIR     = SCRIPT_DIR.parent.parent / "data"
FOLDER_ROOT  = DATA_DIR / "kmeans251" / "b3lyp_dft" / "molecule_folders"
JSON_ROOT    = DATA_DIR / "nist1185" / "b3lyp_dft" / "results"
OUT_DIR      = DATA_DIR / "processed"

HA_TO_KJMOL  = 2625.5          # Hartree -> kJ/mol
H_HPLUS_KJMOL = 6.197          # 5/2 * R * T at 298.15 K, kJ/mol
KJMOL_TO_KCAL = 0.239006       # kJ/mol -> kcal/mol
RT_298         = 298.15 * 8.314e-3  # kJ/mol, used for documentation only

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — PA calculation
# ---------------------------------------------------------------------------

def compute_pa_kjmol(H_neutral_Ha: float, H_prot_Ha: float) -> float:
    """PA = (H_neutral - H_prot) * 2625.5 + H(H+), all in kJ/mol."""
    return (H_neutral_Ha - H_prot_Ha) * HA_TO_KJMOL + H_HPLUS_KJMOL


def kjmol_to_kcal(val: float | None) -> float | None:
    if val is None:
        return None
    return round(val * KJMOL_TO_KCAL, 4)


# ---------------------------------------------------------------------------
# Helpers — folder parsing
# ---------------------------------------------------------------------------

def _parse_log(log_path: Path) -> dict:
    """Extract scalars from a PySCF DFT summary .log file."""
    text = log_path.read_text()

    def _float(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    def _int(pattern):
        m = re.search(pattern, text)
        return int(m.group(1)) if m else None

    smiles   = None
    m = re.search(r"SMILES:\s+(\S+)", text)
    if m:
        smiles = m.group(1)

    charge   = _int(r"Charge:\s+(-?\d+)")
    n_atoms  = _int(r"N atoms:\s+(\d+)")
    n_elec   = _int(r"N electrons:\s+(\d+)")
    n_basis  = _int(r"N basis:\s+(\d+)")

    E_elec   = _float(r"E\(elec\)\s+=\s+([-\d.]+)")
    nuc_rep  = _float(r"Nuclear repul\.\s+=\s+([-\d.]+)")
    ZPE_kjmol= _float(r"ZPE\s+=\s+([\d.]+) kJ/mol")
    H_total  = _float(r"H\(total\)\s+=\s+([-\d.]+) Ha")

    HOMO_eV  = _float(r"HOMO\s+=\s+([-\d.]+) eV")
    LUMO_eV  = _float(r"LUMO\s+=\s+([-\d.]+) eV")
    gap_eV   = _float(r"HOMO-LUMO gap\s+=\s+([-\d.]+) eV")

    dip_mag  = _float(r"\|mu\|\s+=\s+([\d.]+) Debye")
    dip_x    = _float(r"mu_x\s+=\s+([-\d.]+)")
    dip_y    = _float(r"mu_y\s+=\s+([-\d.]+)")
    dip_z    = _float(r"mu_z\s+=\s+([-\d.]+)")

    n_imag   = _int(r"Imaginary frequencies:\s+(\d+)")
    wall     = _float(r"Wall time:\s+([\d.]+) s")

    # ZPE in Hartree (back-convert for H_total consistency check)
    ZPE_Ha   = ZPE_kjmol / HA_TO_KJMOL if ZPE_kjmol is not None else None

    return {
        "smiles": smiles,
        "charge": charge,
        "n_atoms": n_atoms,
        "n_electrons": n_elec,
        "n_basis": n_basis,
        "E_elec_Ha": E_elec,
        "nuclear_repulsion_Ha": nuc_rep,
        "ZPE_Ha": ZPE_Ha,
        "ZPE_kjmol": ZPE_kjmol,
        "H_total_Ha": H_total,
        "HOMO_eV": HOMO_eV,
        "LUMO_eV": LUMO_eV,
        "HOMO_LUMO_gap_eV": gap_eV,
        "dipole_debye": dip_mag,
        "dipole_x": dip_x,
        "dipole_y": dip_y,
        "dipole_z": dip_z,
        "n_imaginary": n_imag,
        "wall_time_s": wall,
    }


def _parse_freq(freq_path: Path) -> dict:
    """Extract frequency list (unscaled cm-1) from _freq.txt."""
    freqs = []
    for line in freq_path.read_text().splitlines():
        parts = line.split()
        # lines look like: "    1    22.9057    22.5621"
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                freqs.append(float(parts[1]))   # unscaled
            except ValueError:
                pass
    result = {"frequencies_cm": freqs}
    if freqs:
        result["freq_min_cm"] = min(freqs)
        result["freq_max_cm"] = max(freqs)
        result["n_low_freq"]  = sum(1 for f in freqs if f < 100)
    else:
        result["freq_min_cm"] = None
        result["freq_max_cm"] = None
        result["n_low_freq"]  = None
    return result


def _parse_xyz(xyz_path: Path) -> dict:
    """Extract optimised coordinates from .xyz file."""
    lines = xyz_path.read_text().splitlines()
    symbols, coords = [], []
    for line in lines[2:]:   # skip count and comment
        parts = line.split()
        if len(parts) == 4:
            symbols.append(parts[0])
            coords.append([float(x) for x in parts[1:]])
    return {"opt_coords_symbols": symbols, "opt_coords_angstrom": coords}


def parse_folder_molecule(mol_dir: Path) -> dict | None:
    """
    Parse one mol_XXXXX folder. Returns a standardised molecule record
    (source='folder') or None if critical files are missing.
    """
    mol_id = mol_dir.name
    warnings = []

    # --- neutral ---
    neutral_log  = mol_dir / "neutral" / "neutral.log"
    neutral_freq = mol_dir / "neutral" / "neutral_freq.txt"
    neutral_xyz  = mol_dir / "neutral" / "neutral_optimized.xyz"

    if not neutral_log.exists():
        return None   # can't do anything without this

    neutral = _parse_log(neutral_log)
    if neutral_freq.exists():
        neutral.update(_parse_freq(neutral_freq))
    else:
        warnings.append("neutral_freq.txt missing")
    if neutral_xyz.exists():
        neutral.update(_parse_xyz(neutral_xyz))

    # --- sites ---
    site_dirs = sorted(mol_dir.glob("site_*"))
    if not site_dirs:
        warnings.append("no site directories found")
        return None

    all_sites = []
    best_site = None
    best_pa   = None

    for site_dir in site_dirs:
        site_name = site_dir.name   # e.g. site_1
        m = re.search(r"(\d+)$", site_name)
        site_idx = int(m.group(1)) if m else -1

        # find log/freq/xyz with flexible naming
        logs  = list(site_dir.glob("protonated_*.log"))
        freqs = list(site_dir.glob("protonated_*_freq.txt"))
        xyzs  = list(site_dir.glob("protonated_*_optimized.xyz"))

        if not logs:
            warnings.append(f"{site_name}: no .log found, skipping")
            continue

        prot = _parse_log(logs[0])
        if freqs:
            prot.update(_parse_freq(freqs[0]))
        else:
            warnings.append(f"{site_name}: freq.txt missing")
        if xyzs:
            prot.update(_parse_xyz(xyzs[0]))

        # PA
        H_n = neutral.get("H_total_Ha")
        H_p = prot.get("H_total_Ha")
        pa  = compute_pa_kjmol(H_n, H_p) if (H_n and H_p) else None

        # delta features
        gap_n = neutral.get("HOMO_LUMO_gap_eV")
        gap_p = prot.get("HOMO_LUMO_gap_eV")
        dip_n = neutral.get("dipole_debye")
        dip_p = prot.get("dipole_debye")
        zpe_n = neutral.get("ZPE_kjmol")
        zpe_p = prot.get("ZPE_kjmol")

        site_record = {
            "site_idx":           site_idx,
            "site_name":          site_name,
            "protonated_smiles":  prot.get("smiles"),
            "status":             "OK" if pa is not None else "FAILED",
            "pa_kjmol":           round(pa, 4) if pa else None,
            "pa_kcalmol":         kjmol_to_kcal(pa),
            # protonated state scalars
            "H_total_Ha":         prot.get("H_total_Ha"),
            "HOMO_eV":            prot.get("HOMO_eV"),
            "LUMO_eV":            prot.get("LUMO_eV"),
            "HOMO_LUMO_gap_eV":   prot.get("HOMO_LUMO_gap_eV"),
            "dipole_debye":       prot.get("dipole_debye"),
            "ZPE_kjmol":          prot.get("ZPE_kjmol"),
            "n_imaginary":        prot.get("n_imaginary"),
            "freq_min_cm":        prot.get("freq_min_cm"),
            "freq_max_cm":        prot.get("freq_max_cm"),
            "n_low_freq":         prot.get("n_low_freq"),
            "n_atoms":            prot.get("n_atoms"),
            "n_electrons":        prot.get("n_electrons"),
            "n_basis":            prot.get("n_basis"),
            # delta features (DFT captures what PM7 cannot)
            "delta_HOMO_LUMO_gap_eV": round(gap_p - gap_n, 6) if (gap_p and gap_n) else None,
            "delta_dipole_debye":     round(dip_p - dip_n, 6) if (dip_p and dip_n) else None,
            "delta_ZPE_kjmol":        round(zpe_p - zpe_n, 6) if (zpe_p and zpe_n) else None,
            # full freq list for future use
            "frequencies_cm":     prot.get("frequencies_cm", []),
            "opt_coords_symbols": prot.get("opt_coords_symbols", []),
            "opt_coords_angstrom":prot.get("opt_coords_angstrom", []),
        }
        all_sites.append(site_record)

        if pa is not None and (best_pa is None or pa > best_pa):
            best_pa   = pa
            best_site = site_record

    if not all_sites:
        return None

    return {
        "record_id":  f"{mol_id}_folder",
        "mol_id":     mol_id,
        # ---- METADATA ----
        "metadata": {
            "source":       "folder",
            "mol_dir":      str(mol_dir.relative_to(SCRIPT_DIR.parent.parent)),
            "dft_level":    "B3LYP/def2-TZVP",
            "has_freq":     True,
            "n_sites":      len(all_sites),
            "best_site_idx":best_site["site_idx"] if best_site else None,
            "warnings":     warnings,
            "parsed_at":    datetime.now(timezone.utc).isoformat(),
        },
        # ---- LABELS ----
        "labels": {
            "exp_pa_kjmol":      None,   # merged from external source later
            "exp_pa_kcalmol":    None,
            "dft_pa_kjmol":      round(best_pa, 4) if best_pa else None,
            "dft_pa_kcalmol":    kjmol_to_kcal(best_pa),
            "pm7_pa_kjmol":      None,   # merged from PM7 results later
            "pm7_pa_kcalmol":    None,
            "delta_dft_exp":     None,   # filled after exp_pa merge
            "delta_pm7_exp":     None,   # ML target — filled after PM7 merge
            "dft_correction":    None,   # dft_pa - pm7_pa, filled after PM7 merge
        },
        # ---- NEUTRAL STATE ----
        "neutral": {
            "smiles":             neutral.get("smiles"),
            "charge":             neutral.get("charge"),
            "H_total_Ha":         neutral.get("H_total_Ha"),
            "ZPE_kjmol":          neutral.get("ZPE_kjmol"),
            "HOMO_eV":            neutral.get("HOMO_eV"),
            "LUMO_eV":            neutral.get("LUMO_eV"),
            "HOMO_LUMO_gap_eV":   neutral.get("HOMO_LUMO_gap_eV"),
            "dipole_debye":       neutral.get("dipole_debye"),
            "dipole_x":           neutral.get("dipole_x"),
            "dipole_y":           neutral.get("dipole_y"),
            "dipole_z":           neutral.get("dipole_z"),
            "n_atoms":            neutral.get("n_atoms"),
            "n_electrons":        neutral.get("n_electrons"),
            "n_basis":            neutral.get("n_basis"),
            "n_imaginary":        neutral.get("n_imaginary"),
            "freq_min_cm":        neutral.get("freq_min_cm"),
            "freq_max_cm":        neutral.get("freq_max_cm"),
            "n_low_freq":         neutral.get("n_low_freq"),
            "frequencies_cm":     neutral.get("frequencies_cm", []),
            "opt_coords_symbols": neutral.get("opt_coords_symbols", []),
            "opt_coords_angstrom":neutral.get("opt_coords_angstrom", []),
        },
        # ---- ALL SITES ----
        "all_sites": all_sites,
    }


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

def parse_json_molecule(json_path: Path) -> dict | None:
    """
    Load a pre-computed JSON record and reformat into the unified schema.
    source='json'.
    """
    try:
        data = json.loads(json_path.read_text())
    except Exception as e:
        log.warning(f"Failed to load {json_path.name}: {e}")
        return None

    mol_id   = f"mol_{data.get('global_idx', 0):05d}"
    warnings = []

    # neutral
    n = data.get("neutral", {})
    neutral_out = {
        "smiles":             data.get("smiles"),
        "charge":             n.get("charge", 0),
        "H_total_Ha":         n.get("H_total"),
        "ZPE_kjmol":          n.get("ZPE_kjmol"),
        "HOMO_eV":            n.get("HOMO_eV"),
        "LUMO_eV":            n.get("LUMO_eV"),
        "HOMO_LUMO_gap_eV":   n.get("HOMO_LUMO_gap_eV"),
        "dipole_debye":       n.get("dipole_debye"),
        "dipole_x":           n.get("dipole_x"),
        "dipole_y":           n.get("dipole_y"),
        "dipole_z":           n.get("dipole_z"),
        "n_atoms":            n.get("n_atoms"),
        "n_electrons":        n.get("n_electrons"),
        "n_basis":            n.get("n_basis"),
        "n_imaginary":        n.get("n_imaginary"),
        "frequencies_cm":     n.get("frequencies_cm", []),
        "opt_coords_symbols": n.get("opt_coords_symbols", []),
        "opt_coords_angstrom":n.get("opt_coords_angstrom", []),
    }
    freqs_n = n.get("frequencies_cm", [])
    neutral_out["freq_min_cm"] = min(freqs_n) if freqs_n else None
    neutral_out["freq_max_cm"] = max(freqs_n) if freqs_n else None
    neutral_out["n_low_freq"]  = sum(1 for f in freqs_n if f < 100) if freqs_n else None

    H_n = neutral_out["H_total_Ha"]

    # all sites
    all_sites = []
    best_site = None
    best_pa   = None

    gap_n = neutral_out.get("HOMO_LUMO_gap_eV")
    dip_n = neutral_out.get("dipole_debye")
    zpe_n = neutral_out.get("ZPE_kjmol")

    for raw_site in data.get("all_sites", []):
        H_p  = raw_site.get("H_total")
        pa   = compute_pa_kjmol(H_n, H_p) if (H_n and H_p) else raw_site.get("pa_kjmol")

        gap_p = raw_site.get("HOMO_eV")    # note: JSON stores HOMO not gap at site level
        # JSON all_sites has HOMO_eV and LUMO_eV, so compute gap
        homo_p = raw_site.get("HOMO_eV")
        lumo_p = raw_site.get("LUMO_eV")
        gap_p  = round(lumo_p - homo_p, 6) if (homo_p and lumo_p) else None
        dip_p  = raw_site.get("dipole_debye")
        zpe_p  = raw_site.get("ZPE_kjmol")

        freqs_p = []   # not stored per-site in JSON format
        site_record = {
            "site_idx":            None,   # not indexed in JSON format
            "site_name":           raw_site.get("atom", ""),
            "protonated_smiles":   raw_site.get("protonated_smiles"),
            "status":              raw_site.get("status", "OK"),
            "pa_kjmol":            round(pa, 4) if pa else None,
            "pa_kcalmol":          kjmol_to_kcal(pa),
            "H_total_Ha":          H_p,
            "HOMO_eV":             homo_p,
            "LUMO_eV":             lumo_p,
            "HOMO_LUMO_gap_eV":    gap_p,
            "dipole_debye":        dip_p,
            "ZPE_kjmol":           zpe_p,
            "n_imaginary":         None,
            "freq_min_cm":         None,
            "freq_max_cm":         None,
            "n_low_freq":          None,
            "n_atoms":             None,
            "n_electrons":         None,
            "n_basis":             None,
            "delta_HOMO_LUMO_gap_eV": round(gap_p - gap_n, 6) if (gap_p and gap_n) else None,
            "delta_dipole_debye":     round(dip_p - dip_n, 6) if (dip_p and dip_n) else None,
            "delta_ZPE_kjmol":        round(zpe_p - zpe_n, 6) if (zpe_p and zpe_n) else None,
            "frequencies_cm":         freqs_p,
            "opt_coords_symbols":     [],
            "opt_coords_angstrom":    [],
        }
        all_sites.append(site_record)

        if pa is not None and (best_pa is None or pa > best_pa):
            best_pa   = pa
            best_site = site_record

    if not all_sites:
        warnings.append("no sites found in JSON")

    exp_pa    = data.get("exp_pa")        # kJ/mol in JSON
    dft_pa    = round(best_pa, 4) if best_pa else data.get("dft_pa")
    delta_de  = round(abs(dft_pa - exp_pa), 4) if (dft_pa and exp_pa) else None

    return {
        "record_id":  f"{mol_id}_json",
        "mol_id":     mol_id,
        "metadata": {
            "source":        "json",
            "json_path":     str(json_path.relative_to(SCRIPT_DIR.parent.parent)),
            "dft_level":     data.get("level", "B3LYP/def2-TZVP"),
            "has_freq":      data.get("freq", False),
            "n_sites":       data.get("n_sites", len(all_sites)),
            "best_site_idx": None,
            "warnings":      warnings,
            "parsed_at":     datetime.now(timezone.utc).isoformat(),
        },
        "labels": {
            "exp_pa_kjmol":   exp_pa,
            "exp_pa_kcalmol": kjmol_to_kcal(exp_pa),
            "dft_pa_kjmol":   dft_pa,
            "dft_pa_kcalmol": kjmol_to_kcal(dft_pa),
            "pm7_pa_kjmol":   None,
            "pm7_pa_kcalmol": None,
            "delta_dft_exp":  delta_de,
            "delta_pm7_exp":  None,
            "dft_correction": None,
        },
        "neutral":   neutral_out,
        "all_sites": all_sites,
    }


# ---------------------------------------------------------------------------
# CSV row builder
# ---------------------------------------------------------------------------

# Columns for features.csv — one row per (record_id, site)
CSV_COLUMNS = [
    # identifiers
    "record_id", "mol_id", "source", "site_idx", "site_name",
    "neutral_smiles", "protonated_smiles",
    # labels
    "exp_pa_kjmol", "exp_pa_kcalmol",
    "dft_pa_kjmol", "dft_pa_kcalmol",
    "pm7_pa_kjmol", "pm7_pa_kcalmol",
    "delta_dft_exp", "delta_pm7_exp", "dft_correction",
    # neutral features
    "neutral_HOMO_eV", "neutral_LUMO_eV", "neutral_HOMO_LUMO_gap_eV",
    "neutral_dipole_debye", "neutral_ZPE_kjmol",
    "neutral_freq_min_cm", "neutral_freq_max_cm", "neutral_n_low_freq",
    "neutral_n_atoms", "neutral_n_electrons", "neutral_n_basis",
    "neutral_n_imaginary",
    # protonated site features
    "prot_HOMO_eV", "prot_LUMO_eV", "prot_HOMO_LUMO_gap_eV",
    "prot_dipole_debye", "prot_ZPE_kjmol",
    "prot_freq_min_cm", "prot_freq_max_cm", "prot_n_low_freq",
    "prot_n_atoms", "prot_n_electrons", "prot_n_basis",
    "prot_n_imaginary",
    # delta features (most informative for ML)
    "delta_HOMO_LUMO_gap_eV",
    "delta_dipole_debye",
    "delta_ZPE_kjmol",
]


def record_to_csv_rows(record: dict) -> list[dict]:
    """Expand one record into one CSV row per site."""
    rows = []
    lbl  = record["labels"]
    neu  = record["neutral"]
    meta = record["metadata"]

    base = {
        "record_id":           record["record_id"],
        "mol_id":              record["mol_id"],
        "source":              meta["source"],
        "neutral_smiles":      neu.get("smiles"),
        "exp_pa_kjmol":        lbl.get("exp_pa_kjmol"),
        "exp_pa_kcalmol":      lbl.get("exp_pa_kcalmol"),
        "dft_pa_kjmol":        lbl.get("dft_pa_kjmol"),
        "dft_pa_kcalmol":      lbl.get("dft_pa_kcalmol"),
        "pm7_pa_kjmol":        lbl.get("pm7_pa_kjmol"),
        "pm7_pa_kcalmol":      lbl.get("pm7_pa_kcalmol"),
        "delta_dft_exp":       lbl.get("delta_dft_exp"),
        "delta_pm7_exp":       lbl.get("delta_pm7_exp"),
        "dft_correction":      lbl.get("dft_correction"),
        "neutral_HOMO_eV":            neu.get("HOMO_eV"),
        "neutral_LUMO_eV":            neu.get("LUMO_eV"),
        "neutral_HOMO_LUMO_gap_eV":   neu.get("HOMO_LUMO_gap_eV"),
        "neutral_dipole_debye":        neu.get("dipole_debye"),
        "neutral_ZPE_kjmol":           neu.get("ZPE_kjmol"),
        "neutral_freq_min_cm":         neu.get("freq_min_cm"),
        "neutral_freq_max_cm":         neu.get("freq_max_cm"),
        "neutral_n_low_freq":          neu.get("n_low_freq"),
        "neutral_n_atoms":             neu.get("n_atoms"),
        "neutral_n_electrons":         neu.get("n_electrons"),
        "neutral_n_basis":             neu.get("n_basis"),
        "neutral_n_imaginary":         neu.get("n_imaginary"),
    }

    for site in record["all_sites"]:
        row = {**base}
        row["site_idx"]              = site.get("site_idx")
        row["site_name"]             = site.get("site_name")
        row["protonated_smiles"]     = site.get("protonated_smiles")
        row["prot_HOMO_eV"]          = site.get("HOMO_eV")
        row["prot_LUMO_eV"]          = site.get("LUMO_eV")
        row["prot_HOMO_LUMO_gap_eV"] = site.get("HOMO_LUMO_gap_eV")
        row["prot_dipole_debye"]     = site.get("dipole_debye")
        row["prot_ZPE_kjmol"]        = site.get("ZPE_kjmol")
        row["prot_freq_min_cm"]      = site.get("freq_min_cm")
        row["prot_freq_max_cm"]      = site.get("freq_max_cm")
        row["prot_n_low_freq"]       = site.get("n_low_freq")
        row["prot_n_atoms"]          = site.get("n_atoms")
        row["prot_n_electrons"]      = site.get("n_electrons")
        row["prot_n_basis"]          = site.get("n_basis")
        row["prot_n_imaginary"]      = site.get("n_imaginary")
        row["delta_HOMO_LUMO_gap_eV"]= site.get("delta_HOMO_LUMO_gap_eV")
        row["delta_dipole_debye"]    = site.get("delta_dipole_debye")
        row["delta_ZPE_kjmol"]       = site.get("delta_ZPE_kjmol")
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset       = {}   # record_id -> record
    parse_report  = {"skipped": [], "warnings": {}, "summary": {}}
    csv_rows      = []

    # ---- Parse folder-based molecules (251) ----
    folder_dirs = sorted(FOLDER_ROOT.glob("mol_*"))
    log.info(f"Found {len(folder_dirs)} molecule folders in {FOLDER_ROOT.relative_to(SCRIPT_DIR.parent.parent)}")

    folder_ok = folder_skip = 0
    for mol_dir in folder_dirs:
        record = parse_folder_molecule(mol_dir)
        if record is None:
            parse_report["skipped"].append({"id": mol_dir.name, "source": "folder", "reason": "missing critical files"})
            folder_skip += 1
            continue
        if record["metadata"]["warnings"]:
            parse_report["warnings"][record["record_id"]] = record["metadata"]["warnings"]
        dataset[record["record_id"]] = record
        csv_rows.extend(record_to_csv_rows(record))
        folder_ok += 1

    log.info(f"  Folder: {folder_ok} OK, {folder_skip} skipped")

    # ---- Parse JSON molecules (1185) ----
    json_files = sorted(JSON_ROOT.glob("mol_*.json"))
    log.info(f"Found {len(json_files)} JSON files in {JSON_ROOT.relative_to(SCRIPT_DIR.parent.parent)}")

    json_ok = json_skip = 0
    for jf in json_files:
        record = parse_json_molecule(jf)
        if record is None:
            parse_report["skipped"].append({"id": jf.stem, "source": "json", "reason": "parse error"})
            json_skip += 1
            continue
        if record["metadata"]["warnings"]:
            parse_report["warnings"][record["record_id"]] = record["metadata"]["warnings"]
        dataset[record["record_id"]] = record
        csv_rows.extend(record_to_csv_rows(record))
        json_ok += 1

    log.info(f"  JSON:   {json_ok} OK, {json_skip} skipped")

    # ---- Write dataset.json ----
    dataset_path = OUT_DIR / "dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)
    log.info(f"Wrote {dataset_path.relative_to(SCRIPT_DIR.parent.parent)}  ({len(dataset)} records)")

    # ---- Write features.csv ----
    csv_path = OUT_DIR / "features.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info(f"Wrote {csv_path.relative_to(SCRIPT_DIR.parent.parent)}  ({len(csv_rows)} rows)")

    # ---- Write parse_report.json ----
    parse_report["summary"] = {
        "total_records":   len(dataset),
        "folder_records":  folder_ok,
        "json_records":    json_ok,
        "total_skipped":   folder_skip + json_skip,
        "total_csv_rows":  len(csv_rows),
        "run_at":          datetime.now(timezone.utc).isoformat(),
    }
    report_path = OUT_DIR / "parse_report.json"
    with open(report_path, "w") as f:
        json.dump(parse_report, f, indent=2)
    log.info(f"Wrote {report_path.relative_to(SCRIPT_DIR.parent.parent)}")

    # ---- Summary ----
    print("\n" + "="*55)
    print(f"  Records:   {len(dataset)}  ({folder_ok} folder + {json_ok} JSON)")
    print(f"  CSV rows:  {len(csv_rows)}  (one per site)")
    print(f"  Skipped:   {folder_skip + json_skip}")
    print(f"  Outputs:   {OUT_DIR.relative_to(SCRIPT_DIR.parent.parent)}/")
    print("="*55)


if __name__ == "__main__":
    main()
