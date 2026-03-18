"""
build_pm7_dataset.py
====================
Extracts PM7 semiempirical data from two source files and builds a unified
dataset in the same schema as build_dataset.py (DFT), enabling direct
comparison and ML feature alignment.

Sources  (relative to script location, expected in ../data/pm7/)
-------
  pm7_nist1185_pa_per_site.csv   — 1155 molecules, 1867 site rows  (NIST set)
  pm7_kmeans251_pa_per_site.csv  —  252 molecules,  823 site rows  (k-means set)

PA formula (PM7 convention)
---------------------------
  PA (kcal/mol) = HOF_neutral - HOF_protonated + H(H+)_PM7
  H(H+)_PM7    = 365.7 kcal/mol  (standard PM7 proton heat of formation)
  1 kcal/mol   = 4.184 kJ/mol

  Note: HOF (heat of formation) in PM7 plays the role of H_total in DFT.
  The H(H+) correction differs from DFT (6.197 kJ/mol = 1.481 kcal/mol)
  because PM7 references HOF, not absolute enthalpy.

PM7-specific features (no DFT equivalent)
------------------------------------------
  cosmo_area, cosmo_volume   — solvation surface descriptors
  ionization_potential       — Koopmans' theorem IP from HOMO
  molecular_weight
  spin_state                 — closed_shell vs open_shell (open = radical cation)
  heat_of_formation (HOF)    — PM7 formation enthalpy, kcal/mol

Outputs  (written to ../data/processed/)
-------
  pm7_dataset.json     — full archive keyed by record_id
  pm7_features.csv     — flat table, one row per (record_id, site)
  pm7_parse_report.json
"""

import json
import csv
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR      = Path(__file__).parent
DATA_DIR        = SCRIPT_DIR.parent / "data"
PM7_DIR         = DATA_DIR / "pm7"
OUT_DIR         = DATA_DIR / "processed"

H_HPLUS_PM7_KCAL = 365.7        # kcal/mol, PM7 proton heat of formation
KCAL_TO_KJMOL    = 4.184
KJMOL_TO_KCAL    = 1 / KCAL_TO_KJMOL

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column maps  (raw CSV col -> unified schema name)
# ---------------------------------------------------------------------------

# NIST file (prefix style: neutral_*, protonated_*)
NIST_NEUTRAL_MAP = {
    "neutral_homo_ev":           "HOMO_eV",
    "neutral_lumo_ev":           "LUMO_eV",
    "neutral_gap_ev":            "HOMO_LUMO_gap_eV",
    "neutral_dipole_moment":     "dipole_debye",
    "neutral_dipole_x":          "dipole_x",
    "neutral_dipole_y":          "dipole_y",
    "neutral_dipole_z":          "dipole_z",
    "neutral_num_atoms":         "n_atoms",
    "neutral_heat_of_formation": "HOF_kcalmol",
    "neutral_total_energy_kcal_mol": "total_energy_kcalmol",
    "neutral_total_energy_ev":   "total_energy_eV",
    "neutral_ionization_potential":  "ionization_potential_eV",
    "neutral_cosmo_area":        "cosmo_area",
    "neutral_cosmo_volume":      "cosmo_volume",
    "neutral_molecular_weight":  "molecular_weight",
    "neutral_spin_state":        "spin_state",
    "neutral_computation_time":  "wall_time_s",
}

NIST_PROT_MAP = {
    "protonated_homo_ev":        "HOMO_eV",
    "protonated_lumo_ev":        "LUMO_eV",
    "protonated_gap_ev":         "HOMO_LUMO_gap_eV",
    "protonated_dipole_moment":  "dipole_debye",
    "protonated_dipole_x":       "dipole_x",
    "protonated_dipole_y":       "dipole_y",
    "protonated_dipole_z":       "dipole_z",
    "protonated_num_atoms":      "n_atoms",
    "protonated_heat_of_formation": "HOF_kcalmol",
    "protonated_total_energy_kcal_mol": "total_energy_kcalmol",
    "protonated_total_energy_ev":    "total_energy_eV",
    "protonated_ionization_potential":   "ionization_potential_eV",
    "protonated_cosmo_area":     "cosmo_area",
    "protonated_cosmo_volume":   "cosmo_volume",
    "protonated_molecular_weight":   "molecular_weight",
    "protonated_spin_state":     "spin_state",
    "protonated_computation_time":   "wall_time_s",
    "protonated_charge":         "charge",
    "protonated_multiplicity":   "multiplicity",
    "protonated_unpaired_electrons": "unpaired_electrons",
}

# k-means master file (suffix style: *_neutral, *_protonated)
KMEANS_NEUTRAL_MAP = {
    "homo_ev_neutral":           "HOMO_eV",
    "lumo_ev_neutral":           "LUMO_eV",
    "gap_ev_neutral":            "HOMO_LUMO_gap_eV",
    "dipole_moment_neutral":     "dipole_debye",
    "dipole_x_neutral":          "dipole_x",
    "dipole_y_neutral":          "dipole_y",
    "dipole_z_neutral":          "dipole_z",
    "num_atoms_neutral":         "n_atoms",
    "heat_of_formation_neutral": "HOF_kcalmol",
    "total_energy_ev_neutral":   "total_energy_eV",
    "ionization_potential_neutral":  "ionization_potential_eV",
    "cosmo_area_neutral":        "cosmo_area",
    "cosmo_volume_neutral":      "cosmo_volume",
    "molecular_weight_neutral":  "molecular_weight",
    "spin_state_neutral":        "spin_state",
    "computation_time_neutral":  "wall_time_s",
}

KMEANS_PROT_MAP = {
    "homo_ev_protonated":        "HOMO_eV",
    "lumo_ev_protonated":        "LUMO_eV",
    "gap_ev_protonated":         "HOMO_LUMO_gap_eV",
    "dipole_moment_protonated":  "dipole_debye",
    "dipole_x_protonated":       "dipole_x",
    "dipole_y_protonated":       "dipole_y",
    "dipole_z_protonated":       "dipole_z",
    "num_atoms_protonated":      "n_atoms",
    "heat_of_formation_protonated":  "HOF_kcalmol",
    "total_energy_ev_protonated":    "total_energy_eV",
    "ionization_potential_protonated":   "ionization_potential_eV",
    "cosmo_area_protonated":     "cosmo_area",
    "cosmo_volume_protonated":   "cosmo_volume",
    "molecular_weight_protonated":   "molecular_weight",
    "spin_state_protonated":     "spin_state",
    "computation_time_protonated":   "wall_time_s",
    "alpha_electrons_protonated":    "alpha_electrons",
    "beta_electrons_protonated":     "beta_electrons",
    "unpaired_electrons_protonated": "unpaired_electrons",
    "multiplicity_protonated":       "multiplicity",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val):
    """Convert numpy scalars / NaN to plain Python types."""
    if val is None:
        return None
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return None
    except Exception:
        pass
    if hasattr(val, "item"):   # numpy scalar
        return val.item()
    return val


def _map_row(row: pd.Series, col_map: dict) -> dict:
    return {unified: _safe(row.get(raw)) for raw, unified in col_map.items()}


def compute_pa_kcal(hof_neutral: float, hof_protonated: float) -> float:
    """PA (kcal/mol) = HOF_neutral - HOF_protonated + H(H+)_PM7"""
    return hof_neutral - hof_protonated + H_HPLUS_PM7_KCAL


def build_site_record(
    site_idx,
    site_element: str,
    prot_smiles: str,
    pa_kcal: float,
    prot_data: dict,
    neu_data: dict,
) -> dict:
    """Assemble one site record in the unified schema."""
    pa_kjmol = round(pa_kcal * KCAL_TO_KJMOL, 4) if pa_kcal is not None else None

    gap_n = neu_data.get("HOMO_LUMO_gap_eV")
    gap_p = prot_data.get("HOMO_LUMO_gap_eV")
    dip_n = neu_data.get("dipole_debye")
    dip_p = prot_data.get("dipole_debye")

    return {
        "site_idx":               _safe(site_idx),
        "site_name":              site_element,
        "protonated_smiles":      prot_smiles,
        "status":                 "OK" if pa_kcal is not None else "FAILED",
        "pa_kcalmol":             round(pa_kcal, 4) if pa_kcal else None,
        "pa_kjmol":               pa_kjmol,
        # protonated state
        "HOMO_eV":                prot_data.get("HOMO_eV"),
        "LUMO_eV":                prot_data.get("LUMO_eV"),
        "HOMO_LUMO_gap_eV":       prot_data.get("HOMO_LUMO_gap_eV"),
        "dipole_debye":           prot_data.get("dipole_debye"),
        "dipole_x":               prot_data.get("dipole_x"),
        "dipole_y":               prot_data.get("dipole_y"),
        "dipole_z":               prot_data.get("dipole_z"),
        "HOF_kcalmol":            prot_data.get("HOF_kcalmol"),
        "total_energy_eV":        prot_data.get("total_energy_eV"),
        "ionization_potential_eV":prot_data.get("ionization_potential_eV"),
        "cosmo_area":             prot_data.get("cosmo_area"),
        "cosmo_volume":           prot_data.get("cosmo_volume"),
        "molecular_weight":       prot_data.get("molecular_weight"),
        "spin_state":             prot_data.get("spin_state"),
        "n_atoms":                prot_data.get("n_atoms"),
        "charge":                 prot_data.get("charge", 1),
        "multiplicity":           prot_data.get("multiplicity"),
        "unpaired_electrons":     prot_data.get("unpaired_electrons"),
        # delta features (mirrors DFT schema)
        "delta_HOMO_LUMO_gap_eV": round(gap_p - gap_n, 6) if (gap_p and gap_n) else None,
        "delta_dipole_debye":     round(dip_p - dip_n, 6) if (dip_p and dip_n) else None,
    }


def build_molecule_record(
    record_id: str,
    mol_smiles: str,
    source_tag: str,
    source_file: str,
    neutral_data: dict,
    all_sites: list[dict],
    exp_pa_kjmol=None,
) -> dict:
    """Assemble the full molecule record in the unified schema."""
    best_site = max(all_sites, key=lambda s: s["pa_kjmol"] or 0) if all_sites else None
    best_pa   = best_site["pa_kjmol"] if best_site else None
    best_pa_kcal = best_site["pa_kcalmol"] if best_site else None

    delta_pm7_exp = round(abs(exp_pa_kjmol - best_pa), 4) if (exp_pa_kjmol and best_pa) else None

    return {
        "record_id": record_id,
        "mol_id":    record_id,
        "metadata": {
            "source":        source_tag,
            "source_file":   source_file,
            "method":        "PM7/MOPAC",
            "n_sites":       len(all_sites),
            "best_site_idx": best_site["site_idx"] if best_site else None,
            "warnings":      [],
            "parsed_at":     datetime.now(timezone.utc).isoformat(),
        },
        "labels": {
            "exp_pa_kjmol":   exp_pa_kjmol,
            "exp_pa_kcalmol": round(exp_pa_kjmol * KJMOL_TO_KCAL, 4) if exp_pa_kjmol else None,
            "pm7_pa_kjmol":   best_pa,
            "pm7_pa_kcalmol": best_pa_kcal,
            "dft_pa_kjmol":   None,   # filled by cross-reference with DFT dataset
            "dft_pa_kcalmol": None,
            "delta_pm7_exp":  delta_pm7_exp,
            "dft_correction": None,
        },
        "neutral": {
            "smiles":                 mol_smiles,
            "charge":                 0,
            "HOMO_eV":                neutral_data.get("HOMO_eV"),
            "LUMO_eV":                neutral_data.get("LUMO_eV"),
            "HOMO_LUMO_gap_eV":       neutral_data.get("HOMO_LUMO_gap_eV"),
            "dipole_debye":           neutral_data.get("dipole_debye"),
            "dipole_x":               neutral_data.get("dipole_x"),
            "dipole_y":               neutral_data.get("dipole_y"),
            "dipole_z":               neutral_data.get("dipole_z"),
            "HOF_kcalmol":            neutral_data.get("HOF_kcalmol"),
            "total_energy_eV":        neutral_data.get("total_energy_eV"),
            "ionization_potential_eV":neutral_data.get("ionization_potential_eV"),
            "cosmo_area":             neutral_data.get("cosmo_area"),
            "cosmo_volume":           neutral_data.get("cosmo_volume"),
            "molecular_weight":       neutral_data.get("molecular_weight"),
            "spin_state":             neutral_data.get("spin_state"),
            "n_atoms":                neutral_data.get("n_atoms"),
            "wall_time_s":            neutral_data.get("wall_time_s"),
        },
        "all_sites": all_sites,
    }


# ---------------------------------------------------------------------------
# CSV columns  (mirrors DFT features.csv + PM7-specific extras)
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    # identifiers
    "record_id", "source", "neutral_smiles", "protonated_smiles",
    "site_idx", "site_name",
    # labels
    "exp_pa_kjmol", "exp_pa_kcalmol",
    "pm7_pa_kjmol", "pm7_pa_kcalmol",
    "dft_pa_kjmol", "dft_pa_kcalmol",
    "delta_pm7_exp", "dft_correction",
    # neutral features (shared with DFT schema)
    "neutral_HOMO_eV", "neutral_LUMO_eV", "neutral_HOMO_LUMO_gap_eV",
    "neutral_dipole_debye",
    # neutral features (PM7-specific)
    "neutral_HOF_kcalmol", "neutral_ionization_potential_eV",
    "neutral_cosmo_area", "neutral_cosmo_volume",
    "neutral_molecular_weight", "neutral_spin_state", "neutral_n_atoms",
    # protonated features (shared with DFT schema)
    "prot_HOMO_eV", "prot_LUMO_eV", "prot_HOMO_LUMO_gap_eV",
    "prot_dipole_debye",
    # protonated features (PM7-specific)
    "prot_HOF_kcalmol", "prot_ionization_potential_eV",
    "prot_cosmo_area", "prot_cosmo_volume",
    "prot_spin_state", "prot_n_atoms", "prot_multiplicity",
    # delta features (directly comparable to DFT delta features)
    "delta_HOMO_LUMO_gap_eV",
    "delta_dipole_debye",
]


def record_to_csv_rows(record: dict) -> list[dict]:
    lbl = record["labels"]
    neu = record["neutral"]
    base = {
        "record_id":                   record["record_id"],
        "source":                      record["metadata"]["source"],
        "neutral_smiles":              neu.get("smiles"),
        "exp_pa_kjmol":                lbl.get("exp_pa_kjmol"),
        "exp_pa_kcalmol":              lbl.get("exp_pa_kcalmol"),
        "pm7_pa_kjmol":                lbl.get("pm7_pa_kjmol"),
        "pm7_pa_kcalmol":              lbl.get("pm7_pa_kcalmol"),
        "dft_pa_kjmol":                lbl.get("dft_pa_kjmol"),
        "dft_pa_kcalmol":              lbl.get("dft_pa_kcalmol"),
        "delta_pm7_exp":               lbl.get("delta_pm7_exp"),
        "dft_correction":              lbl.get("dft_correction"),
        "neutral_HOMO_eV":             neu.get("HOMO_eV"),
        "neutral_LUMO_eV":             neu.get("LUMO_eV"),
        "neutral_HOMO_LUMO_gap_eV":    neu.get("HOMO_LUMO_gap_eV"),
        "neutral_dipole_debye":        neu.get("dipole_debye"),
        "neutral_HOF_kcalmol":         neu.get("HOF_kcalmol"),
        "neutral_ionization_potential_eV": neu.get("ionization_potential_eV"),
        "neutral_cosmo_area":          neu.get("cosmo_area"),
        "neutral_cosmo_volume":        neu.get("cosmo_volume"),
        "neutral_molecular_weight":    neu.get("molecular_weight"),
        "neutral_spin_state":          neu.get("spin_state"),
        "neutral_n_atoms":             neu.get("n_atoms"),
    }
    rows = []
    for site in record["all_sites"]:
        row = {**base}
        row["protonated_smiles"]       = site.get("protonated_smiles")
        row["site_idx"]                = site.get("site_idx")
        row["site_name"]               = site.get("site_name")
        row["prot_HOMO_eV"]            = site.get("HOMO_eV")
        row["prot_LUMO_eV"]            = site.get("LUMO_eV")
        row["prot_HOMO_LUMO_gap_eV"]   = site.get("HOMO_LUMO_gap_eV")
        row["prot_dipole_debye"]       = site.get("dipole_debye")
        row["prot_HOF_kcalmol"]        = site.get("HOF_kcalmol")
        row["prot_ionization_potential_eV"] = site.get("ionization_potential_eV")
        row["prot_cosmo_area"]         = site.get("cosmo_area")
        row["prot_cosmo_volume"]       = site.get("cosmo_volume")
        row["prot_spin_state"]         = site.get("spin_state")
        row["prot_n_atoms"]            = site.get("n_atoms")
        row["prot_multiplicity"]       = site.get("multiplicity")
        row["delta_HOMO_LUMO_gap_eV"]  = site.get("delta_HOMO_LUMO_gap_eV")
        row["delta_dipole_debye"]      = site.get("delta_dipole_debye")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_nist_file(csv_path: Path) -> dict:
    """Parse pm7_nist1185_pa_per_site.csv -> records keyed by smiles."""
    df = pd.read_csv(csv_path)
    records = {}

    for smiles, group in df.groupby("neutral_smiles"):
        first = group.iloc[0]
        neutral_data = _map_row(first, NIST_NEUTRAL_MAP)

        all_sites = []
        for _, row in group.iterrows():
            prot_data = _map_row(row, NIST_PROT_MAP)
            hof_n = _safe(row["hof_neutral"])
            hof_p = _safe(row["hof_protonated"])
            pa_kcal = compute_pa_kcal(hof_n, hof_p) if (hof_n is not None and hof_p is not None) else None

            # cross-check against stored value (should be ~0 diff)
            stored_pa = _safe(row.get("proton_affinity_kcal_mol"))
            if stored_pa and pa_kcal and abs(pa_kcal - stored_pa) > 0.01:
                log.warning(f"PA mismatch for {smiles}: computed={pa_kcal:.3f} stored={stored_pa:.3f}")

            site = build_site_record(
                site_idx     = _safe(row.get("site_index")),
                site_element = str(row.get("site_element", "")),
                prot_smiles  = str(row.get("protonated_smiles", "")),
                pa_kcal      = pa_kcal,
                prot_data    = prot_data,
                neu_data     = neutral_data,
            )
            all_sites.append(site)

        # mol_id from SMILES hash index in group — use row order as proxy
        record_id = f"nist_{abs(hash(smiles)) % 10**8:08d}"
        record = build_molecule_record(
            record_id   = record_id,
            mol_smiles  = smiles,
            source_tag  = "pm7_nist",
            source_file = str(csv_path.name),
            neutral_data= neutral_data,
            all_sites   = all_sites,
            exp_pa_kjmol= None,   # merged from NIST lookup separately
        )
        records[record_id] = record

    return records


def parse_kmeans_file(csv_path: Path) -> dict:
    """Parse pm7_kmeans251_pa_per_site.csv -> records keyed by smiles."""
    df = pd.read_csv(csv_path)
    records = {}

    for smiles, group in df.groupby("neutral_smiles"):
        first = group.iloc[0]
        neutral_data = _map_row(first, KMEANS_NEUTRAL_MAP)

        all_sites = []
        for _, row in group.iterrows():
            prot_data = _map_row(row, KMEANS_PROT_MAP)
            hof_n = _safe(row.get("heat_of_formation_neutral"))
            hof_p = _safe(row.get("heat_of_formation_protonated"))
            pa_kcal = compute_pa_kcal(hof_n, hof_p) if (hof_n is not None and hof_p is not None) else None

            site = build_site_record(
                site_idx     = _safe(row.get("protonation_site_index_protonated")),
                site_element = str(row.get("protonation_site_element_protonated", "")),
                prot_smiles  = str(row.get("protonated_smiles", "")),
                pa_kcal      = pa_kcal,
                prot_data    = prot_data,
                neu_data     = neutral_data,
            )
            all_sites.append(site)

        record_id = f"kmeans_{abs(hash(smiles)) % 10**8:08d}"
        record = build_molecule_record(
            record_id   = record_id,
            mol_smiles  = smiles,
            source_tag  = "pm7_kmeans",
            source_file = str(csv_path.name),
            neutral_data= neutral_data,
            all_sites   = all_sites,
            exp_pa_kjmol= None,
        )
        records[record_id] = record

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nist_file   = PM7_DIR / "pm7_nist1185_pa_per_site.csv"
    kmeans_file = PM7_DIR / "pm7_kmeans251_pa_per_site.csv"

    dataset  = {}
    csv_rows = []
    report   = {"missing_files": [], "warnings": [], "summary": {}}

    # ---- NIST 1185 set ----
    if nist_file.exists():
        log.info(f"Parsing {nist_file.name} ...")
        nist_records = parse_nist_file(nist_file)
        dataset.update(nist_records)
        for rec in nist_records.values():
            csv_rows.extend(record_to_csv_rows(rec))
        log.info(f"  {len(nist_records)} molecules, {sum(len(r['all_sites']) for r in nist_records.values())} site rows")
    else:
        log.warning(f"Not found: {nist_file}  — expected at {nist_file.relative_to(SCRIPT_DIR.parent)}")
        report["missing_files"].append(str(nist_file.name))

    # ---- k-means 251 set ----
    if kmeans_file.exists():
        log.info(f"Parsing {kmeans_file.name} ...")
        km_records = parse_kmeans_file(kmeans_file)
        dataset.update(km_records)
        for rec in km_records.values():
            csv_rows.extend(record_to_csv_rows(rec))
        log.info(f"  {len(km_records)} molecules, {sum(len(r['all_sites']) for r in km_records.values())} site rows")
    else:
        log.warning(f"Not found: {kmeans_file}  — expected at {kmeans_file.relative_to(SCRIPT_DIR.parent)}")
        report["missing_files"].append(str(kmeans_file.name))

    # ---- Write outputs ----
    json_path = OUT_DIR / "pm7_dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=2)
    log.info(f"Wrote {json_path.relative_to(SCRIPT_DIR.parent)}  ({len(dataset)} records)")

    csv_path = OUT_DIR / "pm7_features.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info(f"Wrote {csv_path.relative_to(SCRIPT_DIR.parent)}  ({len(csv_rows)} rows)")

    nist_count   = len([r for r in dataset.values() if r["metadata"]["source"] == "pm7_nist"])
    kmeans_count = len([r for r in dataset.values() if r["metadata"]["source"] == "pm7_kmeans"])
    report["summary"] = {
        "total_records":   len(dataset),
        "nist_records":    nist_count,
        "kmeans_records":  kmeans_count,
        "total_csv_rows":  len(csv_rows),
        "run_at":          datetime.now(timezone.utc).isoformat(),
    }
    report_path = OUT_DIR / "pm7_parse_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Wrote {report_path.relative_to(SCRIPT_DIR.parent)}")

    print("\n" + "="*55)
    print(f"  Records:   {len(dataset)}  ({nist_count} NIST + {kmeans_count} k-means)")
    print(f"  CSV rows:  {len(csv_rows)}  (one per site)")
    print(f"  Outputs:   {OUT_DIR.relative_to(SCRIPT_DIR.parent)}/")
    print("="*55)
    print()
    print("Next step: place renamed PM7 CSVs in data/pm7/ then run.")
    print("  Rename: FINAL_PM7_ALL_proton_affinities.csv  -> pm7_nist1185_pa_per_site.csv")
    print("  Rename: FINAL_PM7_DFT_master.csv             -> pm7_kmeans251_pa_per_site.csv")


if __name__ == "__main__":
    main()
