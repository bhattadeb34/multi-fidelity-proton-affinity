#!/usr/bin/env python3
"""Merge Roar PM7 checkpoint JSON files into screening parquet artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCREENING = SCRIPT_DIR.parent.parent
PROJECT = SCREENING.parent
DATA_ROOT = next((p for p in (PROJECT / "data", PROJECT.parent / "data") if p.exists()), PROJECT / "data")
DATA_DIR = DATA_ROOT / "screening"


def load_chunk_json(chunk_dir: Path, name: str) -> dict | list:
    return json.loads((chunk_dir / name).read_text())


def property_fields(prefix: str, values: dict) -> dict:
    return {
        f"{prefix}_HOF_kcalmol": values.get("heat_of_formation"),
        f"{prefix}_HOMO_eV": values.get("homo_ev"),
        f"{prefix}_LUMO_eV": values.get("lumo_ev"),
        f"{prefix}_HOMO_LUMO_gap_eV": values.get("gap_ev"),
        f"{prefix}_dipole_debye": values.get("dipole_moment"),
        f"{prefix}_dipole_x": values.get("dipole_x"),
        f"{prefix}_dipole_y": values.get("dipole_y"),
        f"{prefix}_dipole_z": values.get("dipole_z"),
        f"{prefix}_ionization_potential_eV": values.get("ionization_potential"),
        f"{prefix}_electronic_energy_eV": values.get("total_energy_ev"),
        f"{prefix}_core_core_repulsion_eV": np.nan,
        f"{prefix}_cosmo_area": values.get("cosmo_area"),
        f"{prefix}_cosmo_volume": values.get("cosmo_volume"),
    }


def convert_chunk(chunk_dir: Path, mol_id_by_smiles: dict[str, int]) -> tuple[list, list]:
    neutral = load_chunk_json(chunk_dir, "properties_neutral.json")
    protonated = load_chunk_json(chunk_dir, "properties_protonated.json")
    protonation_map = load_chunk_json(chunk_dir, "protonation_map.json")
    failed_raw = load_chunk_json(chunk_dir, "failed_molecules.json")

    records = []
    successful_smiles = set()
    for smiles, neutral_values in neutral.items():
        if smiles not in mol_id_by_smiles or not neutral_values.get("success"):
            continue
        sites = protonation_map.get(smiles, [])
        n_sites = len(sites)
        for site_index, site in enumerate(sites):
            protonated_smiles = site["smiles"]
            protonated_values = protonated.get(protonated_smiles)
            if not protonated_values or not protonated_values.get("success"):
                continue
            hof_neutral = neutral_values.get("heat_of_formation")
            hof_protonated = protonated_values.get("heat_of_formation")
            if hof_neutral is None or hof_protonated is None:
                continue
            records.append({
                "mol_id": mol_id_by_smiles[smiles],
                "smiles": smiles,
                "site_idx": site["site_index"],
                "site_element": site["site_element"],
                "site_index": site_index,
                "site_normalized_index": site_index / max(n_sites - 1, 1),
                "site_n_sites": n_sites,
                "protonated_smiles": protonated_smiles,
                "pa_pm7_kcalmol": hof_neutral + 365.7 - hof_protonated,
                **property_fields("neutral", neutral_values),
                **property_fields("protonated", protonated_values),
            })
            successful_smiles.add(smiles)

    failed_by_parent = {}
    for failure in failed_raw:
        parent = failure.get("parent_neutral") or failure.get("smiles")
        if parent in mol_id_by_smiles:
            failed_by_parent[parent] = failure.get("error", "HPC PM7 failure")
    for smiles in set(protonation_map) - successful_smiles:
        if smiles in mol_id_by_smiles:
            failed_by_parent.setdefault(smiles, "no successful protonation sites")

    failed = [
        {
            "mol_id": mol_id_by_smiles[smiles],
            "smiles": smiles,
            "reason": reason,
        }
        for smiles, reason in failed_by_parent.items()
    ]
    return records, failed


def main(iteration: int, hpc_dir: Path) -> None:
    iter_dir = DATA_DIR / f"iter{iteration}"
    candidate_path = iter_dir / "candidates.parquet"
    result_path = iter_dir / "pm7_results.parquet"
    failure_path = iter_dir / "pm7_failed.csv"

    candidates = pd.read_parquet(candidate_path)
    mol_id_by_smiles = {
        smiles: mol_id for mol_id, smiles in enumerate(candidates["smiles"])
    }
    existing = pd.read_parquet(result_path)
    existing = existing[existing["smiles"].isin(mol_id_by_smiles)].copy()
    existing["mol_id"] = existing["smiles"].map(mol_id_by_smiles)
    existing_failures = (
        pd.read_csv(failure_path)
        if failure_path.exists()
        else pd.DataFrame(columns=["mol_id", "smiles", "reason"])
    )
    existing_failures = existing_failures[
        existing_failures["smiles"].isin(mol_id_by_smiles)
    ].copy()

    new_records = []
    new_failures = []
    chunk_dirs = sorted(hpc_dir.glob("results_chunk_*"))
    if not chunk_dirs:
        raise FileNotFoundError(f"No results_chunk_* directories under {hpc_dir}")
    for chunk_dir in chunk_dirs:
        records, failures = convert_chunk(chunk_dir, mol_id_by_smiles)
        new_records.extend(records)
        new_failures.extend(failures)

    new_df = pd.DataFrame(new_records)
    if new_df.empty:
        raise RuntimeError("HPC checkpoint conversion produced no successful site records")
    merged = (pd.concat([existing, new_df], ignore_index=True)
              .drop_duplicates(["smiles", "site_idx", "protonated_smiles"], keep="last")
              .sort_values(["mol_id", "site_idx"])
              .reset_index(drop=True))
    successful = set(merged["smiles"])
    failures = pd.concat(
        [existing_failures, pd.DataFrame(new_failures)], ignore_index=True
    )
    failures = failures[~failures["smiles"].isin(successful)]
    failures["mol_id"] = failures["smiles"].map(mol_id_by_smiles)
    failures = (failures.drop_duplicates("smiles", keep="last")
                .sort_values("mol_id")
                .reset_index(drop=True))
    classified = successful | set(failures["smiles"])
    missing = set(candidates["smiles"]) - classified
    if missing:
        raise AssertionError(f"Unclassified candidates after merge: {len(missing)}")

    # Do not replace either artifact until the candidate partition is complete.
    merged.to_parquet(result_path, index=False)
    failures.to_csv(failure_path, index=False)

    print(f"HPC chunks: {len(chunk_dirs)}")
    print(f"New successful molecules: {new_df['smiles'].nunique()}")
    print(f"New site records: {len(new_df)}")
    print(f"Merged successful molecules: {merged['smiles'].nunique()}")
    print(f"Merged site records: {len(merged)}")
    print(f"Failed molecules: {len(failures)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--hpc-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.iter, args.hpc_dir)
