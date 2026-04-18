"""
03_run_pm7.py
=============
Run PM7 calculations on screening candidates using the existing
mopac_calculator.py + run_pm7_parallel.py infrastructure.

Reads from:
    data/screening/iter{N}/candidates.parquet

Writes to:
    data/screening/iter{N}/pm7_results.parquet   -- flat per-site records
    data/screening/iter{N}/pm7_raw/              -- raw JSON checkpoints
    screening/logs/iter{N}_pm7.log

Usage:
    python screening/scripts/03_run_pm7.py --iter 1
    python screening/scripts/03_run_pm7.py --iter 1 --workers 8
    python screening/scripts/03_run_pm7.py --iter 1 --dry-run
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

# Add screening/scripts to path for mopac_calculator import
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
SCREENING  = SCRIPT_DIR.parent.parent
PROJECT    = SCREENING.parent
DATA_DIR   = PROJECT / "data" / "screening"

from mopac_calculator import MOPACCalculator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROTON_ENTHALPY_KCAL = 365.7

# ---------------------------------------------------------------------------
# Site enumeration and protonation
# ---------------------------------------------------------------------------

def get_protonation_sites(smiles: str) -> list[dict]:
    """Enumerate all N and O protonation sites."""
    import copy
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    sites = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ("N", "O"):
            continue
        if atom.GetFormalCharge() != 0:
            continue
        # Generate protonated SMILES
        try:
            mol_copy = copy.deepcopy(mol)
            a = mol_copy.GetAtomWithIdx(atom.GetIdx())
            a.SetFormalCharge(1)
            Chem.SanitizeMol(mol_copy)
            prot_smiles = Chem.MolToSmiles(Chem.AddHs(mol_copy), canonical=True)
            # Deduplicate
            if not any(s["protonated_smiles"] == prot_smiles for s in sites):
                sites.append({
                    "site_idx":          atom.GetIdx(),
                    "element":           atom.GetSymbol(),
                    "protonated_smiles": prot_smiles,
                })
        except Exception:
            continue

    n = len(sites)
    for i, s in enumerate(sites):
        s["site_index"]            = i
        s["site_normalized_index"] = i / max(n - 1, 1)
        s["site_n_sites"]          = n
    return sites


# ---------------------------------------------------------------------------
# Per-molecule PM7 worker
# ---------------------------------------------------------------------------

def process_molecule(smiles: str, mol_id: int, calc: MOPACCalculator) -> list[dict]:
    """
    Run PM7 for neutral + all protonation sites.
    Returns list of per-site records.
    """
    records = []

    # Neutral calculation
    neutral = calc.calculate_properties(smiles, charge=0, cleanup=True)
    if not neutral.get("success") or neutral.get("heat_of_formation") is None:
        return records

    hof_neutral = neutral["heat_of_formation"]

    # Per-site
    sites = get_protonation_sites(smiles)
    if not sites:
        return records

    for site in sites:
        prot = calc.calculate_properties(
            site["protonated_smiles"], charge=1, cleanup=True)
        if not prot.get("success") or prot.get("heat_of_formation") is None:
            continue

        hof_prot = prot["heat_of_formation"]
        pa_pm7   = hof_neutral + PROTON_ENTHALPY_KCAL - hof_prot

        record = {
            "mol_id":                  mol_id,
            "smiles":                  smiles,
            "site_idx":                site["site_idx"],
            "site_element":            site["element"],
            "site_index":              site["site_index"],
            "site_normalized_index":   site["site_normalized_index"],
            "site_n_sites":            site["site_n_sites"],
            "protonated_smiles":       site["protonated_smiles"],
            "pa_pm7_kcalmol":          pa_pm7,
            # Neutral properties
            "neutral_HOF_kcalmol":         neutral.get("heat_of_formation"),
            "neutral_HOMO_eV":             neutral.get("homo_ev"),
            "neutral_LUMO_eV":             neutral.get("lumo_ev"),
            "neutral_HOMO_LUMO_gap_eV":    neutral.get("gap_ev"),
            "neutral_dipole_debye":        neutral.get("dipole_moment"),
            "neutral_dipole_x":            neutral.get("dipole_x"),
            "neutral_dipole_y":            neutral.get("dipole_y"),
            "neutral_dipole_z":            neutral.get("dipole_z"),
            "neutral_ionization_potential_eV": neutral.get("ionization_potential"),
            "neutral_electronic_energy_eV":    neutral.get("electronic_energy"),
            "neutral_core_core_repulsion_eV":  neutral.get("core_core_repulsion"),
            "neutral_cosmo_area":              neutral.get("cosmo_area"),
            "neutral_cosmo_volume":            neutral.get("cosmo_volume"),
            # Protonated properties
            "protonated_HOF_kcalmol":          prot.get("heat_of_formation"),
            "protonated_HOMO_eV":              prot.get("homo_ev"),
            "protonated_LUMO_eV":              prot.get("lumo_ev"),
            "protonated_HOMO_LUMO_gap_eV":     prot.get("gap_ev"),
            "protonated_dipole_debye":         prot.get("dipole_moment"),
            "protonated_dipole_x":             prot.get("dipole_x"),
            "protonated_dipole_y":             prot.get("dipole_y"),
            "protonated_dipole_z":             prot.get("dipole_z"),
            "protonated_ionization_potential_eV": prot.get("ionization_potential"),
            "protonated_electronic_energy_eV":    prot.get("electronic_energy"),
            "protonated_core_core_repulsion_eV":  prot.get("core_core_repulsion"),
            "protonated_cosmo_area":              prot.get("cosmo_area"),
            "protonated_cosmo_volume":            prot.get("cosmo_volume"),
        }
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Module-level worker functions (must be at module level for pickling)
# ---------------------------------------------------------------------------

def _worker_init(script_dir: str) -> None:
    """Initialize each worker process with the correct sys.path."""
    import sys
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)


def _worker_fn(args):
    """Top-level worker — picklable by ProcessPoolExecutor."""
    import sys
    from mopac_calculator import MOPACCalculator
    smi, mol_id = args
    with MOPACCalculator() as calc:
        return process_molecule(smi, mol_id, calc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(iteration: int, workers: int = 8, dry_run: bool = False) -> None:
    iter_dir  = DATA_DIR / f"iter{iteration}"
    cand_path = iter_dir / "candidates.parquet"
    out_path  = iter_dir / "pm7_results.parquet"
    fail_path = iter_dir / "pm7_failed.csv"

    # File logging
    log_path = SCREENING / "logs" / f"iter{iteration}_pm7.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)

    if not cand_path.exists():
        log.error(f"Candidates not found: {cand_path}")
        sys.exit(1)

    cand_df = pd.read_parquet(cand_path)
    log.info(f"Loaded {len(cand_df):,} candidates for iteration {iteration}")

    if dry_run:
        log.warning("DRY RUN — first 5 molecules only")
        cand_df = cand_df.head(5)

    smiles_list = cand_df["smiles"].tolist()
    log.info(f"Starting PM7 on {len(smiles_list):,} molecules ...")

    all_records = []
    failed      = []
    t0 = time.time()

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(str(SCRIPT_DIR),)) as executor:
        futures = {
            executor.submit(_worker_fn, (smi, i)): (i, smi)
            for i, smi in enumerate(smiles_list)
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="PM7 calculations"):
            mol_id, smi = futures[future]
            try:
                recs = future.result()
                if recs:
                    all_records.extend(recs)
                else:
                    failed.append({"mol_id": mol_id, "smiles": smi,
                                   "reason": "no_sites_or_convergence"})
            except Exception as e:
                failed.append({"mol_id": mol_id, "smiles": smi,
                               "reason": str(e)})

    elapsed = (time.time() - t0) / 60
    log.info(f"\nFinished in {elapsed:.1f} min")
    log.info(f"  Site records:    {len(all_records)}")
    log.info(f"  Molecules OK:    {len(set(r['mol_id'] for r in all_records))}")
    log.info(f"  Failed:          {len(failed)}")

    if all_records:
        df = pd.DataFrame(all_records)
        df.to_parquet(out_path, index=False)
        log.info(f"Saved → {out_path}")
        log.info(f"  PA_PM7 range: "
                 f"{df['pa_pm7_kcalmol'].min():.1f} – "
                 f"{df['pa_pm7_kcalmol'].max():.1f} kcal/mol")
        log.info(f"  PA_PM7 mean:  {df['pa_pm7_kcalmol'].mean():.1f} kcal/mol")

        # Quick sanity — known PAs
        log.info("\n  Sanity check (known PA values):")
        log.info("    imidazole:     223 kcal/mol (exp)")
        log.info("    benzimidazole: ~230 kcal/mol (exp)")
        for smi, name in [("c1cn[nH]c1", "imidazole"),
                          ("c1ccc2[nH]cnc2c1", "benzimidazole")]:
            sub = df[df["smiles"] == smi]
            if len(sub) > 0:
                best = sub["pa_pm7_kcalmol"].max()
                log.info(f"    {name}: {best:.1f} kcal/mol (PM7)")
    else:
        log.error("No records — check MOPAC installation")

    if failed:
        pd.DataFrame(failed).to_csv(fail_path, index=False)
        log.info(f"Failed → {fail_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PM7 screening calculations")
    parser.add_argument("--iter",    type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(iteration=args.iter, workers=args.workers, dry_run=args.dry_run)