#!/usr/bin/env python3
"""
DFT proton-affinity calculator (B3LYP/def2-TZVP by default) - preserves
mol_idx from an input CSV. Per-molecule JSON results are written to the
directory named by the RESULTS_DIR environment variable (default: ./dft_results).

Input CSV columns:
    mol_idx,smiles

Usage (direct):
    export RESULTS_DIR=dft_results
    python run_dft_pa.py --csv my_molecules.csv --save-files --files-dir dft_files

Usage (SLURM):
    sbatch submit_dft.sh      # see submit_dft.sh for placeholders to fill in
"""
import argparse
import json
import time
import os
import sys
import traceback
import pandas as pd
import numpy as np

from pa_calculator import (
    calculate_pa, enumerate_protonation_sites,
    HARTREE_TO_KJMOL, R_KJMOL
)

RESULTS_DIR = os.environ.get("RESULTS_DIR", "dft_results") # per-molecule JSON files go here


def result_path(mol_idx_str):
    """Path for per-molecule result JSON - preserves mol_idx string."""
    return os.path.join(RESULTS_DIR, f"{mol_idx_str}.json")


def is_done(mol_idx_str):
    """Check if molecule already has a completed result (checkpoint)."""
    p = result_path(mol_idx_str)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            data = json.load(f)
        return data.get("status") in ("OK", "NO_SITES")
    except Exception:
        return False


def make_serializable(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_result(mol_idx_str, result):
    """Save per-molecule result to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = make_serializable(result)
    with open(result_path(mol_idx_str), "w") as f:
        json.dump(result, f, indent=2)


def process_molecule(row, args):
    """Process a single molecule: compute PA for all protonation sites."""
    mol_idx_str = str(row["mol_idx"])
    smiles = row["smiles"]

    result = {
        "mol_idx": mol_idx_str,
        "smiles": smiles,
        "level": f"{args.xc}/{args.basis}",
        "freq": args.do_freq,
    }

    t0 = time.time()

    try:
        # Set up per-molecule output directory for standard DFT files
        mol_outdir = None
        if getattr(args, "save_files", False):
            mol_outdir = os.path.join(args.files_dir, mol_idx_str)

        best_pa, res_n, res_p, all_sites = calculate_pa(
            smiles,
            basis=args.basis,
            xc=args.xc,
            do_freq=args.do_freq,
            output_dir=mol_outdir,
        )

        if best_pa is None:
            result["status"] = "NO_SITES"
            result["dft_pa"] = None
            result["neutral"] = make_serializable(res_n) if res_n else None
        else:
            result["status"] = "OK"
            result["dft_pa"] = round(best_pa, 2)

            # Find best site info
            best_site_label = "unknown"
            for sr in all_sites:
                if sr["status"] == "OK" and sr["pa_kjmol"] == best_pa:
                    s = sr["site"]
                    best_site_label = f"{s['atom_symbol']}({s['neighbor_info']})"
                    break
            result["best_site"] = best_site_label
            result["n_sites"] = len(all_sites)

            # Save neutral properties
            result["neutral"] = make_serializable(res_n)

            # Save best protonated properties
            result["protonated_best"] = make_serializable(res_p)

            # Save all site PAs (summary)
            site_summary = []
            for sr in all_sites:
                s = sr["site"]
                entry = {
                    "atom": s["atom_symbol"],
                    "neighbors": s["neighbor_info"],
                    "protonated_smiles": s["protonated_smiles"],
                    "status": sr["status"],
                }
                if sr["status"] == "OK":
                    entry["pa_kjmol"] = round(sr["pa_kjmol"], 2)
                    sr_res = sr.get("result", {})
                    for key in ["E_elec", "H_total", "HOMO_eV", "LUMO_eV",
                                "dipole_debye", "ZPE_kjmol"]:
                        if key in sr_res:
                            entry[key] = sr_res[key]
                site_summary.append(entry)
            result["all_sites"] = site_summary

    except Exception as e:
        result["status"] = f"FAILED: {e}"
        traceback.print_exc()

    result["wall_time_s"] = round(time.time() - t0, 1)
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark PA Calculator")
    parser.add_argument("--csv", required=True,
                        help="Path to CSV with mol_idx,smiles columns")
    parser.add_argument("--basis", default="def2-TZVP")
    parser.add_argument("--xc", default="B3LYP")
    parser.add_argument("--no-freq", dest="do_freq", action="store_false")
    parser.add_argument("--save-files", action="store_true",
                        help="Save standard DFT output files")
    parser.add_argument("--files-dir", default="dft_files",
                        help="Directory for DFT output files")
    args = parser.parse_args()

    print("=" * 70)
    print("  BENCHMARK MOLECULES: PROTON AFFINITY CALCULATIONS")
    print(f"  Level: {args.xc}/{args.basis}")
    print(f"  Freq:  {'Yes' if args.do_freq else 'No'}")
    print("=" * 70)

    df = pd.read_csv(args.csv)
    if "mol_idx" not in df.columns:
        print("ERROR: CSV must have 'mol_idx' column")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_total = len(df)
    n_done = 0
    n_skipped = 0
    n_ok = 0
    n_failed = 0
    t_start = time.time()

    for _, row in df.iterrows():
        mol_idx_str = str(row["mol_idx"])

        # Checkpoint: skip if already done
        if is_done(mol_idx_str):
            n_skipped += 1
            n_done += 1
            print(f"\n  [{n_done}/{n_total}] {mol_idx_str} "
                  f"({row['smiles'][:30]}) -- SKIPPED (already done)")
            continue

        n_done += 1
        print(f"\n\n{'*'*70}")
        print(f"  [{n_done}/{n_total}] {mol_idx_str}: {row['smiles']}")
        print(f"{'*'*70}")

        result = process_molecule(row, args)
        save_result(mol_idx_str, result)

        if result["status"] == "OK":
            n_ok += 1
            print(f"\n  >>> {mol_idx_str}: DFT PA = {result['dft_pa']:.1f} kJ/mol"
                  f"  [{result['wall_time_s']:.0f}s]")
        elif result["status"] == "NO_SITES":
            n_ok += 1
            print(f"\n  >>> {mol_idx_str}: No protonation sites found")
        else:
            n_failed += 1
            print(f"\n  !!! {mol_idx_str}: {result['status']}")

        elapsed = time.time() - t_start
        rate = (n_done - n_skipped) / elapsed if elapsed > 0 else 0
        remaining = n_total - n_done
        if rate > 0:
            eta_s = remaining / rate
            print(f"  Progress: {n_done}/{n_total}, "
                  f"OK={n_ok}, Failed={n_failed}, Skipped={n_skipped}, "
                  f"ETA={eta_s/3600:.1f}h")

    # Final summary
    total_time = time.time() - t_start
    print("\n\n")
    print("=" * 70)
    print(f"  COMPLETE")
    print(f"  Total: {n_total}, OK: {n_ok}, Failed: {n_failed}, "
          f"Skipped: {n_skipped}")
    print(f"  Wall time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print("=" * 70)


if __name__ == "__main__":
    main()

