#!/usr/bin/env python3
"""
Collect per-molecule JSON results into a single CSV + JSON summary.

Usage:
    python collect_results.py                     # default results/ dir
    python collect_results.py --results-dir results --output pa_nist_full
"""
import argparse
import json
import os
import glob
import pandas as pd
import numpy as np


def collect(results_dir, output_prefix):
    files = sorted(glob.glob(os.path.join(results_dir, "mol_*.json")))
    print(f"Found {len(files)} result files in {results_dir}/")

    if not files:
        print("No results to collect.")
        return

    rows = []
    all_data = []

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        all_data.append(data)

        row = {
            "global_idx": data.get("global_idx"),
            "smiles": data.get("smiles"),
            "exp_pa": data.get("exp_pa"),
            "dft_pa": data.get("dft_pa"),
            "error": data.get("error"),
            "status": data.get("status"),
            "best_site": data.get("best_site"),
            "n_sites": data.get("n_sites"),
            "level": data.get("level"),
            "wall_time_s": data.get("wall_time_s"),
        }

        # Neutral properties
        neutral = data.get("neutral", {})
        if neutral:
            for key in ["E_elec", "H_total", "ZPE_kjmol",
                        "HOMO_eV", "LUMO_eV", "HOMO_LUMO_gap_eV",
                        "dipole_debye", "n_basis", "n_electrons", "n_atoms",
                        "nuclear_repulsion_Ha", "n_imaginary"]:
                row[f"neutral_{key}"] = neutral.get(key)

        # Best protonated properties
        prot = data.get("protonated_best", {})
        if prot:
            for key in ["E_elec", "H_total", "ZPE_kjmol",
                        "HOMO_eV", "LUMO_eV", "HOMO_LUMO_gap_eV",
                        "dipole_debye", "n_imaginary"]:
                row[f"prot_{key}"] = prot.get(key)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Statistics
    ok = df[df["status"] == "OK"]
    print(f"\nResults: {len(df)} total, {len(ok)} OK, "
          f"{len(df) - len(ok)} failed/no-sites")

    if len(ok) > 0 and "error" in ok.columns:
        errors = ok["error"].dropna()
        if len(errors) > 0:
            print(f"\nPA Error Statistics (DFT - Exp):")
            print(f"  MAE  = {errors.abs().mean():.1f} kJ/mol "
                  f"({errors.abs().mean()/4.184:.1f} kcal/mol)")
            print(f"  RMSE = {np.sqrt((errors**2).mean()):.1f} kJ/mol")
            print(f"  MSD  = {errors.mean():+.1f} kJ/mol")
            print(f"  Max  = {errors.abs().max():.1f} kJ/mol")
            print(f"  N    = {len(errors)}")

    # Save CSV (flat table, no coords/frequencies)
    csv_path = f"{output_prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}")

    # Save full JSON (with coords, frequencies, all site details)
    json_path = f"{output_prefix}.json"
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=1)
    print(f"JSON saved to: {json_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect PA results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="pa_nist_full")
    args = parser.parse_args()

    collect(args.results_dir, args.output)
