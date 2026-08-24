"""
compute_cost_analysis.py
========================
Self-contained DFT vs PM7+ML compute cost analysis.
All input data is in docs/dft_logs/ (no HPC access needed).

Inputs:
    docs/dft_logs/hpc_chunk_timing.csv       -- per-chunk wall times from SLURM logs
    docs/dft_logs/kmeans_per_molecule_timing.csv  -- per-molecule k-means DFT wall times
    docs/dft_logs/pm7_ml_inference_cost.json  -- PM7+ML route timing (from audit2)
    docs/dft_compute_cost_summary.csv         -- ICDS billing summary

Outputs to stdout. No files modified.

Usage:
    python scripts/analysis/compute_cost_analysis.py
"""

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent.parent
LOGS = HERE / "docs" / "dft_logs"
DOCS = HERE / "docs"
OUTPUT = HERE / "results" / "computational_cost"


def load_chunk_timing():
    rows = []
    with open(LOGS / "hpc_chunk_timing.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_per_molecule_timing():
    rows = []
    with open(LOGS / "kmeans_per_molecule_timing.csv") as f:
        for r in csv.DictReader(f):
            if r["status"] == "OK":
                rows.append(r)
    return rows


def load_inference_cost():
    with open(LOGS / "pm7_ml_inference_cost.json") as f:
        return json.load(f)


def load_billing_summary():
    rows = []
    with open(DOCS / "dft_compute_cost_summary.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_compute_environment():
    with open(LOGS / "compute_environment.json") as f:
        return json.load(f)


def select_billing_row(rows, label):
    """Select one billing row by a case-insensitive label fragment."""
    matches = [row for row in rows if label.lower() in row["Calculation"].lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected one billing row containing {label!r}")
    return matches[0]


def parse_job_id(text):
    """Extract the unique long numeric job identifier."""
    matches = re.findall(r"\b(\d{6,})\b", text)
    if len(matches) != 1:
        raise ValueError(f"Expected one job identifier in {text!r}")
    return matches[0]


def parse_count(text):
    """Parse the first comma-formatted integer from a metadata field."""
    match = re.search(r"[\d,]+", text)
    if match is None:
        raise ValueError(f"No count found in {text!r}")
    return int(match.group(0).replace(",", ""))


def main():
    chunks = load_chunk_timing()
    per_mol = load_per_molecule_timing()
    inference = load_inference_cost()
    billing = load_billing_summary()
    environment = load_compute_environment()

    nist_billing = select_billing_row(billing, "NIST dataset")
    km_billing = select_billing_row(billing, "k-means dataset")
    prospective_billing = select_billing_row(
        billing, "Corrected prospective candidates"
    )
    nist_job = parse_job_id(nist_billing["Calculation"])
    km_job = parse_job_id(km_billing["Calculation"])
    prospective_job = parse_job_id(prospective_billing["Calculation"])

    # --- DFT chunk-level timing (from SLURM output logs) ---
    nist_chunks = [c for c in chunks if c["job"].endswith(nist_job)]
    km_chunks = [c for c in chunks if c["job"].endswith(km_job)]
    prospective_chunks = [
        c for c in chunks if c["job"].endswith(prospective_job)
    ]
    if not nist_chunks or not km_chunks:
        raise ValueError("DFT timing rows do not match the billing job identifiers")
    if len(prospective_chunks) != 1:
        raise ValueError("Expected exactly one corrected prospective timing row")

    nist_total_h = sum(float(c["wall_h"]) for c in nist_chunks)
    # The billing record gives the analyzed molecule count. Raw job chunks also
    # contain the records later excluded during feature construction.
    nist_total_mols = parse_count(nist_billing["Molecules"])
    km_total_h = sum(float(c["wall_h"]) for c in km_chunks)
    km_total_mols_ok = sum(int(c["n_ok"]) for c in km_chunks)
    km_total_mols = sum(int(c["n_mols"]) for c in km_chunks)
    km_failed = sum(int(c["n_failed"]) for c in km_chunks)
    pareto_h = float(prospective_chunks[0]["wall_h"])
    pareto_mols = int(prospective_chunks[0]["n_ok"])

    print("=" * 70)
    print("DFT COMPUTE COST ANALYSIS")
    print("=" * 70)
    print()
    print(
        f"All DFT jobs ran on {environment['hardware']}, "
        f"{environment['cpus_per_job']} CPUs, "
        f"{environment['memory_gb_per_job']} GB RAM"
    )
    print(
        f"Level of theory: {environment['level_of_theory']} "
        f"({environment['software']})"
    )
    print(f"Platform: {environment['platform']}")
    print()

    print("--- Per-chunk wall times (from SLURM output logs) ---")
    print()
    print(f"NIST dataset (job {nist_job}): {len(nist_chunks)} parallel A100 chunks")
    for c in nist_chunks:
        print(f"  Chunk {c['chunk']}: {c['wall_h']}h "
              f"({c['n_ok']}/{c['n_mols']} molecules OK)")
    print(f"  Total compute: {nist_total_h:.1f}h, {nist_total_mols} molecules")
    print(f"  Per molecule: {nist_total_h / nist_total_mols:.3f}h "
          f"= {nist_total_h * 3600 / nist_total_mols:.1f}s")
    print()

    print(f"K-means dataset (job {km_job}): {len(km_chunks)} parallel A100 chunks")
    for c in km_chunks:
        print(f"  Chunk {c['chunk']}: {c['wall_h']}h "
              f"({c['n_ok']}/{c['n_mols']} molecules OK, "
              f"{c['n_failed']} failed)")
    print(f"  Total compute: {km_total_h:.1f}h, "
          f"{km_total_mols_ok} OK / {km_total_mols} total "
          f"({km_failed} failed)")
    print(f"  Per molecule: {km_total_h / km_total_mols_ok:.3f}h "
          f"= {km_total_h * 3600 / km_total_mols_ok:.1f}s")
    print()

    print(f"Corrected prospective candidates (job {prospective_job}): 1 A100 job")
    print(f"  Wall time: {pareto_h}h, {pareto_mols} molecules")
    print(f"  Per molecule: {pareto_h / pareto_mols:.3f}h "
          f"= {pareto_h * 3600 / pareto_mols:.1f}s")
    print()

    # --- Per-molecule timing (k-means, from JSON result files) ---
    wall_s = np.array([float(r["wall_s"]) for r in per_mol])
    n_sites = np.array([int(r["n_sites"]) for r in per_mol])

    print("--- Per-molecule k-means DFT timing ---")
    print(f"  Source: wall times extracted from per-molecule JSON results")
    print(f"  Molecules: {len(per_mol)} OK")
    print(f"  Total: {wall_s.sum() / 3600:.1f}h")
    print(f"  Per molecule: mean={wall_s.mean() / 3600:.3f}h "
          f"({wall_s.mean():.1f}s), "
          f"median={np.median(wall_s) / 3600:.3f}h")
    print(f"  Range: {wall_s.min() / 3600:.2f} - {wall_s.max() / 3600:.2f}h")
    print(f"  Sites/mol: mean={n_sites.mean():.1f}, "
          f"range={n_sites.min()}-{n_sites.max()}")
    print()

    # --- Cross-check chunk totals vs per-molecule totals ---
    print("--- Cross-check ---")
    print(f"  Chunk total: {km_total_h:.1f}h")
    print(f"  Per-mol sum: {wall_s.sum() / 3600:.1f}h")
    diff_pct = abs(km_total_h - wall_s.sum() / 3600) / km_total_h * 100
    print(f"  Difference: {diff_pct:.2f}%")
    print()

    # --- Billing cross-check ---
    print("--- ICDS billing summary (from dft_compute_cost_summary.csv) ---")
    for b in billing:
        print(f"  {b['Calculation']}")
        print(f"    Wall time: {b['Wall time']}, "
              f"Credits: {b['Credits']}, "
              f"Cost: {b['Cost (USD)']}")
    print()

    nist_cr = float(nist_billing["Credits"])
    km_cr = float(km_billing["Credits"])

    print("  Credit rates (credits/h):")
    print(f"    NIST:   {nist_cr / nist_total_h:.4f}")
    print(f"    Kmeans: {km_cr / km_total_h:.4f}")
    print("    Corrected prospective job: billing credits not separately retrieved")
    print()

    # --- PM7+ML inference cost ---
    totals = inference["totals"]
    ml_s = totals["ml_route_seconds_per_molecule"]
    pm7_frac = totals["fraction_of_ml_route_that_is_pm7"]

    print("--- PM7+ML inference route ---")
    print(f"  Source: audit2_inference_cost.py (measured on local machine)")
    print(f"  PM7 leg: {inference['pm7_leg']['seconds_per_molecule_mean']:.2f} s/mol "
          f"({inference['pm7_leg']['jobs_per_molecule_mean']:.1f} MOPAC jobs/mol)")
    print(f"  Feature leg: "
          f"{inference['feature_leg']['seconds_per_molecule_mean']:.4f} s/mol")
    print(f"  Model leg: "
          f"{inference['model_leg']['predict_seconds_per_molecule']:.5f} s/mol")
    print(f"  Total ML route: {ml_s:.3f} s/mol "
          f"({pm7_frac * 100:.0f}% is PM7)")
    print()

    # --- Speedup ---
    nist_s_per_mol = nist_total_h * 3600 / nist_total_mols
    km_s_per_mol = wall_s.mean()

    print("=" * 70)
    print("SPEEDUP SUMMARY")
    print("=" * 70)
    print()
    print(f"  DFT (NIST):     {nist_s_per_mol:.1f} s/mol "
          f"({nist_total_h / nist_total_mols:.3f} h/mol)")
    print(f"  DFT (k-means):  {km_s_per_mol:.1f} s/mol "
          f"({km_s_per_mol / 3600:.3f} h/mol)")
    print(f"  PM7+ML route:   {ml_s:.3f} s/mol")
    print()
    speedup_nist = nist_s_per_mol / ml_s
    speedup_km = km_s_per_mol / ml_s
    print(f"  Speedup (NIST basis):    {speedup_nist:.0f}x "
          f"(10^{np.log10(speedup_nist):.2f})")
    print(f"  Speedup (k-means basis): {speedup_km:.0f}x "
          f"(10^{np.log10(speedup_km):.2f})")
    print()
    print(f"  k-means DFT range: {wall_s.min() / 3600:.2f} - "
          f"{wall_s.max() / 3600:.2f} h/mol "
          f"(mean {wall_s.mean() / 3600:.2f})")
    print()
    print("  Hardware caveat: PM7 and ML measured on a single MOPAC thread")
    print("  on a local machine. DFT measured on A100 GPUs. The ratio is")
    print("  hardware-crossed. Quote to one significant figure.")

    source_files = [
        LOGS / "hpc_chunk_timing.csv",
        LOGS / "kmeans_per_molecule_timing.csv",
        LOGS / "pm7_ml_inference_cost.json",
        LOGS / "deployed_model_inference_benchmark.json",
        LOGS / "compute_environment.json",
        DOCS / "dft_compute_cost_summary.csv",
    ]
    summary = {
        "compute_environment": environment,
        "nist": {
            "molecules": nist_total_mols,
            "total_gpu_wall_hours": nist_total_h,
            "seconds_per_molecule": nist_s_per_mol,
        },
        "kmeans": {
            "molecules": int(len(per_mol)),
            "sites": int(n_sites.sum()),
            "total_gpu_wall_hours": km_total_h,
            "seconds_per_molecule_mean": km_s_per_mol,
            "seconds_per_molecule_median": float(np.median(wall_s)),
        },
        "corrected_prospective": {
            "slurm_job": int(prospective_job),
            "molecules": pareto_mols,
            "elapsed_seconds": int(float(prospective_chunks[0]["wall_s"])),
            "elapsed_hours": pareto_h,
        },
        "pm7_ml_route": {
            "seconds_per_molecule": ml_s,
            "pm7_fraction": pm7_frac,
            "benchmark_caveat": inference["hardware_caveat"],
        },
        "speedup": {
            "nist_basis": speedup_nist,
            "kmeans_basis": speedup_km,
            "caveat": "hardware-crossed comparison; quote to one significant figure",
        },
        "source_sha256": {
            str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved machine-readable summary: {OUTPUT / 'summary.json'}")


if __name__ == "__main__":
    main()
