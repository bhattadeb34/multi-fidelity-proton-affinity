# DFT Proton-Affinity Pipeline (B3LYP / def2-TZVP)

Compute gas-phase proton affinities from first principles using PySCF
(+ optional `gpu4pyscf` acceleration) and the geomeTRIC optimizer. For
every input molecule the pipeline:

1. Enumerates all neutral N / O / S protonation sites from the SMILES.
2. Builds a 3D starting geometry (RDKit ETKDGv3 + MMFF94).
3. Runs a B3LYP/def2-TZVP geometry optimization on the neutral and on
   every protonated form.
4. Computes the analytical Hessian (GPU if available, else CPU), obtains
   vibrational frequencies, and applies ZPE + thermal corrections with
   ZPE scale factor 0.9850 (see `ZPE_SCALE_FACTORS` in `pa_calculator.py`).
5. Reports the site-wise PA

   $$\mathrm{PA} = H_{298}(\text{neutral}) + H(\mathrm{H}^{+}) - H_{298}(\text{protonated}),\quad H(\mathrm{H}^{+}) = \tfrac{5}{2}RT = 1.48\ \text{kcal/mol}$$

   and flags the most-favourable site (highest PA) as the canonical value.

This is the exact pipeline used to generate the DFT reference data in
the paper.

## Files

| File                       | Purpose                                                                                |
|----------------------------|----------------------------------------------------------------------------------------|
| `pa_calculator.py`         | Core: site enumeration, SCF + geom-opt + Hessian, thermo, PA assembly.                 |
| `run_dft_pa.py`            | Driver: iterates over an input CSV, checkpoints per molecule, writes JSON per mol.     |
| `collect_results.py`       | Aggregates per-molecule JSON files into a flat CSV + a combined JSON.                  |
| `generate_protonated.py`   | Stand-alone RDKit utility: emit protonated SMILES for a single input or a CSV.         |
| `submit_dft.sh`            | SLURM submission template (single A100; edit the `<...>` placeholders).               |

All Python modules have **no hardcoded paths or credentials**; everything
is command-line driven. Cluster-specific settings are confined to
`submit_dft.sh`.

## Requirements

* Python ≥ 3.10.
* Packages: `pyscf`, `rdkit`, `numpy`, `pandas`, `geometric`.
* Optional (recommended for >20 heavy atoms):
  `gpu4pyscf` with CUDA ≥ 12.x and an NVIDIA GPU (A100/H100/RTX tested).
  Without it the code transparently falls back to CPU.

Reference install (GPU):
```bash
conda create -n gpu_dft -c conda-forge python=3.11 \
    pyscf rdkit numpy pandas geometric
conda activate gpu_dft
pip install "gpu4pyscf-cuda12x"          # match your CUDA toolkit
```

## Input CSV format

Two required columns, `mol_idx` and `smiles`. `mol_idx` is preserved
verbatim as the JSON filename, so choose something globally unique:

```csv
mol_idx,smiles
mol_001,CCO
mol_002,c1ccncc1
mol_003,CC(=O)N
```

## Running locally

```bash
export RESULTS_DIR=dft_results
python run_dft_pa.py \
    --csv        my_molecules.csv \
    --save-files \
    --files-dir  dft_files
```

Options:
| flag              | default         | meaning                                              |
|-------------------|-----------------|------------------------------------------------------|
| `--csv`           | *(required)*    | CSV with `mol_idx,smiles`                            |
| `--basis`         | `def2-TZVP`     | any PySCF basis string                               |
| `--xc`            | `B3LYP`         | any PySCF XC functional                              |
| `--no-freq`       | off             | skip Hessian + thermochem (electronic-only energies) |
| `--save-files`    | off             | write optimized geometry / molden files per site     |
| `--files-dir`     | `dft_files`     | parent directory for per-molecule subfolders         |

Resume behaviour: molecules already present in `$RESULTS_DIR` with
`status in {OK, NO_SITES}` are skipped.

## Running on a SLURM cluster

1. Copy this folder to your cluster scratch.
2. Edit the placeholders at the top of `submit_dft.sh`:
   * `--account=<YOUR_SLURM_ACCOUNT>`
   * `--partition=<YOUR_GPU_PARTITION>`
   * `CONDA_ACTIVATE` / `CONDA_ENV` (path to your env)
   * `INPUT_CSV`, `RESULTS_DIR`, `FILES_DIR` (or override via env vars)
3. Submit:
   ```bash
   mkdir -p logs
   INPUT_CSV=my_molecules.csv \
   RESULTS_DIR=dft_results \
   FILES_DIR=dft_files \
   sbatch submit_dft.sh
   ```

## Output layout

```
dft_results/                           # one JSON summary per molecule
    mol_001.json
    mol_002.json
    ...

dft_files/                             # only if --save-files
    mol_001/
        neutral/
            neutral_initial.xyz        # RDKit ETKDGv3 + MMFF94 start
            neutral_optimized.xyz      # geomeTRIC-optimized geometry
            neutral.molden             # MO coefficients (load into Avogadro etc.)
            neutral.log                # energies, dipole, HOMO/LUMO, thermo
            neutral_freq.txt           # vib frequencies (raw + scaled, cm-1)
            neutral_hessian.txt        # full Hessian matrix (Ha/Bohr^2)
        site_1/
            protonated_site1_initial.xyz
            protonated_site1_optimized.xyz
            protonated_site1.molden
            protonated_site1.log
            protonated_site1_freq.txt
            protonated_site1_hessian.txt
        site_2/
            protonated_site2_*         # same 6 files, 1-indexed per N/O/S site
        ...
    mol_002/
        ...
```

`site_k/` is created for every enumerated protonation site (1-indexed),
so a molecule with 3 N/O/S sites yields `site_1/`, `site_2/`, `site_3/`
under `mol_XXX/`. This is the exact layout used to generate the paper's
DFT reference datasets.

Each `mol_<idx>.json` contains:
```jsonc
{
    "mol_idx": "mol_001",
    "smiles":  "CCO",
    "level":   "B3LYP/def2-TZVP",
    "status":  "OK",          // or "NO_SITES" or "FAILED: ..."
    "dft_pa":  776.52,         // kJ/mol, most-favourable site
    "best_site":   "O(C,H)",
    "n_sites":     1,
    "neutral":       { "E_elec": ..., "H_total": ..., "HOMO_eV": ..., ... },
    "protonated_best": { ... },
    "all_sites":   [ { "atom": "O", "pa_kjmol": 776.52, "status": "OK", ... }, ... ],
    "wall_time_s": 81.3
}
```

## Aggregate results

```bash
python collect_results.py --results-dir dft_results --output pa_summary
# -> pa_summary.csv    (flat table)
# -> pa_summary.json   (full records, incl. coords / frequencies)
```

## Enumerate protonated SMILES standalone

```bash
python generate_protonated.py "c1ccncc1"          # single SMILES
python generate_protonated.py my_molecules.csv    # CSV -> new CSV
```

## Method details (for reproducibility)

* Functional / basis: **B3LYP / def2-TZVP** (edit via `--xc` / `--basis`).
* SCF: tight convergence (`conv_tol = 1e-10`), grid level 4,
  max 200 SCF iterations.
* Geometry optimisation: geomeTRIC, max 200 steps; failed optimisations
  fall back to the input geometry.
* Thermochemistry: IGRRHO at 298.15 K; vibrational frequencies scaled by
  0.9850; imaginary frequencies |ν| < 50 cm⁻¹ are retained without
  reoptimization.
* Proton enthalpy: $H(\mathrm{H}^{+}) = \tfrac{5}{2}RT$ = 1.48 kcal/mol.

See Sec. *B3LYP/def2-TZVP DFT Calculation Details* of the Supporting
Information for a full description.

## Citation

If you use this pipeline, please cite the paper (see the top-level
[`README.md`](../README.md)).
