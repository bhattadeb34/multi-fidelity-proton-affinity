# PM7 Proton-Affinity Pipeline

Compute PM7 (MOPAC) proton affinities for arbitrary SMILES. For every input
molecule the pipeline:

1. Enumerates all neutral N/O/S/P protonation sites.
2. Runs a PM7 geometry-opt + property calc on the neutral and on every
   protonated form in parallel.
3. Computes the site-wise proton affinity

   $$\mathrm{PA}_{\mathrm{PM7}} = H_f(\text{neutral}) + 365.7 - H_f(\text{protonated})
   \ \ [\text{kcal/mol}]$$

   where 365.7 kcal/mol is the gas-phase proton enthalpy at 298 K.

This is the exact pipeline used to generate the 16,384-molecule and
251-molecule PM7 datasets in the paper.

## Files

| File                    | Purpose                                                                      |
|-------------------------|------------------------------------------------------------------------------|
| `mopac_calculator.py`   | Thin wrapper around the `mopac` executable (writes `.mop`, parses `.out`).   |
| `run_pm7_parallel.py`   | Driver: site enumeration, `ProcessPoolExecutor`, checkpointing, PA math.     |
| `submit_pm7.sh`         | SLURM array-job template (chunks a CSV into N parallel array tasks).         |

All Python modules have **no hardcoded paths**; everything is driven from
command-line arguments. Site-specific details (SLURM account, partition,
email, conda env) are confined to `submit_pm7.sh` and are clearly marked
with `<...>` placeholders.

## Requirements

* MOPAC 2016 or later on `PATH` (free academic license at
  [openmopac.net](http://openmopac.net/)). Verify with `which mopac`.
* Python ≥ 3.9 with:
  `rdkit`, `numpy`, `pandas`, `tqdm`.

Minimal install (conda):
```bash
conda create -n pm7_calc -c conda-forge python=3.11 rdkit numpy pandas tqdm
conda activate pm7_calc
```

## Input CSV format

A single column named `smiles` (default; change with `--smiles_column`):

```csv
smiles
CCO
CC(=O)N
c1ccncc1
```

## Running locally (one machine, no SLURM)

```bash
python run_pm7_parallel.py my_molecules.csv \
    --project_name  my_pm7_run \
    --n_processes   8 \
    --total_chunks  1 \
    --current_chunk 1 \
    --force
```

Output is written to `./my_pm7_run/` as a set of JSON checkpoint files:
```
protonation_map.json        # {neutral_smiles: [{smiles, site_index, site_element}, ...]}
properties_neutral.json     # {neutral_smiles: {heat_of_formation, homo_ev, ..., success}}
properties_protonated.json  # {prot_smiles:    {..., site_index, site_element, parent_neutral}}
proton_affinities.json      # {neutral_smiles: {protonation_sites: [...], max_pa, min_pa, num_sites}}
progress.json               # resume bookkeeping
failed_molecules.json       # molecules/sites that failed
```
Re-running the same command resumes from the last checkpoint (every 100
calculations).

## Running on a SLURM cluster

1. Copy this folder to your cluster scratch.
2. Edit the placeholders at the top of `submit_pm7.sh`:
   * `--account=<YOUR_SLURM_ACCOUNT>`
   * `--partition=<YOUR_CPU_PARTITION>`
   * `--mail-user=<YOUR_EMAIL@EXAMPLE.COM>`
   * `CONDA_ACTIVATE` / `CONDA_ENV` (path to your env)
3. Make sure `--array=1-N` matches the `N_CHUNKS` argument.
4. Submit:
   ```bash
   sbatch submit_pm7.sh /absolute/path/to/my_molecules.csv 10
   ```
   Each array task processes `ceil(N_mols / N_CHUNKS)` molecules. Results
   land in `run_my_molecules_job<JOBID>/results_chunk_<k>_of_10/`.

## PM7 keyword line (for reproducibility)

`PM7 PRECISE GNORM=0.001 SCFCRT=1.D-8`; `CHARGE=1 UHF` is added for
protonated species. See `mopac_calculator.py::write_mopac_input`.

## Tuning notes

* Performance is single-threaded per MOPAC process, so set
  `--n_processes` equal to `$SLURM_CPUS_PER_TASK`. The template already
  does this.
* `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` are exported in
  `submit_pm7.sh` — do not remove; over-subscription destroys throughput.
* Wall-clock estimate is ~144 s × (1 neutral + ~3 protonated sites) per
  molecule on a single core.

## Citation

If you use this pipeline, please cite the paper (see the top-level
[`README.md`](../README.md)).
