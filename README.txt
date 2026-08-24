# Proton Affinity Prediction via Multi-Fidelity Delta Learning

**Authors**: Debjyoti Bhattacharya, Yifan Liu, Valentino R. Cooper, Wesley F. Reinhart
**Affiliations**: Pennsylvania State University; Oak Ridge National Laboratory

Code and data-processing pipeline accompanying the manuscript. The model
predicts a signed correction to a PM7 semi-empirical proton affinity
against a higher-fidelity reference (experimental or B3LYP/def2-TZVP
DFT). See the manuscript + SI for methods, datasets, and results.

## Release layout

The Zenodo release keeps source code and scientific data in separate sibling
directories:

```text
multi-fidelity-proton-affinity_2026-08-19/
├── code/
├── data/
└── artifacts/
```

Run commands from `code/`. Scripts first look for `code/data/` to support a
standard Git clone, then use the sibling `data/` directory provided by the
Zenodo release. The optional sibling `artifacts/` directory contains compact
plot inputs, selected-feature tables, and verified fitted-model checkpoints.
It does not replace or modify the authoritative `data/` tree.

## Code layout

```
code/
├── scripts/                 # ML / analysis / plotting
│   ├── calculations/            # Dataset build + model training
│   │   ├── build_dataset.py         # DFT JSONs -> processed/dataset.json
│   │   ├── build_pm7_dataset.py     # PM7 CSVs  -> processed/pm7_dataset.json
│   │   ├── build_targets.py         # Features + targets -> data/targets/
│   │   ├── featurize/               # MACCS/Morgan/RDKit/Mordred/PM7/site
│   │   ├── train_models.py          # 5-fold CV, up to 15 models, PM7 features
│   │   ├── train_models_dft.py      # Same + B3LYP ablation
│   │   ├── learning_curve.py        # Learning-curve sweeps
│   │   ├── compute_shap.py          # SHAP over each dataset's best tree model
│   │   ├── analyze_results.py       # Summarize CV results
│   │   ├── dataset_molecules_selection.py  # k-means selection of DFT/PM7 molecules from ZINC
│   │   └── select_kmeans_1024.py    # k-means selection of 1024 ZINC mols
│   ├── plotting/                # All manuscript + SI figures
│   └── analysis/                # Site-agreement, feature counts, Tanimoto bias
│
├── screening/               # Prospective-screening code (data -> data/screening/)
│   └── scripts/{execution,plotting}/
│
├── pm7_scripts/             # Stand-alone PM7 (MOPAC) PA pipeline
├── dft_scripts/             # Stand-alone B3LYP/def2-TZVP DFT PA pipeline
│
├── results/                 # CV / SHAP / learning-curve outputs (~15 MB, in-repo)
├── figures/                 # Manuscript + SI PDFs/PNGs (~16 MB, in-repo)
├── logs/                    # Training logs (gitignored)
├── docs/                    # Reference material shipped with the repo
├── make_figures.py         # regenerate every figure in the paper (one entry point)
├── requirements.txt
└── README.txt                # this file
```

`data/`, `logs/`, and all `__pycache__/` / `*.log` / `catboost_info/`
are gitignored. `results/` and `figures/` are kept in-repo (~31 MB
total) so the paper's numbers and figures are viewable without
retraining.

## Headline results

- NIST: ExtraTrees, **2.8731 +/- 0.2643 kcal/mol** (five-fold CV).
- k-means: RandomForest, **7.4355 +/- 1.0013 kcal/mol**, using
  molecule-grouped outer folds across 251 molecules / 821 sites.
- Prospective screening: **3.9932 kcal/mol MAE** over 30 completed DFT
  candidates; 29 lie in the 210-235 kcal/mol target window.

See `docs/publication_results_summary.txt` for the validated ablations,
learning curves, screening counts, computational-cost correction, and
interpretation notes. Machine-readable validation reports and SHA-256
manifests accompany the corresponding directories under `results/`,
`figures/`, and the Zenodo `data/screening/` tree.

See `docs/REPRODUCIBILITY.txt` for the distinction between configured scientific
choices, data-derived values, the final candidate review step, and archived
machine-specific provenance.

See `docs/EXACT_REPRODUCIBILITY.txt` for environment locking, artifact hashes,
safe scratch outputs, and feature-selection reproducibility.

For an independent reproduction, including how to combine this repository
with the complete Zenodo data release, see
`docs/repository_data_and_revision_guide.txt`.

## Data

The dated release is about 9.7 GB after extraction. Its compressed
size depends on the archive format. It contains the authoritative DFT and PM7
calculations,
features, targets, screening records, raw LLM responses, and prospective DFT
files. These files are too large for Git and are distributed with the complete
Zenodo release.

The complete release contains the target files used for the reported results. No manual
file replacement is required. Their hashes and the controlled input hashes
are recorded in `docs/input_data_fingerprints.json`.

To run code that reads scientific data, extract the complete release and run
commands from its `code/` directory:

```
multi-fidelity-proton-affinity_2026-08-19/
├── code/
│   ├── scripts/
│   ├── results/
│   └── ...
└── data/
│   ├── nist1185/
│   ├── kmeans251/
│   ├── pm7/ | pm7_source_raw/
│   ├── features/
│   ├── processed/
│   ├── targets/
    └── screening/
```

Scripts resolve either `code/data/`, which is convenient for a standalone Git
clone, or the sibling `data/` directory in the complete release. A symbolic
link from `code/data` to another data location is also supported. The
versioned `results/` and `figures/` directories are enough to inspect the
reported model results. Figures that use raw or screening data require the
complete release.

## Setup

Python 3.10.18 is the reference interpreter. Create the reference environment
from the repository root:

```bash
conda env create -f environment.yml
conda activate mfpa-py310
```

`requirements.txt` lists the direct dependencies. For an exact Linux x86_64
reproduction, create the recorded conda base and install the hash-locked pip
packages instead:

```bash
conda create --prefix ./mfpa-exact-env \
  --file conda-lock-py310-linux-64.txt
conda activate ./mfpa-exact-env
python -m pip install --require-hashes \
  -r requirements-lock-py310-linux-x86_64.txt
```

The pip resolver report used to generate the lock is archived under
`docs/environment/`. This exact lock is platform-specific. Use
`environment.yml` when portability matters more than bitwise agreement.

Core deps: `rdkit`, `numpy`, `pandas`, `scikit-learn`, `xgboost`,
`lightgbm`, `catboost`, `mordred`, `matplotlib`, `faiss-cpu`, `scipy`,
`openpyxl`. `pyscf` + `gpu4pyscf` are needed only for the DFT pipeline.

## Reproducing the main paper

```bash
# 1. Parse raw data
python scripts/calculations/build_dataset.py
python scripts/calculations/build_pm7_dataset.py

# 2. Featurize
python scripts/calculations/featurize/build_features.py --dataset all

# 3. Build ML targets into a separate reproduction directory
python scripts/calculations/build_targets.py \
  --output-dir reproduction/targets

# 4. Train (5-fold CV, up to 15 models)
# k-means automatically uses molecule-grouped folds on record_id.
python scripts/calculations/train_models.py     --dataset all   # PM7 + cheminformatics
python scripts/calculations/train_models_dft.py --dataset all   # + B3LYP ablation

# Focused NIST sensitivity reported in the ESI
python scripts/calculations/run_nist_pm7_descriptor_ablation.py

# 5. Summarize + plot
python scripts/calculations/analyze_results.py
python scripts/calculations/learning_curve.py
python scripts/calculations/compute_shap.py
python make_figures.py              # regenerate every figure in one go
```

Figures land in `figures/` as both `.pdf` and `.png`.

Use `build_targets.py --dry-run` to validate target construction without
writing files. Existing target outputs are protected from replacement unless
`--force` is supplied explicitly.

### Just the plots

If `results/` is already present (it ships with the repo), one can
skip training and regenerate every figure directly:

```bash
python make_figures.py --list              # show the ordered steps
python make_figures.py --only main         # main-paper figures only
python make_figures.py --only screening    # prospective-screening figures
python make_figures.py                     # everything (~80 s on a laptop)
```

Each step reports OK/FAIL with timing; use `-v` to stream a failing
step's output or `--stop-on-error` to abort on the first failure.

The complete release also provides a faster checkpoint path. It reads only
the saved tables under `artifacts/plot_inputs/`. It does not featurize,
select features, train models, run PM7 or DFT, or call the LLM service.

```bash
python scripts/reproducibility/validate_quick_artifacts.py
python scripts/reproducibility/make_figures_quick.py
```

To construct the convenience layer in a prepared release, run:

```bash
python scripts/reproducibility/build_quick_artifacts.py
python scripts/reproducibility/export_model_checkpoints.py --dataset all
python scripts/reproducibility/validate_quick_artifacts.py
```

The checkpoint exporter uses the archived fold membership and selected
feature names. It saves a fitted model only after its held-out predictions
match the archived `predictions.csv` values. See
`docs/EXACT_REPRODUCIBILITY.txt` for the entry points and validation rules.

**Note on paper figures vs. generated output.** `make_figures.py`
regenerates every plot produced by the code, including intermediate and
diagnostic plots that are not used in the manuscript or SI. Some manuscript
figures are composites assembled from generated quantitative panels. Final
layout elements and schematic workflow diagrams were prepared separately.

## Dataset molecule selection (k-means)

`scripts/calculations/dataset_molecules_selection.py` reproduces the
k-means-based DFT/PM7 molecule selection from the filtered ZINC library.
Input: `data/screening/zinc_raw/filtered_821k.csv` (available in the Zenodo archive).

```bash
# Reproduce the paper's 256 DFT + PM7 selection
python scripts/calculations/dataset_molecules_selection.py \
    --input data/screening/zinc_raw/filtered_821k.csv \
    --output-dir data/screening/kmeans_selection/ \
    --n-dft 256 --n-pm7-per-dft 64

# Optional: also generate latent-space plot
python scripts/calculations/dataset_molecules_selection.py \
    --input data/screening/zinc_raw/filtered_821k.csv --plot
```

Running this on `filtered_821k.csv` with `random_state=42` reproduces
the exact 256 DFT molecules used in the paper (verified against
`DFT_enhanced_256molecules_20250904_140333.csv`).

For bit-level comparison with the archived reference run, verify the
three primary input files against `docs/input_data_fingerprints.json`. Parquet
files rewritten by a different Arrow version may be table-equivalent while
having a different byte hash. The archived copies identified by the recorded
hashes are authoritative for exact artifact reproduction.

## Prospective screening

See `screening/README.txt` for the 7-stage pipeline
(`01_build_index.py` -> `07_pareto_select.py` -> DFT -> `11_parse_dft_files.py`).

**LLM verification (Stage 6):** The archived screening run used Claude Opus
4.6 via Google Vertex AI (`vertex_ai/claude-opus-4-6@default`) at
`temperature=0`. All 2,368 molecules reaching the rule/LLM stage were
evaluated. Failed or invalid API responses were retried with the same model,
prompt, and scientific criteria. The final verdict table, raw responses,
run configuration, attempt logs, stage counts, and checksums are archived under
`data/screening/iter1/`. The 30 completed prospective DFT
calculations and provisional top-five selection are under
`data/screening/dft_validation/iter1_20260813/`. The exact-structure PubChem and
ZINC checks supporting the top-five discussion in the ESI are under
`data/screening/database_validation/top5_20260816/`. That bundle includes the
retrieval script, normalized tables, raw API responses, request metadata,
structure files, and a SHA-256 manifest.

## Quantum-chemistry pipelines (PM7 / DFT)

Self-contained, so others can compute fresh PA data for arbitrary SMILES
without touching the ML code:

| Folder         | What it does                                | Requirements                                                |
|----------------|---------------------------------------------|-------------------------------------------------------------|
| `pm7_scripts/` | Parallel PM7 PA via MOPAC                   | `mopac`, `rdkit`, `pandas`, `tqdm`                          |
| `dft_scripts/` | B3LYP/def2-TZVP PA via PySCF (+ gpu4pyscf)  | `pyscf`, `rdkit`, `geometric`; optional `gpu4pyscf` + CUDA  |

Each folder has its own README with install, CSV format, local and SLURM usage,
and output layout. The reusable `submit_*.sh` templates use placeholders for
cluster-specific settings. The archived prospective submission script is retained
under `data/screening/dft_validation/` with the corresponding calculations.

## Sanity checks / auxiliary analyses

```bash
python scripts/analysis/verify_features.py         # feature-count check
python scripts/analysis/analyze_site_agreement.py  # PM7 site-selection
python scripts/analysis/tanimoto_bias_analysis.py  # Tanimoto-bias figure
python scripts/analysis/audit_release.py --verify-checksums  # release audit
```

## Release validation

Run the release audit from the repository root:

```bash
python scripts/analysis/audit_release.py --verify-checksums
git diff --check
```

For a complete extracted code-and-data release, run the read-only scientific
validation entry point:

```bash
# Run from code/. Build the release manifest only after code and data are final.
python scripts/reproducibility/build_release_manifest.py ..
python scripts/analysis/validate_complete_release.py
python scripts/analysis/validate_complete_release.py --verify-checksums
python scripts/analysis/validate_complete_release.py \
  --verify-checksums --publication-ready
```

The checksum validation command hashes every file and can take several
minutes. Neither validation command reruns PM7, DFT, LLM screening, or
external database queries. If the release contents change after the manifest
is created, rebuild it with the manifest command and `--force`.

## License

The source code is released under the MIT License. Dataset reuse is governed
by the license stated in the corresponding Zenodo record.
