# Proton Affinity Prediction via Multi-Fidelity Delta Learning

**Authors**: Debjyoti Bhattacharya, Yifan Liu, Valentino R. Cooper, Wesley F. Reinhart
**Affiliations**: Pennsylvania State University; Oak Ridge National Laboratory

Code and data-processing pipeline accompanying the manuscript. The model
predicts a signed correction to a PM7 semi-empirical proton affinity
against a higher-fidelity reference (experimental or B3LYP/def2-TZVP
DFT). See the manuscript + SI for methods, datasets, and results.

## Repository layout

```
proton-affinity-paper/
├── data/                    # All datasets (gitignored; hosted externally)
│   ├── 1185_molecules/          # NIST B3LYP DFT results
│   ├── 251_molecules/           # k-means B3LYP DFT results
│   ├── pm7/ | pm7_source_raw/   # PM7 inputs + raw outputs
│   ├── features/                # Featurized parquet/csv per dataset
│   ├── processed/               # Parsed dataset.json / pm7_dataset.json
│   ├── targets/                 # ML-ready parquet (NIST + k-means)
│   └── screening/               # Prospective-screening datasets
│       ├── zinc_raw/                # Raw ZINC + Morgan fingerprints
│       ├── processed/               # FAISS index, PCA metadata
│       └── iter{1,2,3}/             # Per-iteration candidates/PM7/DFT
│
├── scripts/                 # ML / analysis / plotting
│   ├── calculations/            # Dataset build + model training
│   │   ├── build_dataset.py         # DFT JSONs -> processed/dataset.json
│   │   ├── build_pm7_dataset.py     # PM7 CSVs  -> processed/pm7_dataset.json
│   │   ├── build_targets.py         # Features + targets -> data/targets/
│   │   ├── featurize/               # MACCS/Morgan/RDKit/Mordred/PM7/site
│   │   ├── train_models.py          # 5-fold CV, 14 models, PM7 features
│   │   ├── train_models_dft.py      # Same + B3LYP ablation
│   │   ├── learning_curve.py        # Learning-curve sweeps
│   │   ├── compute_shap.py          # SHAP over trained ExtraTrees
│   │   ├── analyze_results.py       # Summarize CV results
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
└── README.md                # this file
```

`data/`, `logs/`, and all `__pycache__/` / `*.log` / `catboost_info/`
are gitignored. `results/` and `figures/` are kept in-repo (~31 MB
total) so the paper's numbers and figures are viewable without
retraining.

## Data

The raw + processed datasets (~7 GB — NIST / k-means DFT outputs, PM7
outputs, features, screening inputs and per-iteration outputs) are too
large for GitHub and are **not** included in this repository. They
will be released on **Zenodo** at the time of publication; the DOI and
download link will be added here once available.

To run any code that reads from `data/`, download the Zenodo archive,
extract it at the repo root, and confirm the resulting layout matches
the `data/` tree shown in *Repository layout* above:

```
proton-affinity-paper/
├── data/                     # <-- extracted Zenodo archive goes here
│   ├── 1185_molecules/
│   ├── 251_molecules/
│   ├── pm7/ | pm7_source_raw/
│   ├── features/
│   ├── processed/
│   ├── targets/
│   └── screening/
├── scripts/
├── ...
```

The scripts expect `data/` to sit at the repo root; no other path is
configurable. `results/` and `figures/` (shipped in-repo) are the
outputs of running the full pipeline against this `data/` tree, so
reviewers who only want to inspect the paper's numbers or regenerate
the figures (`python make_figures.py`) do **not** need to download
the Zenodo archive.

## Setup

```bash
pip install -r requirements.txt
```

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

# 3. Build ML targets
python scripts/calculations/build_targets.py

# 4. Train (5-fold CV, 14 models)
python scripts/calculations/train_models.py     --dataset all   # PM7-only
python scripts/calculations/train_models_dft.py --dataset all   # + B3LYP ablation

# 5. Summarize + plot
python scripts/calculations/analyze_results.py
python scripts/calculations/learning_curve.py
python scripts/calculations/compute_shap.py
python make_figures.py              # regenerate every figure in one go
```

Figures land in `figures/` as both `.pdf` and `.png`.

### Just the plots (for reviewers)

If `results/` is already present (it ships with the repo), reviewers can
skip training and regenerate every figure directly:

```bash
python make_figures.py --list              # show the 12 ordered steps
python make_figures.py --only main         # main-paper figures only
python make_figures.py --only screening    # prospective-screening figures
python make_figures.py                     # everything (~80 s on a laptop)
```

Each step reports OK/FAIL with timing; use `-v` to stream a failing
step's output or `--stop-on-error` to abort on the first failure.

**Note on paper figures vs. generated output.** `make_figures.py`
regenerates *every* plot produced by the code, which is a superset of
what appears in the manuscript and SI — some intermediate/diagnostic
plots are not used in the paper. Several published figures are also
composites: individual panels are generated here, then joined,
re-laid-out, or visually polished in PowerPoint (labels, arrows,
panel letters, colour tweaks), and all schematic / workflow diagrams
are drawn in PowerPoint from scratch. The underlying quantitative
content always comes from the scripts in this repo; only the layout
and schematics are hand-edited.

## Prospective screening

See `screening/README.md` for the 7-stage pipeline
(`01_build_index.py` -> `07_pareto_select.py` -> DFT -> `11_parse_dft_files.py`).

## Quantum-chemistry pipelines (PM7 / DFT)

Self-contained, so others can compute fresh PA data for arbitrary SMILES
without touching the ML code:

| Folder         | What it does                                | Requirements                                                |
|----------------|---------------------------------------------|-------------------------------------------------------------|
| `pm7_scripts/` | Parallel PM7 PA via MOPAC                   | `mopac`, `rdkit`, `pandas`, `tqdm`                          |
| `dft_scripts/` | B3LYP/def2-TZVP PA via PySCF (+ gpu4pyscf)  | `pyscf`, `rdkit`, `geometric`; optional `gpu4pyscf` + CUDA  |

Each folder has its own README with install, CSV format, local + SLURM
usage, and output layout. All cluster-specific settings
(`--account`, `--partition`, conda env path) are confined to the two
`submit_*.sh` templates and marked with `<...>` placeholders.

## Sanity checks / auxiliary analyses

```bash
python scripts/analysis/verify_features.py         # feature-count check
python scripts/analysis/analyze_site_agreement.py  # PM7 site-selection
python scripts/analysis/tanimoto_bias_analysis.py  # Tanimoto-bias figure
```
