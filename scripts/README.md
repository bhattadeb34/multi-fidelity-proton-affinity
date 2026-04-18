# `scripts/` — training, analysis, and plotting pipeline

Everything here is code for the **main paper** (training, evaluation,
figures). Prospective-screening code lives under `../screening/` — see
`../screening/README.md`.

All scripts assume they are run from the repo root and resolve paths via
`Path(__file__)`, so they work whether you `cd` into the repo root or invoke
them with a relative path.

---

## `calculations/` — dataset build + model training

| Script | Purpose |
|---|---|
| `build_dataset.py`            | Parse B3LYP DFT JSON / folder output → `data/processed/dataset.json` |
| `build_pm7_dataset.py`        | Parse raw PM7 CSVs → `data/processed/pm7_dataset.json` |
| `build_targets.py`            | Join features with experimental / DFT targets → `data/targets/*.parquet` |
| `featurize/build_features.py` | Orchestrate MACCS + Morgan + RDKit + Mordred + PM7 + site features |
| `featurize/fp_maccs.py`       | 167 MACCS keys |
| `featurize/fp_morgan.py`      | 1024 count-based Morgan fingerprints (r=2) |
| `featurize/desc_rdkit.py`     | 210 RDKit descriptors × 3 states (neutral / protonated / delta) |
| `featurize/desc_mordred.py`   | 2D + 3D Mordred descriptors × 3 states |
| `featurize/desc_pm7.py`       | 13 PM7 quantum descriptors × 3 states |
| `featurize/desc_site.py`      | Site-level descriptors (element one-hot, indices, n_sites) |
| `train_models.py`             | 5-fold CV over 14 regressors, PM7-only features |
| `train_models_dft.py`         | Same + B3LYP DFT features (Section 4 ablation) |
| `learning_curve.py`           | Learning-curve sweeps (NIST + k-means, PM7 / DFT) |
| `compute_shap.py`             | SHAP over the deployed ExtraTrees model |
| `analyze_results.py`          | Print / summarize CV outputs |
| `select_kmeans_1024.py`       | k-means selection of 1024 ZINC molecules in latent space (dataset construction for the k-means DFT split) |

## `plotting/` — all manuscript and SI figures

| Script | Produces |
|---|---|
| `plot_results.py`             | Parity plots, model-comparison bar charts |
| `plot_exploration.py`         | PA distributions, PA-vs-MW, dataset exploration |
| `plot_chemical_analysis.py`   | Correction-by-class figures + representative molecules (combined NIST/k-means) |
| `plot_chemical_space_zinc.py` | ZINC latent-space scatter |
| `plot_learning_curves.py`     | Individual-panel learning curves |
| `plot_lc_combined.py`         | Combined 2×2 learning-curve figure |
| `plot_shap.py`                | SHAP summary + dependence plots |
| `plot_site_histogram.py`      | k-means site-count histogram (SI) |
| `plot_style.py`               | Shared matplotlib `rcParams` + journal-spec helpers |

## `analysis/` — one-off analyses and sanity checks

| Script | Purpose |
|---|---|
| `analyze_site_agreement.py`   | Compare PM7 best-site vs experimental proxy (SI §3.2 table) |
| `verify_features.py`          | Cross-check ML-parquet feature counts against SI Algorithm 1 claims |
| `tanimoto_bias_analysis.py`   | Bias vs max Tanimoto similarity for the 30 DFT-validated candidates (SI figure) |
