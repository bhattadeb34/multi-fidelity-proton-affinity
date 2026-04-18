# `scripts/analysis/` — one-off analyses

Standalone scripts that consume the already-generated data / results and
produce SI figures or verification tables. They do not fit into the main
training pipeline.

| Script | What it does | Output |
|---|---|---|
| `analyze_site_agreement.py` | Compares the PM7-predicted best protonation site against the site whose PM7 PA is closest to the experimental value (used as a proxy for the experimentally preferred site). Covers the 508 NIST molecules with ≥2 candidate sites. | Printed summary table → SI §3.2, Table "Site selection agreement". |
| `verify_features.py`        | Loads `data/targets/nist1155_ml.parquet` and `kmeans251_ml.parquet` and counts every feature category (MACCS / Morgan / RDKit × 3 states / Mordred × 3 states / PM7 × 3 states / site). Checks the totals against the claims in SI Algorithm 1. | Printed per-dataset feature tally. |
| `tanimoto_bias_analysis.py` | Plots prediction bias (PA_pred − PA_DFT) vs maximum Tanimoto similarity to the k-means training set for the 30 DFT-validated Pareto candidates. Top-5 leads are marked by shape; colour encodes in-window vs out-of-window. | `figures/tanimoto_bias_plot_final.pdf` → SI "Tanimoto bias" figure. |

Run from anywhere:

```bash
python scripts/analysis/verify_features.py
python scripts/analysis/analyze_site_agreement.py
python scripts/analysis/tanimoto_bias_analysis.py
```
