# Corrected screening figures

This directory contains only figures retained from the corrected August 2026
prospective-screening rerun.

- `iter1_pareto.{pdf,png}` uses the frozen 615-molecule eligible PA-window
  pool and the 30 selected Pareto candidates.
- `iter1_pa_parity_final.pdf` compares corrected ML predictions with the 30
  completed prospective B3LYP/def2-TZVP calculations.
- `iter1_pa_parity_with_leads.{pdf,png}` combines the same quantitative parity
  panel with the five leads loaded from the deterministic lead table.
- `pa_distribution.pdf` uses the corrected molecule-level maximum-site
  aggregation.
- `iter1_top5_molecule_panels/` contains the five rule-selected priority
  leads from `data/screening/dft_validation/iter1_20260813/top5_leads.csv`.

The submission-era `pa_parity_selected.pdf`, `pareto_scatter.pdf`,
`screening_funnel.pdf`, `top_candidates.png`, and
`iter1_top5_molecule_markers.pdf` were removed because they were not products
of the corrected rerun. `SHA256SUMS` covers every retained figure.
