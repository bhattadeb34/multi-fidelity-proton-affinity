# NIST PM7-descriptor sensitivity

This bundle supports the ESI sensitivity analysis that removes the 13 neutral
and 13 protonated PM7 property descriptors while retaining PM7 proton affinity
as the low-fidelity baseline. The production feature selector and fixed NIST
folds were used. ExtraTrees gives 2.9252 +/- 0.2571 kcal/mol with fold feature
counts 190, 214, 222, 212, and 170.

Reproduce the analysis from the corrected target table with:

```bash
python scripts/calculations/run_nist_pm7_descriptor_ablation.py
```

`validation_report.json` recomputes every fold MAE from `predictions.csv`.
`SHA256SUMS` covers all files in this directory except the manifest itself.
