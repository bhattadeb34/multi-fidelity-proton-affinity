# `docs/` — reference material shipped with the repo

These files record inputs, commands, environment details, validation checks,
and numerical results for the controlled rerun.

| File | Description |
|---|---|
| `REPRODUCIBILITY.txt` | Scientific settings, derived values, human decisions, leakage controls, and reproducibility boundaries. |
| `dft_compute_cost_summary.csv` | Raw data behind SI Table 5 (DFT wall times and credit/USD costs per stage). |
| `llm_schema_retry_manuscript_note.txt` | Provenance and suggested Methods language for Opus JSON validation and targeted schema repair. |
| `publication_rerun_checklist.txt` | Frozen rerun order, reference results, and artifact acceptance criteria. |
| `publication_methodology_audit.txt` | Leakage audit, corrected inherited issues, and interpretation boundaries. |
| `publication_results_summary.txt` | Validated production, ablation, learning-curve, screening, and cost numbers from the controlled rerun. |
| `repository_data_and_revision_guide.txt` | How to combine GitHub with the Zenodo data, apply corrected targets, reproduce results, and interpret changes from the original submission. |
| `clean_env_versions.json` | Exact software environment used for the controlled rerun. |
| `input_data_fingerprints.json` | Byte sizes and SHA-256 hashes of the targets and parsed DFT archive used by the rerun. |
| `publication_run_logs/` | Verbatim execution logs and SHA-256 manifest for the controlled rerun. |

The screening-specific scientific record is kept under `data/screening/` in
the Zenodo archive. It includes the frozen pipeline outputs, raw Claude
responses and reconciled verdicts, the complete 30-candidate prospective DFT
calculations, and the PubChem/ZINC identity checks for the provisional five
leads. The repository `screening/` directory contains only configuration,
executable code, and its usage guide.
