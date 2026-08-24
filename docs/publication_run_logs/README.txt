# Controlled rerun logs

These logs were copied verbatim from the isolated clean checkout used for the
controlled rerun.
The environment is recorded in `../clean_env_versions.json`.

The two small wrapper logs (`overnight_queue.log` and `light_queue.log`) record
orchestration failures caused by an incorrect completion-string check and two
initially missing synchronized cost-input files. The underlying NIST, learning
curve, SHAP, aggregation, Tanimoto, and figure jobs were not corrupted or
restarted. The cost inputs were synchronized, the cost analysis completed, and
only the failed SI candidate figure was retried after a backward-compatible
column-name fix.

Important logs:

- `core_rerun.log`: DFT descriptor ablations, DFT-only control, row-wise split
  audit, and bundle validation.
- `nist_exact_rerun.log`: final NIST production rerun used in the repository.
- `learning_curves.log`: all 220 learning-curve cells.
- `shap_all.log`: final NIST and k-means SHAP refits and figures.
- `compute_cost_analysis.log`: computational-cost calculation and corrected
  speedup statements.
- `screening_summary.log`: reconciled prospective stage counts.
- `make_figures_main.log`, `make_figures_screening.log`, and
  `plot_si_candidates_retry.log`: figure regeneration and the single targeted
  plotting retry.

`SHA256SUMS` covers this README and every log in the directory.
