# Screening configuration

`pipeline_config.json` is the single source for screening choices that may
change between campaigns. Its defaults reproduce the current pipeline.

- `pa_window_kcalmol` is shared by prediction summaries, LLM-verification
  summaries, Pareto filtering, and DFT result parsing.
- `consensus_min_folds`, `featurization`, and `random_forest` define the
  deployed feature/model specification.
- `rule_filter` contains policy thresholds. The training PA maximum and the
  correction mean/standard deviation are intentionally not configured here;
  `pipeline_config.py` derives them from the corrected training parquet.
  The loader rejects the stale broadcast-target parquet by checking that
  multi-site molecules have distinct per-site DFT targets.
- `llm`, `pareto`, and `retrieval` contain operational selection settings.

Changing this file affects future runs only. Existing parquet, CSV, JSONL, and
figure artifacts are not rewritten automatically and must be regenerated from
the earliest affected step. `pareto_report.json` embeds the configuration used
for that selection run, and step 06 writes a timestamped `llm_run_config` JSON
manifest containing its configuration, runtime overrides, and derived training
references.

Change impact:

- Retrieval settings: rerun from step 02.
- Consensus threshold or RandomForest settings: rerun from step 04 or 05,
  respectively.
- Rule/LLM settings: rerun from step 06.
- PA window or Pareto count: rerun step 07 and downstream reporting (step 05/06
  summaries also reflect the window but do not use it to alter predictions).
- Morgan radius: this is part of the feature definition; changing it requires
  regenerating the fingerprint/index and training features and retraining the
  model before screening. Do not change it only for a screening rerun.
