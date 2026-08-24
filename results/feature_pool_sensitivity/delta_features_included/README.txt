# Descriptor-delta feature-pool sensitivity

The production workflow excludes every column beginning with `delta_`, which
preserves the submitted 5,295-feature definition but also removes legitimate
neutral-to-protonated descriptor differences. This sensitivity run instead
excluded label/target columns by exact name while admitting engineered
descriptor deltas, for 7,344 candidate features. All other corrected k-means
settings were retained, including molecule-grouped outer folds.

RandomForest MAE was 7.1522 +/- 0.8813 kcal/mol. Fold feature counts were 97,
86, 67, 109, and 31. Seventy-four descriptor-delta features appeared in at
least one fold (19 met the >=3/5 consensus rule). The complete prediction
bundle passes `scripts/analysis/validate_result_bundle.py`.

This is a sensitivity analysis only. It does not replace the production
5,295-feature result or the model used for prospective screening.
