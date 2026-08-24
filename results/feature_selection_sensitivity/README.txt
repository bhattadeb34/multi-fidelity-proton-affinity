# Feature-selection sensitivity analysis

This analysis tested three choices in the in-fold selector on the corrected
k-means and NIST workflows:

1. the 1,500-feature training-target-correlation pool versus no cap;
2. the historical 1-SD threshold versus conventional 1-SE; and
3. no feature-count threshold versus the historical minimum-20 fallback.

It also evaluated the simpler standard rule of retaining the nonzero
coefficients at the `LassoCV` alpha minimizing mean five-fold inner-CV MSE.
Outer splits were held fixed. K-means used molecule-grouped folds on
`record_id`; NIST used seeded row-wise folds because it contains one row per
molecule. Every supervised operation was fit only on the outer training fold.

## Result

The recommended rule is:

1. variance filtering;
2. retain the leading 1,500 features by absolute training-fold target
   correlation;
3. remove pairwise-correlated features above 0.95; and
4. retain nonzero coefficients at the minimum-inner-CV-error Lasso alpha.

For k-means, this rule reproduces exactly the five authoritative feature sets
and counts (77, 86, 60, 125, and 79), and therefore preserves the current
Random Forest result of 7.4355 ± 1.0013 kcal/mol. The historical 1-SD plus
minimum-20 code reached these same sets through its fallback, so simplifying
the code does not invalidate the deployed screening model.

For NIST, the official production rerun with the same rule and 1,500-feature
pool gives an ExtraTrees MAE of 2.8756 ± 0.2666 kcal/mol with fold feature
counts of 198, 223, 247, 177, and 179 (mean 204.8). The lightweight sensitivity
driver gave 2.8652 ± 0.2631 kcal/mol; its convenience imputer removed two
all-missing columns before selection in two folds. The production artifacts,
which preserve the established preprocessing path, are authoritative.

The uncapped sensitivity control changed MAE only modestly (k-means RF: 7.4355
to 7.5015; NIST ExtraTrees sensitivity estimate: 2.8652 to 2.8905 kcal/mol)
while taking approximately three times longer in this experiment. This
supports the 1,500-feature pool as a computational dimension-reduction step,
provided it and the minor preprocessing distinction above are reported
explicitly.

See `sensitivity_summary.csv` for the central sensitivity configurations and
the authoritative production NIST row. Values after the ± sign are population
standard deviations across the five fixed outer folds, matching the convention
used in `cv_results.json`.
