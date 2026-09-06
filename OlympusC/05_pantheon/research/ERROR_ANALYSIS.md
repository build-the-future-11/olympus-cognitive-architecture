# Error Analysis

## Full Pantheon
The full adjudicator made **0 localization errors on 160 controlled comparisons**. This should not be read as evidence that the task is solved: the injected faults are deliberately observable and map closely to the taxonomy. The important error-analysis result is therefore the absence of challenging natural failures in the current suite.

## Numeric-only baseline
Numeric-only comparison succeeds only on clean exact reproduction. It cannot distinguish why values differ and misses faults that preserve numerical agreement. In metric-sign faults it reports generic non-reproduction instead of metric mismatch.

## Pairwise-artifact baseline
Pairwise artifact comparison handles one-sided data/environment/implementation/protocol drift, but fails when both agents share the same corrupted code, when the metric changes through runtime configuration, and when a canary exposes hidden-state contamination without a package-file change.

## Public-data case study
Across ten prespecified splits, both implementations improved over the majority baseline. The largest observed absolute difference in the improvement metric was 0.02098 accuracy points (split seed 101); four splits had zero difference at the accuracy level. The numpy implementation had higher mean AUC in this limited comparison, but this is not presented as a superiority claim because hyperparameters were not matched to establish a fair algorithm comparison.
