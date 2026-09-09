# Error Analysis

## Full Pantheon
The full adjudicator made **0 localization errors on 160 controlled comparisons**. This should not be read as evidence that the task is solved: the injected faults are deliberately observable and map closely to the taxonomy. The important error-analysis result is therefore the absence of challenging natural failures in the current suite.

## Numeric-only baseline
Numeric-only comparison succeeds only on clean exact reproduction. It cannot distinguish why values differ and misses faults that preserve numerical agreement. In metric-sign faults it reports generic non-reproduction instead of metric mismatch.

## Pairwise-artifact baseline
Pairwise artifact comparison handles one-sided data/environment/implementation/protocol drift, but fails when both agents share the same corrupted code, when the metric changes through runtime configuration, and when a canary exposes hidden-state contamination without a package-file change.

## Public-data case study
Across ten prespecified splits, both implementations improved over the majority baseline. The largest observed absolute difference in the improvement metric was 0.02098 accuracy points (split seed 101); four splits had zero difference at the accuracy level. The numpy implementation had higher mean AUC in this limited comparison, but this is not presented as a superiority claim because hyperparameters were not matched to establish a fair algorithm comparison.

## CORE-Bench OOD study

The definitive external failure was capability, not provider availability or artifact loss. All 34 trajectories completed without a recorded provider error, but Agent A and Agent B each produced 0/20 non-null answers and 0/20 correct answers. Neither authored a valid report on any task (0/17 each). Existing `report.json` files are harness failure-normalization artifacts whose origin is retained explicitly.

The agents failed differently at the trajectory level: Agent A used 12 tool calls over 41 turns, while Agent B used 284 tool calls over 184 turns and saturated the 20-turn bound on seven tasks. Neither behavior yielded a scored answer. Because every question was an abstention for both agents, the observed 0/20 agreement and 0/20 false-consensus counts do not diagnose capable-agent agreement quality.
