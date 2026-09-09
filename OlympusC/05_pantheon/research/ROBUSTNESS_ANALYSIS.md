# Robustness and Sensitivity Analysis

## Controlled benchmark robustness

The same eight fault conditions are evaluated on twenty independently generated synthetic datasets (seeds 0–19). Full Pantheon gives the same correct localization on every seed. This checks that the result is not tied to one synthetic table, but it does **not** test a broad range of fault severities or natural failure modes.

## Public-data split sensitivity

The cross-implementation case study is repeated across ten prespecified stratified train/test split seeds. Both implementations outperform the test-set majority baseline on all ten splits. The mean absolute difference in the improvement metric is 0.006993 accuracy points; the maximum observed difference is 0.020979.

Because the ten splits overlap, they should be read as sensitivity checks rather than ten independent experiments suitable for a simple hypothesis test.

## Compute sensitivity

The public proposer uses scikit-learn L-BFGS; the replicator uses 7,000 deterministic numpy batch-gradient steps. The latter is intentionally independent in implementation rather than compute-matched. Runtime comparisons are therefore descriptive only; see `results/efficiency.csv`.

## External-agent execution sensitivity

The frozen OOD study spans 17 tasks, 20 questions, four scientific fields, and both Python and R capsules. It used two distinct local model identities with different observed trajectory profiles. Agent A used 12 tool calls across 41 turns; Agent B used 284 tool calls across 184 turns and reached the 20-turn limit on seven tasks. Both nevertheless returned 0/20 non-null answers and authored 0/17 reports.

This variation demonstrates failure preservation under different local trajectories, not robustness of scientific performance. Both models used the same Ollama provider and host, the subset was not randomly sampled, and no agent crossed the capability floor.

## Not tested

- different hardware or OS environments;
- dependency-version reconstruction in a separate virtual environment/container;
- adversarially chosen artifact faults;
- stochastic neural training;
- capable or frontier LLMs;
- provider-diverse or separately administered agents;
- randomized or exhaustive external-task sampling;
- blinded human adjudication of substantive disagreements.
