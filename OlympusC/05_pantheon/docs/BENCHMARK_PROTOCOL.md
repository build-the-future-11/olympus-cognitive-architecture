# Seeded-Fault Benchmark Protocol

1. Generate twenty frozen synthetic research packages using seeds 0–19.
2. Each package contains a claim, protocol, data, analysis script, environment lock, schema, and canonical file hashes.
3. Run the proposer in a fresh temporary workspace and freeze its metrics/provenance.
4. For each condition, run the replicator in a separate fresh temporary workspace. All communication is file-based.
5. Conditions: clean, data mismatch, metric mismatch, implementation mismatch, environment mismatch, protocol ambiguity, hidden-state contamination, shared correlated implementation bug.
6. Compare three methods on the exact same frozen run pair: numeric-only, pairwise artifact diff, and full canonical Pantheon audit.
7. Ground-truth labels are determined by the fault injector, not by inspecting the adjudicator output.
8. The shared-bug condition mutates both proposer and replicator code identically. This deliberately creates pairwise consensus around a noncanonical implementation.
9. Store every subprocess's metrics and provenance under `runs/seeded_faults/` and aggregate predictions under `results/seeded_faults_predictions.csv`.
10. Do not interpret controlled localization accuracy as expected real-world accuracy.
