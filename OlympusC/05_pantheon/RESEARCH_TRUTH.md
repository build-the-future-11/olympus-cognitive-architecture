# Research Truth

## What was actually run

- 12 unit/integration tests (`pytest -q`): all passed in the final local quality gate.
- One clean smoke reproduction package: proposer and replicator executed in separate fresh temporary workspaces and matched exactly.
- Controlled seeded-fault benchmark: **200 subprocess executions** across 20 synthetic seeds and 8 comparison conditions, totaling 160 proposer/replicator comparisons and 480 baseline/adjudicator evaluations.
- Public-data cross-implementation case study: **20 model executions** (2 implementations × 10 prespecified stratified splits) on the Breast Cancer Wisconsin Diagnostic dataset bundled with scikit-learn.
- Evidence validator: passed after checking expected row counts, run status, nonmissing metrics, and required plot/data artifacts.

## What succeeded

- Full Pantheon fault-localization accuracy on the controlled benchmark: **1.000 (160/160)**.
- Pairwise-artifact baseline: **0.625 (100/160)**.
- Numeric-only baseline: **0.125 (20/160)**.
- Shared-bug case: both weaker baselines falsely label all 20 shared-bug cases as reproduced; canonical Pantheon detects implementation drift in all 20.
- Public case direction agreement: **1.000 (10/10)** for improvement over majority accuracy.
- Public case mean absolute proposer/replicator effect difference: **0.006993** accuracy points.

## What failed or was constrained

- An early benchmark implementation exceeded the execution window because every isolated child started scientific Python. The child-side estimator was replaced with a portable AWK implementation while preserving a fresh process/workspace per run; the final benchmark was then restored to 20 seeds. The timed-out partial runs are not used in the final tables.
- No external LLM/model agents were available as independent proposer/replicator systems in this run.
- No Research Atlas or other frozen third-party research package was present in the supplied ZIP, so the original “one genuine real Research Atlas package” milestone remains incomplete.

## Unverified

- Cross-model replication benefits.
- Reduction of false consensus among real LLM agents.
- Performance on naturally occurring research-repository failures.
- Human-vs-Pantheon adjudication quality.
- Generalization beyond the eight injected fault conditions and one public dataset case study.

## Strongest supported claim

> On this controlled seeded-fault suite, canonical frozen-package auditing with provenance and canary checks localizes the injected failure class more accurately than numeric-only or proposer-vs-replicator pairwise comparison, and uniquely detects correlated shared implementation drift.

## Claims that cannot currently be made

- Pantheon makes autonomous AI science reliable.
- Pantheon has been validated across multiple LLM agents or model families.
- Pantheon is state of the art on CORE-Bench, PaperBench, AutoExperiment, or any external agent benchmark.
- Pantheon has reproduced a published research claim end-to-end.
- The controlled 100% localization result estimates real-world failure-localization accuracy.

## Evidence locations

- `results/seeded_faults_predictions.csv`
- `results/execution_registry.csv`
- `results/baselines.csv`
- `results/main_results.csv`
- `results/public_case_results.csv`
- `results/public_case_paired.csv`
- `results/public_case_summary.json`
- `results/ablations.csv`
- `runs/seeded_faults/`
- `runs/public_case/` (summary output is under `results/`)
- `figures/`

## External-validation machinery added after the controlled study

The repository now contains executable provider-backed external-agent machinery under `src/pantheon/external/` and `scripts/run_external_study.py`. It has been locally unit-tested, but **no provider-backed external LLM run is counted in the evidence above unless artifacts exist under `runs/external/`**. API keys and network access are intentionally not fabricated or assumed.

The bundled setup path uses public CORE-Bench training tasks only as a pilot because their canonical answers are public. `scripts/publication_gate.py` refuses to return `PUBLICATION-READY` from public-training evidence alone; held-out/OOD evidence is required.
