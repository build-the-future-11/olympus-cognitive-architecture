# Pantheon
### Canonical Artifact Auditing for Research-Agent Replication

Pantheon is a research prototype for **frozen claim packages, isolated replication, provenance capture, and artifact-grounded disagreement diagnosis**.

The core idea is simple: two research runs agreeing with each other is not sufficient evidence of replication. Both runs should also be compared against a frozen canonical scientific package so shared bugs, hidden-state contamination, and artifact drift can be detected.

## Current evidence boundary

This repository now contains real executable evidence, but the claim is intentionally narrow.

**Verified in this repo:** on a controlled 20-seed × 8-condition injected-fault benchmark, full Pantheon localized 160/160 seeded fault comparisons, versus 100/160 for proposer-vs-replicator artifact comparison and 20/160 for numeric-only comparison. A separate public-data cross-implementation case study produced direction agreement on 10/10 prespecified splits. A frozen study also executed 34 isolated local-agent trajectories on 17 official CORE-Bench v1.1 OOD capsules (20 questions, four fields). Both small models scored 0/20 and authored 0/34 valid reports. This is a retained negative result: it validates execution and failure semantics, but does not estimate false-consensus prevalence among capable agents.

**Not verified:** successful autonomous replication, naturally occurring false-consensus reduction, frontier-agent performance, or independent reproduction of a published claim. See `PROJECT_STATUS.md` and `TRUTH_MAP.md` before using any result.

## Why canonical auditing matters

A pairwise replication check can fail under correlated drift:

```text
frozen package:        canonical analysis.awk
proposer run:          corrupted analysis.awk
replicator run:        same corrupted analysis.awk

pairwise comparison:   MATCH  -> misleading "reproduced"
canonical comparison:  DRIFT  -> implementation mismatch
```

Pantheon makes the frozen package, not the other agent, the reference point.

## Implemented system

- frozen claim package + canonical manifest;
- fresh temporary-workspace execution;
- separate subprocess proposer/replicator runs;
- data/code/protocol/environment hashes;
- run provenance and runtime persistence;
- proposer-only canary contamination probe;
- standardized disagreement metrics (`D_z`, relative error, sign agreement, CI overlap);
- numeric-only and pairwise-artifact baselines;
- rule-based canonical adjudicator;
- seeded fault benchmark;
- public-data cross-implementation case study;
- isolated CORE-Bench OOD runner with safe archive extraction and canonical-answer withholding;
- local Ollama/OpenAI-compatible provider adapters with bounded turns and completion tokens;
- explicit abstention semantics (joint null is not agreement) and report-origin provenance;
- question-level Wilson and task-cluster bootstrap analysis;
- evidence ledger, truth report, manuscript, figures, tests, and one-command reproduction.

## Quickstart

```bash
cd 05_pantheon

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

pytest -q
./scripts/smoke_test.sh
```

## Reproduce all evidence

```bash
./scripts/reproduce_all.sh
```

The workflow runs:

1. environment check;
2. tests;
3. clean smoke reproduction;
4. controlled seeded-fault benchmark;
5. public-data cross-implementation experiment;
6. ablation/figure generation;
7. evidence validation.

## Individual commands

```bash
# Build one frozen synthetic package
python scripts/make_demo_package.py --out packages/demo --seed 42

# Separate proposer / replicator execution
python scripts/run_proposer.py \
  --package packages/demo \
  --run-dir runs/demo/proposer \
  --seed 42

python scripts/run_replicator.py \
  --package packages/demo \
  --run-dir runs/demo/replicator \
  --seed 42 \
  --fresh-workspace

# Canonical comparison
python scripts/compare_results.py \
  --package packages/demo \
  --proposer runs/demo/proposer \
  --replicator runs/demo/replicator \
  --out runs/demo/comparison.json

# Benchmarks
python scripts/run_seeded_faults.py --config configs/seeded_faults.yaml
python scripts/run_public_case_study.py
python scripts/build_paper_assets.py
python scripts/validate_evidence.py
```

## Main controlled benchmark

| Method | Correct | Fault localization accuracy |
|---|---:|---:|
| Numeric only | 20 / 160 | 0.125 |
| Pairwise artifacts | 100 / 160 | 0.625 |
| Full Pantheon | 160 / 160 | 1.000 |

The 100% result is **not** a real-world accuracy estimate. The faults are engineered mechanism tests. See `research/FALSIFICATION_REPORT.md`.

## Public-data case study

The script loads the Breast Cancer Wisconsin Diagnostic dataset from scikit-learn and evaluates two different implementations across ten prespecified stratified splits:

- proposer: scikit-learn logistic regression / L-BFGS;
- replicator: in-repo numpy batch gradient descent.

Mean accuracy is 0.9727 ± 0.0134 for the proposer and 0.9783 ± 0.0101 for the replicator. Both improve over the majority baseline on all 10 splits; mean absolute difference in the improvement metric is 0.00699 accuracy points. These are descriptive cross-implementation results, not a medical claim and not a paper replication.

## Repository layout

```text
05_pantheon/
├── README.md
├── RESEARCH_TRUTH.md
├── EVIDENCE_LEDGER.md
├── FINAL_RESEARCH_REPORT.md
├── audit/
├── configs/
├── docs/
├── figures/
├── packages/
├── paper/
├── research/
├── results/
├── runs/
├── schemas/
├── scripts/
├── src/pantheon/
└── tests/
```

## Evidence map

Start here:

- `RESEARCH_TRUTH.md` — exact claim boundary;
- `EVIDENCE_LEDGER.md` — claim → evidence mapping;
- `FINAL_RESEARCH_REPORT.md` — full completion report;
- `results/seeded_faults_predictions.csv` — all controlled classifications;
- `results/experiment_registry.csv` — run registry;
- `runs/seeded_faults/` — per-run raw metrics/provenance;
- `results/public_case_results.csv` and `runs/public_case/` — public case artifacts;
- `figures/` — plots generated from result files;
- `paper/main.tex` — evidence-grounded manuscript source;
- `output/pdf/pantheon-canonical-auditing.pdf` — compiled six-page preprint;
- `results/external/rigorous_summary.json` — question-level OOD inference;
- `runs/external/` — 34 complete definitive trajectories.

## Research status

**COMPLETE NEGATIVE-RESULT PREPRINT, not ICLR-ready for a broad cross-agent claim.**

The protocol prototype is runnable and evidence-backed. The current paper honestly reports that the tested small models never crossed the task-completion threshold. A strong ICLR iteration still needs capable separately hosted agents, naturally occurring substantive answers and disagreements, stronger container/VM isolation, and a human adjudication baseline.

## Completed external validation

The definitive protocol is `PANTHEON-COREBENCH-OOD-LOCAL-20260903-V4`. It freezes 17 official OOD capsules, 20 questions, four fields, exact source and model hashes, temperature zero, 768 completion tokens per turn, 20 turns, safe extraction, separate ephemeral workspaces, network-denied shell execution, and canonical-answer withholding.

### Reproduce the completed local study

```bash
cd 05_pantheon
ollama serve
PANTHEON_PYTHON=../../.venv/bin/python bash scripts/run_local_ood_study.sh
../../.venv/bin/python analysis/analyze_external_rigor.py
make verify
```

`setup_external.sh` downloads the exact 17-task convenience subset. The selection reaches the frozen 20-question gate while excluding two 400–800 MB capsules; it is not a random sample and must not be described as representative.

The frozen V4 structural gate writes `PUBLICATION-READY` because all preregistered artifacts exist; it is retained byte-for-byte for reproducibility and is not a scientific or venue-readiness judgment. The canonical post-study `poststudy/SCIENTIFIC_READINESS.json` gate instead reports `NEGATIVE_RESULT_ARTIFACT_COMPLETE_CAPABILITY_FLOOR_FAILED`. The decisive measured outcomes are 0/20 canonical accuracy, 0/20 non-null answer rate, 0/34 agent-authored report compliance, 0/20 substantive agreement, and 0/20 false consensus. The Wilson 95% upper bound for each zero event rate is 0.161. With no substantive answers, false-consensus prevalence is unidentified for capable agents.

Useful commands:

```bash
# Verify the frozen protocol and all manuscript evidence
make verify
```

The external runner is resumable (`--resume`) and stores every task under `runs/external/<task_id>/`.
