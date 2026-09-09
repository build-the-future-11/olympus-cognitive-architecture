# Final Research Report

## Executive Summary

Pantheon is an executable research prototype for frozen claim packages, isolated execution, provenance capture, hash-based auditing, canary contamination checks, numerical disagreement metrics, and rule-based adjudication. It has real controlled evidence and a completed external-agent benchmark artifact.

The controlled result is strong but narrow: on 160 engineered fault comparisons, the full canonical audit localized all injected classes, versus 62.5% for pairwise artifact comparison and 12.5% for numeric-only comparison. This demonstrates the intended mechanisms on faults designed to exercise them; it is not an estimate of real-world accuracy.

The follow-on external result is a bounded negative capability result. Two small local agents completed 34 isolated trajectories over 17 frozen CORE-Bench v1.1 OOD tasks containing 20 questions. Both agents answered 0/20 questions, and neither authored a valid report on any task (0/34 combined). The harness is artifact-complete, but the agents failed the post-study capability floor. Therefore the study cannot estimate false-consensus prevalence among capable research agents.

## Research Questions

1. Can canonical frozen-package auditing improve reproducibility-failure localization relative to pairwise or numerical agreement alone?
2. In a frozen OOD task suite, can two independent local agent trajectories produce enough substantive answers to measure agreement and false consensus against canonical answers?

The controlled evidence supports the first question for the injected fault taxonomy. The external run answers the second question negatively for the two evaluated small models.

## What Was Implemented

- package schema and freezer;
- fresh temporary-workspace subprocess runner;
- canonical and observed hash capture;
- provenance files and independence metadata;
- hidden-state canary check;
- disagreement metrics;
- numeric-only, pairwise-artifact, and canonical Pantheon adjudicators;
- seeded fault suite;
- public-data cross-implementation experiment;
- CORE-Bench capsule preparation and hash verification;
- provider-backed local agent loop with bounded decoding and turns;
- safe archive extraction, canonical-answer withholding, network-denied shell execution, and separate ephemeral workspaces;
- failure-preserving normalization and report-origin provenance;
- question-level scoring, Wilson intervals, task-cluster bootstrap intervals, paired analysis, and task diagnostics;
- evidence and scientific-readiness gates.

## Experiments Actually Run

### Controlled seeded faults

200 subprocess executions; 20 synthetic package seeds; 8 conditions; 160 comparisons; 480 adjudicator/baseline evaluations. The fault generator supplies the ground-truth label.

### Public-data case study

Ten prespecified stratified splits; proposer is scikit-learn L-BFGS logistic regression; replicator is an independent numpy batch-gradient-descent implementation. Twenty training/evaluation runs total.

### Frozen CORE-Bench v1.1 OOD study

The definitive protocol, `PANTHEON-COREBENCH-OOD-LOCAL-20260903-V4`, froze 17 official OOD capsules containing 20 questions across four fields and Python/R tasks. Agent A was `qwen3:0.6b`; Agent B was `llama3.2:1b`. The run produced 34 isolated trajectories with no recorded provider errors. The task subset was a size-bounded convenience selection, not a random or representative benchmark sample.

## Controlled Results

| Method | Localization accuracy | Correct / total |
|---|---:|---:|
| Numeric only | 0.125 | 20 / 160 |
| Pairwise artifacts | 0.625 | 100 / 160 |
| Full Pantheon | 1.000 | 160 / 160 |

Removing canonical reference information reduces localization from 1.000 to 0.625; removing artifact evidence entirely reduces it to 0.125. In the shared-bug condition, canonical auditing detects all 20 correlated code-drift cases that both weaker baselines label as reproduced.

## Public-Data Case

| Quantity | Proposer | Replicator |
|---|---:|---:|
| Mean accuracy | 0.9727 ± 0.0134 | 0.9783 ± 0.0101 |
| Mean ROC AUC | 0.9872 ± 0.0101 | 0.9948 ± 0.0050 |
| Mean accuracy improvement over majority | 0.3434 ± 0.0134 | 0.3490 ± 0.0101 |

Direction agreement is 1.000; mean absolute effect difference is 0.006993. No formal significance claim is made because the ten train/test splits overlap and are not independent studies.

## External OOD Result

| Outcome | Agent A | Agent B | Combined / joint |
|---|---:|---:|---:|
| Canonically correct answers | 0 / 20 | 0 / 20 | — |
| Non-null answers | 0 / 20 | 0 / 20 | 0 / 20 jointly answered |
| Agent-authored valid reports | 0 / 17 | 0 / 17 | 0 / 34 trajectories |
| Tool calls | 12 | 284 | 296 |
| Runtime | 666.3 s | 1112.6 s | 1778.8 s |

There were 0/20 substantive agreements and 0/20 observed false-consensus events. Because no question received a substantive answer from either agent, these zeros do not estimate agreement quality or false-consensus prevalence conditional on capable task completion. The Wilson 95% upper bound for each observed zero question-level rate is approximately 0.161, but the failed capability floor remains the decisive interpretive constraint.

## Research Integrity and Gate Interpretation

The frozen V4 `PUBLICATION_READINESS.json` gate passed because it checked structural completeness: task and question counts, distinct identities, domain coverage, OOD labeling, file presence, numeric outputs, and absence of provider errors. Its report check counted harness-normalized `report.json` files as present.

Post-study provenance analysis separated file presence from agent authorship. [`poststudy/SCIENTIFIC_READINESS.json`](poststudy/SCIENTIFIC_READINESS.json) is therefore the authoritative interpretation gate. It records a complete artifact and a failed capability floor. Its thresholds are a post-study reporting safeguard, not preregistered hypothesis tests.

No run was dropped for poor performance, no null answer was converted into agreement, and no harness-normalized report is represented as agent-authored.

## Strongest Supported Conclusions

- Canonical frozen-package auditing adds information that proposer-vs-replicator agreement alone lacks on the engineered fault suite, especially under correlated implementation drift and hidden-state contamination.
- The external harness completed its frozen task matrix and retained failures correctly.
- The two evaluated small local models were not capable enough under the frozen protocol to support the intended false-consensus analysis.

## Unsupported Conclusions

- Pantheon reduces false consensus among capable LLM research agents.
- A 0/20 observed false-consensus count implies a low real-world rate.
- Pantheon generalizes across providers, frontier models, independently administered hosts, or representative research tasks.
- Pantheon has independently reproduced a published claim end to end.
- Perfect controlled localization implies perfect or high natural-failure localization.

## Remaining Limitations

- both evaluated agents failed the minimum answer and report floors;
- both models ran through the same local Ollama provider and host;
- the 17-task subset was a nonrandom convenience selection;
- CORE-Bench OOD is publicly available and therefore not guaranteed contamination-free;
- no blinded human/expert adjudication;
- engineered controlled fault taxonomy;
- no adversarial package generation;
- macOS sandbox isolation is weaker than separately administered VMs or hosts;
- no confirmatory run under a newly frozen capable-agent protocol.

## Reproduction Instructions

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
./scripts/reproduce_all.sh
```

For preserved local OOD evidence and post-study analysis:

```bash
../../.venv/bin/python analysis/analyze_external_rigor.py
../../.venv/bin/python poststudy/scientific_readiness_gate.py
make verify
```

The current repaired source is not the frozen V4 source. The execution wrappers
therefore stop before execution rather than resuming V4 outputs. A confirmatory
run requires a separately frozen V5 protocol with versioned output namespaces
and bound model identities; the draft V5 source snapshot is not a run record.

## File / Evidence Map

See `EVIDENCE_LEDGER.md`, `RESEARCH_TRUTH.md`, `results/`, `runs/`, `configs/local_ood_protocol.json`, and `poststudy/SCIENTIFIC_READINESS.json`.

## Verdict

**COMPLETE NEGATIVE-RESULT ARTIFACT; NOT READY FOR A POSITIVE OR BROAD CROSS-AGENT CLAIM.** Keep the infrastructure and controlled mechanism evidence. A confirmatory scientific study requires capable, preferably provider-diverse agents, a prespecified capability floor, independently administered isolation, representative task sampling, and blinded expert adjudication.
