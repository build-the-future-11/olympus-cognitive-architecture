# Evidence Ledger

## Controlled and public-data evidence

| Claim | Result | Experiment ID / scope | Evidence file | Status |
|---|---|---|---|---|
| Clean R0 packages reproduce exactly | 20/20 controlled seeds | seeded-fault / clean | `results/main_results.csv`; `runs/seeded_faults/` | VERIFIED |
| Full Pantheon localizes seeded faults | 160/160 = 1.000 | 20 seeds × 8 conditions | `results/seeded_faults_predictions.csv` | VERIFIED — CONTROLLED ONLY |
| Numeric-only localization is weaker | 20/160 = 0.125 | same frozen comparisons | `results/baselines.csv` | VERIFIED — CONTROLLED ONLY |
| Pairwise artifact localization is weaker | 100/160 = 0.625 | same frozen comparisons | `results/baselines.csv` | VERIFIED — CONTROLLED ONLY |
| Canonical auditing detects correlated shared bug | 20/20; both weaker baselines call reproduced | shared_bug | `results/seeded_faults_predictions.csv` | VERIFIED — ENGINEERED CONDITION |
| Canary detects seeded hidden-state exposure | 20/20 | hidden_contamination | `results/seeded_faults_predictions.csv` | VERIFIED — ENGINEERED CONDITION |
| Two public-data implementations agree on improvement direction | 10/10 | WDBC, 10 overlapping splits | `results/public_case_paired.csv` | VERIFIED — DESCRIPTIVE |
| Mean absolute public-case effect difference | 0.006993 | WDBC, 10 overlapping splits | `results/public_case_summary.json` | VERIFIED — DESCRIPTIVE |

## Frozen CORE-Bench v1.1 OOD evidence

| Claim | Result | Scope | Evidence file | Status |
|---|---|---|---|---|
| Definitive OOD matrix completed | 17/17 tasks; 34/34 trajectories; 20 questions | protocol V4 | `configs/local_ood_protocol.json`; `runs/external/` | VERIFIED |
| Frozen run used two distinct local model identities | `qwen3:0.6b` and `llama3.2:1b` | all 17 tasks | `configs/local_ood_protocol.json`; `results/external/external_aggregate.json` | VERIFIED |
| Provider errors were absent | 0 recorded provider errors | 34 trajectories | `results/external/task_diagnostics.csv` | VERIFIED |
| Agent A answered canonical questions | 0/20 non-null; 0/20 correct | question-weighted | `results/external/question_level.csv`; `results/external/rigorous_summary.json` | NEGATIVE RESULT |
| Agent B answered canonical questions | 0/20 non-null; 0/20 correct | question-weighted | `results/external/question_level.csv`; `results/external/rigorous_summary.json` | NEGATIVE RESULT |
| Agents authored valid task reports | A 0/17; B 0/17; combined 0/34 | report-origin audit | `results/external/task_diagnostics.csv`; `runs/external/` | NEGATIVE RESULT |
| Harness preserved failed trajectories as normalized artifacts | 34/34 report files present, 0/34 agent-authored | report-origin audit | `runs/external/`; `results/external/task_diagnostics.csv` | VERIFIED |
| Substantive agreement occurred | 0/20 | neither agent answered | `results/external/question_level.csv` | OBSERVED ZERO — ALL ABSTENTIONS |
| False consensus occurred | 0/20 observed | neither agent answered | `results/external/question_level.csv`; `results/external/rigorous_summary.json` | NOT INTERPRETABLE AS CAPABLE-AGENT PREVALENCE |
| Artifact-completeness gate passed | historical `PUBLICATION-READY` | structural checks only | `PUBLICATION_READINESS.json`; `PUBLICATION_READINESS.md` | HISTORICAL / SUPERSEDED |
| Scientific capability floor passed | false | post-study reporting safeguard | `poststudy/SCIENTIFIC_READINESS.json` | FAILED |

## Unsupported or unresolved claims

| Claim | Evidence state | Status |
|---|---|---|
| Pantheon reduces false consensus among capable LLM agents | No substantive answers in the definitive OOD run | UNVERIFIED |
| Observed 0/20 false consensus generalizes beyond this run | Capability floor failed; task subset and agents are not representative | UNSUPPORTED |
| A published research package is independently replicated end to end | No successful external task answer or agent-authored report | UNVERIFIED |
| Controlled localization generalizes to natural failures | Benchmark faults are engineered | UNVERIFIED |
| Results generalize across providers or independently administered hosts | Both models used one local Ollama provider and host | UNVERIFIED |
| Results generalize to frontier agents | Only 0.6B and 1B local models were evaluated | UNVERIFIED |
| Pantheon is publication-ready for a positive cross-agent claim | Superseding capability gate failed | CONTRADICTED BY CURRENT GATE |

## Interpretation rule

Artifact presence is not equivalent to model capability. The frozen structural gate is retained for reproducibility, but [`poststudy/SCIENTIFIC_READINESS.json`](poststudy/SCIENTIFIC_READINESS.json) supersedes its `PUBLICATION-READY` label for scientific interpretation. Null/null is abstention, not agreement; harness-normalized reports are artifacts, not agent-authored scientific reports.
