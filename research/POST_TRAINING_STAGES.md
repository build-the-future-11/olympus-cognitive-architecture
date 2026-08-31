# Hermes Post-Training Stage Control

Only supervised fine-tuning is justified and executed today. Every later stage
has a concrete admission gate; no empty stage is represented as implemented.

| Stage | Current state | Admission evidence | Exit evidence |
|---|---|---|---|
| Supervised fine-tuning | **REAL infrastructure smoke** | Licensed, deduplicated, PII-screened data; immutable splits; base identity | Reproducible checkpoint, validation curve, held-out comparison, resource record |
| Preference optimization | **BLOCKED** | At least 5,000 reviewed chosen/rejected pairs, inter-rater agreement, frozen reference policy, safety review | Held-out preference win rate, no capability regression, calibration and safety deltas |
| Rejection-sampled self-training | **BLOCKED** | A promoted generator, deterministic verifier with measured precision/recall, contamination controls | Accepted-sample audit, ablation against no-self-training baseline, held-out gains |
| Distillation | **BLOCKED** | Authorized teacher, logged teacher revision, rights to derived outputs, student/teacher evaluation contract | Student quality/latency/RAM frontier and teacher leakage review |
| Tool-use specialization | **BLOCKED** | Sandboxed deterministic tools, recorded schemas, execution traces, adversarial injection cases | Tool-selection, argument, execution, recovery, and refusal scores |
| Reasoning specialization | **BLOCKED** | Reviewed problem/answer set with verifiable outcomes and anti-memorization split | Exact outcome accuracy, calibration, length/cost, self-correction success |
| Safety tuning | **BLOCKED** | Written threat model, allowed/refused boundary set, red-team corpus and reviewers | Over-refusal, unsafe compliance, prompt injection, privacy, and regression reports |

The promotion engine in `olympus/foundry/promotion.py` is the enforcement
boundary. A later stage may add evidence, but it cannot waive dataset identity,
held-out scale, task quality, quantization, serving, license, or model-card
requirements.
