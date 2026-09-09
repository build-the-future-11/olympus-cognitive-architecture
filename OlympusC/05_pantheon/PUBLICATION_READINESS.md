# Frozen Structural Artifact Gate — Historical Output

> **SUPERSEDED FOR SCIENTIFIC INTERPRETATION.** This page preserves the output
> of the frozen V4 artifact-completeness gate. `PUBLICATION-READY` below means
> only that the internally prespecified artifact matrix existed and passed
> structural checks. V4 was frozen before execution but was not externally
> timestamped or registered. This is not a scientific-validity,
> model-capability, paper-quality, or venue-readiness verdict.

**Historical structural verdict: `PUBLICATION-READY`**

**Current scientific verdict: `NEGATIVE_RESULT_ARTIFACT_COMPLETE_CAPABILITY_FLOOR_FAILED`**

See [`poststudy/SCIENTIFIC_READINESS.md`](poststudy/SCIENTIFIC_READINESS.md) and [`RESEARCH_TRUTH.md`](RESEARCH_TRUTH.md). Both agents answered **0/20** questions, and neither authored a valid task report (**0/34** trajectories combined). Consequently, the completed run cannot identify false-consensus prevalence among capable agents.

## What the historical gate checked

- PASS — **local_evidence_validator**: 480 seeded-fault classifier evaluations, 200 controlled executions, and 20 public-case runs.
- PASS — **tests**: 22 tests passed at the time the frozen structural gate was generated.
- PASS — **external_tasks**: 17 tasks; threshold 15.
- PASS — **external_questions**: 20 scored questions; threshold 20.
- PASS — **agent_independence**: model identities differed on every task.
- PASS — **cross_domain**: four fields: Computer Science, Economics, Engineering, and Physics.
- PASS — **heldout_or_ood**: the frozen subset was labeled OOD; public-training evidence was excluded from the definitive run.
- PASS — **complete_reports**: numeric artifacts and `report.json` files existed, with no recorded provider errors.

## Why this gate is insufficient

The historical `complete_reports` check accepted harness-created failure-normalization files as present reports. It did not require an agent-authored valid report and did not impose a minimum non-null answer rate. Later provenance-aware analysis found:

- Agent A answered 0/20 and authored 0/17 reports.
- Agent B answered 0/20 and authored 0/17 reports.
- Jointly answered questions: 0/20.
- Agent-authored valid reports: 0/34.
- Substantive agreements: 0/20.
- Observed false-consensus events: 0/20, uninterpretable as a capable-agent prevalence estimate because both agents abstained on every question.

The immutable machine-readable historical output remains in `PUBLICATION_READINESS.json`. It should be cited only as the frozen gate result, together with the superseding post-study capability verdict.
