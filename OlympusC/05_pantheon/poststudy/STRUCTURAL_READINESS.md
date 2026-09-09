# Structural Readiness Gate

**Verdict: ARTIFACT_COMPLETE**

> This gate checks artifact completeness only. It cannot emit a publication-ready verdict, and it does not establish model capability or scientific interpretability.

- PASS — **local_evidence_validator**: {
  "evidence_validation": "PASS",
  "seeded_fault_classifier_evaluations": 480,
  "actual_seeded_fault_executions": 200,
  "public_case_runs": 20
}
- PASS — **tests**: 31 passed, 2 skipped
- PASS — **external_tasks**: 17 tasks; threshold 15
- PASS — **external_questions**: 20 scored questions; threshold 20
- PASS — **agent_identity_difference**: agent identities differ on every task
- PASS — **cross_domain**: 4 fields: ['Computer Science', 'Economics', 'Engineering', 'Physics']
- PASS — **heldout_or_ood**: visibility=['ood']; public_train is pilot evidence only
- PASS — **normalized_artifacts_present**: numeric=True; normalized_reports=True; provider_errors_absent=True; this does not establish agent authorship

Structural completeness is necessary but insufficient. Run poststudy/scientific_readiness_gate.py for the authoritative scientific verdict.
