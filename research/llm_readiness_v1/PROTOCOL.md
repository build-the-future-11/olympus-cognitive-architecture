# Six-family large-scale readiness protocol v1

## Claim boundary

This protocol decides whether one exact candidate from each BU1LD family is
ready for large-scale use. Reference modules, synthetic studies, green CI,
plans, and unbound benchmark summaries cannot satisfy any gate. A missing or
hash-mismatched receipt fails closed. Passing this protocol does not itself
authorize deployment.

## Frozen inputs and decision

Thresholds are frozen in `configs/llm-readiness-v1.json`. Evidence conforms to
`ReadinessEvidenceSet` in `olympus/foundry/readiness.py`. The evaluator always
emits all eleven gates for all six canonical families. A family is ready only
when every gate passes; the portfolio is ready only when all six families pass.

1. **Data receipt integrity:** the manifest exists and its SHA-256 matches.
2. **Data quality:** approved rights and provenance, exact and normalized
   deduplication, cross-split contamination scanning, PII/secret scanning,
   frozen splits, and capability coverage all pass.
3. **Data scale:** at least 10,000 reviewed training records, 1,200 protected
   records, and 100 protected records per category.
4. **Three-seed training:** at least three unique seeds use one exact dataset,
   base model, and training configuration. Every run completes with finite
   metrics and no protected access before the scientific freeze.
5. **Strong controls:** base model, matched alternative, retrieval ablation,
   tool ablation, and routing ablation each have an immutable receipt. The
   confidence-interval lower bound of the primary improvement is at least 0.02
   with no material regression.
6. **Capability evaluation:** reasoning, instruction following, tool
   completion, structured output, long context, domain-held-out behavior,
   distribution shift/adversarial behavior, and calibration/abstention meet
   their frozen floors without critical failures.
7. **Safety evaluation:** prompt injection, data exfiltration, harmful-output
   policy, tool permissions, and tenant/memory isolation pass with zero
   critical failures.
8. **Production qualification:** P50/P95/P99 latency, throughput, error rate,
   cost, a 24-hour soak, failure injection, queue/cancel/retry/backpressure,
   rollback, zero-downtime upgrades, and drift/failure/cost monitoring pass.
9. **Independent reproduction:** a different operator in a clean environment
   reproduces the exact source, dataset, checkpoint, and metrics.
10. **Family-specific gates:** every gate in the canonical target specification
    for that family passes.
11. **Release documentation:** exact model card, evaluation report, load/cost
    report, and safety/rollback report hashes are present.

## Execution order

1. Curate and audit a rights-cleared corpus. Freeze its manifest before any
   candidate training.
2. Freeze base-model and training-config hashes. Run three seeds in isolated
   cloud jobs and retain complete receipts and checkpoints.
3. Freeze the protected evaluation manifest before opening it. Run controls,
   then capability and safety suites once per candidate.
4. Admit only candidates that clear scientific gates to an isolated staging
   service. Run the production qualification and rollback exercise.
5. Give an independent operator only the protocol and immutable identities.
6. Add the four release documents and run `olympus foundry readiness-check`.

The current evidence file deliberately contains no candidate receipts. Its
generated report is the honest baseline and lists every unresolved gate.
