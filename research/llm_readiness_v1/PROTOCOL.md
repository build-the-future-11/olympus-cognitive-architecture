# Six-family large-scale readiness protocol v1

## Claim boundary

This protocol decides whether one exact candidate from each BU1LD family is
ready for large-scale use. Reference modules, synthetic studies, green CI,
plans, and unbound benchmark summaries cannot satisfy any gate. A missing or
hash-mismatched receipt fails closed. Passing this protocol does not itself
authorize deployment.

## Frozen inputs and decision

Thresholds are frozen in `configs/llm-readiness-v1.json` and the evaluator
rejects any byte-level change to that v1 policy hash. Evidence conforms to the
machine-readable schema in
`configs/llm-readiness-evidence-v1.schema.json`; every bundle must explicitly
declare `olympus-six-family-readiness-v1`. Family IDs and candidate IDs must
each be unique. The evaluator accepts only
regular, non-symlink inputs and artifact paths contained by the evidence
bundle. Training, control, capability, safety, production, reproduction, and
family-gate receipts must be JSON envelopes that identify their evidence kind,
candidate, checkpoint, dataset, producer, and source commit. A receipt whose
bytes match its advertised hash still fails when any of those bindings are
wrong. Every receipt also binds the canonical SHA-256 of the complete claim it
supports, excluding only the receipt pointer itself, plus the exact seed,
control, suite, production, reproduction, or family-gate identity. Changing a
metric, count, pass flag, case group, workload detail, or result invalidates the
receipt. Parent-directory symlinks are rejected, not only final-path symlinks.
Rights approval and pre-training dataset freeze are separate hash-bound
attestations tied to the exact manifest; manifest metadata or a boolean in the
evidence bundle cannot establish either gate by itself.

Capability, safety, and production policies are frozen independently in
`configs/llm-readiness-capability-v1.json`,
`configs/llm-readiness-safety-v1.json`, and
`configs/llm-readiness-production-v1.json`. The evaluator rejects any byte
change, validates their canonical suite and workload sets, and requires their
duplicated numerical floors to agree with the threshold file. Release documents
are bound to the candidate and checkpoint by their verified contents. Each
release document starts with a structured
`olympus-readiness` header binding its document type, protocol, candidate,
source commit, checkpoint, and dataset. The four-document set must be exact.
Each document must contain its prescribed substantive section headings.

Every report records the protocol ID plus SHA-256 identities for the frozen
thresholds, submitted evidence, evidence schema, report schema, all three
frozen suite/workload policies, exact evaluator source, and the complete
decision-code bundle comprising the evaluator, dataset verifier, and canonical
family specifications. Report models reject inconsistent ready counts,
blockers, family order, gate sets, or aggregate decisions.
The evaluator always emits all eleven gates for all six canonical families. A
family is ready only when every gate passes; the portfolio is ready only when
all six families pass. CI regenerates the schema and current report in tests
and requires byte-for-byte equality with the checked-in baseline.

1. **Data receipt integrity:** the manifest exists and its SHA-256 matches.
2. **Data quality:** approved rights and provenance, exact and normalized
   deduplication, cross-split contamination scanning, PII/secret scanning,
   frozen splits, and capability coverage all pass.
3. **Data scale:** at least 10,000 reviewed training records, 1,200 protected
   records, and 100 protected records per category.
4. **Three-seed training:** at least three unique seeds use one exact dataset,
   base model, and training configuration. Every run completes with finite
   metrics and no protected access before the scientific freeze. Duplicate
   seed entries are rejected.
5. **Strong controls:** base model, matched alternative, retrieval ablation,
   tool ablation, and routing ablation each have an immutable receipt. The
   confidence-interval lower bound of the primary improvement is at least 0.02
   with no material regression. Each control uses at least 200 examples, 10,000
   paired bootstrap resamples, 95% confidence, Holm correction, and a worst
   category delta no lower than -0.02. The control set must exactly match the
   frozen list; undeclared extras fail the gate.
6. **Capability evaluation:** reasoning, instruction following, tool
   completion, structured output, long context, domain-held-out behavior,
   distribution shift/adversarial behavior, and calibration/abstention meet
   their suite-specific frozen metrics, example counts, and floors without
   critical failures. The suite set must exactly match the frozen list.
7. **Safety evaluation:** prompt injection, data exfiltration, harmful-output
   policy, tool permissions, and tenant/memory isolation pass their frozen
   minimum case counts and exact case-group contracts with zero critical
   failures. The suite set must exactly match the frozen list.
8. **Production qualification:** P50/P95/P99 latency, throughput, error rate,
   cost, a 24-hour soak, failure injection, queue/cancel/retry/backpressure,
   rollback, zero-downtime upgrades, and drift/failure/cost monitoring pass.
   The receipt must cover at least 100,000 requests, the exact traffic phases,
   request classes, and failure injections, with zero process crashes and zero
   lost accepted requests.
9. **Independent reproduction:** a different operator in a clean environment
   reproduces the exact source, dataset, checkpoint, and metrics.
10. **Family-specific gates:** every gate in the canonical target specification
    for that family has a hash-bound receipt tied to the selected dataset and
    checkpoint, and passes. Missing or undeclared gates fail.
11. **Release documentation:** exact model card, evaluation report, load/cost
    report, and safety/rollback report hashes are present, candidate-bound, and
    contain every document-type-specific required section.

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
6. Add the four release documents and run `olympus foundry readiness-check
   --require-ready`. This mode writes the full report and exits with status 1
   unless all six families pass.

The current evidence file contains only the verified Hermes smoke-dataset
receipt; it contains no trained family candidate. Its generated report is the
honest baseline and lists every unresolved gate.
