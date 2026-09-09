# Stage 02 — Reference Architecture and Promotion Constitution

**State: IMPLEMENTED.** Dataset, experiment, checkpoint, evaluation, model, and
append-only evidence identities are hash-bound. Candidate promotion is
fail-closed across license, scale, loss/regression, task/tool quality, serving,
and model-card gates. A further hard-coded independent-evidence authority gate
now blocks every release manifest because evaluation, quantization, license,
and serving inputs are still caller-authored; it must later be replaced by real
attestation verification. The current report is
`artifacts/stage-execution/deep/promotion.json`.

Observed result: Hermes is `NOT_PROMOTED`; no release manifest was emitted.
Next action: satisfy existing gates without moving their thresholds.
