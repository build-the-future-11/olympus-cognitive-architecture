# Olympus Source of Truth

The working implementation is this repository root. The audit restored Git
metadata and anchored branch `audit-repair-20260906` to declared upstream
revision `2b0e3930d4403a9df830bd470c64f2d3c5028088`. Repair commits on this branch
remain **non-release candidates** until the complete CI gate passes and a human
reviews the imported research evidence. File hashes describe artifacts; they do
not substitute for the source commit named by a release tag.

`OlympusC/05_pantheon` is a separate research sidecar and the repository's only
paper-shaped empirical program. It is not a trained Olympus model. Until it is
moved to an independently versioned repository, its own validator and tests
must run as an explicit CI job rather than being silently covered by root test
claims. Other discovered copies are historical or unverified until a
maintainer reconciles them.

The canonical runtime surfaces are `olympus/`, `datasets/`, `apps/forge-web/`,
and the Foundry CLI/API. Generated evidence lives under
`artifacts/stage-execution/`; release-facing summaries live under `evidence/`.

## Family truth

| Family | State | Claim boundary |
| --- | --- | --- |
| FoundryVerificationBigram | OUTCOME_VERIFIED | Lifecycle verifier only; not a family model. |
| Hermes | EXPERIMENTAL_SMOKE_NOT_PROMOTED | An authorized-workspace grounding/output bridge and separate head optimization execute on synthetic fixtures. The historical tiny infrastructure checkpoint scored 0% exact task/tool; neither artifact is a qualifying Hermes checkpoint. |
| Prometheus | EXPERIMENTAL_SMOKE_NOT_PROMOTED | Canonical allowlisted receipts are bound to workspace/model/branch/claim during deterministic selection; proposer/verifier optimization remains separate. No qualifying checkpoint or synthesis result exists. |
| Perseus | EXPERIMENTAL_SMOKE_NOT_PROMOTED | Host-registered action approval, defensive transaction, commit-time reauthorization, and action/recovery optimization paths execute. There is no sandbox, durable recovery, committed-effect rollback, qualifying checkpoint, or stateful benchmark result. |
| Olympus-Atlas | EXPERIMENTAL_SMOKE_NOT_PROMOTED | ACL-first hybrid retrieval uses opaque keyed process-local versions and ACL-filtered index views; retriever optimization is separate. No qualifying checkpoint, durable index, corpus study, or comparative result exists. |
| Kronos | EXPERIMENTAL_SMOKE_NOT_PROMOTED | Temporal optimization and local staging checks require candidate/metric-bound allowlisted evidence identifiers, but reports are not parsed and rollback/deletion replay are not performed. No qualifying checkpoint or temporal result exists. |
| Aion | EXPERIMENTAL_SMOKE_NOT_PROMOTED | Host-registered approvals and audits bind the run, protocol, and current head; transition evidence IDs are restricted to the frozen protocol, and model authority cannot promote. Governed transitions and masked-router optimization execute in process-local code. No signed/durable controller, qualifying checkpoint, or autonomous-research result exists. |

The bounded component-run record is
`evidence/model_family_smoke_20260906.json`. It covers independent synthetic
component execution paths, not an integrated family system or capability
benchmark. Partial Hermes/Prometheus shared-workspace bridges do not change that
boundary.

These names identify source-level architecture roles only. They cannot become
external model or checkpoint identities until immutable checkpoints pass the
family-specific, hash-bound promotion constitution. Focused unit-gradient tests
establish differentiability, not trained capability.

The research priority and claim boundaries are maintained in
`research/RESEARCH_PROGRAM.md`.
