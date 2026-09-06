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
| Hermes | SMOKE_TESTED, NOT_PROMOTED | Tiny infrastructure checkpoint; 0% exact task/tool score. |
| Prometheus | SPECIFIED | Roadmap only; no qualifying checkpoint. |
| Perseus | SPECIFIED | Roadmap only; no qualifying checkpoint. |
| Atlas | SPECIFIED | Roadmap only; no qualifying checkpoint. |
| Kronos | SPECIFIED | Roadmap only; no qualifying checkpoint. |
| Aion | SPECIFIED | Governed autonomous-research roadmap only. |

These names cannot become model identities until immutable checkpoints pass the
family-specific, hash-bound promotion constitution.

The research priority and claim boundaries are maintained in
`research/RESEARCH_PROGRAM.md`.
