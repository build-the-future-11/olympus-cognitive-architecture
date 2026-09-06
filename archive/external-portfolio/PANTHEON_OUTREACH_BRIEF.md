# Pantheon: Canonical Artifact Auditing

Status: collaboration brief; controlled evidence only  
Evidence date: 2026-09-02

## Problem

Two research agents can agree with each other and still share the same implementation drift. Pantheon evaluates each run against a frozen canonical scientific package so agreement cannot hide correlated faults.

## Supported result

On a controlled 20-seed × 8-condition injected-fault benchmark, the full canonical audit localized 160/160 seeded fault comparisons. Proposer-versus-replicator artifact comparison localized 100/160, while numeric-only comparison localized 20/160. A separate public-data cross-implementation case study produced direction agreement on 10/10 prespecified splits.

## What exists

- frozen claim-package and manifest format;
- isolated proposer/replicator execution;
- artifact, environment and provenance comparison;
- injected-fault benchmark with retained outputs;
- public-data cross-implementation case study;
- truth ledger and explicit unsupported-claim boundary.

## Requested collaboration

We seek a partner with an existing research-agent or evaluation workflow to test whether canonical package comparison reduces false agreement under naturally occurring drift. The preferred study would freeze one real claim package, execute independent agent replications, introduce no undisclosed interventions, and score localization plus false alarms before inspecting outcomes.

## Explicit non-claims

- No cross-LLM or autonomous-research-agent benefit has yet been demonstrated.
- Controlled fault localization is not evidence of production reliability.
- The 10/10 direction agreement case study is small and is not a broad replication benchmark.

## Public alignment

The methodology is directly relevant to foundation-model evaluation, ML systems artifact review, AI measurement and reproducibility standards.

Sender identity, affiliation, public repository URL and release status must be confirmed before external distribution.
