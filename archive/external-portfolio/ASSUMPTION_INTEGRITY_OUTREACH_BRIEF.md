# Assumption Integrity Under Distribution Shift

Status: anonymous collaboration brief for ICLR 2027 preparation  
Evidence date: 2026-09-02

## Question

When inference-time regime or context information is available, when does conditioning on it improve prediction under shift, and how dangerous is semantically wrong context?

## Supported result

Valid context can substantially improve OOD prediction in controlled settings, but misspecified context can create a severe high-confidence failure surface. A simple concatenation baseline matches the initially proposed specialized conditioning architecture, falsifying the stronger architecture claim. On the natural UCI Adult benchmark, the unconditioned model is best against every tested conditioned method on all five frozen seeds, and that ordering reproduces in a separate local environment.

## Evidence package

- 167 registered runs: 164 research runs and 3 smoke runs;
- 30-run main matrix and independent 30-run frozen-seed replication;
- 20 integrity runs plus explicitly separated exploratory pilots;
- 40 real-measurement calibration runs;
- 20 frozen Adult runs and a cross-environment reproduction;
- deep ensembles, MC dropout, temperature scaling and split conformal baselines;
- exact 5,000-observation non-identifiability witness;
- loadable checkpoints for every headline and confirmatory neural run;
- 20 passing tests, claim preflight and artifact validator.

## Requested collaboration

We seek one of:

1. an independent reproduction of the assumption-validity intervention;
2. a stronger baseline or natural-context dataset that could falsify the present conclusion;
3. technical feedback on the identifiability boundary and claim language.

## Explicit non-claims

- The specialized ACP architecture is not established as superior to concatenation.
- Context conditioning is not generally beneficial; the Adult result points the other way.
- The validity of claimed context is not identifiable without additional observables or assumptions.

## Reproduction entry points

- Repository: https://github.com/THE-BU1LD/Assumption-Integrity
- Paper PDF: `Assumption-Integrity/output/pdf/assumption_integrity_iclr2027.pdf`
- Artifact gate: `python3 scripts/validate_artifacts.py`
- Claim gate: `python3 scripts/iclr_claim_preflight.py`

Sender identity and public artifact-release status must be confirmed before external distribution.
