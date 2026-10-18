# Paper specification: Reliability-Supervised Shrinkage Fusion after a Controlled Negative Result

This is the pre-results paper contract. Results language remains blank until protected evidence is
locked.

## Intended paper

The paper asks whether explicit reliability identification repairs a learned multimodal router
that failed against uniform fusion under corruption. It reports the v2 negative finding as the
motivation, not as evidence for v3. The contribution is an auditable mechanism and a disciplined
successor-study design, not a claim of general multimodal or cognitive capability.

## Required sections

1. Abstract: problem, negative predecessor, v3 mechanism, protocol, and eventual honest disposition.
2. Introduction: robust fusion problem, why semantic gates can encode source preference, falsifier.
3. Related work: dynamic fusion, missing modalities, collapse, calibration, uncertainty, shrinkage.
4. Problem formulation: five states, reliability variables, protected boundary, objectives.
5. Method: reliability heads, log-space product, uniform shrinkage, recomputed decision/risk path.
6. Experimental design: fresh splits, matched variants, fixed seeds, primary endpoint, bootstrap.
7. Results: generated only from `EVIDENCE_SUMMARY.json` and retained per-run JSON files.
8. Ablations: supervision-off, semantic, uniform, reliability-only.
9. Failure analysis: routing collapse, calibration failure, risk ordering, OOD limitations.
10. Limitations and ethics: synthetic scope, no deployment or safety efficacy, compute constraints.
11. Reproducibility: hashes, commands, environment, frozen configs, failure receipts.

## Figures and tables

- Architecture diagram with semantic and reliability paths kept visually distinct.
- Gate lower-bound diagram showing fixed shrinkage geometry on the simplex.
- Primary paired-seed plot with all five points and 95% bootstrap interval.
- Reliability calibration plot for clean, perturbed, and missing states.
- Full variant-by-regime matrix with uncertainty and invariant counts.
- Runtime/parameter table proving matched controls.

No number may enter the paper without a raw-artifact path and an analysis transform. No result
paragraph is to be written before the protected matrix is complete and the evidence audit passes.

