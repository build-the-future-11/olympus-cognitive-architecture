# Confirmatory result audit

Disposition: **NEGATIVE**  
Scope: controlled synthetic generator only  
Protected outcomes opened: 2026-10-11, after `SCIENTIFIC_FREEZE.json`

## Primary result

Across five paired seeds, every variant reached 1.000 clean-test decision accuracy. The full-minus-uniform clean difference was therefore 0.000 with paired seed-bootstrap 95% CI [0.000, 0.000], below the registered +0.02 threshold.

On the corrupted protected split, full-model selective accuracy at 50% all-example coverage averaged 0.8703; uniform fusion averaged 0.9172. The paired difference was -0.0469 with 95% CI [-0.0953, 0.0016], failing both the directional and +0.03 practical thresholds.

The conjunctive hypothesis is rejected within this generator. Risk monotonicity and transition legality had zero violations across 75 protected evaluations.

## Secondary observations

| Variant | Clean accuracy | Corrupted accuracy | Corrupted selective@50 | OOD accuracy | Corrupted risk Brier |
|---|---:|---:|---:|---:|---:|
| Full | 1.0000 | 0.8289 | 0.8703 | 0.7227 | 0.7407 |
| Uniform | 1.0000 | 0.8664 | 0.9172 | 0.7773 | 0.7397 |
| No counterfactual | 1.0000 | 0.8273 | 0.8734 | 0.7211 | 0.7401 |
| No disagreement risk | 1.0000 | 0.8289 | 0.8703 | 0.7227 | 0.7483 |
| Mechanism off | 1.0000 | 0.8641 | 0.9141 | 0.7758 | 0.7484 |

Values are means over five independent training seeds. Seed-level artifacts and standard deviations are retained in `paper/figures/confirmatory_metrics.csv`.

No secondary ablation contrast survived the registered Holm adjustment. The full and no-disagreement-risk variants were identical in task accuracy, so disagreement-conditioned risk did not influence decisions in this experiment. Removing counterfactual consistency changed corrupted selective accuracy by only +0.0031 in the mean. The mechanism-off control tracked the uniform baseline.

## Mechanism audit

Observation → the learned router underperformed exact uniform fusion under corrupted and OOD conditions while tying on clean data.

Hypothesized cause → the router learned source preferences rather than reliable per-example reliability inference. On corrupted data it assigned mean weights of 0.281 to Atlas and 0.265 to Prometheus, but only 0.100 to Hermes. Because the corrupted regime removes one modality and perturbs another independently per example, this non-uniform preference amplified unreliable channels. Uniform fusion preserved ensemble diversity and was more robust.

Evidence → full-model corrupted gate entropy remained high (0.922; 4.42 effective components), so the failure was not hard collapse. It was a moderate but systematically harmful reweighting. The risk head also severely underestimated the registered corruption target: mean intrinsic risk was about 0.143 on a split whose target is 1.0, consistent with the high 0.741 Brier score. The 0.1 risk-loss weight was insufficient to learn the synthetic corruption indicator, but changing that weight after protected inspection would define a successor hypothesis.

Alternative explanation → the three text-derived families receive the same token channel, so the five-way gate is not equivalent to five independent modalities. The controlled generator is useful for falsification but does not establish that learned routing must fail on richer aligned data.

## Validity audit

- Data integrity: PASS; 1,408 examples, immutable hashes, zero source-ID overlap.
- Protocol timing: PASS; protocol preceded data generation, clarification preceded protected inspection, and code/data/checkpoints were frozen before confirmation.
- Execution: PASS; 25/25 training runs and 75/75 protected evaluations completed.
- Parameter matching: PASS; all controls instantiate the same parameter count and execute the router.
- Invariants: PASS; zero host-risk monotonicity or transition-legality violations.
- Statistical unit: PASS; contrasts are paired by independent training seed, not example.
- Multiplicity: PASS for registered secondary sign-flip tests with Holm adjustment.
- External validity: NOT ESTABLISHED by design.
- Independent reproduction: NOT YET PERFORMED.

## Scientific conclusion

The registered evidence rules out the claim that this learned router and regularization package improves reliability-shift robustness over uniform fusion in the tested controlled regime. The most defensible interpretation is not that fusion is useless, but that normalized learned gates without an explicit reliability target can encode brittle source preference. Any reliability-supervised router, stronger risk objective, or modality-independent redesign is a new successor study and must not rewrite this negative result.

