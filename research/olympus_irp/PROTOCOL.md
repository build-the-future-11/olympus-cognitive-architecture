# Olympus individual research protocol

## Status and boundary

This protocol was frozen before the expanded backtest matrix was run. It is a
development and engineering protocol. It is not a confirmatory test of the central
Olympus fusion hypothesis. The only available corpus contains 36 proprietary smoke
instruction records, and its 12-record test split has already been observed.

The central hypothesis remains falsifiable but not currently executable: on a new,
rights-reviewed, source-disjoint aligned corpus, risk-monotone counterfactual fusion
must improve the preregistered task and selective-risk endpoints over the strongest
matched baseline without exceeding the safe-case STOP budget. Failure to obtain a
positive lower confidence bound, or worse robustness at matched clean quality and
compute, falsifies the corresponding claim within the tested scope.

## Bounded engineering question

On the fixed smoke corpus, do cosine learning-rate decay and validation-best
restoration improve loss stability over matched constant-rate and last-epoch controls?
How do parameter scale and sequence packing change validation/test loss, latency, and
format compliance?

The answer can validate or reject engineering choices for this implementation. It
cannot validate multimodal fusion, safety efficacy, general capability, or JEPA quality.

## Frozen matrix

All runs use full-parameter CPU training, byte tokenization, batch size 4, gradient
accumulation 2, learning rate 0.001, weight decay 0.01, gradient clipping at 1.0, 30
attempted epochs, deterministic seeds, and no early stopping. The paired seed set is
20261101 through 20261105.

| Variant | Width/layers/heads | Packing | Schedule | Restore best | Seeds |
|---|---:|---|---|---|---:|
| cosine_large | 128/4/8 | yes | 10-step warmup, cosine to 0.05 | yes | 5 |
| constant_large | 128/4/8 | yes | constant | yes | 5 |
| no_restore_large | 128/4/8 | yes | 10-step warmup, cosine to 0.05 | no | 5 |
| no_pack_large | 128/4/8 | no | 10-step warmup, cosine to 0.05 | yes | 3 |
| cosine_medium | 64/2/4 | yes | 10-step warmup, cosine to 0.05 | yes | 3 |
| cosine_small | 32/1/4 | yes | 10-step warmup, cosine to 0.05 | yes | 3 |

The first three variants use identical seeds and are paired comparisons. Scale and
packing analyses are exploratory and not compute matched.

## Endpoints and rules

Primary engineering endpoint: final validation loss restored according to the frozen
variant definition. Secondary endpoints: observed smoke-test loss, format compliance,
exact match, runtime, peak resident memory, parameters, training trajectory, and failed
runs. Test-set endpoints are explicitly contaminated development measurements.

Report every seed, arithmetic mean, sample standard deviation, deterministic bootstrap
95% confidence interval, median, and matched paired differences where applicable. No
multiplicity-adjusted confirmatory p-values will be calculated because the study is not
confirmatory and the sample size is too small for credible asymptotic inference.

Any exception, non-finite value, resource-governor rejection, or missing artifact is a
retained failure. The matrix is not revised after results. No score-driven rerun is
authorized.

## Adversarial flaw register

| Flaw | Cause | Severity | Repair | Required evidence |
|---|---|---:|---|---|
| Novelty collision | Fusion, MoE routing, counterfactuals, and risk control each have prior art | High | Claim only the combined contract after updated search | Independent literature audit and exact mechanism comparison |
| No aligned corpus | Current data are instruction-only smoke records | Critical | Acquire rights-reviewed aligned modalities and governance labels | Provenance, licenses, source hashes, and frozen splits |
| Test contamination | Smoke test was inspected during development | Critical | Treat it as exploratory and create a new protected test | Access log and never-observed split hash |
| Tiny independent-unit count | Twelve records per split | Critical | Power analysis and many source-independent units | Precision target and achieved confidence interval |
| Confounded fusion gains | More parameters, compute, or information | High | Parameter-, compute-, and information-matched controls | Activated FLOPs, parameter counts, wall time, and paired results |
| Routing non-identifiability | Similar components can substitute for each other | High | Intervention and component corruption studies | Stable counterfactual and causal intervention effects |
| Risk-label ambiguity | Intrinsic risk is task- and policy-dependent | High | Define labels and calibrate on a separate split | Annotator protocol, agreement, calibration, and coverage curves |
| Counterfactual inconsistency | Leave-one-out heads do not prove causal necessity | Medium | Call them diagnostic and add rerun/ablation interventions | Agreement between analytic and executed ablations |
| Optimization instability | Joint objectives and routing can conflict | High | Per-loss gradients, multiple seeds, and stability diagnostics | Gradient conflict, collapse rate, and failure logs |
| JEPA objective over-complexity | Added losses may duplicate simpler reconstruction priors | High | Term-by-term and compute-matched ablations | Source-disjoint PSNR, SSIM, spectral, edge, runtime, and memory results |
| Governance overclaim | Architectural monotonicity is not safety efficacy | Critical | Limit claim to the invariant | Formal proof, tests, and downstream hazard evaluation |
| Scaling uncertainty | Reference sizes do not predict target-scale behavior | High | Predefined scaling study | Data/model/compute curves with confidence intervals |

## Terminal rule

Negative, mixed, inconclusive, invalid, and blocked outcomes are legitimate terminal
states. This protocol does not continue until a metric becomes positive. A successor
hypothesis requires a new version, falsifier, protocol, and untouched evaluation set.
