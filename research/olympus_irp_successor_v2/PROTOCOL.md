# Olympus Controlled Fusion Study v2 — Frozen Protocol

Status: **FROZEN BEFORE DATA GENERATION OR OUTCOME INSPECTION**  
Freeze date: 2026-10-11  
Scientific role: controlled synthetic mechanism test; not external-validity evidence

## Registered question and hypothesis

Existing heterogeneous fusion systems can fail when the reliability of text, vision, and temporal inputs changes per example. We test whether a learned normalized router, trained with leave-one-component-out consistency and disagreement-conditioned intrinsic risk, improves source-disjoint decision quality and failure ranking over an otherwise identical uniform-fusion control under controlled missingness and corruption.

The confirmatory hypothesis is directional: the full mechanism must improve mean clean-test decision accuracy by at least 0.02 over the parameter-matched uniform-fusion control and improve corrupted-test selective accuracy at 50% coverage by at least 0.03. Both endpoints must have a paired seed-bootstrap 95% confidence interval whose lower bound is greater than zero. Host-risk monotonicity and transition legality must have zero violations in every run.

The hypothesis is falsified if either practical threshold is missed, either interval includes zero, any invariant violation occurs, or the mechanism-off control performs equivalently within the registered thresholds.

## Scope and boundaries

This study tests implementation-level causal behavior in a known synthetic generator. It does not establish performance on natural images, human language, real workflows, deployment safety, target-scale capability, or JEPA super-resolution. A negative result is retained as a valid negative result. A positive result can justify—but cannot replace—a rights-reviewed real-data study.

No protected outcome may be used to change the generator, architecture, optimizer, hyperparameters, metrics, exclusions, aggregation, or prose claim. A defect that invalidates the run must be recorded, corrected, re-frozen, and rerun only for affected work.

## Independent units and data generation

The generator creates source-disjoint aligned examples from independent 64-bit source seeds. Each example has a four-class latent decision, token sequence, RGB image, temporal state/action trajectory, legal-transition mask, host risk, intrinsic-risk target, transition target, and next-state target. Text, image, and temporal channels carry complementary noisy evidence. Missingness and corruption are sampled independently of the class conditional on the registered regime.

Splits are generated once with disjoint source identities:

- train: 512 examples, seed 20261201, mixed clean and single-channel corruption;
- development: 128 examples, seed 20261202, mixed clean and single-channel corruption;
- protected clean test: 256 examples, seed 20261203, no channel removed;
- protected corrupted test: 256 examples, seed 20261204, one channel removed and one additional channel perturbed;
- protected OOD test: 256 examples, seed 20261205, higher noise and class-conditional rendering permutations not used in training.

The generated bundles and manifests are immutable and SHA-256 bound. Source IDs may not overlap between any split. Protected bundle metrics are not inspected until the scientific freeze receipt exists.

## Registered model and controls

All variants instantiate the same six-family reference system, dimensions, and parameters. The gate router is executed in all variants so the primary control remains parameter- and forward-compute-matched.

1. `full`: learned normalized gates; counterfactual consistency weight 0.05; disagreement-conditioned intrinsic risk.
2. `uniform`: router is executed but its output is replaced by exact uniform gates; all other components and losses are unchanged.
3. `no_counterfactual`: learned gates; counterfactual consistency weight 0; disagreement-conditioned risk.
4. `no_disagreement_risk`: learned gates; counterfactual consistency weight 0.05; the disagreement coordinate is replaced by zero before the parameter-matched risk network.
5. `mechanism_off`: uniform gates, counterfactual consistency weight 0, and zeroed disagreement coordinate.

The primary contrast is `full - uniform`. The remaining variants are registered explanatory ablations, not alternative primary hypotheses.

## Optimization and repetitions

- seeds: 20261301, 20261302, 20261303, 20261304, 20261305;
- device: CPU;
- deterministic algorithms: enabled with warning-only fallback recorded;
- steps: 300;
- batch size: 32;
- AdamW learning rate: 0.0005;
- minimum learning-rate ratio: 0.1;
- warmup steps: 20;
- weight decay: 0.01;
- gradient clipping: 1.0;
- no early stopping and no protected-outcome selection;
- one training workload at a time;
- the repository ResourceGovernor remains enabled with its 2 GiB available-memory floor and 90% swap ceiling.

The bounded matrix contains 25 training runs (5 variants × 5 seeds), followed by frozen-checkpoint evaluation on the three protected test regimes. Resource rejection is retained and does not authorize weakening the guard.

## Metrics and statistics

Primary endpoints:

- decision accuracy on decision-available clean-test examples;
- risk-ranked selective decision accuracy at 50% coverage on corrupted test.

For the second endpoint, coverage is defined over all examples: sort every example by effective risk, retain the lowest-risk `ceil(0.5 N)` examples, and score their raw decision argmax. Forced-stop examples are not removed before ranking. This clarification was recorded during development, before the scientific freeze and before any protected metric was opened, because the generic evaluator's diagnostic curve conditions on decision availability.

Secondary endpoints:

- transition accuracy;
- next-state MSE;
- intrinsic-risk Brier score and ECE;
- accuracy at 25%, 50%, 75%, and 100% risk coverage;
- counterfactual flip rate and KL influence per component;
- gate entropy and effective component count;
- runtime, peak RSS, and parameter count;
- the same metrics on OOD data.

Contrasts are paired by seed. Uncertainty uses 20,000 deterministic paired bootstrap resamples over the five seed-level differences with seed 20261401. No example-level pseudo-replication is permitted. Holm correction is applied to secondary ablation contrasts within each metric family; primary endpoints are conjunctive and require both registered criteria.

## Failure handling and decision rule

A run is invalid only for a documented implementation, data-integrity, leakage, metric, or protocol defect. Non-finite loss, resource rejection, or optimizer failure is retained as an observed failure. No failed seed is silently excluded.

Disposition:

- `SUPPORTED`: both primary thresholds and confidence rules pass, with zero invariant violations;
- `MIXED`: some registered evidence is positive but the conjunctive rule fails;
- `NEGATIVE`: valid evidence rejects the practical claim;
- `INCONCLUSIVE`: valid execution lacks precision because registered runs cannot complete;
- `INVALID`: a validity defect prevents interpretation.

## Strongest falsification test

The corrupted protected split deliberately shifts reliability twice per example. If the learned router and risk signal encode the proposed mechanism, the full model should preserve useful decisions and rank failures better than uniform fusion. Failure on this split, especially with comparable clean accuracy, directly falsifies the claimed reliability-adaptation mechanism within this controlled setting.

