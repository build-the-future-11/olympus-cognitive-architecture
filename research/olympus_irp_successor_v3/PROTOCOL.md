# Olympus v3: Reliability-Supervised Shrinkage Fusion

Status: preregistered successor specification; protected outcomes unobserved.

## 1. Research statement

Existing learned fusion in Olympus v2 struggled under missing and corrupted modalities because
its gates encoded global source preference without an identifiable per-example reliability
signal. We propose reliability-supervised shrinkage routing because directly predicting each
component's observable corruption state, then conservatively shrinking adaptive weights toward
uniform, should improve corrupted-regime selective decision accuracy without violating host-risk
monotonicity or transition legality.

This is a successor study. The v2 negative result is immutable and was used to formulate v3;
therefore no v2 protected source, example, metric, or outcome is a v3 confirmatory observation.

## 2. Scientific objective and falsifier

The primary objective is to determine whether explicit per-example reliability identification,
rather than unconstrained semantic gating, explains robust fusion gains in a controlled setting.

Primary hypothesis: across five paired training seeds, v3 `full` exceeds the parameter- and
forward-compute-matched `semantic` control on corrupted-test selective accuracy at 50% coverage
by at least 0.02, with a paired bootstrap 95% interval strictly above zero.

Central falsifier: the hypothesis is rejected if the interval includes zero, or if the mean gain
is below 0.02. Any risk-monotonicity or transition-legality violation invalidates the affected run.

## 3. Prior-art and novelty boundary

Dynamic fusion and missing-modality robustness are established areas. Provable Dynamic Fusion
derives dynamically weighted fusion with guarantees; SimMLM uses a dynamic mixture of modality
experts and a More-vs-Fewer ranking loss; recent work studies multimodal representation collapse,
confidence-guided fusion, calibrated missing-modality learning, heterogeneous aligned fusion,
and prototype-anchored robust fusion. See:

- Zhang et al., *Provable Dynamic Fusion for Low-Quality Multimodal Data*, ICML 2023,
  <https://proceedings.mlr.press/v202/zhang23ar.html>.
- Li et al., *SimMLM*, ICCV 2025,
  <https://openaccess.thecvf.com/content/ICCV2025/html/Li_SimMLM_A_Simple_Framework_for_Multi-modal_Learning_with_Missing_Modality_ICCV_2025_paper.html>.
- Chaudhuri et al., *A Closer Look at Multimodal Representation Collapse*, ICML 2025,
  <https://proceedings.mlr.press/v267/chaudhuri25a.html>.
- Jorf et al., *MedPatch*, MLHC 2025,
  <https://proceedings.mlr.press/v298/jorf25a.html>.
- Ma et al., *Are Multimodal Transformers Robust to Missing Modality?*, CVPR 2022,
  <https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Are_Multimodal_Transformers_Robust_to_Missing_Modality_CVPR_2022_paper.html>.

No claim is made that reliability prediction, confidence weighting, mixture routing, or shrinkage
alone is new. The testable contribution is narrower: an auditable Olympus-specific mechanism
that combines explicit component-level reliability supervision with a fixed uniform-shrinkage
constraint, evaluated against matched controls after a recorded negative predecessor.

## 4. Model contract

Let the five family states from Hermes, Prometheus, Perseus, Atlas, and Kronos be
`z_i in R^d`, and let the frozen-v2 semantic router produce simplex weights `s_i`.
Each family has a reliability head:

`a_i = h_i(z_i)`, `q_i = sigmoid(a_i)`.

For full routing, the adaptive distribution is

`p_i = softmax_i(log(max(s_i, eps)) + beta log(max(q_i, eps)))`.

For fixed adaptive mass `lambda`, final gates are

`g_i = (1 - lambda) / m + lambda p_i`,

where `m=5`, `beta=1`, `lambda=0.75`, and `eps=1e-6`. Consequently every gate is at least
`(1-lambda)/m = 0.05`. The reliability target is 1.0 for a clean supporting modality, 0.25 for
an explicitly perturbed modality, and 0.0 for an explicitly missing modality. The text target is
shared by Hermes, Prometheus, and Perseus; image maps to Atlas; temporal state maps to Kronos.

The training objective is

`L_v3 = L_v2 + alpha BCEWithLogits(a, r)`,

with `alpha=0.25`. `L_v2` retains decision, transition, temporal, uncertainty calibration, risk
calibration, gate balance, and counterfactual-consistency terms. Reliability targets are never
forward inputs and therefore cannot directly leak protected labels at inference.

The fused state and all dependent risk, Aion, decision, uncertainty, and leave-one-component-out
counterfactual outputs are recomputed from `g`. Host risk remains combined monotonically as
`1 - (1-host_risk)(1-intrinsic_risk)`. Aion retains the host-supplied legal-transition mask and
forced-stop authority.

## 5. Data and protected boundary

Dataset: deterministic synthetic `olympus-reliability-fusion` version 3.0.0, CC-BY-4.0.

| Split | Examples | Seed | Use |
|---|---:|---:|---|
| train | 512 | 20261501 | optimization |
| development | 128 | 20261502 | inspectable development |
| clean_test | 256 | 20261503 | protected confirmation |
| corrupted_test | 256 | 20261504 | protected primary endpoint |
| ood_test | 256 | 20261505 | protected generalization |

Source identifiers are content-derived, split-specific, and must have zero overlap. The protected
bundles may be materialized before freeze but may not be evaluated, summarized, or used for design
until development is complete and the scientific-freeze receipt exists.

The dataset is intentionally synthetic. It licenses a mechanism conclusion only, not natural-data,
JEPA, target-scale, deployment, cognitive, or safety efficacy claims.

## 6. Matched experimental matrix

Every variant instantiates and executes the base Olympus model and all reliability heads.

| Variant | Routing | Reliability loss | Role |
|---|---|---:|---|
| full | semantic + reliability + shrinkage | 0.25 | proposed mechanism |
| no_reliability_supervision | semantic + reliability + shrinkage | 0 | mechanism-off supervision |
| semantic | v2 semantic gates | 0 | primary matched control |
| uniform | fixed uniform gates | 0 | simple matched control |
| reliability_only | reliability + shrinkage | 0.25 | semantic-mechanism ablation |

Each variant uses seeds 20261601–20261605, 300 steps, batch size 32, AdamW, learning rate
`5e-4`, 20 warm-up steps, cosine decay to 10%, weight decay 0.01, and gradient clipping at 1.0.
The matrix contains 25 development fits and 75 protected evaluations.

Primary endpoint: corrupted-test selective accuracy at 50% coverage, ranked by effective risk.

Secondary endpoints: full decision accuracy, clean and OOD results, reliability BCE/MAE/Brier,
clean-minus-damaged reliability separation, risk violations, transition violations, runtime,
parameters, and gradient stability. Secondary findings cannot rescue a failed primary hypothesis.

Inference: paired bootstrap over the five common seeds, 10,000 resamples, fixed bootstrap seed
20261701, two-sided percentile 95% interval. No protected-result-driven tuning is allowed.

## 7. Adversarial flaw audit

| Flaw | Cause | Severity | Mitigation | Required evidence |
|---|---|---:|---|---|
| Reliability labels may make the synthetic task too easy | corruption is generated and observable | high | narrow claim; fresh OOD split | protected reliability separation and task result |
| More supervision may explain gains | full sees auxiliary targets | high | reliability-only and supervision-off controls | paired ablations |
| Extra parameters/compute may explain gains | added reliability heads | critical | heads exist and execute in every control | exact parameter counts and source inspection |
| Uniform shrinkage may be the true mechanism | lower-bounded gates regularize routing | high | semantic, uniform, reliability-only controls | matched comparisons |
| Selective accuracy can improve from risk ordering alone | endpoint retains only low-risk examples | high | also report full accuracy and risk calibration | complete secondary matrix |
| Five seeds may be imprecise | bounded compute budget | medium | paired analysis and honest inconclusive result | interval width |
| OOD construction may not represent real shifts | synthetic permutation/noise | critical | no external-validity claim | explicit scope statement |
| Shared text target conflates three families | common input modality | medium | component order is fixed and disclosed | per-family diagnostics |
| Protected outcomes could leak through design | v3 follows v2 results | critical | fresh seeds and freeze-before-confirmation | hashes and freeze receipt |
| Simpler method may win | v2 uniform was stronger | critical | uniform is a first-class control | protected uniform comparison |

## 8. Resource, failure, and stopping rules

The resource preflight requires at least 2 GiB available memory, swap utilization at or below 90%,
and an exclusive workload lock. These guards must not be weakened to force execution. A preflight
failure creates a retained block receipt and ends the campaign stage without touching protected
outcomes.

No failed, negative, invalid, or blocked receipt may be deleted. Development stops after the fixed
25 fits. Confirmation starts only after all development reports exist and source, dataset,
contract, and checkpoint hashes are frozen. An implementation or metric defect makes affected
evidence INVALID; fix under a new corrected freeze and rerun only scientifically affected work.

## 9. Disposition rules

- SUPPORTED: primary CI lower bound > 0 and mean delta >= 0.02, with zero safety invariants broken.
- NEGATIVE: primary CI upper bound < 0.02.
- MIXED: statistically positive but practically sub-threshold, or prespecified regime disagreement.
- INCONCLUSIVE: the interval cannot distinguish the required effect.
- INVALID: leakage, broken metric, illegal transition, risk violation, hash mismatch, or protocol defect.

The disposition is a controlled synthetic result. It is never a model-promotion authorization.

