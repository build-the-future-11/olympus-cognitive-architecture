# Olympus: risk-monotone counterfactual fusion for governed multimodal prediction

## Status

This Markdown file preserves the original preregistered architecture and experiment
contract. The arXiv manuscript in `research/olympus_irp/paper/main.tex` is now the
evidence-updated paper. A separately versioned controlled synthetic successor completed
25 training runs and 75 protected evaluations with a **NEGATIVE** disposition: the full
learned mechanism tied uniform fusion on clean accuracy and underperformed its corrupted
selective-accuracy point estimate. The result does not establish natural-data or JEPA
performance, and the novelty search still requires independent repetition.

## Abstract

Specialized models can encode complementary linguistic, scientific, operational,
multimodal, temporal, and governance signals, but learned fusion can hide routing
collapse and allow model confidence to weaken an external safety signal. Olympus tests a
different composition: normalized heterogeneous representations are fused by an explicit
gate; leave-one-component-out predictions expose marginal influence; representation
disagreement produces an intrinsic risk estimate; and intrinsic risk combines with a
host risk floor through a monotone noisy-OR rule. A host-supplied legal-transition mask
and deterministic STOP threshold remain outside learned authority. We preregister
comparisons against concatenation, uniform ensembles, ordinary softmax gating, and
standard load-balancing variants. Primary outcomes are task quality, selective risk,
calibration, routing collapse, robustness to missing or corrupted components, latency,
and memory. Super-resolution is studied separately through a geometry-conditioned JEPA
with latent prediction, multiscale reconstruction, and phase-aware spectral coherence.
No capability or superiority result is reported before source-disjoint datasets,
matched-compute trials, confidence intervals, and independent reproduction exist.

## 1. Research question

Can a heterogeneous family of bounded predictive models be fused so that:

1. fusion improves source-disjoint task performance over matched baselines;
2. each component's marginal influence is observable and stable;
3. learned evidence can increase, but never decrease, host-declared risk;
4. forbidden transitions remain impossible and high-risk states fail closed; and
5. gains survive missing-modality, corruption, distribution-shift, and compute-matched
   ablations?

The question is architectural. It does not assume that the six Olympus family names
correspond to trained frontier-capability models.

## 2. Method

Let component `i` emit representation `h_i`. Olympus maps heterogeneous widths into a
shared space and normalizes them:

```text
z_i = LayerNorm(W_i h_i + b_i)
g = softmax(r([z_1; ...; z_m]) / tau)
z = LayerNorm(sum_i g_i z_i)
```

`tau > 0` is a fixed preregistered gate temperature. The implementation emits `g` and
every `z_i`; routing is therefore an evaluated output, not hidden telemetry.

### 2.1 Counterfactual component influence

For each component, Olympus computes a leave-one-out state without rerunning its
backbone:

```text
z_{-i} = LayerNorm((sum_j g_j z_j - g_i z_i) / max(1 - g_i, epsilon))
```

The decision head is evaluated on `z` and on every `z_{-i}`. Training may minimize the
gate-weighted divergence

```text
L_cf = mean_b sum_i stopgrad(g_bi) KL(stopgrad(p_b) || p_{b,-i})
```

where `p` is the full decision distribution. This is a robustness objective, not proof
that a family is causally necessary. Evaluation reports the unregularized ablation effect
as well as the trained consistency term so that robustness cannot be mistaken for
specialization.

### 2.2 Disagreement and monotone risk

Let `c` be the normalized gated centroid. Representation disagreement is

```text
d = 1/2 sum_i g_i [1 - cosine(z_i, c)],       0 <= d <= 1.
```

An intrinsic estimator returns `q = sigmoid(f([z; d]))`. Given a host risk score `r_h`,
effective risk is

```text
r_eff = 1 - (1 - r_h)(1 - q).
```

For `r_h, q` in `[0, 1]`, `r_eff - r_h = q(1 - r_h) >= 0`. Learned state therefore
cannot reduce the host risk floor. The risk head requires explicit labels; without them,
its loss is exactly zero and no calibration claim is permitted.

### 2.3 Governed transition boundary

The host supplies a Boolean legal-transition set `A(x)`. Aion masks all other logits to
negative infinity. When `r_eff` reaches the fixed kill threshold, the selected transition
is STOP, provided STOP is host-allowed; otherwise execution raises an error. A forced
STOP zeroes the exposed decision logits. Raw logits remain available for audit and loss
calculation but never authorize execution.

### 2.4 Joint objective

The implemented reference objective is

```text
L = L_decision
  + lambda_t L_transition
  + lambda_k L_temporal
  + lambda_u L_heteroscedastic
  + lambda_r L_intrinsic-risk
  + lambda_b KL(mean(g) || Uniform)
  + lambda_cf L_cf.
```

All weights must be finite and non-negative. Forbidden transition targets, out-of-range
labels, empty masks, non-finite targets, and misaligned batches are rejected before an
optimizer step. The uniform gate term is a baseline regularizer, not assumed optimal;
loss-free balancing is a required comparator.

### 2.5 JEPA super-resolution objective

The upscaler predicts EMA-target high-resolution patch embeddings from masked bicubic
context plus spatial and temporal coordinates. Its objective combines Charbonnier pixel
error, edge error, normalized latent prediction, latent-gradient geometry, variance
floor, log-magnitude spectrum error, SSIM, multiscale pyramid consistency, and
target-magnitude-weighted phase coherence. Tiled inference bounds working memory;
overlapping predictions use separable squared-sine ramps at internal boundaries to
reduce hard tile seams without attenuating true image edges. Eight-way dihedral
self-ensemble is an optional quality/compute tradeoff.

## 3. Relationship to prior work

- I-JEPA established masked prediction in representation space and emphasized large
  semantic target blocks and informative context
  ([Assran et al., 2023](https://arxiv.org/abs/2301.08243)). Olympus uses the predictive
  representation principle but its current convolutional super-resolution reference is
  not an I-JEPA reproduction.
- V-JEPA 2 extended predictive representations to video understanding, action-conditioned
  world modeling, and planning
  ([Assran et al., 2025](https://arxiv.org/abs/2506.09985)). Olympus must compare temporal
  prediction against a matched single-world-model baseline before claiming benefit.
- Expert Choice routing addresses expert under-training through capacity-constrained
  expert-selected routing
  ([Zhou et al., 2022](https://arxiv.org/abs/2202.09368)). Olympus currently uses
  sample-level softmax fusion and therefore cannot claim sparse-compute efficiency.
- Auxiliary-loss-free balancing avoids interference gradients from conventional MoE
  balance losses
  ([Wang et al., 2024](https://arxiv.org/abs/2408.15664)). It is a mandatory routing
  baseline because Olympus currently includes an explicit balance penalty.
- Conformal Risk Control provides finite-sample control for monotone losses
  ([Angelopoulos et al., 2022](https://arxiv.org/abs/2208.02814)). Olympus's monotone
  noisy-OR is an architectural invariant, not a conformal guarantee; a calibration split
  and conformal procedure would be required for such a guarantee.

The candidate contribution is the combined, testable contract of normalized
heterogeneous fusion, emitted leave-one-family-out decisions, disagreement-conditioned
intrinsic risk, and a host-monotone governance boundary. Novelty is rejected if a prior
system already implements the same contract or if matched ablations show no benefit.

## 4. Preregistered hypotheses and falsifiers

| ID | Hypothesis | Primary measure | Falsifier |
|---|---|---|---|
| H1 | Fusion improves task quality | paired task metric | no positive confidence-bound improvement over the best matched single family |
| H2 | Counterfactual training improves component-loss robustness | corruption and family-drop delta | equal or worse delta at matched clean quality and compute |
| H3 | Disagreement helps detect failures | AUROC and risk-coverage | no gain over entropy, max probability, and host risk alone |
| H4 | Monotone risk reduces unsafe continuation | unsafe-continuation rate | no reduction, or excessive safe-case STOP rate beyond the preregistered budget |
| H5 | JEPA additions improve reconstruction | source-disjoint PSNR, SSIM, edge and frequency metrics | no matched-compute improvement over bicubic and the prior objective |

Negative, mixed, inconclusive, and invalid outcomes are retained. Hyperparameters may not
be changed after inspecting protected test results.

## 5. Required baselines and ablations

1. Best individual family at matched trainable parameters.
2. Concatenation plus MLP with matched fusion parameters.
3. Uniform averaging after the same projections.
4. Softmax gate without disagreement, risk head, counterfactuals, or balance loss.
5. Gate balance loss versus loss-free bias balancing.
6. Host risk only, intrinsic risk only, `max(host, intrinsic)`, and noisy-OR.
7. Every leave-one-family-out inference ablation.
8. Shuffled-family and corrupted-modality controls.
9. Frozen-backbone versus jointly trained variants.
10. JEPA: bicubic, pixel-only, latent-only, prior v3 objective, plus each new term removed.
11. JEPA tiled and self-ensemble quality, latency, and peak-memory curves.

All comparisons use identical source IDs, optimizer budgets, stopping rules, seeds, and
evaluation code. Parameter count, activated FLOPs, wall time, energy proxy, peak memory,
and failed runs are reported.

## 6. Data and evaluation protocol

Every source must record ownership, license, retrieval identity, transformations, and
content hash. Train, calibration, validation, and protected test source identities are
disjoint. Near-duplicate and contamination scans run before training. Public benchmarks
are never mixed into optimization data.

Required data strata are:

- instruction and abstention supervision for Hermes;
- cited synthesis, contradiction, and uncertainty labels for Prometheus;
- workflow traces with tool outcomes and recovery cases for Perseus;
- rights-cleared image-text, localization, retrieval, and corruption pairs for Atlas;
- temporally ordered state/action/next-state trajectories for Kronos;
- legal-transition masks, risk labels, STOP cases, and adversarial policy cases for Aion;
- native high/low-resolution pairs with source-level disjointness for the upscaler.

The repository currently contains only 36 proprietary smoke instruction records. They
exercise plumbing and cannot qualify any hypothesis above. The aligned Olympus tensor
manifest refuses missing fields, source overlap, digest drift, shape drift, non-finite
values, forbidden transition labels, and undeclared extensions. The implemented
development runner additionally refuses model/data dimensional mismatches and records
initial and final held-out evaluations without authorizing promotion.

The diagnostic surface is fixed before confirmatory use: decision accuracy is computed
only where a decision is available; STOP and transition outcomes are always counted;
temporal error respects the registered mask; routing reports mean gate mass, normalized
entropy, and effective component count; leave-one-component-out predictions report KL
shift and top-class flip rate; labeled intrinsic risk reports Brier score and ten-bin
ECE; and selective performance is reported at fixed 25%, 50%, 75%, and 100% coverage.
These diagnostics are descriptive until adequate calibration and test-set sample sizes
are established.

## 7. Statistics

- Freeze the protocol and test set before the first confirmatory run.
- Use at least five preregistered seeds for stochastic comparisons unless a power analysis
  justifies more.
- Report per-seed values, paired differences, bootstrap 95% confidence intervals, and a
  robust location estimate.
- Correct the confirmatory family of primary tests for multiplicity; label all other
  analyses exploratory.
- Report selective risk as a risk-coverage curve, not a single favorable threshold.
- Calibrate thresholds only on the calibration split. Evaluate once on protected test.
- Treat NaN, OOM, timeout, and resource-governor rejection as observed outcomes.

## 8. Security and operational invariants

- The model never executes tools or approves its own transition mask.
- Checkpoints and datasets are digest-pinned and non-symlink files.
- Outputs are not promotion receipts.
- A resource governor obtains an exclusive workload lock and enforces memory/swap floors.
- Existing artifacts are never overwritten.
- STOP and unavailable decisions remain visible in all metrics.

## 9. Current evidence

The reference implementation has software tests for differentiability, extension
channels, monotone risk, STOP suppression, counterfactual outputs, immutable checkpoint
round trips, aligned dataset verification, a complete composed training/evaluation
run, and JEPA runtime paths. A bounded 34,528
parameter infrastructure run completed one epoch on the 36-record smoke corpus. Candidate
held-out loss was lower than its deterministic untrained baseline, but exact-match and
format-compliance rates were both 0%, so the smoke gate failed and promotion remains
forbidden. This is useful negative pipeline evidence, not evidence for H1-H5.

## 10. Release gates

The research claim may advance only when all of the following exist:

1. rights-reviewed, source-disjoint datasets for the claimed modalities;
2. frozen configs, source snapshot, environments, seeds, and artifact hashes;
3. every required baseline and ablation at matched compute;
4. preregistered metrics with uncertainty and failure accounting;
5. adversarial safety and distribution-shift evaluation;
6. a second clean replay matching the original receipt; and
7. independent review or reproduction.

Until then the correct status is `REFERENCE_IMPLEMENTED; SCIENCE_NOT_RUN`.
