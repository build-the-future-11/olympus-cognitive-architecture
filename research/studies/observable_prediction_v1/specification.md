# Observable forecasts before latent-performance claims

## Status and research boundary

This specification precedes development and confirmatory outcomes. The study is
an applicability and falsification study of a small Olympus forecasting model.
It introduces no claim of a new JEPA family, a trained Hermes assistant, Ghost
execution, universal physical prediction, or superiority over published methods.
The exact executable contract is protocol.json. Historical Olympus model
results and all predecessor studies remain unchanged.

## Introduction and precise question

The existing prediction demo fits a fixed random target representation on a
single affine construction. Its second evaluation supplies the true next state
again; it never feeds a prediction back as the following input. That engineering
demo cannot establish observable multi-step prediction. The missing scientific
question is whether a latent auxiliary objective adds useful forecast accuracy
after the inference network, data, and training resources are controlled.

One-step latent agreement can conceal observable recursive forecasting failure.
We test a temporal EMA-target objective with coordinate variance regularization
because a stable target and nonconstant features might constrain a useful
transition representation. The falsifier is failure to achieve the prespecified
10% practical OOD error reduction over the same network given the same training
CPU time. Failure remains a valid result. No positive-result search is permitted.

## Related work and novelty boundary

BYOL [1] establishes slowly averaged target networks and predictor asymmetry;
VICReg [2] combines invariance, variance, and covariance regularization. Our
variance term borrows only one part of that design. I-JEPA [3] predicts target
representations from informative contexts. Learned latent dynamical models
already combine encoders, future-state prediction, and reconstruction [4].
Polynomial regularized forecasting has a strong inexpensive precedent [5].
These mechanisms are established and cannot be renamed into a novelty claim.

Time-series JEPA applications include LaT-PFN [6] and TS-JEPA [7]. LeJEPA [8]
uses an alternative objective that does not need an EMA teacher. Consequently,
an absent EMA alone is not a universal defect; the Olympus issue is the actual
fixed random target and the interpretation of its reported metric. A recent
assessment of one JEPA instantiation reports architecture-dependent benefits
and harms [9]. It uses a different two-stage setup and does not predict the
outcome of this simultaneous supervised auxiliary-loss study.

Our contribution is a fully specified, source-bound diagnostic experiment,
an observable forecasting implementation, and a retained outcome under strong
controls. It is not the first latent forecasting or JEPA comparison. There is
no theorem that this objective must improve forecasting.

## Problem formulation and data

Each system has dimensionless fully observed state x=(q,v) in R² and an
autonomous deterministic transition at observation interval0.1. The pendulum
obeys q'=v, v'=-sin(q)-0.15v. The Duffing oscillator obeys q'=v,
v'=-q-0.5q³-0.2v. Each family is trained separately, avoiding ambiguity from an
unobserved system identifier. RK4 uses four substeps per observation in float64;
neural training uses float32. These are controlled low-dimensional systems without nuisance variables, not
an empirical benchmark for broad physical intelligence.

Training comprises64 complete trajectories of33 states per family/replicate.
Development trajectories and protected test trajectories have independent
initial conditions and RNG streams. ID starts are uniform in[-1.5,1.5]².
Amplitude OOD changes the initial position magnitude to[1.75,2.5] while leaving
the law and velocity range fixed. This is initial-condition OOD, not transfer
to a new law. Shifted initial distributions do not guarantee disjoint visited
state clouds: trajectories may overlap the training state range. Normalization
uses training data alone. Protocol.json fixes all
sample counts, seed streams, robustness conditions and parameter values.

## Complete chronological model specification

Normalize each coordinate using training mean μ and population std σ, with
σ≥1e-6. Denote normalized state u. E:R²→R⁸ has affine2→32, tanh, affine32→8.
P:R⁸→R⁸ has affine8→32, tanh, affine32→8. D:R⁸→R² has affine8→32, tanh,
affine32→2. All affine maps include biases and use PyTorch default Linear
initialization. For a batch of B states, latent tensors are B×8, hidden activations are B×32,
and state inputs/outputs are B×2. E, P and D respectively contain360,552 and354
trainable parameters:1266 active parameters (5064 float32 bytes). The frozen
teacher adds360 parameters (1440 bytes) during training only.

The forward path is z=E(u), zhat=z+P(z), uhat_next=D(zhat). Thus zhat means
future-state representation; the decoder outputs a future state, not a state
increment. The reconstruction D(E(u)) has an unambiguous current-state target.
All neural arms use these same active trainable modules and initial tensors.

The target encoder Ebar is an exact initial copy of E, with gradients disabled.
For a paired batch(u,w), optimize the arithmetic mean over all coordinates:

    Lstate = mean((D(E(u)+P(E(u))) - w)²)
    Lrecon = mean((D(E(u))-u)²)
    Llatent = mean((E(u)+P(E(u))-stopgrad(Ebar(w)))²)
    Lvar = mean_j max(0,0.2-sqrt(Var_population(E(u)[:,j])+1e-4))
    L = Lstate + 0.1 Lrecon + λlatent Llatent + λvar Lvar

After backpropagation, reject nonfinite losses/gradients, clip the global
gradient norm to1, update with AdamW(lr0.003, weight decay0.0001, betas0.9/0.999, epsilon1e-8,
amsgrad=False, foreach=False, fused=False), and then
update Ebar←0.99Ebar+0.01E. No target gradient enters the optimizer. Full and
mechanism-off models use identical initialization and minibatch-stream prefixes.
Train500 updates with batches of128 sampled with replacement. No early stopping
or best-test-checkpoint selection is allowed. Two auxiliary coefficient pairs
are reserved exclusively for the declared development calibration.

At inference, discard the target encoder and optimizer. Starting from the
initial observed state, repeatedly call D(E(u)+P(E(u))) on the preceding
predicted state for50 steps. No future state enters the inference call. Retain
forecasts in training-normalized coordinates with the original raw trajectories
and training mean/std, so original coordinates are exactly recoverable by the
declared affine inverse. NMSE uses the frozen training scale. Missing/noisy initial-state conditions
modify only the starting observation; evaluation targets stay clean.

The per-state neural inference cost is the sum of the three small affine
networks, O(dh+hk+kh+hk+kh+hd), d=2,h=32,k=8. Training additionally evaluates
the decoder reconstruction and, when used, the target encoder. The same
trainable parameter count therefore does not imply equal training cost.

## Mathematical limits and hand-checkable guarantees

If D is L_D-Lipschitz, triangle inequalities give

    ||D(zhat)-w|| ≤ L_D||zhat-Ebar(w)||
                    +L_D||Ebar(w)-E(w)||+||D(E(w))-w||.

This requires three quantities, not only latent error: target prediction,
target/online alignment, and observable reconstruction. For a transition with
uniform one-step error ε throughout the visited rollout region and Lipschitz
constant L, recursive error satisfies e_h≤εΣ[j=0..h-1]L^j, by induction from
e_(h+1)≤L e_h+ε and e_0=0. Training MSE does not establish a uniform ε.
Neither bound proves superiority. Coordinate variance does not prove rank:
duplicated nonconstant coordinates satisfy every variance floor. We therefore
retain effective covariance rank and coordinate standard deviations.

## Methodology, controls and decision rules

The four loss arms isolate latent and variance terms in a2×2 design. Fixed
targets test the EMA update; shuffled auxiliary future targets test temporal
alignment while preserving correct supervised labels. The direct control has
exactly the same trainable E/P/D. A second direct control receives the full
arm's measured training process CPU time, including optimizer/loop overhead;
its update count is measured rather than assumed equal. Training wall time,
CPU time and all inference costs are reported separately. A1500-update cap or
more than5% CPU mismatch invalidates that matched comparison.

Persistence, affine ridge and cubic polynomial ridge use only the same training
states/targets. Polynomial features include all monomials up to the declared
degree, an intercept and ridge1e-6 on mean normal equations. Features are ordered
by total degree0..p, then q exponent0..degree; v receives the remaining exponent.
Coefficients use a float64 linear solve. This current-state polynomial baseline
is inspired by nonlinear vector autoregression, not a full reproduction of NGRC. They are legitimate
strong controls; no weak baseline is selected after viewing test results.

The independent unit is a paired block containing one separately trained model
per arm in each of two fixed families, with new training and evaluation data. Initialization uses seed100r+f; batch
sampling uses10000r+100f+11 and latent shuffling uses10000r+100f+12, where r
is the declared replicate seed and f the family index0/1.
Eight confirmation blocks are fixed in advance. Time steps and trajectories
are nested measurements, not extra independent training repetitions. The sole
primary endpoint averages normalized squared errors over the first20 OOD
steps, coordinates and trajectories. Paired log ratios aggregate families
equally. A10000-draw paired-block bootstrap supplies95% uncertainty; a
predeclared one-sided exact sign test supplements it. The10% practical effect,
ID regression guard, family guard, failure treatment and total classification
precedence are fixed in protocol.json. The ID guard takes the exponential of
minus the mean paired ID log-ratio (first across families, then blocks) and
requires a ratio≤1.05. Each family mean OOD log-ratio must be positive. Any full
arm divergence in any declared condition fails the divergence guard. The sign
test addresses positive block contrasts; it does not alone test10% magnitude.
The eight-block percentile interval is nominal95%, with no finite-sample
coverage guarantee. All other comparisons are descriptive secondary
analyses, avoiding multiple uncorrected confirmatory claims.

Development may inspect only seeds701/702 and their development trajectories,
under two predefined coefficient pairs. The lower geometric mean full-arm
development20-step NMSE selects one pair; ties select a. Then freeze the
configuration, code commit and source hashes before generating confirmation
seeds1701–1708. The chosen method may lose. No extra seed or hypothesis is
permitted to rescue a frozen result.

## Reproducibility, failure handling and limits

Every attempt owns a fresh directory with source hashes, command, environment,
config, raw trajectories, all forecasts/losses, model and optimizer state,
normalization, RNG state, memory and timing. Checkpoint reload must reproduce
the saved forecast. For each trajectory separately, a nonfinite normalized state or absolute
coordinate above1e6 incurs coordinate-mean squared error1e12 on the offending
and all later steps. Its next input is reset to zero while raw forecasts and
cumulative failure flags are retained; other trajectories continue normally. Resource failures stop execution and remain in the record. Analysis and
figures must regenerate from these artifacts without retraining.

The study does not establish Mac/MPS performance, external-host replication,
language understanding, causal discovery, large-model scaling, Ghost speedup,
or literature-wide superiority. Eight blocks may be insufficient for a narrow
interval. Only two auxiliary coefficient pairs are tuned; other models are not claimed
to be individually optimized published baselines. Inactive auxiliary loss fields
are logged as0 meaning not computed, not measured zero. Teacher, optimizer and
parameter bytes are distinguished from allocator/process memory. A clean-environment
analysis/figure reproduction and adversarial
independent review are required before the evidence is called verified.

## Pre-outcome resource-contract revision

The first dry attempt stopped before warmup/training at a study-chosen2GiB
remaining working-set estimate; its peak RSS was353230848bytes and no family
was generated. Its missing preflight counters cannot be reconstructed. Preserve
that report and protocol-resource-v1.json. The revised resource contract uses
at least1GiB remaining estimated headroom and an external supervisor sampling
child RSS, host/cgroup headroom, swap and OOM counters every0.25seconds. It
terminates on a768MiB sampled child RSS ceiling or headroom violation. This is
a sampled operational guard, not a kernel hard memory reservation; RLIMIT_RSS
is not used. The estimate credits only clean inactive file cache, and does not
credit active cache or slab. The one-lane, two-thread,900second total budget
remains. Every launched attempt, including failures, is charged; no launch is
allowed unless100seconds plus5seconds termination margin remain. No development
or protected outcome informed this revision. A complete monitored dry pass is
required before development. The supervisor stays outside measured model
training CPU time; its wall time and RSS are separately retained.

## Primary references

1. Grill et al. Bootstrap Your Own Latent (2020). https://arxiv.org/abs/2006.07733
2. Bardes, Ponce, LeCun. VICReg (2022). https://arxiv.org/abs/2105.04906
3. Assran et al. I-JEPA (2023). https://arxiv.org/abs/2301.08243
4. Lusch, Kutz, Brunton. Deep learning for universal linear embeddings of nonlinear dynamics (2018). https://arxiv.org/abs/1712.09707
5. Gauthier et al. Next generation reservoir computing (2021). https://pmc.ncbi.nlm.nih.gov/articles/PMC8455577/
6. Verdenius, Zerio, Wang. LaT-PFN (2024). https://arxiv.org/abs/2405.10093
7. Ennadir, Golkar, Sarra. Joint Embeddings Go Temporal (2025). https://arxiv.org/abs/2509.25449
8. Balestriero, LeCun. LeJEPA (2025). https://arxiv.org/abs/2511.08544
9. Feng et al. Does JEPA Pretraining Help Time Series Forecasting? (2026 preprint). https://arxiv.org/abs/2609.31680
