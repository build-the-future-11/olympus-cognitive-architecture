# Olympus six-family reference system

For the maintained bounded training, interruption/resume, and Kaggle commands,
see [the 2026-10-16 campaign handoff](kaggle-campaign.md). Its readiness status is
BLOCKED pending numerical and CUDA checks; older study outcomes remain historical.

`OlympusReferenceSystem` is the bounded composition layer for the six family
reference implementations. It is executable and differentiable, but it is not
a promoted Olympus checkpoint or an autonomous agent runtime.

## Composition

The model runs five capability references over aligned inputs:

1. Hermes supplies the grounded assistant representation.
2. Prometheus supplies the domain-specialized research representation.
3. Perseus supplies the sparse operational representation and routing record.
4. Olympus-Atlas supplies multimodal grounding and retrieval state.
5. Kronos supplies the final valid temporal state and planning representation.

Each representation is projected into a common fusion dimension. A learned
gate returns an explicit weight for every contributing family, and the weighted
state feeds a bounded decision head, a positive uncertainty head, and Aion.
The family weights are part of the output rather than hidden orchestration
state.

`fusion_strategy="uniform"` is a registered parameter-matched control. The
router is still executed, but its output is multiplied by zero and replaced by
exact uniform weights, so the control retains the same instantiated parameters
and forward router computation. `disagreement_risk=false` similarly preserves
the risk network and replaces only its disagreement coordinate by zero. These
switches exist for causal ablation; they are not runtime quality modes.

Projections include per-component normalization, and the composition accepts
typed extension feature channels declared in `OlympusSystemConfig`. It emits
every projected component state and leave-one-component-out decision, enabling
routing and marginal-influence audits without rerunning the family backbones.

Aion remains a controller boundary rather than another voting expert. The host
provides the legal research-stage transition mask and the risk score. Aion can
rank only allowed transitions. When the deterministic risk threshold forces
STOP, `decision_available` becomes false and the exposed decision logits are
zeroed. The raw logits remain available for training and audit, never as an
execution authorization.

Family-to-centroid disagreement feeds a separately supervised intrinsic-risk
head. Intrinsic risk combines with host risk as
`1 - (1 - host) * (1 - intrinsic)`, so learned state can raise but never lower
the host risk floor. This invariant is not described as calibrated risk unless
labeled calibration data and held-out evaluation exist.

## Training objective

`olympus_system_loss` combines:

- decision classification only for examples where Aion did not force STOP;
- legal research-stage transition classification;
- Kronos next-state error over an optional temporal mask;
- heteroscedastic uncertainty calibration; and
- a small family-gate balance penalty to expose and discourage collapse onto a
  single family during development;
- optional intrinsic-risk supervision; and
- a gate-weighted leave-one-family-out consistency objective.

Targets forbidden by the host transition mask are rejected before loss
calculation. Model outputs do not execute tools, mutate external state, approve
research, or promote checkpoints.

## Evidence boundary

The local suite exercises forward composition, gradient flow through every
family backbone, forced STOP behavior, forbidden-target rejection, and input
alignment. This establishes an engineering contract only. Real training still
requires licensed task data, family-specific supervision, frozen splits,
matched ablations, safety evaluation, and independent review.

`OlympusSystemRuntime` adds resource admission, device selection, optional
mixed precision and compilation, finite-output checks, clipped finite-gradient
training steps, and digest-pinned immutable checkpoints.
`OlympusSystemDatasetManifest` binds aligned multimodal tensors to
source-disjoint IDs, shapes, dtypes, and hashes. See
`research/OLYMPUS_PAPER_DRAFT.md` for the formal method and preregistered
falsification protocol.

## End-to-end development run

The composed pathway now has a single fail-closed runner. It verifies the
manifest and every tensor digest, checks model/data dimensional compatibility,
acquires the memory/swap workload guard, trains with a warmup-plus-cosine
schedule and clipped gradients, evaluates the untouched source-disjoint split,
and atomically writes a digest-pinned checkpoint plus `report.json`. The report
records the complete validated training config and its canonical SHA-256:

```bash
.venv/bin/olympus bu1ld train-olympus-system \
  path/to/aligned/manifest.json \
  configs/olympus-reference-system.json \
  artifacts/olympus-system/run-001 \
  --steps 1000 --batch-size 8 --warmup-steps 100
```

The held-out report includes decision and legal-transition accuracy, masked
next-state MSE, STOP counts, mean uncertainty, intrinsic-risk Brier score and
ECE when labels exist, a risk-coverage curve, mean gates, normalized routing
entropy, effective component count, and per-component leave-one-out KL and
class-flip rates. The report always sets `promotion_authorized` to false.

The repository does not currently contain a rights-reviewed aligned Olympus
multimodal/temporal/risk corpus. The existing 36-record instruction smoke
dataset is intentionally not coerced into this schema. A run is scientifically
eligible only after the data requirements in the paper are satisfied.

Two separately versioned synthetic studies tested increasingly explicit fusion
mechanisms without reopening protected outcomes. The v2 controlled router and
the v3 reliability-supervised shrinkage router both produced NEGATIVE primary
results across 25 fitted checkpoints and 75 protected evaluations per study.
All tested host-risk and legal-transition invariants held. Those results narrow
the synthetic mechanisms; they do not answer the natural-data composition or
JEPA super-resolution questions.

Large-image JEPA inference uses position-aware squared-sine overlap windows,
which blend internal tile boundaries while leaving true exterior image edges
unattenuated. Eight-way self-ensemble remains a separate compute/quality option.
