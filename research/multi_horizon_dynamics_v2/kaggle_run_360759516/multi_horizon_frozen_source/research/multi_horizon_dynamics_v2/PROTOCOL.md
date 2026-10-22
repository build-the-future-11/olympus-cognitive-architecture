# Multi-horizon dynamics v2 prospective protocol

Status: **FROZEN BEFORE PROTECTED GENERATION**. Authorized by the user on
2026-10-21 for one free Kaggle execution. This is a fresh synthetic successor
to the Internet-enabled invalid v1 attempt; it does not reuse v1 or v5
checkpoints, examples, seeds, or protected outcomes. The hypothesis, models,
thresholds, and analysis are unchanged from v1. A pre-protected outbound TCP
probe must refuse execution unless Internet access is disabled.

## Question and hypothesis

Can a recurrent uncertainty-aware residual dynamics model improve long-horizon
prediction under clean, corrupted-observation, and shifted-dynamics regimes
relative to identically structured direct and residual controls and a
persistence baseline?

The primary hypothesis is supported only if the uncertainty-aware residual
candidate has at least 10% lower horizon-8 MSE than the best learned control in
the worst protected regime, the paired dataset-cluster bootstrap interval for
that difference is entirely below zero, clean horizon-8 MSE is no more than
0.02 above the ordinary residual control, and empirical 90% interval coverage
is within 0.10 of nominal. Failure of any criterion rejects the primary.

## Data and separation

Three dataset clusters use base seeds 20268100, 20268200, and 20268300. Each
cluster has 512 training and 128 development trajectories. Protected clean,
corrupted-observation, and shifted-dynamics sets contain 256 trajectories each
and use disjoint offsets +11, +12, and +13. Protected tensors are generated only
after the protocol/source freeze and all learned checkpoints are written.

Trajectories have four state dimensions, two action dimensions, six observed
history steps, and eight prediction horizons. The nonlinear controlled system,
noise levels, corruption, and shift are fixed in the source before execution.

## Models and budget

Controls are persistence, recurrent direct prediction, and recurrent residual
prediction. The candidate uses the residual architecture with a Gaussian
negative-log-likelihood objective and an explicit log-variance head. Learned
models use two paired optimizer seeds, 200 AdamW updates, batch size 64, learning
rate 0.002, one Torch thread, deterministic algorithms, and CPU execution.
No model or threshold selection uses protected outcomes.

The campaign is limited to 18 learned fits and 54 protected learned-model
evaluations plus matched persistence evaluations. Wall time is capped at 3,600
seconds. It must run on a free Kaggle CPU allocation with Internet disabled.

## Reporting and validity

Report every horizon-wise MSE, candidate interval coverage, recovery error,
parameter count, elapsed fit time, checkpoint hash, prediction hash, and paired
cluster delta. Use 10,000 percentile bootstrap samples with seed 20268999.
Source/protocol/config identities, checkpoint files, raw cell reports, summary,
environment, and failure receipts must be retained.

Any source mismatch, overlapping seed, non-finite value, missing denominator,
resource refusal, timeout, or post-freeze change makes the campaign INVALID.
Negative results are retained. This study supports only a bounded synthetic
dynamics conclusion; it cannot establish target-scale capability, SOTA,
deployment readiness, safety, or model promotion.
