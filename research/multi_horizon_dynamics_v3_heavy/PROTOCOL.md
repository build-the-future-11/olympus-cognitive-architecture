# Multi-horizon dynamics v3 heavy prospective protocol

Status: **FROZEN BEFORE PROTECTED GENERATION**. Authorized by the user on
2026-10-21 as one queued heavy run using only free Kaggle compute. This is a
fresh confirmatory successor after the verified negative v2 result. It does not
reuse any v1 or v2 checkpoint, training example, development example, protected
example, optimizer seed, dataset seed, or bootstrap seed. V2's negative result
is not a tuning input: the question, candidate, controls, dynamics, regimes,
four decision thresholds, and analysis are unchanged.

## Question and hypothesis

Can the same recurrent uncertainty-aware residual dynamics model improve
horizon-8 prediction under clean, corrupted-observation, and shifted-dynamics
regimes relative to identically structured direct and residual controls and a
persistence baseline when training and evaluation receive a much larger fixed
budget?

The primary is supported only if all four unchanged gates pass: candidate
horizon-8 MSE is at least 10% below the best learned control in its worst
protected regime; the paired dataset-cluster bootstrap interval for recovery
error difference lies entirely below zero; clean horizon-8 MSE is at most 0.02
above the residual control; and empirical 90% interval coverage is within 0.10
of nominal. Failure of any gate rejects the primary.

## Data and separation

Three fresh dataset clusters use base seeds 20269100, 20269200, and 20269300.
Each contains 4,096 training and 1,024 development trajectories. Protected
clean, corrupted-observation, and shifted-dynamics partitions contain 2,048
trajectories each and use disjoint offsets +21, +22, and +23. Protected tensors
are generated only after the source/protocol/network freeze and after every
learned checkpoint has been written.

The state/action dimensions, six-step history, eight-step horizon, nonlinear
controlled system, corruption, shift, direct/residual/uncertainty-aware model
definitions, and hidden width remain identical to v2.

## Heavy budget

The three learned models use two fresh paired optimizer seeds, 3,000 AdamW
updates per fit, batch size 256, learning rate 0.002, deterministic algorithms,
and one CPU thread. The fixed campaign contains 18 learned fits, 54 learned
protected evaluations, and 18 matched persistence rows. Relative to v2, the
training, development, and protected partitions are each 8 times larger, the
update count is 15 times larger, and the batch is 4 times larger.

Wall time is capped at 21,600 seconds. The job must use a private free Kaggle
CPU allocation with Internet disabled. Paid compute, accelerators, model
promotion, and follow-on favorable reruns are not authorized.

## Reporting and validity

Report horizon-wise MSE, interval coverage, recovery error, parameter count,
fit time, checkpoint and prediction hashes, and the paired cluster delta. Use
10,000 percentile-bootstrap samples with fresh seed 20269999. Retain exact
wrapper, base runner, protocol, environment, checkpoints, raw rows, summary,
manifest, and any failure receipt.

The embedded base runner must have SHA-256
`45a0375eb5297e8d3071487b61ee44df194f765e4254e9cd1cacc2be6844b636`.
Any mismatch, seed overlap, enabled Internet, non-finite value, missing
denominator, resource refusal, timeout, or post-freeze change makes the run
INVALID. Negative results are retained. This can support only a bounded
synthetic confirmatory conclusion; it cannot establish target-scale capability,
SOTA, deployment readiness, safety, family identity, or model promotion.
