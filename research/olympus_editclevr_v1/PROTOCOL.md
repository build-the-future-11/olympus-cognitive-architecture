# Olympus EditCLEVR v1 prospective protocol

Status: **COMPLETED, PRIMARY REJECTED**. The protocol was frozen before model
development, and the protected evaluation was opened only after all five
development checkpoints and the source/checkpoint freeze existed. The learned
router tied the development-selected control in four seeds and was slightly
worse in one: shifted mean difference `-0.0000500` (95% descriptive interval
`[-0.0001500, 0]`) and ID mean difference `-0.0000667` (interval
`[-0.0002000, 0]`). The predeclared +2 percentage-point shifted-data endpoint
failed; ID non-regression passed. The arithmetic/provenance verifier passed for
five seeds, 20 protected reports and 20 prediction hashes. This is a negative
result and does not authorize promotion.

## Question

Can an observable learned router over before-image, after-image, absolute-change
and direct paired-image experts improve edit-factor classification under hard
distractors and CoGenT distribution shift without materially reducing
in-distribution accuracy?

This is a narrow paired-image representation test. EditCLEVR does not contain
the language, action trajectories, risk labels, legal transition masks or tool
outcomes needed to validate the full six-family Olympus architecture.

## Data and rights

- Dataset: `torux/EditCLEVR`, revision
  `734efa9b0164ba742cd2c39b9c2945c494497c23`.
- Dataset license: CC-BY-4.0; upstream code: MIT.
- The existing payload receipt validates 20,000 unique pair IDs and all 120,000
  referenced files across train, validation, ID, no-op, hard and CoGenT splits.
- Train and validation select model state and the strongest control. Protected
  `test_id`, `test_hard`, `test_cogent` and `test_noop` remain unread until the
  checkpoint/source/protocol freeze exists.
- A deterministic 25% derived no-edit augmentation pairs each selected training
  or validation before-image with itself. It supplies the `none` class without
  reading protected no-op examples.

## Model and matched controls

The model uses shared convolutional encoders for before/after images, a second
shared encoder for signed and absolute changes, four independently supervised
experts and an explicit router. Inputs are bilinearly resized to 96x128 and
mapped from `[0,1]` to `[-1,1]`.

Every checkpoint reports these modes from exactly the same parameters:

1. `learned`: input-dependent routed probability mixture;
2. `uniform`: equal probability mixture over all experts;
3. `pair_only`: the direct paired-image expert;
4. `delta_only`: the absolute-change expert.

All experts receive direct class supervision. The primary control is the best
of pair-only, uniform and delta-only on validation accuracy. Protected results
cannot change that choice.

## Training and endpoints

Five declared optimizer seeds `20266101..20266105`; 15 epochs; AdamW learning
rate 0.0005, weight decay 0.01; batch 64; gradient norm 1.0. The checkpoint with
maximum learned validation accuracy is retained independently for each seed.

Primary effect: learned minus validation-selected control, averaged equally
across `test_hard` and `test_cogent`. Required mean is at least +2 percentage
points and the descriptive five-seed bootstrap interval lower bound must exceed
zero. ID non-regression requires mean loss no worse than 1 point and interval
lower bound at least -2 points. Both conditions are required.

Mandatory reports: accuracy, classwise accuracy, NLL, ECE, learned gate means,
no-op performance, parameter count, training history, checkpoint/source hashes,
all per-seed deltas and all protected prediction hashes. Five optimizer seeds on
one dataset do not constitute independent dataset replication; intervals are
descriptive.

## Execution

CUDA is required and CPU fallback is forbidden. Before Torch import require at
least 2 GiB available memory and no more than 90% swap. The full development,
freeze, protected evaluation and summary command is:

```sh
python -m scripts.run_editclevr_v1 --output artifacts/editclevr-v1 run
```

No model promotion, production claim, full-Olympus claim, SOTA claim or paper
submission is authorized by this bounded study.
