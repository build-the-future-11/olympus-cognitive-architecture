# Mean-preserving uncertainty dynamics: engineering contract

## Purpose and scope

The retained multi-horizon v2 candidate uses Gaussian likelihood gradients for
both its predictive mean and variance. Those gradients need not follow the
mean-only squared-error trajectory. This separate engineering successor gives
the residual mean its original full mean-squared-error objective while fitting
variance through an independent, detached head. It does not change any frozen
v1, v2, or v3 study, and it does not assert improved protected performance.

This applies the established principle in [Stirn et al., *Faithful
Heteroscedastic Regression with Neural Networks*, AISTATS 2023](https://proceedings.mlr.press/v206/stirn23a.html).
[Seitzer et al., *On the Pitfalls of Heteroscedastic Uncertainty Estimation with
Probabilistic Neural Networks*, ICLR 2022](https://arxiv.org/abs/2203.09168)
provide related analysis of likelihood optimization and propose beta-NLL.
The present implementation claims neither a new uncertainty mechanism nor an
independently demonstrated explanation of the historical negative results.

## Architecture and exact objectives

Defaults reproduce the v2 residual control: history has shape `[B,6,6]`, with
four state and two action coordinates; future actions have shape `[B,8,2]`.
A batch-first GRU maps the history to width 32. At each future step a separate
GRUCell receives the current four-coordinate predicted state and the next
two-coordinate action. A linear 32-to-4 head predicts a residual that is added
to the current state. The output mean has shape `[B,8,4]`. No dropout or
stochastic sampling is performed by either forward pass.

The uncertainty model takes an explicit deep copy of an existing mean model.
Its independent linear 32-to-4 variance head receives detached recurrent
features and produces log variance clamped to `[-8,5]`. The loss is

\[
L_\mu=\operatorname{mean}[(\mu_\theta-y)^2],\qquad
L_v=\tfrac12\operatorname{mean}
 [(\operatorname{stopgrad}(\mu_\theta)-y)^2 e^{-v_\phi}+v_\phi].
\]

The variance head sees `stopgrad(h_theta)`; the variance loss also detaches the
mean. Both barriers are necessary because the mean feeds the autoregressive
trajectory. Targets are treated as observations. Gaussian normalizing constants
are omitted because they do not affect either optimizer.

Two AdamW optimizers own disjoint parameter sets. The mean uses precisely
`L_mu`, without a factor of one half or uncertainty weighting. Gradient norms
are clipped separately for each parameter set, with a default ceiling of one.
Defaults are learning rate 0.002 and weight decay 0.01. Configuration and input
validation reject malformed dimensions and non-finite values before updates.

## Mean-trajectory invariance

Assume that a mean-only control and the candidate begin with identical mean
parameters, AdamW mean state, optimizer options, and mean clipping threshold.
They receive identical batches in the same order, and use identical
deterministic kernels and precision. There must be no external parameter
mutation, uncertainty-dependent batch selection, loss scaling, early stopping,
optimizer scheduling, or random-stream divergence.

At update zero their mean forwards are equal. Because both paths from variance
loss to the mean parameters are detached, `grad_theta(L_v)=0`; therefore the
candidate's mean gradient is exactly the control's gradient of `L_mu`.
Mean-only clipping applies the same norm and scaling to the same ordered
gradient tensors. AdamW then receives the same parameters, state and gradient,
and produces the same updated parameters and state. Induction gives equality
at every subsequent update and forward pass under these assumptions.

This is a statement about the implemented optimization trajectory. It implies
equal predictive means for equal evaluation inputs under the same execution
conditions. It does not imply calibrated variance, superior mean accuracy,
valid extrapolation, faster execution, or a generally positive research result.
Bitwise equality across different devices, library versions or kernels is not
promised.

**Combined global clipping violates the premises.** If variance gradients enter
the norm used to clip mean gradients, changing the variance head changes the
mean update. The tests include a deliberately coupled clipping counterexample.

## Checkpoints and continuation

Checkpoints retain the architecture, optimizer configuration, both parameter
sets, both AdamW states, and the completed update count. Publication uses a
temporary file followed by an atomic hard-link create, so an existing target
cannot be overwritten. Loading uses `weights_only=True`, validates schema,
parameter identities/shapes/finiteness, optimizer configuration and state, and
reconstructs independent optimizers. Data order and external RNG state are the
caller's responsibility: the model itself is deterministic, and a resumed
caller must supply the same next batch.

Optimizer steps are not transactional. An exception during either optimizer's
update or the subsequent finite-parameter check can leave a partially advanced
trainer. Discard that object and reload the last valid checkpoint; do not
continue from it. Checkpoint publication validates the update counters and
finite optimizer/model state, but this component does not promise rollback of
an optimizer failure.

## Bounded verification plan

Artificial tensor regressions check exact mean gradients and several AdamW
updates against a mean-only control, variance-head updates, a checkpoint round
trip and the next continued step, malformed and non-finite input rejection,
invalid checkpoint rejection, and refusal to overwrite an existing checkpoint.
These tests use no scientific datasets, protected examples, or benchmark metric.
The engineering budget is at most 100 tiny artificial optimizer updates, one
CPU thread, and 120 seconds per validation command. A future efficacy study
requires its own hypothesis, controls, budget, prospective freeze and fresh
protected data; this contract does not authorize one.

## Observed engineering verification

On 22 October 2026 the 28 artificial regression cases passed. The final pass
took 3.09 seconds in Python 3.14.8 with Torch 2.14.1+cpu and one thread. Ruff
0.17.0 and strict MyPy 2.4.0 passed for the implementation and tests. The local
CPU Torch build differs from the repository's complete hosted dependency
environment; these results are scoped to the recorded environment.

The regressions compare the mean forward directly with the untouched frozen
v2 class without invoking its campaign, every mean gradient, and four complete
AdamW mean-state updates. They also verify variance-head learning, both
optimizer states after checkpoint reload, the identical next update, and a
counterexample using the deliberately incorrect global clipping rule. These
are engineering outcomes only. No efficacy, calibration, protected evaluation,
or promotion claim follows.
