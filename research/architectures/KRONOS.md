# Kronos Architecture

## Purpose and falsifiable claim

Kronos forecasts, plans, observes, and revises under temporal drift. It is not
an always-learning chatbot. Initial experiments are offline and claim only that
explicit temporal state plus guarded adapters improves planning without
measurable regression on retained tasks.

## Architecture

1. **Event encoder:** converts timestamped observations, interventions, delayed
   outcomes, missingness, and environment version into typed event tokens.
2. **Temporal state model:** a compact selective state-space encoder summarizes
   long event streams; a Transformer readout attends to retrieved raw episodes.
   The SSM is an ablation candidate, not a foregone conclusion.
3. **Forecaster:** emits calibrated outcome distributions at specified horizons
   and distinguishes aleatoric uncertainty from missing evidence metadata.
4. **Planner adapter:** return/constraint-conditioned sequence model proposes a
   receding-horizon plan DAG with checkpoints and stop conditions.
5. **Observer and replanner:** compares predicted with observed state, records
   residuals, and invokes Perseus for authorized execution.
6. **Adaptation manager:** trains a new versioned adapter offline on a
   deletion-aware replay set. It tests replay, regularization, and no-update
   baselines; only the promotion service may swap adapters.
7. **Rollback:** every plan and adapter identifies environment, data, base, and
   predecessor hashes; rollback is atomic and tested before promotion.

## Data and losses

Use timestamped trajectories with interventions, delayed rewards, drift
episodes, censored outcomes, counterfactual simulations clearly labeled as
synthetic, and replay/deletion provenance. Add horizon-weighted forecasting,
action prediction, return conditioning, constraint violation, and retention
losses. Never random-split adjacent time points across train and test.

## Evaluation and stop rule

Use forward-chaining evaluation and report calibration by horizon, planning
success, regret, recovery, adaptation gain, backward transfer/forgetting,
deletion compliance, and rollback identity. Compare static Hermes/Perseus
replanning, Transformer-only temporal encoding, SSM hybrid, naive fine-tuning,
replay, and regularization. Online weight mutation remains forbidden until a
fresh-process rollback test and a predeclared forgetting bound both pass.

Input: `TemporalWorkspace + TargetHorizon`. Output: `Forecast + Plan | Stop`.
Current state: **specified, no qualifying checkpoint**.
