# Stage 07 — Matched Baselines and Ablations

**State: ABLATION_RUN.** Frozen seeds 17/31/47 and an equal 138,112
parameter-step budget compared the reference, simple SFT, and no-packing
conditions. All obtained validation mean 5.5031697 (stdev 0.0947228).

The packing mechanism is explicitly demoted as `mechanism_not_supported`
because the bounded corpus did not create a behaviorally distinct condition.
Artifacts are under `artifacts/stage-execution/research-smoke/`.
