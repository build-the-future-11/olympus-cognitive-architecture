# Olympus v4: oracle routing and matched mechanism diagnostics

This is a new, user-authorized successor. v2/v3 negatives and all earlier Kaggle
results remain unchanged. `PRIOR_INVENTORY.json` hashes 1,102 pre-existing files,
including dirty source files. Those files are checked again before freeze and
after evaluation. No external campaign or historical protected set is resumed.

## Question and scope

Can an input-dependent router exploit useful complementary information from the
existing five Olympus reference encoders after removing target-derived metadata,
covering training corruptions, and learning per-expert usefulness out of fold?

The five original family encoders and their projections are reused. V4 adds
individually supervised expert classification heads and probability fusion.
This differs from v3 latent fusion. Consequently absolute cross-version scores
cannot establish that any single repair caused a gain. Matched v4 ablations can.
The counterfactual penalty is an analogous leave-one-expert-out KL penalty in
probability space, not an unchanged v3 training loss.

## Prospective design

The machine-readable protocol is `PROTOCOL` in `scripts/run_oracle_fusion_v4.py`.
Five independent dataset seeds, two common training seeds, two corruption coverage
conditions, and two counterfactual weights give **40 cells / 80 expert-bank fits**.
Each expert bank trains 250 updates of batch 64, AdamW at 0.001, weight decay 0.01,
gradient clipping 1. The only difference in paired CF runs is weight 0 versus 0.05.
Each cell trains all router heads for 300 updates at the same learning rate.
No post-outcome hyperparameter search, extra seed selection, or favorable rerun.

Train: 1,024 examples; development: 256; protected clean, corrupted and OOD:
1,024 each. Each dataset seed produces fresh examples. Train/dev/test IDs and
tensor bytes are bound. Test data is generated only after all 40 development
reports and checkpoints are frozen. Independent repeated runs use new output
directories, preserving all failures. Frozen source or checkpoint drift aborts.

## Data repairs and controls

Domains are sampled independently of targets. Legal-action masks are constant
all-legal and contain no answer. Domain-only lookup and mask-only majority
classifiers are fit on training labels and reported on every protected split.
These are classification diagnostics; no transition-policy efficacy is claimed.

`mixed` training includes clean, missing-only, perturbed-only, and simultaneous
missing/perturbed examples. `missing_only` retains the old support gap as a
matched ablation on identical underlying labels and clean trajectories. Test
missing interventions are paired identically between coverage conditions;
only the added perturbation interventions differ. Test
corrupted data has one missing, one perturbed, one clean modality. OOD reserves
new interventions (token reversal, image inversion, state bias); class semantics
are unchanged. The modalities remain synthetic and intentionally interpretable.

Future state follows a nonlinear latent recurrence. Inputs observe noisy,
possibly erased/corrupted states; targets always contain clean future states.
Report temporal MSE against persistence and the old observed-state-plus-action
formula. No privileged clean state is provided to learned inference.

## Cross-fitting and frozen-bank controls

Each cell partitions training IDs deterministically into two disjoint halves.
Bank 0 trains on even IDs and generates router targets on odd IDs; bank 1 does
the converse. Router inputs are detached family states, class probabilities and
entropy. Utility targets are whether that held-out expert's prediction is correct.
Corruption targets encode clean=1, missing=0, perturbed=0.25. Atlas consumes both
text and image, so its proxy target averages both modalities' quality.

Router fitting consumes only out-of-fold predictions from training examples.
The final frozen predictor averages outputs from the two half-trained banks;
there is no full-data refit that would change calibration or invalidate pairing.
All modes use identical banks and execute all router heads. Learned router heads
have independent parameters, without shared trainable features or cross-loss
gradient transfer. Heads have matched architecture; all exist in the common
supernetwork, including for the uniform and privileged controls. Inference
parameter allocation and head execution are matched, not total scientific cost
or oracle information access. Report runtime, updates and actual parameter counts.

Routes:

1. Uniform probability averaging.
2. Learned input-independent fixed mixture.
3. Semantic router trained directly on fused classification NLL.
4. Utility router supervised by out-of-fold expert correctness.
5. Corruption router supervised by corruption-state proxy labels.
6. Oracle corruption: known generator quality, privileged diagnostic only.
7. Oracle task: select the expert with highest true-class probability, privileged
   label-informed diagnostic (not a universal accuracy upper bound).
8. Best single expert, selected on development accuracy.
9. Semantic training-mean gates: freeze the semantic router's mean gates on its
   out-of-fold training features, isolating dynamic versus fixed routing.

## Endpoints and analysis

Primary: full corrupted classification accuracy, utility minus the strongest
development-selected deployable control (uniform/fixed/semantic/best single/
semantic training-mean), under mixed coverage and zero CF weight. A supported
synthetic signal requires >=0.02 mean improvement and a paired 95% interval above
zero. Otherwise the registered hypothesis is NOT_SUPPORTED. Report all outcomes.

Secondary: every routing contrast, clean/OOD accuracy, NLL, confidence-ranked
selective accuracy at 50% population coverage, multiclass Brier/ECE, per-expert
accuracy, gate means, utility Brier, and temporal MSE. Selective ranking uses class
confidence; it is not the old v3 risk-ranked endpoint. Controlled coverage and CF
contrasts use common seeds and identical protected examples. Paired bootstrap:
10,000 draws of five dataset-cluster means, averaging two training seeds within
each cluster. Only five clusters and multiple secondary comparisons mean intervals
are descriptive; individual seed or example resampling must not inflate precision.

The external-benchmark/SOTA requirement is not satisfied by this synthetic
diagnostic. No claim of superiority to public task-specific methods, natural-data
transfer, independence, policy safety, or promoted Olympus capability is allowed.
The appropriate next external task must be frozen separately after identifying
which mechanism, if any, survives this diagnostic.

## Execution and limits

Before Torch import: existing 2 GiB available-memory minimum, <=90% swap, stable
pageouts, shared campaign-validation lock. Memory is checked every 50 expert
updates and between cells. CPU only, one Torch thread, deterministic algorithms,
fixed 3,600-second development budget. All receipts/checkpoints are exclusive
writes; exceptions leave a dated failure receipt and partial files. No overwrite,
lock removal, threshold weakening or deletion is allowed.

Run: `rtk proxy .venv/bin/python -m scripts.run_oracle_fusion_v4 --output artifacts/oracle-fusion-v4 run`.
Verification: guarded focused tests, Ruff, complete expected cell counts, artifact
hashes, disjoint fold IDs, independent summary recalculation, prior-file hashes.

For a private remote CPU run, the bundle may contain `V4_REMOTE_PRESERVATION.json`
binding a successful local pre-upload verification to the inventory hash. That
receipt verifies the baseline only; it does not claim access to the original
host's files during remote training. The original host must be rechecked after
retrieving results. The same source/data/memory/freeze checks apply remotely.
