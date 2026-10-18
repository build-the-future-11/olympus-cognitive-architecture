# Olympus v5 prospective protocol

Fresh successor authorized 2026-10-17. All prior files are inventoried before
implementation. Every prior result, failure, source change, and v4 checkpoint
remains unchanged. This protocol and scientific sources are hash-bound before
development and again before protected evaluation. No protected tuning or
favorable reruns are allowed.

## Questions and matched controls

Can broader corruption coverage and probability-based routing reduce v4's OOD
regression? Can an explicit residual temporal predictor beat the learned v4
predictor AND persistence on clean observations without losing corrupted-input
recovery? These are synthetic reference-model hypotheses, not SOTA claims.

Five new dataset seeds 20265100..20265500 (increments 100), two paired training
seeds 20265601/20265602, and core/broad training give 20 cells and 40 bank fits.
Train 1024 examples; development 256 per regime; protected 1024 per regime.
Offsets +1 train, +2/+3/+4 development; +11/+12/+13/+14 protected. Source IDs and
tensor hashes bind each split. No v4 evaluation example or trained checkpoint is
used. The v4 implementation is reused as an immutable dependency.

Core training uses v4 mixed missing/perturbed inputs. Broad changes half of the
damaged observations to previously observed v4 shift types: token reversal,
image inversion, positive constant state bias. Latents, labels, missingness,
clean observations, parameter initialization, minibatches and update budgets
are matched. The controlled coverage difference adds no examples or updates.

Protected regimes: clean, original corruption, known shift (v4 OOD families,
now explicitly part of development), and novel shift. Novel shift reserves token
relocation by four positions, checkerboard image occlusion, and negative
time-varying state drift. These families occur in neither training nor
development. Every protected regime is freshly generated after all development
checkpoints are frozen; the same protected examples are used for every paired
condition. Novel families are synthetic and do not establish natural-data OOD.

Each bank fits one training half and supplies router/temporal features only for
the other half. Two-bank averaging remains unchanged. Banks use 250 updates,
batch 64, AdamW lr .001, weight decay .01, CF=0, temporal weight .25. v4's clean
nonlinear future target, independent domains and all-legal masks remain intact.

Routing controls: uniform, learned fixed, full-latent semantic, prediction-only
semantic, utility, corruption proxy, fixed 50/50 probability/utility blend, and
privileged task oracle. Prediction-only zeroes the 80 latent coordinates of the
same 105-dimensional router feature vector; network architecture, initial
weights and 300-update budget match full semantic. Heads have separate weights;
all execute for all modes. Oracle uses true labels and is not deployable.

Candidate route selection uses only development clean/corrupted/known-shift:
maximize worst-regime accuracy, then mean accuracy, then declared order
(semantic, prediction_only, utility, blend, fixed, corruption). The selected
route is frozen per cell. Both selection results and all individual routes are
reported, including failures. No protected route switching is permitted.

Temporal controls: original bank's v4 predictor, persistence, new direct MLP,
new residual MLP. Both MLPs read observed state/action and detached bank context,
have identical 88→64→5 architecture and initialization, zero final layer, paired
500-update batches, and independent optimizers and gradient clipping. Only the
residual adds the observed state. This isolates the skip connection; comparing
either new MLP with v4 also changes architecture and training budget and cannot
attribute the entire difference to residual learning. No target, clean state,
corruption metadata, or generator dynamics parameters enter learned inference.

## Registered endpoints

Routing primary: broad development-selected candidate minus core full semantic
on novel-shift accuracy, >=2 pp mean and descriptive 95% interval lower >0.
Original-corruption nonregression requires mean loss <=1 pp and interval lower
>=−2 pp. Both criteria must pass. Known-shift and clean outcomes are mandatory
secondary reports; the full table exposes possible selection tradeoffs.

Temporal primary: broad residual clean MSE <=90% of BOTH core v4 and persistence,
residual-minus-persistence interval upper <0, and corrupted MSE no more than
core v4 +.02. Direct-versus-residual, coverage and all shifted regimes are
mandatory matched ablations. Failure of any primary criterion is reported.

Paired dataset-cluster means average two training seeds. 10,000 percentile
bootstrap samples, fixed seed 20265999. Five clusters and multiple secondary
comparisons make intervals descriptive. Report accuracy, NLL, Brier/ECE,
confidence-ranked 50% selective accuracy, temporal MSE, every cluster delta,
checkpoint/source/prediction hashes, parameter counts and elapsed times.

## Execution and preservation

CPU, one Torch thread, deterministic algorithms, shared exclusive campaign lock.
Before Torch import require 2 GiB available, <=90% swap, stable pageouts; check
memory during training and between evaluations. Development cap 3600 seconds;
remote outer execution cap 4200 seconds for checks, evaluation and export.
No threshold weakening or overwriting receipts. Failure leaves partial artifacts.

Run `python -m scripts.run_robust_fusion_v5 --output artifacts/robust-fusion-v5 run`.
Qualification uses the same entry point with `test` and a separate output path.
The preservation check hashes every inventoried file, blocks any change or loss
under `artifacts/`, and records live source, test, notebook, and documentation
differences separately. This keeps prior claim-bearing receipts immutable without
making legitimate successor implementation changes impossible. Private remote
bundles may carry V5_REMOTE_PRESERVATION.json binding the local pre-upload check
to the complete inventory hash. Recheck the original host after download.
Verify complete denominators, fold disjointness, development-only selection,
common test tensors, freeze order, all raw artifact hashes, and aggregate means.

External benchmark, independent replication, publication and deployment remain
outside this bounded study. Preserve the result even if neither fix succeeds.
