# Olympus Reality Ledger

Initialized from the repository and local runtime on 2026-08-21; reconciled
through 2026-10-18. Status terms are literal; no model-family capability is
inferred from architecture code.

## Real and verified

- Cognitive research demos: executable Python implementations with automated
  tests. These establish engineering behavior only.
- Foundry registry: durable dataset, experiment, checkpoint, evaluation, model,
  and append-only evidence records backed by SQLite.
- Foundry golden path: real maximum-likelihood character-bigram training,
  held-out baseline comparison, checkpoint hashing, export, promotion, restart
  recovery, generation, and tamper rejection.
- Stable model interface: model listing and non-streaming chat completion API.
- Local provider discovery: Ollama is installed and reports `qwen3:8b`, GGUF
  Q4_K_M, 8.2B parameters, with completion/tool/thinking capabilities in its
  local manifest.
- Deep Foundry controls: licensed instruction-record schema, immutable
  split/manifest hashes, deduplication, contamination and PII rejection,
  exclusive RAM/swap governor, real tiny causal SFT, LoRA, nibble-packed 4-bit
  QLoRA, checkpoint resume, held-out category evaluation, int4/int8
  quantization, and fail-closed hash-bound promotion gates.

## Implemented but not a trained Olympus model

- `HermesNano`: deterministic interpretation-and-memory integration fixture. It
  has no trained weights and must not be represented as Hermes Alpha.
- JEPA and transform demos: small synthetic training routines for testing
  research mechanisms, not general-purpose model checkpoints.
- As of 2026-10-11, bounded differentiable reference architectures exist for
  Hermes, Prometheus, Perseus, Atlas, Kronos, and Aion. They cover
  tool/abstention heads, domain adapters, sparse expert dispatch, multimodal
  patch grounding, temporal world-model planning, and a host-constrained
  research-stage router with a deterministic STOP rule. All are
  `REFERENCE_IMPLEMENTATION_NOT_PROMOTED`; none is a target-scale checkpoint.
- The JEPA upscaler successor now predicts masked high-resolution subpixel
  targets rather than embeddings of a downsampled target. It has multiscale
  prediction, robust pixel/edge/embedding/geometry/variance/frequency/SSIM
  losses, extended held-out metrics, and digest-pinned checkpoint reload. This
  is engineering evidence only; no natural-image superiority claim exists.
- `OlympusReferenceSystem` now composes the six reference contracts through
  explicit family gates, joint coordination/temporal/uncertainty losses, and
  Aion's host-constrained STOP boundary. It performs no tool execution and
  confers no authority or capability promotion.
- The composition now emits normalized extension and leave-one-family-out
  states, combines learned disagreement risk with host risk monotonically,
  supports digest-pinned checkpoints and resource-governed inference, and has
  a source-disjoint aligned tensor dataset contract. These remain engineering
  mechanisms, not validated improvements.
- On 2026-10-11, the 36-record internal smoke corpus was materialized into 12
  train, 12 validation, and 12 test records with all 12 categories in each
  split. A one-epoch, 34,528-parameter readiness run completed and reduced
  held-out loss from 5.6282 to 5.4858, but exact-match and format-compliance
  rates were both 0%. The smoke quality gate failed and promotion remains
  forbidden.
- The same hash-bound 34,528-parameter candidate was subsequently resumed to
  12 total epochs and 24 optimizer steps. Validation loss reached 3.3812 and
  untouched-test loss reached 3.4746 with no category regressions. Format
  compliance reached 66.7%, but exact match and every workflow exact score
  remained 0%; the gate still failed. No post-test tuning was performed.
- After trainer hardening, a fresh 884,480-parameter exploratory candidate used
  validation-only early stopping and restored epoch 18 after 30 attempted
  epochs. Held-out loss was 3.1644 with no category regressions, format
  compliance was 83.3%, and the infrastructure smoke gate passed. Exact and
  tool matches remained 0%, latency increased materially, and the corpus is
  still only 36 records; promotion remains forbidden.
- Controlled synthetic v2 completed 25 fits and 75 protected evaluations with
  a NEGATIVE disposition: its learned router did not beat the registered
  controls. The separately frozen v3 reliability-supervised successor also
  completed 25 fits and 75 protected evaluations and was NEGATIVE; its primary
  full-minus-semantic corrupted selective-accuracy delta was -0.0125 (95%
  paired-bootstrap interval -0.0281 to 0.0031) versus a required +0.0200. Both
  studies recorded zero host-risk monotonicity and transition-legality
  violations. These are controlled synthetic results, not natural-data or JEPA
  validation.
- V4 completed 40 matched cells, 80 expert-bank fits and 120 protected
  evaluations. Its utility-routing primary failed by 6.82 percentage points
  against semantic routing. Mixed-corruption training improved corrupted
  accuracy by 27.42 points and semantic routing beat fixed routing by 10.21
  points, but semantic routing regressed on the reserved OOD shifts and the
  temporal predictor lost to persistence on clean inputs.
- V5 is source- and protocol-prepared for 20 cells, 40 bank fits and 80 protected
  evaluations. It has not run and has no result. EditCLEVR is being added as a
  separate image-edit benchmark; it does not supply temporal or governance data.

## Not present

- A trained Hermes Alpha checkpoint.
- A promoted Prometheus, Perseus, Atlas, Kronos, or Aion checkpoint.
- A completed Hermes base-model comparison and licensing decision.
- Hermes-specific SFT, tool-use post-training, quantization comparison, or Percy
  task evaluation.
- A paid or distributed large-scale training run.
- Aligned rights-reviewed multimodal, temporal, risk, tool-outcome, or native
  super-resolution corpora required for the composed Olympus study.
- An executed EditCLEVR benchmark or JEPA efficacy result. The complete
  revision-pinned snapshot is locally admitted and materialized, and a bounded
  real-payload loader smoke passed, but data readiness is not an experiment.

## Current external-runtime findings

The installed `qwen3:8b` artifact is discoverable, but a minimal 256-context,
one-output-token request returned zero bytes before its 120-second bound.
Ollama's log shows swap exhaustion during tensor loading and abort when the
bounded client closed. The 8B configuration therefore fails the local RAM gate.

The separately installed `qwen3:0.6b` Q4_K_M artifact loaded in 15.743 seconds
and returned exact output through the real Olympus adapter in 0.118 seconds of
model-reported time. It was unloaded before training. It is a verified local
provider baseline, not an Olympus-trained checkpoint.

The deep smoke transformer reduced held-out loss with no category regressions,
and its int4/int8 artifacts preserved loss within 2%. It nevertheless scored
0% exact task completion and 0% tool exact match. The promotion engine returned
`NOT_PROMOTED`. Exact measurements are in `OLYMPUS_MODEL_FOUNDRY_LEDGER.md`.

## 2026-10-18 canonical completion correction

This append-only correction supersedes the earlier launch-time statements that
V5 had not run and that no EditCLEVR benchmark had executed. It does not alter
the preserved V1-V4 outcomes above.

- V5 subsequently completed its frozen 20 cells, 40 bank fits, and 80 protected
  evaluations on Kaggle CPU. The provenance/arithmetic verifier passed. The
  routing primary remained unsupported: selected routing was effectively tied
  with core semantic routing on novel shift and was 5.87 percentage points
  worse under corruption. The temporal primary was supported within this
  controlled synthetic study: residual clean MSE was 0.01457 versus 0.07133 for
  the V4 control, a 79.58% reduction, and it also beat persistence. The
  hash-bound remote archive has not been retrieved into this repository, and
  there is no independent retraining, external-benchmark, or promotion claim.
  Canonical details are in `research/olympus_irp_successor_v5/RESEARCH_STATE.json`.
- EditCLEVR V1 subsequently completed five frozen training seeds and 20
  protected reports on the revision-pinned public synthetic image benchmark.
  Its verifier passed, but the learned router tied the selected control in four
  seeds and was marginally worse in one, missing the preregistered +2 percentage
  point shifted-data threshold. The result is NEGATIVE. Its hash-bound remote
  archive has not been retrieved, no independent reproduction exists, and the
  study does not evaluate temporal prediction or the six-family contract.
  Canonical details are in `research/olympus_editclevr_v1/RESEARCH_STATE.json`.

## 2026-10-18 consolidated publication artifact

A new successor manuscript now synthesizes the complete preserved routing and
temporal sequence without changing any predecessor outcome. It reports five
learned-routing primaries as unsupported, retains the narrow V4 secondary gains,
and reports the V5 residual temporal primary as supported only for its frozen
synthetic dynamics task. The missing V5 and EditCLEVR remote archives,
descriptive-interval boundary, and absence of independent reproduction are
prominent limitations.

The seven-page PDF and self-contained arXiv source archive passed repository and
clean-extraction compilation, embedded-font and unresolved-reference checks, and
full-page visual inspection. The exact final repository state passed 595 tests
with two expected skips outside the sandbox boundary. The preceding sandboxed
attempt is retained: 520 passed, two skipped, 14 failed, and 61 errored because
macOS `sysctl` memory inspection was denied and the resource governor failed
closed. The artifact is `ARXIV_READY_NOT_SUBMITTED`; no external upload,
submission, endorsement, model promotion, scientific run, protected evaluation,
or paid compute occurred. Canonical publication state and hashes are in
`research/olympus_consolidated_2026/RESEARCH_STATE.json` and
`research/olympus_consolidated_2026/RELEASE_RECEIPT_20261018.json`.

## Naming rule

No artifact may use Hermes, Prometheus, Perseus, Atlas, Kronos, or Aion as its
model identity until a real immutable checkpoint exists and passes the
family-specific evaluation and deployment gates.

## 2026-10-19 flagship engineering v2 successor

The user authorized changes to the six flagship engineering implementations.
The v2 protocol was frozen before implementation. New opt-in mechanisms cover
tool-view agreement (Hermes), causal trust-radius adapters (Prometheus),
resident-set expert routing (Perseus), patch evidence exclusion (Atlas), evolving
residual rollouts (Kronos), and affordable authorized transitions (Aion). The
Olympus core exposes their risk terms. Separate additions implement FFN parameter
growth, a JEPA-style latent adapter, tensor-storage accounting, and inference
leases that unload to meta. These are candidate combinations, not established
scientific novelty or target-scale trained models.

Prometheus is corrected to the user-specified 12–20B target range, not 128B.
Kronos retains its 8T target, requiring 4 TB of raw backing weights even at 4 bits
before metadata. Lazy residency does not create absent weights or remove their
backing-storage requirement. Atlas and Aion target counts remain unspecified.

The old seven-model receipt retains its exact bytes and is now explicitly
historical, not a verification of changed source. Target-specification identity
was refreshed in the mutable readiness baseline; its exact verifier still
reports 0/6. Prior scientific dispositions above are unchanged.

The local numerical qualification was refused before importing Torch: free
memory was 65,863,680 then 56,442,880 bytes, and pageouts rose from 403,263 to
403,280. The retained refusal is
`research/flagship_engineering_v2/RESOURCE_BLOCKED_20261019.json`. Earlier
unsandboxed qualification approval reviews timed out twice; those commands did
not execute. Static and tensor-free checks are recorded separately in the v2
receipt. CPU/CUDA Kaggle notebooks are engineering preparations only; neither
was uploaded or run. There was no scientific training, protected evaluation,
paid compute, promotion, or change to a predecessor outcome.

## 2026-10-19 authorized Kaggle qualification attempt

The user subsequently authorized Kaggle execution of the bounded flagship,
parameter-growth and ghost-residency qualification. The source is now optionally
embedded in a single hash-pinned notebook, with separate source/device working
directories and visible retained failure logs. A first local build failed on a
missing string-concatenation operator; that failure was retained and repaired.
The subsequent metadata/archive/notebook checks passed 35 tests; these did not
execute tensors. Ruff, strict MyPy (166 files), and exact 0/6 readiness also passed.

A private Kaggle CPU draft was created with Internet off. Its embedded editor
remained uninitialized after reload and a direct official-resource check, leaving
Import Notebook, Run All and Start session disabled. No source was uploaded, no
session started, no CPU/CUDA numerical tests or optimizer steps executed, and no
other user's notebook was modified or stopped. CPU/CUDA qualification and the
full release suite remain pending. The local memory and swap refusals are still
preserved. See `research/flagship_engineering_v2/KAGGLE_EDITOR_BLOCKED_20261019.json`.

## 2026-10-19 downloaded Kaggle CPU engineering qualification

After editor recovery, private offline Kaggle version 359942456 passed 28 CPU
regressions (one CUDA test deselected) and eight artificial optimizer steps.
The six-family integrated model fixture had 6,563 parameters; its grown Hermes
FFN had 416 rather than 280 parameters. The smoke covered all 191 trainable
parameter tensors, retained finite gradients, excluded the absent Perseus expert,
and passed exact checkpoint-output reload. Qualification took 14.965 seconds
inside the unchanged 120-second budget on Python 3.13.15 / Torch 2.11.0+cpu.

The recovery archive was downloaded; its external SHA-256, manifest, every member,
runtime source files, protocol, regression source and recorded artifacts were
verified against the checkout. Parsed JUnit confirms zero failures/errors.
No numerical repair was needed. This is bounded CPU engineering evidence only,
not target-scale training, efficacy, novelty, independent scientific replication,
or full release qualification. The preceding refusals remain unchanged.

CUDA execution has not started. Kaggle offered only GPU T4 x2, so paired runtime
allocation was referred to the user. The CUDA wrapper now sets
`CUDA_VISIBLE_DEVICES=0` for all qualification children; that packaging change
passed all 35 tensor-free runner/archive tests. Actual training and benchmarking
remain unauthorized in this engineering phase. See
`research/flagship_engineering_v2/KAGGLE_CPU_VERIFIED_20261019.json`.

## 2026-10-19 approved CUDA allocation, platform-blocked submission

The user approved the free T4 x2 allocation with computation restricted to GPU 0.
The unchanged CUDA r2 notebook was selected with Internet off and its child
environment `CUDA_VISIBLE_DEVICES=0` verified. Save & Run All was rejected by
Kaggle's maximum two batch-GPU sessions limit. Manage active sessions instead
showed zero active events, an unresolved platform-status contradiction. One
refresh and same-source retry produced no saved version; version history remained
Starting Fresh and the editor stalled on Loading with interactive execution
disabled. No other project's session was stopped or modified.

This is an access/submission blocker, not a failed or passed CUDA regression.
No CUDA output receipt exists and no CUDA numerical result is accepted. The
downloaded CPU evidence and all predecessor receipts remain unchanged. No runtime
or test source, budget, resource guard, scientific outcome, or promotion status
changed. Actual training and benchmarking have not started. See
`research/flagship_engineering_v2/KAGGLE_CUDA_BLOCKED_20261019.json`.
