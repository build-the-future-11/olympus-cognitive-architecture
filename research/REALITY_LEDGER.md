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

## Naming rule

No artifact may use Hermes, Prometheus, Perseus, Atlas, Kronos, or Aion as its
model identity until a real immutable checkpoint exists and passes the
family-specific evaluation and deployment gates.
