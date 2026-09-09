# Olympus Reality Ledger

Observed on 2026-09-06 from the repository and local runtime. Status terms are
literal; no model-family capability is inferred from architecture code.

## Real and verified

- Cognitive research demos: executable Python implementations with automated
  tests. These establish engineering behavior only.
- Foundry registry: durable dataset, experiment, checkpoint, evaluation, model,
  and append-only evidence records backed by SQLite.
- Foundry golden path: real maximum-likelihood character-bigram training,
  held-out baseline comparison, checkpoint hashing, export, promotion, restart
  recovery, generation, and tamper rejection.
- Stable model interface: model listing and non-streaming chat completion API.
- Local provider discovery: the dated operator ledger reports discovery of a
  `qwen3:8b` GGUF Q4_K_M artifact with 8.2B parameters and
  completion/tool/thinking capabilities. No hash-bound raw manifest or resource
  snapshot was retained, so this is historical rather than current evidence.
- Deep Foundry controls: licensed instruction-record schema, immutable
  split/manifest hashes, deduplication, contamination and PII rejection,
  exclusive RAM/swap governor, real tiny causal SFT, LoRA, nibble-packed 4-bit
  QLoRA, checkpoint resume, held-out category evaluation, int4/int8
  quantization, and fail-closed hash-bound candidate-promotion gates. A
  signature verifier now checks evaluation, quantization, license-review and
  serving attestations against an operator-pinned trust policy. This is verified
  protocol machinery, not evidence that independent runners or qualified models exist.
- Typed task/tool scoring, matched-seed and matched-budget comparisons that
  demote unsupported mechanisms, and resource admission with cancellation.
- A locally built Forge API/web control surface with provenance, live admission,
  job lifecycle, start, cancel, and retry. It is not deployed.
- Shared family contracts and adapter-decoder code plus executable Hermes,
  Olympus-Atlas, Prometheus, Perseus, Kronos, and Aion reference components.
  Focused tests exercise real PyTorch gradient paths and scoped deterministic
  validation/gating failures; most state, trust, and receipt stores are
  process-local. The bounded release-facing record is
  `evidence/model_family_smoke_20260906.json`; it marks every checkpoint
  non-qualifying and unpromoted. This is software evidence, not family-level
  empirical evidence.
- A fixed synthetic six-role contract replay in
  `olympus/models/composition.py`. It is a two-phase, process-local path that
  exposes an action and execution manifest before execution, then requires a
  host-registered Aion approval bound to that manifest. Retrieval and final
  response requests are explicitly public and hash-bound to their applicable
  authorized views; Hermes is additionally limited to frozen-protocol evidence.
  Runtime-secret HMACs derive public pending/scope IDs and a private reservation
  token protects scoped Perseus operations. After the gate, Perseus finishes
  before Kronos and Hermes run; closed post-gate failure/budget outcomes are
  audited through Aion STOP, with calls charged before invocation and completed
  work cached only for same-process recovery. The manifest includes a
  host-asserted executor profile and only permits tools declared non-material.
  The sanitized replay record is
  `evidence/model_composition_smoke_20260906.json`. This proves only bounded
  contract interoperability; it does not prove sandboxing, actual absence of
  material effects, crash durability, model quality, scientific validity, or
  promotion eligibility. The declared non-material policy is not enforcement
  of executor effects.

## Implemented but not a trained Olympus model

- `HermesNano`: deterministic interpretation-and-memory integration fixture. It
  has no trained weights and must not be represented as Hermes Alpha.
- JEPA and transform demos: small synthetic training routines for testing
  research mechanisms, not general-purpose model checkpoints.
- The family modules under `olympus/models/`: source-level components initialized
  and optimized from scratch on synthetic fixtures for contract and
  training-path verification. None is a rights-cleared, benchmark-qualified,
  or serving-promoted family model. Hermes has a shared-workspace output bridge
  and Prometheus resolves authorized trusted-receipt evidence. One fixed
  synthetic path now composes all six roles, but no general-purpose,
  configurable, crash-durable, sandboxed, or autonomous end-to-end runtime
  exists. HMAC-bound in-memory identifiers and capabilities improve isolation;
  they are not durable credentials or evidence of a secure deployment.

## Not present

- A qualifying trained Hermes checkpoint.
- A promoted Prometheus, Perseus, Atlas, Kronos, or Aion checkpoint.
- A completed Hermes base-model comparison and licensing decision.
- Hermes-specific SFT, tool-use post-training, quantization comparison, or Percy
  task evaluation.
- A paid or distributed large-scale training run.

## Current external-runtime findings

The dated operator ledger reports that the installed `qwen3:8b` artifact was
discoverable, but a minimal 256-context,
one-output-token request returned zero bytes before its 120-second bound.
Ollama's log shows swap exhaustion during tensor loading and abort when the
bounded client closed. The 8B configuration therefore fails the local RAM gate.

It separately reports that `qwen3:0.6b` Q4_K_M loaded in 15.743 seconds and
returned exact output through the real Olympus adapter in 0.118 seconds of
model-reported time before being unloaded. No hash-bound raw request/response,
resource snapshot, or Ollama log was retained for either observation. Treat
them as historical operator reports, not current reproducible evidence or an
Olympus-trained checkpoint.

The deep smoke transformer reduced held-out loss from 5.78961 to 3.90434 with no
category regressions, and its int4/int8 artifacts preserved loss within 2%. It
nevertheless scored 0% exact task completion and 0% tool exact match. Both
quantization quality gates are false and the promotion engine returned
`NOT_PROMOTED`. The three-seed packing ablation produced no distinct condition,
so its mechanism claim is explicitly demoted. Exact measurements are in
`OLYMPUS_MODEL_FOUNDRY_LEDGER.md`.

## Naming rule

Source modules may use Hermes, Prometheus, Perseus, Olympus-Atlas, Kronos, or
Aion as architecture-role codenames. No external model, checkpoint, or serving
identity may use one until a real immutable checkpoint passes the
family-specific evaluation and deployment gates.
