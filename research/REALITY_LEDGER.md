# Olympus Reality Ledger

Observed on 2026-08-21 from the repository and local runtime. Status terms are
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
- Local provider discovery: Ollama is installed and reports `qwen3:8b`, GGUF
  Q4_K_M, 8.2B parameters, with completion/tool/thinking capabilities in its
  local manifest.

## Implemented but not a trained Olympus model

- `HermesNano`: deterministic interpretation-and-memory integration fixture. It
  has no trained weights and must not be represented as Hermes Alpha.
- JEPA and transform demos: small synthetic training routines for testing
  research mechanisms, not general-purpose model checkpoints.

## Not present

- A trained Hermes Alpha checkpoint.
- A promoted Prometheus, Perseus, Atlas, or Kronos checkpoint.
- A completed Hermes base-model comparison and licensing decision.
- Hermes-specific SFT, tool-use post-training, quantization comparison, or Percy
  task evaluation.
- A paid or distributed large-scale training run.

## Current external-runtime finding

The installed `qwen3:8b` artifact is discoverable, but discovery alone is not a
generation pass. On 2026-08-21, a bounded request using a 1,024-token context and
16-token output allowance received zero response bytes before its 600-second
timeout. Ollama's server log shows the 8.2B Q4_K_M GGUF entering tensor loading,
then aborting when the bounded client closed. Therefore local model discovery is
verified and local generation is failed under the measured environment. This is
an external-runtime failure, not an Olympus checkpoint claim.

## Naming rule

No artifact may use Hermes, Prometheus, Perseus, Atlas, or Kronos as its model
identity until a real immutable checkpoint exists and passes the family-specific
evaluation and deployment gates.
