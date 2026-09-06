# Olympus Model Foundry Ledger

Run date: 2026-09-01. Host: Apple M4, 10 CPU cores, 8 GPU cores, 16 GB unified
memory, macOS 26.5.2. Status words are literal: **REAL**, **PLANNED**,
**BLOCKED**, and **FAILED**.

## FOUNDATION

**REAL.** Fresh-root validation passed: Ruff; strict MyPy; more than the required
64 Python tests with branch coverage above 85%; Python dependency audit with no
known vulnerabilities; seven web tests; zero npm audit findings; production web
build; sdist/wheel build; and Twine metadata validation. A clean wheel install
outside the repository reproduced the golden path and restart integrity.

The pre-existing durable Foundry golden path was re-run from an empty root. It
registered corpus SHA-256
`6f4823e0503426bda11aa4bffe8357963f3bc97a62ea40c2b69700af166a8eab`,
created checkpoint `ckpt_a6e7cfa9ad55681a` with SHA-256
`a6e7cfa9ad55681a40472f6a4ac3a30e2b8f331d0eb2204768c5999758a2b0ee`,
beat its frozen uniform baseline (candidate perplexity 6.68695 versus 24.0),
exported and served it, recovered after restart, and retained SQLite integrity.
Tampered datasets and checkpoints are rejection-tested. This is a character
bigram lifecycle verifier, not Hermes.

## LOCAL INFERENCE

**FAILED — qwen3:8b.** The installed 8.2B Q4_K_M artifact was tested with
context 256, output allowance 1, temperature 0, streaming HTTP, and a 120-second
bound. It returned zero bytes. Ollama logs show all 37 layers targeted for Metal,
4,643.78 MiB Metal model allocation plus 333.84 MiB CPU model allocation, mmap
disabled for partial Metal offload, only 3.6 GiB free host memory and zero free
swap at admission, then abort during tensor loading when the bounded client
closed. Swap had grown to 2.704 GiB used. Prompt, context, output length,
streaming, and Olympus routing are ruled out; the failure is memory pressure
during load.

**REAL — qwen3:0.6b local baseline.** Apache-2.0 Q4_K_M artifact, 522 MB on
disk, local manifest 751.63M parameters and 40,960 context. At context 256 it
loaded in 15.743 seconds, occupied 567 MB on GPU according to Ollama, and
returned `OK.`. Through the Olympus Ollama adapter it returned exactly
`OLYMPUS_LOCAL_MODEL_OK`; model-reported duration was 0.118 seconds and adapter
process peak RSS was 261,029,888 bytes. It was explicitly unloaded before
training. It is a frozen provider baseline, not an Olympus-trained model.

## BASE CANDIDATES

**PLANNED.** `Qwen/Qwen3-0.6B-Base` is selected for the first external-base
QLoRA experiment because it is a small public pretrained base under Apache-2.0
with 32,768-token context. The revision and weights have not been downloaded in
this run. Qwen3 1.7B, Gemma 3 1B IT, Llama 3.2 1B, and Phi-4-mini-instruct are
documented and deferred or rejected in `research/HERMES_BASE_CANDIDATES.md`.

## DATA

**REAL infrastructure dataset.** `datasets/hermes-smoke/source.jsonl` contains
36 human-reviewed, repository-owned records: 12 train, 12 validation, 12 test.
Every split has instruction, reasoning, mathematics, code, science, tool use,
extraction, planning, agent behavior, self-correction, research, and safety.
Source SHA-256 is
`9cd0e6455b45b40f4d222b6b9dc4d8be35a2fccb80c7793c4cb45615c5ec3e4f`.
The manifest SHA-256 is
`f64a41e9d3d79d5911e5abcf34e635bb76582abbdbb5d858b6405a1921c89285`
and is identical when prepared into different output roots.
Preparation enforces schema, source/license, immutable split hashes, exact and
normalized deduplication, cross-split eight-token contamination rejection,
quality bounds, PII/secret rejection, stable transformations, and versioned
manifests. Test data never enters optimization.

**BLOCKED for Hermes.** The promotion floor is 10,000 reviewed training records
and 1,200 held-out records with at least 100 per category. Current counts are 12
and 12 with one held-out record per category.

## TRAINING

**REAL smoke jobs, not Hermes.** A causal byte-level transformer with masked
prompt loss, sequence packing, category mixing controls, deterministic seeds,
gradient accumulation, clipping, AdamW, validation logging, atomic checkpoints,
optimizer/generator resume state, and checkpoint hashing was trained locally.
The full-SFT run used 145,664 parameters, four epochs, eight optimizer steps.
Validation loss improved 5.81517 → 3.80487. Checkpoint SHA-256:
`e483df22322345a985507944fc81d1ad1bcc81a620cb112f5e7cdd37bf4e375a`.
Resume rejects architecture, optimizer, and data-contract mismatches rather than
silently relabeling loaded weights with caller configuration.

LoRA trained 9,488 of 155,152 parameters for two epochs; validation loss
improved 3.80487 → 3.72659. QLoRA trained 9,488 of 39,056 parameters over
genuinely nibble-packed symmetric 4-bit bases; validation loss improved 3.78839
→ 3.71108. Both emitted adapter-only artifacts bound to the exact base and dataset
hashes. Mixed precision was requested, safely unavailable on CPU, and recorded
as ineffective rather than claimed.

The RAM governor held an exclusive model/training lock, required at least 2 GiB
available memory, rejected swap above 90%, and recorded pre/final memory. The
resumed full-SFT process recorded peak RSS of 323,911,680 bytes. A stricter
medium tier requires 8 GiB available and <=50% swap and was correctly refused.

## POST-TRAINING

**REAL:** SFT, LoRA, and QLoRA infrastructure. **BLOCKED:** preference
optimization, rejection-sampled self-training, distillation, tool-use
specialization, reasoning specialization, and safety tuning because their
licensed datasets, verifiers, or authorized teachers do not yet exist. Concrete
admission and exit gates are in `research/POST_TRAINING_STAGES.md`.

## EVAL

**REAL negative result.** Twelve untouched test tasks cover every category and
three Percy-compatible workflow groups. Full SFT held-out loss improved 5.78961
→ 3.90434 with zero category regressions. Exact match was 0%, format compliance
was 83.33%, and every workflow exact score was 0%. The typed harness separately
scores answers, tool names and arguments, refusals, uncertainty, and Percy
workflows, and separates model, orchestration, and tool failures. The candidate
failed the bounded capability gate. Loss reduction is not competence.

## QUANTIZATION

**REAL.** Weight-only symmetric int8 reduced model storage 593,194 → 171,765
bytes (71.04%) with loss change −0.01360%. Nibble-packed int4 reduced it to
98,357 bytes (83.42%) with loss change −0.76061%. Both round-tripped into the
runtime, enforced the 192-token context boundary, and stayed within the 2% loss
gate. Int4 startup was 15.81 ms in the measured process. Tool exact match was
0%, so both quality gates are false and promotion remains blocked.

## PROMOTED MODELS

**None.** `hermes-alpha` returned `NOT_PROMOTED`. Passed gates: checkpoint
identity, dataset identity, and loss/no-regression. Failed gates: approved
external-base license binding, training scale, held-out scale, task quality,
quantized tool reliability, fresh-process serving for this exact checkpoint,
and hash-bound model card. No release manifest was emitted.

The existing `FoundryVerificationBigram` registry entry is a lifecycle verifier
and is not part of the reserved model family. Prometheus, Perseus, Atlas,
Kronos, and Aion are **SPECIFIED ONLY** in
`research/FUTURE_MODEL_FAMILY_ROADMAPS.md`.

## MATCHED ABLATION

**REAL negative result.** Seeds 17, 31, and 47 compared the reference, simple
SFT, and no-packing conditions at an equal 138,112 parameter-step budget. Every
condition obtained validation mean 5.5031697 (stdev 0.0947228). The packing
mechanism is demoted as `mechanism_not_supported`: this bounded corpus did not
create a distinct condition, so no mechanism claim is made.

## BLOCKERS

1. Download and hash a revision-pinned, license-reviewed Qwen3-0.6B-Base only
   after confirming the local 4-bit training runtime and disk budget.
2. Expand reviewed licensed data to the enforced training and held-out floors;
   retain document/source-level contamination controls.
3. Run the bounded external-base QLoRA experiment and ablate rank, sequence
   length, packing, and dataset mixing against the frozen base.
4. Reach the exact, format, workflow, safety, latency, RAM, and quantization
   gates without category regression.
5. Serve the exact candidate checkpoint through fresh CLI, API, OpenAI contract,
   Ollama-compatible export, and web processes; bind results to its hash.
6. Publish a complete hash-bound model card. Only then may the gate emit a
   release manifest and allow the Hermes identity.

No paid compute, remote training, private-data upload, or false model-family
instantiation occurred.
