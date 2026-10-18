# Olympus Model Foundry Ledger

Run date: 2026-08-21. Host: Apple M4, 10 CPU cores, 8 GPU cores, 16 GB unified
memory, macOS 26.5.2. Status words are literal: **REAL**, **PLANNED**,
**BLOCKED**, and **FAILED**.

## FOUNDATION

**REAL.** Fresh-root validation passed: Ruff; strict MyPy across 65 source files;
72 tests; branch coverage 87.90%; Python dependency audit with no known
vulnerabilities; six web tests; zero npm audit findings; production web build.

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
The full-SFT run used 90,816 parameters, four epochs, eight optimizer steps, and
0.767 seconds. Validation loss improved 5.69029 → 4.45411. Checkpoint SHA-256:
`59d0fed6ac4938428c967f227545cd271d3eed77449e76781786745fa2936f7d`.

LoRA trained 7,376 of 98,192 parameters for three epochs; validation loss
improved 4.45411 → 4.18589. QLoRA trained 7,376 parameters over genuinely
nibble-packed symmetric 4-bit bases; validation loss improved 4.44891 →
4.18467. Both emitted adapter-only artifacts bound to the exact base and dataset
hashes. Mixed precision was requested, safely unavailable on CPU, and recorded
as ineffective rather than claimed.

The RAM governor held an exclusive model/training lock, required at least 2 GiB
available memory, rejected swap above 90%, and recorded pre/final memory. Full
SFT peak RSS was 339,263,488 bytes; swap did not increase during the run.

## POST-TRAINING

**REAL:** SFT, LoRA, and QLoRA infrastructure. **BLOCKED:** preference
optimization, rejection-sampled self-training, distillation, tool-use
specialization, reasoning specialization, and safety tuning because their
licensed datasets, verifiers, or authorized teachers do not yet exist. Concrete
admission and exit gates are in `research/POST_TRAINING_STAGES.md`.

## EVAL

**REAL negative result.** Twelve untouched test tasks cover every category and
three Percy-compatible workflow groups: tool workflow, result comparison, and
multi-step planning. Full SFT held-out loss improved 5.73703 → 4.50894 with zero
category regressions. LoRA improved 4.50894 → 4.23669; QLoRA improved 4.50894 →
4.23407. However, exact-match, structured-format compliance, and every workflow
exact score were 0%. All three failed the capability smoke gate. Loss reduction
is not treated as assistant competence.

## QUANTIZATION

**REAL.** Weight-only symmetric int8 reduced model storage 373,802 → 115,061
bytes (69.22%) with loss change −0.00645%. Nibble-packed int4 reduced it to
69,813 bytes (81.32%) with loss change −0.25035%. Both round-tripped into the
runtime, enforced the 192-token context boundary, and stayed within the 2% loss
gate. Int4 startup was 6.25 ms in the measured process. Tool exact match was
0%, so quantized promotion remains blocked regardless of numerical fidelity.

## PROMOTED MODELS

**None.** `hermes-alpha` returned `NOT_PROMOTED`. Passed gates: checkpoint
identity, dataset identity, and loss/no-regression. Failed gates: approved
external-base license binding, training scale, held-out scale, task quality,
quantized tool reliability, fresh-process serving for this exact checkpoint,
and hash-bound model card. No release manifest was emitted.

The existing `FoundryVerificationBigram` registry entry is a lifecycle verifier
and is not part of the reserved model family. Prometheus, Perseus, Atlas, and
Kronos are **PLANNED ONLY** in `research/FUTURE_MODEL_FAMILY_ROADMAPS.md`.

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

## 2026-10-11 ENGINEERING ADDENDUM

**REAL reference implementations, NOT PROMOTED MODELS.** All six family
contracts now have bounded differentiable PyTorch references. Hermes exposes
language, tool, and abstention heads; Prometheus exposes domain-adapter and
evidence routing; Perseus performs sparse top-k token-to-expert dispatch; Atlas
returns multimodal patch grounding and retrieval embeddings; Kronos returns
transition, reward, value, uncertainty, and multi-step planning outputs; Aion
ranks only host-allowed research transitions and forces STOP at its risk gate. The
full local Python gate passed 171 tests with 87.13% branch coverage. This does
not change the `PROMOTED MODELS: None` verdict above.

**REAL JEPA upscaler engineering upgrade, efficacy UNESTABLISHED.** The v3
mechanism uses masked high-resolution pixel-unshuffle targets, multiscale
dilated prediction, frequency and SSIM losses/metrics, component loss receipts,
and digest-verified checkpoint reload. Its zero-initialized reconstruction head
still reproduces bicubic at initialization. No rights-reviewed natural-image
campaign, matched modern baseline matrix, multi-seed confidence interval, or
independent reproduction was run, so superiority and promotion remain absent.

**REAL composed-system run path, scientific execution BLOCKED ON DATA.** The
reference system now verifies a hash-bound aligned dataset, enforces model/data
compatibility and resource admission, trains with deterministic bounded steps,
evaluates a source-disjoint split, and writes an immutable checkpoint and report.
The report includes routing entropy/utilization, leave-one-component-out shifts,
STOP behavior, labeled risk calibration, and fixed risk-coverage points. No
rights-reviewed aligned Olympus corpus is present, so no scientific campaign or
promotion has been performed.

## 2026-10-11 TRAINING CONTINUATION

**REAL bounded training, NOT A PROMOTED OLYMPUS MODEL.** The exact one-epoch
34,528-parameter instruction candidate was resumed from checkpoint SHA-256
`db69d98c4d517e333149046a5028d9893567939894f7a9498656e1c77652be6a`
with its optimizer and generator state intact. It completed 12 total epochs and
24 optimizer steps on the immutable 12-record training split. Validation loss
improved from 5.46788 at resume to 3.38116; every recorded epoch reduced
validation loss. The final checkpoint SHA-256 is
`cd1cdae3016f6d7d063243b3e1f0a1297c2c560a6186d4f1ffb3cd18991ec610`.

The untouched 12-record test evaluation improved loss from the deterministic
untrained baseline 5.62821 to 3.47457 with zero category regressions. Format
compliance rose to 66.67%, but exact match and all workflow exact-match scores
remained 0%. The smoke gate therefore failed and the checkpoint is ineligible
for promotion. No additional tuning was performed after inspecting the
protected test result.

## 2026-10-11 IMPROVED EXPLORATORY CANDIDATE

**REAL TRAINING AND SMOKE PASS; NOT HERMES PROMOTION.** The trainer was first
hardened with configuration-hash run identities, overwrite refusal, strict
resume compatibility, optimizer-step warmup plus cosine decay, early stopping,
and restoration of a mutually consistent best model/optimizer/generator state.
A fresh 884,480-parameter candidate (width 128, four layers, eight heads, seed
20261012) was then trained without using protected-test feedback for stopping.
It attempted 30 epochs, early-stopped after a 12-epoch validation plateau, and
restored epoch 18 / optimizer step 36. Validation loss improved 5.75673 →
2.96258. Checkpoint SHA-256:
`e95ff77e5465707a49dbd08b43077108bbff373979fb834e1c29593015102eaa`.

Its single held-out evaluation reduced loss 5.75433 → 3.16436 with zero
category regressions, 83.33% format compliance, and nonempty outputs for all
workflow examples. It passed the bounded infrastructure smoke gate. Exact match
and tool exact match remained 0%, median generation latency rose to 615.52 ms,
and the dataset contains only 12 training and 12 test records. Therefore this
is a better plumbing candidate, not evidence of general task competence or a
promotable Hermes model.

Int4 and int8 derivatives passed the quantization quality gate. Int4 reduced
weight storage 86.04% (3,556,754 → 496,699 bytes) with a −0.064% measured loss
change; int8 reduced it 73.60% with a +0.012% loss change. Tool exact match
remained 0%, so quantization does not alter the promotion verdict.

## 2026-10-11 CONTROLLED RESEARCH OUTCOMES

**NEGATIVE v2 and v3; invariants retained.** Controlled fusion v2 and the
separately frozen reliability-supervised shrinkage v3 each completed 25 fitted
checkpoints and 75 protected evaluations. The v3 primary full-minus-semantic
corrupted selective-accuracy delta was -0.0125 with a 95% paired-bootstrap
interval of [-0.0281, 0.0031], below the preregistered +0.0200 effect. Both
studies recorded zero host-risk monotonicity and transition-legality violations.
The results are controlled synthetic evidence only and do not establish
natural-data fusion, JEPA efficacy, promoted model capability, or independent
reproduction.
