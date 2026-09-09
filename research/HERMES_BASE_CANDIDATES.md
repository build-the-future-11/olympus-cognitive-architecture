# Hermes Alpha Base-Model Decision

Decision date: 2026-08-21. This is a source-backed selection for a future real
Hermes training run. It is not a Hermes checkpoint and does not authorize a
remote or paid job.

## Candidate matrix

Memory figures are engineering estimates for a 16 GB unified-memory Mac and
must be re-measured with the exact revision. "Local result" is the only column
that contains observations from this machine.

| Candidate | License/access | Parameters / context | Local memory and quantization | Training fit | Capability evidence | Restrictions / decision |
|---|---|---|---|---|---|---|
| [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | Apache-2.0; public weights | Card: 0.6B class, 0.44B non-embedding; 32,768 tokens | Estimated BF16 weights ~1.5 GB including embeddings; LoRA ~2–4 GB working set; 4-bit QLoRA ~1.5–3 GB | Best fit: actual pretrained base, small enough for bounded local LoRA/QLoRA, broad adapter ecosystem | Base model has no assistant/tool-use claim; those must be trained and measured | **Selected for the first external-base SFT/QLoRA experiment.** Pin a revision and re-run license review before download. |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) / [Ollama tag](https://ollama.com/library/qwen3/tags) | Apache-2.0; public | Dated operator ledger reports 751.63M parameters and 40,960 context | Historical report: Q4_K_M artifact 522 MB, load 15.743 s, GPU allocation 567 MB at context 256 | Potential inference baseline, not the preferred clean SFT base because it is already post-trained | A local exact-format smoke is reported, but its raw response, provider log, resource snapshot, and full artifact identity were not retained | Treat as a historical local serving observation. Re-identify and hash the exact artifact, then rerun before freezing it as a baseline; do not relabel or fine-tune it as Hermes Alpha. |
| [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | Apache-2.0; public | 1.7B class; 32,768 tokens | Ollama Q4 tag is about 1.4 GB; QLoRA estimated 3–6 GB | Plausible second-stage scale-up after the 0.6B data and evaluation pipeline is credible | Not locally measured | Deferred: higher memory and iteration cost without first proving data quality at 0.6B. |
| [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it) | Gemma Terms; access acceptance required | 1B; card documents 32K context for 1B | Card estimates BF16 weights around 2 GB before runtime overhead | Small, but instruction-tuned rather than a clean base and carries custom terms | Model-card claims only; not locally measured | Rejected for first run: custom terms and weaker provenance simplicity than Apache-2.0 Qwen base. |
| [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/README.md) | Llama 3.2 Community License; attribution, naming, and use terms | 1B class | Estimated QLoRA 2–5 GB; exact revision unmeasured | Technically feasible | Model-card claims only; not locally measured | Rejected for first run: custom license/attribution burden when an Apache-2.0 alternative exists. |
| [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | MIT | 3.8B; 128K context | Estimated local QLoRA 6–10+ GB; long context increases KV/activation pressure | Too close to the machine's safe limit for rapid iteration | Model-card claims only; not locally measured | Deferred: strong license but excessive training risk on the current 16 GB host. |

## Selection and exact next experiment

Use a revision-pinned `Qwen/Qwen3-0.6B-Base` snapshot for the first real
external-base run. Store the upstream commit, every file SHA-256, the full
Apache-2.0 text, tokenizer identity, and immutable dataset manifest. Start with
4-bit QLoRA, rank 8, sequence length 512, micro-batch 1, gradient accumulation
16, deterministic seed, validation every 25 optimizer steps, and a hard RAM/swap
abort. This configuration is a planned run, not an executed result; the current
repository run validates the machinery on a 90,816-parameter transformer only.

Promotion remains impossible until the external base, dataset scale, held-out
quality, quantized tool reliability, and fresh-process serving gates pass.
