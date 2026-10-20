# Six flagship engineering successor

The six flagship families are Hermes, Prometheus, Perseus, Atlas, Kronos, and
Aion. Prometheus now records the user's **12–20B parameter range**, correcting
the former 128B specification. Kronos remains an **8T target**. Hermes retains
the existing 12B target and Perseus the existing 1.024T target. Atlas and Aion
remain unspecified rather than receiving invented sizes. These targets do not
describe the weights currently instantiated by the development configurations.

## Implemented feature candidates

| Family | Successor mechanism | Engineering falsifier |
|---|---|---|
| Hermes | Last-token and pooled-context tool votes; disagreement raises abstention; a host tool mask removes forbidden probability mass | A denied tool receives probability or disagreement lowers abstention |
| Prometheus | Per-token causal adapter routing with disagreement-limited residual norm | A future token changes an earlier logit or the residual exceeds its radius |
| Perseus | Top-k routing restricted to an explicit resident expert set; padded tokens do not execute experts | An absent expert or padded token executes |
| Atlas | Host-selected patch evidence excludes masked keys, values, influence, and gradients | Changing an excluded patch changes outputs |
| Kronos | Residual next-state predictions and evolving planning states; uncertainty accumulates across the horizon | Rollout holds the state fixed or cumulative uncertainty decreases |
| Aion | Exact intersection of host authority and per-transition affordability, with authorized zero-cost STOP | A transition exceeds its budget or STOP grants new authority |
| Olympus | Named family evidence diagnostics raise a monotone risk envelope before the final Aion decision | Increased host risk lowers effective risk or a forced STOP exposes decisions |

These mechanisms are implemented in `olympus/models/flagship.py`. Instantiate
`default_flagship_config()` for artificial tests, or explicitly set
`architecture_revision="flagship_v2"` in an `OlympusSystemConfig` and construct
with `build_olympus_system`. Training and checkpoint loading use this factory.
The default remains `reference_v1`, keeping existing configurations explicit.
The v2 core additionally accepts `allowed_tools`, `resident_experts`, `patch_mask`,
`transition_costs`, and `remaining_budget` for constrained inference. These extra
inputs must be present in a future training dataset before their benefit is tested.

The risk contributions are **uncalibrated diagnostics**. Held-out risk calibration,
abstention cost, task accuracy, and compute comparisons remain required. Finite
gradients and code execution do not establish feature usefulness.

## JEPA parameter growth

`olympus/models/parameter_growth.py` is distinct from the existing image upscaler.
`widen_feedforward` duplicates hidden units and divides outgoing weights by their
replication counts, preserving the initial function and input gradient within
floating-point tolerance. It returns a new block and a parameter-count receipt,
admits the combined old/new weight budget, and requires rebuilding the optimizer.
It supports a documented Linear/activation[/zero-dropout]/Linear block, not
arbitrary transformer width expansion or optimizer-trajectory preservation.

`JEPAParameterGrowth` adds an identity-initialized residual adapter, an online
context encoder, a frozen EMA target encoder, and masked latent prediction.
Target positions are removed from the prediction context; targets are detached.
This is latent-position exclusion: it cannot remove target information already
mixed into other positions by an upstream encoder. A raw-token masking study
must separately encode masked context inputs and freeze that data boundary.
The qualification exercises this path alongside an enlarged Hermes FFN.
Training a larger model still requires new data and compute.

## Lazy ghost parameters

The parameter store now validates a complete plan before disk reads, rejects
duplicates, omitted mandatory shards, forged byte totals, parent symlinks, and
nonfinite shard tensors. Its manifest records actual tensor backing-storage bytes.
Serialized bytes and tensor residency are conservatively accounted separately.

Use `store.inference_lease(meta_module.eval(), plan)` inside
`torch.inference_mode()` to load one complete inference module and return it to
meta when the lease exits, including on an exception. Evicting a cache entry alone
does not unload weights attached to a module. Leases require exclusive ownership;
callers must not retain parameter references or attach an optimizer. Activation,
temporary read-buffer, allocator, and process memory remain the outer governor's
responsibility. This is a CPU shard lease, not distributed training or a CUDA
offloading scheduler. Resident-set routing and lease ownership are building blocks
for a later expert scheduler; they are not automatically a full sparse serving engine.

`estimate_capacity` computes total/backing, active, resident, training-state, and
activation/KV budgets without allocating weights. At 4 bits, 12–20B parameters
require 6–10 GB of raw backing weights; 8T requires 4 TB. These figures exclude
quantizer metadata. Multiple GPUs do not automatically pool their memory.

## Kaggle handoff

Build a new immutable output directory:

```sh
python -m notebooks.build_flagship_runner --output-dir artifacts/flagship-kaggle --device cpu --embed-source
```

The notebook and `olympus-source.tar.gz` are paired by SHA-256. `--embed-source`
places the exact archive inside the notebook, so only one private notebook upload
is needed. Without it, attach the archive as a private input. Keep internet
disabled and preserve the platform's Torch. Each source/device pair uses a
distinct working directory to avoid reusing another qualification's outputs.
The notebook runs `scripts.qualify_flagship_v2`, retains regression XML/logs and
failure receipts, and exports a recovery archive. CUDA must be explicitly selected
and available; no CPU fallback can count as GPU verification.

The 2026-10-19 execution attempt stopped before upload: Kaggle's private draft
remained on `Editor Loading`, with import and execution disabled. This is an
access/UI blocker, not a numerical test failure or pass. The full retained record
is `research/flagship_engineering_v2/KAGGLE_EDITOR_BLOCKED_20261019.json`.

After editor recovery, version 359942456 passed 28 CPU regressions and eight
optimizer steps. The downloaded archive, source/protocol/test identities and
every output hash were verified. All 191 trainable parameter tensors were covered
with finite gradients and exact output reload. This qualifies the tiny CPU
engineering configuration only; CUDA and the complete release suite remain pending.
The CUDA wrapper sets `CUDA_VISIBLE_DEVICES=0`; provisioned hardware can still
contain two GPUs, and paired allocation requires explicit authorization under
the existing one-device budget. Current evidence is in
`research/flagship_engineering_v2/KAGGLE_CPU_VERIFIED_20261019.json`.

Qualification uses at most four examples per fixture and eight optimizer steps,
one CPU thread, and a 120-second cooperative budget. It verifies new feature
contracts, growth, finite updates, and checkpoint reload. The optimizer smoke
rotates the absent Perseus expert, checks that absent
experts receive no gradients, and requires full trainable-parameter coverage
across the eight updates. Finite or zero gradients do not prove useful learning.
It does not benchmark an LLM. A subsequent scientific campaign still needs revision-pinned base weights,
rights-reviewed train/development/protected splits, a frozen benchmark protocol,
controls, seed counts, effect criteria, and an admitted hardware budget.

## Novelty and next study

The new combinations are candidate contributions. Function-preserving widening
comes from [Net2Net](https://arxiv.org/abs/1511.05641); masked latent prediction
and EMA targets build on [I-JEPA](https://arxiv.org/abs/2301.08243).
[Counterfactual routing analysis](https://arxiv.org/abs/2605.07260) already studies
alternative expert assignments. Neither sparse execution nor abstention alone
establishes novelty. The proposed differentiator is a measured end-to-end contract
linking evidence exclusion, resident expert availability, evolving temporal risk,
and host-authorized budget decisions. Its contribution requires matched ablations,
fresh data, actual memory/latency measurements, and independent reproduction.

Numerical verification status is recorded in the v2 research receipt. Historical
seven-model receipts retain their exact bytes and are explicitly historical;
they do not certify this successor's changed source.
## CUDA allocation follow-up, 2026-10-19

Paired T4 allocation was explicitly approved with computation restricted to GPU 0.
Kaggle rejected saved-run submission at its two batch-GPU session cap. A bounded
refresh/retry did not create a version and the editor returned to Loading. CUDA
therefore remains platform-blocked, not numerically failed or qualified. The
accepted CPU result is unchanged. See
`research/flagship_engineering_v2/KAGGLE_CUDA_BLOCKED_20261019.json`.

