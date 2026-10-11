# BU1LD model target specifications

Status: **architecture and mechanism development only**. Parameter counts and
roles below are targets, not existing checkpoints or capability claims.

| Family | Target parameters | Intended role | Current evidence boundary |
|---|---:|---|---|
| Hermes | 12B | Private local-first assistant | No qualified 12B checkpoint |
| Prometheus | 128B | Adaptable research, coding, and design specialist | No qualified 128B checkpoint |
| Perseus | 1.024T | Sparse, globally accessible generalist | No qualified trillion-parameter checkpoint |
| Atlas | Unspecified | Deep multimodal and long-context intelligence | “Sentience” is not claimed or treated as a measurable engineering result |
| Kronos | 8T | Long-horizon general-intelligence research system | No qualified 8T checkpoint or AGI evidence |

Olympus is the umbrella architecture, not an additional parameter target. Aion
is retained as a historical/deferred family because it was not included in this
revised lineup. “Perfect” is an aspiration: promotion requires the explicit
held-out gates in `olympus.models.specifications` and independent review.

## Implemented pathways

### Encoder → quantization

`olympus.models.encoder_quantization` now provides a trainable encoder followed
by masked residual vector quantization. Multiple codebooks progressively encode
the residual, inactive tokens receive no codes, the straight-through path keeps
encoder gradients, and commitment, codebook, error, and utilization signals are
returned for training and evaluation. This is an executable pathway; it is not
a trained tokenizer or evidence that quantization preserves downstream quality.
`olympus.training.encoder_quantization` adds a hash-bound, source-disjoint
rate-distortion training and held-out evaluation runner. It writes immutable
checkpoint/report pairs and never authorizes promotion automatically.

### JEPA upscaler v2 mechanism

`olympus.upscaler.jepa` predicts an EMA target representation conditioned on
normalized image coordinates, radius, and time. A zero-initialized pixel-shuffle
head begins at the deterministic bicubic baseline. Its objective separates
reconstruction, edge, joint-embedding, latent-geometry, and anti-collapse
variance terms. This is a new successor mechanism, not a reinterpretation of
the retained negative synthetic upscaler result; it still needs a frozen,
rights-reviewed multi-seed comparison against bicubic, pixel-only, and strong
super-resolution baselines.
`olympus.upscaler.training` adds a provenance-bound training runner and a held-out
PSNR/edge comparison against bicubic. A completed run retains negative results
and is explicitly marked development-only.

### Spatiotemporal geometric lazy parameter residency

`olympus.models.spatiotemporal_lazy` describes parameter shards with immutable
hashes, unique tensor ownership, 3D bounds, time intervals, resolution ranges,
priority, and mandatory/global status. A query projects position using velocity
and a lookahead horizon, ranks relevant shards, respects a byte budget, rejects
path traversal/symlinks/tampering, and loads selected tensors through an LRU
cache. This makes residency planning space-, time-, motion-, and scale-aware; it
does not make inactive parameters free, prove quality, or solve distributed 8T
training and inference.

The store can also materialize selected shards directly into meta-device PyTorch
modules while reporting every tensor that remains unloaded. It rejects unknown
names, shape/dtype mismatches, incomplete materialization when completeness is
required, path escapes, symlinks, byte changes, and hash changes.

## Command line entry points

- `olympus bu1ld specs`
- `olympus bu1ld train-encoder-quantizer MANIFEST OUTPUT`
- `olympus bu1ld train-jepa-upscaler MANIFEST OUTPUT`
- `olympus bu1ld lazy-plan MANIFEST BUDGET_BYTES X Y Z TIMESTAMP RESOLUTION`

The two training commands produce development checkpoints and evidence reports;
neither command promotes a model or upgrades a family capability claim.

## Gates before scaling

1. Integrate a real, provenance-bound multimodal dataset and encoder training
   protocol with train/development/protected splits.
2. Measure rate–distortion and task quality for residual VQ against float,
   scalar int8/int4, and unquantized encoder baselines.
3. Run the JEPA upscaler successor under a preregistered multi-seed protocol on
   rights-reviewed natural images; preserve the older negative outcome.
4. Replay lazy-shard decisions under recorded trajectories and compare latency,
   peak memory, I/O, miss rate, and downstream quality with eager and ordinary
   layer-wise loading.
5. Only after those gates, validate sparse active-parameter scaling on smaller
   models before requesting 12B, 128B, 1.024T, or 8T compute.
