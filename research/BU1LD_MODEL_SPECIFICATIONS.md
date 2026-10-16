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
| Aion | Unspecified | Governed research-protocol orchestration | No qualified router/controller checkpoint or autonomous-research evidence |

Olympus is the umbrella architecture, not an additional parameter target.
“Perfect” is an aspiration: promotion requires the explicit held-out gates in
`olympus.models.specifications` and independent review.

## Implemented pathways

### Six-family reference architectures

`olympus.models.reference_families` makes every active family contract
executable at bounded development scale:

- Hermes couples a causal backbone to explicit language, tool-selection, and
  abstention heads.
- Prometheus adds auditable domain-adapter routing and per-token evidence
  classification.
- Perseus uses sparse top-k expert dispatch and exposes routing probabilities,
  selected experts, and its load-balance objective.
- Atlas grounds text in spatiotemporal image patches, returning patch-level
  grounding weights and normalized retrieval embeddings.
- Kronos predicts state transitions, reward, value, calibrated-positive
  uncertainty, and a differentiable multi-step action plan.
- Aion ranks only host-allowed research-stage transitions and applies a
  deterministic risk-triggered STOP rule over the learned router.

The default configurations are intentionally small enough for contract tests.
They validate architecture wiring and gradient flow only; they are not family
checkpoints and do not inherit the target parameter counts in the table above.

### Olympus composition layer

`olympus.models.olympus_system` composes the five capability representations
under an explicit family gate, then sends the fused state to Aion for
host-constrained research-stage routing. Its joint objective covers decisions,
legal transitions, temporal prediction, uncertainty calibration, and gate
balance. A forced Aion STOP removes the exposed decision. This makes the
umbrella architecture trainable and inspectable without granting it execution
or promotion authority.

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

### JEPA upscaler v3 mechanism

`olympus.upscaler.jepa` now predicts masked EMA targets made from a lossless
pixel-unshuffle of the real high-resolution image. The context encoder sees the
same representation of a bicubic baseline, so the target retains subpixel
detail that downsampling would erase. A multiscale dilated predictor is
conditioned on normalized image coordinates, radius, and time. A
zero-initialized pixel-shuffle head still begins exactly at bicubic. The
objective separates robust reconstruction, edge, masked joint-embedding,
latent geometry, anti-collapse variance, Fourier-amplitude, and SSIM structure
terms. This successor mechanism is not a reinterpretation of the retained
negative synthetic result; it still needs a frozen, rights-reviewed multi-seed
comparison against bicubic, pixel-only, and strong super-resolution baselines.
`olympus.upscaler.training` adds a provenance-bound training runner and a held-out
PSNR, edge, SSIM, and frequency comparison against bicubic, block-mask
generation, component-level loss reporting, and digest-pinned checkpoint
reload. A completed run retains negative results and is explicitly marked
development-only.

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
- `olympus bu1ld reference-models`
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
   rights-reviewed natural images; include matched pixel-only and modern strong
   baselines, and preserve the older negative outcome.
4. Replay lazy-shard decisions under recorded trajectories and compare latency,
   peak memory, I/O, miss rate, and downstream quality with eager and ordinary
   layer-wise loading.
5. Only after those gates, validate sparse active-parameter scaling on smaller
   models before requesting 12B, 128B, 1.024T, or 8T compute.
