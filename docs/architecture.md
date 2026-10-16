# Architecture

Olympus is organized around six collaborating layers:

1. Core cognition.
   The `olympus.core` package implements multimodal artifacts, latent workspace state, interpretive branching, representation selection, learned transforms, JEPA prediction, retrodiction, outcome scoring, compute allocation, task division, verification, tracing, and safety guards.
2. Forge compilation and runtime.
   The `olympus.forge` package defines the behavior language, natural-language compiler, graph execution runtime, and experiment controller.
3. Memory and data.
   The `olympus.memory` and `olympus.data` packages provide SQLite-backed memory, source registries, manifests, ingestion, chunking, sharding, and local retrieval.
4. Model Foundry.
   The `olympus.foundry` package owns immutable dataset materialization,
   experiment state, checkpoint hashing, held-out evaluation, model promotion,
   append-only evidence, portable export, local generation, and Ollama provider
   integration. The registry uses SQLite foreign keys, WAL journaling, full
   synchronization, explicit transactions, and integrity checks.
5. Model-family references.
   The `olympus.models` package contains a deterministic Hermes Nano integration
   fixture. Despite its historical name, it is not a trained language model and
   is not a Hermes checkpoint.
   It also contains bounded, differentiable reference architectures for all six
   Hermes, Prometheus, Perseus, Atlas, Kronos, and Aion contracts: tool and
   abstention heads, domain adapters, sparse expert routing, multimodal patch
   grounding, temporal world-model planning, and governed research-transition
   routing respectively. These validate interfaces and mechanisms, not
   target-scale capabilities.
   The BU1LD target programme also includes executable research primitives for
   masked residual encoder quantization and hash-bound spatiotemporal parameter
   residency. The latter ranks lazy shards using position, velocity, time, and
   resolution while enforcing containment and content identity.
6. Interfaces.
   FastAPI lives in `olympus.api`, Typer CLI lives in `olympus.cli`, and the React dashboard lives in `apps/forge-web`.

The six references can also be composed through `OlympusReferenceSystem`. Five
family representations are projected into a common, explicitly gated state;
Aion then applies host-supplied transition legality and a deterministic STOP
rule. The resulting decision surface is trainable but has no tool-execution or
promotion authority. See `docs/olympus-system.md` for the contract and loss.

The current implementation is intentionally small-scale but real. The Foundry
verification model proves infrastructure behavior only; future Olympus family
names are not registered until a real checkpoint passes its declared gates.
The image upscaler successor in `olympus.upscaler` uses masked, high-resolution
pixel-unshuffle EMA targets, multiscale spatial/time-conditioned prediction,
and a bicubic residual path. Its evaluation reports PSNR, edge, SSIM, and
frequency errors and its checkpoints can be digest-pinned on reload. These are
mechanisms, not promoted families or target-scale capability evidence.
