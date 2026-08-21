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
6. Interfaces.
   FastAPI lives in `olympus.api`, Typer CLI lives in `olympus.cli`, and the React dashboard lives in `apps/forge-web`.

The current implementation is intentionally small-scale but real. The Foundry
verification model proves infrastructure behavior only; future Olympus family
names are not registered until a real checkpoint passes its declared gates.
