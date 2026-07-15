# Architecture

Olympus is organized around five collaborating layers:

1. Core cognition.
   The `olympus.core` package implements multimodal artifacts, latent workspace state, interpretive branching, representation selection, learned transforms, JEPA prediction, retrodiction, outcome scoring, compute allocation, task division, verification, tracing, and safety guards.
2. Forge compilation and runtime.
   The `olympus.forge` package defines the behavior language, natural-language compiler, graph execution runtime, and experiment controller.
3. Memory and data.
   The `olympus.memory` and `olympus.data` packages provide SQLite-backed memory, source registries, manifests, ingestion, chunking, sharding, and local retrieval.
4. Model-family references.
   The `olympus.models` package contains miniature reference families, including a runnable Hermes Nano local assistant.
5. Interfaces.
   FastAPI lives in `olympus.api`, Typer CLI lives in `olympus.cli`, and the React dashboard lives in `apps/forge-web`.

The current implementation is intentionally small-scale but real: every demo exercises functioning code rather than placeholders.

