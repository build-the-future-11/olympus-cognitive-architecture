# Architecture

Olympus is organized around six collaborating layers:

1. Core heuristic and synthetic demonstrations.
   The `olympus.core` package contains multimodal fixtures, workspace state,
   heuristic interpretive branching and representation selection, synthetic
   learned-transform and JEPA-inspired experiments, retrodiction utilities,
   scoring, compute allocation, dependency-aware task scheduling, verification,
   tracing, and policy definitions. These modules are executable prototypes,
   not validated cognitive or neural mechanisms.
2. Forge compilation and runtime.
   The `olympus.forge` package defines a behavior language, keyword-to-graph
   compiler, graph runtime, and experiment controller. External effects occur
   only when callers inject a tool handler or memory store; absent effects are
   reported as skipped.
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

Pantheon under `OlympusC/05_pantheon` is a separate empirical research package,
not a seventh runtime layer. LabOS is an incubation-stage research-operations
tool whose parent-workspace reports are not Olympus scientific evidence. See
`research/RESEARCH_PROGRAM.md` for the maintained paper and claim boundaries.

The current implementation is intentionally small-scale but real. The Foundry
verification model proves infrastructure behavior only; future Olympus family
names are not registered until a real checkpoint passes its declared gates.

## Canonical experimental contract

Every candidate follows one compact interface:

`immutable dataset manifest → frozen run config → content-addressed checkpoint
→ held-out capability report → quantization report → serving verification
→ fail-closed promotion report`.

The capability report separates exact task completion, tool selection, typed
arguments, refusal, uncertainty, Percy-style workflow completion, and failures
originating in the model, orchestration layer, or tool runtime. Comparative
claims require the same registered seeds and parameter-step budget within the
pre-registered tolerance. A mechanism is demoted when removing it does not
reduce the frozen primary metric by the declared minimum effect.

Local execution has two admission tiers. `small` permits one active run and one
queued run with a 2 GiB available-memory floor. `medium` permits one active run,
no queue, an 8 GiB available-memory floor, and at most 50% swap use. Refusal is
an expected safety result when the host does not satisfy a tier.
