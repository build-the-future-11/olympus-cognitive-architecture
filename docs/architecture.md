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
   The `olympus.models` package contains the historical Hermes Nano fixture plus
   executable references for a shared typed/adapter substrate and the Hermes,
   Olympus-Atlas, Prometheus, Perseus, Kronos, and Aion roles. Learned
   components have real PyTorch losses. Deterministic references provide scoped
   ACL, schema/capability, process-local transaction/checkpoint, protocol, and
   approval validation or gating paths. `OlympusReferenceReplay` composes them
   along one fixed synthetic two-phase contract path. Preparation performs
   authorized Atlas retrieval and Prometheus selection, preregisters Aion, and
   returns the proposed action/manifest without executing a tool. Resume first
   validates the host-registered, manifest-bound Aion approval and executor
   profile, then runs Perseus; only after that terminal execution receipt does it
   compute Kronos from host-supplied pre-execution features, run Hermes, audit,
   and reach Aion STOP. The retrieval query and final request are explicit
   public artifact fields whose traces also bind the applicable authorized-view
   hashes. Hermes receives the intersection of authorized evidence and frozen
   protocol evidence, not every otherwise visible workspace item.

   Pending IDs and public execution-scope IDs are domain-separated HMACs under a
   runtime-private secret. A different HMAC-derived scope token remains private
   and reserves the public scope in `TransactionManager`; scoped
   prepare/commit/receipt/rollback operations require that token. This blocks a
   caller sharing the manager from pre-creating or replaying a transaction into
   the approved scope, but it is an in-memory capability rather than durable
   authentication. Model-call use is charged before each attempted family call;
   terminal receipts and post-execution outputs are cached so an in-process
   retry neither re-executes a committed tool nor double-counts cached calls.
   Kronos exceptions/budget exhaustion become closed failed observations, and
   Hermes exceptions/budget exhaustion become closed abstentions; execution or
   Kronos failure also forces Hermes to abstain. Those degraded outcomes are
   audited and reach STOP rather than being reported as success or leaving the
   authorized run pending.

   The path rejects tools that either shared or Perseus policy *declares*
   material, but it cannot prove that an injected executor has no material
   effects. Executor-profile and non-materiality claims are host assertions. It
   does not implement a sandbox, durable/crash-safe transaction or replay state,
   real effect rollback, or a production authority service.
   These references are not trained family checkpoints and have not passed
   family evaluation or promotion gates. The exact class map is in
   `docs/model-families.md`.
6. Interfaces.
   FastAPI lives in `olympus.api`, Typer CLI lives in `olympus.cli`, and the React dashboard lives in `apps/forge-web`.

Pantheon under `OlympusC/05_pantheon` is a separate empirical research package,
not a seventh runtime layer. LabOS is an incubation-stage research-operations
tool whose parent-workspace reports are not Olympus scientific evidence. See
`research/RESEARCH_PROGRAM.md` for the maintained paper and claim boundaries.

The current implementation is intentionally small-scale but real. The Foundry
verification model proves infrastructure behavior only. Family codenames may
identify source-level roles, but no external model or checkpoint may take a
family identity until an immutable checkpoint passes its declared gates.
The fixed reference replay demonstrates bounded interoperability among all six
roles. It is not a configurable or autonomous research runtime, does not wire
the trainable family heads into one learned system, and confers no scientific,
model-quality, qualification, or promotion evidence.

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
