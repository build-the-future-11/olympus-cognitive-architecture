# OLYMPUS Cognitive Architecture

Olympus is a local-first research platform for building and evaluating cognitive behaviors:

- multimodal state representation,
- interpretive branching with calibrated scoring,
- dynamic representation selection,
- retrodiction and forward prediction,
- behavior compilation and graph execution,
- structured verification, memory, data ingestion, and retrieval,
- synthetic training/evaluation demos,
- API, CLI, and a lightweight web console.

The repository now also includes a durable Model Foundry that performs a real,
bounded dataset → experiment → checkpoint → held-out evaluation → export →
serving lifecycle. Its built-in character-bigram run exists to verify the
infrastructure cheaply; it is explicitly not Hermes and carries no assistant or
reasoning capability claim.

Olympus is currently an alpha research system. Its tests establish executable
engineering behavior; they do not establish broad scientific superiority or
regulated-production fitness.

## Quick Start

```bash
uv sync --locked --all-extras
uv run --locked python -m pytest
uv run --locked python -m olympus.cli demo run-all
uv run --locked python -m olympus.cli foundry verify-pipeline
uv run --locked uvicorn olympus.api:app --reload
```

The default development server is loopback-only. Before binding the API to any
other interface, set `OLYMPUS_API_TOKEN` and send it as a bearer token to
stateful or compute-bearing routes. `OLYMPUS_MAX_CONCURRENT_WORKLOADS` defaults
to one and rejects excess work with HTTP 429.

The committed `uv.lock` is the canonical Python environment resolution used by
CI and release verification. Regenerate it only when dependency policy changes,
and reject drift with `uv lock --check`. Release distributions use the locked
build backend with `python -m build --no-isolation`.

To run the real web console against that API:

```bash
cd apps/forge-web
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. The console displays only live API results and
reports API failures explicitly. It can run the Foundry verification pipeline,
inspect promoted models, generate through the OpenAI-compatible local contract,
and report Ollama health without fabricating provider state.

To inspect an already-installed Ollama model through the provider adapter:

```bash
.venv/bin/python -m olympus.cli foundry ollama-smoke qwen3:0.6b \
  --context-tokens 256 --max-tokens 16
```

Model discovery and successful generation are reported separately. A model
being installed does not prove that it can load or generate on the current
machine.

The publication-grade Foundry controls can prepare an immutable instruction
dataset and run full SFT, LoRA, or genuine 4-bit QLoRA smoke jobs:

```bash
.venv/bin/olympus foundry prepare-dataset \
  --source datasets/hermes-smoke/source.jsonl \
  --output artifacts/foundry/deep/dataset
.venv/bin/olympus foundry train-sft \
  artifacts/foundry/deep/dataset/manifest.json \
  --output artifacts/foundry/deep/training --mode full
```

Held-out evaluation, int4/int8 quantization, checkpoint resume, RAM/swap
governance, and fail-closed promotion are implemented and tested. The current
tiny transformer is an infrastructure smoke checkpoint and failed task-quality
gates; no Hermes model is promoted. See
[`OLYMPUS_MODEL_FOUNDRY_LEDGER.md`](OLYMPUS_MODEL_FOUNDRY_LEDGER.md) for exact
measurements and blockers.

All six model-family contracts also have bounded executable reference
architectures. Inspect their exact local parameter counts and non-promotional
status with:

```bash
.venv/bin/olympus bu1ld reference-models
```

These reference modules exercise Hermes tool/abstention heads, Prometheus
domain adapters, Perseus sparse expert routing, Atlas multimodal grounding, and
Kronos temporal planning, and Aion governed research-transition routing. They
are not trained 12B-8T family checkpoints.

`OlympusReferenceSystem` composes those references into an explicit five-way
family gate followed by Aion control. Its joint loss trains coordination,
temporal prediction, uncertainty, and legal research transitions while a
host-supplied Aion STOP suppresses the decision surface. The detailed boundary
is documented in [docs/olympus-system.md](docs/olympus-system.md).

The composition also supports normalized extension channels, emitted
leave-one-family-out decisions, disagreement-conditioned risk that cannot lower
the host risk floor, resource-governed inference, immutable checkpoints, and a
content-addressed aligned dataset contract. The JEPA upscaler provides
multiscale and phase-aware objectives plus seam-weighted tiled and eight-way
self-ensemble inference. A guarded `bu1ld train-olympus-system` path now binds a
verified dataset and explicit model config to training, held-out routing/risk
diagnostics, and an immutable checkpoint/report. The formal research proposal,
falsifiers, controls, and release
gates are in
[research/OLYMPUS_PAPER_DRAFT.md](research/OLYMPUS_PAPER_DRAFT.md).

A separately versioned controlled study has now completed 25 training runs and
75 protected evaluations. Its disposition is **NEGATIVE**: learned fusion tied
uniform fusion on clean accuracy and did not improve corrupted selective
accuracy. All tested host-risk and transition invariants held. This result is
limited to the synthetic generator; natural-data fusion and JEPA efficacy remain
unevaluated. The protocol, state, audit, and evidence map live under
`research/olympus_irp_successor_v2/`.

A second, separately frozen reliability-supervised successor also completed 25
fits and 75 protected evaluations. Its disposition is **NEGATIVE**: the full
router's corrupted selective accuracy was 0.0125 below the semantic control
(95% paired-bootstrap interval -0.0281 to 0.0031), short of the preregistered
+0.0200 effect. Risk monotonicity and transition legality still had zero
violations. The complete evidence lock and arXiv-ready, unsubmitted negative
paper live under `research/olympus_irp_successor_v3/`.

The v4 oracle-routing diagnostic subsequently completed 40 cells, 80 expert-bank
fits, and 120 protected evaluations. Its preregistered utility-routing primary
failed by 6.82 percentage points against the development-selected semantic
control. Two bounded repairs were supported: mixed-corruption training improved
corrupted accuracy by 27.42 points, and semantic routing exceeded learned fixed
routing by 10.21 points. The v5 distribution-shift and residual-temporal
successor is frozen but not yet executed. The authoritative cross-iteration
status is `RESEARCH_STATE.json`.

The bounded Foundry SFT path now uses configuration-hashed immutable runs,
strict resume compatibility, warmup plus cosine decay, validation early
stopping, and best-state restoration. These mechanisms improve development-run
integrity but do not turn the bundled 36-record smoke corpus into a capability
dataset.

The historical Olympus O1 four-arm preregistration has also been migrated from
the closed, unmerged VertexED provenance surface as nine byte-exact,
SHA-256-bound files under `research/olympus_o1/`. It remains
`PREREGISTERED_NOT_RUN` / `UNFROZEN_DO_NOT_RUN`: no 100-task manifest,
provider/runtime freeze, prompt hashes, or execution authorization exists.
This migration preserves protocol ownership; it is not an O1 result.

The revision-pinned EditCLEVR paired-vision snapshot is fully downloaded,
validated, and locally materialized under `datasets/external/editclevr/`.
`EditCLEVRDataset` streams the 186 MB split registry without loading it in full,
enforces path containment, and deterministically decodes RGB tensors, scene
objects, and no-pickle NPZ masks. The archives and materialized payload are
ignored; the manifest, provenance, validation, deviation, and loader-smoke
receipts are reviewable. See
[`datasets/external/editclevr/PROVENANCE.md`](datasets/external/editclevr/PROVENANCE.md)
for the pinned acquisition and materialization commands. Dataset readiness is
not a JEPA result or an Olympus confirmatory result.

For the optional local Postgres/Redis services, create an untracked `.env` with
a real random password before starting Compose:

```bash
printf 'OLYMPUS_POSTGRES_PASSWORD=%s\n' "$(openssl rand -hex 32)" > .env
docker compose up -d
docker compose config --quiet
```

Compose binds both service ports to loopback by default. Override
`OLYMPUS_POSTGRES_PORT` or `OLYMPUS_REDIS_PORT` only when the local ports are
already occupied; do not commit `.env`.

## Repository Layout

```text
apps/               API, CLI entrypoints, and Forge web console
behaviors/          Example behavior specifications
datasets/           Schemas, manifests, and local samples
docs/               Architecture and API documentation
examples/           End-to-end usage examples
infrastructure/     Local infrastructure configuration
olympus/            Core Python implementation
research/           Design notes and model cards
scripts/            Utility scripts, including LOC accounting
tests/              Unit and integration coverage
```

## Core Demonstrations

The implemented demos correspond to the attached research brief:

1. Ambiguous interpretation analysis.
2. Code-state retrodiction.
3. JEPA forward prediction on synthetic sequences.
4. Learned transform decomposition.
5. Dynamic representation selection.
6. Neural division and reassembly.
7. Olympus Forge natural-language behavior compilation and execution.
8. Hermes Nano interpretation and SQLite memory integration.

## LabOS Portfolio Commands

Olympus also acts as the writable LabOS orchestration layer for the parent
workspace:

```bash
.venv/bin/python -m olympus.cli labos discover --workspace ..
.venv/bin/python -m olympus.cli labos validate --workspace ..
.venv/bin/python -m olympus.cli labos run-smoke --workspace .. --project olympus
.venv/bin/python -m olympus.cli labos generate-report --workspace ..
```

The LabOS documentation lives in [docs/labos.md](docs/labos.md).

## Release verification

The automated release gate runs the Python tests, lint, strict type check,
distribution build, web tests, and production web build. Exact local commands
and the remaining owner-controlled publication decisions are documented in
`RELEASE.md`.

The Foundry lifecycle, registry invariants, API contract, artifact layout, and
truth boundaries are documented in [docs/foundry.md](docs/foundry.md). The
current factual model-program state is recorded in
[research/REALITY_LEDGER.md](research/REALITY_LEDGER.md) and the detailed
[model Foundry ledger](OLYMPUS_MODEL_FOUNDRY_LEDGER.md).
