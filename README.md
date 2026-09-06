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

The shared model substrate and the Hermes, Olympus-Atlas, Prometheus, Perseus,
Kronos, and Aion role architectures now have executable reference
implementations under `olympus/models/`. They include typed contracts,
an access-filtered model view, scoped deterministic validation/gating paths,
trainable PyTorch components, and focused tests. Hermes and Prometheus have
partial shared-workspace integration; the remaining family services retain
local contracts, and the process-local guards are not durable authority or
rollback services.
Their status is `EXPERIMENTAL_SMOKE_NOT_PROMOTED`: a bounded synthetic component
run established executable optimizer/check paths, but no qualifying family
checkpoint or family-level result exists. See
[`docs/model-families.md`](docs/model-families.md) for the exact implemented
surface and claim boundary.

## Quick Start

```bash
python3.14 -m pip install --user uv==0.12.5
uv sync --locked --all-extras
uv run pytest
uv run python -m olympus.cli demo run-all
uv run python -m olympus.cli foundry verify-pipeline
uv run uvicorn olympus.api:app --host 127.0.0.1 --reload
```

To run the real web console against that API:

```bash
cd apps/forge-web
corepack npm ci
corepack npm run dev
```

Open `http://127.0.0.1:5173`. The console displays only live API results and
reports API failures explicitly. It shows hash-bound provenance and resource
admission, runs one bounded Foundry job at a time, and supports cooperative
cancel/retry. It can also inspect promoted models, generate through the
OpenAI-compatible local contract, and report Ollama health without fabricating
provider state.

Mutating and generation endpoints are loopback-only by default. Before putting
the API behind any proxy or binding it to a non-loopback address, set a strong
`OLYMPUS_API_TOKEN` and send it as a bearer token. The development Vite proxy is
intended only for the local loopback workflow above.

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

The complete twelve-stage execution index is machine-readable at
[`evidence/status.json`](evidence/status.json). Canonical-source decisions and
model-family truth are in
[`research/SOURCE_OF_TRUTH.md`](research/SOURCE_OF_TRUTH.md); the one highest
value next experiment and its frozen stop rules are in [`NEXT.md`](NEXT.md).

The former PostgreSQL/pgvector and Redis Compose scaffold is archived under
`archive/unused-infrastructure/`. It is not used by the implementation and is
not part of the supported runtime.

## Repository Layout

```text
apps/               API, CLI entrypoints, and Forge web console
behaviors/          Example behavior specifications
datasets/           Schemas, manifests, and local samples
docs/               Architecture and API documentation
examples/           End-to-end usage examples
archive/            Inactive scaffolds retained only for provenance
olympus/            Core Python implementation
research/           Design notes and model cards
scripts/            Utility scripts, including LOC accounting
tests/              Unit and integration coverage
evidence/           Twelve-stage execution records and machine-readable status
```

## Heuristic and Synthetic Demonstrations

These executable demonstrations are engineering fixtures, not validated
cognitive or neural mechanisms. Their historical names are retained only for
API compatibility:

1. Ambiguous interpretation analysis.
2. Code-state retrodiction.
3. JEPA-inspired forward prediction on synthetic sequences.
4. Learned transform decomposition.
5. Rule-based representation selection.
6. Dependency-aware specialist scheduling and reassembly.
7. Olympus Forge keyword-to-graph compilation and execution with optional,
   explicitly injected tool and memory effects.
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
