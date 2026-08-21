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
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e .[dev]
.venv/bin/pytest
.venv/bin/python -m olympus.cli demo run-all
.venv/bin/python -m olympus.cli foundry verify-pipeline
.venv/bin/uvicorn olympus.api:app --reload
```

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
