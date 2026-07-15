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

## Quick Start

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e .[dev]
.venv/bin/pytest
.venv/bin/python -m olympus.cli demo run-all
.venv/bin/uvicorn olympus.api:app --reload
```

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
benchmarks/         Benchmark definitions and demo harnesses
behaviors/          Example behavior specifications
datasets/           Schemas, manifests, recipes, and local samples
docs/               Architecture and API documentation
examples/           End-to-end usage examples
experiments/        Experiment configurations and reports
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

## LabOS Portfolio Commands

Olympus also acts as the writable LabOS orchestration layer for the parent
workspace:

```bash
.venv/bin/python -m olympus.cli labos discover --workspace ..
.venv/bin/python -m olympus.cli labos validate --workspace ..
.venv/bin/python -m olympus.cli labos run-smoke --workspace .. --project olympus
.venv/bin/python -m olympus.cli labos generate-report --workspace .. --output-root .
```

The LabOS documentation lives in [docs/labos.md](/Users/ryan/Documents/Olympus/docs/labos.md:1).
