# Project Status

Evidence refreshed on 2026-09-02. This file distinguishes verified engineering
capability from unrun research claims.

## Current objective

Ship Olympus as a reproducible local cognitive-research, Model Foundry, and
portfolio-execution platform whose claims are bounded by immutable evidence.

## Working

- Python cognitive behaviors, Forge compiler/runtime, typed SDK, FastAPI API,
  and React Forge control surface.
- Durable Foundry lifecycle for datasets, experiments, checkpoints,
  evaluations, model records, evidence events, exports, and generation.
- Deep smoke infrastructure for tiny causal SFT, LoRA/QLoRA, resume,
  quantization, evaluation, resource admission, cancellation, and fail-closed
  promotion.
- LabOS bounded discovery, validation, scheduling, run history, event logging,
  and portfolio reports across Git and marker-based project roots.

## Implemented this run

- Expanded LabOS from Git-only discovery to top-level marker-based discovery;
  the real parent workspace report increased from 8 eligible Git checkouts to
  36 recognized project roots without recursive scanning.
- Redacted credential-like environment overrides from LabOS lifecycle events
  and captured child-process output.
- Added malformed optional README, TOML, and package metadata resilience.
- Isolated malformed declared LabOS manifests so one invalid project is marked
  with an error without aborting discovery of healthy siblings.
- Honored pinned npm versions through Corepack in inferred LabOS commands,
  release instructions, and CI/release workflows.
- Patched the web dependency lock to remove the Browserslist advisory.
- Excluded internal outreach and tiering documents from source distributions.

## Tested

- `ruff check .`: passed.
- `mypy --strict olympus tests`: passed for 78 source files.
- `pytest --cov=olympus --cov-branch`: 88 passed; 88.31% total coverage;
  required gate 85%.
- Forge web: 7 tests passed; production Vite build passed; 208.89 kB JS,
  64.83 kB gzip.
- `pip-audit --strict .` and `npm audit --audit-level=moderate`: no known
  vulnerabilities.
- Fresh sdist/wheel: package metadata check passed and the 150-file sdist
  contained no outreach material.
- Wheel installed in a new virtual environment outside the source tree. Its
  real Foundry verifier trained a character bigram, beat the frozen uniform
  baseline on held-out perplexity (6.68695 versus 24.0), exported a hashed
  checkpoint, and persisted one dataset, experiment, checkpoint, evaluation,
  and model plus five evidence events. A new process reopened the registry with
  SQLite integrity `ok`.

## Partially working

- The local Forge UI/API are production-buildable but not deployed.
- Deep Foundry training is scientifically useful as infrastructure validation;
  its tiny smoke model is not an assistant and did not pass task/tool promotion
  gates.
- LabOS discovered 36 projects, but 34 remain `discovered_only` until their own
  declared manifests and real validation commands are run.

## Broken

- The installed local `qwen3:8b` provider cannot pass this machine's RAM/swap
  admission boundary. The verified `qwen3:0.6b` provider is the usable local
  baseline.

## Blockers

- No trained or promoted Hermes, Prometheus, Perseus, Atlas, Kronos, or Aion
  checkpoint exists.
- Publication-quality claims still require frozen baselines, multi-seed
  experiments, ablations, confidence intervals, compute accounting, and
  independent review per research repository.
- Deployment, package publication, large-model training, and external outreach
  require account credentials, budget, and explicit external authorization.

## Metrics / experimental evidence

- Foundry verifier: held-out candidate perplexity 6.68695; uniform baseline
  24.0; decision passed.
- Deep smoke checkpoint: held-out loss 5.78961 to 3.90434, but 0% exact task and
  tool scores; promotion correctly returned `NOT_PROMOTED`.
- Portfolio inventory: 36 recognized project roots; 35 inferred manifests and
  1 declared manifest.

## Highest-value next actions

1. Add declared LabOS manifests and project-specific release gates to the top
   research candidates, then run their smoke profiles through the portfolio DB.
2. Freeze one Hermes base-model/licensing decision and execute matched-budget
   post-training comparisons on hardware that passes admission.
3. Add browser-level end-to-end coverage for the live Forge API/UI boundary.
4. Deploy only after an operator selects a host and supplies account-controlled
   credentials.

## Reproduction commands

```bash
.venv/bin/ruff check .
.venv/bin/mypy --strict olympus tests
.venv/bin/python -m pytest --cov=olympus --cov-branch --cov-report=term-missing -q
.venv/bin/pip-audit --strict .
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
.venv/bin/python -m olympus.cli foundry verify-pipeline
.venv/bin/python -m olympus.cli foundry status
.venv/bin/python -m olympus.cli labos generate-report --workspace ..
cd apps/forge-web
corepack npm ci
corepack npm audit --audit-level=moderate
corepack npm test
corepack npm run build
```
