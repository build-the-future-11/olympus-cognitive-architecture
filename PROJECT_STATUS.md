# Project Status

Evidence refreshed on 2026-09-06. This file distinguishes verified engineering
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
  candidate-promotion evaluation. Signed evidence verification is implemented,
  including operator-pinned trust policy and internally consistent report metrics.
  Independent production signer/runners and qualifying model evidence remain absent.
  Background job state/history is durable; interrupted workers are reconciled on
  manager startup and require explicit retry. Built-in background verification runs
  in an isolated process with a deadline and forced cancellation; custom in-process
  runner extensions still require cooperative cancellation.
- LabOS bounded discovery, validation, scheduling, run history, event logging,
  and portfolio reports across Git and marker-based project roots.
- Executable reference implementations for the shared model substrate and all
  six family roles, including typed state/evidence/action contracts,
  an access-filtered model view, differentiable component losses, and selected
  deterministic validation/gating paths. A fixed synthetic two-phase replay
  composes the six roles for one contract-interoperability path. It exposes the
  proposed action and policy manifest, pauses for a host-registered
  manifest-bound Aion approval, and permits only tools declared non-material.
  The retrieval query and final request are public, hash-bound inputs; the
  Hermes step is restricted to authorized evidence listed in the frozen
  protocol. A runtime-secret HMAC derives pending and execution-scope IDs, and
  a separate private reservation token protects scoped Perseus
  prepare/commit/receipt/rollback operations. After Aion authorization, the
  order is Perseus execution, Kronos observation, Hermes response, Aion audit,
  then Aion STOP. Closed Kronos/Hermes degraded records prevent a post-gate
  component failure or exhausted model budget from stranding the run, and
  attempted model calls are charged before invocation and cached across an
  in-process retry. State, pending work, authority, tokens, transactions, and
  cached outputs remain process-local; the executor profile and non-materiality
  are host/declaration assertions, and no sandbox verifies actual behavior.

## Implemented this run

- Expanded LabOS from Git-only discovery to top-level marker-based discovery;
  the real parent workspace report increased from 8 eligible Git checkouts to
  36 recognized project roots without recursive scanning.
- Redacted credential-like environment values from LabOS lifecycle events and
  captured child-process output, including secrets inherited from the host.
- Added malformed optional README, TOML, and package metadata resilience.
- Isolated malformed declared LabOS manifests so one invalid project is marked
  with an error without aborting discovery of healthy siblings.
- Honored pinned npm versions through Corepack in inferred LabOS commands,
  release instructions, and CI/release workflows.
- Patched the web dependency lock to remove the Browserslist advisory.
- Excluded internal outreach and tiering documents from source distributions.
- Added source-level Hermes, Olympus-Atlas, Prometheus, Perseus, Kronos, and
  Aion component architectures. These are reference implementations, not
  qualifying family checkpoints or benchmark results.
- Added `OlympusReferenceReplay`, a deliberately fixed Atlas → Prometheus →
  Aion approval → Perseus → Kronos → Hermes → Aion STOP composition. Public
  pending/result artifacts omit trusted workspace state and raw executor output
  but intentionally disclose the retrieval/final requests and their authorized
  view bindings. The replay reserves each public execution scope with a private
  runtime capability, filters Hermes evidence to the frozen protocol, reports
  closed degraded post-gate outcomes, and accounts for every attempted family
  call. The implementation is still an in-process reference, not a durable or
  secure execution system.

## Tested

- Current audit-repair scope: 195 root Python tests passed with the unrelated,
  untracked O1 experiment excluded; Ruff passed; strict MyPy passed for 99
  source files; and branch-aware coverage reached 85.13%, above the 85% gate,
  with that O1 module explicitly omitted. This verifies the intended repair
  scope, not a releasable commit: the live-tree distribution payload validator
  correctly rejects untracked Python sources.
- Pantheon: 31 tests passed and 2 platform-dependent sandbox tests skipped; its
  evidence validator rechecked 480 classifier evaluations, 200 actual seeded-
  fault executions, and 20 public-case runs.
- Forge web: 7 tests passed; production Vite build passed; 208.89 kB JS,
  64.83 kB gzip.
- The locked-runtime dependency audit (`scripts/audit_locked_dependencies.py`)
  and `npm audit --audit-level=moderate`: no known vulnerabilities.
- Fresh sdist/wheel: package metadata check passed. The release payload gate
  correctly remains blocked because untracked Python sources and tests cannot
  be tied to the candidate commit.
- A temporary clean-candidate wheel installed into an isolated target outside
  the source checkout, using the locked build environment's dependencies. Its
  real Foundry verifier trained a character bigram, beat the frozen uniform
  baseline on held-out perplexity (6.68695 versus 24.0), exported a hashed
  checkpoint, and persisted one dataset, experiment, checkpoint, evaluation,
  and model plus five evidence events. A new process reopened the registry with
  SQLite integrity `ok`.
- The bounded 2026-09-06 family-component run passed for the shared substrate
  and all six roles using deterministic synthetic contract fixtures. Every
  emitted checkpoint is explicitly non-qualifying and unpromoted; this is
  execution evidence only. The release-facing record is
  `evidence/model_family_smoke_20260906.json`.
- The fixed six-role composition smoke is recorded in
  `evidence/model_composition_smoke_20260906.json`. It exercises one synthetic,
  declaratively non-material action with manifest-bound approval and sanitized
  output. It establishes contract composition only, not model quality,
  scientific validity, sandboxing, durability, or promotion eligibility.

## Partially working

- The local Forge UI/API are production-buildable but not deployed.
- The component smoke still executes each family independently. Separately, the
  fixed reference replay connects all six roles along one predetermined
  synthetic path. There is no configurable or autonomous end-to-end research
  service graph, crash-durable recovery, external authority service, or sandbox.
  Its recovery works only while the same runtime, transaction manager, scope
  token, Aion state, and cached outputs remain alive in the same process.
- Deep Foundry training is scientifically useful as infrastructure validation;
  its tiny smoke model is not an assistant and did not pass task/tool promotion
  gates.
- LabOS discovered 36 projects, but 34 remain `discovered_only` until their own
  declared manifests and real validation commands are run.

## Broken

- The dated 2026-09-01 ledger reports that local `qwen3:8b` failed the
  RAM/swap boundary and `qwen3:0.6b` returned a response. Its raw provider log,
  resource snapshot, and response artifact were not retained, so neither
  observation is a current reproducible result. The adapter and admission
  policy remain test-covered.

## Blockers

- No qualifying or promoted Hermes, Prometheus, Perseus, Olympus-Atlas, Kronos,
  or Aion checkpoint exists. Trainable reference modules and synthetic gradient
  tests do not satisfy family training, evaluation, or promotion gates.
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

These are separate scopes rather than one interchangeable ranking. Research
priority is frozen in `NEXT.md`; platform/release work cannot substitute for
that experiment, and a composition smoke cannot satisfy either scope's
promotion gates.

1. **Research:** externally timestamp and freeze the planned Pantheon
   capable-agent confirmation described in `NEXT.md`, then run it; afterward,
   pursue the separately gated Foundry external-base experiment.
2. **Portfolio engineering:** add declared LabOS manifests and project-specific
   release gates to the top candidates, then run their smoke profiles through
   the portfolio database.
3. **Model-reference engineering:** add crash-safe persistence for pending
   state, private scope capabilities, transactions, cached outputs, and Aion
   heads; add authenticated external authority and an enforceable sandbox before
   expanding beyond declaratively non-material synthetic actions.
4. **Product quality:** add browser-level end-to-end coverage for the live Forge
   API/UI boundary.
5. **Deployment:** deploy only after an operator selects a host and supplies
   account-controlled credentials.

## Reproduction commands

```bash
.venv/bin/ruff check .
.venv/bin/mypy --strict olympus tests
.venv/bin/python -m pytest --cov=olympus --cov-branch --cov-report=term-missing -q
.venv/bin/python scripts/audit_locked_dependencies.py
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
.venv/bin/python -m olympus.cli foundry verify-pipeline
.venv/bin/python -m olympus.cli foundry status
.venv/bin/python -m olympus.cli models composition-smoke \
  --output-dir artifacts/model-composition-smoke/verification --seed 20260906
.venv/bin/python -m olympus.cli labos generate-report --workspace ..
cd apps/forge-web
corepack npm ci
corepack npm audit --audit-level=moderate
corepack npm test
corepack npm run build
```
