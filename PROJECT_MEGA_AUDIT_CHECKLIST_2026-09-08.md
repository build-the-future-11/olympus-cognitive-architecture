# Olympus: current audit and completion checklist

Audit date: 2026-09-08. Decision: **FIX — NOT READY for general public model use or comparative model publication.**

This document supersedes earlier status descriptions for the areas inspected here. It is an audit and work plan, not a declaration that its unchecked work has been implemented. No finite checklist establishes perfection. Completion means meeting a frozen, testable release contract and recording remaining limitations.

## Scope and evidence limits

### Astra follow-up execution (2026-09-08)

After the initial execution below, validation/test sampling was separated from
training mixture weights and packing. Quantization now checks the training-manifest
identity before producing artifacts and binds it in schema-v3 evidence. Mixture
weights and the fixed byte-tokenizer vocabulary have explicit bounds.

Built-in background verification now uses a child process/session with independent
SQLite ownership, wall-clock deadline, TERM/KILL cancellation, result validation and
supervisor-death monitoring. Tests exercise a real lifecycle and kill/reap an
uncooperative process. The custom Python runner hook remains cooperative and is not
an untrusted-code sandbox. Historical v1/v2 results were not overwritten or relabelled.

Follow-up verification: **295 Python tests passed, 85.40% combined coverage**;
Ruff and strict mypy (105 files) passed. Web tests (16) and production build passed.
Two clean package builds were byte-identical; Twine, installed-wheel lifecycle,
six-family reference composition and installed-wheel child execution passed.
The tracked-payload gate still fails on untracked sources; no files were staged to
bypass it. See [Astra execution evidence](PROJECT_ASTRA_EXECUTION_2026-09-08.md).
The older observations below remain historical, including their then-open deadline
limitation, which this follow-up resolves for the built-in background workflow.

### Initial execution update (2026-09-08)

The initial measurements below are retained as the pre-execution audit, not current
test failures. Final validation: 277 Python tests, 85.26% combined statement/branch
coverage, strict typing, repository Ruff, 16 web tests, production web build,
reproducible wheel/sdist and installed-wheel verification pass. Payload validation
still rejects untracked sources. Full evidence is in
[the execution report](PROJECT_EXECUTION_REPORT_2026-09-08.md).
The checked items now carry implementation evidence. This pass does
not claim the entire 108-item checklist or the model portfolio is complete.

- B04: operator trust-policy SHA-256 pinning implemented; independently controlled
  signer credentials and production release configuration still require provisioning.
- B05/F05: bounded regular-file snapshots for reports/cards/attestations and streaming
  promotion artifact hashes implemented. Large model loading and downstream artifact
  immutability remain separate unresolved requirements.
- B06: unique per-run release records, input-path alias rejection and preservation of
  history implemented. Latest-summary transaction/replay-consumer guarantees remain open.
- B12: invalid-input attempt records and explicit CLI exit codes (0/1/2) implemented;
  previous success outputs are never presented as the result of an invalid attempt.
- C05/C07: corrected batch-weighting bugs in evaluation and gradient accumulation;
  restored RNG state for dropout resume; CPU resume is bitwise checked. New objective
  and report versions prevent silent mixing with historical v1 measurements. Full
  packing/isolation studies and CUDA/MPS parity remain unverified.
- E03/E06/G06: durable job history API and cross-worker cancellation implemented.
  Process-isolated hard deadlines, full multi-worker registry certification and a
  history browser UI remain unfinished, not external-blocker excuses.
- I-series: repaired setuptools metadata-directory validation without accepting
  untracked source or arbitrary metadata payloads. Made package API export lazy,
  retaining compatibility and avoiding eager API/Torch loading in lightweight workers.

No frozen Pantheon results, model-family claims, historical metric artifacts or
private training data were modified. See `docs/foundry.md` for the v2 compatibility
boundary and the execution report for final verification results.

The current inspection covers Olympus runtime, model roles, Foundry, promotion, API, SDK, ingestion, jobs, frontend, tests, build/release configuration, and research status artifacts. The sibling-project section uses the dated 2026-09-07 portfolio audit; those repositories were not all rerun on September 8. Earlier test counts are historical observations, not fresh certification of this changing checkout. The checkout contains extensive modified and untracked work; HEAD alone does not reproduce it.

The current request is for status, an audit, and a specific checklist. This pass adds this report; implementation items below remain proposals. Prior execution work is described separately.

## Fresh verification on this checkout

| Check | Result | Interpretation |
| --- | --- | --- |
| Python tests with coverage | **FAIL: 257 passed, 1 failed** | `test_release_workflows_enforce_clean_sources_and_distribution_validation` still requires `brew install tectonic`; workflows now use the checksum-pinned installer |
| Coverage | **PASS: 85.14%** | Meets the 85% combined statement/branch-aware coverage gate; this is not 85.14% of all branches individually |
| Ruff (`olympus tests scripts`) | **PASS** | Scoped to maintained Python directories, not a claim about every sibling repository |
| Strict mypy (`olympus tests`) | **FAIL: 3 errors in 2 files** | One attestation fixture uses an unconstrained string for a Literal; two Foundry tests access non-exported imports through the service module |
| Frontend component tests, first run | **FAIL: 10 passed, 6 timed out** | All six failures hit the existing 5-second limit while host load was severe |
| Frontend component tests, isolated rerun | **PASS: 16/16 in 5.19 seconds** | Same code and existing timeout settings; supports resource contention as the cause of the earlier timeout cluster |
| Frontend TypeScript/production build | **PASS** | Vite generated 216.11 kB JS, 66.77 kB gzip |
| Diff whitespace check | **PASS** | Does not validate semantics |
| Final package build/install/payload, dependency audit, remote CI/deployment | **NOT RUN in this refresh** | Prior results and existing workflow code cannot certify this final checkout |
| Pantheon and sibling study results | **Existing artifacts/prior audit inspected** | Not fresh experiments or complete reruns in this refresh |

The Python suite took 455.78 seconds. During the concurrent verification run, host load averages were 234.37/157.50/84.92. The subsequent unchanged web suite passed all 16 tests when run separately. Preserve both observations and admit expensive verification workloads sequentially when this host is overloaded.

## Executive assessment

Olympus has substantial working software. Its most credible product is a local research platform that records datasets, experiments, checkpoints, evaluations and provenance, then applies explicit release gates. The character-bigram lifecycle is real; the tiny-transformer training, adapter and quantization code is executable. These are the strongest engineering assets to keep.

The six named families contain actual Python/PyTorch reference components and bounded deterministic services. They are not trained 12B/128B/1.024T/8T assistants. The registry explicitly records zero qualifying checkpoints. The current implementation roles also differ from the original product aspirations: Perseus is a tool-action component, Atlas is retrieval, Kronos is temporal planning, and Aion is governed orchestration. Those decisions must be resolved before more architecture is added.

The most consequential remaining defects are incomplete evaluation validation, incomplete operational control of promotion evidence, unsafe assumptions about process-local execution, and lack of a clean reproducible release. The new attestation verifier improves evidence identity but does not independently establish the truth of measurements or the independence of signers.

## What was implemented well

| Component | Actual implementation | Practical limit |
| --- | --- | --- |
| Foundry lifecycle | Dataset/experiment/checkpoint/evaluation/model records, SQLite integrity, hashes, exports, restart checks and generation | The promoted verifier is a character bigram |
| Tiny training | Causal transformer, full SFT, LoRA, packed 4-bit QLoRA bases, resume state and resource controls | Small smoke data; no qualified pretrained Hermes integration |
| Artifact integrity | Dataset verification, checkpoint checks, lineage checks, strict finite JSON and atomic service writes | Not every evidence reader/writer has the same guarantees |
| Promotion signatures | Ed25519 verification, exact subjects, four evidence kinds, issuer threshold, runner pins, expiration and revocation fields | Trust roots are caller-selected; operational authority is not established |
| Model components | Real trainable modules, losses and deterministic authorization/validation paths | Learned modules and deterministic services are mostly separate |
| Reference composition | Fixed two-phase six-role replay, approval binding, budget accounting and closed degraded outcomes | Process-local state; no enforced tool sandbox or crash recovery |
| API/SDK/provider | Stronger local access checks, secret-safe failures, request bounds, HTTPS checks and provider response validation | No production identity/tenant model or rate limits |
| Web | Live API state, validation, confirmation, polling backoff, provenance, loading/error/empty states | Mocked component tests do not exercise a deployed browser/API journey |
| Release tooling | Locked dependencies, archive validation, isolated-wheel checks and reproducible-build tooling | Final immutable release candidate has not passed all gates |
| Research reporting | Explicit negative findings and structural/scientific gate separation | Most claimed mechanisms lack external confirmatory evidence |

## What is pseudocode, scaffolding, or a limited implementation?

The search did not identify a blanket TODO/NotImplementedError implementation in the inspected active runtime. Calling the entire project pseudocode would be inaccurate. The important distinction is below.

| Area | Classification | Evidence and consequence |
| --- | --- | --- |
| `olympus/models/*.py` | Executable research prototypes | Actual modules and tensor operations; no qualifying family weights |
| Hermes grounded service | Deterministic extractive baseline | Selects an authorized sentence using token overlap; confidence is overlap, not calibrated truth probability |
| Atlas service | Deterministic hybrid retrieval baseline | BM25 and hashed token vectors; separate trainable retriever is not proof of learned production retrieval |
| Prometheus | Trainable proposer/verifier components plus receipt-based selector | Does not constitute a trained specialization or research-synthesis product |
| Perseus | Action/transaction reference implementation | In-process approval and transaction controls; external effects are not contained by an OS sandbox |
| Kronos | Trainable temporal prototype plus guarded adapter manager | No demonstrated AGI, long-term world model, deletion replay or broad continual learning |
| Aion | Protocol controller/router reference | No general autonomous research system |
| `olympus/models/composition.py` | Fixed integration replay | Meaningful contract test, not general graph execution or durable autonomy |
| `olympus/core/interpretation.py` | Heuristic demo | Keyword scores and templated interpretations; the historical class name does not make it a learned network |
| `olympus/forge/compiler.py` | Keyword compiler | Builds a fixed pattern from words such as verify/tool/memory; not arbitrary natural-language program synthesis |
| `olympus/forge/runtime.py` | Working runtime with optional adapters | Missing tool/memory adapters return explicit skipped states; these are honest unavailable integrations |
| `research/architectures/` and paper | Specifications/proposal | Describes intended designs; document existence is not experimental completion |
| `archive/unused-infrastructure/` | Explicitly archived scaffold | PostgreSQL/Redis are not active runtime dependencies |
| Test mocks | Legitimate test doubles | Useful for edge cases; cannot stand in for live integration tests |
| Frontier size, perfection, sentience, AGI labels | Unsupported aspirations | No corresponding qualified artifacts were established |

## Scoring

Scores are reviewer judgments about the inspected implementation, not measured benchmark results. Targets below mean a credible supported local alpha, not perfection. General public hosting and publication have additional gates.

| Dimension | Current / 10 | Local-alpha target | Reason |
| --- | ---: | ---: | --- |
| Architecture | 6 | 8 | Useful separation; duplicated execution/persistence boundaries |
| Correctness | 6 | 8 | Many regression tests; promotion report consistency remains weak |
| Maintainability | 6 | 8 | Typed code; duplicated validators and status documents |
| Reliability | 5 | 8 | Durable artifacts; volatile jobs and composition state |
| Test quality | 7 | 8 | Substantive unit/integration coverage; missing crash/browser evidence |
| Security | 5 | 8 for local scope | Local safeguards; no hosted or sandbox boundary |
| Observability | 5 | 8 | Events/provenance exist; failure diagnosis and run correlation incomplete |
| Performance evidence | 3 | 7 | Little representative latency, throughput or concurrency measurement |
| Developer experience | 6 | 8 | CLI and lockfiles; inconsistent docs and unfinished release validation |
| Product usefulness | 6 | 8 | Useful to researchers; not yet a capable personal assistant |
| UX/onboarding | 6 | 8 | Real controls and states; incomplete end-to-end validation |
| Model readiness | 2 | Separate trained-beta gate | Reference code exists; no qualified family model |
| Research reproducibility | 6 | 8 | Some retained artifacts; source/evidence ownership still uneven |
| Evaluation rigor | 4 | 8 | Baselines and negative results exist; metric contracts need strengthening |
| Novelty established | 2 | Requires experiments | Architecture combinations do not establish novelty |
| Claim/evidence alignment | 5 | 9 | Truthful registry; stale docs and aspirational naming conflict |
| Paper readiness | 3 | Paper-specific | Proposal and bounded negative evidence; no general comparative result |
| Release provenance | 4 | 9 | Tooling exists; dirty/untracked candidate cannot reproduce from HEAD |

## How to use the checklist

Every item states a problem, the affected component, a proposed solution, and a verification condition. P0 blocks the relevant release; P1 completes the supported product; P2 is research/extension work; P3 is later polish. A hosted-only P0 does not force a local alpha to become a hosted service. Choose a release scope first. Do not add every proposed extension indiscriminately.

### A. Product scope and honest identities

- [ ] **A01 — P0: Freeze the deliverable.** Problem: platform, assistant, six model families and research portfolio have incompatible completion criteria. Component: README, `docs/model-families.md`, `research/RESEARCH_PROGRAM.md`. Solution: define local platform alpha, trained Hermes beta and paper submissions as separate deliverables. Verify: each has its own acceptance gate and none can claim completion through another's tests.
- [ ] **A02 — P0: Resolve model-role drift.** Problem: current Perseus/Atlas/Kronos roles differ from the original generalist/deep-intelligence/AGI descriptions. Component: `olympus/models/registry.py`, architecture specs. Solution: approve one capability contract per name before expanding implementation. Verify: registry, UI, architecture and evaluation use the same roles.
- [ ] **A03 — P0: Replace unsupported release claims.** Problem: model names and parameter aspirations can be mistaken for released capabilities. Component: model cards, README, paper, web copy. Solution: publish implementation status, actual parameter count, checkpoint hash and qualification separately. Verify: an untrained or synthetic artifact cannot appear as an available family assistant.
- [ ] **A04 — P1: Define a first user journey.** Problem: broad architecture lacks one proved value outcome. Component: Forge web, CLI, Foundry docs. Solution: make one local dataset-to-evidence-to-export journey complete. Verify: a new user completes it from the installation instructions without editing source.

### B. Promotion and evaluation correctness

- [x] **B01 — P0: Recompute report decisions.** Problem: `promotion.py` trusts `regression_count` and `passed_quality_gate`; report schemas do not enforce consistency with component metrics. Component: `eval_suite.py`, `quantization.py`, `promotion.py`. Solution: derive regression and quantization decisions from validated metrics, with versioned rules. Verify: correctly signed but contradictory reports are rejected. **Completed 2026-09-08:** Validated derived regression, perplexity, smoke and quantization decisions; contradictory payloads are rejected in tests/test_deep_foundry.py.
- [x] **B02 — P0: Validate complete evaluation coverage.** Problem: empty/missing/duplicate workflows or categories are insufficiently constrained; empty workflows can reach `min()`. Component: `HeldOutEvaluation`, promotion. Solution: require the declared suite, unique categories/workflows, matching manifest counts and exact workflow definitions. Verify: omitted safety/tool category, duplicated easy category, empty workflows and count mismatches fail explicitly. **Completed 2026-09-08:** Required exact v2 suite/category/workflow membership and manifest counts; missing coverage and mismatched counts are rejected.
- [ ] **B03 — P0: Bind provenance to the actual base and training run.** Problem: a permitted license string is weaker than a record of the exact base revision and training ancestry. Component: training checkpoint schema, promotion subject, model card. Solution: carry immutable base hash/revision, training configuration, dataset identity and rights-review identity through the chain. Verify: substituting a base or claiming a different ancestry invalidates promotion.
- [ ] **B04 — P0: Put trust policy outside candidate control.** Problem: CLI callers can supply their own trust store; signatures alone do not prove independent review. Component: `attestation.py`, CLI, release automation. Solution: pin the reviewed trust-policy digest in release configuration and isolate signer credentials from training. Verify: candidate-provided keys cannot qualify a release through the authoritative runner.
- [ ] **B05 — P0: Eliminate evidence read/hash races.** Problem: reports, cards and attestations are parsed and hashed in separate file reads. Component: promotion/attestation readers. Solution: read bounded immutable bytes once, parse and hash the same bytes, use snapshots for large artifacts. Verify: concurrent replacement cannot make accepted evidence and recorded hashes disagree.
- [ ] **B06 — P0: Scope release output per candidate.** Problem: promotion uses a shared `release-manifest.json` in the output directory and unlinks it on reevaluation; concurrent candidates can interfere. Component: `promotion.py`. Solution: isolate candidate/run directories, reject input/output aliases, serialize finalization and preserve previous decisions as history. Verify: two simultaneous candidates cannot overwrite or delete each other's release record; failed parsing cannot leave an apparently current success.
- [ ] **B07 — P1: Specify re-verification and replay policy.** Problem: expiry/revocation fields exist, but downstream release consumers and replay semantics are not demonstrated. Component: attestation/release consumption. Solution: define when signatures must be live, how historical attestations are retained and how revocations affect deployment. Verify: an expired or revoked bundle cannot authorize a new release, while historical reports remain inspectable.
- [ ] **B08 — P1: Produce real independent attestations.** Problem: a signing helper and ephemeral test keys are the only demonstrated producer path. Component: runner integration, CI, evidence registry. Solution: have isolated evaluation, quantization, serving and rights-review processes emit signed reports over retained raw evidence. Verify: a fresh verifier reconstructs the subject and validates the exact release candidate.
- [ ] **B09 — P0 for model release: Strengthen task scoring.** Problem: exact string matching and checking only that JSON is an object do not establish tool correctness or safety. Component: `_format_compliant`, workflow scoring, evaluation tools. Solution: use declared schemas, executable tool outcomes and reviewed task-specific scoring; retain raw predictions. Verify: valid-but-wrong JSON, wrong arguments, false citations and unsafe completions are rejected by the appropriate metrics.
- [ ] **B10 — P1: Test the quantized execution claim.** Problem: loading packed weights dequantizes into a float model; storage compression is not proof of integer inference acceleration or lower resident memory. Component: `quantization.py`, runtime, model card. Solution: label storage-only quantization accurately; integrate an actual supported quantized backend if required. Verify: measure resident memory, output quality and latency on the actual deployed runtime.
- [ ] **B11 — P1: Separate smoke and qualification policies.** Problem: generic reports can be confused with Hermes-specific promotion rules. Component: schemas, registry, promotion. Solution: encode purpose, family and policy version; reject unsupported family qualification. Verify: an infrastructure smoke or a Hermes-policy result cannot qualify Kronos or another family.
- [ ] **B12 — P1: Make invalid evidence diagnostic.** Problem: malformed files can raise before writing a decision; prior outputs may remain. Component: CLI and promotion output lifecycle. Solution: emit explicit invalid-input outcomes into a new run record and return a documented nonzero exit code. Verify: automation can distinguish failed quality, invalid evidence, missing dependencies and success.

### C. Data and training

- [ ] **C01 — P0 for model release: Obtain a usable base.** Problem: no qualifying revision-pinned pretrained Hermes checkpoint is established. Component: base-candidate ledger, SFT integration. Solution: select a feasible base, record revision/hash and approval, implement its real tokenizer/adapter loading path. Verify: a clean process loads the exact base and reproduces baseline inference.
- [ ] **C02 — P0 for model release: Meet the frozen data contract.** Problem: the recorded smoke corpus is 12 train/12 validation/12 test records. Component: dataset manifests and preparation. Solution: acquire/review data against the declared floors; record provenance and review decisions. Verify: actual files, not edited counts, satisfy 10,000 train and 1,200 test with required category counts.
- [ ] **C03 — P1: Split at source level.** Problem: normalized/exact and eight-token checks do not rule out paraphrased or document-level leakage. Component: data pipeline. Solution: source/document/group IDs, near-duplicate analysis and frozen split assignment. Verify: shared sources and paraphrases are detected without looking at outcomes to choose exclusions.
- [ ] **C04 — P1: Protect held-out data operationally.** Problem: explicit split labels alone do not prevent iterative tuning on test outcomes. Component: run orchestration and experiment protocol. Solution: separate development evaluation and protected final evaluation; record each test access. Verify: training and tuning jobs cannot read protected answers.
- [ ] **C05 — P1: Validate training objectives.** Problem: successful optimization can conceal loss-mask, padding, packing or resume bugs. Component: `sft.py`, training tests. Solution: hand-check target masks and compare uninterrupted/resumed runs under documented determinism. Verify: masks and gradients match tiny analytical fixtures and resume parity holds within declared tolerance.
- [ ] **C06 — P1: Make ablations change the mechanism.** Problem: the retained no-packing comparison did not create a distinct condition. Component: packing/mixing configs and research scripts. Solution: include manipulation checks for sequence composition and effective budgets before running comparisons. Verify: each ablation demonstrably changes the intended operation while controlling compute.
- [ ] **C07 — P1: Preserve full training provenance.** Problem: a checkpoint alone does not reconstruct environment, sampling and effective runtime options. Component: SFT run manifest/checkpoint. Solution: capture code digest, dependencies, device, seeds, data order, actual precision and resource admission. Verify: a fresh run can reconstruct the declared configuration and distinguish requested from effective options.
- [ ] **C08 — P2: Add post-training stages only after admission.** Problem: preference tuning, distillation and specialization remain plans. Component: `research/POST_TRAINING_STAGES.md`. Solution: implement only the next stage with approved data, a frozen baseline and measurable exit conditions. Verify: retained held-out gains and safety/regression results justify keeping that stage.

### D. Each named model

- [ ] **D01 — Hermes / P1: Integrate the learned path.** Problem: current serving selects a sentence by overlap; the trainable grounding head is separate. Component: `hermes.py`, substrate, Foundry serving. Solution: connect one trained generator/grounder to authorized evidence and structured citation output. Verify: real unseen questions exercise trained weights and preserve authorization/abstention.
- [ ] **D02 — Hermes / P1: Calibrate confidence.** Problem: overlap is exposed as confidence. Component: grounded service and evaluation. Solution: label overlap as retrieval support and fit/test calibration on separate validation data. Verify: reliability diagrams, abstention coverage and selective error on untouched data.
- [ ] **D03 — Hermes / P1: Complete memory approval.** Problem: memory proposals do not constitute approved durable memory. Component: Hermes proposals, memory store, authorization service. Solution: approve/reject proposals with provenance, conflict handling and deletion. Verify: unapproved writes never persist; deletion survives restart.
- [ ] **D04 — Prometheus / P1: Connect proposal, verification and selection.** Problem: components are tested independently without a full synthesis workflow. Component: `prometheus.py`, composition. Solution: run one bounded domain task through proposal, evidence collection, verifier and selector. Verify: unsupported branches fail and decisions reference original evidence.
- [ ] **D05 — Prometheus / P1: Give receipts real authority.** Problem: host-allowlisted receipts are not independently signed runner evidence. Component: selector and receipt registry. Solution: authenticate receipt issuer, scope, source and expiry using a reviewed common evidence contract. Verify: invented, stale and cross-task receipts fail.
- [ ] **D06 — Prometheus / P2: Demonstrate specialization.** Problem: no retained coding/research/UI specialization quality comparison. Component: adapter selection/training and task suite. Solution: evaluate one domain against the same base and simple fine-tuning controls. Verify: specialization improves the selected task without unacceptable forgetting or cost.
- [ ] **D07 — Perseus / P0 before real effects: Enforce isolation.** Problem: declarative permissions cannot constrain a host-injected tool handler. Component: `perseus.py`, Forge executor. Solution: run tools in a restricted process/container with bounded filesystem, network, time and resource access. Verify: adversarial tools cannot exceed granted capabilities.
- [ ] **D08 — Perseus / P1: Persist transaction state.** Problem: idempotency, reservations and pending effects are process-local. Component: transaction manager. Solution: durable effect IDs, execution journal and reconciliation of uncertain outcomes. Verify: crash after each step produces no duplicate effect on retry.
- [ ] **D09 — Perseus / P1: Define compensation honestly.** Problem: rollback cannot automatically reverse arbitrary committed external effects. Component: action schema/executors. Solution: declare reversible, compensatable and irreversible actions and their approval/recovery rules. Verify: tests exercise actual executor behavior and expose uncertain outcomes.
- [ ] **D10 — Atlas / P1: Persist index and ACL versions.** Problem: retrieval state is in memory. Component: `atlas.py`. Solution: durable versioned documents/passages, access policy, index metadata and atomic activation. Verify: restart retains the same authorized results and revoked/deleted material disappears.
- [ ] **D11 — Atlas / P1: Evaluate learned retrieval.** Problem: hashed vectors are not trained semantic embeddings, and the trainable retriever is separate. Component: Atlas service/retriever. Solution: compare BM25, current hashing, learned dense retrieval and hybrid on a fixed corpus. Verify: recall/ranking gains, latency and unauthorized-result rate are reported.
- [ ] **D12 — Atlas / P2: Measure retrieval maintenance.** Problem: no representative scale/update evidence. Component: index implementation. Solution: benchmark corpus growth, incremental changes and concurrent reads. Verify: documented memory/latency budget at a declared corpus size.
- [ ] **D13 — Kronos / P1: Verify adaptation evidence contents.** Problem: allowlisted bound identifiers do not themselves parse evaluation reports or prove replay results. Component: `kronos.py`, adapter manager. Solution: ingest immutable validated evaluations and require regression/deletion replay evidence. Verify: a named but absent or contradictory report cannot promote an adapter.
- [ ] **D14 — Kronos / P1: Add real temporal tasks.** Problem: synthetic temporal gradients do not establish useful forecasting/planning. Component: temporal data and tests. Solution: freeze time-ordered tasks and compare against simple temporal baselines under equal budgets. Verify: no future leakage and retained per-seed results.
- [ ] **D15 — Kronos / P1: Prove recovery.** Problem: broad rollback and continual-adaptation claims lack complete replay evidence. Component: adapter lifecycle. Solution: retain previous versions, replay protected evaluations and implement recovery from interrupted activation. Verify: restart and deliberate bad updates restore a known-good state.
- [ ] **D16 — Aion / P1: Persist protocol authority.** Problem: approvals, heads and receipts are local to a process. Component: `aion.py`, composition state. Solution: durable signed approval records and transactional state transitions. Verify: crash/restart cannot reuse stale authority or skip protocol steps.
- [ ] **D17 — Aion / P2: Add bounded configurable orchestration.** Problem: one fixed replay does not test general workflows. Component: controller/composition. Solution: support a small declared graph with budgets, stop rules and failure transitions after durability/isolation are established. Verify: malformed cycles, exhausted budgets and missing evidence end deterministically.
- [ ] **D18 — All families / P0 for release: Qualify actual checkpoints.** Problem: component loss tests are not model evaluations. Component: registry/model cards/family evaluation. Solution: checkpoint-bound datasets, baselines, multiple seeds, operational tests and independent evidence per family. Verify: only exact passing checkpoints receive family release status.

### E. Reliability, persistence and architecture

- [x] **E01 — P1: Unify workload admission.** Problem: synchronous `/foundry/verify` calls the service directly while background jobs use admission and cancellation. Component: `api.py`, `jobs.py`, service. Solution: route both through one admitted workload coordinator. Verify: concurrent foreground/background submissions cannot bypass the limit. **Completed 2026-09-08:** Foreground and background verification share the service's root-scoped resource lease; admission failure returns HTTP 409. Lease/refusal regression tests added.
- [x] **E02 — P1: Persist job state.** Problem: thread snapshots disappear on process death. Component: `jobs.py`, registry. Solution: durable states, leases/heartbeats and restart reconciliation. Verify: killed workers become interrupted/recoverable rather than silently IDLE. **Completed 2026-09-08:** SQLite job journal, cross-process ownership, cross-worker cancellation and startup reconciliation implemented; an actual killed worker retains its identity and becomes FAILED.
- [x] **E03 — P1: Bound shutdown.** Problem: cooperative cancellation depends on runner cooperation. Component: job shutdown/API lifespan. Solution: finite deadlines and isolated worker termination policy. Verify: a stuck runner cannot indefinitely block server shutdown or corrupt a registry. **Completed for built-in background verification:** `worker.py` enforces a subprocess deadline, kills/reaps uncooperative workers and separates database connections. `test_foundry_worker.py` covers timeout, cancellation, parent monitoring and retained registry integrity. Trusted custom in-process runner hooks remain cooperative; this is not a general-purpose sandbox.
- [ ] **E04 — P1: Consolidate artifact writes.** Problem: multiple `_atomic_json` variants have different cleanup/durability behavior. Component: Foundry modules. Solution: a small shared writer with finite JSON, cleanup, fsync and alias checks. Verify: failure injection leaves previous artifact intact and no misleading partial result.
- [ ] **E05 — P1: Add migrations and restore drills.** Problem: durable storage is incomplete without versioned upgrades and recovery procedures. Component: Foundry/memory SQLite stores. Solution: explicit schema versioning, backups and tested restore. Verify: old supported databases upgrade and a fresh process restores a snapshot with integrity intact.
- [ ] **E06 — P1: Test multiprocess behavior.** Problem: process-local locks do not establish safety with multiple API workers. Component: service/store/jobs/deployment. Solution: enforce single-worker mode or implement interprocess ownership. Verify: competing processes cannot duplicate a workload or corrupt state.
- [ ] **E07 — P2: Simplify by shared contracts, not mass rewrites.** Problem: role and platform layers duplicate evidence IDs, hashes, approvals and errors. Component: model substrate, Foundry, LabOS. Solution: consolidate proven common contracts while preserving independent policy. Verify: consumer compatibility tests pass and fewer implementations own the same invariant.
- [ ] **E08 — P1: Make retained evidence tamper-evident.** Problem: append-only application behavior is not cryptographic protection against file/database editing. Component: evidence store. Solution: immutable exports or hash-linked events with externally pinned roots. Verify: removal/reordering/modification is detected on replay.

### F. Security and privacy

- [ ] **F01 — P0 for hosting: Add identity and authorization.** Problem: local/shared-token mode has no tenant ownership boundary. Component: API, store, gateway. Solution: define users/service identities, object ownership, scoped access and token rotation. Verify: cross-user reads/writes and privilege escalation fail.
- [ ] **F02 — P0 for hosting: Add admission and rate limits.** Problem: authenticated clients can exhaust workload resources. Component: API/gateway/workers. Solution: quotas, request/concurrency limits and explicit overload responses. Verify: sustained load is bounded and recovery works.
- [x] **F03 — P0 when enabling network ingestion: Close DNS and proxy gaps.** Problem: DNS validation and connection resolve separately; urllib may use ambient proxies. Component: `olympus/data/ingestion.py`. Solution: connect to a validated address with correct TLS hostname verification or use controlled egress; disable unintended proxy inheritance. Verify: rebinding and proxy-based routes to prohibited networks fail. **Completed 2026-09-08:** HTTPS connects to the checked numeric address with original TLS hostname; proxy inheritance disabled. Connection and proxy regression tests added.
- [x] **F04 — P1: Harden file write boundaries.** Problem: ingestion reads use descriptor-based protection, while shard output uses resolve/check/write. Component: ingestion `shard`. Solution: anchor writes to trusted directory descriptors and prevent symlink races. Verify: attacker-controlled path swaps cannot redirect output. **Completed 2026-09-08:** Shard writes use directory descriptors, no-follow traversal, unique temporary files, fsync and atomic rename. Parent-symlink swap regression added.
- [ ] **F05 — P1: Bound evidence and model loading.** Problem: several promotion/hash readers use whole-file reads. Component: attestation/promotion/checkpoint loaders. Solution: explicit byte limits, streaming hashes and constrained checkpoint formats. Verify: oversized inputs fail before exhausting RAM; loading cannot execute arbitrary code.
- [ ] **F06 — P1: Test privacy through every output surface.** Problem: secret-safe exceptions do not prove safe prompts, tool output, traces and persisted evidence. Component: logs/API/exports/UI. Solution: documented sensitive fields, access controls, redaction and deletion policy. Verify: seeded private values cannot escape permitted views or routine logs.
- [ ] **F07 — P1: Audit dependencies and history.** Problem: previous zero-advisory runs do not cover the final dependency set or committed secret history. Component: lockfiles, CI, Git history. Solution: rerun locked audits and redacted secret scanning on the release revision. Verify: all findings are remediated or explicitly accepted; exposed credentials are rotated outside Git.
- [ ] **F08 — P0 for hosting: Validate the browser trust boundary.** Problem: static web build success does not prove authenticated same-origin deployment. Component: proxy/gateway, API, web. Solution: explicit TLS/origin/session/CSRF policy suited to the chosen authentication design. Verify: hostile origins, missing credentials and revoked sessions fail in browser tests.

### G. Product, UX and API quality

- [ ] **G01 — P1: Add browser-to-real-API tests.** Problem: mocked fetch tests miss server/proxy drift. Component: Forge web and API. Solution: isolated temporary Foundry root with real start/status/cancel/retry/export/generate flows. Verify: browser tests pass against the built assets and actual API.
- [ ] **G02 — P1: Make verification resource-aware.** Problem: web component tests timed out under severe host load, then passed unchanged in an isolated run. Component: local verification orchestration and test runner. Solution: avoid concurrent heavy suites when admission indicates contention; keep test deadlines intact. Verify: repeated admitted runs pass and overloaded runs are identified explicitly rather than misreported as application regressions.
- [ ] **G03 — P1: Complete accessibility verification.** Problem: labels and responsive inspection do not establish keyboard/screen-reader usability. Component: forms, dialogs, state announcements. Solution: automated accessibility checks plus keyboard/focus and assistive-technology review. Verify: primary journey is usable without a mouse and status changes are announced correctly.
- [ ] **G04 — P1: Explain available model capability.** Problem: a user may confuse the lifecycle verifier with an assistant. Component: model picker, generation panel. Solution: show actual family/status/limitations and disable unsupported modes. Verify: no UI action implies unavailable Hermes capability.
- [ ] **G05 — P1: Validate API contracts across clients.** Problem: manual TypeScript/SDK schemas can drift from FastAPI. Component: OpenAPI, `api.ts`, `sdk.py`. Solution: shared contract fixtures or generated schemas with compatibility tests. Verify: unknown states and malformed responses fail explicitly in both clients.
- [ ] **G06 — P1: Make operations inspectable.** Problem: users need history and actionable failure reasons beyond the latest snapshot. Component: jobs UI/API. Solution: durable run list, safe diagnostics, cancellation outcome and links to evidence. Verify: returning users can inspect the exact completed/interrupted run.
- [ ] **G07 — P2: Measure latency and resource use.** Problem: tiny smoke timings do not establish real user experience. Component: API/web/serving benchmarks. Solution: first-response, throughput, p50/p95, memory and concurrency tests on declared hardware. Verify: a published workload budget and measured degradation behavior.
- [ ] **G08 — P3: Split large UI components only where needed.** Problem: a growing App component can complicate state ownership. Component: `App.tsx`. Solution: extract stable job/model/generation state boundaries after tests cover transitions. Verify: behavior stays identical and responsibilities become independently testable.

### H. Research and publication

- [ ] **H01 — P0 for comparative publication: Freeze one falsifiable question.** Problem: a collection of architectural ideas is not a single supported contribution. Component: research program and paper. Solution: select one precise claim and define a disconfirming result. Verify: evidence can reject the claim without being relabeled as success.
- [ ] **H02 — P1: Create a claim-to-raw-evidence ledger.** Problem: summary JSON and prose can diverge. Component: research ledgers/generated tables. Solution: every result links to protocol, code digest, raw outputs and analysis command. Verify: tables regenerate without manual metric edits.
- [ ] **H03 — P1: Run fair baselines.** Problem: loss versus an untrained model is weak evidence of architectural advantage. Component: evaluation/experiment configs. Solution: compare a competent base and simplest relevant alternative under matched data/compute. Verify: budgets and tuning opportunities are reported for every condition.
- [ ] **H04 — P1: Report uncertainty and failures.** Problem: single runs and aggregate success counts hide variability. Component: analysis scripts. Solution: multiple declared seeds, appropriate confidence intervals/effect sizes, complete failures and multiplicity policy. Verify: raw results reproduce every statistic, including null outcomes.
- [ ] **H05 — P1: Validate causal controls.** Problem: failed negative controls invalidate the intended causal interpretation. Component: Causal-Memory-Use and mechanism studies. Solution: preserve the negative finding, diagnose exchangeability, freeze redesigned controls before new experiments. Verify: any renewed causal claim has passing controls and a separate confirmatory run.
- [ ] **H06 — P0 for Pantheon claims: Respect capability admission.** Problem: retained scientific artifact has 0/20 answered questions for both agents and 0/17 valid agent-authored reports. Component: Pantheon scientific gate. Solution: retain this as a negative capability result; freeze a new study with capable conditions and blinded review. Verify: capable-agent claims use newly admitted conditions, not the existing artifact.
- [ ] **H07 — P1: Distinguish post-study and preregistered rules.** Problem: a safeguard added after outcomes cannot be retroactively preregistered. Component: protocols/gates/paper. Solution: timestamp versions and document deviations. Verify: every analysis identifies whether its decision was frozen before outcomes.
- [ ] **H08 — P1: Complete independent reproduction.** Problem: author reruns do not establish external reproducibility. Component: reproduction package. Solution: provide bounded commands, dependencies, data acquisition and expected artifacts to an independent reviewer. Verify: retained independent logs reproduce the stated result or document deviations.
- [ ] **H09 — P1: Rebuild the paper from verified results.** Problem: a polished proposal PDF can look empirically complete. Component: `research/paper/`, Pantheon paper. Solution: separate proposal, method, results and limitations; verify citations and regenerate figures. Verify: every empirical sentence maps to evidence and no acceptance/publication claim is invented.

### I. Release, operations and documentation

- [ ] **I01 — P0: Review the complete working tree.** Problem: significant changes and required files are untracked. Component: Git source state. Solution: review ownership and scope, form a clean candidate commit and retain unrelated work. Verify: the candidate clone includes every intended source and test; no dirty-tree dependency.
- [ ] **I02 — P0: Finish the complete gate on one immutable revision.** Problem: prior test/build results came from changing revisions; fresh checks are resource-affected. Component: CI/release. Solution: run tests with coverage, lint, strict typing, frontend, Pantheon and package checks on one candidate. Verify: all required results reference the same source digest.
- [ ] **I03 — P0: Prove archive identity and reproducibility.** Problem: builder/validator code exists but final artifacts remain unqualified. Component: build/payload/installed-wheel scripts. Solution: double-build in isolated directories, compare hashes and exact tracked payload, install outside source. Verify: wheel/sdist metadata, payload and runtime all pass.
- [ ] **I04 — P1: Test pinned tool bootstrap.** Problem: installation commands and action pins must work on actual clean runners. Component: `install_tectonic.sh`, CI, release docs. Solution: verify checksum-pinned compiler and locked Python/npm setup. Verify: empty-cache supported runners complete the documented build.
- [ ] **I05 — P1: Add operational runbooks.** Problem: installation alone omits backup, restore, disk-full, migration, worker death and rollback. Component: deployment docs/scripts. Solution: document and exercise each supported failure/recovery path. Verify: drills preserve data and make failures visible.
- [ ] **I06 — P0: Reconcile stale truth documents.** Problem: README/Foundry/status/reality/post-training docs still describe a permanent promotion blocker removed in code. Component: those documents and the older finish checklist. Solution: one canonical status source, generated summaries, explicit historical dates. Verify: documentation states verifier implemented, operational trust and qualifying evidence missing.
- [ ] **I07 — P1: Separate configuration from measured evidence.** Problem: static stage status can be mistaken for a current run. Component: `evidence/status.json`, smoke records. Solution: bind results to source/config/artifact hashes and label stale records. Verify: code changes cannot silently retain a current-verification badge.
- [ ] **I08 — P1: Record rights and publication ownership.** Problem: proprietary/no-license/research-data status varies across repositories. Component: license/model/data cards and release metadata. Solution: owner reviews intended distribution and third-party permissions; record choices. Verify: release materials identify permitted use and all unresolved approvals. This is a documentation/review task, not a claim of legal clearance.
- [ ] **I09 — P1: Configure external release controls.** Problem: local code cannot prove branch protection, publisher identity or deployment status. Component: GitHub/PyPI/host configuration. Solution: owner-configured required checks, publishing identity and approved tag/deployment process. Verify: an authorized release runs successfully with retained provenance.
- [ ] **I10 — P3: Reduce dead documentation/scaffolds.** Problem: overlapping ledgers and archived infrastructure confuse contributors. Component: root docs/archive. Solution: retain one entry point, mark historical records and link archives only for provenance. Verify: onboarding has one supported path and no unused service requirement.
- [x] **I11 — P0: Repair the actual workflow-test mismatch.** Problem: `tests/test_packaging_metadata.py:244` expects Homebrew installation and an inline version assertion; CI uses `scripts/install_tectonic.sh`. Solution: test the current checksum-pinned installer, its required version and its use by both workflows. Verify: focused packaging tests and the real supported bootstrap pass without deleting the compiler-integrity requirement. **Completed 2026-09-08:** Workflow tests match the checksum-pinned installer; the actual macOS arm64 0.17.0 bootstrap completed with checksum verification.
- [x] **I12 — P0: Fix the current strict-type errors.** Problem: `tests/test_promotion_attestation.py:104` supplies `str` for an outcome Literal; `tests/test_foundry.py:388` and `:414` reference non-exported service imports. Solution: type the fixture outcome explicitly and patch/import the actual owner modules or declared test seam. Verify: `mypy olympus tests` passes without broad ignores or weakened strict settings. **Completed 2026-09-08:** Typed attestation outcomes and corrected monkeypatch targets; strict mypy passes without weakened settings.

### J. Wider portfolio: specific next decisions

These items come from the September 7 audit. Recheck each target before implementation; its state may have changed. Component names below deliberately do not invent source filenames that were not inspected in this refresh.

- [ ] **J01 — Causal-Memory-Use / KEEP:** problem: failed naturalistic negative control. Solution: complete a bounded diagnostic paper and separately preregister redesigned controls. Verify: 20,250 controlled and 720 bounded naturalistic records remain traceable; no unsupported causal conclusion.
- [ ] **J02 — FIM / KEEP:** problem: no consistent full-mechanism advantage. Solution: report the negative component matrix, strengthen external controls and delayed-credit tests. Verify: all conditions/seeds retained and conclusions match them.
- [ ] **J03 — Fabric-Induced-Memory and QFIM / MERGE:** problem: duplicated FIM implementations/evidence ownership. Solution: compare behavior, retain unique validated features and tests in one canonical implementation, archive the duplicate origins. Verify: parity and provenance survive consolidation.
- [ ] **J04 — GaussianMemory / KEEP as bounded study:** problem: detached write path and no full-mechanism win. Solution: determine whether delayed credit is required by the hypothesis; test it as a new experiment if so. Verify: gradients and new results support only the new claim; retain the frozen 96-cell negative study.
- [ ] **J05 — Adaptive-Theory-Geometry / KEEP as falsification:** problem: more elaborate selector/prior does not beat simpler controls. Solution: finish transparent negative analysis or narrow to a new falsifiable condition. Verify: original five-seed evidence is preserved.
- [ ] **J06 — DRPT / FIX:** problem: training does not preserve trajectory order needed to test hysteresis. Solution: sequence-preserving data/optimization and history-controlled comparisons. Verify: model distinguishes equal present states with different histories for the intended reason.
- [ ] **J07 — Eigen-JEPA / FIX:** problem: one-seed synthetic evidence. Solution: freeze multiple seeds, useful baselines, generalization splits and mechanism ablations. Verify: rigor gate passes using actual retained runs.
- [ ] **J08 — FI-JEPA / FIX or merge after comparison:** problem: overlapping outputs, weak tests and missing canonical multiseed artifact. Solution: select canonical implementation/protocol and run its declared evidence matrix. Verify: gate consumes one complete, immutable result set.
- [ ] **J09 — NPMS / FIX:** problem: external memory acceptance run incomplete. Solution: run the frozen external acceptance protocol and analyze control failures. Verify: actual predictions/metrics/versions meet its predeclared gate.
- [ ] **J10 — SCDMIT / FIX:** problem: substantial implementation without a complete frozen evidence package. Solution: freeze protocol, add truth ledger, complete splits/seeds and raw artifacts. Verify: another researcher regenerates comparison tables.
- [ ] **J11 — Saphir-Whoof / FIX:** problem: toolkit lacks a real-model result ledger. Solution: execute one bounded multilingual diagnostic and appropriate controls. Verify: real outputs and scoring support the stated diagnostic only.
- [ ] **J12 — RIPII / FIX:** problem: missing external baselines and generalization evidence. Solution: narrow the structured-latent hypothesis and execute a controlled multiseed comparison. Verify: improvement or null result survives held-out evaluation.
- [ ] **J13 — CDW / FIX:** problem: internal/synthetic pilots and deviations limit external claims. Solution: reconcile protocol versions and obtain independent validation before broader claims. Verify: all deviations and retained negative results remain visible.
- [ ] **J14 — Synthica/APEN / KEEP as falsification:** problem: APEN loses to retained baselines. Solution: finish the negative study with fair budgets and failure analysis. Verify: paper does not reverse the observed conclusion through selective reporting.
- [ ] **J15 — IRIS-Draft / ARCHIVE pending evidence recovery:** problem: missing canonical trajectories/source archives. Solution: preserve existing negative manuscript; reopen only if provenance can be recovered. Verify: no claim depends on reconstructed or fabricated data.
- [ ] **J16 — RIS / ARCHIVE pending executable scope:** problem: unsupported scaffold. Solution: archive claims or define one small executable experiment before reviving. Verify: no institutional-grade capability language without evidence.
- [ ] **J17 — EigenFinance / ARCHIVE or remove after owner review:** problem: prior audit found only a license. Solution: preserve history and consolidate the idea into a proposal registry if still wanted. Verify: no empty repository is counted as an implemented project.
- [ ] **J18 — Umbrella foundries / CONSOLIDATE:** problem: many directories represent variants of a few mechanisms, not independent validated contributions. Component: ML4Science, MLInvention and sibling umbrella registries. Solution: canonical mechanism owners, shared runners, separate experiment configs. Verify: each retained project has a distinct falsifiable claim and evidence owner.

### K. Extensions worth considering after the blockers

- [ ] **K01 — P2: Portable evidence replay.** Extend existing provenance into a bundle another machine can inspect, validate and selectively reproduce. Value: the current Foundry already has most artifact identities. Verify: fresh offline inspection detects missing/changed evidence; reproduction uses declared dependencies. Novelty is unproven.
- [ ] **K02 — P2: Evidence-aware grounded research assistant.** Integrate a modest trained Hermes with Atlas retrieval and strict citations/abstention. Value: a concrete useful product path. Verify: task quality, citation correctness, unauthorized evidence exclusion and resource use against a simple retrieval baseline.
- [ ] **K03 — P2: Crash-recoverable bounded experiment runner.** Combine durable jobs, isolated tools and Aion protocol state for one approved workflow. Value: reliable interrupted research work. Verify: fault injection at every transition, no duplicate external effects and complete evidence lineage. Add generality only after this path works.

## First five implementation tasks

1. **Evaluation consistency:** B01/B02/B09 in Foundry evaluation/quantization/promotion. Reject inconsistent reports and inadequate task coverage before they can qualify anything.
2. **Promotion authority and atomicity:** B03–B07 in attestation, promotion and release automation. Pin trust, bind ancestry, parse/hash one snapshot and isolate candidate outputs.
3. **Durable admitted execution:** E01–E03/E06 in API, jobs and store. One workload coordinator, durable state and tested crash/shutdown behavior.
4. **One reproducible release candidate:** I01–I07 and G01/G02. Reconcile docs, isolate the source revision and run the actual complete gate with resource admission.
5. **One trained Hermes research beta:** C01–C07 and D01–D03, after data/base/compute decisions. Integrate a feasible base and prove useful grounded behavior before expanding the other families.

## Completion decision

**Local platform alpha:** conditionally plausible after the local P0/P1 gates and clean release checks pass.

**Public trained models:** NOT READY. No family has a qualifying checkpoint; the recorded tiny smoke result remains 0% exact task/tool completion.

**Top-conference model paper:** NOT READY. Novelty and comparative performance are not established. A bounded software/methodology or negative-result paper may be viable on its own evidence, subject to independent review.

**Project verdict: FIX.** Preserve the working infrastructure and honest negative results. Complete one reliable research product and one measurable model experiment before expanding the portfolio.
