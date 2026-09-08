# Olympus — single-prompt execution contract for usable model releases

Copy this entire document into the implementation task. Execute the requirements,
not another checklist rewrite. A single prompt coordinates the work; it cannot
guarantee trained models, compute, rights or scientific results exist. Keep an item
open until its acceptance evidence exists. This document is a specification, not
a release announcement.

## Mission and boundaries

Deliver downloadable, runnable, accurately labelled releases for Hermes, Prometheus,
Perseus, Atlas, Kronos and Aion. Distinguish standalone trained models, adapters,
deterministic components and orchestration applications. Do not relabel one as another.
No parameter-scale, sentience, AGI, superiority or publication claim without evidence.
Do not silently replace the original large-model ambitions with small components
and declare them fulfilled. Smaller experimental releases must be separately named.

Inspect current source, AGENTS instructions, git status/history, manifests, locks,
CI, tests, model registry, docs, research protocols and all prior audit/execution
reports available in the checkout. Some previously discussed implementation may
be uncommitted or absent from GitHub: verify presence before relying on it.
Preserve unrelated edits, frozen protocols, raw outcomes and negative findings.
No paid compute, public deployment, weight upload, data destruction or external
tool effects without the required authority. Do not print secrets.

Build a dependency-aware internal execution queue. Implement, test, record evidence
and continue. Do not stop at documentation or green mocks. Report external blockers
precisely and continue independent local work. Missing implementation is not an
external blocker. Do not weaken tests, promotion gates, ACLs or licensing checks.

## A. Freeze and represent six deliverables

- [ ] In `docs/model-families.md`, `research/RESEARCH_PROGRAM.md` and
  `olympus/models/registry.py`, define a versioned capability contract per family:
  intended users/tasks, inputs/outputs, prohibited behavior, supported hardware,
  runtime, base identity, actual parameter count and release acceptance criteria.
- [ ] Resolve Perseus generalist-versus-tool-action, Atlas intelligence-versus-retrieval
  and Kronos AGI-versus-planning role drift explicitly with the owner. Continue
  reusable infrastructure while those materially different product decisions await approval.
- [ ] Represent implementation, checkpoint availability, installation verification,
  capability evaluation and promotion as separate machine-readable states.
- [ ] Label synthetic/reference artifacts non-qualifying. Tests must prevent them
  from appearing as available trained assistants in CLI, API, UI or release manifests.

Acceptance: every family has one consistent contract across registry, documentation,
model card, user interface and evaluation. Unavailable releases remain unavailable.

## B. Shared runnable model foundation

- [ ] Measure target host RAM, free disk and available acceleration. Record facts,
  not guessed hardware capacity. Select one supported backend first; inspect the
  existing `olympus/foundry/ollama.py` integration before adding another abstraction.
- [ ] Research candidate checkpoint licenses and official runtime documentation.
  Select immutable base revisions; record publisher, tokenizer, chat template,
  actual parameter count, quantization format, hashes and redistribution rights.
- [ ] Implement explicit acquisition with size/disk checks, interruption recovery,
  bounded network timeouts, integrity verification and atomic completed installs.
  Reject partial/corrupt files and unsafe archive paths. Never execute unreviewed
  remote model code or automatically download a different model on failure.
- [ ] Define a strict inference interface: conversation, model identity, context/output
  budgets, streaming events, completion reason, cancellation, timeout and usage.
  Implement the actual supported backend and real integration tests.
- [ ] Apply the correct tokenizer/template. Bound history using measured tokenization
  where available; disclose conservative approximations. Truncate whole turns with
  notice, and reject single overlong inputs. Never silently truncate required policy.
- [ ] Detect identity changes, missing weights, malformed responses, empty outputs,
  OOM and interrupted generation. Do not report length-limited output as normal completion.
- [ ] Provide CLI commands to list availability, inspect identity, run, diagnose and
  remove only explicitly selected artifacts. Document backend storage and cleanup.
- [ ] Test CPU and each advertised accelerator independently. Do not advertise
  Windows or hardware combinations unsupported by the actual POSIX/runtime code.

Acceptance: install outside the checkout, acquire an approved real checkpoint,
generate repeatedly, cancel, restart and reproduce model identity. Measure loading,
peak memory, first-token latency and throughput on every advertised configuration.

## C. Hermes — complete local assistant

Primary paths: `olympus/models/hermes.py`, `olympus/models/hermes_chat.py` if present,
`olympus/cli.py`, `olympus/foundry/ollama.py`, `tests/test_hermes_model.py`.

- [ ] Complete `olympus hermes chat`: explicit base identity, multi-turn conversation,
  streaming, cancellation, `/clear`, `/exit`, readable failures and visible budgets.
  No conversation persistence by default; disclose backend logging separately.
- [ ] Separate ordinary chat from document-grounded mode. Ordinary chat must not
  claim citation verification. Preserve the existing extractive baseline for comparison.
- [ ] Implement bounded document input, authorization filtering before retrieval,
  immutable evidence identifiers and hash checks before prompt construction.
- [ ] Treat evidence as untrusted data. Validate generated citation identifiers and
  exact quoted spans. Distinguish quotations from paraphrases; refuse invented support.
- [ ] Implement insufficient/conflicting evidence handling; evaluate abstention.
  Do not present lexical overlap or uncalibrated heads as factual probabilities.
- [ ] Integrate and evaluate learned grounding heads, or label them experimental
  rather than implying the user-facing runtime uses them.
- [ ] Implement explicit memory proposal approval, persistence, inspect/delete and
  authorization if memory is included in the frozen release contract. No silent writes.
- [ ] Qualify one actual checkpoint/adapter; record training lineage if trained.

Acceptance: clean-machine multi-turn chat and authorized-document QA, including
malicious documents, absent support, altered sources and unauthorized evidence.
Frozen evaluation compares base-only, extractive baseline and generative Hermes.

## D. Prometheus — specialization

Primary paths: `olympus/models/prometheus.py`, Foundry training/evaluation/registry.

- [ ] Freeze one initial domain and observable adaptation workflow before adding domains.
- [ ] Implement validated data import, rights/provenance, disjoint train/dev/test,
  preprocessing isolation and contamination checks.
- [ ] Implement actual adaptation, checkpoint/adapter save/load, immutable base
  binding and explicit selection at inference. No dummy domain routing responses.
- [ ] Add held-out in-domain, out-of-domain and forgetting evaluations against the
  unchanged base with disclosed training/inference budgets.
- [ ] Preserve failed runs and support rollback to the previous approved adapter.

Acceptance: a user can adapt or install a real specialist artifact and run it from
a fresh process. Evidence establishes the claimed domain behavior and its tradeoffs.
No claim of a trained 128B model without a real corresponding checkpoint.

## E. Perseus — execute the approved role

Primary paths: `olympus/models/perseus.py`, composition and executor integrations.

- [ ] If tool-action release: typed tool schemas, allowlisted executors, server-side
  authorization, human approvals for material actions, argument validation, bounded
  execution, timeouts and safe error handling.
- [ ] Bind approvals to exact action parameters and identity. Implement durable
  idempotency/receipts, retries and compensation where semantically possible.
  Reject stale/replayed approvals and unauthorized scope changes.
- [ ] Test executor crash before/after effects; do not promise exactly-once behavior
  for arbitrary external systems. Never call a host assertion a security sandbox.
- [ ] If generalist release: acquire/train the approved actual model, implement its
  inference bundle and evaluate generalist capability separately from tool execution.

Acceptance: approved role works end-to-end with retained failure/permission evidence.
A tool component cannot satisfy the proposed 1.024T generalist requirement.

## F. Atlas — grounded retrieval system

Primary paths: `olympus/models/atlas.py`, `olympus/data/ingestion.py`, retrieval/store.

- [ ] Implement supported ingestion formats with size/path validation, provenance,
  deduplication, updates, deletion and restart persistence.
- [ ] Enforce ACLs before retrieval and prompt construction, including caches.
  Ensure deleted/revoked content does not remain accessible through stale indexes.
- [ ] Integrate ranked retrieval with grounded answers and resolvable source spans.
- [ ] Evaluate retrieval recall/ranking and answer/citation quality separately against
  a simple retrieval baseline; retain no-answer and conflict cases.

Acceptance: fresh install can ingest, query, update, revoke and delete real documents,
with verified access boundaries. No sentience or unsupported deep-intelligence claim.

## G. Kronos — bounded planning system

Primary paths: `olympus/models/kronos.py`, workspace, persistence and execution feedback.

- [ ] Freeze bounded planning tasks, budgets, success/failure conditions and controls.
- [ ] Implement typed plans, precondition validation, feedback, replanning, budget
  exhaustion, explicit STOP and recovery from interrupted execution.
- [ ] Test impossible goals, conflicting constraints, stale state and tool failures.
- [ ] Compare with a simple planner/base-model baseline using equal disclosed budgets
  and held-out tasks. Record failed plans and external effects.

Acceptance: reproducible planning behavior within declared bounds. An 8T or AGI
release remains unfulfilled without actual corresponding evidence and artifacts.

## H. Aion — durable governed composition

Primary paths: `olympus/models/aion.py`, `composition.py`, Foundry jobs/store/worker.

- [ ] Integrate the approved family implementations through explicit versioned
  contracts; propagate missing capability and failures instead of fabricating results.
- [ ] Persist workflow transitions, approvals, budgets, cancellation and receipts.
- [ ] Implement crash recovery at admission, execution and finalization, including
  already-running observers and preserved abandoned-job history.
- [ ] Test multiple processes, supervisor death, duplicate submissions, stale approvals,
  cancellation races and restart. Keep external-effect semantics explicit.
- [ ] Separate historical evidence inspection from new release authorization;
  implement expiry/revocation/replay policy in downstream evidence consumers.

Acceptance: one complete approved workflow survives injected failures with correct
history, bounded work, intact permissions and no falsely reported success.

## I. Product, security and evidence

- [ ] Finish relevant Forge web/API flows: actual availability, model selection,
  loading/streaming/cancellation, readable errors, durable history and explicit retries.
- [ ] Verify keyboard access, focus, mobile layout, empty/loading/error states and
  returning-user behavior in the rendered app. Preserve existing visual identity.
- [ ] Default local services to loopback; require explicit remote exposure and
  authentication. Enforce authorization server-side, validation, request limits and
  appropriate rate/concurrency limits. No secrets/prompts/documents in default logs.
- [ ] Freeze per-family evaluation protocols before final outcomes: tasks, datasets,
  splits, seeds, metrics, baselines, ablations, budgets and acceptance thresholds.
- [ ] Retain raw predictions, versions, artifact hashes, failures and uncertainty.
  Use independent runners/signers for claimed release evidence, not candidate keys.
- [ ] Distinguish software tests, synthetic smoke checks, capability results and
  comparative research. Publication readiness is a separate gate, not a consequence
  of installing successfully. Preserve null and negative findings.

## J. Release and final verification

- [ ] Add regression tests for each fixed requirement; run targeted tests immediately.
- [ ] Run repository Ruff, strict mypy, full pytest with the existing coverage gate,
  frontend tests/typecheck/build, dependency audits and relevant real-backend tests.
- [ ] Build reproducible wheel/sdist; run Twine and tracked-payload validation.
  Review intended source contents; never stage unrelated work merely to pass a gate.
- [ ] Extend installed-wheel tests to all advertised family commands using actual
  approved artifacts. Exercise restart, streaming, errors and offline operation
  wherever offline use is promised. Mocks alone do not pass model readiness.
- [ ] Write one versioned manifest/model card per release: actual identity, supported
  hardware, installation/run commands, rights, evidence, limits and hashes.
- [ ] Test the exact final download on a clean supported machine. A developer's
  virtual environment or local uncommitted source is not release evidence.
- [ ] Commit only reviewed in-scope changes; inspect staged diff for secrets/artifacts.
  Push only with authorization, without force-pushing. Publish weights/services only
  after rights, qualification and distribution authorization are satisfied.

## Completion rule and final response

For EACH family report separately: implementation complete, real artifact available,
clean install/run verified, capability gate passed and release authorized. Every YES
must link to exact retained evidence and tested revision. Otherwise report NO with
the precise unfinished task or external dependency. Do not average these into a
misleading portfolio percentage.

Final report: completed behavior; important files; commands with PASS/FAIL/NOT RUN/
BLOCKED; real model identities and measurements; remaining local work; exact external
inputs required; six individual READY/EXPERIMENTAL/NOT READY decisions; commit/push
status. No universal perfection or all-model readiness claim from a partial pass.
