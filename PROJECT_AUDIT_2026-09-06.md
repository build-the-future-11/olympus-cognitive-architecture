# Olympus Project Audit — 2026-09-06

## Audit scope and decision

This audit inspected the local repository at commit `d77d49d` plus the
uncommitted audit-repair working tree: source, tests, workflows, package
metadata, Git history, evidence manifests, model-family code, Pantheon run
artifacts, research ledgers, paper sources, and rendered PDFs. README statements
were treated as hypotheses until matched to executable code or immutable
evidence.

The working tree is intentionally dirty. The audit-repair and release-candidate
metrics explicitly exclude the four unrelated, untracked O1 paths. A pre-existing
Ollama adapter modification was preserved and exercised incidentally, but is not
attributed to this mission. The current working tree has not run on hosted CI
and is not a published release. Local `origin/main` was also stale during
inspection, so remote green status cannot certify this exact tree.

**Decision: preserve and repair the project.** Olympus contains a real, useful
verification and research-governance core. It does not yet contain six trained,
qualified model families, a production autonomous-research system, or a deployable
safe tool runtime. The correct investment is to narrow claims, finish one model
and one trustworthy runtime path, and stop expanding names and scaffolds until
those gates pass.

## What the project actually is

Olympus is currently four related workstreams:

1. A working Python control plane: typed cognitive behaviors, Forge compilation
   and execution, Foundry lifecycle/evidence management, a local API/web surface,
   and model-adapter infrastructure.
2. A LabOS portfolio layer with tested discovery, validation, scheduling,
   execution-history, and reporting paths. It recognizes 36 sibling project
   roots, but 34 remain discovery-only rather than scientifically executable.
3. A serious research-auditing experiment, Pantheon, with controlled positive
   results and preserved evidence that its evaluated agents failed a post-study
   capability floor.
4. An experimental six-role model architecture. Each family has executable
   deterministic components and small trainable PyTorch heads; a bounded
   synthetic replay now composes the six roles. None has a qualifying dataset,
   checkpoint, benchmark result, or promotion.

That distinction matters. The repository is **real engineering**, but its
highest-level product and model claims remain prospective.

## Where the research is going

The technically coherent direction is one revision-pinned decoder substrate,
small family adapters/heads, external retrieval and memory, and deterministic
authority outside the neural model. The six names are roles, not justification
for six independently pretrained base models.

The bounded flow is:

`Olympus-Atlas retrieval -> Prometheus proposal/process check -> Aion approval
gate -> Perseus action -> Kronos observation -> Hermes response -> Aion STOP`.

The implementation now proves those contracts can be composed for one fixed
synthetic, declared-nonmaterial scenario. It does **not** prove the roles are
learned, generally capable, scientifically valid, crash-safe, sandboxed, or
production-ready.

The research sequence that has the best chance of yielding publishable evidence
is:

1. confirm Pantheon's false-consensus question with agents that pass a
   prespecified, externally timestamped capability admission floor;
2. train and evaluate one small Hermes checkpoint on admitted data against a
   matched base-model baseline;
3. use that checkpoint as the only learned base for later role adapters;
4. productionize authority, durability, and sandboxing before allowing material
   tools;
5. add Prometheus/Atlas/Perseus/Kronos/Aion only when each incremental hypothesis
   beats a simpler deterministic or retrieval baseline.

## Research ideas and paper basis

The proposal paper is
[`research/paper/olympus_model_architecture.tex`](research/paper/olympus_model_architecture.tex),
with the claim-to-source boundary in
[`research/paper/report-source.md`](research/paper/report-source.md). Its cited
literature supports design hypotheses, not Olympus results.

| Component | Research hypothesis and literature basis | Implemented evidence | Still unsupported |
| --- | --- | --- | --- |
| Shared substrate | Reuse a decoder-only Transformer with RoPE/GQA rather than inventing a base architecture ([Transformer](https://arxiv.org/abs/1706.03762), [RoPE](https://arxiv.org/abs/2104.09864), [GQA](https://arxiv.org/abs/2305.13245), [Qwen3](https://arxiv.org/abs/2505.09388)); keep formal outputs constrained in the spirit of [PICARD](https://arxiv.org/abs/2109.05093). | Typed workspace, evidence, claim, permission, budget, event-chain, model-identity, tool, approval, and output contracts; a compact adapter decoder trains on synthetic fixtures. | No revision-pinned pretrained backbone is integrated into the family runtime; validation is post-generation, not grammar-constrained decoding; event and authority state are process-local. |
| Hermes | Grounded response, calibrated abstention, and permissioned memory informed by [RAG](https://arxiv.org/abs/2005.11401), [RETRO](https://arxiv.org/abs/2112.04426), [Lost in the Middle](https://arxiv.org/abs/2307.03172), [MemGPT](https://arxiv.org/abs/2310.08560), and calibration work ([arXiv:2207.05221](https://arxiv.org/abs/2207.05221)). | ACL-filtered, extractive workspace response or abstention; response bindings and synthetic attribution-head optimization are tested. | No learned end-to-end generator, admitted 10k/1.2k train/eval corpus, multi-seed baseline, durable memory-approval service, or qualifying checkpoint. |
| Prometheus | Bounded branch proposal and process verification informed by [Tree of Thoughts](https://arxiv.org/abs/2305.10601), [self-consistency](https://arxiv.org/abs/2203.11171), and [process supervision](https://arxiv.org/abs/2305.20050). | Deterministic selection requires host-allowlisted receipts bound to workspace, model, branch, claim, and transition; proposer/verifier heads optimize on synthetic data. | Receipt issuers are not signed; scores are not a scientific oracle; no synthesis benchmark shows better claims than a simpler baseline. |
| Perseus | Typed tool use and recovery informed by [ReAct](https://arxiv.org/abs/2210.03629), [Toolformer](https://arxiv.org/abs/2302.04761), [PICARD](https://arxiv.org/abs/2109.05093), and [ToolSandbox](https://arxiv.org/abs/2408.04682). | Host-registered approvals, exact action/schema/profile binding, defensive snapshots, protected transaction scopes, commit-time reauthorization, sanitized failures, idempotent replay, and recovery-state tests. | `SandboxExecutor` remains a host-supplied protocol, not an enforced sandbox; state is process-local; committed effects cannot be generally rolled back; no stateful benchmark or qualifying policy. |
| Olympus-Atlas | Retrieval-first private-corpus analysis informed by [RAG](https://arxiv.org/abs/2005.11401), [RETRO](https://arxiv.org/abs/2112.04426), [ColBERT](https://arxiv.org/abs/2004.12832), and the distinct published [Atlas](https://arxiv.org/abs/2208.03299). | ACL-first hybrid retrieval, opaque keyed versions, collision-safe passage IDs, ACL-filtered views, and a separately trained synthetic scorer. | No persistent/cross-process index, learned reranker integrated into the service, retrieval benchmark, leakage study, or qualifying checkpoint. |
| Kronos | Offline temporal planning/continual adaptation informed by [Decision Transformer](https://arxiv.org/abs/2106.01345), [Mamba](https://arxiv.org/abs/2312.00752), and [EWC](https://arxiv.org/abs/1612.00796). | Temporal tensors optimize on synthetic fixtures; local adapter admission checks bind a candidate/config/metric record to allowlisted evidence; composition labels its observation nonauthoritative. | Host features are unauthenticated; no causal or temporal benchmark; evidence reports are not parsed; deletion/rollback experiments are not executed; no safe online adaptation. |
| Aion | Governed research lifecycles informed by [Constitutional AI](https://arxiv.org/abs/2212.08073), [AgentBench](https://arxiv.org/abs/2308.03688), [tau-bench](https://arxiv.org/abs/2406.12045), and [The AI Scientist](https://arxiv.org/abs/2408.06292). | A deterministic controller enforces legal stages, frozen-protocol evidence scope, budgets, run/head-bound approvals and audits, model-authority exclusion, and STOP. A learned router trains separately. | No durable/signed authority, general configurable research loop, autonomous-science result, independent audit service, or proof that a learned router improves on the state machine. |
| Pantheon | Test whether pairwise agreement can conceal shared scientific drift by comparing runs to frozen canonical artifacts. | Controlled fault localization is 160/160 versus 100/160 pairwise and 20/160 numeric-only; WDBC direction agreement is 10/10; 17-task OOD execution completed. | The OOD agents produced 0/20 substantive answers and 0/34 agent-authored valid reports. The capability floor failed, so the run cannot estimate naturally occurring false consensus. |

## What is demonstrably working

### Core engineering

- The root Python package imports and its CLI exposes family status, component
  smoke, and bounded composition smoke commands.
- Forge and Foundry have executable lifecycle, compilation, SQLite provenance,
  dataset/checkpoint hashing, local-model adapter, SFT/LoRA/QLoRA, resume, and
  quantization paths. Historical evidence shows a character-bigram candidate at
  perplexity `6.68695` versus a uniform `24.0`, and a tiny deep smoke loss moving
  from `5.78961` to `3.90434`; neither is an assistant-capability result.
- The dated Foundry ledger reports a real `qwen3:0.6b` adapter response and an
  8B RAM/swap failure, but no hash-bound raw response, resource snapshot, or
  Ollama log was retained. These are historical operator observations, not
  reproducible evidence for the current tree.
- Root verification passes 195 tests with the unrelated, untracked O1
  experiment excluded. Ruff and strict mypy pass across 99 source files.
  Branch-aware coverage reaches 85.13% with the O1 module explicitly omitted,
  above the configured 85% gate.
- The web test/build/audit path executes locally. A temporary clean-candidate
  simulation built the intended tracked tree, passed metadata/Twine and exact
  tracked-Python-source wheel/sdist payload checks, then passed an
  installed-wheel composition smoke.
  The dirty root `dist/` directory is stale and is not release evidence.

### Six-role model work

- [`evidence/model_family_smoke_20260906.json`](evidence/model_family_smoke_20260906.json)
  is bound to an explicit nine-file source scope, including the core schema.
  All six synthetic objectives decrease; every checkpoint is explicitly
  nonqualifying and unpromoted.
- [`evidence/model_composition_smoke_20260906.json`](evidence/model_composition_smoke_20260906.json)
  is a source-bound fixed replay. It exercises all six roles, one
  manifest-approved action declared nonmaterial, bounded model/tool accounting,
  sanitized public artifacts, and Aion STOP. It explicitly claims only
  `contract_composition_only` and authorizes no promotion.
- Adversarial tests cover ACL projection, mutable-container revalidation,
  finite/strict JSON hashing, action-bound approvals, stale/forged heads,
  transaction aliasing and TOCTOU, output exfiltration, reservation tokens,
  retries, failure terminalization, evidence/citation binding, and public-record
  cross-validation.

### Pantheon research integrity

- The controlled benchmark, public WDBC study, and frozen OOD run have preserved
  manifests, hashes, normalized artifacts, analyses, and negative results.
- The controlled validator reports 480 classifications, 200 executions, and 20
  public-case runs.
- The Pantheon suite passes 31 tests with 2 platform-specific macOS sandbox tests
  skipped in the current environment.
- The project no longer interprets null/null as agreement. The authoritative
  post-study verdict is
  `NEGATIVE_RESULT_ARTIFACT_COMPLETE_CAPABILITY_FLOOR_FAILED`.

## What is broken or materially incomplete

### 1. The named model product does not exist yet

There are zero qualifying family checkpoints. No tracked model weights were
found. Synthetic loss decreases demonstrate that modules execute, not that they
perform their named tasks. None of the six families has the admitted dataset,
frozen multi-seed evaluation, matched baseline, safety evaluation, model card,
and fresh-process serving proof required by its own roadmap.

Foundry's local promotion evaluator can parse and hash-link candidate reports,
but evaluation, quantization, license approval, and serving fields are supplied
by the caller rather than reproduced by an independent authority. It now adds
an unconditional `independent_evidence_authority` blocker and currently cannot
emit a release manifest. That hard blocker must later be replaced with actual
verification of runner-produced, independently verifiable attestations.

This is the highest-impact gap because it is the difference between an
architecture/reference implementation and a model project.

### 2. The composed runtime is a process-local reference, not a safe agent

The repaired composition is meaningful, but deliberately narrow:

- pending state, approvals, audit heads, transaction state, secrets, and indexes
  live in memory;
- the executor's sandbox/profile and “nonmaterial” behavior are host assertions;
- there is no durable recovery after process death or a general compensation
  mechanism after an external side effect;
- authority records are not signed by an external identity service;
- the one smoke path uses fixed synthetic evidence and a fixed tool fixture;
- most learned heads do not drive the deterministic services; the replay's one
  randomly initialized Kronos forecast is explicitly nonauthoritative and
  cannot select the action.

Calling this a production agent, end-to-end learned system, sandbox, autonomous
researcher, or scientific result would be incorrect.

### 3. The audited tree is not a released artifact

The repository has only seven local commits and no local tags. The exact audited
tree is uncommitted and therefore not covered by the
[latest hosted CI result](https://github.com/build-the-future-11/olympus-cognitive-architecture/actions/runs/34040612580),
which ran against an earlier remote `main`. The repository's
[Releases page](https://github.com/build-the-future-11/olympus-cognitive-architecture/releases)
has no release, the expected
[PyPI project page](https://pypi.org/project/olympus-cognitive-architecture/)
does not exist, and there is no verified remote deployment. The only Docker
Compose file is under archived, unused infrastructure.

Release automation is substantially better after this audit—it now rejects a
dirty source tree and checks distribution metadata, tracked Python payloads,
and required resource presence. Owner-controlled repository/PyPI configuration,
branch protection, a private vulnerability-reporting channel, an exact tagged
commit, and authorized publication remain external gates.

### 4. Pantheon answered an infrastructure question, not its target empirical question

The OOD run is complete and useful precisely because it is negative: both agents
failed the post-study capability floor. It therefore cannot test whether capable,
independent research agents exhibit false consensus. Re-running the same weak
agents would add volume, not evidence.

The publication workflow also required a repair during this audit because a
structural artifact-completeness gate could regenerate a conflicting,
misleading `PUBLICATION-READY` artifact after the later negative scientific
readiness verdict. The durable rule must remain: file completeness is not
scientific readiness.

### 5. Architecture and truth maintenance are too concentrated

The audited Python package scope is about 17.6k lines and tests about 6.6k, but
`olympus/models/composition.py` alone exceeds 2k lines. It mixes public record
schemas, integrity validation, orchestration, recovery, privacy projection, and
family translation. That concentration made repeated cross-field security bugs
possible.

Truth is manually duplicated across README, project status, truth maps, stage
evidence, family documentation, architecture notes, paper source, TeX, and
Pantheon sidecars. Tests now catch important drift, but the maintenance model is
still brittle.

### 6. Residual security and reliability limits

- No common credential or private-key patterns were found; `.env` and `.env.*`
  files are ignored except for an intentional `.env.example` template;
  mutations default to loopback/token controls, ingestion defaults to network
  denial, URLs are bounded/allowlisted, and redirects are denied.
  Configured filesystem roots are enforced, but the default policy permits
  byte-bounded reads without a root restriction.
- Network ingestion still has a potential DNS rebinding/time-of-check versus
  connection-resolution gap unless the validated address is pinned at connect
  time.
- Local ingestion resolves/checks and then reopens read and shard-write paths;
  in an attacker-writable directory, a symlink/path replacement can escape a
  configured root. Descriptor-relative, no-follow open/write handling is still
  needed.
- Python `>=3.14` is a deliberate but narrow deployment requirement and will
  reduce ecosystem compatibility.
- The system data volume was nearly full during the audit and initially broke
  ordinary temporary-file/SQLite tests. Temporary files were redirected to the
  external workspace; free space later recovered into roughly the 16--19 GiB
  range. Disk capacity is therefore an environment incident to monitor, not a
  current code failure.

## Slop and dead weight

The central model/runtime code is no longer placeholder-only, but several things
should stop being mistaken for progress:

- Family names, architecture prose, and tiny synthetic heads are not models.
  Keep them as reference implementations, but do not add another named family
  until one existing family qualifies.
- The character bigram, cognitive fixture, JEPA experiments, and `HermesNano`
  style artifacts are valid test devices only. Product-facing wording should
  never imply they establish language-model capability.
- Pantheon contains 2,045 of 2,235 tracked files, including a large historical
  run corpus with many seeded-fault and invalidated artifacts. The evidence
  should be retained, but exposed through a content-addressed index and compact
  release bundle rather than treated as hand-browsable source.
- Manual status duplication is dead-weight risk. Generate derived tables and
  paper artifact inventories from one machine-readable registry/evidence index.
- Ignored virtual environments, build outputs, test artifacts, and caches make
  the working directory multiple gigabytes versus approximately 15.1 MiB of
  tracked logical payload. They are operational residue, not project output.
- `archive/unused-infrastructure` is honestly labeled; keep it outside release
  payloads or delete it after any unique historical rationale is recorded.
- The unrelated O1 protocol files were preserved but excluded from both the
  audited repair metrics and the release-candidate boundary. They require their
  own review and provenance before they can become project evidence.

## Architecture assessment

### Strong choices

- Authority is designed to remain outside model logits.
- Claims, evidence, tool calls, identities, permissions, budgets, events, and
  promotion state have explicit types and negative tests.
- The family registry separates implementation, validation, and promotion.
- Retrieval filters before model exposure; public composition artifacts are
  redacted and hashes/IDs avoid simple private-state oracles.
- Aion makes STOP a successful terminal outcome and binds approvals/audits to
  run, protocol, head, and action context.
- Evidence generation is deterministic/source-bound and refuses to convert
  synthetic smoke into promotion.

### Structural problems

- `composition.py` is a validation/orchestration god module.
- `WorkspaceState` has one runtime identity rather than a first-class map of
  independently versioned component identities.
- Trainable heads and deterministic services are mostly parallel demonstrations,
  not one learned inference graph.
- Storage, authority, sandbox, and audit interfaces are not backed by durable,
  separately administered implementations.
- Registry, source-digest lists, smoke runners, docs, and exact-family tests must
  all be edited to add a family; this invites drift.

The smallest correct refactor is to split public immutable records, trusted
state/recovery, and orchestration without changing behavior; then add durable
ports behind those contracts. A framework rewrite would destroy useful tested
invariants and is not justified.

## Research-integrity assessment

The repaired repository now makes the most important distinctions correctly:

- **Completed positive evidence:** controlled Pantheon fault localization,
  public WDBC comparison, engineering tests, component optimization smokes, and
  one bounded contract replay.
- **Completed negative evidence:** the artifact-backed Pantheon OOD
  capability-floor failure and deep-model task/tool gates that did not pass.
  The local 8B resource failure is only a historical operator report because
  no hash-bound raw resource or Ollama log was retained.
- **Hypotheses:** specialized adapters improve named family metrics; capable
  agents exhibit detectable false consensus; temporal adapters help planning;
  learned Aion routing beats deterministic control.
- **Frozen decisions:** Pantheon V4 protocol/task selection and source identity
  were internally frozen before execution but not externally preregistered;
  family claim boundaries, evidence source digests, and explicit no-promotion
  fields.
- **Unsupported conclusions:** model-family capability, autonomous science,
  robust continual learning, production sandbox safety, scientific synthesis
  validity, or venue-quality novelty.

No result should be manufactured to fill those gaps. A failed, capability-limited
study is more valuable than a fabricated success.

## Changes implemented during this audit

This was not a cosmetic documentation pass. The working tree now includes:

- strict workspace/output boundaries and authorized pre-model projections;
- strict `Authorization: Bearer` parsing for token-protected API mutations;
- loopback peer/host validation, credential-safe SDK transport checks, public
  error redaction, inherited-secret redaction, and lossless typed memory JSON;
- deep snapshot revalidation and finite JSON/hash-integrity checks;
- six import-verified family implementations and explicit unpromoted registry;
- hardened Aion run/head/protocol/evidence/approval/audit bindings;
- hardened Perseus approval, schema, transaction, reservation, retry, receipt,
  error-sanitization, and mutable-state boundaries;
- safer Atlas identity/ACL boundaries, Prometheus receipt/claim bindings, Kronos
  checkpoint evidence binding, and extractive Hermes citations;
- a two-phase six-role reference replay with manifest-bound approval, exact
  budgets, public/private separation, terminal fail-closed behavior, and a
  fixed nonqualifying composition smoke;
- canonical source-bound family and composition evidence;
- a promotion authority blocker that prevents caller-authored reports from
  producing a release manifest;
- package payload validation, untracked-file release rejection, and an
  isolated installed-wheel lifecycle/composition regression;
- corrected Pantheon negative-result truth, V4 source-drift stop, draft V5
  source snapshot, scoped integrity checks, pytest import behavior, publication
  workflow, research ledgers, architecture documentation, and rebuilt,
  visually inspected model-architecture and Pantheon papers.

These improvements make the reference implementation defensible. They do not
substitute for data collection, training, evaluation, external replication, or
deployment.

---

**PROJECT STATUS:** Salvageable

**VERIFIED WORKING:**

The Forge/Foundry control plane, evidence lifecycle, root API/CLI/web checks,
195-test in-scope root suite, six synthetic family component smokes, one bounded
source-bound six-role replay, package build/install checks, and Pantheon's
controlled/public/OOD artifact pipeline are demonstrably real. The authoritative
evidence explicitly records zero qualifying or promoted family checkpoints.

**CRITICAL FAILURES:**

1. No named model family has a qualifying dataset, checkpoint, comparison, or
   capability result; the primary model-product claim is unfinished.
2. Composition is process-local and host-trusting, with no enforced sandbox,
   durable recovery, signed authority, or safe general material execution.
3. The audited repair tree is uncommitted and has no exact hosted-CI, release,
   PyPI, or deployment proof.
4. Pantheon's definitive OOD agents failed the post-study capability floor,
   so the target false-consensus hypothesis remains untested.
5. A monolithic composition boundary and duplicated truth surfaces make future
   correctness/security drift likely.

**SLOP / DEAD WEIGHT:**

Do not count family branding, synthetic losses, fixture models, generated prose,
ignored artifacts, archived deployment scaffolds, or invalidated historical runs
as model progress. Retain evidence, but index and package it compactly. Generate
duplicated status prose from one registry, and refuse a seventh family until one
of the six qualifies.

**MISSING TO COMPLETE:**

A licensed/revision-pinned base, admitted train/eval data, frozen matched
baselines, multi-seed family evaluations, qualifying checkpoints, durable state,
an enforced sandbox, externally verifiable approval/audit identity, crash and
compensation tests, capable-agent Pantheon confirmation, exact-commit hosted CI,
security reporting, a tagged release, and a verified deployment.

**TOP 5 NEXT ACTIONS:**

1. **Run the capable-agent Pantheon confirmation.** Files/components:
   [`NEXT.md`](NEXT.md),
   `OlympusC/05_pantheon/configs/local_ood_protocol.json` (historical V4; never
   overwrite), `configs/local_ood_protocol_v5_source_manifest.json` (draft
   only), `external/corebench/manifest.jsonl`, `scripts/run_external_study.py`,
   `scripts/run_final_external.sh`, and `src/pantheon/external/providers.py`
   under `OlympusC/05_pantheon/`. Change: create versioned
   output namespaces, bind exact model identities, then preregister and freeze a
   V5 confirmation with capable provider-diverse agents, a non-null completion
   floor, separate hosts, and blinded human adjudication. The existing V5 source
   manifest is a draft snapshot, not execution authority. Why: the
   existing 0/20 result cannot answer the false-consensus question. Expected
   outcome: either an estimable cross-agent result or another explicit bounded
   negative. Verify: frozen manifest hash predates execution; capability floor
   passes before consensus analysis; reports are agent-authored; independent
   adjudicators reproduce the tables.
2. **Qualify one Hermes 0.6B checkpoint before expanding families.** Files/components:
   [`research/HERMES_BASE_CANDIDATES.md`](research/HERMES_BASE_CANDIDATES.md),
   `datasets/hermes-smoke/source.jsonl`, `olympus/foundry/data_pipeline.py`,
   `olympus/foundry/sft.py`, `olympus/foundry/eval_suite.py`,
   `olympus/foundry/quantization.py`, `olympus/foundry/promotion.py`, and
   `olympus/models/hermes.py`. Change: pin license/revision, admit at least
   the planned 10k/1.2k grounded-response split, train three seeds, and compare
   against the untouched base and retrieval-only baseline on citation,
   abstention, leakage, calibration, latency, and task quality. Why: one real
   checkpoint is more informative than six synthetic heads. Expected outcome:
   the first honestly promotable family or a useful falsification. Verify:
   immutable data/checkpoint hashes, held-out document splits, preregistered
   gates, multi-seed confidence intervals, and fresh-process serving identity.
3. **Turn the trust boundary into a durable runtime.** Files/components:
   `olympus/models/composition.py`, `olympus/models/perseus.py`,
   `olympus/models/aion.py`, `olympus/models/substrate.py`, plus new durable
   store, authority, and sandbox adapters. Change: persist replay,
   head, budget, transaction, audit, and private capability state; run tools in
   an OS-enforced deny-by-default sandbox; bind signed external approval to the
   execution manifest; add compensation protocols for material effects. Why:
   process-local equality checks cannot safely authorize production actions.
   Expected outcome: restartable, auditable, least-privilege execution. Verify:
   kill/restart at every transition, duplicate delivery, stale/replayed approval,
   network/filesystem escape, partial side-effect, secret-exfiltration, and
   independent audit tests.
4. **Split composition and generate truth from one source.** Files/components:
   `olympus/models/composition.py`, `olympus/models/registry.py`,
   `olympus/models/smoke.py`, `olympus/models/composition_smoke.py`,
   [`evidence/status.json`](evidence/status.json), `README.md`,
   `PROJECT_STATUS.md`, `TRUTH_MAP.md`, `research/SOURCE_OF_TRUTH.md`, and the
   exact-list/evidence tests in `tests/test_model_registry.py`,
   `tests/test_model_smoke.py`, `tests/test_model_composition_smoke.py`, and
   `tests/test_stage_registry.py`. Change: separate immutable public records,
   trusted recovery state, orchestration, and family adapters; generate status
   tables and artifact lists from registry/evidence rather than editing copies.
   Why: the current concentration and duplication caused real invariant and
   documentation drift. Expected outcome: smaller review surfaces with identical
   behavior and automatic truth consistency. Verify: existing adversarial suite
   remains green; mutation/property tests cover every boundary; a deliberate
   registry/evidence mismatch fails CI.
5. **Ship one exact, supportable release.** Files/components: `SECURITY.md`,
   `RELEASE.md`, `.github/workflows/ci.yml`, `.github/workflows/release.yml`,
   `pyproject.toml`, `MANIFEST.in`, `scripts/validate_distribution_payload.py`,
   `scripts/validate_installed_wheel.py`, `olympus/api.py`,
   `apps/forge-web/src/api.ts`, and a new explicit deployment profile. Change:
   configure private vulnerability intake,
   branch protection, trusted publishing, clean-checkout builds, exact revision
   evidence, release SBOM/attestation, and one documented deployment profile.
   Why: local success is not distribution or operations evidence. Expected
   outcome: a reproducible package and a bounded deployed control plane, still
   labeled experimental. Verify: hosted CI on the tagged SHA, isolated install,
   signature/attestation and payload checks, PyPI import/CLI smoke, authenticated
   live health/mutation tests, and rollback rehearsal.

**FINAL VERDICT:** FIX — keep the verified control plane, evidence discipline,
Pantheon artifacts, and bounded model-role contracts. Stop presenting architecture
and synthetic optimization as model completion; qualify Hermes, confirm Pantheon
with capable agents, productionize the trust boundary, simplify the monolith, and
release only from an exact green revision. Archiving or deleting now would discard
substantial real engineering and unusually honest negative evidence; continuing
without those constraints would turn it into slop.
