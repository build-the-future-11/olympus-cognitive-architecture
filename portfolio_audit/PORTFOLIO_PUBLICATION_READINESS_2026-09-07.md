# Olympus research portfolio: adversarial audit and execution report

Audit date: 2026-09-07  
Canonical report source: this file  
Machine-readable companion: `portfolio_registry.json`  
External-source ledger: `claim_source_ledger.md`

## 1. Executive summary

Olympus is a real, tested alpha research platform. It has a usable Python API,
CLI, local web console, evidence-aware Foundry, portfolio discovery, compact
PyTorch reference components for six named roles, and one fixed synthetic
six-role contract replay. Its software is substantially better than a mock-up.

It is **not** a family of trained frontier models. Hermes, Prometheus, Perseus,
Olympus-Atlas, Kronos, and Aion are role architectures and compact reference
implementations. There are no 12B, 128B, 1.024T, or 8T checkpoints; no licensed
training corpus at those scales; no qualifying multi-seed capability suite; no
independent safety evaluation; and no production inference deployment. Words
such as “perfect,” “sentient,” and “AGI” have no supporting artifact and must
not appear as factual release claims.

Across the wider portfolio, several repositories contain credible engineering
and research artifacts, but the strongest scientific result is usually a
negative or bounded diagnostic result. The best near-paper candidates are:

1. **Causal-Memory-Use** as a diagnostic/falsification paper. It has 20,250
   controlled local-model calls and 720 bounded LongMemEval-S calls. Its
   strongest naturalistic result is that the matched negative control fails,
   preventing the intended causal interpretation.
2. **MIT-STAN-064 / the research foundry data-contract work** as a systems or
   methodology paper after independent, distributed, and deployed validation.
3. **GaussianMemory**, **FIM**, and **Adaptive-Theory-Geometry** as carefully
   bounded negative mechanism studies after licensing and external baselines.
4. **Synthica/APEN** as a falsification report: APEN loses clearly to standard
   neural-operator baselines in the retained experiments.

No model is ready for unrestricted public use. Olympus itself is suitable for
an **expert-only alpha source release** after a clean, reviewed commit passes the
distribution gate. Scientific publication is at best conditional and must be
paper-specific.

## 2. Internal project model

The codebase is three projects sharing one repository:

1. **Olympus runtime/product** — cognitive behavior graphs, Forge compiler,
   FastAPI service, SDK, CLI, memory/data components, and web console.
2. **Model Foundry and governance** — datasets, experiments, checkpoints,
   evaluations, registry, promotion rules, resource admission, and evidence
   logs.
3. **Research portfolio and named roles** — Hermes, Olympus-Atlas, Prometheus,
   Perseus, Kronos, Aion, Pantheon, LabOS, and the sibling research repositories.

The most coherent product is not “an 8T AGI.” It is an **evidence-governed local
research foundry that makes unsupported promotion difficult**. That is both more
credible and more differentiated than the current aspirational branding.

## 3. What actually works

### Olympus

- Python suite: **231 passed**, with **85.21% branch coverage**.
- Ruff: pass on maintained source.
- strict mypy: pass across `olympus` and `tests` (101 source files).
- web suite: **13 passed**; production Vite build passes.
- dependency audits: Python locked graph reports no known vulnerabilities;
  npm reports zero vulnerabilities.
- fresh Foundry golden path: dataset, experiment, character-bigram checkpoint,
  held-out evaluation, export, model registry, and evidence status all execute.
- wheel: builds, passes Twine metadata validation, installs outside the source
  tree, and passes isolated identity/Foundry/composition verification.
- Pantheon boundary: **31 passed, 2 skipped**; evidence validator passes; the
  structural publication gate passes; the scientific gate correctly reports a
  complete negative-result artifact whose capability floor failed; PDF builds.
- six named reference roles: importable typed implementations, compact trainable
  modules, deterministic guardrails, and focused smoke tests.
- fixed six-role replay: a manifest-bound, process-local, declared non-material
  action path reaches Aion STOP and fails closed under tested component errors.

### Strong sibling artifacts

- MIT-Stanford-Princeton-Research: **364 tests pass** and **14/14** foundry
  projects verify.
- CDW: **268 tests pass** and the full lint gate now passes.
- Synthica/APEN: **102 tests pass**; retained experiments reject the proposed
  APEN advantage.
- Causal-Memory-Use: **25 tests pass**; controlled and bounded naturalistic raw
  evidence is persisted and hash-bound.
- FIM: **38 passed, 8 skipped**; real 24-cell component matrix.
- GaussianMemory: **9 passed**; real frozen 96-cell negative study.
- NPMS: **30 passed**; large controlled and causal matrices exist.
- SCDMIT: **73 passed**; substantial model and baseline code exists.
- Adaptive-Theory-Geometry, Eigen-JEPA, DRPT, QFIM, RIPII, Saphir-Whoof, and
  IRIS-Draft all have executable test coverage, although evidence maturity is
  much lower.

Test success verifies the tested engineering surface. It does not validate
novelty, broad generalization, scale, deployment, or the named model claims.

## 4. Initial audit: ranked problems

### P0 — critical

#### P0.1 — Named-model claims do not correspond to trained models

- **Evidence:** `docs/model-families.md` explicitly marks every role
  `EXPERIMENTAL_SMOKE_NOT_PROMOTED`; the artifacts are compact synthetic
  components and one reference replay.
- **Affected:** `olympus/models/`, `research/architectures/`, model-family
  marketing and any public model card.
- **Why it matters:** users could mistake interface-level implementations for
  released 12B/128B/1.024T/8T models.
- **Smallest correct fix:** preserve the architectures, keep the status labels,
  and train one rights-cleared Hermes-sized beta before discussing a family.
- **Safe now:** documentation correction is safe; large training is not
  possible or evidenced in this environment.

#### P0.2 — Release provenance gate correctly fails on the dirty source state

- **Evidence:** the built wheel and sdist contain six new Python files that are
  not tracked by Git. `scripts/validate_distribution_payload.py` rejects them.
  The isolated installed-wheel check nevertheless passes.
- **Affected:** `olympus/evaluation/o1.py`, `olympus/models/composition.py`,
  `olympus/models/composition_smoke.py`, and their three test files.
- **Why it matters:** an artifact assembled from unreviewed/untracked source is
  not reproducible from the release commit.
- **Smallest correct fix:** review, track, commit, rebuild from a clean checkout,
  then rerun the unchanged gate.
- **Safe now:** no. Weakening the validator or staging an unreviewed 129-path
  working tree would create false release confidence.

#### P0.3 — Scientific promotion is intentionally nonfunctional

- **Evidence:** `PROJECT_STATUS.md` and the Foundry ledger state that candidate
  promotion has a hard-coded independent-evidence blocker because evaluation,
  quantization, licensing, and serving inputs are not runner-attested.
- **Affected:** Foundry promotion/report paths and release policy.
- **Why it matters:** this prevents false promotion, but also means no named
  model can become a real release through the current pipeline.
- **Smallest correct fix:** define signed/hash-bound runner attestations and
  verify them in a fresh process; retain human approval.
- **Safe now:** schema and tests are safe; inventing attestations is not.

#### P0.4 — Multiple research repos fail their own evidence gates

- **Evidence:** MLInvention has three collection errors from a missing
  independent-reproduction handoff; ML4SematicIntelligence has one stale COGS
  assertion; FI-JEPA's evidence gate lacks a canonical three-seed artifact;
  Eigen-JEPA has one seed.
- **Why it matters:** these projects cannot support publication or release
  claims even when most unit tests pass.
- **Smallest correct fix:** repair stale expectations only where evidence is
  already present; otherwise run the predeclared experiment or retain FAIL.
- **Safe now:** the umbrella repos were read-only in this environment. Their
  failures are reported, not hidden.

### P1 — high leverage

#### P1.1 — The composition proves interoperability, not autonomy

Its authority, event history, reservations, transactions, and cached outputs
are process-local. It lacks an enforcing sandbox, durable recovery, authenticated
executor profile, and general configurable graph. The current implementation is
well bounded; the risk is overinterpretation.

#### P1.2 — Research projects are fragmented and duplicated

FIM, Fabric-Induced-Memory, and QFIM share substantial code and concepts.
ML4Science and MLInvention each expand 64 ideas into many shallow directories.
This obscures ownership and multiplies maintenance without multiplying evidence.

#### P1.3 — Most focused repos cannot legally support an open public release

Synthica/APEN, Adaptive-Theory-Geometry, DRPT, Eigen-JEPA, FI-JEPA, FIM,
Fabric-Induced-Memory, GaussianMemory, IRIS-Draft, NPMS, QFIM, RIPII, and RIS
have no repository license. CDW explicitly grants no distribution permission.
Olympus is intentionally proprietary. Public visibility is not permission to
reuse or redistribute.

#### P1.4 — External validity is the dominant research gap

Most evidence is synthetic, local, single-dataset, single-split, or
resource-bounded. Stronger systems such as SCDMIT and NPMS lack a canonical
frozen external result package. This cannot be solved by more internal unit
tests.

#### P1.5 — Product deployment remains unverified

The web console builds and API tests pass, but no remote deployment, load test,
production auth boundary, backup/restore drill, or operational SLO is evidenced.

### P2 — important

- Add durable signed evidence/authority storage and crash-recovery tests.
- Add accessibility and browser end-to-end tests for the web console.
- Introduce common result schemas and a single project registry across sibling
  repos.
- Separate frozen evidence from mutable development outputs everywhere.
- Resolve vendored OpenML Electricity redistribution terms in the MIT portfolio.
- Expand model cards with exact hardware, energy/cost, intended users, and
  prohibited uses after real checkpoints exist.

### P3 — polish

- Normalize spelling (`ML4SematicIntelligence` versus “Semantic”).
- Replace historical hype in repository summaries with registry-derived state.
- Consolidate duplicated setup commands and paper-generation instructions.
- Archive old generated logs after preserving hashes.

## 5. Claims versus evidence

| Claim | Verdict | Evidence |
| --- | --- | --- |
| “Hermes is a perfect 12B local assistant” | Unsupported | No 12B checkpoint, training run, assistant benchmark, or safety evaluation |
| “Prometheus is a perfect 128B specialist” | Unsupported | Branch/proposer/verifier reference components only |
| “Perseus is a 1.024T lightweight generalist” | Internally contradictory and unsupported | Action/transaction reference components only; a trillion-parameter model is not lightweight by ordinary deployment constraints |
| “Atlas is sentient/beyond the market” | Unsupported and not scientifically operationalized | Hybrid retrieval service and scorer only; no sentience test or comparative evaluation |
| “Kronos is an 8T AGI” | Unsupported | Compact temporal planner and local checkpoint gate only |
| “Aion autonomously governs research” | Unsupported as a general claim | Deterministic process-local controller for one bounded protocol path |
| “Olympus has executable family architectures” | Verified, bounded | Source, tests, component smoke, and fixed composition smoke |
| “Olympus is production ready” | False | Alpha label, dirty release state, no deployment evidence, process-local authority |
| “Causal memory is proven on naturalistic tasks” | False | The naturalistic irrelevant-deletion control is behaviorally active |
| “APEN improves neural operators” | Refuted in retained studies | FNO/Transformer/ResNet baselines outperform APEN; no-memory ablation is slightly better |
| “Negative results are absent or failures of progress” | False | Several of the portfolio's most credible contributions are falsification studies with preserved evidence |

The external prior-art boundary is recorded in `claim_source_ledger.md`. In
particular, LongMemEval already decomposes long-term memory evaluation, CMI
already applies causal intervention to agent memory, FNO/DeepONet are established
operator baselines, and model/dataset cards and ML data validation are mature
areas. Novelty must be narrower than those concepts.

## 6. Model-by-model architecture and genuine state

### Hermes

- **Intended role:** evidence-grounded local assistant and final response layer.
- **Implemented:** ACL-filtered workspace views, content hashes, extractive span
  citations, abstention, failure-aware runtime, and separate trainable
  attribution/confidence/memory heads.
- **Missing:** pretrained/generative integration, approved persistent memory,
  held-out calibration, safety suite, rights-cleared dataset, checkpoint, and
  serving artifact.
- **Decision:** keep the architecture; it is the only sensible first model
  product. Start with a licensed 1B--3B base or adapter, not a fictional 12B.

### Olympus-Atlas

- **Intended role:** ACL-first hybrid retrieval and evidence assembly.
- **Implemented:** BM25, hashed dense retrieval, reciprocal-rank fusion, late
  interaction, opaque corpus version IDs, and a separate trainable scorer.
- **Missing:** durable cross-process index, real private-corpus study, integrated
  learned retrieval, scale/latency measurements, and comparative recall metrics.
- **Decision:** keep as an infrastructure component. “Sentient” is categorically
  unsupported.

### Prometheus

- **Intended role:** specialist branch generation, comparison, and verification.
- **Implemented:** typed hypotheses, canonical receipts, host-bound receipt
  hashes, proposer/verifier losses, diversity loss, and fail-closed selection.
- **Missing:** authenticated independent receipt issuer, integrated learned
  selection, equal-compute baselines, autonomous research validity, and domain
  specialist checkpoints.
- **Decision:** keep as a research planning component, not a 128B model product.

### Perseus

- **Intended role:** tool/action execution and recovery.
- **Implemented:** versioned action rules, preconditions, action-bound approvals,
  commit-time reauthorization, defensive snapshots, sanitized receipts,
  idempotency, and trainable policy/recovery heads.
- **Missing:** actual sandbox, durable transactions, authenticated executor
  behavior, material-effect rollback, crash recovery, and stateful benchmarks.
- **Decision:** keep, but do not market it as a generalist model. It is a
  capability kernel.

### Kronos

- **Intended role:** temporal state, forecasting, planning, and adaptation.
- **Implemented:** validated temporal events, time-decayed recurrence,
  probabilistic forecast/action heads, joint loss, tensor-only checkpoint
  loading, and local evidence-bound staging.
- **Missing:** causal/calibrated forecasting, real drift/retention study,
  parsed signed reports, deletion/rollback replay, and useful action-selection
  evidence.
- **Decision:** keep as experimental temporal research; no AGI or 8T claim.

### Aion

- **Intended role:** research protocol/state governance and kill switch.
- **Implemented:** protocol-bound state machine, legal transitions, budgets,
  host-registered approvals, evidence-bound audits, and explicit STOP behavior.
- **Missing:** durable signed authority/audit store, configurable service graph,
  external replication, and evidence that learning improves on deterministic
  control.
- **Decision:** keep as the governance layer. It is not autonomous science.

## 7. Research-project disposition

| Project | Genuine state | Decision |
| --- | --- | --- |
| MIT-Stanford-Princeton-Research | Best foundry implementation; 364 tests and 14/14 verification, but 0 submission-ready projects | KEEP |
| Causal-Memory-Use | Strong bounded diagnostic with a credible naturalistic falsification | KEEP |
| GaussianMemory | Frozen 96-cell negative mechanism study; detached write path explains a core failure | KEEP |
| FIM | Real component matrix; no consistent advantage for the full mechanism | KEEP |
| Adaptive-Theory-Geometry | Five-seed/WDBC study; response geometry transfers but proposed blend/selector do not win | KEEP |
| Synthica/APEN | Strong falsification infrastructure; APEN underperforms FNO/Transformer/ResNet in retained tests | KEEP |
| CDW | Four retained three-seed pilots, mostly negative or protocol-deviating | FIX |
| NPMS | Strong controlled infrastructure; external LoCoMo/LongMemEval run absent | FIX |
| SCDMIT | Substantial code and baselines; missing canonical frozen multi-seed evidence | FIX |
| Saphir-Whoof | Multilingual diagnostic toolkit; no real-model result ledger | FIX |
| Eigen-JEPA | Executable synthetic prototype; one seed and failed rigor gate | FIX |
| DRPT | Early implementation; does not yet test its stated hysteresis mechanism | FIX |
| RIPII | Executable structured-latent prototype with no credible novelty/evaluation package | FIX |
| FI-JEPA | One-test scaffold; evidence gate fails | FIX |
| ML4Industry | Useful audit framework, but no deployment/outcome/conference-ready study | FIX |
| ML4SematicIntelligence | Negative COGS analysis with one stale test and one Transformer seed | FIX |
| ML4Science | 64 directories collapse to eight shared mechanisms; consolidate | MERGE |
| MLInvention | 64 proxy mechanisms, 0/64 worst-condition support; missing independent handoff | MERGE |
| Fabric-Induced-Memory | Older weaker duplicate of FIM | MERGE into FIM |
| QFIM | Duplicate FIM branch with no independent claim/evidence package | MERGE into FIM |
| IRIS-Draft | Negative archival manuscript with missing canonical trajectory provenance | ARCHIVE |
| RIS | Unsupported “institutional-grade” scaffold with no tests/evidence | ARCHIVE |
| EigenFinance | License-only empty repository | DELETE or merge the name into a real project |

## 8. Architecture quality

### Strengths

- Contracts make authority and evidence boundaries explicit.
- Promotion defaults to failure, which is the right direction for research.
- Source hashes, dataset identities, provenance, and result ledgers are first-
  class concepts rather than README prose.
- The local Foundry golden path is cheap enough to run routinely.
- The fixed composition has unusually careful action/approval binding and
  sanitized failure behavior for a reference implementation.

### Structural weaknesses

- Named roles mix “model,” service, controller, and policy-kernel concepts. A
  single lifecycle label makes unlike artifacts look equivalent.
- `OlympusC/05_pantheon` is effectively a nested research project inside the
  product repo, with its own environment, evidence, paper, and gates.
- Portfolio orchestration knows many project roots but lacks one authoritative
  registry for license, evidence stage, owner, and frozen release.
- Process-local security semantics are described in great detail but cannot
  survive crashes or hostile executors.
- Duplicated focused repos fragment evidence and invite conflicting fixes.

### Simplification direction

Use four explicit artifact types: **runtime service**, **trainable component**,
**checkpointed model**, and **research study**. A named family may own several
types, but only a checkpointed model can receive a model release status. Merge
the FIM variants, collapse 64-idea umbrellas to mechanism families, and make the
portfolio registry the source for status pages.

## 9. Security and reliability

### Verified protections

- No confirmed committed credential, private key, or `.env` secret was found in
  the scanned scope.
- mutating/generation API endpoints are loopback-first; the SDK refuses bearer
  tokens over non-loopback plaintext HTTP.
- action schemas, authorization bindings, ACL filtering, finite JSON, path
  safety, idempotency, and sanitized failure cases have targeted tests.
- Python and web dependency audits found no known vulnerabilities in the locked
  graphs at audit time.

### Remaining risks

- The “sandbox executor” is only an injected interface plus a host assertion;
  it does not enforce OS isolation.
- authority, reservations, transactions, and audit state are process-local.
- no production reverse-proxy/auth deployment is verified.
- no rate/abuse/load test or disaster-recovery exercise is evidenced.
- numerous sibling repos lack licenses and formal data-use records.
- the MIT foundry's vendored Electricity data needs redistribution review.

## 10. Research integrity

### Completed evidence

- Raw and aggregate artifacts exist for the retained Causal Memory, FIM,
  GaussianMemory, Adaptive-Theory-Geometry, CDW, and APEN studies.
- Multiple projects retain negative/null findings instead of rewriting the
  hypothesis after the fact.
- Olympus and Pantheon gates separate artifact completeness from scientific
  readiness.

### Negative results that must remain visible

- APEN loses to standard baselines; no-memory is slightly better in its ablation.
- FIM's full mechanism does not consistently win.
- GaussianMemory's full system is not best and its write path is not trained by
  delayed loss.
- Adaptive-Theory-Geometry's proposed blend/selector do not improve on simpler
  controls; the neural prior ablation favors removing the prior.
- Causal Memory's naturalistic irrelevant-deletion control fails badly.
- Pantheon is structurally complete but both agents answer 0/20 retained
  questions, so its capability floor fails.

### Unsupported conclusions

- No portfolio project establishes AGI, sentience, perfect specialization,
  frontier generality, state of the art, or public-production safety.
- Internal synthetic seed counts do not establish domain generalization.
- A compiled PDF is not a publication result.
- A fail-closed gate is evidence of integrity, not evidence that the candidate
  passed.

## 11. Scores

These are reviewer judgments, not measured KPIs. “Initial” is the state before
this pass; “current” reflects the concrete fixes below. The release target is
the minimum for a credible expert alpha, not a perfect system.

### Engineering

| Dimension | Initial | Current | Release target | Justification |
| --- | ---: | ---: | ---: | --- |
| Architecture | 6.5 | 6.8 | 8.0 | Strong contracts; fragmented roles and nested projects |
| Correctness | 6.5 | 7.0 | 8.5 | Broad tests; fixed real numeric/admission/input bugs |
| Code quality | 7.0 | 7.2 | 8.0 | Typed and linted core; large legacy/duplicate surface |
| Maintainability | 5.5 | 5.8 | 8.0 | 24 projects and duplicate families remain costly |
| Reliability | 6.0 | 6.4 | 8.5 | Fail-closed behavior; process-local durability |
| Test coverage | 7.5 | 7.8 | 8.5 | Olympus 85.21%; several projects have 0--2 tests |
| Observability | 6.0 | 6.2 | 8.0 | Evidence logs exist; no production telemetry/SLO |
| Performance | 3.5 | 3.5 | 7.0 | Almost no representative load/latency evidence |
| Security | 6.0 | 6.4 | 8.5 | Good validation; sandbox and durable authority absent |
| Developer experience | 6.0 | 6.3 | 8.0 | Locked setup and docs; multi-repo state is confusing |

### Product

| Dimension | Initial | Current | Release target | Justification |
| --- | ---: | ---: | ---: | --- |
| Usefulness | 5.5 | 5.8 | 8.0 | Foundry/audit platform is useful; models are not products |
| UX | 5.0 | 5.0 | 7.5 | Functional local console; no complete user study/journey |
| Onboarding | 6.0 | 6.3 | 8.0 | Quick start exists; Python 3.14 and nested envs add friction |
| Feature completeness | 4.0 | 4.2 | 8.0 | Core lifecycle works; promotion/deployment do not |
| Differentiation | 5.5 | 6.0 | 8.0 | Evidence governance is real; model branding is not |
| Consistency | 5.0 | 5.4 | 8.0 | Truth maps help; sibling status/docs conflict |
| Polish | 5.0 | 5.1 | 7.5 | Builds cleanly; no deployment/accessibility evidence |
| Accessibility | 3.5 | 3.5 | 7.5 | No focused accessibility audit or browser E2E |
| Deployment readiness | 3.0 | 3.2 | 8.5 | No production deployment or operations proof |

### Research and credibility

| Dimension | Initial | Current | Release target | Justification |
| --- | ---: | ---: | ---: | --- |
| Novelty | 4.0 | 4.0 | 7.0 | Broad ideas overlap prior work; narrower audit novelty possible |
| Hypothesis clarity | 6.0 | 6.5 | 8.5 | Best projects are falsifiable; many scaffolds are not |
| Methodological rigor | 5.5 | 6.0 | 8.5 | Strong gates in pockets; weak external validity overall |
| Reproducibility | 6.5 | 7.0 | 9.0 | Many frozen artifacts; dirty/untracked release state remains |
| Baseline quality | 5.0 | 5.0 | 8.5 | Some fair baselines; many prototypes lack them |
| Evaluation quality | 5.0 | 5.3 | 8.5 | Real bounded studies; no family-level qualifying suite |
| Ablations | 5.0 | 5.2 | 8.0 | Good FIM/GMF/ATG pockets; uneven elsewhere |
| Statistical validity | 4.5 | 4.8 | 8.0 | Multi-seed pockets; single-seed/small selected samples remain |
| Leakage prevention | 6.5 | 6.8 | 9.0 | Explicit split checks in Olympus; inconsistent portfolio-wide |
| Falsifiability | 7.0 | 7.8 | 9.0 | Negative results and gates are the strongest asset |
| Claim/evidence alignment | 6.0 | 7.2 | 9.0 | Truth docs corrected; brand aspirations still need restraint |
| Paper readiness | 4.5 | 4.8 | 8.0 | Several credible drafts, none unconditionally ready |
| Repository hygiene | 4.0 | 4.2 | 8.0 | 129 dirty paths and many missing licenses |
| External credibility | 3.5 | 3.8 | 8.0 | No independent replication, users, deployment, or acceptance |

Overall reviewer score: **5.4 initial → 5.9 current**. The pass improved real
correctness and truth alignment; it did not create missing models or experiments.

## 12. Work completed in this pass

1. **DRPT float32 safety:** trajectories that remain finite in NumPy float64
   but overflow during Torch float32 export now fail closed instead of returning
   `inf` tensors.
2. **FIM/Fabric/QFIM memory admission:** `min_store_probability=0` no longer
   causes default random low-salience writes. The threshold is deterministic;
   optional stochastic exploration is explicit and training-only. Diagnostics
   separate effective write probability from raw score probability.
3. **QFIM image training:** rank-4 image batches are no longer mistaken for
   sequence batches, which previously discarded target structure and caused a
   Conv2d shape failure.
4. **CDW lint boundary:** the immutable Project 00 snapshot is excluded as a
   whole, preventing nested Ruff configuration from linting frozen provenance
   as maintained source. Full Ruff now passes.
5. **Causal Memory truth reconciliation:** `RESEARCH_TRUTH.md`,
   `FINAL_RESEARCH_REPORT.md`, and `EVIDENCE_LEDGER.md` now record the completed
   20,250-call and 720-call studies and distinguish the bounded evidence from the
   still-failing legacy full-benchmark gate.
6. **Cross-project verification:** all focused suites were executed; six paper
   PDFs were rendered and inspected; Olympus package, web, Foundry, Pantheon,
   dependency, lint, type, and test gates were exercised.
7. **Portfolio consolidation:** this report, external claim/source ledger, and
   machine-readable registry now give one evidence-first disposition for every
   project.

## 13. Important files changed

- `DRPT/src/data.py` — fail-closed float32 trajectory conversion.
- `FIM/fim/memory/consolidation.py` — correct write admission semantics.
- `Fabric-Induced-Memory/fim/memory/consolidation.py` — aligned diagnostics.
- `QFIM/fim/memory/consolidation.py` — correct write admission semantics.
- `QFIM/fim_experiments/train.py` — preserve image-batch targets.
- `CDW/pyproject.toml` — treat the frozen snapshot as provenance, not lint input.
- `Causal-Memory-Use/{RESEARCH_TRUTH.md,FINAL_RESEARCH_REPORT.md,EVIDENCE_LEDGER.md}`
  — reconcile current bounded evidence and negative-result boundary.
- `Olympus/portfolio_audit/` — canonical report, registry, and source ledger.

Other pre-existing Olympus working-tree changes were inspected and preserved.
They were not rewritten or staged wholesale.

## 14. Verification matrix

### Olympus

- **BUILD: PASS (with release caveat)** — Python wheel/sdist and web production
  assets build. Twine passes. Installed-wheel verification passes.
- **DISTRIBUTION PAYLOAD: FAIL** — correctly detects six untracked Python files.
- **TESTS: PASS** — 231 passed; 85.21% branch coverage.
- **LINT: PASS** — maintained-source Ruff.
- **TYPECHECK: PASS** — strict mypy, 101 files.
- **WEB TESTS: PASS** — 13 passed.
- **DEPENDENCY AUDIT: PASS** — no known Python or npm vulnerabilities found.
- **FOUNDRY GOLDEN PATH: PASS** — fresh temporary root.
- **PANTHEON STRUCTURAL GATE: PASS**.
- **PANTHEON SCIENTIFIC GATE: NEGATIVE RESULT** — artifact complete; capability
  floor failed, which is the correct retained outcome.
- **REMOTE DEPLOYMENT: NOT RUN** — not authorized and no target configured.

### Focused repositories

| Repository | Result |
| --- | --- |
| CDW | 268 passed; Ruff pass |
| Synthica/APEN | 102 passed; one numerical warning |
| Causal-Memory-Use | 25 passed; legacy full-benchmark publication gate FAIL |
| Adaptive-Theory-Geometry | 10 passed; one environment/core-count warning |
| DRPT | 5 passed after regression fix |
| Eigen-JEPA | 8 passed; research rigor gate FAIL (one seed) |
| FI-JEPA | 1 passed; evidence gate FAIL |
| FIM | 38 passed, 8 skipped |
| Fabric-Induced-Memory | 26 passed, 7 skipped |
| GaussianMemory | 9 passed |
| QFIM | 19 passed, 7 skipped after fixes |
| RIPII | suite passed |
| SCDMIT | 73 passed |
| Saphir-Whoof | 9 passed |
| IRIS-Draft | 2 passed |
| NPMS | 30 passed |
| RIS | NOT RUN — no tests exist |
| EigenFinance | NOT RUN — no implementation exists |

### Umbrella repositories

| Repository | Result |
| --- | --- |
| ML4Science | 63 passed |
| ML4Industry | 125 passed, 3 skipped |
| ML4SematicIntelligence | 54 passed, 1 failed (stale COGS label) |
| MLInvention | 91 passed, 3 collection errors (missing handoff artifact) |
| MIT-Stanford-Princeton-Research | 364 passed; 14/14 foundry verification |

## 15. Venue and publication direction

The ICLR 2027 abstract and paper deadlines are 2026-09-18 and 2026-09-25.
Submitting a model-family paper by that deadline would reward schedule pressure
over evidence. The official guide also requires double blindness and a nine-page
main text. **Do not submit the Olympus family claim to ICLR 2027.**

Better routes:

- **Causal-Memory-Use:** TMLR after independent semantic scoring and a
  preregistered replacement negative control. TMLR's rolling process and stated
  emphasis on technical correctness suit a diagnostic/negative result. ARR's
  2026-10-12 cycle is an alternative if the work is reframed squarely as NLP.
- **MIT-STAN-064 / Foundry:** a future MLSys submission after a multi-process,
  distributed, deployed comparison against existing data validation and ML
  metadata systems.
- **GaussianMemory/FIM/ATG/APEN/CDW:** TMLR or a focused workshop only after
  licensing, a frozen common baseline harness, external datasets, and independent
  reproduction. Their value is mechanism falsification, not positive SOTA.
- **Olympus models:** no venue until at least one checkpointed family model has
  a rights-cleared dataset, multi-seed baselines, calibration/safety analysis,
  and an independently reproducible evaluation.

## 16. Unfair advantages worth building

### A. Evidence graph that controls promotion

Combine Olympus's hash-bound lifecycle, Pantheon's distinction between
structural and scientific readiness, and MIT-STAN-064's data contracts into a
single attestable evidence graph. Most experiment trackers record metrics;
fewer make it impossible to confuse “artifact complete” with “scientifically
supported.” Feasibility is high because the primitives exist. Difficulty:
medium. Expected impact: the strongest product and systems-paper story.

### B. Cross-project falsification benchmark

Turn the retained negative mechanisms (memory use, operator memory, geometry,
APEN, consensus) into a common suite that requires nulls, counterfactuals,
baselines, and stop rules. The portfolio already has diverse failure modes and
real negative evidence. Difficulty: medium-high. Expected impact: a distinctive
benchmark for whether proposed mechanisms actually contribute.

### C. Authority-safe local assistant beta

Use Hermes as the response layer, Olympus-Atlas as ACL-first retrieval, Perseus
as the action kernel, and Aion as governance around a licensed small base model.
Do not train six models first. Difficulty: high but bounded. Expected impact: a
real user-facing artifact that exercises the platform's strongest architecture.

The first credible version of A already exists: the Foundry golden path,
structural/scientific gate split, machine-readable status, and fixed composition
smoke all execute. The next work is durability and independent attestation.

## 17. Final adversarial review

### Senior production engineer

Rejects because the tree is dirty, payload provenance fails, authority is
process-local, the sandbox is declarative, and no production operations evidence
exists.

### Top-conference reviewer

Rejects a unified “Olympus models” paper because there are no model checkpoints
or comparative results and novelty is an architecture proposal. May engage with
the bounded negative-result papers if claims and controls remain strict.

### Skeptical user

Rejects the named models because there is nothing to download and run as a
12B/128B/1T/8T assistant. May value the local Foundry after a clean alpha
release and an end-to-end tutorial using a real small model.

### Open-source maintainer

Rejects broad contribution because Olympus is proprietary, many sibling repos
are unlicensed, duplicate repos lack ownership, and the working tree is huge.

### Technical diligence reviewer

Values the unusually honest evidence boundaries and working infrastructure but
discounts all frontier-model valuation claims to zero. Investable technical
value is the evidence-governance platform and the disciplined negative-result
pipeline.

## 18. Remaining P0/P1 issues

1. No trained, evaluated, promoted named model exists.
2. Olympus cannot produce a release from the current dirty/untracked tree.
3. Foundry scientific promotion lacks real independently verified attestations.
4. No production sandbox, durable authority store, or crash-safe transaction
   layer exists.
5. No remote deployment or operational reliability evidence exists.
6. Research external validity is weak and inconsistent across the portfolio.
7. Most focused repositories lack a reusable public license.
8. Duplicate and empty projects obscure the small number of credible studies.
9. Causal Memory's naturalistic negative control fails and must be redesigned,
   not statistically massaged.
10. MLInvention, ML4SemanticIntelligence, FI-JEPA, and Eigen-JEPA fail their
    current evidence/test gates.

## 19. Next five highest-leverage actions

### 1. Produce one clean, reviewable Olympus release candidate

- **Files/components:** the more than 120 current working-tree paths; especially the six
  untracked composition/O1 source and test files, `pyproject.toml`, CI/release
  workflows, and evidence ledgers.
- **Change:** split changes into auditable commits, remove generated scratch,
  ensure every packaged source is tracked, and build from a clean clone.
- **Why:** this is the first causal release blocker and makes every artifact
  reproducible from a commit.
- **Expected outcome:** payload validator, Twine, isolated-wheel, tests, lint,
  typecheck, web build, and Pantheon all pass on the same clean SHA.
- **Verify:** `test -z "$(git status --porcelain)"`, then execute the CI/release
  commands without exclusions or local path tricks.

### 2. Replace the Foundry's hard-coded evidence blocker with attestation checks

- **Files/components:** Foundry promotion, evaluation, quantization, license
  review, serving verification, schemas, and promotion tests.
- **Change:** define versioned runner-attestation envelopes bound to code SHA,
  dataset, checkpoint, config, environment, raw outputs, and signer/authority;
  validate them in a fresh process while preserving human approval.
- **Why:** promotion is currently safe but permanently impossible.
- **Expected outcome:** invalid/missing/mismatched attestations fail; one
  repository-owned synthetic candidate can pass structurally without gaining a
  scientific/model-quality claim.
- **Verify:** mutation tests for every binding, restart tests, independent
  verifier test, and a retained negative scientific gate.

### 3. Build a real Hermes alpha, not the whole mythic model family

- **Files/components:** `olympus/models/hermes.py`, Atlas retrieval, Perseus/Aion
  boundaries, dataset manifests, Foundry trainers/evaluators, model card.
- **Change:** select a revision-pinned, rights-cleared 1B--3B base; train an
  adapter on a documented corpus; integrate grounded generation; freeze
  baselines and safety/calibration suites before outcomes.
- **Why:** one useful model validates the architecture more than six disconnected
  synthetic heads or speculative parameter counts.
- **Expected outcome:** downloadable checkpoint with bounded intended use,
  reproducible training, held-out results, abstention calibration, and explicit
  failure cases.
- **Verify:** three or more seeds where stochastic training matters, equal-
  compute base/RAG baselines, contamination checks, red-team set, fresh-process
  inference, and independent reproduction.

### 4. Consolidate the research portfolio around mechanism families

- **Files/components:** FIM, Fabric-Induced-Memory, QFIM; the 64-project
  ML4Science and MLInvention trees; `portfolio_registry.json`.
- **Change:** preserve frozen evidence, choose one canonical implementation per
  mechanism, archive duplicates, add migration maps, and make one registry own
  license/stage/evidence/status.
- **Why:** duplication is consuming maintenance effort and makes evidence easy
  to misattribute.
- **Expected outcome:** roughly eight coherent research families with clear
  ownership instead of dozens of shallow “projects.”
- **Verify:** no evidence hash changes, archived READMEs point to canonical
  homes, shared suites pass, and registry entries resolve to one active owner.

### 5. Finish the two best papers through new evidence, not prose

- **Files/components:** Causal-Memory-Use protocol/scoring/results/paper and
  MIT-STAN-064 data-contract/evidence pipeline.
- **Change:** for Causal Memory, preregister an exchangeable negative control,
  add blinded semantic scoring with agreement, and evaluate production-like
  memory systems. For MIT-STAN-064, add distributed/fault-injection/deployed
  comparisons against established data-validation/metadata baselines and resolve
  data licensing.
- **Why:** these are the only contributions presently close enough that new
  experiments can plausibly produce credible submissions.
- **Expected outcome:** two bounded papers whose claims survive skeptical review,
  including if results remain negative.
- **Verify:** immutable protocol before calls, raw row-level outputs, multi-seed
  or clustered uncertainty as appropriate, independent reproduction, complete
  claim/evidence table, and venue-format gate.

## 20. Release/publication decision

**NOT READY**

Olympus is a credible alpha engineering project, but the current source state is
not releasable and the named models do not exist as trained products. The best
research projects are conditionally publishable only after targeted external
validation and licensing. The portfolio should continue, but it should continue
as an evidence-governed Foundry plus a small number of consolidated falsification
studies—not as five unsupported frontier models.

---

**PROJECT STATUS:** Major Repair

**VERIFIED WORKING:** Olympus runtime/Forge/Foundry engineering, the fixed
six-role synthetic composition, the web build, isolated wheel runtime,
Pantheon's structural negative-result artifact, and the listed focused research
test/evidence packages.

**CRITICAL FAILURES:** no trained/promoted named models; dirty/untracked release
state; nonfunctional scientific promotion attestation; no enforcing sandbox or
durable authority; weak external validity; missing licenses.

**SLOP / DEAD WEIGHT:** EigenFinance, RIS's unsupported institutional-grade
shell, duplicate FIM/Fabric/QFIM implementations, and the shallow expansion of
64-idea umbrella repositories beyond their shared mechanism families.

**MISSING TO COMPLETE:** one clean release SHA, attested promotion, a real
Hermes checkpoint, durable security boundaries, production deployment evidence,
consolidated ownership, licensing, and independent external experiments.

**TOP 5 NEXT ACTIONS:** clean release candidate; real attestation verification;
one bounded Hermes alpha; portfolio consolidation; targeted new evidence for
Causal Memory and MIT-STAN-064.

**FINAL VERDICT:** **FIX** — preserve the working infrastructure and strongest
negative studies, merge duplicates, archive unsupported scaffolds, delete the
empty repository, and stop presenting architecture names as completed models.
