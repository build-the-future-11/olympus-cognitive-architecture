# Olympus Research Feature Mega Checklist

## Purpose and scientific boundary

This is an implementation-and-evidence roadmap for turning Olympus into a more
useful experimental research system.  It is **not** evidence that these
capabilities already work, nor authorization to run a new protected study.
Items marked **[GATED SCIENCE]** require a separately frozen protocol, admitted
data, budget, controls, and falsifier before protected outcomes are observed.
Items marked **[ENG]** can be developed against artificial fixtures and
unprotected data.  A completed checkbox must link a hash-bound receipt, source
revision, configuration, and the listed acceptance artifact.

### Scientific starting point

- The credible present contribution is an auditable reference platform for
  constrained multimodal routing, residual temporal prediction, evidence
  controls, and reproducible experiment lifecycle management.
- Five registered learned-routing primaries (v2--v5 and EditCLEVR) did **not**
  beat their strongest matched controls.  Those negative results are terminal
  for their frozen protocols and are not tuning targets.
- The v5 residual temporal predictor is supported only on its frozen synthetic
  dynamics task; it is not evidence for the whole six-family system.
- The six flagship v2 mechanisms, parameter growth, and ghost residency have
  bounded CPU engineering evidence only.  They have no efficacy, target-scale,
  CUDA, deployment, or promotion evidence.

### Evidence used to tailor this roadmap

- `RESEARCH_STATE.json` and `research/REALITY_LEDGER.md`
- `OLYMPUS_MODEL_FOUNDRY_LEDGER.md`
- `docs/architecture.md`, `docs/olympus-system.md`, and `docs/flagship-v2.md`
- `research/olympus_consolidated_2026/paper/main.tex` and its evidence map
- Frozen v2--v5, EditCLEVR, Foundry, and flagship engineering records

## Completion rules for every deliverable

- Preserve all predecessor protocols, raw outputs, negative outcomes, and
  source identities.  Make successors rather than altering completed studies.
- Keep development fixtures and protected evaluation physically and logically
  separate.  A test or dashboard screenshot is engineering evidence, never a
  capability result.
- Record the exact source commit, dependency lock, dataset/manifest hashes,
  config, seed, device allocation, wall time, and artifact hashes for any run.
- Fail closed on missing provenance, split overlap, resource-guard failure,
  invalid statistical input, or checkpoint/report identity mismatch.

## A. Core model capabilities

- [ ] **A01 — Typed six-family interchange contract [ENG].** Replace the
  current loosely coupled projected-state handoff with versioned tensor and
  metadata schemas for Hermes tool evidence, Prometheus adapter evidence,
  Perseus expert availability, Atlas patch provenance, Kronos state history,
  and Aion authority/budget state.  **Done when:** a single contract validator
  accepts all six references, rejects every missing/ambiguous field, and emits
  a machine-readable compatibility receipt.

- [ ] **A02 — Family capability manifest [ENG].** Add a versioned manifest that
  states each family’s supported modalities, required inputs, output semantics,
  parameter scale declaration, residency requirements, and known non-claims.
  **Done when:** model construction and CLI validation refuse incompatible
  combinations before tensor allocation, with a rendered manifest table.

- [ ] **A03 — Hermes calibrated abstention head [GATED SCIENCE].** Turn the
  current disagreement diagnostic into a trainable abstention mechanism with a
  fixed coverage/utility objective and explicit denied-tool behavior.  **Done
  when:** a frozen held-out protocol reports calibration, risk--coverage,
  tool-denial false-allow rate, and matched no-abstention control for every
  seed.

- [ ] **A04 — Prometheus causal adapter bank [ENG].** Implement multiple
  revisioned adapter experts with prefix-causal routing, bounded residual
  composition, and route provenance rather than only one adapter path.  **Done
  when:** intervention tests prove future-token invariance, cap enforcement,
  and deterministic route replay from a saved trace.

- [ ] **A05 — Prometheus transfer study harness [GATED SCIENCE].** Build a
  domain-shift experiment that compares causal adapter routing, static adapters,
  and no-adapter control on source-disjoint tasks.  **Done when:** it freezes a
  domain matrix, reports per-domain confidence intervals and regressions, and
  cannot select routes using protected-domain outcomes.

- [ ] **A06 — Perseus sparse expert execution engine [ENG].** Evolve
  resident-set masking into actual sparse batched dispatch with capacity limits,
  overflow policy, load accounting, and absent-expert gradients guaranteed zero.
  **Done when:** profiler traces and adversarial tests show only selected,
  resident experts execute for non-padding tokens.

- [ ] **A07 — Perseus load-balance and specialization objective [GATED
  SCIENCE].** Add declared auxiliary objectives and a fixed suite of
  specialization/collapse metrics.  **Done when:** a preregistered ablation
  compares identical experts with and without the objective, publishing
  utilization, task metric, routing stability, and cost.

- [ ] **A08 — Atlas grounded evidence graph [ENG].** Represent image patches,
  text spans, retrieved chunks, masks, and exclusions as a queryable evidence
  graph rather than an opaque patch tensor.  **Done when:** every output can
  export the exact admitted evidence nodes and excluded-node counterfactual
  trace with content hashes.

- [ ] **A09 — Atlas evidence-faithfulness benchmark [GATED SCIENCE].** Build a
  protocol for patch/span insertion, deletion, and swapping that tests whether
  reported evidence predicts output change.  **Done when:** the study reports
  deletion/insertion curves, leakage checks, random-mask controls, and
  per-modality intervals on untouched examples.

- [ ] **A10 — Kronos recurrent residual world model [ENG].** Generalize the
  current one-step residual predictor to an explicit recurrent latent state,
  horizon-conditioned decoder, and trajectory validity mask.  **Done when:**
  serialized rollouts replay exactly, padded timesteps are inert, and uncertainty
  aggregation remains nondecreasing under the documented semantics.

- [ ] **A11 — Kronos multi-horizon dynamics comparison [GATED SCIENCE].**
  Compare residual, direct, persistence, and uncertainty-aware models at
  predeclared horizons and corruption regimes.  **Done when:** a frozen
  protocol reports horizon-wise MSE, calibration, recovery error, compute, and
  paired seed/trajectory intervals without reusing v5 protected outcomes.

- [ ] **A12 — Aion constrained decision executor [ENG].** Add a deterministic
  executor that consumes Aion’s legal transitions, affordability ledger, STOP
  cause, and host policy without granting tool authority itself.  **Done when:**
  replayed traces prove every chosen action was legal and affordable and that
  STOP is non-bypassable across API, CLI, and web surfaces.

## B. Composed-system mechanisms and architectural extensions

- [ ] **B01 — Explicit cross-family message-passing layer [ENG].** Implement a
  bounded, typed coordination layer with named messages instead of only shared
  projection/gating.  **Done when:** messages, recipients, and causal order are
  logged; a no-message control reproduces the current composition exactly.

- [ ] **B02 — Contribution attribution ledger [ENG].** For each composed
  decision, export family gate weight, leave-one-out shift, risk contribution,
  evidence availability, and Aion decision delta in a single schema.  **Done
  when:** an offline verifier recomputes the ledger from checkpoint, input, and
  trace and detects any changed contribution.
  **2026-10-20 progress:** the new decision-dossier bundle exports gates,
  leave-one-family-out prediction deltas, host/effective risk, and Aion
  transition/STOP state with checkpoint and dataset binding.  It does not yet
  contain the full typed evidence-availability graph.  An independent guarded
  verifier now checks the serialized-file hash and content digest, reloads the
  declared checkpoint and source-bound evaluation split on the declared device,
  and rejects any recomputed attribution difference.  The item remains open
  until A01/A08 supply the typed evidence-availability graph.

- [ ] **B03 — Counterfactual family availability simulator [ENG].** Support
  reproducible removal, degradation, delayed arrival, and stale-output scenarios
  for each family.  **Done when:** an experiment config can enumerate these
  conditions and reject illegal combinations with a condition manifest.

- [ ] **B04 — Six-family composition ablation protocol [GATED SCIENCE].**
  Freeze full, leave-one-out, pairwise-critical, and matched parameter/compute
  controls for a single admitted task.  **Done when:** the runner emits a
  complete ablation matrix with declared primary endpoint and no unplanned
  family-removal decisions.

- [ ] **B05 — Function-preserving growth planner [ENG].** Extend FFN widening
  into a declarative growth plan that records supported block topology, target
  width, parameter delta, optimizer reset policy, and exact parity tolerance.
  **Done when:** plans validate before allocation and parity/input-gradient
  receipts are produced for every supported topology.

- [ ] **B06 — Growth-versus-restart experiment [GATED SCIENCE].** Test whether
  function-preserving growth changes learning relative to a parameter-matched
  fresh model, under identical data/step budgets.  **Done when:** a frozen
  campaign reports learning curves, final held-out metric, stability failures,
  active/backing memory, and matched seed intervals.

- [ ] **B07 — Masked-latent boundary audit [ENG].** Add an instrumented check
  proving which raw input regions can influence JEPA context positions before
  latent prediction.  **Done when:** synthetic leakage cases fail, allowed
  cases pass, and every run records an encoder-boundary diagram and mask hash.

- [ ] **B08 — JEPA objective family switchboard [ENG].** Support documented
  masked-latent, reconstruction, contrastive, and predictive objectives behind
  a common target/stop-gradient contract.  **Done when:** each objective has a
  finite-gradient, target-isolation, checkpoint-resume, and no-op baseline test.

- [ ] **B09 — Objective ablation campaign [GATED SCIENCE].** Create a
  controlled, parameter- and compute-matched comparison of the objective
  family on one frozen task.  **Done when:** a report includes all objectives,
  strong simple baselines, multiple seeds, predeclared effect threshold, and
  null/negative outcomes without suppression.

- [ ] **B10 — Ghost-residency scheduler [ENG].** Build a scheduler that maps
  Perseus demand and model plans to exclusive CPU inference leases, with
  admission estimates for weights, KV cache, activations, read buffers, and
  allocator reserve.  **Done when:** stress tests demonstrate safe eviction,
  exception cleanup, no retained parameters, and a replayable residency trace.

- [ ] **B11 — Residency cost model validation [ENG].** Measure actual CPU and,
  when separately authorized, single-device CUDA memory/latency against the
  static estimate.  **Done when:** the report quantifies estimation error over
  a fixed development fixture matrix and rejects underestimated plans.

- [ ] **B12 — Architecture version migration tool [ENG].** Provide explicit
  state-dict/config migration between `reference_v1` and later revisions, with
  unsupported changes refused rather than silently coerced.  **Done when:**
  migrations create signed before/after manifests, preserve declared tensors,
  and fail on missing semantic mappings.

## C. Training, checkpoints, and evaluation execution

- [ ] **C01 — Unified experiment specification [ENG].** Define one strict
  schema covering task, split identities, model revision, controls, seeds,
  budgets, hardware policy, metrics, and stopping rule.  **Done when:** CLI,
  API, Kaggle builder, and Python runner consume the same validated file and
  render its canonical digest.

- [ ] **C02 — Run state machine with safe recovery [ENG].** Replace
  ad-hoc status handling with durable states for admission, running,
  checkpointing, interrupted, resumed, evaluated, invalidated, and archived.
  **Done when:** forced interruption at each boundary resumes deterministically
  or produces a precise invalidation receipt.

- [ ] **C03 — Full stochastic checkpoint contract [ENG].** Expand checkpoint
  capture to include model, optimizer, scheduler, RNGs, sampler cursor,
  gradient-accumulation state, data-manifest digest, source tree digest, and
  implementation revision.  **Done when:** resume parity is checked at several
  interruption points and mismatch injection is rejected before training.

- [ ] **C04 — Baseline registry and runner [ENG].** Register simple,
  literature-motivated, and privileged-diagnostic baselines with explicit
  deployability labels and parameter/compute accounting.  **Done when:** every
  experiment must name its primary control and the runner refuses an unregistered
  or post-outcome-selected baseline.

- [ ] **C05 — Metric registry with direction and unit semantics [ENG].**
  Centralize metrics such as exact match, selective accuracy, calibration,
  MSE, latency, memory, energy proxy, and abstention cost.  **Done when:**
  invalid aggregation, wrong direction, missing denominator, or mixed unit is
  rejected and metric definitions export with results.

- [ ] **C06 — Statistical analysis module [ENG].** Implement paired bootstrap,
  hierarchical/cluster-aware resampling where applicable, effect-size tables,
  uncertainty labels, and multiplicity policy chosen before evaluation.  **Done
  when:** synthetic known-answer datasets validate the intervals and the module
  refuses invalid pairing or too few independent units for the requested claim.

- [ ] **C07 — Protected-evaluation firewall [ENG].** Enforce one-way access:
  configurations and checkpoints may enter the evaluator, while protected
  outcomes are sealed until the protocol’s declared release point.  **Done
  when:** audit logs show no protected labels reachable by training/development
  code and intentional leakage attempts fail.

- [ ] **C08 — Deterministic data-loader audit [ENG].** Record record order,
  sharding, augmentation seeds, skipped examples, collation, and token/sample
  denominators.  **Done when:** replay reproduces batches byte-for-byte where
  deterministic and explicitly records permitted device nondeterminism.

- [ ] **C09 — Multi-device parity qualification [ENG].** When CUDA access is
  available under the existing one-GPU policy, implement a distinct CPU/GPU
  parity suite rather than treating GPU execution as a fallback.  **Done when:**
  it records device identity, precision, deterministic flags, numerical
  tolerances, and CPU/GPU checkpoint/evaluation divergences.

- [ ] **C10 — Failure-injection test suite [ENG].** Deliberately test disk
  exhaustion, corrupt checkpoint, lost process, unavailable resident shard,
  nonfinite gradient, data mutation, and time-budget expiration.  **Done when:**
  each fault leaves a durable, non-promotable run state and no partially valid
  artifact can be selected as a result.

- [ ] **C11 — Capability-gate evaluator [ENG].** Generalize Foundry’s current
  promotion gates into a configurable evaluator that distinguishes plumbing,
  task, safety, serving, reproducibility, and release evidence.  **Done when:**
  it produces a signed pass/fail matrix and refuses promotion whenever any
  required evidence is absent or bound to another candidate.

- [ ] **C12 — Real-task campaign template [GATED SCIENCE].** Provide a blank,
  protocol-first template for an admitted future task, with hypotheses,
  falsifiers, fixed controls, stopping rule, sample-size rationale, and data
  governance fields.  **Done when:** template validation blocks execution until
  every field is frozen and the project state registers its successor identity.

## D. Data, robustness, and generalization studies

- [ ] **D01 — Data-card and rights-review pipeline [ENG].** Add source-level
  licensing, consent/rights, intended-use, PII, contamination, revision, and
  retention attestations for each dataset component.  **Done when:** missing or
  conflicting attestation prevents materialization and produces a review queue.

- [ ] **D02 — Cross-modal provenance graph [ENG].** Bind text, image, temporal,
  tool-outcome, and risk records to source identities, transformations, and
  split membership.  **Done when:** a query from any training example to its
  sources and descendants is reproducible and detects split contamination.

- [ ] **D03 — Robustness scenario language [ENG].** Define declarative
  corruption, missingness, delay, contradiction, OOD, and authority/budget
  scenarios with known severity and seed.  **Done when:** the same scenario
  renders across synthetic and admitted real data without changing clean data.

- [ ] **D04 — Shift taxonomy and coverage report [ENG].** Classify each
  evaluation condition by covariate, label, modality, temporal, intervention,
  and policy shift.  **Done when:** every result table automatically lists
  tested versus untested shift classes and refuses an unclassified condition.

- [ ] **D05 — Missing-modality stress matrix [GATED SCIENCE].** Evaluate
  complete, missing, corrupted, delayed, and conflicting input conditions
  against simple fusion and best-single controls.  **Done when:** a frozen
  matrix reports per-condition utility, calibration, abstention, and family
  availability, including failures.

- [ ] **D06 — Evidence-exclusion robustness study [GATED SCIENCE].** Test
  Atlas masks and retrieval exclusions under naturally irrelevant, adversarial,
  and semantically essential evidence.  **Done when:** the protocol publishes
  both faithfulness and task-cost results, with random/exclusion-size controls.

- [ ] **D07 — Temporal drift and delayed-observation benchmark [GATED
  SCIENCE].** Evaluate Kronos with irregular cadence, stale observations,
  multi-step rollout, and regime change.  **Done when:** it reports horizon
  degradation, recovery time, uncertainty calibration, and persistence/direct
  baselines over fixed seeds.

- [ ] **D08 — Route stability and sensitivity audit [ENG].** Quantify whether
  small semantically null perturbations, input reorderings, and random seeds
  alter gates or expert choice disproportionately.  **Done when:** route
  stability metrics and thresholded alerts appear in every run report.

- [ ] **D09 — Negative-control generator [ENG].** Build conditions where no
  modality or route should carry predictive information, plus oracle-label
  leakage traps.  **Done when:** learned methods fail to exceed chance under the
  negative controls and the harness detects deliberately injected leakage.

- [ ] **D10 — Distribution-generalization protocol [GATED SCIENCE].** Freeze
  source, target, and withheld-shift domains before model development, with
  no target tuning.  **Done when:** the campaign reports source/target tradeoffs,
  transfer gaps, and confidence intervals against matched simple controls.

- [ ] **D11 — Safety and authority red-team suite [ENG].** Generate illegal
  tool masks, forged cost ledgers, contradictory host rules, empty evidence,
  and malicious manifest/path inputs.  **Done when:** every attack is rejected
  or forced to STOP with a machine-readable reason and no side effects.

- [ ] **D12 — Resource robustness envelope [ENG].** Evaluate behavior near
  memory, storage, time, and resident-expert limits rather than only at nominal
  settings.  **Done when:** charts identify safe admission boundary, graceful
  rejection behavior, and any performance degradation before refusal.

## E. Experiment interface and observability

- [ ] **E01 — Experiment lifecycle CLI [ENG].** Provide `plan`, `validate`,
  `admit`, `run`, `resume`, `evaluate`, `compare`, `archive`, and `verify`
  commands over the unified specification.  **Done when:** `--help` documents
  all state transitions and each command emits a parseable receipt.

- [ ] **E02 — Configuration composer [ENG].** Let researchers assemble model,
  data, corruption, baseline, budget, and evaluation fragments with conflict
  detection.  **Done when:** composition emits one canonical config and shows
  the resolved values/provenance before a run can start.

- [ ] **E03 — Live run dashboard [ENG].** Extend Forge web to show validated
  live state: admission, resource guard, checkpoint, metrics, route usage,
  errors, and STOPs.  **Done when:** it uses only persisted API data, clearly
  labels stale data, and can never fabricate a passed run.

- [ ] **E04 — Run-comparison workspace [ENG].** Provide side-by-side selection
  of exact candidate/control runs with config, data, seed, budget, metric, and
  artifact-digest diffs.  **Done when:** incomparable runs are visibly refused
  or annotated with the exact incompatibility.

- [ ] **E05 — Evidence lineage explorer [ENG].** Create an interactive graph
  from protocol to config, source, data manifest, checkpoint, report, plot,
  and paper table.  **Done when:** any displayed statistic can navigate to its
  raw artifact and an offline verifier confirms all edges.

- [ ] **E06 — Resource and residency monitor [ENG].** Expose estimates and
  observed memory, active/resident parameters, cache, shard transfers,
  latency, and budget consumption.  **Done when:** it records peak values and
  alerts on planner underestimation or unauthorized device use.

## F. Model inspection and paper-ready analysis

- [ ] **F01 — Per-example decision dossier [ENG].** Export a self-contained
  report for one prediction: inputs/provenance, masks, route/expert choices,
  family contributions, uncertainty, authority constraints, output, and
  counterfactuals.  **Done when:** dossier replay regenerates the output or
  identifies an exact nondeterminism reason.
  **2026-10-20 progress:** implemented a hash-bound evaluation-split dossier
  bundle (`olympus.evaluation.decision_dossiers`) and automatic export from the
  composed trainer.  It binds source IDs, checkpoint and dataset identities,
  host authority, STOP, risk, gates, and leave-one-family-out diagnostics while
  excluding raw tensors.  Ruff and strict MyPy pass; numerical pytest is
  deliberately deferred because local free memory is below the 512 MiB guard.

- [ ] **F02 — Routing diagnostic suite [ENG].** Plot gate entropy, effective
  expert count, utilization, availability, collapse, agreement, and route
  sensitivity by regime.  **Done when:** figures derive only from recorded
  traces and include denominators, seed units, and baseline overlays.

- [ ] **F03 — Evidence and patch inspection viewer [ENG].** Render Atlas patch
  masks, text spans, retrieved records, exclusions, and insertion/deletion
  counterfactual outputs.  **Done when:** the viewer loads content-hashed
  artifacts and distinguishes reported evidence from causal validation.

- [ ] **F04 — Temporal rollout inspector [ENG].** Visualize observed state,
  residual/direct/persistence forecasts, uncertainty bands, missingness, and
  error by horizon.  **Done when:** plots are generated from a saved evaluation
  bundle and include all trajectories, not selected examples only.

- [ ] **F05 — Calibration and selective-prediction report [GATED SCIENCE].**
  Produce reliability diagrams, ECE/Brier with uncertainty, risk--coverage,
  abstention utility, and subgroup/regime breakdowns.  **Done when:** a frozen
  evaluation defines bins and coverage choices before protected data are read.

- [ ] **F06 — Automatic paper figure/table compiler [ENG].** Generate every
  manuscript plot, comparison table, appendix cell, and limitations table from
  versioned result bundles.  **Done when:** clean rebuild reproduces a paper
  artifact and fails if a displayed number lacks an evidence-lineage edge.

- [ ] **F07 — Results completeness checker [ENG].** Verify every planned cell,
  seed, baseline, failure, exclusion, and denominator appears in the report or
  is explicitly marked invalid.  **Done when:** a deliberately omitted bad run
  makes the paper build fail.

- [ ] **F08 — Reproducibility and artifact management bundle [ENG].** Implement
  one-command export of source snapshot, lockfiles, environment, config,
  manifests, checkpoints, traces, reports, and hash index; support verified
  import into a clean location.  **Done when:** clean-machine or isolated-root
  verification reproduces the recorded engineering result and identifies any
  unavailable external dependency.

- [ ] **F09 — Public, bounded demonstration [ENG].** Ship an offline demo using
  only a clearly labeled development-scale reference or recorded result bundle:
  it should let users perturb modalities, inspect Aion STOPs, view routes, and
  replay residual trajectories.  **Done when:** the demo exposes no fabricated
  capability claim, ships its source/data provenance, and has an automated
  end-to-end smoke test.

- [ ] **F10 — Independent reproduction kit [GATED SCIENCE].** Package a frozen
  completed study for a second operator: exact archive, environment, task
  instructions, blinded acceptance criteria, and verifier.  **Done when:** an
  independent operator can execute the protocol or report a bounded failure
  without access to the original machine or protected outcomes.

## Recommended execution order

1. Deliver A01--A02, B01--B03, C01--C08, D01--D04, E01--E06, and F01--F08 as
   engineering infrastructure.  They create one auditable experiment surface.
2. Implement isolated mechanisms (A04/A06/A08/A10/A12, B05/B07/B10/B12), each
   with its falsifier and development-only receipt.
3. Freeze one successor protocol per scientific question.  Do not combine the
   routing failures, residual-temporal result, and v2 flagship mechanisms into
   a single post hoc campaign.
4. Run only the corresponding **[GATED SCIENCE]** items after data, budget, and
   protected-evaluation admission.  Promote neither model nor paper claim from
   engineering receipts alone.

## Non-goals preserved by this checklist

- It does not reopen or repair the terminal v2--v5/EditCLEVR routing outcomes.
- It does not claim that the six reference families are trained 12B--8T models,
  nor that lazy residency makes their backing weights available.
- It does not authorize protected evaluation, target-scale training, paid
  compute, external dataset acquisition, or model promotion.
