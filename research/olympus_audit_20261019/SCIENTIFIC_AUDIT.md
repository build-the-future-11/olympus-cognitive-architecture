# Olympus: source-bound scientific audit and next development contract

Current disposition is recorded in the final scientific-status addendum below: Foundry engineering passed; the residency diagnostic was resource-blocked; broader model research is not complete.

Prepared 2026-10-19. **The new diagnostic is frozen but unexecuted at the time of this audit.** All earlier numerical results below are explicitly source-reported; this audit did not retrain models, open protected datasets, replay benchmarks, run a GPU job, or import the subject module. It changed no shared repository source.

## 1. Current question and source identity

The immediate research question worth answering is whether the PR29 lazy loader's bounded shard cache also bounds weights that remain owned by a repeatedly materialized module. This is a necessary systems premise for future memory-saving inference claims. It has a small, falsifiable, deterministic test with an ownership control and does not reuse a consumed model study. It does not establish a new reasoning model or a novel prefetching method.

There are several distinct source states, and joining their claims without joining their evidence would be a mistake:

| State | Exact identity | What it establishes |
|---|---|---|
| Main audit base | `dc46b831e2aa88a5e669e1343463dc74ef1270a4` | Reviewed engineering lineage and the older canonical SFT-continuation question. The core heuristic/toy mechanisms were inspected here. |
| Open PR29 | `12b7c785958b741ec48834a74e81df0c674d327b`, branch `codex/olympus-final-deliverables-20261016` | A separate, substantial unmerged implementation and newer study declarations. At comparison, it was 10 commits ahead, zero behind main, with 225 changed files. |
| PR29 v5 execution source | `2bbf6d1e73d6eb57d9748dc65a89231f184cf4f3` | The source named by the newer v5 per-study completion record. It is not interchangeable with the later PR29 head. |
| PR29 EditCLEVR execution source | `1d2591788cbc7a52766f18c959d4114c16888910` | The source named by the newer external-benchmark completion record. |
| Parent's active engineering work | Separate main-based Foundry admission/checkpoint work; local HEAD observed as `37cb1b678e1666d33208147964fedcdf4a1f5167` during final audit preparation, with further working-tree changes | Its current test receipts and pending edits belong to that engineering revision. They are not evidence that the main base or PR29 scientific studies passed new evaluations. |

The main/PR29 comparison and pinned per-study records supply these identities. The parent working-tree observation is a local read, not an immutable evidence claim for its pending edits. [S1, S6, S7, S9]

At the main base, `RESEARCH_STATE.json` asks whether the actual SFT trainer can continue an unchanged schedule without changing its optimizer trajectory or overwriting previous attempts. Its scope explicitly excludes scientific training, held-out capability reevaluation, architecture changes, and model promotion. The parent has since added a separate Foundry admission study. The historical top-level SFT fields must remain identifiable as historical when an active-study key points elsewhere. [S1, S2]

PR29's newer scientific records also need careful precedence. Its v5 and EditCLEVR per-study states say execution completed and report outcomes; an older reality ledger still describes those studies as not executed. Prefer the newer, specifically identified record when describing its declaration, retain the disagreement, and do not turn that editorial precedence into independent verification. Both newer states explicitly say their verification covers provenance/report arithmetic, not independent retraining; their complete archives remain marked unretrieved. [S4, S7, S9]

The precise broader scientific portfolio question is whether learned interpretation/routing, retrodictive constraints, and temporal representations improve task performance under distribution changes at matched resources. Current evidence does not support that portfolio as a unified successful model. Individual tests have different objectives, fixtures, controls, sources, and dispositions. The one limited positive temporal result cannot transfer to failed routing claims or to the main repository's `TinyJEPA`, which is a different implementation. [S5, S7, S8, S9, S10, S13]

## 2. Strongest available results, including failures

### Engineering and Foundry evidence

The main ledgers establish a real tiny bigram lifecycle on a reviewed 674-byte corpus with SHA256 `6f4823e0503426bda11aa4bffe8357963f3bc97a62ea40c2b69700af166a8eab`. That is a useful executable infrastructure result. It is not a Hermes capability result or evidence of modern language-model quality. Historical deep smoke data contained only 36 records, split 12/12/12; the stated promotion floor was 10,000 reviewed training records and 1,200 held-out records. The earlier smoke model had zero exact task and tool completion, and no Hermes model was promoted. [S3, S4]

The later PR29 ledger describes genuine training and checkpoint continuation, including a 34,528-parameter run and an 884,480-parameter candidate with validation-selected restoration. The latter's reported format compliance reached 83.33%, but exact and tool completion remained zero. Thus a successful training/checkpoint/pipeline smoke check is compatible with no demonstrated target-task capability. Those outcomes must not be overwritten or relabeled because newer admission plumbing passes tests. [S4]

### Routing and temporal evidence

| Study and source | Strongest relevant observation | Scientific disposition and limit |
|---|---|---|
| v3, PR29 ledger | Primary corrupted selective-accuracy delta `-0.0125`, descriptive 95% interval `[-0.0281, 0.0031]`, against a declared `+0.02` effect. | Improvement not supported. A failed criterion cannot be rescued by reporting only legality or host-risk invariants. [S4] |
| v4, `RESULTS.md` | Utility minus semantic routing on corruption `-0.0682`, interval `[-0.0771,-0.0589]`. Semantic minus fixed `+0.1021`, interval `[0.0962,0.1076]`. | Learned utility routing failed its primary contrast. A simpler semantic route helped on this corruption fixture, not uniformly across regimes. [S5] |
| v4, coverage ablation | Mixed rather than clean-only training improved corrupted semantic accuracy by `+0.2742`, interval `[0.2530,0.2928]`; the corresponding counterfactual-penalty delta was `+0.0001`, interval `[-0.0002,0.0004]`. | Training-distribution coverage is a substantial explanation of this fixture's improvement. These numbers do not establish a useful counterfactual penalty or a general architecture advance. [S5] |
| v5, newer completion state | Novel-shift selected routing minus core semantic `-0.00068359375`, interval `[-0.00244140625,0.00107421875]`; original-corruption difference `-0.05869140625`. | Routing primary unsupported. Selection on development data did not produce the declared protected-regime improvement. [S7] |
| v5, newer completion state | Clean residual temporal MSE `0.014565714355558156` versus core-v4 `0.07133172154426574`; residual minus persistence `-0.015398251730948687`, interval `[-0.015830382984131575,-0.014920006040483713]`. | Source-reported temporal primary supported within this protocol. The roughly 79.58% reduction versus core-v4 is not attributable entirely to the residual skip because other architecture/training changes were present. [S7, S8] |
| EditCLEVR, newer completion state | Shifted learned-minus-control mean `-0.000050002336502075194`, interval `[-0.00015000700950622558,0]`: four seed ties, one slightly worse. | The `+2` percentage-point improvement was rejected. This is a bounded negative benchmark result, not evidence that the full Olympus architecture improves vision reasoning. [S9] |

V4's corrupted semantic accuracy was `0.9574`, compared with `0.8554` for fixed routing, but its OOD accuracy was `0.9199`, compared with `0.9864` for fixed routing. The direction changes with regime. Its `oracle_task` route reached `0.9959` corrupted accuracy using privileged task information, so that oracle is a diagnostic ceiling, not a deployable baseline whose performance belongs to the learned route. All intervals in the v4 summary are descriptive across five independently generated datasets, with two training seeds per dataset; ten optimizer runs are not ten independent sampled worlds. [S5]

V5 declares 20 cells, 40 bank fits, and 80 protected evaluations. Its verifier checked 80 prediction hashes, recomputed aggregates and cluster contrasts, and checked source/checkpoint chronology. The declared archive SHA256 is `7dd4b614d2895331e2000a68d2ae988cf8badd54d2776be26857471a5ac391d6`, at `/kaggle/working/olympus-v5-evidence.tar.gz`, with retrieval still false. Its protocol includes a matched direct-versus-residual temporal ablation, but the per-study summary does not give that numerical contrast. The full retained contrast is required before making a specific residual-skip attribution. Rerunning consumed data would not repair this evidence-retrieval gap. [S7, S8]

EditCLEVR declares five training seeds, 15 epochs each, four protected splits, 20 protected reports, and a 741,064-parameter model, using dataset revision `734efa9b0164ba742cd2c39b9c2945c494497c23`. Its archive SHA256 is `a947b3f86afe71ff712e7f203fdfc421cb7c66d3a45e5e61d2d95f6ffeed13c1`, also marked unretrieved. The record says provenance/arithmetic verification, no independent reproduction, and no authorization for promotion or a full-Olympus claim. The negative finding remains useful even though it cannot support the desired positive conclusion. [S9]

### The newer O1 declaration

O1 is a distinct, migrated, unrun proposal about explicit hypothesis → falsification → execution decomposition versus a monolithic agent under the same provider, tasks, context, tools, and budget. Its four arms are monolithic, full decomposition, no-Perseus, and no-evidence. The proposal envisages at least 100 prescored tasks from at least two families and a `+5` percentage-point primary threshold with a bootstrap interval above zero. Its task manifest, provider/runtime, prompts, decoding, and exact execution budgets remain unfrozen. The state explicitly says `UNFROZEN_DO_NOT_RUN`; the prior migration and closed PR ancestry establish source lineage, not an experiment. It must not inherit outcomes from routing, temporal, or EditCLEVR studies. [S10, S11]

## 3. What the main core mechanisms actually do

`olympus/core/prediction.py` defines a small context encoder, target encoder, and predictor. The target is evaluated under `no_grad`, with no EMA or other target-encoder update in this file. Its training fixture is 32 deterministic four-dimensional samples with a fixed additive shift; the repeated full-batch objective predicts a frozen target embedding and includes a standard-deviation regularizer. A training-loss reduction there is not held-out natural-data prediction or evidence of reconstructing large-model weights. The relevant test thresholds check a bounded synthetic learning behavior. [S13, S18]

`olympus/core/retrodiction.py` enumerates caller-supplied candidate generators, calls an invariant and a forward simulator, and ranks candidates with a fixed weighted compatibility formula. This is inspectable constrained candidate search. It is not a learned posterior over histories, and multiple candidates can explain the same observation. A claim of uniquely recovering the actual cause needs a stated identifiable generative process and ambiguity handling. [S14]

`olympus/core/interpretation.py` uses keyword/feature heuristics and a numerical confidence formula. A temporal interpretation enum/scoring branch exists, but the analyzed candidate list omits that temporal branch. The evidence-weight bonus does not by itself establish branch-specific support or statistical calibration. These are concrete limitations of this implementation; they do not establish the failure cause of a different learned router in later research code. [S15]

`olympus/core/representation.py` uses fixed representation scores and simple projections. The spherical path uses L1 normalization, while the low-rank path subtracts a mean. Those operations can be valid transformations, but their names do not prove Euclidean unit-sphere geometry or learned low-rank factorization. The required remedy is an explicit mathematical specification, an implementation that meets that specification, and evidence attached to the actual algorithm. [S16]

`olympus/models/families.py`'s main-base HermesNano selects a heuristic branch and returns its statement. Its prompt memory key uses Python's process-dependent `hash(prompt)`, and the response path does not retrieve a learned or trained model from that memory. This is a façade over prototype mechanisms, not demonstrated trained conversational reasoning. A stable content key would repair durable identity; it would not create learned capability. [S17]

The new PR29 lazy module is more concrete systems work: validated shard metadata, content hashes, deterministic planning, lazy CPU loading, a dictionary cache, and assignment into meta-initialized modules. Nevertheless, deleting a cached tensor dictionary need not release a parameter storage also owned by the receiving module. In source, `_admit` evicts cache entries and `materialize_module` assigns selected tensors with `assign=True`; there is no corresponding removal of previously assigned module weights. This is a **source-based prediction of an ownership issue**, not a reported experimental observation. The proposed fresh diagnostic directly tests that prediction. [S12, S19]

## 4. Flaw, cause, remedy, and required evidence matrix

Severity means impact on a defensible claim or next execution decision. “Known” means directly established by inspected source or declared results. “Unknown cause” is deliberate: an outcome summary cannot identify a unique optimization or architectural cause. The fixes below are recommendations for separately scoped work, not permission to rerun protected studies.

The matrix labels its evidence basis explicitly:

- **D — Direct source observation:** inspected code, schema, source identity, or state fields establish the behavior. This can identify a defect or a specification mismatch; source inspection alone does not create an experimental outcome.
- **H — Historical source-reported evidence:** a pinned result or execution log declares the finding. This audit did not independently rerun it. Arithmetic/provenance verification and independent retraining remain distinct.
- **U — Untested scientific interpretation or limitation:** the stronger claim lacks the required evidence, or its mathematical specification is missing. This is not automatically a demonstrated software defect.
- **R — Fresh artificial regression/diagnostic proof:** none was created by this audit. The root's separately recorded engineering regressions belong to their own source/test receipts. The frozen residency outcome remains pending; root can append it below without changing the protocol.

A row may combine classes, such as a directly observed ownership path (**D**) with an untested whole-module memory interpretation (**U**).

### Source and evidence identity

| Flaw | Cause and confidence | Severity | Concrete remedy | Required proof |
|---|---|---|---|---|
| **D / U** — Main, PR29, and study execution sources can be conflated. | Known: unmerged PR29 contains 225 changes; later per-study execution SHAs differ. [S1, S4, S7, S9] | Critical | Attach repository path, exact source SHA, dataset/checkpoint identity, and study ID to every result. Preserve the unmerged branch. | A source/result map whose artifacts verify against their recorded identities; no numerical claim silently moved to a newer implementation. |
| **D** — Older canonical/ledger text disagrees with newer completion records. | Known editorial chronology conflict, not evidence that either experiment was rerun. [S1, S4, S7, S9] | High | Append a dated reconciliation, keep predecessor entries, and state which per-study record supplies the current declaration. | Reviewable state diff and links to the exact source records; no erased historical failures. |
| **H** — PR29 hosted status and coverage log disagree. | Known: run `39105840776` is green in metadata; Python job output reports 532 passed, two skipped, 84.62% coverage and failure of the required 85% threshold. Cause unresolved. [S20] | High | Inspect the exact job/gate wiring before calling it a clean release gate; retain the failing log. | Exact-source release receipt explaining or resolving the status/exit discrepancy. Passing tests alone do not satisfy the reported coverage requirement. |
| **D / H** — Full scientific evidence is still remote/unretrieved. | Known: v5 and EditCLEVR state `retrieved_to_repository=false`. [S7, S9] | High | Retrieve the already-existing archives through an authorized supported interface and verify their recorded hashes. | Complete immutable predictions, controls, checkpoints, configuration, chronology, and recomputed contrasts; no new protected evaluation. |
| **D / U** — Provenance replay can be mistaken for independent reproduction. | Known: both state files explicitly disclaim independent retraining. [S7, S9] | High | Label arithmetic/provenance verification separately from a new operator/runtime reproduction. | A genuinely separate, predeclared reproduction for that stronger claim; current claims remain source-reported until then. |

### Scientific mechanism and inference

| Flaw | Cause and confidence | Severity | Concrete remedy | Required proof |
|---|---|---|---|---|
| **H** — Utility/learned routing is presented as an improvement despite failed primaries. | Failure is known across v3-v5; the causal reason for the learning/control gap is unknown from summaries. [S4, S5, S7] | Critical | Preserve rejection; localize a new mechanism question on fresh development data before another confirmatory study. | A new frozen contrast isolating a proposed cause, then one protected test if separately authorized. Do not tune on consumed failures. |
| **H / U** — A corruption benefit is generalized to all shifts. | Known regime interaction: v4 semantic beats fixed on corruption but loses on OOD. [S5] | High | State the supported regime and report all predeclared regimes together. | Regime-specific paired effects with common controls, complete uncertainty, and no average masking of harm. |
| **H / U** — Training-coverage gains are attributed to a routing architecture. | Known matched coverage contrast is large; route/counterfactual effects differ. [S5] | High | Keep data coverage, model, training steps, and routing contrasts distinct. | Equal-data/equal-update ablations and explicit factorial contrasts on fresh development material. |
| **H / U** — Counterfactual penalty is asserted beneficial despite near-zero contrasts. | Known reported effect is near zero; absence of any possible effect is not established. [S5] | Medium | Report the observed bound and avoid a positive mechanism claim. | A justified effect size and a new powered/matched contract if revisited; current results remain unchanged. |
| **D / H / U** — The full v5 temporal gain is credited to the residual skip. | Known protocol confound: the core-v4 comparison also changes architecture/training; direct/residual matched control exists. [S7, S8] | High | Retrieve and report the matched direct-minus-residual contrast before assigning the cause. | The frozen paired ablation, its predictions/checkpoints and cluster-level arithmetic; no post hoc rerun. |
| **D / U** — Descriptive intervals are treated as broad population certainty. | Known small number of independently generated datasets; optimizer seeds share a dataset. [S5, S8] | High | Respect the sampling hierarchy and mark intervals descriptive. | Cluster-level data, complete paired contrasts, and an independent dataset design for any broader inference. |
| **H / U** — EditCLEVR's failed improvement is hidden behind non-regression. | Known: four ties/one small loss misses the declared +2-point improvement; ID non-regression is a separate criterion. [S9] | Critical | Record the primary as rejected and retain all five seed results. | Exact declared primary/control/split identities and the full negative report. No positivity-driven repetition. |
| **D / U** — A portfolio-level “full Olympus works” claim exceeds study scope. | Known: routing, temporal, vision, and O1 studies exercise distinct subsets and sources. [S7–S11, S13–S17] | Critical | Define a concrete integrated model/task contract before making a unified claim. | One integrated, reproducible trained/executed model with matched baselines and a separately frozen evaluation. |

### Core implementation and mathematical specification

| Flaw | Cause and confidence | Severity | Concrete remedy | Required proof |
|---|---|---|---|---|
| **D / U** — TinyJEPA training loss is read as learned predictive world modeling. | Known: frozen target encoder, fixed small translation fixture, repeated same-batch objective. [S13, S18] | High | Specify this exact frozen-target toy objective; reserve broader claims for an independently designed model. | Target-update/source audit, held-out latent/state tests with controls, non-collapse evidence and matched capacity/data. |
| **D / U** — Retrodiction scores are read as unique causal recovery or posterior probabilities. | Known: supplied candidates plus deterministic simulator checks and fixed scoring. [S14] | High | Define the candidate class and ambiguity semantics; expose ties and unverifiable histories. | Identifiable/non-identifiable controlled cases and calibrated probabilistic evidence only if a probability model is actually specified. |
| **D** — A temporal category is declared but absent from analyzed candidates. | Known source omission; no new behavior test was executed in this audit. [S15] | Medium | Align the implemented candidate set with the declared interpretation contract, or narrow the contract. | A bounded regression proving the declared temporal path can be selected and remains evidence-supported. |
| **D / U** — Heuristic confidence is interpreted as calibrated likelihood. | Known fixed formula and common evidence bonus; calibration is unmeasured. [S15] | High | Label it a heuristic score; define branch-specific evidence semantics before calibrating. | Reliability/error curves and proper scoring on untouched examples under a separate evaluation contract. |
| **D / U** — Spherical/low-rank labels imply mathematics not implemented. | Known L1 normalization and mean-centering; the intended mathematical spec is not established. [S16] | Medium | State the actual map, norm, dimensions, rank objective and invariants; rename or implement accordingly. | Algebraic checks of the chosen specification and a matched task test for a usefulness claim. |
| **D** — Durable memory identity depends on Python's process hash. | Known source uses `abs(hash(prompt))`; persisted lookup identity can vary by process seed. [S17] | Medium | Use a versioned deterministic content identity, with explicit memory retrieval semantics. | Cross-process identity/retrieval evidence and collision handling; no inference that this alone improves reasoning. |

### Residency and system claims

| Flaw | Cause and confidence | Severity | Concrete remedy | Required proof |
|---|---|---|---|---|
| **D / U; R pending** — A bounded cache is interpreted as bounded whole-module weights. | Known source retains assigned module parameters; actual trace effect remains unmeasured at freeze. [S12, S19] | High | Execute the one frozen ownership diagnostic. If falsified, define module/cache ownership leases in a later engineering contract. | Live unique-storage measurements and exact output checks for persistent and recreated-module conditions; this is the selected fresh experiment. |
| **D / U** — Serialized shard size is treated as actual memory use. | Known cache counter counts file bytes; tensors, serialized payload and deserialization workspace are distinct allocations. [S12] | High | Separate file-I/O, tensor-storage, staging, activation, and process-memory budgets. | Storage accounting plus transient peak/staging tests before any whole-process RAM guarantee. The current diagnostic only supplies a lower bound. |
| **D / U** — A cache eviction is assumed to precede the next allocation. | Known `_load(record)` evaluates before `_admit`; temporary over-allocation is possible. [S12] | High | Reserve staging memory or define an admission protocol that accounts for old and incoming ownership. | Allocation-lifetime measurements under a separate fixed peak-memory contract; RSS alone is insufficient. |
| **D / U** — One selected layer's output is called full-model inference. | Known partial meta-materialization test executes only the loaded layer, leaving other meta tensors. [S19] | High | Name the tested path accurately; separately specify complete-network execution and missing-tensor behavior. | End-to-end complete forward/generation with exact checkpoint and output identity, without silently filling untested layers. |
| **U; primary prior art** — Generic predictive or spatiotemporal prefetching is claimed novel. | Known primary literature already covers predictive expert loading, native-route-preserving prefetch, and spatiotemporal correlations. [P1–P7] | High | Define any proposed difference as an exact algorithm and compare with the closest work. | A mechanism-level novelty map plus equal-memory/equal-I/O baselines including prediction and wrong-prefetch costs. No novelty is established now. |
| **H / U** — Historical provider loading is treated as current Mac/16-GB viability. | Known historical load/swap records concern specific providers/runtimes, not this new CPU fixture. [S3, S4] | High | Keep the device and model identity explicit; do not extrapolate from tiny CPU work. | Measurements on the target machine/model, including peak resident memory, swap, latency and full execution. |

### Execution discipline and stopping

| Flaw | Cause and confidence | Severity | Concrete remedy | Required proof |
|---|---|---|---|---|
| **D / U** — O1 is run while critical task/provider/budget fields are unset. | Known unfrozen proposal and missing execution specification. [S10, S11] | Critical | Complete a fresh immutable specification only when its actual provider/tasks/budget are available. | A source/task/prompt/decoding/control freeze before outputs; O1 remains unrun now. |
| **H / U** — CI or smoke success is promoted into model efficacy. | Known mismatch between engineering assertions and scientific endpoints, including zero task success in training smoke. [S1, S3, S4, S18] | Critical | Keep software validation, model admission, and scientific efficacy as separate typed outcomes. | Exact software receipts plus independent declared capability evidence for any promotion. |
| **U; methodological contract** — Repeating until positive changes the experiment after outcomes. | Methodological cause is certain: outcome-dependent continuation selects results and defeats a fixed stopping rule. The detailed user contract explicitly requires fixed budgets and preserved negatives. [U1] | Critical | Freeze the one permitted new matrix, stop after it, and preserve negative/blocked/invalid results. | Protocol and executable hashes predating outputs, all attempt receipts, and no silent threshold or seed changes. |

## 5. Prior art and the actual contribution boundary

This is a targeted primary-source review, not a claim that all relevant literature has been exhausted. The following work is enough to reject novelty based solely on the concepts “predict the next expert,” “keep exact native routing,” or “use spatial/temporal correlations.” No reported performance number from these papers is transferred to Olympus.

| Primary source | Relevant mechanism | Implication for Olympus |
|---|---|---|
| [P1: LLM in a Flash](https://machinelearning.apple.com/research/efficient-large-language) | On-demand flash-to-memory loading, recent-activation reuse, and storage/layout choices informed by hardware costs. | Sharding and loading under limited memory have established prior art; the full ownership and I/O path must be measured. |
| [P2: ProMoE](https://arxiv.org/abs/2410.22134) | Predicts future experts from intermediate model features and proactively prepares/caches them. | Predictive expert caching alone is not an original contribution. |
| [P3: Pre-gated MoE](https://arxiv.org/abs/2308.12066) | Forecasts sparse expert use to prepare future expert execution/loading. | A lookahead mechanism must be compared to existing pre-gating approaches. |
| [P4: FlexGen](https://proceedings.mlr.press/v202/sheng23a.html) | CPU/GPU/disk offloading and resource allocation for throughput-oriented language-model inference. | Compare the actual workload and objective; batched offload throughput does not establish interactive local CPU performance. |
| [P5: SpecPrefetch](https://arxiv.org/abs/2607.24787) | A small predictor guides expert transfer while the frozen native router determines the executed experts. | Separating prediction for loading from authoritative routing is already described. |
| [P6: APEX](https://arxiv.org/abs/2608.11688v1) | Adaptive prefetch with an exact-routing mode and an approximate mode designed to avoid stalls. | Exactness versus speculation is an established systems tradeoff, not a novelty claim by itself. |
| [P7: ST-MoE](https://arxiv.org/abs/2606.15453v1) | Spatiotemporal expert-use correlations guide prefetch while retaining native routing. | The name “spatiotemporal” is insufficient evidence of a new mechanism or new result. |

The defensible immediate contribution is smaller and concrete: determine whether an existing, exactly identified implementation has the parameter-ownership behavior required for a proposed memory interpretation, using an independent observable rather than its own cache counter. A result could be an informative falsification. A future originality claim would need a specified predictor, target, admission/eviction ownership model, correctness contract, accounting of its cost, and a demonstration of a difference from the closest papers.

## 6. Selected fresh development contract

The complete frozen contract is in `PROTOCOL.md` and `protocol.json`; `FREEZE.json` binds the exact diagnostic and source bytes. Study ID is `olympus-residency-ownership-dev-v1-20261019`, protocol version 2. This protocol is a new development study, not a retry of v2-v5, EditCLEVR, O1, or a failed scientific run.

The diagnostic uses three deterministic 128-by-128 diagonal float32 shards, seed `2026101901`, and trace `[0,1,2,0,1,2]`. Each tensor has 65,536 mathematical weight bytes. The cache budget is fixed by the maximum actual serialized shard size, with a precondition that exactly one shard fits. It runs the exact PR29 module in two fresh sequential processes: one retains the receiving module; the other releases and recreates an identical meta module between requests. Both use the same files, plans, six selected-layer forwards, and six disk loads. No training or data fitting occurs.

The primary is the sum of unique live CPU storage reachable from the current module and real cache after request temporaries are deleted. It deduplicates shared storage and is distinct from the serialized cache counter and child RSS. Weak references to actual previous tensor/parameter owners supply supporting lifetime observations without retaining them. Exact selected-layer outputs are checked against an analytical reference and preserved.

If the persistent arm exceeds the budget while the recreated-module control stays within it and all validity checks pass, the result is `RESIDENCY_INTERPRETATION_FALSIFIED`. This rejects the stronger whole-module interpretation for this source/trace; it does not accuse the class of violating an explicit dictionary-cache guarantee. If both remain within the budget, the finite test did not falsify the interpretation. A bad control is inconclusive; a bad source/runtime/fixture/output is invalid; failed admission is blocked. None requires a positive answer.

The runner freezes Linux Python 3.14.8, Torch 2.14.1+cpu, Pydantic 2.14.1, CPU-only tensors, and one Torch thread. It acquires the existing ResourceGovernor, requires 2 GiB effective headroom, and limits the full matrix to 90 seconds with 30 seconds per child. The parent's pre-execution resource-policy amendment credits only conservatively reclaimable inactive file cache, records raw headroom separately, and leaves the 2 GiB floor unchanged. It retains a blocked attempt without retries. No GPU, remote/paid job, external dataset, protected evaluation, model promotion, repository mutation, or network request is part of this diagnostic.

The decision after this fixed matrix is determined by the outcome. A falsification motivates a separately scoped ownership/eviction/staging repair before any inference-performance study. A non-falsification leaves the broader claim unproved. An invalid or blocked attempt is retained with its cause and does not silently become a new seed or altered trace. The complete system cannot honestly be declared perfect or scientifically successful by repairing every visible software flaw; successful execution and a failed scientific hypothesis can coexist.

## Source index

All GitHub paths below are pinned. A source-reported result means the cited file declares it; this audit did not independently reproduce the run.

- **S1:** [Main `RESEARCH_STATE.json`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/dc46b831e2aa88a5e669e1343463dc74ef1270a4/RESEARCH_STATE.json).
- **S2:** [Main `AGENTS.md`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/dc46b831e2aa88a5e669e1343463dc74ef1270a4/AGENTS.md).
- **S3:** [Main `OLYMPUS_MODEL_FOUNDRY_LEDGER.md`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/dc46b831e2aa88a5e669e1343463dc74ef1270a4/OLYMPUS_MODEL_FOUNDRY_LEDGER.md) and [main `research/REALITY_LEDGER.md`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/dc46b831e2aa88a5e669e1343463dc74ef1270a4/research/REALITY_LEDGER.md).
- **S4:** [PR29 `OLYMPUS_MODEL_FOUNDRY_LEDGER.md`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/12b7c785958b741ec48834a74e81df0c674d327b/OLYMPUS_MODEL_FOUNDRY_LEDGER.md), [PR29 reality ledger](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/12b7c785958b741ec48834a74e81df0c674d327b/research/REALITY_LEDGER.md), and [PR29 canonical state](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/12b7c785958b741ec48834a74e81df0c674d327b/RESEARCH_STATE.json).
- **S5:** [PR29 v4 `RESULTS.md`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/12b7c785958b741ec48834a74e81df0c674d327b/research/olympus_irp_successor_v4/RESULTS.md), Git blob `039d0961e121a0de66443ea1450170106305f5ac`.
- **S6:** [PR29](https://github.com/build-the-future-11/olympus-cognitive-architecture/pull/29) and [pinned main-to-PR29 comparison](https://github.com/build-the-future-11/olympus-cognitive-architecture/compare/dc46b831e2aa88a5e669e1343463dc74ef1270a4...12b7c785958b741ec48834a74e81df0c674d327b), inspected through GitHub's repository/PR comparison connectors.
- **S7:** [PR29 v5 `RESEARCH_STATE.json`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/12b7c785958b741ec48834a74e81df0c674d327b/research/olympus_irp_successor_v5/RESEARCH_STATE.json), Git blob `43c9e468d7708487aad3c9c11e86eed590bd455e`.
- **S8:** [PR29 v5 `PROTOCOL.md`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/12b7c785958b741ec48834a74e81df0c674d327b/research/olympus_irp_successor_v5/PROTOCOL.md), Git blob `be010e9dc5f79d12ce8dcb5f899c6734489a01d8`.
- **S9:** [PR29 EditCLEVR `RESEARCH_STATE.json`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/12b7c785958b741ec48834a74e81df0c674d327b/research/olympus_editclevr_v1/RESEARCH_STATE.json), Git blob `cbacf95c74cc9015e4dda0bce7be5b525a617547`.
- **S10:** [PR29 O1 `README.md`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/12b7c785958b741ec48834a74e81df0c674d327b/research/olympus_o1/README.md), Git blob `c8d5d6c35506304f0ffa90bd11bfa7084529755e`.
- **S11:** [PR29 O1 `RESEARCH_STATE.json`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/12b7c785958b741ec48834a74e81df0c674d327b/research/olympus_o1/RESEARCH_STATE.json).
- **S12:** [PR29 `olympus/models/spatiotemporal_lazy.py`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/12b7c785958b741ec48834a74e81df0c674d327b/olympus/models/spatiotemporal_lazy.py), SHA256 `09c7a7aa4a1edcf19f3e4c4084e44d627dc606a43172e8853c02d3fa59fff939`.
- **S13:** [Main `olympus/core/prediction.py`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/dc46b831e2aa88a5e669e1343463dc74ef1270a4/olympus/core/prediction.py), SHA256 `beef7ad50e25a54006abc3d95d46da259bf30673c956d2b0b50ed0999f611b16`.
- **S14:** [Main `olympus/core/retrodiction.py`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/dc46b831e2aa88a5e669e1343463dc74ef1270a4/olympus/core/retrodiction.py), SHA256 `81f4baf7dda3a5e89bcefb2df665019b4b01646d3ea0848e2df4f12d94b57c2c`.
- **S15:** [Main `olympus/core/interpretation.py`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/dc46b831e2aa88a5e669e1343463dc74ef1270a4/olympus/core/interpretation.py), SHA256 `6110433e0965c62c33514f6977e8fcb3c3d6ade86ac943264b68b63751ff2ebe`.
- **S16:** [Main `olympus/core/representation.py`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/dc46b831e2aa88a5e669e1343463dc74ef1270a4/olympus/core/representation.py), SHA256 `e369de60759ddfbfb2aa58624621e88a404c5db4c53b2b2127b6fa319ff4e465`.
- **S17:** [Main `olympus/models/families.py`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/dc46b831e2aa88a5e669e1343463dc74ef1270a4/olympus/models/families.py), SHA256 `1144b12e02ba5840157c0546bd541db4147edfa30da4f41a71f553a41a5616a2`.
- **S18:** [Main `tests/test_learning_and_retrodiction.py`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/dc46b831e2aa88a5e669e1343463dc74ef1270a4/tests/test_learning_and_retrodiction.py).
- **S19:** [PR29 `tests/test_bu1ld_model_pathways.py`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/12b7c785958b741ec48834a74e81df0c674d327b/tests/test_bu1ld_model_pathways.py), Git blob `09bee95ab81cd3580a571b264e50781db27e7f70`.
- **S20:** [PR29 workflow run 39105840776](https://github.com/build-the-future-11/olympus-cognitive-architecture/actions/runs/39105840776) and [Python job 117611371531](https://github.com/build-the-future-11/olympus-cognitive-architecture/actions/runs/39105840776/job/117611371531); both metadata and detailed logs were inspected through the GitHub connector.
- **U1:** User steering attachment `/workspace/scratch/5d7183878ded/upload/Pasted text(2).txt`, read in full during this audit. Its detailed fixed-budget, no-positive-selection requirements govern the development contract.

## Prepared artifacts and limits of this audit

`diagnose_residency.py`, `protocol.json`, `PROTOCOL.md`, `SOURCE_MANIFEST.json`, `RESOURCE_POLICY_ORIGIN.json`, `FREEZE.json`, and the three copied source modules are under `/tmp/olympus-science-audit/`. Only stdlib metadata/hashing/AST operations and static Ruff lint were used during preparation. The diagnostic's exact source SHA256 is `fd63cd8aacab03b1e80733e3d9b8546400307784dc5be7ae70d6c47eae6b9fee`; the final protocol SHA256 is `d937e38dc7b3d3c32f0fcafecd2f65a98ed532b7e368ee8ff17cd495c2af7a25`. Root owns execution and any repository integration.

This is a source-bound audit of the inspected mechanisms and declared studies, not proof that every possible defect has been found. Unsupported claims are limited or rejected here rather than repaired by invented evidence. Callable GitHub tools exposed no billing/usage endpoint, so no billing figure was obtained; unsupported endpoints and credentials were not probed.


## Reserved outcome addendum — pending root execution

No residency result is entered here at audit freeze. Root may append the fresh output directory, receipt SHA256, source/protocol identities, execution disposition, both complete storage traces, analytical output checks, resource observations, and the predeclared outcome after the one admitted matrix. A blocked or invalid attempt must remain visible. This section must not be used to rewrite historical study outcomes or change the frozen protocol.


## Final scientific status — 2026-10-19

**The broader new scientific model has not reached `IDEA_READY` or `RESEARCH_COMPLETE`.** This revision supplies a source-bound flaw/evidence matrix, a minimum candidate model contract, and an audit of 20 primary sources. It also preserves a fully specified, resource-blocked development diagnostic. These are concrete research preparation and evidence-preservation deliverables; they do not establish a novel learned mechanism or a completed model paper. The [related-work and minimum-contract document](RELATED_WORK_AND_MODEL_CONTRACT.md) and [source audit records](sources.json) retain the existing paper links, source locations, and access limitations.

Attachments `Pasted text(2).txt`, `(5).txt`, `(6).txt`, and `(8).txt` are byte-identical: SHA256 `afd9d4f11cc233fae8811f742017ddfe39d9fc8f392a1044256d39d2e57a7fa1`. They are recorded as one continuing research specification; prior work and frozen outcomes were not reset.

### The ten checkpoints apply to the broader proposed model

| Checkpoint | Current disposition | Evidence and missing requirement |
|---|---|---|
| 1. `IDEA_READY` | **Not met** | Prior-art boundaries and candidate falsifiers are documented. A differentiated contribution and complete, explicit learned mechanism remain unresolved. |
| 2. `MODEL_SPECIFICATION_READY` | **Not met; minimum contract available** | The contract identifies reconstruction, identity, resource, and selector requirements. It does not freeze every scientifically consequential architectural and training choice. |
| 3. `METHODOLOGY_READY` | **Not met for the broader model** | The narrow ownership diagnostic is fully specified. A complete broader-model methodology with eligible data, controls, statistical units, and fixed budgets is absent. |
| 4. `MODEL_IMPLEMENTED` | **Not met** | Existing Foundry and PR29 modules are credited under their own sources. They do not implement the complete proposed novel model. |
| 5. `EXPERIMENT_SYSTEM_IMPLEMENTED` | **Not met for the broader model** | Foundry has an executable engineering lifecycle, and the diagnostic has a frozen runner. Neither constitutes the complete new scientific methodology. |
| 6. `RUN_READY` | **Not met** | The broader pipeline is incomplete. The narrow diagnostic's fixture worker was stopped by resource admission before producing a fixture or arm result. |
| 7. `CONFIRMATORY_PROTOCOL_FROZEN` | **Not met** | The frozen protocol is a bounded development diagnostic. No final confirmatory protocol for the broader new model has been frozen. |
| 8. `EVIDENCE_LOCKED` | **Not met for a new-model claim** | Engineering and blocked-attempt artifacts have preserved identities. No new-model confirmatory evidence exists to lock. |
| 9. `VERIFIED` | **Not met for scientific completion** | Saved-artifact hashes and recorded resource arithmetic were checked. This does not verify a new model's efficacy or independently reproduce historical studies. |
| 10. `RESEARCH_COMPLETE` | **Not met** | There is no complete novel-model paper, confirmatory run, or complete reproducibility package supporting a terminal new-model conclusion. |

### Actual residency outcome and evidence limits

The single protocol-v2 attempt began at **13:56:27 UTC** and ended **`STOPPED` / `BLOCKED`**. Only the fixture worker launched. It ran for approximately 1.014 seconds and exited `-9` after the guard recorded effective available memory of **2,123,210,752 bytes**, below the unchanged **2,147,483,648-byte** floor. Recorded swap utilization was zero. Child maximum RSS was **177,549,312 bytes**, with 115 major page faults. These are worker-startup observations, not live-weight residency measurements or proof of a kernel OOM. [Recorded outcome](RESIDENCY_OUTCOME.json)

Both stdout and stderr were empty; no fixture directory, fixture metadata, or comparison-arm result was produced. Consequently, there is **no primary storage metric, analytical output check, or residency-hypothesis result**. The source-based ownership concern remains untested. No retry was performed, and the frozen source/protocol bytes and failed receipt remain preserved.

All 22 recorded artifact/freeze/identity hash checks agreed; the three copied source files matched their SHA256, size, and Git blob identities. Recorded headroom arithmetic also agreed. Individual historical `memory.stat` counters were not retained, so the underlying cache-credit derivation cannot be independently reconstructed from raw counters. Receipt SHA256 is `c11c69be67e64aa4daa40e8dd79e991336072a4445102c5681e70b8da7dcce4a`.

A future harness limitation is also retained: the runner lacks a final post-child-exit deadline check, which could matter if the parent is descheduled near the time boundary. That review note arrived after this early memory stop and does not change its disposition. The executed bytes were not revised afterward. Ownership execution remains deferred to a separately admitted host and reviewed harness.

### Engineering completion and historical findings remain distinct

The **Foundry v4 engineering lifecycle passed** on source `92cc121e6f1074de72eaee5a8cb670170107bbc7`, including the recorded train/reload/export/generation/API identity path. Its earlier stopped attempts remain retained. That pass demonstrates the declared engineering lifecycle; it supplies no novel-model capability result. [Workflow receipt](../verification/foundry_admission_20261019/workflow-receipt.json)

Historical source-reported routing negatives across v3–v5 and the rejected EditCLEVR improvement remain unchanged. V5's limited temporal positive also remains: its comparison against core-v4 changes architecture/training as well as residual prediction, so the full gain cannot be attributed to the residual skip. The historical provenance/arithmetic verifications are not independent retraining, and the retained archives still require retrieval and inspection before stronger attribution.

### One recommended next research action

**Retrieve the retained v5 and EditCLEVR evidence archives through the supported authenticated notebook-output capability, then verify their declared hashes and reconcile their full evidence records.** This means retrieving existing outputs, including the frozen matched temporal contrast, predictions, controls, and checkpoint/configuration identities. Complete that evidence audit before designing another protected study. It can resolve current provenance and attribution gaps while preserving all prior outcomes; another protected run cannot substitute for the missing retained evidence.
