# Olympus: related work and minimum scientific model contract

Read-only research audit, 2026-10-19. The final engineering source reviewed for handoff is `92cc121e6f1074de72eaee5a8cb670170107bbc7`, on `fix/olympus-foundry-admission-20261019`. It is distinct from the observed main base and from PR #29, identified below. This is a proposed research contract, not an implemented new method, experiment result, novelty certification, or approval to use protected data. The source manifest retains the earlier inspection history and adds this final review. Every W/R identifier has a linked primary bibliography entry in section 6; `sources.json` supplies supporting locations, paraphrases, retrieval references, and access limitations.

## Decision

The broad ideas of generating weights, storing weights in learned latent codes, loading selected weights on demand, choosing latent geometries, and allocating computation across reasoning branches already have close precedents. A bounded October 2026 search found particularly direct recent work: Neural Weight Compression, PocketLLM, Motion-Compensated Weight Compression, DFloat11, and EntroPack. A proposal that only combines these labels has not established a new scientific contribution. [W1–W8]

The main-derived engineering source supplies useful foundations, but its inspected checkpoint paths do not instantiate the proposed learned weight reconstructor and lazy tile executor. Its core representation and branch modules implement deterministic heuristics. Separately, PR #29 contains a real lazy parameter-shard store; it must be credited and assessed under its own source contract. Neither a blanket claim that Olympus has no lazy implementation nor a transfer of PR #29 research results onto main would be accurate. A new codec or selector hypothesis still needs an executable mechanism, comparator, resource envelope, and failure criterion before confirmation. The attached research pipeline permits honest negative, mixed, inconclusive, and invalid terminal states; it should not be interpreted as an instruction to keep changing the method until results become positive.

## 1. Closest mechanisms and the remaining claim boundary

| Mechanism | Closest primary work | What an Olympus claim must add and demonstrate |
|---|---|---|
| A network produces another network's weights | HyperNetworks (2016); NeRN (ICLR 2023) [W1, W2] | A specified predictor, target weight distribution, training objective, and measured benefit. Weight generation itself is established. NeRN's CNN experiments do not validate a local language model. |
| Compressed latent codes decode into LLM weights | Neural Weight Compression (2025, revised 2026); PocketLLM (AAAI 2026) [W3, W4] | A meaningful algorithmic or system difference from learned weight codecs; matched complete storage, quality, and inference cost. NWC's reported tensor decoding timing is not a whole-model resident-memory guarantee. |
| Predict later weights from prior weights and store residuals | Motion-Compensated Weight Compression (2026 preprint) [W5] | Explicit alignment, dependency depth, random-access/keyframe policy, and better measured tradeoffs. Layer-sequential prediction plus residuals is already directly proposed. |
| Exact compressed weights decoded during execution | DFloat11 (NeurIPS 2025) [W6] | Comparison to an eligible exact entropy codec and an honest account of decode kernels, scratch memory, and offload baselines. Exact compression and online decompression are established. |
| Independently decodable tiles and fused weight reconstruction | EntroPack (September 2026 preprint) [W7] | An explicit distinction and end-to-end measurements. Its language-model discussion directly separates prefill from decode and reports reconstruction overhead; prefill observations cannot establish interactive decoding speed. |
| Weight selection, flash loading, cache reuse | LLM in a flash (ACL 2024); PowerInfer and PowerInfer-2 [W8–W10] | An actual supported backend and measured sparsity/cache behavior. Predicting which neurons activate differs from predicting numerical weights. These systems depend on activation patterns, memory tiers, and hardware-specific operations. |
| Out-of-core throughput under limited accelerator memory | FlexGen (2023) [W11] | Interactive latency comparisons must use the corresponding workload. A throughput result for large, latency-insensitive batches does not answer a single-user response-time question. |
| Select or learn representation geometry | Mixed-curvature product spaces (ICLR 2019); Neural Latent Geometry Search (NeurIPS 2023) [R1, R2] | Executable geometric mappings, explicit information and dimension budgets, a search objective, and improvements beyond a fixed representation or simple metadata policy. |
| Adaptive computation and branch selection | Adaptive Computation Time; Mixture-of-Depths; Tree of Thoughts; Selecting Computations [R3–R6] | A specific action policy with all generation, evaluation, and routing costs charged. A priority formula is not automatically an estimate of value of information. |

These are mechanism-level precedents, not claims that every cited implementation is immediately compatible with Olympus. Recent preprints are identified as such; their author-reported results were not independently reproduced. This search is sufficient to reject broad novelty assertions, but cannot prove exhaustive priority coverage.

## 2. Source-level contract actually present

### Distinct source states

| Scope | Exact source | Interpretation |
|---|---|---|
| Main audit base | [`dc46b831e2aa88a5e669e1343463dc74ef1270a4`](https://github.com/build-the-future-11/olympus-cognitive-architecture/tree/dc46b831e2aa88a5e669e1343463dc74ef1270a4) | The observed main base. Main is not assumed to contain the later repair or PR #29. |
| Engineering handoff | [`92cc121e6f1074de72eaee5a8cb670170107bbc7`](https://github.com/build-the-future-11/olympus-cognitive-architecture/tree/92cc121e6f1074de72eaee5a8cb670170107bbc7) | The main-based Foundry admission/checkpoint repair branch. The table below describes inspected files in this revision. |
| Separate PR #29 subject | [`12b7c785958b741ec48834a74e81df0c674d327b`](https://github.com/build-the-future-11/olympus-cognitive-architecture/blob/12b7c785958b741ec48834a74e81df0c674d327b/olympus/models/spatiotemporal_lazy.py) | A separate PR source, not merged into the observed main base or repair tree. The lazy module was inspected from the hash-bound source copy identified in `source-contract.json`. |

This distinction is about exact trees, not a claim about future branch or PR status. The audit did not refresh remote references or rerun any study. PR #29 also contains separate learned research implementations and study declarations; this main-core inspection does not identify their results or explain their failures. Those need their own execution-source and artifact records.

### Main-derived engineering implementation

Paths below are relative to the repository at the engineering handoff revision. They describe code, not measured model capability.

| Source | Observed behavior | Research implication |
|---|---|---|
| `olympus/core/transform.py`, `LearnedTransformNetwork` | A linear encoder, tanh, and decoder reconstruct input signals. | This is an input autoencoder, not a coordinate-to-model-weight reconstructor. |
| `olympus/core/representation.py`, `select` | Metadata rules return tree, graph, grid, sequence, or Euclidean labels with constants 0.89, 0.84, 0.82, 0.80, or 0.72. | These numbers are heuristic scores, not empirically estimated quality or calibrated probabilities. |
| Same file, `project` | The spherical branch uses L1 normalization; low-rank subtracts the mean of a vector; other unsupported families fall through to rounding. | This does not implement conventional unit-L2 sphere projection, matrix rank reduction, or the named hyperbolic/product geometries. Defining an enum does not supply that geometry's operations or training objective. |
| `olympus/core/interpretation.py` | Eight template branches are formed; scores derive from keywords, evidence weights, and category constants. | This does not demonstrate learned alternate reasoning trajectories or empirical uncertainty estimation. |
| `olympus/core/schemas.py`, `BranchScore.priority`; `olympus/core/budget.py`, `schedule` | A product of heuristic factors divided by assigned cost ranks branches; scheduling enforces a branch-count rule. | This alone does not enforce the declared latency, RAM, token, or model-call budgets, nor estimate marginal decision improvement. |
| `olympus/core/verification.py`, `CalibrationVerifier` | A single outcome and two confidence thresholds determine its result. | Calibration concerns a population of predictions and outcomes. This rule cannot establish it. [R7] |
| `olympus/foundry/quantization.py`, `load_quantized_model` | All archived tensors are dequantized eagerly and loaded into a normal `TinyCausalLM`. | Archive size reduction does not establish low-bit inference or reduced runtime residency. This observation is scoped to the archive loader, not the separately implemented packed QLoRA linear layers. |
| `olympus/foundry/sft.py` | A real dense GELU tiny transformer; checkpoint loading materializes full state; text generation recomputes a bounded prefix. | The inspected path has neither a learned weight codec nor a lazy tile executor. Sparse ReLU execution results from other models cannot simply be transferred. Its current generator is not a KV-cache backend. |

At the engineering handoff revision, the consumed-checkpoint SHA repair binds the candidate, explicit baseline, and quantization source to the bytes actually deserialized. That is evidence identity infrastructure. It does not change the scientific meaning of these components or validate their quality. It is not attributed to the earlier main base.

### PR #29 lazy-shard implementation

The separately inspected `olympus/models/spatiotemporal_lazy.py` has SHA-256 `09c7a7aa4a1edcf19f3e4c4084e44d627dc606a43172e8853c02d3fa59fff939`. It defines spatial/temporal bounds, shard manifests, queries with velocity/lookahead, and deterministic shard scores. `SpatiotemporalParameterStore.plan` chooses shards by mandatory flags, scores, and serialized byte counts. `_load` verifies actual shard bytes before CPU deserialization. `_admit` maintains an LRU dictionary cache, and `materialize_module` assigns selected tensors into a supplied module, including a meta-initialized module.

That is implemented lazy shard loading and module materialization. Its predictor is an explicit position extrapolation plus metadata scoring; the inspected module does not learn a predictor of numerical weights. The stated cache budget counts serialized shard bytes, so it is not automatically an upper bound on live tensor storage, staging allocations, or process RAM. Eviction removes a dictionary entry, while materialization uses `assign=True` and contains no explicit unloading of older module parameters. This gives a source-based reason to test persistent module ownership separately from cache accounting; this review supplies no executed ownership result. A source-bound diagnostic can falsify a stronger whole-module residency interpretation without claiming that the narrower dictionary-cache contract failed.

The future learned-codec contract in section 3 is therefore a proposed successor mechanism. It should not be presented as already implemented by this shard store, nor as the first lazy-loading idea.

## 3. Candidate A: learned weight codec plus bounded execution

### Representation and identity

Freeze one source checkpoint with named tensors, exact shapes, dtype, byte order, and deterministic tile partition. For each tile \(j\), define a code \(z_j\), predictor \(h_\phi\), context \(c_j\), and optional residual \(r_j\). The decoder computes

\[
\widehat W_j = D(h_\phi(c_j,z_j),r_j).
\]

The context must specify whether it uses coordinates, layer identity, previous reconstructed tiles, or activations. Dependencies cannot be hidden: a layer-sequential predictor may require earlier keyframes and intermediate tiles before a requested tile is available. Account for those reads, computations, and buffers. [W2–W5]

Choose one of two distinct contracts before experimentation:

1. **Exact codec:** decoded canonical tensor bytes equal the source tensor bytes. A floating-point subtraction/addition residual is not inherently byte-exact. Define a reversible residual over a canonical representation and verify equality. Equal weights alone do not guarantee bit-identical logits if tiling changes floating-point accumulation order or kernels.
2. **Approximate codec:** reconstruction changes weights. Assign a new artifact identity; specify quality non-inferiority margins and exact execution semantics. Weight error is diagnostic, not a substitute for task performance.

Bind source SHA, codec parameters, predictor weights, code/residual payloads, normalization metadata, dependency index, runtime version, arithmetic precision, cache policy, and output artifact SHA. A hash identifies bytes; it does not show those bytes implement a valid reconstruction.

### Objective and complete size

A candidate rate–distortion objective is

\[
\mathcal L = R_{\rm total} + \lambda_w D_w(W,\widehat W)
 + \lambda_a D_a(f_W(x),f_{\widehat W}(x)),
\]

with all coefficients frozen after development. \(R_{\rm total}\) includes predictor, codes, residuals, indices, entropy tables, scales, alignment information, and metadata. Count actual serialized bits, not only nominal code width or trainable parameters. A surrogate entropy loss is not the achieved encoded size. The activation term must use authorized development data; code fitting or recovery training must never consume the protected final evaluation. Learned rate–distortion codecs and importance-aware compression already exist. [W3–W5, W12]

For exact reconstruction of arbitrary \(P\) weights stored with \(b\) bits each, no injective codec can map every possible checkpoint to fewer than \(Pb\) bits: there are \(2^{Pb}\) inputs and fewer shorter bitstrings. This elementary counting argument does not forbid compression of a structured model distribution; it forbids universal lossless-compression promises and excluding the predictor from the bit count.

### Memory and runtime model

For disjoint allocations at time \(t\), the required accounting is

\[
M_{\rm peak}=\max_t\{M_{\rm runtime}+M_{\phi}+M_{\rm codes/residuals}
+M_{\rm cache}+M_{\rm decoded/workspace}+M_{\rm activations}
+M_{\rm KV}+M_{\rm I/O}\}(t).
\]

The sum of the separate component maxima is an upper bound, not generally the exact peak. Avoid double-counting aliased buffers. Also record process resident memory, accelerator allocations, OS cache pressure, and swap/page faults under a stated measurement policy. Neither a smaller archive nor a fresh process establishes cold-storage access. Full reconstruction before generation forfeits the intended bounded-weight residency.

For a future ordinary autoregressive backend with a KV cache, its ideal payload is \(2LBT H_{KV}d_h s\) bytes, for layers, batch, cached positions, KV heads, head width, and bytes per element; allocator overhead is additional. Olympus's inspected generator has no such cache, so this is a planning term for a future backend, not its current allocation.

For a cache miss, let \(S_j\) be raw tile bytes, \(C_j\) coded bytes, \(B\) measured effective transfer bandwidth, \(R_j\) decode time, and \(O_j\) added scheduling/management time. With equal request overhead and serialized stages, a necessary local improvement condition is

\[
R_j+O_j < (S_j-C_j)/B.
\]

If request count or access layout differs, include both request-latency terms. With real overlap, critical-path stages may combine through a maximum rather than a sum; overlap must be measured and its buffers charged. Small tiles reduce workspace but can increase per-request overhead and reduce matrix-kernel efficiency. Decode work may compete with inference for the same compute and bandwidth. These are explanatory cost models, not performance proofs. Flash and GPU codec studies give concrete precedents for these distinctions. [W6–W8]

Approximation satisfies the elementary bound \(\|(W-\widehat W)x\|_2\leq\|W-\widehat W\|_2\|x\|_2\). This does not bound complete-model task loss without additional assumptions. As a limited sufficient condition, an infinity-norm logit error smaller than half the original top-two margin preserves greedy token selection at the same prefix; applying that condition at every successive identical prefix preserves a deterministic sequence. It says nothing about probabilistic calibration or broad assistant competence. Published findings on reconstruction objectives are conditional and not uniform. [W2, W12, W13]

### Primary falsifier

Freeze a quality margin, minimum practical latency improvement, peak-memory cap, input/context distribution, and eligible baselines before final measurement. Compare the same source checkpoint, prompt/tokenization, context, and output workload against resident execution where feasible, bounded exact streaming/caching, and a competent supported quantized execution path. An archive loader that expands every tensor is not a fair proxy for a native low-bit runtime.

**Reject the specified advantage if the method cannot improve the declared quality–memory–latency operating point after all decoder, dependency, cache, I/O, and routing costs are charged.** If quality passes but only disk size improves, report storage compression only. If quality is preserved and memory falls while latency worsens, report that tradeoff against the declared hypothesis. If exact residual storage cancels code savings or decode costs exceed saved transfer time throughout the fixed development grid, stop that candidate rather than enlarging the search to manufacture a win.

Report cold-start/first-token and steady decoding separately, with paired latency distributions and the hardware/cache regime. Kernel timing, prefill throughput, generation throughput, and full user-perceived latency are separate endpoints. Do not treat weights within one checkpoint as independent replicates of model-level capability.

## 4. Candidate B: representation and branch selection

Study this separately from the codec initially. Otherwise a combined success cannot identify which mechanism caused it.

Define each representation as an actual mapping \(g_r:x\mapsto z_r\), its shape and precision, any learned parameters, compatible solver, and measured construction/execution cost. Specify which information each mapping retains. A graph encoder given extra relations cannot be fairly credited as a superior geometry when its vector comparator never receives those relations. A geometry claim additionally needs the metric, curvature constraints, valid operations, and objective. [R1, R2]

The selector observes only input and available evidence \(e\), chooses an executable action \(a\), observes its result, updates state, and stops or abstains. One candidate metalevel target is

\[
V(a\mid e)=\mathbb E[U(\text{decision after }a)\mid e]
-U(\text{current decision})-\lambda_t\mathbb E[T_a\mid e]
-\lambda_m\Delta M_a.
\]

Use positive predicted value only among actions that satisfy remaining hard limits; otherwise stop. This is a proposed objective inspired by metareasoning, not a value estimate provided by Olympus's current score product. Count every branch generation, verifier/model/tool call, discarded candidate, representation construction, and policy inference. Selecting the best result after secretly running all options is an offline oracle, not cheap online routing. [R3–R6]

Required controls are the best fixed eligible representation, the current metadata heuristic, equal-budget simple/random routing, and an ablation with the learned selector removed while keeping the candidate pool and downstream solver fixed. An offline oracle may diagnose headroom but must be labeled with its extra computation. Freeze development, calibration, and protected final partitions by source/task family; repeated variants of the same template should not silently span them.

Evaluate task utility at the same enforced resource budget, regret to the declared comparator, abstention/risk coverage, measured costs, and calibration with held-out labeled examples. Proper scoring losses such as Brier or log loss can accompany reliability diagnostics; a single threshold check cannot establish a calibrated population. [R7]

**Selector falsifier:** the claimed improvement disappears against the strongest fixed or cheap eligible policy after matching available information, candidate pool, and total computation, or vanishes when the proposed learned selection mechanism is ablated. That is evidence against the claimed mechanism even if the combined system sometimes answers correctly.

## 5. Bounded next decision

These notes support a narrow successor protocol, not immediate broad training. Choose one explicit mechanism and a fixed development grid whose full cost fits the observed hardware/time envelope. First verify byte identity or controlled approximation, measured buffer lifetimes, deterministic dependencies, and budget enforcement on artificial engineering fixtures. Then, only under a separately frozen scientific protocol, evaluate eligible checkpoints and held-out tasks.

Record every attempted configuration and failure. Freeze selection rules, paired units, uncertainty procedure, quality margins, timeout/OOM handling, and terminal classifications before protected evaluation. Timeouts and missing measurements are not zero-quality successes and must remain visible. A tiny tensor crossover experiment can establish an engineering mechanism on that scale; it cannot establish that a multi-billion-parameter assistant is feasible on a 16 GB Apple device. Hardware availability and runtime compatibility must be observed, not inferred from a paper.

No training, scientific/capability evaluation, protected-data access, external model download, paid compute, or repository mutation was performed for this audit.

## 6. Linked primary bibliography

Publication labels distinguish an identified conference version from an arXiv preprint or an author-reported venue. Dates are the research publication/submission dates, not the date a page was crawled. The bibliography does not adopt any source's self-declared priority or performance claim as an Olympus result.

| ID | Primary source | Version or publication |
|---|---|---|
| W1 | Ha, Dai, and Le. [HyperNetworks](https://arxiv.org/abs/1609.09106). | arXiv 1609.09106v4, 2016. |
| W2 | Ashkenazi et al. [NeRN: Learning Neural Representations for Neural Networks](https://arxiv.org/abs/2212.13554). [Authors' implementation](https://github.com/maorash/NeRN). | ICLR 2023; arXiv v2. |
| W3 | Ryu et al. [Neural Weight Compression for Language Models](https://arxiv.org/abs/2510.11234). [Full text](https://arxiv.org/html/2510.11234v3). | Submitted October 2025; v3, May 2026. No conference acceptance inferred. |
| W4 | Tian et al. [PocketLLM: Ultimate Compression of Large Language Models via Meta Networks](https://ojs.aaai.org/index.php/AAAI/article/view/40610). | AAAI 2026, 40(39), 33250–33258. |
| W5 | Lamaakal. [Motion-Compensated Weight Compression](https://arxiv.org/abs/2605.24754). | May 2026 preprint, v1. |
| W6 | Zhang et al. [DFloat11](https://papers.nips.cc/paper_files/paper/2025/hash/8e8b4449c80879215ff2ee47b6dd01a6-Abstract-Conference.html). | NeurIPS 2025. |
| W7 | Zhang, Duan, and Chen. [EntroPack](https://arxiv.org/abs/2609.34185). [Full text](https://arxiv.org/html/2609.34185v1). | September 2026 preprint, v1. |
| W8 | Alizadeh et al. [LLM in a flash](https://arxiv.org/abs/2312.11514). [Full text](https://arxiv.org/html/2312.11514v3). | ACL 2024; arXiv v3. |
| W9 | Song et al. [PowerInfer](https://arxiv.org/abs/2312.12456). | SOSP 2024; arXiv v2. |
| W10 | Xue et al. [PowerInfer-2](https://arxiv.org/abs/2406.06282). | June 2024 submission; v3, December 2024. |
| W11 | Sheng et al. [FlexGen](https://arxiv.org/abs/2303.06865). | arXiv v2, 2023. |
| W12 | Chowdhury et al. [IMPACT: Importance-Aware Activation Space Reconstruction](https://arxiv.org/abs/2507.03828). | July 2025 submission; v4, April 2026. Author metadata identifies ACL 2026. |
| W13 | Choi, Thiagarajan, Glatt, and Liu. [Enhancing Accuracy and Parameter-Efficiency of Neural Representations for Network Parameterization](https://arxiv.org/abs/2407.00356). | June 2024 preprint, v1. |
| R1 | Gu, Sala, Gunel, and Ré. [Learning Mixed-Curvature Representations in Product Spaces](https://openreview.net/forum?id=HJxeWnCcF7). [Authors' implementation](https://github.com/HazyResearch/hyperbolics); [indexed primary submission](https://openreview.net/references/pdf?id=BJl-BLqTQ). | ICLR 2019. Final OpenReview access was blocked by a verification wall; only the supported mechanism is used. |
| R2 | Sáez de Ocáriz Borde et al. [Neural Latent Geometry Search](https://papers.neurips.cc/paper_files/paper/2023/hash/78efbc5386c5a7c241e7fcc482d3c3dc-Abstract-Conference.html). | NeurIPS 2023. |
| R3 | Graves. [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983). | Originally 2016; arXiv v6 reviewed. |
| R4 | Raposo et al. [Mixture-of-Depths](https://arxiv.org/abs/2404.02258). | arXiv, 2024. |
| R5 | Yao et al. [Tree of Thoughts](https://arxiv.org/abs/2305.10601). | NeurIPS 2023; arXiv v2. |
| R6 | Hay, Russell, Tolpin, and Shimony. [Selecting Computations: Theory and Applications](https://proceedings.mlr.press/r10/hay12a.html). | UAI 2012; PMLR reissue. The reissue date is not the original research date. |
| R7 | Guo, Pleiss, Sun, and Weinberger. [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). | ICML 2017. |
