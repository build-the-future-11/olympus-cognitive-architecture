# Frozen development diagnostic: parameter ownership and residency

Study ID: `olympus-residency-ownership-dev-v1-20261019`  
Status at freeze: **NOT EXECUTED**. Protocol version **2**. This document, `protocol.json`, the executable, and the exact source copies precede any outcome. `FREEZE.json` records their final hashes. The preceding resource-policy draft is retained in `pre_execution_drafts/protocol-resource-v1.json`; only the pre-execution admission estimate was amended, on the parent's instruction. No hypothesis outcomes informed that change.

## Question and claim boundary

Does the one-shard dictionary-cache budget in the exact PR29 `SpatiotemporalParameterStore` also bound persistent CPU parameter storage owned by a module across changing calls to `materialize_module`?

The explicit implementation variable is `cache_budget_bytes`. Its dictionary counter accounts for serialized shard file sizes. A functioning dictionary bound is a narrower claim than a bound on all materialized weights or process memory. We test whether the stronger whole-module residency interpretation is justified. A counterexample will not be described as a violation of an explicit guarantee that the implementation never made.

This is a fresh artificial systems mechanism diagnostic. It is a prerequisite to credible claims about memory-saving inference, not a new learning algorithm, model-quality experiment, architecture improvement, or LLM benchmark. It consumes no prior scientific study, held-out sample, or checkpoint. There is no requirement to obtain a positive outcome.

## Exact sources and isolation

The subject is `olympus/models/spatiotemporal_lazy.py` at commit `12b7c785958b741ec48834a74e81df0c674d327b` of `build-the-future-11/olympus-cognitive-architecture` (open PR29). Its SHA256 is `09c7a7aa4a1edcf19f3e4c4084e44d627dc606a43172e8853c02d3fa59fff939`; its Git blob is `8cb9839530bdb165f50b9c716deebd21417b0472`.

The exact `olympus/core/schemas.py` dependency from that commit is copied beside it, with SHA256 `e89c498176fb86b52f1a49b8c9961aa8f5cb431ef6a953abfff09cbef1e597f0`. The exact main ResourceGovernor from commit `dc46b831e2aa88a5e669e1343463dc74ef1270a4` has SHA256 `83c95b1ba70b5e3fd6ff4d29179f5b48212eadb184da122fe6a40f4c83ac26f0`. `SOURCE_MANIFEST.json` binds every byte and repository path. None of these three source files is edited.

The runner verifies SHA256 and Git blob identities before execution. Each fresh child creates namespace-only `olympus`, `olympus.core`, and `olympus.models` modules, then executes the exact copied schema and subject files by absolute path. It does not execute `olympus/__init__.py` or import the working repository. The actual Pydantic schema is used; there is no substitute validator or loader. This deliberately isolates a module-level API and is not a test of PR29 package integration or a port to main.

The frozen runtime is Linux, Python `3.14.8`, Torch `2.14.1+cpu`, and Pydantic `2.14.1`, as observed through installed-package metadata before freezing. All tensors are explicit CPU float32; Torch intra-op and inter-op threads are one; deterministic algorithms are requested. The metadata check imports no Torch during protocol preparation. The diagnostic has not been executed during this audit.

## Fixture, workload, and budget

There are three parameter shards. Shard `i` contains exactly one tensor named `layer{i}.weight`: the 128 by 128 matrix `W_i = (i+1) I`. Each tensor occupies 65,536 bytes in float32, for 196,608 mathematical weight bytes across the fixture. `torch.save` creates local shard files once, in a separate fixture process. The manifest binds their actual file hashes and serialized lengths. Those serialization hashes identify this fixture realization; cross-version byte-for-byte serialization stability is not claimed.

The input is the 1 by 128 row vector `x[j] = j/128`, `j = 0,...,127`. The exact reference for selecting shard `i` is `x * (i+1)`. These dyadic values and diagonal integer weights make exact output equality an appropriate, declared check. No labels, gradients, optimization, data selection, or external corpus are used. The RNG seed is fixed to `2026101901`; the fixture itself is deterministic and does not use random draws to choose values.

The meta model is `nn.ModuleDict` containing three bias-free 128-to-128 linear layers. Spatial boxes are centered at `(0,0,0)`, `(100,0,0)`, and `(200,0,0)`, each with half-width one on all axes. Queries use the selected center, zero velocity, zero lookahead, resolution one, and the integer step as timestamp. The trace is exactly `[0,1,2,0,1,2]`. The actual PR29 planner must select exactly the requested single shard at every step.

Let `B` be the largest of the three actual serialized file sizes. Before either arm starts, the fixture must satisfy all of the following: every serialized shard is at most 100,000 bytes; `65,536 <= B < 131,072`; twice the smallest serialized shard exceeds `B`; and all fixture files together occupy at most 8 MiB. These conditions establish that one shard fits and two cannot fit. The same manifest, files, `B`, and six queries serve both arms.

## Two arms and what is matched

| Arm | Module ownership | Same fixed work |
|---|---|---|
| `persistent_module` | Retain one meta-initialized module across the six calls. Previously assigned layers remain owned by that module unless the unchanged implementation releases them. | Same fixture, one-shard budget, plans, six selected-layer forwards, and exactly six disk loads. |
| `recreated_module` | Before each call, release the preceding module, run garbage collection, and create a fresh identical meta module. Retain the same store across calls. | Same fixture, one-shard budget, plans, six selected-layer forwards, and exactly six disk loads. |

Each arm executes in its own fresh child process. The control removes persistent module ownership without modifying the loader, planner, cache, shard representation, or selected computation. Module construction and collection add CPU overhead, so elapsed-time differences are not an efficacy metric and cannot support a speed claim. This is an equal weight-budget, equal I/O-count, equal selected-mathematical-work ownership control, not a production replacement algorithm.

## Primary measurement

After each selected-layer forward, delete output/reference tensors and collect garbage. Inspect the actual current module parameters and buffers and actual cached tensor objects. Exclude meta tensors. For each CPU tensor obtain its `untyped_storage`, deduplicate by `(data_ptr, nbytes)`, and sum storage sizes over the union of module and cache owners:

`P_t = bytes(unique CPU storage reachable from current module or store cache at completed request t)`.

This is a measured lower bound on persistent live process storage. It includes storage that remains referenced by module parameters after cache dictionary eviction. It deliberately excludes Python interpreter memory, activations, serializer staging, allocator reserves, native-library workspace, and other non-parameter allocations. If this lower bound alone exceeds `B`, the stronger whole-module bound is false. If it remains within `B`, that does not prove a full process or transient peak memory bound.

The report separately records module-only bytes, cache-only tensor bytes, unique storage identifiers and owner names, and the implementation's serialized-file cache counter. The observer retains only scalar metadata and weak references between requests; it does not keep old tensor or storage objects alive. Weak references to previous actual parameter/tensor owners provide supporting lifetime evidence. Their expiration does not prove that an allocator returned pages to the OS. Child maximum RSS, page faults, and host/cgroup memory and swap samples remain secondary environment observations.

Every request retains the exact output values, output shape, equality against the analytical reference, and SHA256 of little-endian float32 output values. The expected output is not inferred from another run. Every plan, assignment's remaining-meta list, resident shard list, and cumulative disk-load count is retained. This validates only the requested layer's mathematical output. A full forward through all partially loaded layers is outside this experiment.

## Predeclared outcome rules

| Outcome | Rule and permissible conclusion |
|---|---|
| `RESIDENCY_INTERPRETATION_FALSIFIED` | All 12 requests and controls are valid, the recreated-module arm has `P_t <= B` throughout, and the persistent arm has `P_t > B` at least once. Reject the stronger whole-module residency interpretation for this exact source and trace. |
| `NOT_FALSIFIED_IN_BOUND` | All controls pass and both arms stay within `B` for all six steps. This finite test did not falsify the interpretation; it does not establish a general memory guarantee. |
| `INCONCLUSIVE` | Execution completes but the recreated-module arm also exceeds `B`. The contrast does not isolate persistent ownership under this protocol. Preserve all values and do not invent another baseline after inspection. |
| `INVALID` | A source/runtime/fixture/plan/output/disk-count/measurement or execution contract fails. Preserve all partial artifacts. No primary mechanism conclusion follows. |
| `BLOCKED` | The exclusive workload lock, effective memory/swap guard, or fixed time budget prevents or stops execution. Preserve the blocked attempt; it is not a negative scientific result. |

A valid falsification exits with process status zero and `status=COMPLETED`, because a negative hypothesis result is successful execution of the protocol. An invalid or blocked attempt exits with status two. The primary outcome is a deterministic finite counterexample test. Confidence intervals and p-values would not add meaningful information here and are not computed. The six steps are not treated as independent population samples.

## Resource admission and stopping rule

The parent acquires the existing ResourceGovernor lock `/tmp/olympus-{uid}.workload.lock` before importing Torch in a child. It requires at least 2 GiB effective available memory, taking the minimum of host `MemAvailable` and conservative cgroup v1/v2 available-memory estimates along the process's mounted ancestry. The estimate is `max(0, limit-used+credit)`, where `credit=min(used,max(0,min(inactive_file,file)-shmem-file_mapped-file_dirty-file_writeback))`. All required counters must exist and be nonnegative; otherwise credit is zero. Cgroup v1 uses the corresponding `total_*` counters. Raw `limit-used`, actual usage, and credit are recorded separately. This conservatively discounts potentially reclaimable inactive file cache; it is an availability estimate, not a guarantee that reclamation is immediate. The 2 GiB floor is unchanged. `RESOURCE_POLICY_ORIGIN.json` binds the parent's reviewed working-tree policy source; the exact main ResourceGovernor itself remains unchanged.

Unknown Linux cgroup limits fail closed. Host swap utilization and any finite nonzero cgroup v2 swap limit must remain at most 90%. Cgroup v1 swap is represented by the host observation; no cgroup-v1 swap-specific limit guarantee is claimed. The parent samples these conditions while each child runs and kills the child process group on a resource/time violation. A blocked result is retained without retries.

Only one child runs at a time. The full budget is one fixture process, two arm processes, 12 selected-layer forwards, zero training updates, zero protected evaluations, 30 seconds per child, and 90 seconds total. Each output directory must be fresh; an existing path is refused. The source invokes no network operation and an audit hook rejects common Python socket connection/bind/name-resolution operations; this is not a claim of OS-level network isolation. No GPU, paid or remote workload, model-provider call, or external data access is part of the command.

Execute this matrix once, then stop. Do not alter seeds, sizes, trace, threshold, control, or mechanism after inspecting outputs. A necessary harness repair requires a new protocol/executable version and retention of the original failed attempt. A negative scientific result requires no repair to produce a positive one. Root owns any execution and subsequent repo publication; this audit only prepares reviewable source and protocol.

## Lineage and subsequent work

This new study follows PR29's partial-materialization engineering tests. Those tests exercise a single meta-module assignment and cache identity; they do not establish model-wide storage bounds across repeated eviction. This study is distinct from the consumed v2-v5 routing/temporal experiments, EditCLEVR, Foundry promotion suites, and the unfrozen O1 proposal. None of their samples, checkpoints, or metrics are inputs.

If the interpretation is falsified, a future engineering contract should define ownership leases or module unloading, reserve peak staging space, and verify a complete execution path before asserting a local-RAM budget. Such a repair is not implemented or evaluated here. If not falsified, the result remains restricted to the finite declared trace. Any later predictive-prefetch study needs its own frozen protocol and a demand-loading/LRU control with equal physical memory and I/O budgets, actual route correctness, predictor overhead, and miss costs. Existing predictive and spatiotemporal expert-prefetch work prevents a novelty claim based on those names alone.

No outcome here demonstrates LLM quality, better reasoning, parameter reconstruction, faster inference, large-model support, safe 16-GB operation, or Mac support. The valuable result is a source-bound answer to one necessary ownership question.
