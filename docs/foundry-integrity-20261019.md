# Foundry artifact integrity — 19 October 2026

## Disposition and ownership

**IMPLEMENTED and LOCALLY_TESTED** for the declared artifact-admission and serving
slice. This is infrastructure engineering. It establishes no new assistant,
Hermes, reconstruction, large-model efficiency, or scientific capability result.

Repository: `build-the-future-11/olympus-cognitive-architecture`.
Development branch: `codex/olympus-foundry-integrity-20261019`.
Base: `dc46b831e2aa88a5e669e1343463dc74ef1270a4`.
The containing commit identifies this deliverable; the retained workflow also
hashes 101 source/test/configuration files. Execution occurred on that base plus
the explicitly recorded dirty-tree patch, so the checkpoint correctly retains
the `+dirty` code identity. It is not represented as an execution of clean main.

The root engineering agent owns this branch. A separate local worktree was used
for the deep-artifact implementation, followed by root integration and one CPU
test lane. Review agents made no writes to the delivery branch and ran no tests.
No default-branch merge, release, external deployment, protected capability run,
paid job, schedule, or changes to related repositories are part of this work.

## Defects repaired

The durable registry previously accepted a model record without independently
establishing the full dataset/checkpoint/evaluation relationship. A caller could
substitute dataset metadata or rely on a status field, while serving could trust
an unsupported record. The repaired admission path validates the canonical
registered dataset, consumes and hashes checkpoint bytes once, checks the model's
supported identity and provenance, and requires matching passing evidence.

Initial bigram admission additionally reconstructs the exact training-prefix
counts and recomputes held-out metrics. This prevents fitting the held-out suffix
while claiming a training-prefix identity, and prevents numeric comparison
tolerance from promoting an actual uniform-baseline tie. Serving revalidates
artifact identity and recorded admission evidence; it does not repeat training
and evaluation on every request or retrospectively re-admit old registry entries.

The deep path previously rehashed some mutable paths after consuming them and
could emit a promotion manifest without loading the referenced quantized weights.
It now binds reports to the consumed snapshots, validates real 4/8-bit artifact
structure and loadability, and rejects incomplete or contradictory reports.
Successful artificial promotion tests use real tiny tensor artifacts but explicit
fabricated gate inputs for scale and quality. Those tests are not model promotion
evidence. The actual historical tiny Transformer remains **NOT_PROMOTED**.

API and CLI model listings now expose current availability separately from the
historical status. The web console disables unavailable or stale selections and
refreshes after failures. Atomic portable exports preserve an existing destination
when admission fails and protect registered artifacts and symlink aliases. Deep
evaluation, quantization, and promotion reject reused output files and collisions
with their inputs or sibling release manifest.

## Source-to-execution map

| Contract | Implementation | Retained verification |
| --- | --- | --- |
| Canonical dataset and checkpoint identity | `admission.py`, `store.py`, `service.py`, `schemas.py` under `olympus/foundry/` | `tests/test_foundry_admission.py`; healthy pipeline and fresh recovery |
| Exact prefix fitting and strict held-out improvement | `store.py` initial model registration | count-substitution and actual-baseline-tie regressions |
| Same consumed bytes for evaluation/quantization | `eval_suite.py`, `quantization.py` | `tests/test_deep_artifact_integrity.py`; existing dataset snapshot tests |
| Actual quantized weights and coherent quality reports | `quantization.py`, `promotion.py` | real tiny Torch payloads; missing/corrupt/malformed/contradictory evidence cases |
| Non-destructive output paths and exports | `artifacts.py`, `service.py`, deep consumers | source, sibling, earlier-output, inward/outward symlink and external-registration cases |
| Current serving availability and failures | `olympus/api.py`, `olympus/cli.py`, `apps/forge-web/src/` | API/CLI integration, 11 web tests, fresh loopback API receipts |
| Reproducible measured lifecycle | `scripts/verify_foundry_integrity.py` | `research/verification/foundry_integrity_20261019/workflow.json` |

## Measured lifecycle

The successful attempt ran from **11:38:51 to 11:39:07 UTC** and took
**16.428 seconds**. Its ten asserted lifecycle/error contracts passed.
The fixed corpus has 674 characters and SHA-256
`6f4823e0503426bda11aa4bffe8357963f3bc97a62ea40c2b69700af166a8eab`.
Training used the first 539 characters, seed 7, and smoothing 0.25. The 538 training
target positions and 135 held-out target positions are disjoint; one boundary
character supplies context for the first held-out target. All held-out characters
occur in the training alphabet.

| Held-out metric | Frozen uniform baseline | Trained verification bigram |
| --- | ---: | ---: |
| Negative log likelihood | 3.1780538303 | 1.9001575711 |
| Perplexity | 24.0000000000 | 6.6869480290 |
| Next-character top-1 accuracy | 0.0416666667 | 0.3925925926 |
| Evaluated transitions | 135 | 135 |

The admitted model is `foundry-verification-bigram-b749d3f9`. Its 1,718-byte
checkpoint and both exports have SHA-256
`b749d3f93fcef0c182a22da3a80ce1ee7f49807dc06227a4ca5c49626cc8dace`.
The CLI, a fresh recovery process, and a fresh API process generated identical
80-character results for `user: request verify evidence`, temperature 0.7, seed 7.
Each generation's evidence was explicitly compared to the admitted checkpoint.
The output is character-bigram text, not a meaningful assistant answer.

The final 110 added regression cases were also run against the exact unmodified
base in a detached worktree, using the same test files: **108 failed and two
controls passed**. The repaired full suite includes all 110 cases and passes.

| Local validation | Result |
| --- | --- |
| Full Python suite | 409 passed; zero failures or skips |
| Branch-enabled coverage | 87.61%; required minimum 85% |
| Strict mypy | 89 source files; no issues |
| Ruff | Passed |
| Web tests and production build | 11 tests passed; build passed |
| Python project and locked web dependency audits | No known vulnerabilities reported |
| Wheel/source distribution and metadata | Both built and passed metadata checks |
| Wheel/source binding | 65 runtime files matched the tested snapshot; both bundled corpora verified |

The first test-transport collection failure, restricted-runner timeout, early
regression witnesses, type-check correction, one fixture-schema error, and blocked
workflow preflight remain in the archive. The receipt distinguishes them from
the final accepted results.

A second independently materialized registry was intentionally corrupted. The
checkpoint became unavailable; API generation returned 422, unknown models
returned 404, CLI generation exited 1, and rejected export preserved an existing
destination. The healthy checkpoint remained unchanged. Both loopback API
processes were stopped and reaped; their recorded exit -15 is expected SIGTERM
shutdown, not a failed assertion.

This is one deterministic engineering execution and one artificial failure
fixture. There is no confidence interval, seed selection, independent replication,
or population-level capability inference.

## Host and resource interpretation

The measured host was Linux x86_64, CPython 3.14.8, Torch 2.14.1+cpu, with an
**8 GiB cgroup memory ceiling** and nine visible CPUs. Numeric-library thread
environment variables requested two threads; total OS thread counts were not
measured. Before the successful run, disk free space on the output filesystem was
20.42 GiB. Host and cgroup swap usage were zero at the recorded snapshots.

The largest child process peak RSS was **283,013,120 bytes (269.90 MiB)**; the
coordinator peaked at **277,049,344 bytes (264.21 MiB)**. These include interpreter
and imported-library memory and are measured separately with `wait4`/`getrusage`.
They are not a measured simultaneous process-tree peak. The 1,718-byte checkpoint
is not total RAM. Gradient optimizer state, neural activations, and KV cache are
not applicable to this bigram runtime; no Transformer efficiency claim follows.

The first workflow attempt was blocked before any child command because its
initial `limit - current` calculation treated all cached file pages as unavailable.
The original failure receipt is retained. A read-only process inventory found no
training or serving job. Clean-cache advice on this task's dependency files
released about 604 MiB without deleting or modifying file bytes.

Before the successful attempt, resource accounting was explicitly versioned to
`host-and-cgroup-working-set-v2`: the minimum of host available memory and cgroup
unallocated space plus inactive file cache, conservatively subtracting all dirty
and writeback pages. Active file cache and reclaimable slab receive no credit.
The 2 GiB threshold remained in force before every subprocess stage. Estimated
headroom was approximately **2.78 GiB** at the lowest recorded preflight. This is an estimate
of available working-set capacity, not a physical reservation or continuous peak
measurement. Docker's Linux CLI likewise separates inactive file cache from total
container usage; the kernel documents the underlying counters:

- <https://docs.docker.com/reference/cli/docker/container/stats/>
- <https://docs.kernel.org/admin-guide/cgroup-v2.html>

Ryan's 16 GiB Apple Silicon Mac, MPS, CUDA, Kaggle, and local Ollama were not
executed or independently verified in this session.

## Reproduce and inspect

Use the development branch and a supported CPython 3.14 environment. The measured
Linux installation used `uv` and the CPU Torch backend:

```bash
git switch codex/olympus-foundry-integrity-20261019
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python --torch-backend cpu -e '.[dev]'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/verify_foundry_integrity.py \
  --output artifacts/integrity-my-attempt-1
```

Use a fresh output directory. The script verifies that it imports this checkout,
uses an exclusive workload lock, enforces a 120-second workflow budget and bounded
child deadlines, records per-stage resource snapshots, and retains all outputs.
If preflight fails, inspect that receipt before selecting another permitted host.

For the minimal manual lifecycle:

```bash
.venv/bin/python -m olympus.cli foundry verify-pipeline --root artifacts/my-foundry
.venv/bin/python -m olympus.cli foundry list-models --root artifacts/my-foundry
.venv/bin/python -m olympus.cli foundry generate MODEL_ID \
  'user: request verify evidence' --root artifacts/my-foundry \
  --max-characters 80 --temperature 0.7 --seed 7
```

Replace `MODEL_ID` with an available ID from the list. Experiment timestamps and
the code identity are embedded in checkpoint provenance, so a new attempt obtains
a new checkpoint ID/hash even when counts and generated text are deterministic.

`receipt.json` lists exact validation commands, exits, source identities, and
log hashes. `workflow.json` is the original successful machine-readable report;
`verified-model.json` is the actual exported checkpoint. `verification-logs.zip`
contains the retained test/build/audit logs, failed preflight, and full healthy and
corruption workflow directories. Archived SQLite records retain original absolute
paths as historical evidence. Recreate a registry with the script when moving to
another host; do not reinterpret the archived paths as a portable installation.

## Remaining boundaries and next action

Historical negative model results and earlier study records are preserved. The
deep report contract checks consistency and actual quantized-artifact structure;
it does not authenticate who produced supplied measurements or independently
rerun a protected capability evaluation. Promotion does not separately load every
source-checkpoint tensor when callers supply mutually fabricated reports; the real
evaluation and quantization producers do load the source. That is a distinct
remaining hardening opportunity, not a new positive capability result.

The existing deep evaluation weighting and bigram unknown-character policy were
not changed. New validation does not establish new scientific metric semantics.
Output freshness is an intentional compatibility change, documented above.

The source was prepared for an isolated branch without a pull request. The current
workflow triggers on pushes to main and pull requests; a development-branch push
does not run it. Current account-level Actions allowance was unavailable through
the repository connector, so no hosted validation or CI-budget availability is
claimed. PR29 and related repositories remain separate, unmodified work.

**Next single action:** reproduce this branch's bounded script on Ryan's intended
16 GiB Mac using a fresh output directory, retain the resulting `workflow.json`,
and compare its source hashes and lifecycle assertions before integrating the
repair. This session has not independently reproduced that host result.
