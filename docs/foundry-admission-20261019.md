# Foundry admission and checkpoint identity — 2026-10-19

This change closes a bounded engineering slice: the verification registry admits
only an eligible, evaluated checkpoint, and consumers identify the exact bytes
they use. It also preserves existing exports on failure and shows actual artifact
health in the CLI, API, and console. The only trained model in this workflow is the
built-in `FoundryVerificationBigram`; no Hermes or assistant-quality claim follows.

## Source and ownership

- Repository: `build-the-future-11/olympus-cognitive-architecture`.
- Predecessor: `dc46b831e2aa88a5e669e1343463dc74ef1270a4` on `main`.
- Development branch: `fix/olympus-foundry-admission-20261019`.
- One checkout writer: root assistant. Independent agents reviewed source and
  supplied drafts outside the checkout; they did not execute the reported tests.
- Protocol and append-only history: `RESEARCH_STATE.json`, active study
  `foundry_admission_20261019`.
- Results and source bindings: `research/verification/foundry_admission_20261019/`.

## Defects and mechanism

Previously, `register_model` trusted a `VERIFIED` label. A caller could register an
unevaluated checkpoint from a failed experiment and generate from it. Registration
now reads checkpoint, experiment, dataset, and evaluation records in one SQL
snapshot. It requires the implemented family/runtime and canonical model identity,
an eligible owning experiment, matching dataset provenance, and a passing evaluation
with finite, comparable held-out NLL/perplexity and a positive transition count.

The shared validator reads the checkpoint once, verifies its SHA-256 and byte count,
parses the bigram matrix, and binds its embedded provenance to the registry records.
Health, generation, and export repeat this admission check for existing rows, so a
historical registry label does not bypass current artifact validation. Consumers use
the returned byte snapshot even if the source path changes immediately afterward.

Export used to overwrite a destination before checking the copied bytes, then delete
the destination on a mismatch. It now verifies before touching the target and writes
through a temporary file with atomic replacement. Failed publication cleans up the
temporary file and leaves any earlier destination intact.

The CLI and API retain `registered_status` but expose `UNAVAILABLE` and the live health
reason for an invalid artifact. The console uses that health to count available
models, disable selection, and clear an unavailable selection on refresh.

The tensor checkpoint loader already hashed the bytes passed to safe deserialization.
Evaluation and quantization now retain that digest instead of reopening the path later.
Candidate, explicit baseline, and quantized-source identities therefore describe the
actual consumed tensors even when the path is replaced or deleted after loading.
The existing two-value private loader remains as a compatibility wrapper.

Deep promotion now checks the internal evidence contract as well. It validates
evaluation populations and schema, recomputes category regressions and quality
fractions, binds quantization measurements to the evaluated source loss and
normalization, and hashes the actual quantized artifact. A contradictory summary
flag or missing/mismatched artifact cannot substitute for these checks. Release
digests identify the report, model card, and artifact snapshots actually consumed.
Output alias checks protect inputs, and temporary publication files are cleaned up.
These checks preserve the producer's existing quality thresholds.

CLI and API startup now defer tensor demos and deep-model backends until the
requesting command or endpoint executes. A verification-family process therefore
does not import Torch merely to inspect or serve its small JSON checkpoint.

## Verification contract

The new tests use artificial records and tiny, untrained tensor fixtures. They are
engineering regressions, not research evaluation. The first 52 new cases reproduce
45 predecessor failures and retain seven unchanged controls. Eleven additional
resource-accounting cases validate conservative cgroup estimates. Two fresh-process
import-blocker cases fail before startup deferral and pass afterward. Of 68 new
promotion-evidence cases, 57 fail against the preceding implementation and 11 pass.
These are cases, not a count of independent defects or scientific samples.

The expanded complete Python suite passes 432 tests in 14.42 seconds, with 88.53%
combined statement/branch coverage against an 85% gate. This supersedes the initial
351-test, 87.99% implementation check; both raw results are retained. All seven web
tests and the TypeScript/Vite production build pass. Ruff and strict Mypy pass on
the expanded source. The final wheel and source distribution both build and pass
Twine metadata validation at the executed source revision. The latest web suite
passes seven tests in 2.20 seconds; its complete npm command takes 3.04 seconds.

The dependency checks have different dispositions: npm reports zero vulnerabilities;
the strict Python audit cannot fully audit the installed CPU Torch build because
`torch==2.14.1+cpu` is not represented by that exact version on PyPI. This is an audit
limitation, not a passed Python dependency gate. Raw failures are retained.

The final full Python command is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout 120 .venv/bin/python -m pytest --cov=olympus --cov-branch \
  --cov-report=term-missing --cov-report=json:artifacts/coverage.json -q
```

ASGI tests stalled in this session's restricted execution sandbox. The same actual
API tests and full suite passed using the host executor. The diagnostic timeout is
retained, and no API behavior was altered to accommodate the sandbox.

## Reproduce the complete verification lifecycle

Use a clean checkout of the source commit recorded in the workflow receipt and an
installed Python 3.14 development environment. A Linux CPU wheel can be installed
before `pip install -e '.[dev]'` when GPU dependencies are unnecessary. The receipt
records the actual package versions used here.

```bash
.venv/bin/python scripts/verify_foundry_admission.py \
  --output artifacts/foundry-admission-fresh
```

The output directory must not exist. The script uses one workload lock, requires
2 GiB of effective free memory (including Linux cgroup limits), applies the existing
swap ceiling, and checks disk headroom. It masks GPU devices, sets CPU thread counts
to one, and limits the command sequence to 120 seconds with 30 seconds per child.
Deadline checks are polled; synchronous resource probes can delay termination,
especially on macOS. The measured run here is Linux, and this is not a real-time
scheduler guarantee.
The first attempt stopped before launching any child because raw cgroup headroom
was below the floor while most charged memory was file cache. Resource policy v2
keeps the 2 GiB floor and records both raw usage and a conservative cache credit:
`max(0, min(inactive_file, file) - shmem - mapped - dirty - writeback)`. Missing
counters receive no credit. Every ancestor limit and host available memory still
apply. This estimates usable headroom; it does not guarantee immediate reclamation.
See [Docker's working-set convention](https://docs.docker.com/reference/cli/docker/container/stats/)
and the [kernel counter definitions](https://docs.kernel.org/admin-guide/cgroup-v2.html).
The failed v1 receipt and dependency-cache preparation records are retained.
The v2 attempt passed initial admission but stopped during eager Torch startup when
the memory estimate crossed the same floor. No training output was produced. The
startup import repair has a distinct frozen v3 attempt; it keeps the corpus, seed,
quality thresholds, and resource policy unchanged.
V3 stopped at preflight before any child. One final, separately frozen v4
environment preparation released already-used dependency/build cache pages using
advisory `POSIX_FADV_DONTNEED`, without changing file contents. V4 then completed
on clean source `92cc121e6f1074de72eaee5a8cb670170107bbc7`. All stopped receipts
and preparation records are preserved. Only v4 produced training output; there was
no quality-driven rerun.
It accepts only the reviewed 674-byte built-in corpus, whose SHA-256 is
`6f4823e0503426bda11aa4bffe8357963f3bc97a62ea40c2b69700af166a8eab`.

The sequence runs the actual CLI verification job, reopens the registry for status
and model listing, generates locally, and exports in a fresh process. A separate
fresh process uses the actual FastAPI application through ASGI and its normal Foundry
dependency. It verifies the same checkpoint identity and identical CLI/API generation
for prompt `user: verify evidence`, seed 7, temperature 0, and 24 new characters.
The bigram trains on the prefix; held-out target transitions come from the remaining
suffix, with only the preceding context character shared at the split.

Each run retains `receipt.json`, all stdout/stderr, the registry, the materialized
corpus, checkpoint, and portable export. The receipt includes source and artifact
hashes, commands and exit statuses, held-out metrics, process peak RSS, CPU time,
major page faults, host/cgroup memory observations, and swap change. Checkpoint
provenance contains an experiment timestamp, so hashes bind one run and are not
expected to match across reruns.

### Recorded lifecycle result

The completed sequence took 4.272115 seconds, with a maximum child RSS of
67,338,240 bytes (64.21875 MiB) and no swap use. Minimum recorded estimated headroom
was 2,231,209,984 bytes, 83,726,336 bytes above the 2 GiB floor. On 135 held-out
target transitions, candidate NLL was 1.9001575711 versus the uniform baseline's
3.17805383035, and perplexity was 6.68694802902 versus 24. The checkpoint, internal
export, and portable export were the same 1,712 bytes, SHA-256
`0292e4d08ff43c4638b42ba5013eb897dad60ab4fbbdb609f661b0b731ccdb06`.
The admitted model ID was `foundry-verification-bigram-0292e4d0`.

The actual API was exercised through ASGI transport in a fresh process; this was
not a listening network server or live-browser deployment. Raw `memory.stat`
components were not saved, so historical cache credit cannot be independently
rederived from individual counters. Its recorded headroom arithmetic and all
artifact/source hashes were checked by read-only reviewers. This is evidence
verification, not independent experiment reproduction. Complete commands and
regenerable tables are in the [evidence handoff](../research/verification/foundry_admission_20261019/README.md).

For a manual generation after the workflow, take the actual model ID from
`receipt.json` or `foundry list-models`:

```bash
.venv/bin/python -m olympus.cli foundry list-models \
  --root artifacts/foundry-admission-fresh/registry
.venv/bin/python -m olympus.cli foundry generate MODEL_ID 'user: verify evidence' \
  --root artifacts/foundry-admission-fresh/registry \
  --max-characters 24 --temperature 0 --seed 7
```

## Kaggle launch and retrieval if independently admitted

Kaggle account access, quota, accelerator settings, and running jobs were not verified
in this session: the connected-browser initialization timed out, and no dedicated
Kaggle connector was exposed. No Kaggle job was launched. This CPU script is the
ready-to-run fallback; it does not remove any stop from historical notebooks.

Before a later notebook run, inspect the signed-in account, current jobs, remaining
quota, Internet setting, and CPU accelerator selection. Obtain the exact source
commit from this result's receipt, install its declared environment, and run the
command above with a new output under `/kaggle/working/`. Retain notebook identity
and the environment/installation log alongside the script's receipt. Download the
entire output directory as a notebook output artifact, including the registry and
checkpoint, then compare every `artifact_sha256` entry with the retrieved bytes.
That later execution has no result until the actual notebook output is retrieved
and checked. This procedure authorizes no protected dataset or larger model run.

## Limits and next action

This registry is a consistency boundary for trusted local evidence. It does not
authenticate a malicious database writer, and it requires an applicable passing
receipt rather than implementing later-receipt revocation. Historical artifact
metadata was reviewed, but unavailable historical checkpoint bytes were not rerun.

The deep-promotion repair checks consistency and byte identity. It does not prove
that arbitrary hash-matching artifact bytes load as the claimed model, independently
reproduce quality or fresh-process serving, or authenticate a report writer. Report
and release publication are individually atomic but not a joint crash-atomic
transaction. A rejected later attempt does not revoke an older release file.
Quantization retains its pre-existing fixed-filename overwrite behavior; the
snapshot identity repair does not make repeated quantization publication atomic.
Mac/MPS behavior, QLoRA efficacy, Hermes readiness, lazy reconstruction efficiency,
and scientific capability are unproven by this work.

The next promotion implementation action is to require an executable artifact
contract with fresh-process load and serving evidence bound to the exact artifact
and runtime, before a real model can claim deployment readiness. This requires its
own bounded protocol and preserves the existing negative scientific results.
