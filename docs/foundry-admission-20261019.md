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

## Verification contract

The new tests use artificial records and tiny, untrained tensor fixtures. They are
engineering regressions, not research evaluation. Of 52 new cases, 45 fail on the
exact predecessor and seven are unchanged controls. The repaired complete Python
suite passes 351 tests, with 87.99% combined statement/branch coverage against an 85%
gate. All seven web tests and the TypeScript/Vite production build pass. Ruff, strict
Mypy, distribution construction, and distribution metadata validation pass.

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

The separate deep-promotion gate still trusts some summary flags without fully
validating quantization measurements/artifact identity and evaluation completeness.
Quantization also retains its pre-existing fixed-filename overwrite behavior; the
snapshot identity repair does not make repeated quantization publication atomic.
Mac/MPS behavior, QLoRA efficacy, Hermes readiness, lazy reconstruction efficiency,
and scientific capability are unproven by this work.

The next single implementation action is to make `promotion.py` reject contradictory
quantization/evaluation facts and missing or mismatched artifacts before emitting
`PROMOTED`, with fresh artificial regression fixtures and the existing negative
scientific results preserved.
