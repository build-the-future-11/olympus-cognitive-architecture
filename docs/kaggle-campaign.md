# Olympus bounded campaign handoff — 2026-10-16

## Maintained notebook launcher

Build a fresh upload package from the current checkout:

```sh
rtk proxy .venv/bin/python -m notebooks.build_kaggle_runner --output-dir artifacts/kaggle-runner-new
```

Import the generated `olympus_kaggle.ipynb` into Kaggle and attach its accompanying
`olympus-source.tar.gz` as a private input. The notebook verifies the pinned archive,
extracts into `/kaggle/working`, runs resource/device preflight and numerical
regressions, then executes the eight-step smoke and exports recovery evidence.
It defaults to CPU. Select a GPU session and set `DEVICE = "cuda"` for CUDA;
unavailable CUDA fails explicitly. The six-cell development matrix requires
`RUN_DEVELOPMENT = True` after smoke validation. Keep the image's CUDA Torch;
do not install the Python-3.14 application package into the source-run environment.

For a self-contained GPU notebook, add `--device cuda --embed-source` to the
builder command. This embeds the same hash-verified source bundle and avoids a
separate dataset upload. The code cells can also be pasted into Kaggle's editor
when local-file upload is unavailable. Numerical regression results are retained
as `results/regressions-cuda.xml` (or `regressions-cpu.xml`) in recovery archives.

Re-executing setup verifies the extracted source. Re-executing a campaign validates
completed artifacts or resumes its saved optimizer state. For a fresh session,
attach the recovery archive and original source archive, set `RECOVERY_ARCHIVE`
and its externally saved `RECOVERY_SHA256`, and retain the original device/runtime.
Clear the recovery setting after restoration to continue the existing results.
Recovery archives include data and partial checkpoints; keep the source bundle
separately. Save outputs and explicitly stop the platform session when finished.
The campaign wall budget does not meter account quota or stop the platform GPU.

The old `olympus_kaggle_20261016.ipynb` and `build_kaggle_launch.py` remain disabled
historical drafts. Use the maintained builder above.

Verified Kaggle execution: [version 358974216](https://www.kaggle.com/code/ryangomez010/olympus-gpu-smoke-and-resume-2026-10-16?scriptVersionId=358974216)
completed successfully in 64.2 seconds on Tesla T4 with Python 3.13.15,
Torch 2.11.0+cu128 and CUDA 12.8. All three selected CUDA checks passed
(54 deselected), including interrupted/resumed training with exact model-weight
parity against uninterrupted training. The eight-step GPU smoke completed.
Downloaded recovery checksums, restoration and completed artifacts were verified
locally. See the [verified receipt](../artifacts/kaggle-gpu-qualification-20261016/VERIFIED_GPU_RESULT.json)
and [executed output](../artifacts/kaggle-gpu-qualification-20261016/notebook-output.log).

Local verification also passed 55 numerical CPU tests (two CUDA skips) and all
six maintained launcher tests. The earlier 73 control/archive checks passed,
and the generated CPU notebook completed extraction through recovery export.
These are synthetic engineering checks; scientific results are not promoted.
The original readiness record below is retained as historical context.

## Original handoff

Status: **BLOCKED** pending numerical regressions and target-machine preflight.
The new runner is implemented. No Kaggle session, GPU quota, external tracking or
submission was used. See [readiness](../research/KAGGLE_READINESS_20261016.md).

## Environment and maintained path

Run from this exact source checkout with `python -m olympus.training.campaign`.
It calls the existing `olympus.training.olympus_system.train_olympus_system`.
Do not use the historical controlled/reliability scripts' `all` or `confirm`
commands for this engineering campaign; their negative results stay frozen.

The laptop environment is unchanged: Python 3.14.7, Torch distribution 2.14.1,
NumPy 2.5.4 and Pydantic 2.14.0 were inspected through package metadata. Its
resource gate prevented Torch import, so build/device runtime was not verified.
Normal application packaging still requires Python 3.14 and retains the existing
`pyproject.toml` and `uv.lock`. The source-only campaign has Python 3.11-compatible
syntax and a candidate dependency envelope in `requirements-kaggle.txt`.
Cross-version numerical compatibility is unverified.

On Kaggle preserve the image's CUDA-enabled Torch. Inspect first; do not apply the
laptop's full dependency upgrade. If dependencies are missing, create an isolated
environment using `--system-site-packages`, install only missing non-Torch packages,
and run `python -m pip check`. The minimal requirements deliberately omit Torch.
No dependency installation was performed in this task.

## CPU smoke and preflight

Commands run at the repository root. Locally use `.venv/bin/python` instead of
`python`. Prefix shell commands with RTK where installed; on Kaggle remove the
`rtk`/`rtk proxy` prefixes. Choose new working directories.

```sh
rtk mkdir -p artifacts/kaggle-smoke-cpu
rtk cp configs/kaggle/smoke-cpu.json artifacts/kaggle-smoke-cpu/campaign.json
rtk proxy python -m olympus.training.campaign campaign --manifest artifacts/kaggle-smoke-cpu/campaign.json --run-root artifacts/kaggle-smoke-cpu/runs --dry-run
rtk proxy python -m olympus.training.campaign preflight --run-root artifacts/kaggle-smoke-cpu/runs --device cpu
rtk proxy python -m scripts.validate_olympus_campaign --device cpu
rtk proxy python -m olympus.training.campaign prepare-smoke --run-root artifacts/kaggle-smoke-cpu/runs --output artifacts/kaggle-smoke-cpu/data
rtk proxy python -m olympus.training.campaign run --manifest artifacts/kaggle-smoke-cpu/campaign.json --run-root artifacts/kaggle-smoke-cpu/runs --cell pilot
```

Dry-run enumerates every cell before data exists and imports no Torch. Preflight
and numerical validation acquire the unchanged 2 GiB available-memory / 90% swap
governor before Torch import. Do not weaken a rejected gate. Core execution needs
no network/account. Failures return nonzero; intentional interruption returns 3.

The smoke uses 8 training / 4 validation records, data seeds 2026101601/2026101602,
training seed 2026101603, eight steps, batch four, FP32, one CPU thread and
validation at steps four/eight. It tests execution plumbing, not model efficacy.

## Smallest CUDA check

After explicit authorization to use the chosen Kaggle GPU, copy
`configs/kaggle/smoke-cuda.json` to a new working directory's `campaign.json`.
Repeat the commands above with the new paths and `--device cuda`. CUDA must resolve
or fail; CPU fallback is forbidden. The guarded command
`python -m scripts.validate_olympus_campaign --device cuda` runs target/device tests
and a four-step interrupted-versus-uninterrupted fit before the eight-step smoke.
It fails before test collection if CUDA is unavailable. CPU-suite CUDA skips are
never GPU passes. No CUDA test was executed during this engineering task.

## Interruption, checkpoint resume and frozen evaluation

Exercise the real controller using a fresh run root and prepared smoke data:

```sh
rtk proxy python -m olympus.training.campaign run --manifest artifacts/kaggle-smoke-cpu/campaign.json --run-root artifacts/kaggle-smoke-cpu/interruption-runs --cell pilot --stop-after-steps 4
rtk proxy python -m olympus.training.campaign run --manifest artifacts/kaggle-smoke-cpu/campaign.json --run-root artifacts/kaggle-smoke-cpu/interruption-runs --cell pilot
```

The first command intentionally exits 3; the second resumes automatically. Do not
change the total step budget: warmup/cosine scheduling stays bound to the original
configuration. Changed source/config/data requires a new run root. Completed cells
are reused only after validating identities and required artifacts.

Schema-2 checkpoints store parameters/buffers, AdamW state, functional scheduler
definition, counters, best-validation criterion, Python/NumPy/Torch CPU/CUDA RNG
and the independent CPU subset sampler RNG. FP32's absent scaler is explicit and
validated. Kronos's GRU starts independently for each trajectory batch; it has no
cross-call memory to restore. This is not an FIM implementation.

An fsynced immutable checkpoint precedes atomic latest/best pointers. Load verifies
SHA-256, source/config/data/runtime identity, schema and counters, then uses
`weights_only=True` without unsafe fallback globals. Copy the run directory and
original source together. Relocation is supported; a different runtime/device is
not an exact-resume guarantee.

Saves occur at optimizer boundaries, validation, a configurable interval (default
120 seconds), and cooperative stops. SIGTERM requests a safe boundary. A hard kill
or KeyboardInterrupt may lose work since the last durable checkpoint; saved
RNG/optimizer state replays it. Mid-kernel/backward continuation is unsupported.
The CPU regression checks dropout, all optimizer/RNG/model tensors and metrics at
zero tolerance; CUDA checks model tensors exactly and metrics within 1e-6. These
numerical tests are staged but unexecuted locally because of resource rejection.

`checkpoint.pt` exports the validation-selected best model, separate from latest
training state. Step zero is eligible. No protected test is consulted. Actual
frozen-checkpoint evaluation, analysis and export:

```sh
rtk proxy python -m olympus.training.campaign evaluate --manifest artifacts/kaggle-smoke-cpu/campaign.json --run-root artifacts/kaggle-smoke-cpu/runs --cell pilot --checkpoint best
rtk proxy python -m olympus.training.campaign analyze --run-root artifacts/kaggle-smoke-cpu/runs
rtk proxy python -m olympus.training.campaign export --run-root artifacts/kaggle-smoke-cpu/runs --output artifacts/kaggle-smoke-cpu/completed-artifacts.tar.gz
```

Evaluation runs forward passes on the declared validation data. Analysis retains
incomplete cells. Export includes only verified allowlisted completed artifacts,
not arbitrary workspace files, data or source. For interrupted sessions retain
the entire run directory, its checkpoint pointers, original source and separate data.

## Finite development matrix and quota budget

Executed successfully on Kaggle on 2026-10-16:
[version 359060464](https://www.kaggle.com/code/ryangomez010/olympus-full-development-matrix-2026-10-16?scriptVersionId=359060464)
completed all six fits at 300 steps each (1,800 development steps) in a
206.8-second notebook run with GPU T4 x2 selected. The preceding three CUDA
regressions and eight-step smoke also passed. No execution repair was needed.
The downloaded recovery archive was checksum-verified and restored; the campaign
analyzer verified all six completion records and artifact digests. See the
[verified development receipt](../artifacts/kaggle-development-20261016/VERIFIED_DEVELOPMENT_RESULT.json).
This remains synthetic development evidence, with promotion unauthorized.

After smoke/resume verification, create a new directory, copy
`configs/kaggle/development-cuda.json` to `campaign.json`, run
`prepare-development --run-root <directory>/runs --output <directory>/data`, then
`campaign --manifest <directory>/campaign.json --run-root <directory>/runs --dry-run`
and the same campaign command without `--dry-run`.

The matrix is learned/uniform routing × seeds 2026101621–2026101623: six fits,
300 steps each, batch 32, learning rate 0.0005, 20 warmup steps, FP32, evaluation
every 50 steps and checkpoints every 120 seconds. Fresh generated data has
512 training / 128 validation records, seeds 2026101611/2026101612, no protected
split. This is a development reproducibility/throughput matrix, not a revival of
the rejected v2/v3 scientific hypotheses. There is no new architectural hypothesis.

Both variants retain and execute the router; uniform routing zeroes its influence.
They use equal information, loss weights, selection rules and optimizer steps.
Actual parameter counts and timing are logged. Equal measured runtime or component
benefit has not been established.

The default wall budget is 14,400 seconds, with a schema ceiling of 21,600 seconds.
The controller persists consumption across restarts, conservatively charges
unreconciled crash intervals and reserves 60 seconds to stop. The first unmeasured
cell is a pilot; later estimates use observed time with 25% headroom and may prevent
admission. Budgets cooperate at safe boundaries: a kernel, evaluation or checkpoint
can exceed the reserve. Notebook setup/idle time also consumes Kaggle quota, so this
is not a quota-service meter or hard GPU cutoff. Stop the notebook when work ends.

Compare CPU/CUDA smoke timings before expanding: these tiny models may be more
economical on CPU. No GPU speedup has been measured. AMP, compilation, multi-GPU,
activation checkpointing and gradient accumulation remain unsupported/off.
Naive accumulation changes the nonlinear mean-gate regularizer, so it is not
advertised as equivalent. Any precision/memory variant needs parity tests first.

## Live metrics

Flushed timestamped console events mirror durable `metrics.jsonl`: seed, phase,
step, learning rate, loss, validation, throughput, elapsed time, process RSS, CUDA
allocation, checkpoints and failure status. `campaign_state.json` tracks all cells.
ETA appears after five observations and covers training only, excluding future
validation/checkpoint overhead. Replayed steps remain visible in resumed logs.

Set `tensorboard: true` in a new manifest to log real loop metrics if TensorBoard is
installed, and point its UI at the run's `tensorboard/` directory. Optional tracking
failure is logged without aborting training. No W&B credentials or external tracker
are required or enabled. No curves/results were fabricated for this handoff.
