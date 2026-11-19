# Exact SFT continuation and retained attempts — development record

This is an engineering repair to the actual Foundry trainer. It makes epoch
continuation inspectable and reproducible on the verified CPU runtime. It does
not establish a model capability result or promote any Olympus model family.
The historical zero exact-task/tool-completion and `NOT_PROMOTED` findings in
the dated ledgers remain unchanged.

## Failure and mechanism

The earlier checkpoint saved minibatch-generator state but omitted the global
random state consumed by dropout. Restarting with the same seed therefore
replayed a different stochastic trajectory. Resume also admitted changes to
training settings, and the deterministic run directory could overwrite a
completed parent or a prior failed attempt.

Schema 3 checkpoints retain Python, Torch CPU, and applicable accelerator RNG
state, the resolved device type and Torch version. Resume validates the exact
non-epoch training configuration, dataset identity, model architecture, epoch
and optimizer counters, and complete AdamW parameter-group policy before
continuation. Checkpoint identity is the SHA-256 of the one byte snapshot that
was actually deserialized. The caller must request additional epochs.

The same runtime and device class are required for the recorded stochastic
state. Historical checkpoints remain loadable. A historical checkpoint without
the new RNG record can resume only with zero dropout, where the existing saved
minibatch generator covers this trainer's stochastic behavior. This is an
explicit compatibility boundary, not a claim that unavailable historical RNG
state can be reconstructed.

Each invocation owns a fresh attempt directory. A resumed invocation links its
parent hash and records only its additional epochs. Catchable training failures
retain `FAILED` receipts and any existing metrics; successful attempts receive
`COMPLETED` only after finite losses, gradients, updated model/optimizer state,
and final metrics pass. Invalid resume admission fails before an attempt is
created. Earlier checkpoint, log and failed-attempt bytes are preserved.

## Independent review changed the implementation

The root reviewer reproduced a NaN checkpoint returning a completed NaN result
and a fractional completed-epoch counter being silently truncated. These led to
finite-state/result checks and exact nonnegative counter admission, including
consistency with the saved epoch schedule and AdamW steps.

A second reviewer changed only the saved optimizer learning rate to zero. The
old candidate claimed the configured learning rate, advanced optimizer steps,
and completed with every weight unchanged. Saved parameter groups now must
match the freshly declared AdamW policy and parameter order, including fixed
optimizer options. They are rejected before `load_state_dict`; settings are
never silently replaced to conceal the mismatch. Before/after receipts and
source snapshots are retained in the verification directory.

## Current source basis and verification

The proposal targets current PR 27 at
`80fad05af90fb50a9b6dfb1d7634d5a7b0f92f35`. That parent already integrates PR 26.
Its bounded optimizer-window loop and quantization dataset binding are retained.
The final trainer source SHA-256 is
`6cd3d7b33002f6c4ff69137e42f4985bbeb6c9d907e37acab470b5089c643d27`.

The final local command passes **123 tests**, including 56 continuation/admission
cases and the current dataset snapshot, token-normalization, bounded-window,
quantization-binding and deep Foundry tests. Full, LoRA and QLoRA artificial
dropout trajectories compare uninterrupted and resumed model/optimizer tensors,
generator state, validation loss and metrics exactly. Independent root checks
use a different seed, batching, accumulation and legacy objective, and repeat
bitwise continuation plus five corrupted-parent refusals on the final source.

This environment lacks the eager API imports' `httpx`, `fastapi`, `rich`,
`typer` and `uvicorn` dependencies. An automatic approval review rejected a
local dependency installation. No dependency was installed; further installation
was stopped. The approved module harness supplies only the `olympus` package
namespace so the real Foundry and core modules load. No model or algorithm is
substituted. Ordinary package/API/CLI integration is **not locally verified**.
The normal existing hosted workflow is the separate full integration gate.
Full local Ruff passes; scoped MyPy passes for the two changed implementation
and new-test files. These do not imply a full local package type check.

CUDA and MPS RNG paths are implemented but unexercised locally. No cross-device,
cross-version or scientific efficacy claim is made. The local Torch runtime is
2.14.1+cpu on Python 3.12.14.

## Retained execution history

The first 28-case regression set produced 27 failures and one pass on the exact
integrated predecessor. Later verification includes repaired runs of 28, 88,
102 and 116 cases, followed by the final 123-case current-parent run. These
counts describe successive overlapping suites; they must not be added together.

All material failed attempts remain under
`research/verification/sft_continuation_20261010/`: the original import error,
an initial baseline-path error, stale private-helper and clipping-spy fixtures,
the actual independent NaN/counter/optimizer findings, a first parent-copy path
mistake that ran no Olympus tests, and reviewer-script lint failures. Runnable
reviewer scripts received style-only cleanup, with historical text snapshots
retained. `ENGINEERING_SFT_CONTINUATION_20261010.json` indexes the evidence and
separates pre-fix failures from harness mistakes and final verification.

Reproduce the bounded local scope from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python research/verification/sft_continuation_20261010/run_module_tests.py -q \
  tests/test_sft_continuation.py tests/test_dataset_manifest_integrity.py \
  tests/test_dataset_consumer_snapshot.py tests/test_sft_token_normalization.py \
  tests/test_deep_foundry.py tests/test_sft_window_memory.py \
  tests/test_quantization_dataset_binding.py
```
