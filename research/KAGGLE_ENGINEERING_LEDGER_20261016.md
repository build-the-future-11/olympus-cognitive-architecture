# Olympus engineering change/evidence ledger — 2026-10-16

Status: **BLOCKED** for numerical/GPU validation. HEAD
`20a5687d041859e888bf21390a37483177177dd8`; dirty user tree preserved.

| Defect / requirement | Repair | Evidence and limitation |
|---|---|---|
| Admission after allocation | Gate first; Torch-free preflight/test launcher | Actual resource rejection before import; numerical path unexecuted |
| Final weights only | Immutable schema-2 AdamW/RNG/sampler snapshots, latest/best | 45 pure checkpoint controls pass; numerical serialization/resume tests staged |
| Existence confused with completion | Bound source/config/data/artifact validation, crash recovery | Controller tests pass; real fit not executed |
| Missing finite budget | Persistent wall accounting, reserve, measured estimates, dry-run | Simulated-time tests and six-cell dry-run pass |
| Mode leakage across inference | Restore each module's mode, even on errors | Numerical tests staged |
| Truncated float labels / wrong target device | Reject invalid classes; align target dtype/device | Numerical CPU/CUDA tests staged |
| Entire validation set on GPU | Batched forward, compact CPU predictions, global reductions | Uneven-batch metric test staged; no speedup measurement |
| Conditional risk coverage denominator | Additional population coverage; old omitted field remains null | Numerical test staged; historical outputs unchanged |
| Invalid tensor/source contracts | Dtypes/ranges/masks/disjoint IDs/artifact checks | Numerical regressions staged; provenance still requires review |
| Missing live evidence | Timestamped JSONL, status, optional fail-soft TensorBoard | Controller logs tested; no invented training metrics |
| CUDA wheel replacement risk | Separate minimal source-run environment, actual preflight | No installed dependency changes; Kaggle untested |

## Changed implementation

- `olympus/training/olympus_system.py`: repaired existing trainer and frozen evaluation.
- `olympus/training/checkpoints.py`: versioned training snapshots and validators.
- `olympus/training/provenance.py`: source/config digests and durable JSON.
- `olympus/training/campaign.py`: finite controller around the same trainer.
- `olympus/evaluation/olympus_system.py`: bounded global diagnostic aggregation.
- `olympus/models/olympus_system.py`: compact objective view and target correctness.
- `olympus/models/runtime.py`: guarded inference/modes and exclusive checkpoint save.
- `olympus/data/olympus_dataset.py`: stronger aligned data contracts.
- `scripts/validate_olympus_campaign.py`: guarded CPU/CUDA regression launcher.

## Tests and handoff files

- `tests/test_checkpoint_contract.py`, `tests/test_campaign_source_contract.py`,
  `tests/test_olympus_campaign.py`: 73 executed lightweight control tests.
- `tests/test_olympus_resume.py`, `tests/test_runtime_hardening.py`: numerical
  regressions staged, not executed under the rejected resource gate.
- `configs/kaggle/smoke-cpu.json`, `smoke-cuda.json`, `smoke.json` (CUDA alias),
  `development-cuda.json`: explicit finite plans.
- `requirements-kaggle.txt`, `docs/kaggle-campaign.md`, readiness report, this ledger
  and a pointer in `docs/olympus-system.md`.

## Evidence preservation

`artifacts/engineering/kaggle-readiness-20261016/initial-source.json` records the
original HEAD/dirty tree and hashes; `baseline/` preserves source bytes.
`local-preflight.json` retains the first resource rejection; `numerical-admission.json`
records the guarded numerical launch rejection at 93.3% swap before Torch import.
`verification.json` binds final
source and check commands/exits. Earlier failed static checks, where applicable,
remain in the execution history; only the explicit final receipt establishes a pass.

Historical scripts/protocols/outcomes and pre-existing user work remain. No failure
was relabeled positive. No model superiority, performance gain, independent
reproduction or GPU-readiness claim follows from control/static checks.
