# Astra execution evidence — 2026-09-08

## Completed

This follow-up advances Olympus, not every sibling repository. It does not complete
the whole portfolio checklist or establish public model/publication readiness.

1. **Evaluation population integrity.** Training category repetition and packing
   could change validation, test and quantization populations. `SFTConfig.for_evaluation()`
   now disables both outside training. Tests confirm that excluding safety and
   repeating reasoning in the training mixture cannot change held-out results or
   initial validation loss. Invalid weights and incompatible tokenizer vocabulary
   sizes are rejected before model construction.
2. **Quantization provenance.** Dataset identity is checked against the checkpoint
   before output creation. Schema-v3 quantization evidence binds the manifest hash;
   promotion checks that binding. Evaluation and quantization reports identify
   the all-records-once, unpacked sampling policy. New training identities/checkpoints
   identify the validation policy, and incompatible resumes are rejected.
3. **Bounded background execution.** Built-in verification runs in a dedicated
   child process/session with its own SQLite connection, a finite deadline,
   cancellation with TERM/KILL and reaping, bounded result validation and a parent
   watchdog. The manager records a specific deadline failure. Isolated Python
   startup pins the child to the parent's package location, ignoring shadow
   packages in the working directory or PYTHONPATH. Default deadline: 300 seconds;
   constructor maximum: 3600 seconds. Trusted custom runners remain cooperative;
   this is not a security sandbox for arbitrary code.

## Important files

- `olympus/foundry/sft.py`: mixture/vocabulary bounds and evaluation/resume policy.
- `olympus/foundry/eval_suite.py`: population isolation and schema-v3 reports.
- `olympus/foundry/quantization.py`: dataset binding and input checks before writes.
- `olympus/foundry/promotion.py`: cross-report dataset identity gate.
- `olympus/foundry/worker.py`: new bounded process lifecycle.
- `olympus/foundry/jobs.py`: isolated built-in execution and deadline failures.
- `tests/test_deep_foundry.py`, `tests/test_foundry_worker.py`: regression evidence.
- `docs/foundry.md`, `docs/api.md`, `PROJECT_STATUS.md` and the existing mega
  checklist: compatibility, execution guarantees and scoped completion status.

## Verified

| Command/check | Result |
| --- | --- |
| `.venv/bin/ruff check .` | PASS |
| `.venv/bin/mypy olympus tests` | PASS, 105 source files |
| `.venv/bin/pytest -q --cov=olympus --cov-report=term --tb=short` | PASS, 295 tests, 40.75 seconds, 85.40% combined coverage; unchanged 85% gate |
| `npm test` in `apps/forge-web` | PASS, 16 tests |
| `npm run build` in `apps/forge-web` | PASS, TypeScript and Vite |
| `scripts/build_release_artifacts.py --outdir /private/tmp/olympus-astra-LxFVF5/final --verify-reproducible` | PASS, two byte-identical wheel/sdist builds |
| `python -m twine check` on both final artifacts | PASS |
| `scripts/validate_installed_wheel.py` on final wheel | PASS, six reference families, 15 deterministic composition checks, persisted Foundry lifecycle |
| Additional final-wheel installation into a temporary target, then `run_isolated_verification` outside checkout | PASS, child completes evaluation; reopened registry integrity is ok and contains one fixture model |
| `scripts/validate_distribution_payload.py` on final artifacts | FAIL CLOSED: untracked source payloads |
| Remote CI, deployment, GPU training, independent model evaluation | NOT RUN |

Final build artifacts, before this documentation-only evidence update:

- Wheel SHA-256: `5cdfaacd71c46c1ca83ca91a04ffb7cfa6ef280ab78443c2df4bebcd3d35fcc3`
- Source archive SHA-256: `fe345dcde43b46c0d0c9e86bf410ed77ce68082fb5f9509ff3a8a741753ef24e`

An earlier concurrent-edit coverage run was invalidated by source/line-map changes;
the final results above come from a stable source tree. No test, coverage threshold,
security check or tracked-payload requirement was weakened.

## Research integrity

These are correctness and lifecycle tests, not capability benchmarks. The installed
Foundry model is a character-bigram infrastructure fixture. No named family gained
a qualified trained checkpoint, and no comparative improvement is claimed.
Historical schema-v1/v2 artifacts were retained, not converted into v3 measurements.
Promotion using the new policy needs actual regenerated evidence and new attestations.
Frozen Pantheon results, protocols, manuscripts and sibling studies were not changed
in this follow-up. No AGI or sentience claim is established.

## Remaining and blockers

**NOT READY** for general public model release or comparative model publication.

- The working tree includes extensive existing changes and untracked sources.
  Release payload validation correctly rejects them. Review ownership and intended
  release contents, commit approved files, then rerun the full release gate from
  that clean revision. No unrelated changes were staged or committed here.
- Qualified model weights, governed training data and independent evaluation,
  quantization, serving and rights-review attestations remain absent. Operators must
  provision the approved data/compute and independently controlled signing runners;
  test keys and fixtures cannot substitute for those inputs.
- Local engineering work remains: full multi-worker/crash transition certification,
  job-history UI, downstream replay/revocation semantics and immutable large-artifact
  consumption. These are unfinished work, not external-blocker excuses.
- CUDA/MPS parity, packed-training isolation experiments, generalization comparisons
  and full external reproducibility were not established by this CPU pass.

## Next highest-impact actions

1. Freeze separate platform-alpha and model-release contracts in `README.md`,
   `docs/model-families.md` and the registry; resolve role drift explicitly.
2. Review and commit intended release inputs, then run CI-equivalent packaging and
   payload checks from a clean checkout. Preserve all unrelated user work.
3. Extend `jobs.py`, `store.py` and integration tests with multi-process fault
   injection at admission, cancellation, finalization and restart boundaries.
4. Implement downstream replay/revocation and artifact immutability guarantees in
   promotion/attestation consumers before accepting externally produced evidence.
5. Provision independent evidence runners and execute one approved, modest model
   protocol with retained predictions, baselines and failures before scaling families.
