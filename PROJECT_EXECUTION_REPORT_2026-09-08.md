# Olympus execution evidence — 2026-09-08

Decision: **NOT READY for general public model release or comparative model publication.**
This is an implementation receipt, not a claim that all 108 checklist items are complete.
Existing unrelated changes were preserved; nothing was staged, committed or deployed.

## Completed implementation

- Promotion reports now reject contradictory regression/quality decisions, invalid
  loss/perplexity pairs, missing or duplicate category/workflow coverage and mismatched
  manifest counts. Report schemas are versioned.
- Evidence reports, cards and signatures use bounded regular-file snapshots; their
  recorded hashes refer to the exact parsed bytes. Promotion artifact hashes stream.
  Trust-policy SHA-256 must match operator configuration; self-supplied keys alone
  cannot qualify. Release records use unique run directories and preserve history.
- Invalid CLI evidence gets a separate attempt record and exit 2. Valid unsuccessful
  qualification exits 1; successful qualification exits 0. These are protocol tests,
  not independent validation of a real model candidate.
- Foundry verification now shares root-scoped resource admission across foreground
  and background entrypoints. Admission conflicts produce sanitized HTTP 409 errors.
- Jobs persist in SQLite, retain history, use cross-process kernel ownership leases,
  support cross-worker cancellation and reconcile interrupted workers on startup.
  The recovery test kills a real subprocess and verifies identity retention.
  Authenticated history is available at `/foundry/jobs/history` with bounded pagination.
- HTTPS ingestion pins a validated public address, retains TLS hostname verification
  and disables ambient proxies. Shard writes use descriptor-anchored traversal,
  no-follow directory opens, private temporary files, fsync and atomic replacement.
- Fixed supervised-token loss aggregation, incomplete gradient-accumulation scaling,
  and missing dropout RNG restoration on resume. V2 report/objective identities
  prevent silent reuse of historical v1 measurements or training identities.
  CPU dropout resume is bitwise equal to uninterrupted training in the test fixture;
  accumulated gradients match equivalent full batches within stated float tolerance.
- Package imports no longer eagerly initialize the API/Torch stack for lightweight
  workers. The public `olympus.app` export remains compatible.
- Fixed the packaging workflow-test mismatch, strict typing errors, and the release
  validator's incorrect setuptools metadata-directory expectation. Unknown metadata
  and untracked source payloads remain rejected.
- Updated README, status/truth documents, Foundry/API documentation, research ledgers
  and the two checklists to reflect these boundaries without inventing model results.

## Final verification

| Check | Result |
|---|---|
| `.venv/bin/ruff check .` | PASS |
| `.venv/bin/mypy olympus tests` | PASS — 103 source files |
| `.venv/bin/pytest -q --cov=olympus --cov-report=term --tb=short` | PASS — 277 tests; combined statement/branch coverage 85.26%, unchanged 85% requirement |
| `npm test` in `apps/forge-web` | PASS — 16 tests |
| `npm run build` in `apps/forge-web` | PASS — TypeScript build and Vite production assets |
| Locked Python dependency audit | PASS — no known vulnerabilities; public advisory access required sandbox escalation |
| `npm audit --audit-level=moderate` | PASS — zero reported vulnerabilities; public advisory access required sandbox escalation |
| Checksum-pinned Tectonic bootstrap | PASS — macOS arm64, 0.17.0; Linux clean-runner bootstrap not verified here |
| `build_release_artifacts.py --verify-reproducible` | PASS — two matching wheel/sdist builds |
| `twine check` on final distributions | PASS |
| `validate_distribution_payload.py` | FAIL CLOSED — untracked package/test/script/evidence files; generated setuptools metadata false positive is fixed |
| Final installed-wheel check outside checkout | PASS — integrity `ok`, one complete Foundry lifecycle, six reference families and 15 composition checks |
| `git diff --check` | PASS |
| Live browser-to-real-API journey, hosted deployment, CUDA/MPS parity | NOT RUN |
| Frozen Pantheon studies / sibling repository experiment suites | NOT RUN in this execution; prior results not overwritten |

Final distribution directory: `/private/tmp/olympus-release-5khIvB/final` (temporary,
not a published release). Wheel SHA-256:
`6db7ffb8c1877629a82bff66bdd881f7740314c0e293ac5372151a37e76aecb8`.
Source distribution SHA-256:
`b28bbaeb9a673aed92a95ed9c06af26d108976dfbeeb301be5c983eb2c6cee87`.

An intermediate gradient test exposed Adam's amplification of floating-point noise
near zero. The objective is tested directly through captured gradients rather than
claiming bitwise optimizer parity between different batch shapes. A killed-worker
test failed during eager whole-package startup; the lazy-import fix removes that
dependency and the final complete suite passes without increasing its deadline.

## Remaining blockers and unfinished scope

- **Release source inventory:** the checkout still includes substantial pre-existing
  dirty/untracked work. A maintainer must review the intended files and produce a clean
  release revision, then rerun distribution validation. No validator was weakened to
  accept these files and no unrelated work was staged.
- **Independent model evidence:** operator-controlled evaluation/release runners,
  separate signer credentials, reviewed trust policy, licensed training data and
  qualifying checkpoint/serving evidence are still absent. Provision these through
  reviewed release infrastructure; set `OLYMPUS_PROMOTION_TRUST_SHA256` there, not in
  candidate-controlled training configuration. There are no newly qualified model families.
- **Unfinished local engineering, not external excuses:** process-isolated hard
  deadlines, downstream release re-verification/replay policy, fully bound base/run
  ancestry, latest-summary transaction semantics, history UI, browser integration/
  accessibility verification and remaining model/research checklist items are open.
  These were not marked complete merely because surrounding infrastructure passed.
- **Compatibility:** regenerate v2 evaluation/quantization evidence and signatures.
  Historical artifacts remain unchanged. Old training checkpoints may be evaluated
  but may not silently resume under the corrected v2 objective.

Eight complete checklist items are explicitly checked; partial items are recorded
at the top of `PROJECT_MEGA_AUDIT_CHECKLIST_2026-09-08.md`. Passing software checks
does not establish AGI, sentience, trained family capability, public-service security,
or conference readiness.
