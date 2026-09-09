# Focused re-audit and extension — 2026-09-08

Scope: current Foundry worker, job lifecycle and installed-distribution verification,
using the previous execution report and current dirty checkout. This is not a fresh
certification of every model, sibling repository or deployment.

## Findings and implemented extension

- The release validator exercised only foreground Foundry execution. The new child
  process path had unit/integration tests and a one-off installed-wheel check, but
  no enduring installed-distribution gate. Added child execution to
  `scripts/validate_installed_wheel.py`, which existing CI/release workflows invoke.
  It requires imports from the installed target, evaluation success, all persisted
  record counts, SQLite integrity and cleanup of temporary worker directories.
- Release-verifier subprocess calls lacked deadlines. Added a 120-second timeout
  per command; timeout and nonzero exit propagate as failures. Tests exercise a
  real sleeping subprocess and a failing interpreter, plus successful output.
  This timeout is not a general arbitrary-descendant process sandbox.

The extended validator passes against the previously built final wheel, outside
the source checkout. Runtime package sources were not changed in this pass; the
validator and regression tests changed. No new distribution build or deployment
is claimed. Existing unrelated changes and frozen research were preserved.

## Ranked extensions still to implement

Fresh verification: repository Ruff PASS; strict mypy PASS (105 files); full Python
suite PASS (297 tests, 33.57 seconds, 85.40% combined coverage, unchanged 85% gate).
The final extended installed-wheel validator also passed with command deadlines
enabled. Frontend checks and a new package build were not rerun in this scoped pass.

1. Multi-process crash matrix (`jobs.py`, `store.py`, integration tests): terminate
   supervisors at admission, cancellation and finalization; prove recovery and
   absence of duplicate effects, not merely successful normal execution.
2. Evidence-consumer replay/revocation (`attestation.py`, promotion consumers):
   distinguish historical inspection from authorization of a new deployment;
   reject revoked/expired authority for new releases.
3. Durable job-history UI (Forge web/API): expose persisted failures, cancellations
   and explicit retries without confusing old successful output with a new run.

Public model/publication status remains **NOT READY**. Neither package smoke tests
nor this extension supplies trained qualifying weights or independent capability
evidence. Untracked release inputs still need review and a clean release revision.
