# Olympus finish checklist

Current execution update: see [the September 8 audit and checklist](PROJECT_MEGA_AUDIT_CHECKLIST_2026-09-08.md). The counts below describe an earlier baseline, not final acceptance. The packaging-test mismatch and strict-type errors have been repaired. The latest execution includes validated promotion metrics, protected trust-policy pinning, durable jobs, ingestion hardening and versioned training-objective corrections. Unchecked work is not implicitly complete.

Last updated: 2026-09-08

This is the operating document for the current finish pass. A checked item has
been validated in this checkout; it does not imply scientific validity beyond
the stated boundary. Olympus v0.2.0 is complete only when every P0/P1 item is
checked or explicitly recorded as an external blocker.

## P0 — release, correctness, and safety blockers

- [x] Run the complete Python test suite with branch coverage (231 passed,
  85.21% branch coverage before this finish pass).
- [x] Run maintained-source Ruff and strict mypy.
- [x] Build the web application and run its unit tests.
- [x] Remove the verified web test-runner vulnerabilities and re-run the web
  test, production-build, and dependency-audit lane (16 tests; zero known npm
  vulnerabilities on 2026-09-09).
- [x] Build wheel/sdist, validate metadata, and test the wheel outside the
  source checkout.
- [x] Keep distribution validation fail-closed when built Python sources are
  untracked.
- [ ] Re-run every root, web, Pantheon, packaging, and dependency gate after
  integrating this finish pass.
- [ ] Review the final diff for accidental files, generated scratch, secrets,
  and claim drift.
- [ ] Obtain a clean reviewed Git commit containing every intended package
  source and no unrelated/generated files. **External blocker:** the current
  checkout contains more than 120 pre-existing dirty paths; this pass will not
  stage or commit them without maintainer review.
- [ ] Make `validate_distribution_payload.py` pass from a clean checkout.
  **Blocked by the clean-commit item; the validator must not be weakened.**
- [x] Replace the Foundry's permanent independent-evidence blocker with a
  versioned, hash-bound attestation verifier that still rejects caller-authored
  booleans and untrusted reports.
- [ ] Add regression tests for forged, mismatched, replayed, and incomplete
  promotion attestations.
- [ ] Ensure a structurally valid synthetic attestation can be verified without
  authorizing a scientific/model-quality promotion.
- [x] Reconcile Foundry status/README/model ledger with the implemented
  attestation boundary.
- [ ] Verify API authentication/authorization, loopback defaults, error
  sanitization, input size/path handling, concurrency, and cancellation; fix
  any concrete defect found.
- [ ] Verify the browser console has no dead primary actions and handles live,
  loading, empty, failure, cancellation, and confirmation states.
- [ ] Verify keyboard focus, form labels, status announcements, contrast, and
  narrow/mobile layout for the primary console flow.

## P1 — complete the supported alpha experience

- [x] Verify a fresh Foundry dataset → experiment → checkpoint → held-out
  evaluation → export → registry lifecycle in a temporary root.
- [x] Verify the fixed six-role reference replay and its fail-closed boundary.
- [x] Verify Pantheon's structural/scientific gate separation and retain its
  capability-floor negative result.
- [ ] Trace the CLI/API/web Foundry journeys against the actual implementation
  and remove any contradiction or premature terminal state.
- [ ] Add missing regression coverage for the highest-risk user/API flows found
  during re-audit.
- [ ] Verify environment defaults are safe and `.env.example` lists only real,
  supported settings.
- [ ] Verify a fresh package install can locate package data and execute the CLI
  outside the checkout.
- [ ] Verify release and CI workflows use pinned actions/tools, locked
  dependencies, clean-source checks, payload checks, and isolated-wheel checks.
- [ ] Ensure generated checksums/SBOMs remain outside the PyPI distribution
  upload set.
- [x] Verify cancellation, retry, duplicate submission, and process-reopen
  behavior for Foundry jobs.
- [ ] Verify no sensitive evidence, token, or internal exception is returned in
  public API errors or recorded in routine logs.
- [ ] Ensure README, `PROJECT_STATUS.md`, `TRUTH_MAP.md`, `RELEASE.md`, and the
  model ledger describe the same current state.
- [ ] Make every named family remain explicitly
  `EXPERIMENTAL_SMOKE_NOT_PROMOTED` until a qualifying checkpoint exists.
- [ ] Confirm “perfect,” “sentient,” “AGI,” and exact frontier parameter-count
  claims do not appear as factual current capabilities.

## P2 — quality, maintainability, and research integrity

- [x] Correct DRPT's silent float32 trajectory overflow and cover it.
- [x] Correct FIM/Fabric/QFIM stochastic low-salience writes, preserve novelty
  and write-gate semantics, and add regression tests.
- [x] Correct QFIM image-batch/sequence classification and cover it.
- [x] Correct CDW's frozen-snapshot lint boundary.
- [x] Reconcile Causal-Memory-Use truth documents with the completed local-model
  evidence and failed naturalistic negative control.
- [x] Produce a portfolio-level claim/source ledger, status registry, and
  adversarial report.
- [ ] Re-scan maintained code for TODO/FIXME/HACK, swallowed exceptions,
  unbounded input, misleading names, and dead scaffolds; resolve concrete
  defects rather than style-only churn.
- [x] Re-run dependency vulnerability audits and distinguish network/tooling
  failures from actual vulnerabilities (no known Python or npm vulnerabilities
  reported on 2026-09-09).
- [ ] Verify generated evidence and paper artifacts against their source
  manifests after the final code changes.
- [ ] Document all checks that are NOT RUN or BLOCKED; never turn absence of
  evidence into PASS.
- [ ] Preserve all negative/null findings and frozen protocol decisions.

## P3 — polish and developer experience

- [ ] Remove obsolete or contradictory setup instructions.
- [ ] Verify copy/paste setup, test, build, Foundry, web, and Pantheon commands.
- [ ] Ensure buttons, headings, labels, and status copy use consistent product
  terminology and do not imply nonexistent models.
- [ ] Remove avoidable generic dashboard clutter while preserving Olympus's
  current visual identity.
- [ ] Confirm repository-generated temporary files are ignored and remove only
  scratch produced by this pass.
- [ ] Record exact changed files, commands, outcomes, and remaining blockers in
  the final report.

## External model/research blockers — not executable in this checkout

- [ ] Select and legally approve a revision-pinned Hermes base model.
- [ ] Build 10,000 reviewed training records and 1,200 untouched test records
  with at least 100 records per declared category.
- [ ] Run the frozen multi-seed Hermes comparison on admitted hardware.
- [ ] Complete independent evaluation, quantization, license, safety, and
  fresh-process serving attestations for an exact checkpoint.
- [ ] Conduct independent security review before any non-loopback hosted use.
- [ ] Configure branch protection and required checks in GitHub.
- [ ] Configure the owner-controlled PyPI Trusted Publisher.
- [ ] Merge a reviewed clean commit and create the owner-authorized release tag.
- [ ] Obtain independent research replication and human adjudication before
  making comparative scientific claims.

## Final release acceptance

- [ ] BUILD: pass from the clean release commit.
- [ ] TESTS: pass with required branch coverage.
- [ ] LINT: pass without exclusions that hide maintained source.
- [ ] TYPECHECK: strict pass.
- [ ] WEB: unit tests, accessibility-focused tests, and production build pass.
- [ ] SECURITY: dependency audits and focused security regressions pass.
- [ ] PACKAGE: metadata, exact payload, isolated install, SBOM, and checksums
  pass from the same commit.
- [ ] PANTHEON: integrity, structural, scientific-status, and PDF gates pass
  while preserving the negative capability verdict.
- [ ] DOCUMENTATION: matches the verified runtime and names every limitation.
- [ ] SOURCE: clean, reviewed, tagged, and reproducible.
- [ ] Verdict can be upgraded from `NOT READY — SPECIFIC BLOCKERS REMAIN` only
  when every non-external P0/P1 item is checked.
