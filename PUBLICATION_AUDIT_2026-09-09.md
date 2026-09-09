# Publication audit — 2026-09-09

## Verdict

**Software candidate: functionally verified in this checkout. Publication: not yet
ready because the intended release sources are not committed and the exact
distribution payload therefore fails its deliberate clean-source gate. Scientific
capability publication: not ready; Pantheon records a bounded negative capability
result.**

This verdict separates a working local alpha from a publishable, reviewed Git
revision. No failed or missing evidence is converted into a pass.

## What Olympus is

Olympus is a Python 3.14 research platform for cognitive-behavior experiments,
model-family smoke implementations, data ingestion, evaluation, a durable local
model Foundry, CLI/API access, and a React operator console. The repository also
contains Pantheon, a separately gated research artifact whose structural
completeness does not establish scientific capability.

## Verified working

- Root test suite: `316 passed`; required branch coverage met at `85.26%`.
- Maintained Python lint and strict typing: Ruff and mypy pass.
- Reference runtime: demos, Foundry lifecycle, state reopen, and dataset
  preparation pass.
- Web console: `16 passed`; TypeScript and Vite production build pass.
- Web dependency audit: zero known vulnerabilities after upgrading Vitest to
  5.0.0 and using its supported jest-dom integration.
- Python locked dependency audit: no known vulnerabilities found.
- Packaging: wheel and source archive build reproducibly; Twine metadata passes;
  an isolated wheel install passes CLI, model-composition, Foundry lifecycle, and
  fresh-process state checks.
- Pantheon: `31 passed, 2 skipped`; evidence validation and structural artifact
  gate pass. The authoritative scientific gate correctly remains negative.
- Repository hygiene checks: `git diff --check` passes and the focused credential
  signature scan found no matching private keys or common live-token formats.

## Defect repaired in this pass

The web release gate initially reported two moderate vulnerabilities through
Vitest's mocker dependency. The test runner was upgraded from the vulnerable 3.x
line to 5.0.0, its Testing Library matcher registration was migrated to
`@testing-library/jest-dom/vitest`, and the complete web test/build/audit lane was
rerun successfully.

## Blocking publication issues

### P0 — exact source revision is not reviewable or releasable yet

The checkout contains 156 changed paths, including 42 untracked paths. Several
untracked files are imported package modules, release scripts, and regression
tests. The fail-closed distribution validator therefore reports them as unexpected
wheel/source-archive content. This is correct behavior: a release must not publish
Python code that is absent from its Git revision.

Required owner/reviewer action: review the complete intended diff, add every
intended source and artifact (or remove it from the release), commit it, and rerun
the entire gate from that exact clean commit. The validator must not be bypassed or
weakened.

### P0 — account-controlled release controls

Branch protection, the required `release-gate`, the PyPI Trusted Publisher, final
maintainer approval, merge, and annotated release tag remain external controls.

### P0 — scientific claims

Pantheon's artifact is structurally complete, but both evaluated agents answered
zero of 20 questions and the capability floor fails. This supports a bounded
negative result only. It does not support comparative effectiveness, frontier
capability, AGI, or production-readiness claims.

### P1 — independent review

Non-loopback hosting still requires an independent security review. Model-family
promotion still requires revision-pinned training data/model provenance, qualifying
evaluation evidence, and independent signed attestations bound to the exact
checkpoint.

## Exact validation record

Passed:

```text
.venv/bin/ruff check .
.venv/bin/mypy olympus tests
.venv/bin/python -m pytest --cov=olympus --cov-branch --cov-report=term-missing -q
.venv/bin/python -m olympus.cli demo run-all
.venv/bin/python -m olympus.cli foundry verify-pipeline --root /tmp/olympus-final-foundry
.venv/bin/python -m olympus.cli foundry status --root /tmp/olympus-final-foundry
.venv/bin/python -m olympus.cli foundry prepare-dataset --output /tmp/olympus-final-dataset
.venv/bin/python scripts/audit_locked_dependencies.py
.venv/bin/python scripts/build_release_artifacts.py --outdir /tmp/olympus-final-dist --verify-reproducible
.venv/bin/python -m twine check /tmp/olympus-final-dist/*
.venv/bin/python scripts/validate_installed_wheel.py /tmp/olympus-final-dist/*.whl --temporary-parent /tmp
corepack npm test
corepack npm run build
corepack npm audit --audit-level=moderate
make verify PYTHON=.venv/bin/python
git diff --check
```

Expected failure:

```text
.venv/bin/python scripts/validate_distribution_payload.py /tmp/olympus-final-dist/*.whl /tmp/olympus-final-dist/*.tar.gz
```

Reason: intended distribution Python files are untracked in the current dirty
checkout. This check can only pass after source review and commit.

Not run and not claimed:

- GitHub-hosted CI and tag-release workflows.
- PyPI or GitHub publication.
- Independent security review, model evaluation, or scientific replication.
- Full browser-driven end-to-end testing against a live backend; component/API
  integration tests and the production build passed locally.

## Release sequence

1. Review all 156 changed paths and decide which 42 untracked paths belong in the
   release.
2. Commit the accepted source without generated scratch or secrets.
3. Run every command in `RELEASE.md` from the clean commit; require the payload
   validator to pass.
4. Obtain independent review and merge only after the required CI gate passes.
5. Configure owner-controlled GitHub/PyPI controls, then create the annotated
   `v0.2.0` tag from `main`.

