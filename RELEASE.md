# Release and Publication Gate

## Reproducible code gate

Run these commands from a clean checkout:

```bash
python3.14 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/ruff check .
.venv/bin/mypy olympus tests
.venv/bin/pytest --cov=olympus --cov-branch --cov-report=term-missing -q
.venv/bin/pip-audit --strict .
.venv/bin/python -m olympus.cli demo run-all
.venv/bin/python -m olympus.cli foundry verify-pipeline
.venv/bin/python -m olympus.cli foundry status
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
cd apps/forge-web
npm ci
npm test
npm audit --omit=dev --audit-level=high
npm run build
```

A releasable revision must pass every command. The generated Python artifacts
are `dist/*.whl` and `dist/*.tar.gz`; the generated web assets are
`apps/forge-web/dist/`.

## Source-publication gate

The proprietary license, author name, canonical repository path, dependency
audits, CI gate, SBOM generation, artifact attestations, and release workflow
are implemented. PyPI and GitHub returned `404` for the intended package and
repository paths on 2026-07-23, so they appeared unregistered at verification
time; only successful publication reserves them.

The repository owner must complete these account-controlled actions:

1. Authenticate the `build-the-future-11` GitHub account and create the public
   `olympus-cognitive-architecture` repository at the canonical URL declared in
   `pyproject.toml`.
2. Configure GitHub private vulnerability reporting and require the
   `release-gate` workflow on the `main` branch.
3. Configure a PyPI Trusted Publisher for the `publish-release` workflow,
   environment `pypi`, and package `olympus-cognitive-architecture`.
4. Push the verified `main` revision and create and push the signed or annotated
   `v0.1.0` tag.
5. Confirm the tag-triggered workflow publishes the wheel, source archive,
   CycloneDX SBOMs, build attestations, GitHub release, and PyPI release.

## Scientific-publication gate

The current evidence supports a software alpha and deterministic engineering
demos. It does not support a peer-reviewed claim that Olympus outperforms other
cognitive architectures. Before making comparative scientific claims:

1. Pre-register the benchmark tasks, baselines, metrics, exclusion rules, and
   failure criteria based on the plans declared in `labos.project.yaml`.
2. Run the full benchmark and ablation matrix across multiple declared random
   seeds and preserve raw outputs, environment details, and checksums.
3. Report uncertainty, negative results, compute cost, and all deviations from
   the pre-registered protocol.
4. Record every dataset's provenance, license, consent boundary, checksum, and
   acquisition procedure; do not recategorize smoke tests as benchmark
   evidence.
5. Have every named author approve the manuscript, claims, license, conflicts
   of interest, and data/model-card disclosures before submission.
