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
.venv/bin/python -m olympus.cli foundry prepare-dataset \
  --output artifacts/release-deep-dataset
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

The proprietary license, author name, canonical public GitHub repository,
dependency audits, CI gate, SBOM generation, artifact attestations, and release
workflow are implemented. The source repository exists at the canonical URL.
PyPI publication still depends on its owner-controlled Trusted Publisher.

The repository owner must complete these account-controlled actions:

1. Configure GitHub private vulnerability reporting and require the
   `release-gate` workflow on the `main` branch.
2. Configure a PyPI Trusted Publisher for the `publish-release` workflow,
   environment `pypi`, and package `olympus-cognitive-architecture`.
3. Merge the verified pull request and create and push the signed or annotated
   `v0.2.0` tag.
4. Confirm the tag-triggered workflow publishes the wheel, source archive,
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
