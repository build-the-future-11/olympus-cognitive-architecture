# Release and Publication Gate

## Reproducible code gate

Run these commands from a clean checkout:

```bash
export SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
uv sync --locked --all-extras
uv run --locked python -m ruff check .
uv run --locked python -m mypy olympus scripts tests
uv run --locked python -m pytest --cov=olympus --cov-branch --cov-report=term-missing -q
uv export --locked --all-extras --no-emit-project --no-hashes \
  --format requirements-txt --output-file /tmp/olympus-locked-requirements.txt
uv run --locked python -m pip_audit --strict \
  -r /tmp/olympus-locked-requirements.txt
uv run --locked python -m olympus.cli demo run-all
uv run --locked python -m olympus.cli foundry verify-pipeline
uv run --locked python -m olympus.cli foundry status
uv run --locked python -m olympus.cli foundry prepare-dataset \
  --output artifacts/release-deep-dataset
uv run --locked python -m build --no-isolation
uv run --locked python -m twine check dist/*
tar -tzf dist/*.tar.gz | grep scripts/validate_editclevr_payload.py
! tar -tzf dist/*.tar.gz | grep -q datasets/external/
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

Account-state audit on 2026-10-11 confirmed that private vulnerability
reporting is enabled and `main` has strict required `python` and `web` checks,
admin enforcement, linear history, conversation resolution, and force-push and
deletion denial. The remaining account-controlled actions are:

1. Configure or independently verify the PyPI Trusted Publisher for the `publish-release` workflow,
   environment `pypi`, and package `olympus-cognitive-architecture`.
2. Merge the verified pull request and create and push the signed or annotated
   `v0.2.0` tag.
3. Confirm the tag-triggered workflow publishes the wheel, source archive,
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
