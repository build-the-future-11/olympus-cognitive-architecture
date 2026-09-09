# Release and Publication Gate

## Reproducible code gate

Run these commands from a clean checkout:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install uv==0.12.5
uv sync --locked --all-extras
uv run ruff check .
uv run mypy olympus tests
uv run pytest --cov=olympus --cov-branch --cov-report=term-missing -q
uv run python scripts/audit_locked_dependencies.py
uv run python -m olympus.cli demo run-all
uv run python -m olympus.cli foundry verify-pipeline
uv run python -m olympus.cli foundry status
uv run python -m olympus.cli foundry prepare-dataset \
  --output artifacts/release-deep-dataset
uv run python scripts/build_release_artifacts.py \
  --outdir dist --verify-reproducible
uv run python -m twine check dist/*
uv run python scripts/validate_distribution_payload.py dist/*.whl dist/*.tar.gz
uv run python scripts/validate_installed_wheel.py dist/*.whl
cd apps/forge-web
corepack npm ci
corepack npm test
corepack npm audit --audit-level=moderate
corepack npm run build
cd ../..
cd OlympusC/05_pantheon
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install uv==0.12.5
uv sync --locked --all-extras
test "$(tectonic --version)" = "Tectonic 0.17.0"
make verify PYTHON=.venv/bin/python
```

A releasable revision must pass every root, web, and Pantheon command. Start
with an absent or empty `dist/` directory; the builder refuses to mix a new
release with stale artifacts. The
generated Python artifacts are `dist/*.whl` and `dist/*.tar.gz`; the generated
web assets are `apps/forge-web/dist/`. Tectonic 0.17.0 must be installed before
the Pantheon boundary is run.

The JSON files under `evidence/` and `artifacts/stage-execution/` are historical
run records unless they name an exact Git commit. They must never be treated as
attesting to whichever files happen to be in `dist/`. The tag workflow creates
and attests a fresh `release-metadata/SHA256SUMS` alongside every release; the
checksum file stays outside `dist/` so the PyPI publisher sees distributions only.

The release builder uses the locked build backend, binds timestamps to the Git
commit through `SOURCE_DATE_EPOCH`, normalizes source-archive metadata, and
requires two independent builds to have identical SHA-256 digests. The payload
validator rejects unsafe, non-canonical, duplicate, linked, or internal-outreach
archive members and compares the wheel's package Python files and the source
archive's package, test, example, and script Python files exactly with the
corresponding Git-tracked paths. The installed-wheel validator then installs the
wheel into an isolated target outside the source checkout, rejects a source-tree
import, runs model status and the fixed composition smoke, executes `foundry
verify-pipeline`, and reopens Foundry state in a fresh interpreter. Stage truth must agree with
`evidence/status.json`, and a failed promotion must not emit a release manifest.
The dependency audit exports the locked, non-development runtime graph with
hashes and disables pip resolution before querying vulnerability data.

## Source-publication gate

The proprietary license, author name, configured repository metadata,
dependency audits, CI gate, SBOM generation, artifact attestations, and release
workflow are implemented. The audit repair branch is anchored to upstream but
is not a release until its exact commit passes CI and human review. PyPI
publication still depends on owner-controlled branch protection and Trusted
Publisher setup.

The repository owner must complete these account-controlled actions:

1. Configure GitHub private vulnerability reporting and require the
   `release-gate` workflow on the `main` branch.
2. Configure a PyPI Trusted Publisher for the `publish-release` workflow,
   environment `pypi`, and package `olympus-cognitive-architecture`.
3. Merge the verified pull request and create and push an annotated `v0.2.0`
   tag (a signed annotated tag is preferred). The workflow rejects lightweight
   tags; signature verification remains an owner/repository-policy control.
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
