# Contributing

Use Python 3.14 and Node.js 22. Install the Python and web dependencies from
their locked or declared manifests:

```bash
export SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
uv sync --locked --all-extras
cd apps/forge-web
npm ci
```

Before submitting a change, run the same checks enforced by CI:

```bash
uv run --locked python -m ruff check .
uv run --locked python -m mypy olympus scripts tests
uv run --locked python -m pytest -q
uv run --locked python -m build --no-isolation
cd apps/forge-web
npm test
npm run build
```

Changes must include tests for altered behavior. Generated pass states,
fabricated metrics, hidden credentials, real customer data, and claims not
supported by stored evidence are not accepted.
