# Contributing

Use Python 3.14 and Node.js 22. Install the Python and web dependencies from
their locked or declared manifests:

```bash
python3.14 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cd apps/forge-web
npm ci
```

Before submitting a change, run the same checks enforced by CI:

```bash
.venv/bin/ruff check .
.venv/bin/mypy olympus tests
.venv/bin/pytest -q
.venv/bin/python -m build
cd apps/forge-web
npm test
npm run build
```

Changes must include tests for altered behavior. Generated pass states,
fabricated metrics, hidden credentials, real customer data, and claims not
supported by stored evidence are not accepted.
