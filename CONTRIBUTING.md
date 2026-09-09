# Contributing

Use Python 3.14, Node.js 22, and the pinned `uv` version used by CI. Install
from the lockfiles in isolated project environments:

```bash
python3.14 -m pip install --user uv==0.12.5
uv sync --locked --all-extras
uv run python -m pip check
cd apps/forge-web
corepack npm ci
```

Do not create `.venv` with `--system-site-packages`. That makes local checks
depend on unrelated global packages and can hide an inconsistent environment.
If an existing environment has `include-system-site-packages = true` in
`.venv/pyvenv.cfg`, remove and recreate that ignored environment before relying
on its results.

Before submitting a change, run the same checks enforced by CI:

```bash
uv run ruff check .
uv run mypy olympus tests
uv run pytest --cov=olympus --cov-branch --cov-report=term-missing -q
uv run python scripts/audit_locked_dependencies.py
uv run python scripts/build_release_artifacts.py \
  --outdir dist --verify-reproducible
uv run python -m twine check dist/*
uv run python scripts/validate_distribution_payload.py dist/*.whl dist/*.tar.gz
uv run python scripts/validate_installed_wheel.py dist/*.whl
cd apps/forge-web
corepack npm test
corepack npm audit --audit-level=moderate
corepack npm run build
```

Changes must include tests for altered behavior. Generated pass states,
fabricated metrics, hidden credentials, real customer data, and claims not
supported by stored evidence are not accepted.

## Extending a model family

A new family surface or a material extension to an existing one must update the
implementation and its truth boundary together:

1. Add or change the family module under `olympus/models/` with strict typed
   inputs/outputs and fail-closed validation at every trust boundary.
2. Reuse the shared contracts in `olympus/models/substrate.py`; do not create a
   parallel workspace, identity, evidence, tool, or output-envelope schema
   without documenting why the shared contract cannot represent it.
3. Register public/importable surfaces in `olympus/models/__init__.py` and
   `olympus/models/registry.py`. Registry presence proves importability only; it
   is not an evaluation or promotion state.
4. Add focused success, rejection, mutation, ACL/authority, and serialization
   tests under `tests/`. Trainable code also needs a finite loss and a verified
   parameter update; stateful code needs restart/failure tests appropriate to
   the durability it claims.
5. If the family enters the fixed composition, update
   `olympus/models/composition.py`, its smoke fixture, trace/budget invariants,
   and adversarial tests. Preserve all of the current boundaries: retrieval and
   final requests remain explicit public fields bound to their authorized-view
   hashes; Hermes sees only authorized frozen-protocol evidence; runtime-HMAC
   pending/scope IDs never replace the private reservation token required for
   scoped Perseus prepare/commit/receipt/rollback; execution precedes Kronos and
   Hermes; failures and budget exhaustion produce closed degraded records that
   reach Aion STOP; and a model call is charged before every attempted
   invocation while cached work is not charged twice on same-process recovery.
   Never broaden executor authority or make an external effect solely to
   satisfy a smoke.
6. Regenerate the bounded synthetic evidence with the documented CLI, preserve
   source/config hashes, and label fixtures, checkpoints, qualification, and
   promotion literally. Do not commit ignored training artifacts as evidence.
7. Update `docs/model-families.md`, `docs/architecture.md`,
   `research/SOURCE_OF_TRUTH.md`, and `research/REALITY_LEDGER.md` in the same
   change. Explicitly separate implemented behavior, missing controls,
   hypotheses, negative results, and empirical claims.
8. A family name may be attached to an external checkpoint or served model only
   after the immutable dataset, evaluation, safety, licensing, serving, and
   human-review gates pass. A unit test, optimizer step, component smoke, or
   contract-composition replay is never sufficient.

Composition changes must continue to state what they do **not** provide. A tool
being declared non-material is only policy alignment, not proof of effect
isolation. The executor-profile hash is host-asserted. HMAC-derived identifiers
and scope tokens are process-local capabilities, not crash-safe persistence or
external authentication. Tests must cover those negative boundaries rather than
turning them into unsupported security claims.
