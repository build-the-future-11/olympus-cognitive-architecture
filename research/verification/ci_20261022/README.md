# Current-head release repair, 22 October 2026

## Scope and observed failure

This is an engineering correction based on commit
`a50cad27c66a788c83fe93db63b522559c9e15dd`, not a scientific rerun or a change to
any protected result. The exact-head [release run 39715873686](https://github.com/build-the-future-11/olympus-cognitive-architecture/actions/runs/39715873686)
failed in both jobs. Earlier passing runs refer to different source revisions.

The Python job stopped on ten formatting findings in the three newly submitted
multi-horizon notebooks: three I001 import-order findings and seven E501
findings on embedded frozen source/protocol strings. These differ from the old
129-error notebook failure recorded in issue 30.

The web job passed all seven tests, then its audit failed with a retired quick
endpoint HTTP 400. A fresh registry audit successfully identified two moderate
vulnerable package entries, `@vitest/mocker` and its direct parent `vitest`.

## Changes and evidence boundary

- Exactly three submitted notebook paths receive exemptions for **only E501
  and I001**. All other lint rules remain active, and the standalone scientific
  runners are still fully linted. The notebook bytes are unchanged.
- Both the PR CI and tag-release workflow now verify
  `frozen-artifacts.sha256` before lint. Its nine notebook, runner, and protocol
  hashes match the identities recorded before this repair in the study states.
- Vitest is pinned to patched version **5.0.3**. The lock updates its mocker,
  spy, diagnostic, and resolved string-manipulation dependencies. The required
  moderate-severity audit gate remains unchanged. No audit findings are ignored.
- Scientific code, protocols, original notebooks, results, checkpoints, and
  predecessor receipts are not modified by this repair.

## Security sources

Both maintainer advisories identify 5.0.3 as a fixed release:

- [GHSA-qxg6-89mg-rqv8: manual mock registration code injection](https://github.com/vitest-dev/vitest/security/advisories/GHSA-qxg6-89mg-rqv8).
- [GHSA-xp26-3454-mjgp: Windows redirect allowlist traversal](https://github.com/vitest-dev/vitest/security/advisories/GHSA-xp26-3454-mjgp).

The npm registry metadata for `vitest@5.0.3` binds
`@vitest/mocker` to 5.0.3. This is a targeted patch of the existing test tooling;
Vite remains locked at 7.3.7.

## Verification performed

Local web validation used Node 24.19.0 and npm 11.9.0. The audit was additionally
repeated with npm **10.9.8**, the exact npm version from the failing hosted job;
that repeat still used local Node 24.19.0. Hosted CI uses Node 22.23.2, so local
checks do not constitute a new hosted exact-source release result.

| Check | Observed result |
| --- | --- |
| `sha256sum --check research/verification/ci_20261022/frozen-artifacts.sha256` | All nine files match |
| Integrity negative control in temporary copied files | Appending one byte to the heavy notebook causes failure; the other eight files still pass |
| `.venv/bin/python -m ruff check .` | Full repository lint passes with Ruff 0.17.0 |
| `npm ci --ignore-scripts --no-audit` | Clean locked installation succeeds |
| `npm test` | Seven tests across two files pass on Vitest 5.0.3 |
| `npm audit --audit-level=moderate --json` | Zero vulnerabilities with npm 11.9.0 |
| Same audit through `npm exec --package=npm@10.9.8` | Zero vulnerabilities with the failing hosted job's npm version |
| `npm run build` | TypeScript and Vite production build pass |
| `git diff --check` | Pass |

The local installation disables dependency lifecycle scripts; the existing
hosted workflow retains its normal `npm ci`. No full Python suite or hosted CI
success is asserted by this receipt. The final integration must execute the
unchanged full release checks at the actual published source revision.

## Reproduce the affected gates

From the repository root:

```sh
sha256sum --check research/verification/ci_20261022/frozen-artifacts.sha256
uv run --locked python -m ruff check .
```

From `apps/forge-web`:

```sh
npm ci
npm test
npm audit --audit-level=moderate
npm run build
```

No new research job, model promotion, external submission, or GitHub mutation
was performed during this repair. The canonical research-state update and
final release decision are owned by the integration step.
