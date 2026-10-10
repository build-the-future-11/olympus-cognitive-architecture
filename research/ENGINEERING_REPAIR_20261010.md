# Checkpoint alphabet validation — narrowed repair, 10 October 2026

This revision retains only the independent character-alphabet validation from
PR #24. Duplicate symbols collapse entries in the character-to-index mapping;
empty or multi-character symbols violate the generated-character length
contract. The model constructor now requires unique strings of exactly one
Python character. Valid checkpoint bytes and model numerical behavior are
unchanged.

## Concurrent work and superseded scope

The original PR #24 revision (`080103654a2d5dfdbfcdb04e71b1a1bc73385f3a`)
combined dataset consumption, low-temperature sampling and alphabet checks.
A subsequent overlap audit identified dedicated concurrent PRs:

- [PR #23](https://github.com/build-the-future-11/olympus-cognitive-architecture/pull/23)
  supplies the complete dataset contract and additionally binds preparation to
  a single source-byte snapshot.
- [PR #25](https://github.com/build-the-future-11/olympus-cognitive-architecture/pull/25)
  supplies the sampling repair, including zero-probability and near-maximum
  floating-point cases beyond the original PR #24 implementation.

The duplicated dataset and sampling changes, their tests and the previous
ledger update were removed from PR #24's current diff. Their old 184-test result
belongs to the original commit, not this narrower tree. Those two features must
be attributed to PRs #23 and #25 instead of counted again here. Historical
research records remain unchanged.

## Shared dependency correction

This revision reuses the exact `source-map-js` 1.2.2 lockfile blob
`ee06243f97684b4be861530b9b218caae4ec9ab0` from canonical
[PR #17](https://github.com/build-the-future-11/olympus-cognitive-architecture/pull/17),
also inherited or reused by PRs #18–21, #23 and #25. Only that package's version,
resolved URL and integrity record differ from main. This is a credited existing
dependency repair, not an additional independently discovered improvement.

## Verification of the narrowed source

- New alphabet regressions against unchanged main: **three failed, two passed**.
- Narrowed full Python suite: **165 passed, no skips**, with **87.37%**
  branch-aware coverage against the unchanged 85% gate.
- The five new cases reject duplicate, empty and multi-character symbols and
  preserve ASCII/Unicode checkpoint serialization and seeded generation.
- Scoped Ruff, strict Mypy and patch-whitespace checks pass.
- The reused lockfile's Git blob hash matches canonical PR #17 exactly.

Local runtime: Python 3.12.14, PyTorch 2.5.1+cpu, NumPy 2.5.3,
Pydantic 2.14.0 and pytest 9.1.1. Hosted release checks must verify the final
published head under the declared Python 3.14 environment; the current PR body
records their result separately.

These small generated-model fixtures establish input-contract behavior only.
No research campaign, real-model training, model promotion, release or paid
workload was performed. Retained capability findings and scientific blockers
are not revised.
