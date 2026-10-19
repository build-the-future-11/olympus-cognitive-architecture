# Foundry admission and provenance: evidence handoff

**Date: 2026-10-19. Disposition: bounded engineering verification passed; no model capability claim.**

This record supports the Foundry admission, artifact identity, export, live-health,
promotion-evidence, and startup-import repairs. The expanded Python suite passed
432 tests, and the fixed verification-family lifecycle completed within its
declared CPU resource bounds. Earlier failed attempts remain available. The
separate residency diagnostic stopped at its resource guard before producing an
experimental result. No Hermes model was promoted, and historical scientific
negatives remain unchanged.

## Start with these files

| File | Purpose |
|---|---|
| [RESULTS.md](RESULTS.md) | Tables regenerated from the retained evidence. |
| [workflow-receipt.json](workflow-receipt.json) | Successful v4 commands, measurements, source identities, and artifact hashes. |
| [source-manifest.json](source-manifest.json) | Executed implementation, tests, configuration, corpus, and workflow source identities. |
| [attempt-source-verification.json](attempt-source-verification.json) | Verification of each attempt's saved source hashes against known commits. |
| [v1 receipt](v1-stopped-receipt.json), [v2 receipt](v2-stopped-receipt.json), [v3 receipt](v3-stopped-receipt.json) | Preserved failed lifecycle attempts. |
| [v4-environment-protocol.json](v4-environment-protocol.json) | Frozen final preparation, unchanged thresholds, and one-attempt stopping rule. |
| [verification-model.json](verification-model.json) | Exact portable character-bigram checkpoint from v4. |
| [evidence.tar.gz](evidence.tar.gz) | 82 raw evidence files plus `ARCHIVE_MANIFEST.json`. |
| [archive-identities.json](archive-identities.json) | Archive SHA-256 values, sizes, and verified member counts. |

The archive includes test logs, JUnit XML, coverage JSON, complete lifecycle
stdout/stderr, the SQLite registry, checkpoint/export bytes, built distributions,
resource-preparation records, and historical source-review material. Its internal
manifest binds every raw member by SHA-256 and byte count. The separate
[scientific audit](../../olympus_audit_20261019/SCIENTIFIC_AUDIT.md) and
[residency outcome](../../olympus_audit_20261019/RESIDENCY_OUTCOME.json) retain
their own scope and evidence.

## Exact source and review boundary

The executed v4 source is commit
`92cc121e6f1074de72eaee5a8cb670170107bbc7`, tree
`abed3df4bd3e9309eda1e93884247b11f53e4045`. Later documentation and evidence
commits do not replace that execution identity. The workflow script SHA-256 is
`30051b7281e4147da3d0a41d65b0169904956c10530e692d8a92aa6d67a87fe4`.

The v4 receipt records an empty Git status and the exact commit. Read-only review
recomputed all 22 workflow artifact hashes, compared six JSON stdout files with
the receipt, and matched all 63 recorded Python/configuration source hashes to
the committed bytes. Checkpoint provenance also agrees with the experiment,
dataset, and code commit.

V1 and v3 stopped before launching even the Git identity command. Their receipts
therefore have no recorded Git HEAD. Their 63 saved source hashes and script
hash were subsequently checked against known commits, as documented in
`attempt-source-verification.json`; that comparison must not be described as a
successful in-run Git check.

Independent reviews were source review and saved-evidence consistency checks.
Reviewers did not independently retrain, regenerate model scores, or reproduce
the experiment. Hash agreement binds the available evidence; it does not
authenticate a malicious writer or establish an externally attested execution.

## What the repair verifies

Admission now requires coherent dataset, experiment, checkpoint, and passing
evaluation records, with the supported verification-family identity. Serving,
health, and export use the same current artifact admission. Export validates the
source before publication and preserves an existing destination on failure.
Checkpoint consumers record the identity of the byte snapshot actually loaded.

Deep promotion checks report populations, derived arithmetic, finite
measurements, positive context capacity, and actual quantized-artifact identity.
Existing scientific quality thresholds remain unchanged. API and CLI startup
defer tensor backends until their relevant commands or endpoints execute.
Artificial fixtures challenge these invariants, including deliberate corruption,
contradictory reports, interrupted publication, and replacement after reading.
See the [detailed verification contract](../../../docs/foundry-admission-20261019.md).

## Engineering gate results

| Check | Recorded result |
|---|---|
| Expanded Python suite | **432 passed**, zero failures/errors/skips; pytest reports **14.42 s**. |
| Combined statement/branch coverage | **88.53%**, above the configured 85% combined gate. |
| Branch coverage alone | **73.46%**; this is a different denominator. |
| Web suite | **7 passed**; latest suite duration **2.20 s**. |
| Web production build | Passed. |
| Ruff | Passed. |
| Strict Mypy | Passed across **89 source files**. |
| Wheel and source distribution | Built; both passed `twine check`. |
| npm dependency audit | Reported zero vulnerabilities. |
| Python dependency audit | **Blocked** for the exact installed `torch==2.14.1+cpu`, unavailable to the PyPI audit lookup. |

Use `test_logs/expanded_tests.log`, `expanded_tests.xml`, and
`expanded_coverage.json` for the expanded result. Retained `final_tests.*` and
`final_coverage.json` describe the earlier 351-test, 87.99% check; their filenames
do not make them the latest evidence. Coverage is 4,001/4,348 statements and
739/1,006 branches, yielding 4,740/5,354 combined opportunities. The web command
receipt records 3.037789 seconds including npm startup; 2.20 seconds is the test
runner's duration. Neither dependency-audit disposition is a general security
guarantee.

There are **133 new constructed cases**: 52 admission/export/snapshot cases,
11 resource-accounting cases, 68 promotion-evidence cases, and two startup-import
cases. **104 cases failed on their applicable predecessor implementations**:
45 from the first group, 57 promotion cases, and two import cases. These are
parameterized failure cases, not 104 independently discovered bugs, scientific
samples, or measured capability improvements. Baseline controls and unsuccessful
setup attempts remain in the archive.

## Lifecycle method and observed result

The built-in, repository-owned synthetic corpus contains 674 bytes, SHA-256
`6f4823e0503426bda11aa4bffe8357963f3bc97a62ea40c2b69700af166a8eab`.
The fixed settings are seed 7, training fraction 0.8, and smoothing 0.25.
Training uses 539 characters and 538 target transitions. Evaluation uses 135
held-out target transitions; its 136-character slice includes one boundary
context character. The baseline is the frozen uniform distribution over the
24-symbol training alphabet.

| Held-out measurement | Uniform baseline | Trained bigram |
|---|---:|---:|
| Negative log-likelihood | 3.1780538303479458 | 1.9001575710987069 |
| Perplexity | 24.0 | 6.686948029017607 |
| Top-1 accuracy | 0.041666666666666664 | 0.3925925925925926 |
| Evaluated transitions | 135 | 135 |

The candidate satisfies the declared lower-NLL and lower-perplexity rule. The
checkpoint, internal export, and portable export are identical 1,712-byte files,
SHA-256 `0292e4d08ff43c4638b42ba5013eb897dad60ab4fbbdb609f661b0b731ccdb06`.
Restarted CLI, export health, and API evidence identify those same bytes. CLI and
API produce the same 24-character string, `' re re re re re re re re'`, for the
matched prompt and settings. API listing and completion return 200 through ASGI
transport in a fresh worker process. Reloaded SQLite status reports integrity
`ok`, one record of each principal type, and five evidence events.

All eight child commands exited successfully. Total lifecycle time was
**4.272115 seconds** within 120 seconds; the longest child took 0.803730 seconds
within 30 seconds. Maximum child RSS was 67,338,240 bytes. Across 39 retained
memory snapshots, minimum estimated headroom was **2,231,209,984 bytes**, or
83,726,336 bytes above the unchanged 2 GiB floor. Recorded swap usage was zero.

## Failed attempts and measurement limits

The actual execution host was Linux x86_64 with Python 3.14.8 and CPU-only
Torch 2.14.1+cpu installed. Its cgroup limit was 8 GiB; there was no swap and
19,509,891,072 bytes of free disk before v4. This was the session's Linux executor,
not the target 16 GB Apple Silicon Mac. Local Node was 24.19.0 and npm 11.9.0;
the hosted workflow pins Node 22.23.2. No Kaggle execution or complete hosted
release gate is claimed for this revision.

V1 stopped before any child because memory admission failed. V2 introduced the
documented conservative inactive-cache estimate while retaining the 2 GiB floor;
it then stopped during eager Torch startup, with no training output. V3 contained
the import deferral but again failed preflight before any child. The frozen v4
protocol permitted one final environment-only preparation using advisory cache
release on already-used dependency/build files. File contents, source, data,
seed, resource policy, and quality thresholds were unchanged. Every attempt is
retained; no quality-driven rerun or parameter sweep occurred.

The separate residency diagnostic launched only its fixture worker. The guard
stopped it after 1.014418455 seconds when sampled estimated headroom fell to
2,123,210,752 bytes, below the same floor. No completed fixture, arm result,
ownership metric, or comparative hypothesis result exists. Its status is
`BLOCKED`, not a supported or falsified residency claim.

Headroom is a sampled, cache-adjusted estimate, not continuously observed free
RAM. Raw `memory.stat` components were not saved, so historical cache-credit
derivation cannot be recomputed from individual counters. Hash and arithmetic
checks alone do not establish model capability or browser/network deployment.
The deep-promotion consistency repair does not independently establish artifact
tensor loadability, licensing, or serving reproduction. Promotion
report/release writes are individually atomic, not one crash-atomic transaction;
later rejection does not revoke an older release. These remain explicit limits.

## Minimal use

Run the archive verifier and table generator from the repository root:

```bash
python3 research/verification/foundry_admission_20261019/regenerate_tables.py
```

This verifies both compressed archives and every member hash without extracting
or executing members, then prints Markdown tables to stdout. Add
`--output /tmp/olympus-evidence-results-new.md` to save a separate table file.
It does not import Olympus or execute training. The archive's historical
absolute paths identify the original run; use its member paths when inspecting
a copy on another machine.

For a fresh lifecycle reproduction, use the recorded source and
a prepared Python 3.14 environment with the project dependencies, then choose an
output directory that does not already exist:

```bash
.venv/bin/python scripts/verify_foundry_admission.py \
  --output /tmp/olympus-foundry-reproduction-new
```

The runner owns a fresh directory, keeps its workload lock, checks the built-in
corpus identity, applies the recorded bounds, and preserves failure receipts.
Do not reuse the published registry as a new run directory. Timestamped
provenance makes a new checkpoint hash expected even with identical settings;
compare measured behavior and within-run identity rather than demanding the
historical digest. Preserve any new failure and update the
[canonical state](../../../RESEARCH_STATE.json) with an appended disposition.

The next recorded research action is to retrieve and hash-verify the retained v5
and EditCLEVR evidence archives through supported authenticated notebook outputs.
Their existing results should be reconciled before designing another protected
study. The [scientific audit](../../olympus_audit_20261019/SCIENTIFIC_AUDIT.md)
records the broader model's unmet research gates and the deferred mechanism work.
