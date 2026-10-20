# Final Research Execution Report — 2026-10-20

## Decision

**Submission readiness: NO-GO.**  Olympus is not publication-ready as a new
empirical contribution.  The current manuscript remains an honest,
evidence-bound synthesis of its completed studies; this execution added
engineering inspection infrastructure, not a new scientific result.  No
manuscript metric, figure, conclusion, or submission-status field was changed.

## What was executed

### Wave 1: composed-system audit capability

Implemented a source-bound per-example decision-dossier pipeline:

- `olympus/evaluation/decision_dossiers.py` defines a strict, canonical JSON
  artifact with dataset-manifest and checkpoint identities, source IDs, host
  transition masks, raw/exposed predictions, forced STOP state, risk values,
  uncertainty, family gates, and leave-one-family-out decision deltas.
- Bundle validation rejects duplicate source IDs, bad component weights,
  illegal selected transitions, STOP leakage, risk-floor violations, and
  content-digest tampering.
- `train_olympus_system` now writes `decision-dossiers.json` for each completed
  development run and records its path and SHA-256 in schema-3 `report.json`.
  Interrupted export recovery verifies the existing bundle’s binding and source
  ordering before it can be reused.
- Tests cover a forced STOP, source binding, immutable export, and a modified
  bundle.  The CLI training test now requires the dossier artifact.

This completes useful parts of checklist items B02 and F01, but neither item is
checked off: replay from raw source inputs and a numerically executed artifact
are still required.

### Wave 2: attribution replay integrity repair

An adversarial review found that the first dossier integration used the
bundle's internal content digest as the report's file SHA-256 and allowed a
self-consistent, rehashed partial bundle to be reused without numerical replay.
That was an integrity defect, not a scientific failure.

The repair now:

- records the real serialized-file SHA-256 separately from the internal content
  digest in report schema 4;
- validates file type, external file hash, checkpoint/dataset/split binding, and
  exact source order;
- recomputes every gate, risk value, STOP/transition decision, and leave-one-out
  counterfactual from the declared checkpoint and evaluation tensors;
- requires byte-identical canonical replay on the run's declared device;
- uses that replay both for interrupted-export recovery and the new
  `olympus bu1ld verify-olympus-dossiers` command; and
- includes adversarial tests for a wrong file digest and a rehashed changed
  contribution.

This closes the recomputation-verifier portion of checklist item B02. B02
remains open because the typed evidence-availability graph depends on A01/A08.

### Prepared GPU qualification

Built, but did not upload or execute, an offline CUDA qualification notebook:

- Notebook: `artifacts/flagship-v2-kaggle-hosted-cuda-r3-20261020/olympus_kaggle.ipynb`
- Embedded source SHA-256:
  `c75cbe1272e4e02a3086785481108aeb907bfad6555c2e28f4502c66e6a2f6fa`
- Source archive: 173 files, 2,067,420 bytes; archive SHA-256:
  `c75cbe1272e4e02a3086785481108aeb907bfad6555c2e28f4502c66e6a2f6fa`
- The notebook sets `CUDA_VISIBLE_DEVICES=0`, enables GPU mode, disables
  Internet, validates the embedded archive before execution, and invokes only
  the bounded artificial `scripts.qualify_flagship_v2 --device cuda` path.

Rebuild command (creates a new immutable handoff directory):

```sh
.venv/bin/python -m notebooks.build_flagship_runner \
  --output-dir artifacts/flagship-v2-kaggle-hosted-cuda-r3-20261020 \
  --device cuda --embed-source
```

The job is an engineering qualification only.  It is not a target-scale model
training run, baseline study, protected evaluation, or publication experiment.

## Verification evidence

Passed static checks:

```text
ruff check olympus/evaluation/decision_dossiers.py \
  olympus/training/olympus_system.py tests/test_olympus_system.py tests/test_cli.py
-> pass

.venv/bin/mypy olympus/evaluation/decision_dossiers.py \
  olympus/training/olympus_system.py tests/test_olympus_system.py tests/test_cli.py
-> Success: no issues found in 4 source files
```

After the replay-integrity repair, full Ruff, strict MyPy on 167 source files,
Python byte-compilation, 56 tensor-free/state tests, the exact 0/6 readiness
baseline, the 110-package offline lock check, and `git diff --check` passed.
Four packaging-metadata tests passed, and a fresh source archive and wheel built
successfully; the source archive contains both replay-state records and the
implementation, while the wheel contains the decision-dossier module.
Numerical pytest was not admitted: the unchanged pre-import
guard observed 297,926,656 then 292,700,160 free bytes, below the frozen 512 MiB
floor, with pageouts stable at 1,053,427. The refusal is retained in
`research/DECISION_DOSSIER_REPLAY_HARDENING_20261020.json`.

`RESEARCH_STATE.json` was parsed successfully after recording this engineering
work.  Whitespace validation also passed before this report was written.

The preserved manuscript source also compiled successfully with Tectonic on
2026-10-20 using a temporary output directory on the project volume.  The
result remained seven US-Letter pages; the log contained no unresolved-reference
or overfull-box warning.  The checked-in PDF and arXiv archive still matched the
SHA-256 identities in the 2026-10-18 release receipt exactly.  The desktop
preview compiler could not prepare its own temporary directory because the
system data volume had only about 109 MiB free; this was an environment refusal,
not a TeX diagnostic.

## Explicitly unexecuted work and blockers

- Numerical pytest, composed-model training, ablations, robustness trials, and
  figure generation were **not run**.  Earlier admission observed free memory
  fall from about 272 MiB to 60 MiB while pageouts rose.  A fresh 2026-10-20
  preflight observed 62,423,040 free bytes and 94.366% swap utilization.  These
  observations violate the repository’s 512 MiB/no-rising-pageout and 90% swap
  numerical guards.
- A full six-family scientific study remains blocked by the absence of an
  admitted rights-reviewed aligned corpus, frozen successor protocol, controls,
  seed/budget rationale, and protected-evaluation authorization.
- The approved one-GPU Kaggle CUDA qualification remains platform-blocked until
  Kaggle accepts a saved run; this prepared notebook is not a receipt of GPU
  execution.  The prior draft now returns page not found.  The isolated browser
  session was unauthenticated, and its Google sign-in opened an unintended
  new-account registration flow that was cancelled before account creation.  An
  existing signed-in Kaggle tab belonged to another active task and was left
  untouched.  No upload or CUDA session occurred.
- The v5 and EditCLEVR remote evidence archives remain unretrieved locally;
  therefore no new plot/table should be generated from their unrecovered raw
  outputs.
- Existing routing negatives and the narrowly supported synthetic residual
  temporal finding are unchanged.  No capability, novelty, safety,
  deployment, independent-reproduction, or promotion claim is added.

## Publication package status

The existing consolidated paper and arXiv package remain the only manuscript
package with recorded evidence.  A new submission package would be premature:
the new engineering feature has static validation only, and the essential
empirical prerequisites above remain unsatisfied.  The appropriate next action
is to restore numerical resource admission, execute the focused dossier tests,
then freeze a separately versioned scientific protocol before any new primary
experiment.

The requested personal `arxiv.sty` was not present in the repository, supplied
attachments, or the likely user document/workspace locations searched on
2026-10-20.  The verified manuscript therefore still uses its recorded plain
`article` layout.  Substituting an unrelated style file would not satisfy that
request and would invalidate the existing release hashes.
