# Foundry integration: bounded padded batches and dataset identity

Date: 2026-10-10. This is engineering development using generated fixtures.
The earlier model ledgers, failed capability gates, and absence of a promoted
Hermes model remain unchanged.

## Source and intended behavior

This revision integrates PR #26 at
`2519ef345da556d210a1cb0f705077a835b6f8e6` and PR #27 at
`cbc54b95970891e54d658c085ead03714396aa13`. Both parent histories are preserved
on the existing PR #27 branch. PR #26's supervised-token normalization v2 and
legacy compatibility mode now consume PR #27's exact verified record snapshot.
The public manifest-verification API and prior dated receipts remain intact.

The SFT loop previously padded and retained every batch in an epoch before its
first backward pass. It now prepares one gradient-accumulation window, performs
the same losses and optimizer update, and releases those padded tensors before
preparing the next window. Encoded corpus rows and the permutation remain in
memory; this is a bound on padded-batch retention, not a streaming dataset reader.
Batch order, dedicated generator use, token denominators, optimizer boundaries,
and the intentional legacy tail-window divisor are preserved.

Independent review then identified a second defect: quantization verified a
dataset but did not compare its manifest identity with the source checkpoint,
and wrote weight artifacts before data verification. Quantization now verifies
the complete dataset snapshot and requires the checkpoint's dataset hash to
match before creating the output directory or writing weights. The same admitted
test records feed both loss calculations. Quantization arithmetic is unchanged.

## Bounded falsification and verification

The four actual-trainer memory tests cover both normalization modes and
accumulation sizes two and five, including incomplete final windows and two
epochs. All four failed before the memory correction because twelve batches
were prepared before the first training loss. Three quantization admission
tests failed before the second correction: absent identity, mismatched identity,
and corrupt data all reached the weight-publication boundary. These are
constructed engineering fixtures, not outcome-generating scientific runs.

The final complete local suite passed **220 tests in 34.34 seconds**, with
**87.58% total coverage with branch measurement enabled**. This includes the
existing mathematical normalization checks, verified-snapshot consumer checks,
and seven new regressions. Whole-repository Ruff passed. The workflow's strict
Mypy targets, `olympus tests`, passed for **81 source files** using a fresh cache.
Environment: Python 3.12.14 and CPU Torch 2.14.1; the project release gate uses
Python 3.14 and remains a separate hosted verification requirement.

Development failures are retained rather than counted as successful runs. The
first integration checks exposed an obsolete private helper import in an
inherited test and then a missing verifier import; both test integration errors
were corrected. An exploratory `mypy .` invocation selected duplicate generic
application module names outside the workflow's targets. A subsequent cached
Mypy invocation produced an internal tool error; the documented target command
with a fresh cache passed. No required check or dependency was removed.

Independent source review found no blocker in the window boundary, RNG/order,
loss-normalization, snapshot-consumption, or quantization-admission changes.
That review did not rerun tests. Raw baseline and final test logs, source hashes,
and exact commands are retained in
`research/verification/foundry_integration_20261010/`.

## Interpretation

These checks support the stated execution and provenance contracts. They do not
establish model capability, scientific novelty, a better empirical metric, or
research completion. No protected outcome was inspected, no paid run was started,
and no frozen study or historical artifact was regenerated. Integration into the
default branch and exact-commit hosted release checks are reported separately.
