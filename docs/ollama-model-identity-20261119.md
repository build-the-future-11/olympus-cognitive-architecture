# Ollama response model identity — engineering verification

Date: 2026-11-19. Source basis:
`e4f0944cce4524a2454193bdfeb3f785631b14c0` on the canonical
`build-the-future-11/olympus-cognitive-architecture` repository.

## Defect and change

The provider adapter checked message shape, completion, and termination reason,
but did not inspect the response's `model` field. It returned the requested
model name even when the provider declared another model or supplied no usable
identity. A successful response could therefore be attributed to the wrong
requested model.

The adapter now requires a nonempty, well-formed provider model string that
matches the requested identity. Only the documented omitted-tag / `:latest`
equivalence is admitted. Tags are detected in the final path component, so a
registry's port does not impersonate a model tag. Different names, namespaces,
tags, and registry hosts fail through the existing provider-error boundary.
Successful evidence retains both `requested_model` and `provider_model`.

This verifies a provider-declared name. It is not cryptographic attestation of
model weights and cannot detect a provider lying about the identity it reports.
Other aliases are not guessed. A deployment that rewrites names must make that
mapping explicit before it can be admitted by this adapter.

The official [chat response specification](https://docs.ollama.com/api/chat)
defines the response model field as the model used for generation. The official
[model naming conventions](https://github.com/ollama/ollama/blob/main/docs/api.md)
document the default `latest` tag. Both were inspected on 2026-11-19.

## Falsification and results

The new offline transport regression suite was executed before the source
repair. All 30 cases failed: 25 refusal cases showed missing, malformed, or
different model identity being admitted; five matched-identity controls showed
the requested/provider identity fields were absent. These are constructed
engineering cases, not sampled scientific observations.

After the correction:

- 63 focused provider, API, and Foundry tests passed in 5.35 seconds.
- The complete Python suite passed 378 tests in 48.84 seconds, with 87.47%
  branch-enabled total coverage, exceeding the unchanged 85% requirement.
- Whole-repository Ruff passed.
- Strict MyPy passed for 87 source files.
- The ordinary Foundry CLI lifecycle completed in a fresh output root and a
  second CLI process read the retained registry with integrity `ok`.

The CLI verification used the explicitly synthetic repository-owned character
corpus. It retained experiment `FND_20261119T102014833804`, checkpoint
`ckpt_d9d2000d512d2e23`, evaluation `eval_6efc36587c37f9f7`, and a portable export.
The exact experiment identity is also retained in `lifecycle.log`; that raw
receipt is authoritative if any identifier is corrected later.

The verification model's held-out perplexity was 6.686948029017607 against its
fixed uniform value of 24 over 135 transitions. This repeats the existing
infrastructure lifecycle and does not establish assistant capability or a
competitive research result. Historical deep-model `NOT_PROMOTED` findings,
protected data, and frozen scientific studies remain unchanged.

## Reproduce and inspect

Use Python 3.14 and the declared package dependencies. The exact observed
package versions are retained in the verification directory; this run used
Python 3.14.10 and PyTorch 2.15.0+cpu.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest --cov=olympus --cov-branch \
  --cov-report=term-missing -q
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy olympus tests
.venv/bin/python -m olympus.cli foundry verify-pipeline \
  --root artifacts/verification-20261119
.venv/bin/python -m olympus.cli foundry status \
  --root artifacts/verification-20261119
```

`research/verification/ollama_model_identity_20261119/` retains the pre-fix
failures, commands, exit statuses, full test/lint/type logs, environment, source
hashes, and real CLI receipts. The unit fixtures call no external provider and
load no external model.

The base revision's release workflow was separately observed to have passed
[run 39776856344](https://github.com/build-the-future-11/olympus-cognitive-architecture/actions/runs/39776856344).
That is evidence for the base source only. The current repair has local Python
verification, not a new hosted release run. Web files were not changed or
retested in this repair; full release validation, independent review, and
integration remain distinct from the checks recorded here. No release,
deployment, or scientific submission occurred. Development-branch persistence
is reported separately from this local validation receipt.
