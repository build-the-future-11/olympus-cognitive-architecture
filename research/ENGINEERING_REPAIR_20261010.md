# Dataset admission and character sampling repair — 10 October 2026

This development repair starts from main `88cc23517e758e4fcd48d5bf08e269973a36f125`.
Read `REALITY_LEDGER.md` and `../OLYMPUS_MODEL_FOUNDRY_LEDGER.md` for the retained
model evidence and negative capability findings. The prior Ollama, UI and timing
PRs are separate work; this patch does not replace their evidence.

## Defects and corrected behavior

Dataset verification previously validated hashes and record counts but trusted
the stored quality summary. Re-sealing an incomplete, duplicate or contaminated
dataset could therefore pass the consumption gate despite violating preparation's
rules. Preparation and verification now share one quality validator. Consumption
requires all three splits, complete category coverage, disjoint record IDs,
normalized deduplication, the existing cross-split eight-token rule, and an exact
quality-summary match. The original dataset bytes, manifest schema and valid
prepared-dataset identity remain unchanged.

Character sampling previously raised an exception when every `p ** (1 / T)`
weight underflowed at a valid, positive temperature. Sampling now subtracts the
maximum log probability before division by temperature. This common factor
cancels in normalization and leaves at least one weight equal to one; tied
maxima still sample randomly. Temperature zero remains greedy. Checkpoint
alphabets now reject duplicate, empty or multi-character symbols, which broke
the character-index and output-length contract.

## Verification

- The new 24-case set produced **20 failures and 4 passes** against unchanged
  product source before the repair, then **24 passes** after it.
- Full Python suite: **184 passed**, no skipped cases.
- Branch-aware coverage: **87.45%**, above the unchanged 85% requirement.
- Scoped Ruff, strict Mypy and patch-whitespace checks passed.
- The new tests use modified temporary copies of the repository's existing
  36-record engineering fixture and small in-memory character models. The full
  suite retains its existing bounded synthetic training fixtures.

Local runtime: CPython 3.12.14, PyTorch 2.5.1+cpu, NumPy 2.5.3, Pydantic 2.14.0,
pytest 9.1.1. This is a compatibility test environment; the declared release
environment remains Python >=3.14 and PyTorch >=2.11. Hosted CI must verify that
release environment on the published commit.

Reproduce in the declared project environment:

```sh
python -m pytest tests/test_dataset_consumption.py tests/test_bigram_sampling.py -q
python -m pytest --cov=olympus --cov-branch --cov-report=term-missing -q
python -m ruff check olympus/foundry/bigram.py olympus/foundry/data_pipeline.py tests/test_bigram_sampling.py tests/test_dataset_consumption.py
```

## Research disposition

These are implementation and evidence-admission corrections. They do not train
Hermes, establish task competence, measure live-provider performance, promote a
model, or complete the frozen scientific program. Existing negative results,
dataset scale requirements, licensing, real-workload evaluation and serving
gates remain the canonical blockers. No protected campaign or paid job ran.
