# SFT supervised-token normalization — 10 October 2026

## State and scope

The canonical scientific state remains `OLYMPUS_MODEL_FOUNDRY_LEDGER.md` and
`research/REALITY_LEDGER.md`: the tiny transformer is an infrastructure fixture,
the retained capability smoke failed, and no Hermes model is promoted. Those
ledgers, the dataset bytes, recorded outcomes and promotion thresholds are
unchanged. This correction extends the dataset-contract repair in PR #23 at
`c37066fa5c349eaca0acf43b0e9a57306fdf42ec` without duplicating its changes.

## Reproduced defects

The evaluator averaged per-batch cross-entropy means. A constructed nine-row
fixture with unequal target counts gave **1.2751091774510233**, while its analytic
supervised-token mean is **2.0365126862229523**. Reordering or repartitioning rows
could change the reported metric.

The trainer divided each microbatch mean by the configured accumulation count.
This weighted short and long answers equally by microbatch and underweighted the
last incomplete window. Three actual tiny-transformer comparisons against
unsplit batches failed on the old source, with maximum pre-clipping gradient
differences of approximately 0.011–0.023. A separate two-microbatch fixture with
equal 46-target answers isolates the tail defect: an accumulation setting of five
reduced the old gradient to two-fifths of the intended mean.

These are implementation regression fixtures, not model-performance evidence.

## Corrected objective and compatibility

`supervised_token_mean_v2` is the default for new runs. For one optimizer window,
each microbatch contributes its summed cross entropy divided by the window's
total supervised next-token count. Prompt masks, padding and unpredicted first
columns are excluded. The complete gradient is normalized before clipping and
the optimizer step; incomplete windows use their actual counts. Evaluation and
epoch-loss reporting use the same token denominator.

The version is bound to checkpoint configuration and metadata, training logs,
summaries, held-out evaluation and quantization reports. V2 uses a distinct
`-tokens-v2` run path. Unversioned checkpoints and older reports retain the
explicit `legacy_batch_mean_v1` interpretation. Legacy evaluation arithmetic
remains available, and legacy checkpoints can resume only with the same explicit
configuration. Cross-version resumption and inconsistent version metadata are
rejected.

V2 changes training trajectories and loss values. New values are **not directly
comparable to retained v1 metrics without their source and normalization
version**. No retained checkpoint, loss report or capability conclusion was
rewritten. This change does not claim a gain in capability, efficiency, or model
quality.

## Verification

The new regression module checks the analytic loss, prompt/padding/shift masks,
unequal answer lengths, full and partial accumulation windows, gradients before
clipping, actual AdamW parameter updates, versioned paths and receipts, legacy
checkpoint loading/resumption, cross-version rejection, and held-out/quantization
propagation. Paired training uses the actual 14,192-parameter transformer and
existing reviewed infrastructure fixture with dropout disabled; comparisons
allow ordinary float32 rounding and do not claim bitwise or stochastic-backend
equivalence.

The five initial reproduction/control cases on unchanged PR #23 source produced
**4 failures and 1 pass**. The isolated equal-answer tail case also failed before
the correction and passed afterward. Numerical comparisons use an independent
analytic log-partition reference and actual unsplit-batch gradients.

All **19 new regressions** pass in the complete **198-test** Python suite.
Branch-aware coverage is **87.54%**, above the unchanged **85%** requirement.
Whole-repository Ruff checks pass, and strict mypy reports no issues in all
78 source files.

Verification commands:

```bash
python -m pytest tests/test_sft_token_normalization.py -q
python -m pytest --cov=olympus --cov-branch --cov-report=term-missing -q
python -m ruff check .
python -m mypy olympus tests
```

Local verification uses Python 3.12.14 and PyTorch 2.14.1+cpu. The package's
declared release environment remains Python 3.14 and is checked by the unchanged
hosted release gate. No external model download, provider call, remote training,
large-model campaign, or model promotion was performed.
