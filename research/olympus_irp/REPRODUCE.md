# Reproduction instructions

## Environment

- Python 3.14
- dependency lock and package metadata from this checkout
- CPU execution for the reported matrix
- repository head at protocol initialization:
  `20a5687d041859e888bf21390a37483177177dd8`
- dirty worktree intentionally preserved; run manifests and source hashes bind results
  to the actual files used

## Verify software

```bash
rtk .venv/bin/ruff check .
rtk .venv/bin/mypy olympus
rtk .venv/bin/python -m pytest --cov=olympus --cov-report=term -q
```

## Re-run the frozen exploratory matrix

Choose a new output root because artifacts are immutable:

```bash
rtk .venv/bin/python scripts/run_olympus_backtests.py \
  --manifest artifacts/foundry/deep/dataset/manifest.json \
  --output artifacts/research/olympus-irp-reproduction
```

The resource governor may reject execution. A rejection is a retained outcome; do not
weaken the memory or swap guard. The command observes an already inspected smoke test and
is therefore not a confirmatory reproduction.

## Regenerate the paper

```bash
rtk tectonic -X compile research/olympus_irp/paper/main.tex \
  --outdir output/pdf
```

Copying or submitting the artifact externally is not part of this repository workflow.
