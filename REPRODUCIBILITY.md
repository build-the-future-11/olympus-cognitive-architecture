# Reproducibility

Generated: 2026-07-15T13:44:13.236026+00:00

## LabOS commands

- Discover: `.venv/bin/python -m olympus.cli labos discover --workspace /Users/ryan/Documents --artifacts artifacts`
- Validate: `.venv/bin/python -m olympus.cli labos validate --workspace /Users/ryan/Documents --artifacts artifacts`
- Run all smoke tests: `.venv/bin/python -m olympus.cli labos run-all --workspace /Users/ryan/Documents --artifacts artifacts --profile smoke`
- Generate reports: `.venv/bin/python -m olympus.cli labos generate-report --workspace /Users/ryan/Documents --artifacts artifacts --output-root /Users/ryan/Documents/Olympus`

## Project commands

### free-claude-code

- `smoke` `smoke-test`: `.venv/bin/python -m pytest tests -q -o addopts='' -p no:cacheprovider`
- Expected artifacts: results; artifacts; reports
- Compute: CPU-only=True; GPU optional=True; memory=4GB

### genesis-econ

- `smoke` `unit-tests`: `env PYTHONPATH=src python3 -m unittest discover -s tests -v`
- `development` `portfolio-artifacts`: `env PYTHONPATH=src python3 -m genesis_econ portfolio --steps 120 --output artifacts/labos-portfolio.json`
- `benchmark` `benchmark-ablations`: `env PYTHONPATH=src python3 -m genesis_econ benchmark --steps 120 --seeds 1,2 --output artifacts/labos-benchmark.json`
- Expected artifacts: artifacts/labos-portfolio.json; artifacts/labos-benchmark.json; artifacts/validation.json; artifacts/ventures.json
- Compute: CPU-only=True; GPU optional=False; memory=2GB

### labos

- No runnable entry points are currently declared.
- Expected artifacts: results; artifacts; reports
- Compute: CPU-only=True; GPU optional=False; memory=4GB

### ledger

- `smoke` `node-test`: `npm test`
- Expected artifacts: results; artifacts; reports
- Compute: CPU-only=True; GPU optional=False; memory=4GB

### olympus

- `smoke` `smoke-demos`: `.venv/bin/python -m olympus.cli demo run-all`
- `development` `backend-tests`: `.venv/bin/python -m pytest`
- `development` `portfolio-discovery`: `.venv/bin/python -m olympus.cli labos discover --workspace ..`
- Expected artifacts: artifacts/labos_runs.sqlite3; portfolio_status.json; PORTFOLIO_COMPLETION_REPORT.md
- Compute: CPU-only=True; GPU optional=True; memory=4GB

### project-ascension

- `smoke` `pytest-suite`: `env PYTHONPATH=src /Users/ryan/Documents/Olympus/.venv/bin/python -m pytest -q`
- `benchmark` `full-portfolio`: `env PYTHONPATH=src python3 -m ascension.experiments.full_portfolio --output results/labos-full-portfolio.json`
- `benchmark` `model-benchmark`: `env PYTHONPATH=src python3 -m ascension.experiments.model_benchmark --output results/labos-model-benchmark.json`
- Expected artifacts: results/labos-full-portfolio.json; results/labos-model-benchmark.json; docs/release-audit.md
- Compute: CPU-only=True; GPU optional=False; memory=2GB

### project-atlas-portfolio

- `smoke` `smoke-test`: `.venv/bin/python -m pytest -q -p no:cacheprovider`
- Expected artifacts: results; artifacts; reports
- Compute: CPU-only=True; GPU optional=False; memory=4GB

### project-genesis

- `smoke` `smoke-test`: `.venv/bin/python -m pytest -q -p no:cacheprovider`
- Expected artifacts: results; artifacts; reports
- Compute: CPU-only=True; GPU optional=False; memory=4GB

### tracecompression

- No runnable entry points are currently declared.
- Expected artifacts: results; artifacts; reports
- Compute: CPU-only=True; GPU optional=False; memory=4GB
