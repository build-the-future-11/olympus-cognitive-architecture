#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python - <<'PY'
import sys, numpy, pandas, scipy, sklearn, matplotlib, yaml
print('environment_ok',sys.version.split()[0],numpy.__version__,pandas.__version__,scipy.__version__,sklearn.__version__)
PY
pytest -q
./scripts/smoke_test.sh
./scripts/run_baselines.sh
./scripts/run_experiments.sh
./scripts/build_paper_assets.py
python scripts/build_registry.py
python scripts/validate_evidence.py
