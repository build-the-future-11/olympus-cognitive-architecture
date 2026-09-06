#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python scripts/run_seeded_faults.py --config configs/seeded_faults.yaml
