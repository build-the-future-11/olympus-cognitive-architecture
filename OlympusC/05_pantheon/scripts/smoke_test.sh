#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python scripts/make_demo_package.py --out packages/demo --seed 42
python scripts/run_proposer.py --package packages/demo --run-dir runs/demo/proposer --seed 42
python scripts/run_replicator.py --package packages/demo --run-dir runs/demo/replicator --seed 42 --fresh-workspace
python scripts/compare_results.py --package packages/demo --proposer runs/demo/proposer --replicator runs/demo/replicator --out runs/demo/comparison.json
