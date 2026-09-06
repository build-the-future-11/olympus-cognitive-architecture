#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
./scripts/reproduce_all.sh
./scripts/setup_external.sh
./scripts/run_final_external.sh
