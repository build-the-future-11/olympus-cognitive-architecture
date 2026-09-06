#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=src
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY before running}"
./scripts/reproduce_all.sh
./scripts/setup_external.sh
./scripts/run_final_external.sh
cat PUBLICATION_READINESS.md
