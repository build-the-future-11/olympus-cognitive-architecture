#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
printf '\nSetup complete. Run: source .venv/bin/activate && export PYTHONPATH=src && ./scripts/reproduce_all.sh\n'
