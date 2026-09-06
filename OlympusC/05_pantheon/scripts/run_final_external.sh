#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
MANIFEST="${PANTHEON_EXTERNAL_MANIFEST:-external/corebench/manifest.jsonl}"
python scripts/check_external_ready.py --manifest "$MANIFEST"
python scripts/run_external_study.py --manifest "$MANIFEST" --resume
python scripts/analyze_external_results.py
python scripts/build_registry.py
python scripts/validate_evidence.py
python scripts/publication_gate.py || true
echo
echo "See PUBLICATION_READINESS.md for the evidence-based verdict."
