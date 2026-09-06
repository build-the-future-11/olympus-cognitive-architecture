#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PANTHEON_PYTHON:-../../.venv/bin/python}"
MANIFEST="${PANTHEON_MANIFEST:-external/corebench/manifest.jsonl}"
AGENT_A_MODEL="${PANTHEON_AGENT_A_MODEL:-qwen3:0.6b}"
AGENT_B_MODEL="${PANTHEON_AGENT_B_MODEL:-llama3.2:1b}"
export PANTHEON_AGENT_A_PROVIDER=ollama
export PANTHEON_AGENT_B_PROVIDER=ollama
export PANTHEON_AGENT_A_MODEL="$AGENT_A_MODEL"
export PANTHEON_AGENT_B_MODEL="$AGENT_B_MODEL"
export PANTHEON_PROVIDER_TIMEOUT="${PANTHEON_PROVIDER_TIMEOUT:-900}"
export PANTHEON_MAX_COMPLETION_TOKENS=768
export PYTHONPATH="${PWD}/src"

"$PYTHON_BIN" scripts/check_external_ready.py \
  --manifest "$MANIFEST" \
  --agent-a-provider ollama --agent-a-model "$AGENT_A_MODEL" \
  --agent-b-provider ollama --agent-b-model "$AGENT_B_MODEL"
"$PYTHON_BIN" -m pytest -q
if [[ -f configs/local_ood_protocol.json ]]; then
  "$PYTHON_BIN" scripts/write_local_ood_protocol.py --manifest "$MANIFEST" --verify
else
  "$PYTHON_BIN" scripts/write_local_ood_protocol.py --manifest "$MANIFEST"
fi
"$PYTHON_BIN" scripts/run_external_study.py \
  --manifest "$MANIFEST" \
  --agent-a-provider ollama --agent-a-model "$AGENT_A_MODEL" \
  --agent-b-provider ollama --agent-b-model "$AGENT_B_MODEL" \
  --max-turns "${PANTHEON_MAX_TURNS:-20}" --resume
"$PYTHON_BIN" scripts/analyze_external_results.py
"$PYTHON_BIN" scripts/publication_gate.py
