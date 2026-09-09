#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
PYTHON_BIN="${PANTHEON_PYTHON:-python}"
MANIFEST="${PANTHEON_EXTERNAL_MANIFEST:-external/corebench/manifest.jsonl}"
PROTOCOL="configs/local_ood_protocol.json"
AGENT_A_MODEL="qwen3:0.6b"
AGENT_B_MODEL="llama3.2:1b"
export PANTHEON_AGENT_A_PROVIDER=ollama
export PANTHEON_AGENT_B_PROVIDER=ollama
export PANTHEON_AGENT_A_MODEL="$AGENT_A_MODEL"
export PANTHEON_AGENT_B_MODEL="$AGENT_B_MODEL"
export PANTHEON_PROVIDER_TIMEOUT=900
export PANTHEON_MAX_COMPLETION_TOKENS=768
if [[ ! -f "$PROTOCOL" ]]; then
  echo "Protocol does not exist: $PROTOCOL" >&2
  echo "Freeze a new versioned protocol explicitly before any confirmatory execution." >&2
  exit 4
fi
"$PYTHON_BIN" scripts/write_local_ood_protocol.py \
  --manifest "$MANIFEST" --protocol "$PROTOCOL" --verify
"$PYTHON_BIN" scripts/check_external_ready.py \
  --manifest "$MANIFEST" \
  --agent-a-provider ollama --agent-a-model "$AGENT_A_MODEL" \
  --agent-b-provider ollama --agent-b-model "$AGENT_B_MODEL"
"$PYTHON_BIN" scripts/run_external_study.py \
  --manifest "$MANIFEST" \
  --agent-a-provider ollama --agent-a-model "$AGENT_A_MODEL" \
  --agent-b-provider ollama --agent-b-model "$AGENT_B_MODEL" \
  --max-turns 20 --resume
"$PYTHON_BIN" analysis/analyze_external_rigor.py
"$PYTHON_BIN" scripts/analyze_external_results.py
"$PYTHON_BIN" scripts/validate_evidence.py
"$PYTHON_BIN" scripts/publication_gate.py
"$PYTHON_BIN" poststudy/scientific_readiness_gate.py \
  --legacy-gate poststudy/STRUCTURAL_READINESS.json \
  --require-capability-floor
echo
echo "See poststudy/SCIENTIFIC_READINESS.md for the evidence-based verdict."
