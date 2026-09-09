#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PANTHEON_PYTHON:-python3}"
mkdir -p external/corebench
META="external/corebench/core_ood.json"
META_URL="${PANTHEON_COREBENCH_METADATA_URL:-https://huggingface.co/datasets/agent-evals/core-bench-v1.1-ood/resolve/main/core_test.json}"
CAPSULE_URL="${PANTHEON_COREBENCH_CAPSULE_URL:-}"
if [[ -z "$CAPSULE_URL" ]]; then
  CAPSULE_URL='https://huggingface.co/datasets/agent-evals/core-bench-v1.1-ood/resolve/main/capsules/{task_id}.tar.gz'
fi
# Frozen OOD subset reaches the prespecified >=15-task, >=20-question and
# >=2-field gates while avoiding the two 400-800 MB capsules.
TASK_IDS="${PANTHEON_COREBENCH_TASK_IDS:-capsule-0571975,capsule-0811394,capsule-2151475,capsule-2675546,capsule-3990498,capsule-4407237,capsule-5007288,capsule-5360076,capsule-6562149,capsule-6724161,capsule-7350043,capsule-8185407,capsule-8353473,capsule-8467067,capsule-8610546,capsule-9026204,capsule-5172670}"
if [[ ! -s "$META" ]]; then
  echo "Downloading official CORE-Bench v1.1 OOD metadata..."
  curl -fL --retry 3 --connect-timeout 20 "$META_URL" -o "$META"
fi
"$PYTHON_BIN" scripts/prepare_corebench.py \
  --metadata "$META" \
  --task-ids "$TASK_IDS" \
  --download-capsules \
  --visibility ood \
  --capsule-base-url "$CAPSULE_URL"
"$PYTHON_BIN" scripts/check_external_ready.py --manifest external/corebench/manifest.jsonl
