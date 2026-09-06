#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "local_ood_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files() -> list[Path]:
    roots = [ROOT / "src", ROOT / "scripts", ROOT / "tests"]
    files = [path for root in roots for path in root.rglob("*") if path.is_file() and path.suffix in {".py", ".sh"}]
    files.extend([ROOT / "pyproject.toml", ROOT / "docs" / "EXTERNAL_VALIDATION_PROTOCOL.md"])
    return sorted(files)


def source_identity() -> tuple[str, dict[str, str]]:
    per_file = {str(path.relative_to(ROOT)): sha256(path) for path in source_files()}
    canonical = json.dumps(per_file, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), per_file


def build(manifest_path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as response:
        installed = {item["name"]: item["digest"] for item in json.load(response).get("models", [])}
    models = ["qwen3:0.6b", "llama3.2:1b"]
    missing = [model for model in models if model not in installed]
    if missing:
        raise SystemExit(f"missing frozen model(s): {missing}")
    identity, files = source_identity()
    return {
        "schema_version": 4,
        "protocol_id": "PANTHEON-COREBENCH-OOD-LOCAL-20260903-V4",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity": identity,
        "source_file_sha256": files,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256(manifest_path),
        "tasks": len(rows),
        "questions": sum(len(json.loads((ROOT / row["canonical_answers"]).read_text())) for row in rows),
        "fields": sorted({row["field"] for row in rows}),
        "benchmark_visibility": sorted({row["benchmark_visibility"] for row in rows}),
        "agents": [
            {"provider": "ollama", "model": model, "digest": installed[model]}
            for model in models
        ],
        "decoding": {
            "temperature": 0,
            "max_turns": 20,
            "max_completion_tokens_per_turn": 768,
            "provider_timeout_seconds": 900
        },
        "isolation": {
            "workspace": "distinct ephemeral /private/tmp extraction per agent",
            "filesystem": "macOS sandbox-exec denies /Users/ryan and /Volumes/PRO-BLADE",
            "network": "denied inside agent shell",
            "canonical_answers_exposed_to_agent": False
        },
        "success_gate": {
            "minimum_tasks": 15,
            "minimum_questions": 20,
            "minimum_fields": 2,
            "distinct_agent_identities": True,
            "all_reports_required": True,
            "provider_errors_allowed": False
        },
        "failure_handling": {
            "one_bounded_corrective_turn_after_missing_report": True,
            "missing_or_invalid_exact_question_keys_normalized_to_null": True,
            "joint_null_answers_count_as_agreement": False,
            "agent_submission_origin_retained": True
        },
        "claim_boundary": "Local small-model OOD reproduction evidence; does not estimate frontier-agent or human-adjudication performance.",
        "python": sys.version,
        "platform": platform.platform()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="external/corebench/manifest.jsonl")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest = (ROOT / args.manifest).resolve()
    if args.verify:
        if not PROTOCOL.exists():
            raise SystemExit("frozen protocol does not exist")
        current = build(manifest)
        frozen = json.loads(PROTOCOL.read_text())
        immutable = ["source_identity", "source_file_sha256", "manifest_sha256", "tasks", "questions", "fields", "benchmark_visibility", "agents", "decoding", "isolation", "success_gate", "failure_handling", "claim_boundary"]
        mismatches = [key for key in immutable if current[key] != frozen.get(key)]
        if mismatches:
            raise SystemExit(f"frozen protocol mismatch: {mismatches}")
        print(json.dumps({"status": "PASS", "protocol_id": frozen["protocol_id"], "source_identity": frozen["source_identity"]}, indent=2))
        return 0
    if PROTOCOL.exists():
        raise SystemExit("protocol already exists; use --verify")
    protocol = build(manifest)
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "FROZEN", "path": str(PROTOCOL), "source_identity": protocol["source_identity"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
