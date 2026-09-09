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
DEFAULT_PROTOCOL = ROOT / "configs" / "local_ood_protocol.json"
V4_PROTOCOL_ID = "PANTHEON-COREBENCH-OOD-LOCAL-20260903-V4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files() -> list[Path]:
    roots = [
        ROOT / "src",
        ROOT / "scripts",
        ROOT / "tests",
        ROOT / "analysis",
        ROOT / "poststudy",
    ]
    files = [
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"}
    ]
    files.extend([ROOT / "pyproject.toml", ROOT / "docs" / "EXTERNAL_VALIDATION_PROTOCOL.md"])
    return sorted(files)


def source_identity() -> tuple[str, dict[str, str]]:
    per_file = {str(path.relative_to(ROOT)): sha256(path) for path in source_files()}
    canonical = json.dumps(per_file, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), per_file


def build(
    manifest_path: Path,
    *,
    protocol_id: str = V4_PROTOCOL_ID,
    schema_version: int = 4,
) -> dict[str, object]:
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as response:
        installed = {item["name"]: item["digest"] for item in json.load(response).get("models", [])}
    models = ["qwen3:0.6b", "llama3.2:1b"]
    missing = [model for model in models if model not in installed]
    if missing:
        raise SystemExit(f"missing frozen model(s): {missing}")
    identity, files = source_identity()
    return {
        "schema_version": schema_version,
        "protocol_id": protocol_id,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "source_identity": identity,
        "source_file_sha256": files,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256(manifest_path),
        "tasks": len(rows),
        "questions": sum(
            len(json.loads((ROOT / row["canonical_answers"]).read_text())) for row in rows
        ),
        "fields": sorted({row["field"] for row in rows}),
        "benchmark_visibility": sorted({row["benchmark_visibility"] for row in rows}),
        "agents": [
            {"provider": "ollama", "model": model, "digest": installed[model]} for model in models
        ],
        "decoding": {
            "temperature": 0,
            "max_turns": 20,
            "max_completion_tokens_per_turn": 768,
            "provider_timeout_seconds": 900,
        },
        "isolation": {
            "workspace": "distinct ephemeral /private/tmp extraction per agent",
            "filesystem": "macOS sandbox-exec denies /Users/ryan and /Volumes/PRO-BLADE",
            "network": "denied inside agent shell",
            "canonical_answers_exposed_to_agent": False,
        },
        "success_gate": {
            "minimum_tasks": 15,
            "minimum_questions": 20,
            "minimum_fields": 2,
            "distinct_agent_identities": True,
            "all_reports_required": True,
            "provider_errors_allowed": False,
        },
        "failure_handling": {
            "one_bounded_corrective_turn_after_missing_report": True,
            "missing_or_invalid_exact_question_keys_normalized_to_null": True,
            "joint_null_answers_count_as_agreement": False,
            "agent_submission_origin_retained": True,
        },
        "claim_boundary": (
            "Local small-model OOD reproduction evidence; does not estimate frontier-agent "
            "or human-adjudication performance."
        ),
        "python": sys.version,
        "platform": platform.platform(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="external/corebench/manifest.jsonl")
    parser.add_argument("--protocol", type=Path, default=Path("configs/local_ood_protocol.json"))
    parser.add_argument("--protocol-id")
    parser.add_argument("--schema-version", type=int)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest = (ROOT / args.manifest).resolve()
    protocol = args.protocol if args.protocol.is_absolute() else (ROOT / args.protocol)
    if args.verify:
        if not protocol.exists():
            raise SystemExit("frozen protocol does not exist")
        frozen = json.loads(protocol.read_text())
        current_identity, current_files = source_identity()
        frozen_files = frozen.get("source_file_sha256", {})
        if current_identity != frozen.get("source_identity") or current_files != frozen_files:
            current_names = set(current_files)
            frozen_names = set(frozen_files)
            changed = sorted(
                name
                for name in current_names & frozen_names
                if current_files[name] != frozen_files[name]
            )
            result = {
                "status": "V4_SOURCE_DRIFT_REQUIRES_VERSIONED_PROTOCOL"
                if frozen.get("protocol_id") == V4_PROTOCOL_ID
                else "FROZEN_PROTOCOL_SOURCE_DRIFT",
                "protocol_id": frozen.get("protocol_id"),
                "frozen_source_identity": frozen.get("source_identity"),
                "current_source_identity": current_identity,
                "changed_files": changed,
                "missing_files": sorted(frozen_names - current_names),
                "new_files": sorted(current_names - frozen_names),
                "next_step": "Freeze a new versioned protocol before any confirmatory execution.",
            }
            print(json.dumps(result, indent=2), file=sys.stderr)
            return 4
        current = build(
            manifest,
            protocol_id=str(frozen.get("protocol_id", "UNKNOWN")),
            schema_version=int(frozen.get("schema_version", 0)),
        )
        immutable = [
            "source_identity",
            "source_file_sha256",
            "manifest_sha256",
            "tasks",
            "questions",
            "fields",
            "benchmark_visibility",
            "agents",
            "decoding",
            "isolation",
            "success_gate",
            "failure_handling",
            "claim_boundary",
        ]
        mismatches = [key for key in immutable if current[key] != frozen.get(key)]
        if mismatches:
            print(
                json.dumps(
                    {
                        "status": "FROZEN_PROTOCOL_MISMATCH",
                        "protocol_id": frozen.get("protocol_id"),
                        "mismatches": mismatches,
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 3
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "protocol_id": frozen["protocol_id"],
                    "source_identity": frozen["source_identity"],
                },
                indent=2,
            )
        )
        return 0
    if protocol.exists():
        raise SystemExit("protocol already exists; use --verify")
    is_default_v4 = protocol.resolve() == DEFAULT_PROTOCOL.resolve()
    if is_default_v4:
        raise SystemExit(
            "refusing to recreate the reserved frozen V4 path; choose a new --protocol "
            "and --protocol-id"
        )
    protocol_id = args.protocol_id
    if protocol_id is None:
        raise SystemExit("--protocol-id is required when freezing a new versioned protocol")
    if protocol_id == V4_PROTOCOL_ID:
        raise SystemExit("the frozen V4 protocol identifier is reserved and cannot be reused")
    schema_version = args.schema_version or 5
    frozen_protocol = build(
        manifest,
        protocol_id=protocol_id,
        schema_version=schema_version,
    )
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text(json.dumps(frozen_protocol, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "path": str(protocol),
                "protocol_id": protocol_id,
                "source_identity": frozen_protocol["source_identity"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
