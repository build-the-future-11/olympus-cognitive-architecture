#!/usr/bin/env python3
"""Write or validate a deterministic V5 candidate-source snapshot.

This is deliberately not a frozen protocol and not evidence that V5 ran.  A
confirmatory execution requires a separately frozen protocol containing model
digests and a freeze time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from write_local_ood_protocol import ROOT, V4_PROTOCOL_ID, sha256, source_identity

OUTPUT = ROOT / "configs" / "local_ood_protocol_v5_source_manifest.json"
STATUS = "DRAFT_SOURCE_SNAPSHOT_NOT_FROZEN_NOT_EXECUTED"


def build(manifest_path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    identity, files = source_identity()
    return {
        "schema_version": 1,
        "status": STATUS,
        "candidate_protocol_id": "PANTHEON-COREBENCH-OOD-LOCAL-V5",
        "predecessor_protocol_id": V4_PROTOCOL_ID,
        "source_identity": identity,
        "source_file_sha256": files,
        "benchmark_manifest": str(manifest_path.relative_to(ROOT)),
        "benchmark_manifest_sha256": sha256(manifest_path),
        "tasks": len(rows),
        "questions": sum(
            len(json.loads((ROOT / row["canonical_answers"]).read_text())) for row in rows
        ),
        "fields": sorted({row["field"] for row in rows}),
        "benchmark_visibility": sorted({row["benchmark_visibility"] for row in rows}),
        "claim_boundary": (
            "Current-source snapshot only. It is not a protocol freeze, execution record, "
            "or result. Freeze a versioned V5 protocol before confirmatory execution."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("external/corebench/manifest.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("configs/local_ood_protocol_v5_source_manifest.json")
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    expected = build(manifest_path)
    payload = json.dumps(expected, indent=2, sort_keys=True) + "\n"
    if args.write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload)
        print(
            json.dumps(
                {
                    "status": STATUS,
                    "path": str(output_path),
                    "files": len(expected["source_file_sha256"]),
                },
                indent=2,
            )
        )
        return 0
    if not output_path.exists():
        print(json.dumps({"status": "MISSING", "path": str(output_path)}, indent=2))
        return 3
    actual = json.loads(output_path.read_text())
    if actual != expected:
        print(
            json.dumps(
                {
                    "status": "STALE_V5_SOURCE_MANIFEST",
                    "path": str(output_path),
                    "expected_source_identity": expected["source_identity"],
                    "recorded_source_identity": actual.get("source_identity"),
                },
                indent=2,
            )
        )
        return 3
    print(
        json.dumps(
            {
                "status": "PASS_DRAFT_NOT_FROZEN_NOT_EXECUTED",
                "path": str(output_path),
                "matched_source_files": len(expected["source_file_sha256"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
