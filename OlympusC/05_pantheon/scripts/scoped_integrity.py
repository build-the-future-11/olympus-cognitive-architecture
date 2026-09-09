#!/usr/bin/env python3
"""Generate or validate a scoped Pantheon evidence-integrity manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "audit" / "SCOPED_EVIDENCE_INTEGRITY_V1.json"
STATUS = "CURRENT_SCOPED_EVIDENCE_INTEGRITY"
GROUP_SPECS: dict[str, tuple[tuple[str, ...], int]] = {
    "historical_manifest_and_status": (
        (
            "audit/FINAL_INTEGRITY_MANIFEST.json",
            "audit/FINAL_INTEGRITY_MANIFEST_STATUS.md",
        ),
        2,
    ),
    "frozen_v4_machine_records": (
        (
            "configs/local_ood_protocol.json",
            "PUBLICATION_READINESS.json",
        ),
        2,
    ),
    "frozen_v4_benchmark_definition": (
        (
            "external/corebench/manifest.jsonl",
            "external/corebench/core_ood.json",
            "external/corebench/canonical/*.json",
        ),
        19,
    ),
    "frozen_v4_raw_external_runs": (("runs/external/**/*",), 119),
    "derived_external_evidence": (("results/external/*",), 7),
    "current_poststudy_gates": (
        (
            "poststudy/STRUCTURAL_READINESS.json",
            "poststudy/STRUCTURAL_READINESS.md",
            "poststudy/SCIENTIFIC_READINESS.json",
            "poststudy/SCIENTIFIC_READINESS.md",
        ),
        4,
    ),
    "v5_candidate_source_snapshot": (
        ("configs/local_ood_protocol_v5_source_manifest.json",),
        1,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_groups(
    root: Path,
    specs: dict[str, tuple[tuple[str, ...], int]] = GROUP_SPECS,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name, (patterns, expected_count) in specs.items():
        paths: set[str] = set()
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file():
                    paths.add(str(path.relative_to(root)))
        ordered = sorted(paths)
        if len(ordered) != expected_count:
            raise ValueError(
                f"scope group {name!r} resolved {len(ordered)} files; expected {expected_count}"
            )
        groups[name] = ordered
    return groups


def build_manifest(
    root: Path = ROOT,
    specs: dict[str, tuple[tuple[str, ...], int]] = GROUP_SPECS,
) -> dict[str, object]:
    groups = resolve_groups(root, specs)
    paths = sorted({path for group_paths in groups.values() for path in group_paths})
    files = {
        relative: {
            "bytes": (root / relative).stat().st_size,
            "sha256": sha256(root / relative),
        }
        for relative in paths
    }
    return {
        "schema_version": 1,
        "status": STATUS,
        "algorithm": "sha256",
        "file_count": len(files),
        "groups": groups,
        "files": files,
        "claim_boundary": (
            "Integrity applies only to the explicitly listed V4 OOD inputs/raw runs/derived "
            "evidence, current post-study gates, and the non-executed V5 source snapshot. "
            "It is not a whole-tree immutability or scientific-validity claim."
        ),
    }


def validate_manifest(
    manifest: dict[str, object],
    root: Path = ROOT,
    specs: dict[str, tuple[tuple[str, ...], int]] = GROUP_SPECS,
) -> dict[str, object]:
    expected = build_manifest(root, specs)
    expected_files = expected["files"]
    recorded_files = manifest.get("files", {})
    assert isinstance(expected_files, dict)
    if not isinstance(recorded_files, dict):
        recorded_files = {}
    missing = sorted(set(expected_files) - set(recorded_files))
    unexpected = sorted(set(recorded_files) - set(expected_files))
    mismatched = sorted(
        path
        for path in set(expected_files) & set(recorded_files)
        if expected_files[path] != recorded_files[path]
    )
    metadata_matches = all(
        manifest.get(key) == expected[key]
        for key in [
            "schema_version",
            "status",
            "algorithm",
            "file_count",
            "groups",
            "claim_boundary",
        ]
    )
    matched = len(expected_files) - len(missing) - len(mismatched)
    passed = metadata_matches and not missing and not unexpected and not mismatched
    return {
        "status": "PASS" if passed else "FAIL",
        "matched_files": matched,
        "expected_files": len(expected_files),
        "metadata_matches": metadata_matches,
        "missing_files": missing,
        "unexpected_files": unexpected,
        "mismatched_files": mismatched,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("audit/SCOPED_EVIDENCE_INTEGRITY_V1.json")
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    if args.write:
        result = build_manifest()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(manifest_path)
        print(
            json.dumps(
                {"status": "WRITTEN", "path": str(manifest_path), "files": result["file_count"]},
                indent=2,
            )
        )
        return 0
    if not manifest_path.exists():
        print(json.dumps({"status": "FAIL", "missing_manifest": str(manifest_path)}, indent=2))
        return 3
    result = validate_manifest(json.loads(manifest_path.read_text()))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
