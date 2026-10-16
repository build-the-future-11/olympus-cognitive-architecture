"""Validate a pinned EditCLEVR archive snapshot without extracting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from olympus.data.editclevr import PAYLOAD_FIELDS as PAYLOAD_FIELDS
from olympus.data.editclevr import iter_split_records as iter_split_records

ERROR_SAMPLE_LIMIT = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _member_is_safe(
    member: tarfile.TarInfo,
    expected_root: str | None,
) -> tuple[bool, str | None]:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        return False, "path traversal"
    if not path.parts:
        return False, "empty member path"
    if expected_root is not None and path.parts[0] != expected_root:
        return False, f"unexpected root (expected {expected_root})"
    if member.issym() or member.islnk():
        return False, "link member"
    if member.isdev() or member.isfifo():
        return False, "special member"
    if not (member.isfile() or member.isdir()):
        return False, "unsupported member type"
    return True, None


def inspect_archive(path: Path, expected_root: str | None) -> tuple[dict[str, Any], set[str]]:
    member_names: set[str] = set()
    suffix_counts: Counter[str] = Counter()
    unsafe: list[dict[str, str]] = []
    unsafe_count = 0
    file_count = 0
    directory_count = 0
    uncompressed_bytes = 0

    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            safe, reason = _member_is_safe(member, expected_root)
            if not safe:
                unsafe_count += 1
                if len(unsafe) < ERROR_SAMPLE_LIMIT:
                    unsafe.append({"member": member.name, "reason": reason or "unsafe"})
                continue
            if member.isdir():
                directory_count += 1
                continue
            file_count += 1
            uncompressed_bytes += member.size
            member_names.add(member.name)
            suffix_counts[PurePosixPath(member.name).suffix.lower() or "<none>"] += 1

    summary = {
        "archive": path.name,
        "compressed_bytes": path.stat().st_size,
        "directory_count": directory_count,
        "file_count": file_count,
        "member_types": dict(sorted(suffix_counts.items())),
        "safe": unsafe_count == 0,
        "sha256": _sha256(path),
        "uncompressed_bytes": uncompressed_bytes,
        "unsafe_member_count": unsafe_count,
        "unsafe_member_samples": unsafe,
    }
    return summary, member_names


def _archive_member_sha256(path: Path, member_name: str) -> str:
    digest = hashlib.sha256()
    with tarfile.open(path, mode="r:gz") as archive:
        member = archive.getmember(member_name)
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"{member_name} is not a regular file in {path}")
        with handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_snapshot(
    archive_dir: Path,
    manifest_path: Path,
    splits_path: Path,
) -> dict[str, Any]:
    manifest = _load_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest.schema_version must equal 1")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("manifest.files must be an object")

    archive_roots = {
        "editclevr_atomic_id.tar.gz": "atomic_id",
        "editclevr_cogent_ood.tar.gz": "cogent_ood",
        "editclevr_hard_distractor.tar.gz": "hard_distractor",
        "editclevr_no_edit.tar.gz": "no_edit",
    }
    archive_reports: dict[str, Any] = {}
    members_by_suite: dict[str, set[str]] = {}
    errors: list[str] = []

    splits_archive_name = "editclevr_splits.tar.gz"
    splits_archive_path = archive_dir / splits_archive_name
    splits_expected = files.get(splits_archive_name)
    if not isinstance(splits_expected, dict):
        errors.append(f"manifest is missing {splits_archive_name}")
    elif not splits_archive_path.is_file():
        errors.append(f"payload is missing {splits_archive_name}")
    else:
        split_report, split_members = inspect_archive(splits_archive_path, None)
        expected_split_members = {"phase1_manifest.json", "splits.json", "validation_report.json"}
        split_report["expected_sha256"] = splits_expected.get("sha256")
        split_report["expected_size_bytes"] = splits_expected.get("size_bytes")
        split_report["hash_matches"] = split_report["sha256"] == splits_expected.get("sha256")
        split_report["size_matches"] = (
            split_report["compressed_bytes"] == splits_expected.get("size_bytes")
        )
        split_report["expected_members"] = sorted(expected_split_members)
        split_report["members_match"] = split_members == expected_split_members
        split_report["staged_splits_sha256"] = _sha256(splits_path)
        split_report["archived_splits_sha256"] = (
            _archive_member_sha256(splits_archive_path, "splits.json")
            if "splits.json" in split_members
            else None
        )
        split_report["staged_splits_match"] = (
            split_report["staged_splits_sha256"] == split_report["archived_splits_sha256"]
        )
        archive_reports["splits"] = split_report
        if not split_report["safe"]:
            errors.append(f"{splits_archive_name} contains unsafe archive members")
        if not split_report["hash_matches"]:
            errors.append(f"{splits_archive_name} SHA-256 does not match the pinned manifest")
        if not split_report["size_matches"]:
            errors.append(f"{splits_archive_name} size does not match the pinned manifest")
        if not split_report["members_match"]:
            errors.append(f"{splits_archive_name} has unexpected or missing members")
        if not split_report["staged_splits_match"]:
            errors.append("staged split registry does not match the pinned split archive")

    for filename, root in archive_roots.items():
        path = archive_dir / filename
        expected = files.get(filename)
        if not isinstance(expected, dict):
            errors.append(f"manifest is missing {filename}")
            continue
        if not path.is_file():
            errors.append(f"payload is missing {filename}")
            continue
        report, members = inspect_archive(path, root)
        report["expected_sha256"] = expected.get("sha256")
        report["expected_size_bytes"] = expected.get("size_bytes")
        report["hash_matches"] = report["sha256"] == expected.get("sha256")
        report["size_matches"] = report["compressed_bytes"] == expected.get("size_bytes")
        archive_reports[root] = report
        members_by_suite[root] = members
        if not report["safe"]:
            errors.append(f"{filename} contains unsafe archive members")
        if not report["hash_matches"]:
            errors.append(f"{filename} SHA-256 does not match the pinned manifest")
        if not report["size_matches"]:
            errors.append(f"{filename} size does not match the pinned manifest")

    split_counts: Counter[str] = Counter()
    suite_counts: Counter[str] = Counter()
    pair_ids: set[str] = set()
    referenced_paths = 0
    missing_paths: list[dict[str, str]] = []
    missing_path_count = 0
    invalid_records: list[str] = []
    invalid_record_count = 0

    for split_name, index, raw_record in iter_split_records(splits_path):
        split_counts[split_name] += 1
        if not isinstance(raw_record, dict):
            invalid_record_count += 1
            if len(invalid_records) < ERROR_SAMPLE_LIMIT:
                invalid_records.append(f"{split_name}[{index}] is not an object")
            continue
        pair_id = raw_record.get("pair_id")
        suite = raw_record.get("suite")
        if not isinstance(pair_id, str) or not pair_id:
            invalid_record_count += 1
            if len(invalid_records) < ERROR_SAMPLE_LIMIT:
                invalid_records.append(f"{split_name}[{index}] has no valid pair_id")
            continue
        if pair_id in pair_ids:
            invalid_record_count += 1
            if len(invalid_records) < ERROR_SAMPLE_LIMIT:
                invalid_records.append(f"duplicate pair_id: {pair_id}")
        pair_ids.add(pair_id)
        if not isinstance(suite, str) or suite not in archive_roots.values():
            invalid_record_count += 1
            if len(invalid_records) < ERROR_SAMPLE_LIMIT:
                invalid_records.append(f"{pair_id} has unknown suite {suite!r}")
            continue
        suite_counts[suite] += 1
        members = members_by_suite.get(suite, set())
        for field in PAYLOAD_FIELDS:
            value = raw_record.get(field)
            if not isinstance(value, str) or not value:
                invalid_record_count += 1
                if len(invalid_records) < ERROR_SAMPLE_LIMIT:
                    invalid_records.append(f"{pair_id} has no valid {field}")
                continue
            referenced_paths += 1
            if value not in members:
                missing_path_count += 1
                if len(missing_paths) < ERROR_SAMPLE_LIMIT:
                    missing_paths.append({"field": field, "pair_id": pair_id, "path": value})

    if invalid_record_count:
        errors.append(f"split registry has {invalid_record_count} invalid records or fields")
    if missing_path_count:
        errors.append(f"payload is missing {missing_path_count} split-referenced files")

    return {
        "dataset": manifest.get("dataset"),
        "revision": manifest.get("revision"),
        "status": "PASS" if not errors else "FAIL",
        "archive_reports": archive_reports,
        "split_registry": {
            "invalid_record_count": invalid_record_count,
            "invalid_record_samples": invalid_records,
            "missing_path_count": missing_path_count,
            "missing_path_samples": missing_paths,
            "pair_id_count": len(pair_ids),
            "referenced_path_count": referenced_paths,
            "split_counts": dict(sorted(split_counts.items())),
            "suite_counts": dict(sorted(suite_counts.items())),
        },
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = validate_snapshot(args.archive_dir, args.manifest, args.splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
