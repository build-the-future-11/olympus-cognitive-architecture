"""Atomically materialize an admitted EditCLEVR archive snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from olympus.foundry.resources import ResourceGovernor, memory_snapshot

ARCHIVES = (
    "editclevr_atomic_id.tar.gz",
    "editclevr_cogent_ood.tar.gz",
    "editclevr_hard_distractor.tar.gz",
    "editclevr_no_edit.tar.gz",
    "editclevr_splits.tar.gz",
)


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


def _atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _admitted_archives(
    archive_dir: Path,
    manifest_path: Path,
    validation_path: Path,
) -> tuple[dict[str, dict[str, Any]], int]:
    manifest = _load_object(manifest_path)
    validation = _load_object(validation_path)
    if validation.get("status") != "PASS" or validation.get("errors") != []:
        raise ValueError("EditCLEVR validation report is not an error-free PASS")
    if validation.get("revision") != manifest.get("revision"):
        raise ValueError("validation and upstream manifest revisions differ")
    files = manifest.get("files")
    reports = validation.get("archive_reports")
    if not isinstance(files, dict) or not isinstance(reports, dict):
        raise ValueError("manifest or validation archive maps are invalid")
    expected_files = 0
    bound: dict[str, dict[str, Any]] = {}
    for name in ARCHIVES:
        expected = files.get(name)
        key = "splits" if name == "editclevr_splits.tar.gz" else name.removeprefix(
            "editclevr_"
        ).removesuffix(".tar.gz")
        report = reports.get(key)
        if not isinstance(expected, dict) or not isinstance(report, dict):
            raise ValueError(f"missing admission metadata for {name}")
        if not all(report.get(flag) is True for flag in ("safe", "hash_matches", "size_matches")):
            raise ValueError(f"archive is not admitted: {name}")
        path = archive_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"archive is missing or is a symlink: {name}")
        actual_size = path.stat().st_size
        actual_hash = _sha256(path)
        if actual_size != expected.get("size_bytes") or actual_hash != expected.get("sha256"):
            raise ValueError(f"archive no longer matches its pinned identity: {name}")
        file_count = report.get("file_count")
        if not isinstance(file_count, int) or file_count < 1:
            raise ValueError(f"validation report has no file count for {name}")
        expected_files += file_count
        bound[name] = {"sha256": actual_hash, "size_bytes": actual_size, "file_count": file_count}
    return bound, expected_files


def _admitted_snapshot(governor: ResourceGovernor, stage: str) -> dict[str, int | float]:
    snapshot = memory_snapshot()
    if snapshot.available_bytes < governor.min_available_bytes:
        raise RuntimeError(
            f"{stage}: available memory {snapshot.available_bytes} is below the "
            f"{governor.min_available_bytes}-byte safety floor"
        )
    if snapshot.swap_fraction > governor.max_swap_fraction:
        raise RuntimeError(
            f"{stage}: swap utilization {snapshot.swap_fraction:.1%} exceeds the "
            f"{governor.max_swap_fraction:.1%} safety ceiling"
        )
    return snapshot.as_dict()


def materialize_snapshot(
    archive_dir: Path,
    manifest_path: Path,
    validation_path: Path,
    output_dir: Path,
    receipt_path: Path,
    *,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite materialized payload: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.partial-{uuid4().hex}")
    started = time.time()
    governor = ResourceGovernor(lock_path or output.parent / ".editclevr-materialize.lock")
    try:
        with governor:
            archives, expected_files = _admitted_archives(
                archive_dir.resolve(),
                manifest_path.resolve(),
                validation_path.resolve(),
            )
            staging.mkdir(parents=False)
            stage_memory: dict[str, dict[str, int | float]] = {}
            for name in ARCHIVES:
                with tarfile.open(archive_dir / name, mode="r:gz") as archive:
                    archive.extractall(staging, filter="data")
                stage_memory[name] = _admitted_snapshot(governor, f"after {name}")
            actual_files = sum(1 for path in staging.rglob("*") if path.is_file())
            if actual_files != expected_files:
                raise RuntimeError(
                    f"materialized file count {actual_files} does not match {expected_files}"
                )
            final_memory = _admitted_snapshot(governor, "before atomic promotion")
            marker = {
                "schema_version": 1,
                "dataset": "torux/EditCLEVR",
                "revision": _load_object(manifest_path).get("revision"),
                "validation_sha256": _sha256(validation_path),
                "archives": archives,
                "payload_file_count": actual_files,
            }
            _atomic_json(staging / ".olympus-materialization.json", marker)
            staging.replace(output)
            result = {
                **marker,
                "status": "PASS",
                "output_dir": str(output),
                "marker_sha256": _sha256(output / ".olympus-materialization.json"),
                "elapsed_seconds": time.time() - started,
                "preflight_memory": governor.preflight.as_dict() if governor.preflight else None,
                "stage_memory": stage_memory,
                "final_memory": final_memory,
            }
            _atomic_json(receipt_path.resolve(), result)
            return result
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "retained_staging_dir": str(staging) if staging.exists() else None,
            "elapsed_seconds": time.time() - started,
        }
        _atomic_json(receipt_path.resolve(), failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    result = materialize_snapshot(
        args.archive_dir,
        args.manifest,
        args.validation,
        args.output,
        args.receipt,
        lock_path=args.lock,
    )
    print(json.dumps({"output": result["output_dir"], "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
