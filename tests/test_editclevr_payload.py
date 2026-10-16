from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from olympus.foundry import resources
from olympus.foundry.resources import MemorySnapshot
from scripts import materialize_editclevr_payload
from scripts.materialize_editclevr_payload import _admitted_snapshot, materialize_snapshot
from scripts.validate_editclevr_payload import (
    inspect_archive,
    iter_split_records,
    validate_snapshot,
)


def _write_archive(path: Path, root: str, files: dict[str, bytes]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for relative, content in files.items():
            info = tarfile.TarInfo(f"{root}/{relative}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def _write_flat_archive(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_inspect_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))

    report, members = inspect_archive(archive_path, "atomic_id")

    assert report["safe"] is False
    assert report["unsafe_member_count"] == 1
    assert report["unsafe_member_samples"][0]["reason"] == "path traversal"
    assert members == set()


def test_validate_snapshot_cross_checks_hashes_and_split_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    roots = {
        "editclevr_atomic_id.tar.gz": "atomic_id",
        "editclevr_cogent_ood.tar.gz": "cogent_ood",
        "editclevr_hard_distractor.tar.gz": "hard_distractor",
        "editclevr_no_edit.tar.gz": "no_edit",
    }
    payload_paths = {
        "before_image": "case/before/images/before.png",
        "after_image": "case/after/images/after.png",
        "before_scene_json": "case/before/scenes/before.json",
        "after_scene_json": "case/after/scenes/after.json",
        "instance_masks_before": "case/before/masks_npz/before.npz",
        "instance_masks_after": "case/after/masks_npz/after.npz",
    }
    manifest_files: dict[str, dict[str, int | str]] = {}
    for filename, root in roots.items():
        path = archive_dir / filename
        _write_archive(path, root, {value: b"x" for value in payload_paths.values()})
        report, _ = inspect_archive(path, root)
        manifest_files[filename] = {
            "sha256": report["sha256"],
            "size_bytes": report["compressed_bytes"],
        }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "test/EditCLEVR",
                "revision": "abc",
                "files": manifest_files,
            }
        ),
        encoding="utf-8",
    )
    records = []
    for index, root in enumerate(roots.values()):
        record = {"pair_id": f"pair-{index}", "suite": root}
        record.update({field: f"{root}/{value}" for field, value in payload_paths.items()})
        records.append(record)
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps({"test": records}), encoding="utf-8")
    split_archive = archive_dir / "editclevr_splits.tar.gz"
    _write_flat_archive(
        split_archive,
        {
            "splits.json": splits_path.read_bytes(),
            "validation_report.json": b"{}",
            "phase1_manifest.json": b"{}",
        },
    )
    split_report, _ = inspect_archive(split_archive, None)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["editclevr_splits.tar.gz"] = {
        "sha256": split_report["sha256"],
        "size_bytes": split_report["compressed_bytes"],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_snapshot(archive_dir, manifest_path, splits_path)

    assert report["status"] == "PASS"
    assert report["archive_reports"]["splits"]["staged_splits_match"] is True
    assert report["split_registry"]["pair_id_count"] == 4
    assert report["split_registry"]["referenced_path_count"] == 24

    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps(report), encoding="utf-8")
    gib = 1024**3
    safe = MemorySnapshot(16 * gib, 8 * gib, 4 * gib, 1 * gib, 500)
    monkeypatch.setattr(resources, "memory_snapshot", lambda: safe)
    monkeypatch.setattr(materialize_editclevr_payload, "memory_snapshot", lambda: safe)
    output = tmp_path / "payload"
    receipt = tmp_path / "materialization.json"
    materialized = materialize_snapshot(
        archive_dir,
        manifest_path,
        validation_path,
        output,
        receipt,
    )
    assert materialized["status"] == "PASS"
    assert materialized["payload_file_count"] == 27
    assert (output / "atomic_id/case/before/images/before.png").is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_snapshot(archive_dir, manifest_path, validation_path, output, receipt)

    splits_path.write_text(json.dumps({"test": []}), encoding="utf-8")
    tampered = validate_snapshot(archive_dir, manifest_path, splits_path)
    assert tampered["status"] == "FAIL"
    assert "staged split registry does not match" in tampered["errors"][0]


def test_split_registry_is_parsed_incrementally_across_small_chunks(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps({"train": [{"pair_id": "a"}, {"pair_id": "b"}], "test": []}),
        encoding="utf-8",
    )

    assert list(iter_split_records(path, chunk_size=7)) == [
        ("train", 0, {"pair_id": "a"}),
        ("train", 1, {"pair_id": "b"}),
    ]


def test_split_registry_rejects_trailing_content(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text('{"train": []} false', encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected content"):
        list(iter_split_records(path, chunk_size=4))


def test_materializer_rechecks_resource_ceiling_between_archives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3
    governor = resources.ResourceGovernor(tmp_path / "materialize.lock")
    over_swap = MemorySnapshot(16 * gib, 8 * gib, 10 * gib, 9 * gib + 1, 500)
    monkeypatch.setattr(materialize_editclevr_payload, "memory_snapshot", lambda: over_swap)

    with pytest.raises(RuntimeError, match="after archive: swap utilization"):
        _admitted_snapshot(governor, "after archive")
