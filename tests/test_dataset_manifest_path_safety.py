from __future__ import annotations

from pathlib import Path

import pytest

from olympus.foundry.data_pipeline import (
    prepare_instruction_dataset,
    sha256_bytes,
    verify_dataset_manifest,
)


def _source() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets/hermes-smoke/source.jsonl"


def _rewrite_train_path(manifest_path: Path, replacement: str) -> None:
    manifest = verify_dataset_manifest(manifest_path)
    rewritten = [
        split.model_copy(update={"path": replacement}) if split.name == "train" else split
        for split in manifest.splits
    ]
    forged = manifest.model_copy(update={"splits": rewritten, "manifest_sha256": None})
    forged.manifest_sha256 = sha256_bytes(forged.canonical_bytes(include_hash=False))
    manifest_path.write_bytes(forged.canonical_bytes())


def test_manifest_rejects_parent_traversal_even_when_bytes_and_hash_match(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared"
    prepare_instruction_dataset(_source(), prepared)
    manifest_path = prepared / "manifest.json"

    outside = tmp_path / "outside"
    outside.mkdir()
    outside_train = outside / "train.jsonl"
    outside_train.write_bytes((prepared / "splits/train.jsonl").read_bytes())

    _rewrite_train_path(manifest_path, "../outside/train.jsonl")

    with pytest.raises(
        ValueError, match="split path must stay relative to the manifest root: train"
    ):
        verify_dataset_manifest(manifest_path)


def test_manifest_rejects_absolute_split_path_even_when_bytes_and_hash_match(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared"
    prepare_instruction_dataset(_source(), prepared)
    manifest_path = prepared / "manifest.json"

    outside = tmp_path / "outside"
    outside.mkdir()
    outside_train = outside / "train.jsonl"
    outside_train.write_bytes((prepared / "splits/train.jsonl").read_bytes())

    _rewrite_train_path(manifest_path, str(outside_train.resolve()))

    with pytest.raises(
        ValueError, match="split path must stay relative to the manifest root: train"
    ):
        verify_dataset_manifest(manifest_path)


def test_manifest_rejects_symlink_escape_outside_root(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepare_instruction_dataset(_source(), prepared)
    manifest_path = prepared / "manifest.json"

    outside = tmp_path / "outside"
    outside.mkdir()
    outside_train = outside / "train.jsonl"
    outside_train.write_bytes((prepared / "splits/train.jsonl").read_bytes())
    (prepared / "escape").symlink_to(outside, target_is_directory=True)

    _rewrite_train_path(manifest_path, "escape/train.jsonl")

    with pytest.raises(ValueError, match="split path escapes manifest root: train"):
        verify_dataset_manifest(manifest_path)
