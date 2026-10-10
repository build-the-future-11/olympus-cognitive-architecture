from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from olympus.foundry.data_pipeline import (
    DatasetManifestV2,
    prepare_instruction_dataset,
    sha256_bytes,
    verify_dataset_manifest,
)


def _prepare(root: Path) -> tuple[Path, DatasetManifestV2]:
    source = Path(__file__).resolve().parents[1] / "datasets/hermes-smoke/source.jsonl"
    manifest = prepare_instruction_dataset(source, root)
    return root / "manifest.json", manifest


def _seal(path: Path, manifest: DatasetManifestV2) -> None:
    manifest.manifest_sha256 = sha256_bytes(manifest.canonical_bytes(include_hash=False))
    path.write_bytes(manifest.canonical_bytes())


def _records(path: Path, manifest: DatasetManifestV2, split_name: str) -> list[dict[str, Any]]:
    split = next(item for item in manifest.splits if item.name == split_name)
    return [json.loads(line) for line in (path.parent / split.path).read_text().splitlines()]


def _replace_records(
    path: Path,
    manifest: DatasetManifestV2,
    split_name: str,
    records: list[dict[str, Any]],
) -> None:
    split = next(item for item in manifest.splits if item.name == split_name)
    payload = ("\n".join(json.dumps(item) for item in records) + "\n").encode()
    (path.parent / split.path).write_bytes(payload)
    split.sha256 = sha256_bytes(payload)
    split.records = len(records)
    split.categories = dict(Counter(item["category"] for item in records))
    _seal(path, manifest)


def test_valid_prepared_dataset_still_verifies_without_changing_identity(tmp_path: Path) -> None:
    path, manifest = _prepare(tmp_path)
    payload = path.read_bytes()
    assert verify_dataset_manifest(path) == manifest
    assert path.read_bytes() == payload


@pytest.mark.parametrize("missing", ["train", "validation", "test"])
def test_rehashed_manifest_requires_every_split(tmp_path: Path, missing: str) -> None:
    path, manifest = _prepare(tmp_path)
    manifest.splits = [item for item in manifest.splits if item.name != missing]
    retained = sum(item.records for item in manifest.splits)
    manifest.source.record_count = retained
    manifest.quality.input_records = retained
    manifest.quality.accepted_records = retained
    _seal(path, manifest)
    with pytest.raises(ValueError, match="exactly train, validation, and test"):
        verify_dataset_manifest(path)


def test_rehashed_duplicate_ids_are_rejected_across_splits(tmp_path: Path) -> None:
    path, manifest = _prepare(tmp_path)
    training = _records(path, manifest, "train")
    validation = _records(path, manifest, "validation")
    validation[0]["id"] = training[0]["id"]
    _replace_records(path, manifest, "validation", validation)
    with pytest.raises(ValueError, match="duplicate record IDs"):
        verify_dataset_manifest(path)


def test_rehashed_normalized_duplicates_are_rejected(tmp_path: Path) -> None:
    path, manifest = _prepare(tmp_path)
    records = _records(path, manifest, "train")
    records[1]["prompt"] = "  " + records[0]["prompt"].upper() + "  "
    records[1]["response"] = records[0]["response"].upper()
    _replace_records(path, manifest, "train", records)
    with pytest.raises(ValueError, match="normalized duplicates"):
        verify_dataset_manifest(path)


def test_rehashed_cross_split_contamination_is_rejected(tmp_path: Path) -> None:
    path, manifest = _prepare(tmp_path)
    shared = "Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune"
    for split_name in ("train", "test"):
        records = _records(path, manifest, split_name)
        records[0]["prompt"] = shared + " " + split_name
        _replace_records(path, manifest, split_name, records)
    with pytest.raises(ValueError, match="cross-split 8-token contamination"):
        verify_dataset_manifest(path)


def test_rehashed_category_omission_is_rejected(tmp_path: Path) -> None:
    path, manifest = _prepare(tmp_path)
    records = _records(path, manifest, "train")
    records[0]["category"] = records[1]["category"]
    _replace_records(path, manifest, "train", records)
    with pytest.raises(ValueError, match="required category/split coverage"):
        verify_dataset_manifest(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_records", 40),
        ("normalized_duplicates", 1),
        ("split_leakage_hits", 1),
        ("mean_prompt_characters", 1.0),
        ("missing_category_split_pairs", ["test:science"]),
    ],
)
def test_rehashed_quality_report_is_recomputed(
    tmp_path: Path, field: str, value: object
) -> None:
    path, manifest = _prepare(tmp_path)
    setattr(manifest.quality, field, value)
    _seal(path, manifest)
    with pytest.raises(ValueError, match="quality report mismatch"):
        verify_dataset_manifest(path)
