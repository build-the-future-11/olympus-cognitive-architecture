from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from olympus.foundry.data_pipeline import (
    DatasetManifestV2,
    prepare_instruction_dataset,
    sha256_bytes,
    verify_dataset_manifest,
)


def _source() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets/hermes-smoke/source.jsonl"


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    prepare_instruction_dataset(_source(), tmp_path / "dataset")
    return tmp_path / "dataset/manifest.json"


def _save_manifest(path: Path, payload: dict[str, Any]) -> None:
    manifest = DatasetManifestV2.model_validate(payload)
    manifest.manifest_sha256 = sha256_bytes(manifest.canonical_bytes(include_hash=False))
    path.write_bytes(manifest.canonical_bytes())


def _rewrite_records(
    path: Path, mutate: Callable[[list[dict[str, Any]]], None]
) -> None:
    """Recompute byte identities and counts, so hashes alone cannot reject a defect."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for split in manifest["splits"]
        for line in (path.parent / split["path"]).read_text(encoding="utf-8").splitlines()
    ]
    mutate(records)
    for split in manifest["splits"]:
        selected = [record for record in records if record["split"] == split["name"]]
        payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in selected).encode()
        (path.parent / split["path"]).write_bytes(payload)
        split["sha256"] = sha256_bytes(payload)
        split["records"] = len(selected)
        split["categories"] = dict(Counter(record["category"] for record in selected))
    manifest["source"]["record_count"] = len(records)
    manifest["quality"].update(
        input_records=len(records),
        accepted_records=len(records),
        mean_prompt_characters=sum(len(record["prompt"]) for record in records) / len(records),
        mean_response_characters=sum(len(record["response"]) for record in records) / len(records),
    )
    _save_manifest(path, manifest)


def test_prepared_manifest_is_unchanged_and_split_order_is_irrelevant(prepared: Path) -> None:
    original = verify_dataset_manifest(prepared)
    payload = original.model_dump(mode="json")
    payload["splits"].reverse()
    _save_manifest(prepared, payload)
    reordered = verify_dataset_manifest(prepared)
    assert reordered.quality == original.quality
    assert {split.name for split in reordered.splits} == {"train", "validation", "test"}


@pytest.mark.parametrize("missing", ["train", "validation", "test"])
def test_rehashed_manifest_requires_every_split(prepared: Path, missing: str) -> None:
    payload = json.loads(prepared.read_text(encoding="utf-8"))
    payload["splits"] = [split for split in payload["splits"] if split["name"] != missing]
    count = sum(split["records"] for split in payload["splits"])
    payload["source"]["record_count"] = count
    payload["quality"]["accepted_records"] = count
    _save_manifest(prepared, payload)
    with pytest.raises(ValueError, match="requires train, validation, and test"):
        verify_dataset_manifest(prepared)


@pytest.mark.parametrize("cross_split", [False, True])
def test_rehashed_manifest_rejects_duplicate_record_ids(
    prepared: Path, cross_split: bool
) -> None:
    def mutate(records: list[dict[str, Any]]) -> None:
        records[12 if cross_split else 1]["id"] = records[0]["id"]

    _rewrite_records(prepared, mutate)
    with pytest.raises(ValueError, match="duplicate record IDs"):
        verify_dataset_manifest(prepared)


@pytest.mark.parametrize("cross_split", [False, True])
def test_rehashed_manifest_rejects_normalized_duplicates(
    prepared: Path, cross_split: bool
) -> None:
    def mutate(records: list[dict[str, Any]]) -> None:
        other = records[12 if cross_split else 1]
        other["prompt"] = "  " + records[0]["prompt"].upper() + "  "
        other["response"] = records[0]["response"].upper()

    _rewrite_records(prepared, mutate)
    with pytest.raises(ValueError, match="normalized duplicates"):
        verify_dataset_manifest(prepared)


def test_rehashed_manifest_rejects_eight_token_cross_split_contamination(
    prepared: Path,
) -> None:
    def mutate(records: list[dict[str, Any]]) -> None:
        shared = "albatross birch cobalt dahlia elm falcon granite hazel"
        records[0]["prompt"] = shared + " training suffix"
        records[12]["prompt"] = shared + " validation suffix"

    _rewrite_records(prepared, mutate)
    with pytest.raises(ValueError, match="cross-split 8-token contamination"):
        verify_dataset_manifest(prepared)


def test_rehashed_manifest_requires_category_coverage(prepared: Path) -> None:
    def mutate(records: list[dict[str, Any]]) -> None:
        records[0]["category"] = records[1]["category"]

    _rewrite_records(prepared, mutate)
    with pytest.raises(ValueError, match="required category/split coverage missing"):
        verify_dataset_manifest(prepared)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_records", 37),
        ("duplicate_ids", 1),
        ("normalized_duplicates", 1),
        ("pii_or_secret_hits", 1),
        ("split_leakage_hits", 1),
        ("missing_category_split_pairs", ["train:code"]),
        ("mean_prompt_characters", 1.0),
        ("mean_response_characters", 1.0),
    ],
)
def test_rehashed_quality_report_must_match_records(
    prepared: Path, field: str, value: Any
) -> None:
    payload = json.loads(prepared.read_text(encoding="utf-8"))
    payload["quality"][field] = value
    _save_manifest(prepared, payload)
    with pytest.raises(ValueError, match="quality report does not match"):
        verify_dataset_manifest(prepared)


def test_preparation_hashes_the_exact_source_bytes_it_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.jsonl"
    original = _source().read_bytes()
    source.write_bytes(original)
    replacement = original.replace(b"Return", b"Provide", 1)
    assert replacement != original
    read_bytes = Path.read_bytes
    reads = 0

    def changing_read(path: Path) -> bytes:
        nonlocal reads
        payload = read_bytes(path)
        if path == source:
            reads += 1
            path.write_bytes(replacement)
        return payload

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    manifest = prepare_instruction_dataset(source, tmp_path / "prepared")
    assert reads == 1
    assert manifest.source.sha256 == sha256_bytes(original)
    observed = {
        item["id"]: item
        for split in manifest.splits
        for item in map(
            json.loads,
            (tmp_path / "prepared" / split.path).read_text(encoding="utf-8").splitlines(),
        )
    }
    assert observed == {item["id"]: item for item in map(json.loads, original.splitlines())}
