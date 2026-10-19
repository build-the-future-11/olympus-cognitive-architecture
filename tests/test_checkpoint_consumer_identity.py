"""Artificial checkpoint snapshots remain bound to their consumed tensor bytes."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pytest
import torch

from olympus.foundry import eval_suite, quantization, sft
from olympus.foundry.data_pipeline import REQUIRED_CATEGORIES, prepare_instruction_dataset

Mutation = Literal["replace", "delete", "unchanged"]


@dataclass(frozen=True)
class SnapshotFixture:
    manifest_path: Path
    checkpoint_path: Path
    original: bytes
    replacement: bytes


@pytest.fixture(scope="module")
def snapshot_fixture(tmp_path_factory: pytest.TempPathFactory) -> SnapshotFixture:
    root = tmp_path_factory.mktemp("checkpoint-consumer-artificial")
    records = [
        {
            "id": f"{split}-{category}",
            "category": category,
            "prompt": f"Fixture {split}_{category} question",
            "response": f"Answer {split}_{category}",
            "license": "MIT",
            "source": "artificial://checkpoint-consumer-identity",
            "split": split,
        }
        for split in ("train", "validation", "test")
        for category in REQUIRED_CATEGORIES
    ]
    source = root / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    manifest = prepare_instruction_dataset(source, root / "dataset")

    def serialize(seed: int) -> bytes:
        config = sft.SFTConfig(
            seed=seed,
            pack_sequences=False,
            mixed_precision=False,
            model=sft.TinyModelConfig(width=16, layers=1, heads=2, max_sequence_tokens=128),
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = sft.TinyCausalLM(config.model)
        stream = io.BytesIO()
        torch.save(
            {
                "schema_version": 3,
                "model_config": config.model.model_dump(mode="json"),
                "training_config": config.model_dump(mode="json"),
                "loss_normalization": config.loss_normalization,
                "dataset_manifest_sha256": manifest.manifest_sha256,
                "model_state": model.state_dict(),
            },
            stream,
        )
        return stream.getvalue()

    original, replacement = serialize(37), serialize(91)
    assert original != replacement
    checkpoint_path = root / "unchanged-fixture.pt"
    checkpoint_path.write_bytes(original)
    return SnapshotFixture(root / "dataset/manifest.json", checkpoint_path, original, replacement)


def change_checkpoint_after_read(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    fixture: SnapshotFixture,
    mutation: Mutation,
) -> list[bytes]:
    """Mutate only the source path after the real loader receives its first bytes."""
    path.write_bytes(fixture.original)
    original_read = Path.read_bytes
    reads: list[bytes] = []

    def changing_read(current: Path) -> bytes:
        raw = original_read(current)
        if current == path:
            reads.append(raw)
            if len(reads) == 1:
                if mutation == "replace":
                    current.write_bytes(fixture.replacement)
                elif mutation == "delete":
                    current.unlink()
        return raw

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    return reads


def assert_same_model(left: sft.TinyCausalLM, right: sft.TinyCausalLM) -> None:
    expected, actual = left.state_dict(), right.state_dict()
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[name], actual[name]) for name in expected)


@pytest.mark.parametrize("mutation", ["replace", "delete", "unchanged"])
def test_evaluation_candidate_identity_uses_the_consumed_checkpoint(
    snapshot_fixture: SnapshotFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Mutation,
) -> None:
    fixture = snapshot_fixture
    reference = eval_suite.evaluate_checkpoint(
        fixture.checkpoint_path,
        fixture.manifest_path,
        tmp_path / "reference.json",
        max_generation_tokens=1,
    )
    candidate_path = tmp_path / "candidate.pt"
    reads = change_checkpoint_after_read(monkeypatch, candidate_path, fixture, mutation)
    report = eval_suite.evaluate_checkpoint(
        candidate_path,
        fixture.manifest_path,
        tmp_path / "evaluation.json",
        max_generation_tokens=1,
    )
    expected_sha = hashlib.sha256(fixture.original).hexdigest()
    assert report.checkpoint_sha256 == reference.checkpoint_sha256 == expected_sha
    assert report.overall_candidate_loss == reference.overall_candidate_loss
    assert report.category_results == reference.category_results
    saved_report = json.loads((tmp_path / "evaluation.json").read_bytes())
    assert saved_report["checkpoint_sha256"] == expected_sha
    if mutation != "unchanged":
        assert reads == [fixture.original]


@pytest.mark.parametrize("mutation", ["replace", "delete", "unchanged"])
def test_evaluation_baseline_identity_uses_the_consumed_checkpoint(
    snapshot_fixture: SnapshotFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Mutation,
) -> None:
    fixture = snapshot_fixture
    reference = eval_suite.evaluate_checkpoint(
        fixture.checkpoint_path,
        fixture.manifest_path,
        tmp_path / "reference.json",
        base_checkpoint=fixture.checkpoint_path,
        max_generation_tokens=1,
    )
    baseline_path = tmp_path / "baseline.pt"
    reads = change_checkpoint_after_read(monkeypatch, baseline_path, fixture, mutation)
    report = eval_suite.evaluate_checkpoint(
        fixture.checkpoint_path,
        fixture.manifest_path,
        tmp_path / "evaluation.json",
        base_checkpoint=baseline_path,
        max_generation_tokens=1,
    )
    expected_identity = f"checkpoint:{hashlib.sha256(fixture.original).hexdigest()}"
    assert report.baseline_identity == reference.baseline_identity == expected_identity
    assert report.overall_baseline_loss == reference.overall_baseline_loss
    assert report.category_results == reference.category_results
    if mutation != "unchanged":
        assert reads == [fixture.original]


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("mutation", ["replace", "delete", "unchanged"])
def test_quantization_artifact_and_report_bind_the_consumed_checkpoint(
    snapshot_fixture: SnapshotFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bits: quantization.QuantizationBits,
    mutation: Mutation,
) -> None:
    fixture = snapshot_fixture
    reference = quantization.quantize_checkpoint(
        fixture.checkpoint_path,
        fixture.manifest_path,
        tmp_path / "reference",
        bits=bits,
    )
    checkpoint_path = tmp_path / "source.pt"
    reads = change_checkpoint_after_read(monkeypatch, checkpoint_path, fixture, mutation)
    report = quantization.quantize_checkpoint(
        checkpoint_path,
        fixture.manifest_path,
        tmp_path / "quantized",
        bits=bits,
    )
    expected_sha = hashlib.sha256(fixture.original).hexdigest()
    assert report.source_checkpoint_sha256 == reference.source_checkpoint_sha256 == expected_sha
    payload = cast(
        dict[str, Any],
        torch.load(
            report.artifact_path,
            map_location="cpu",
            weights_only=True,
        ),
    )
    assert payload["source_checkpoint_sha256"] == expected_sha
    saved_report = json.loads((tmp_path / "quantized" / f"int{bits}-report.json").read_bytes())
    assert saved_report["source_checkpoint_sha256"] == expected_sha
    assert report.source_loss == reference.source_loss
    assert report.quantized_loss == reference.quantized_loss
    assert_same_model(
        quantization.load_quantized_model(Path(reference.artifact_path)),
        quantization.load_quantized_model(Path(report.artifact_path)),
    )
    if mutation != "unchanged":
        assert reads == [fixture.original]


def test_checkpoint_loader_preserves_its_existing_two_value_contract(
    snapshot_fixture: SnapshotFixture,
) -> None:
    model, payload = sft._model_from_checkpoint(
        snapshot_fixture.checkpoint_path, torch.device("cpu")
    )
    expected = sft.load_trained_model(snapshot_fixture.checkpoint_path)
    assert_same_model(model, expected)
    assert payload["model_config"] == model.config.model_dump(mode="json")
