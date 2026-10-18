from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from olympus.foundry import eval_suite, quantization, sft
from olympus.foundry.data_pipeline import (
    REQUIRED_CATEGORIES,
    DatasetManifestV2,
    InstructionExample,
    prepare_instruction_dataset,
)


@dataclass
class PreparedDataset:
    path: Path
    manifest: DatasetManifestV2
    examples: dict[str, list[InstructionExample]]
    split_paths: dict[str, Path]


@pytest.fixture
def prepared(tmp_path: Path) -> PreparedDataset:
    records = [
        {
            "id": f"{split}-{category}",
            "category": category,
            "prompt": f"Artificial {split}_{category} prompt",
            "response": f"Original {split}_{category} answer",
            "license": "MIT",
            "source": "artificial://snapshot-regression",
            "split": split,
        }
        for split in ("train", "validation", "test")
        for category in REQUIRED_CATEGORIES
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    root = tmp_path / "dataset"
    manifest = prepare_instruction_dataset(source, root, source_uri="artificial://snapshot")
    examples = sorted(
        (InstructionExample.model_validate(record) for record in records), key=lambda item: item.id
    )
    return PreparedDataset(
        path=root / "manifest.json",
        manifest=manifest,
        examples={
            split.name: [item for item in examples if item.split == split.name]
            for split in manifest.splits
        },
        split_paths={split.name: root / split.path for split in manifest.splits},
    )


def change_after_read(monkeypatch: pytest.MonkeyPatch, path: Path, mutation: str) -> list[bytes]:
    """Alter the file after returning its original bytes to the real verifier."""
    original_read = Path.read_bytes
    reads: list[bytes] = []

    def changing_read(current: Path) -> bytes:
        payload = original_read(current)
        if current == path:
            reads.append(payload)
            if mutation == "replace":
                replacement = payload.replace(b"Original", b"Replaced")
                assert replacement != payload
                current.write_bytes(replacement)
            elif mutation == "delete":
                current.unlink()
        return payload

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    return reads


@pytest.mark.parametrize("mutation", ["replace", "delete", "unchanged"])
def test_evaluation_uses_the_verified_test_snapshot(
    prepared: PreparedDataset, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    reads = change_after_read(monkeypatch, prepared.split_paths["test"], mutation)
    identity, examples = eval_suite._load_test_examples(prepared.path)
    assert identity == prepared.manifest.manifest_sha256
    assert examples == prepared.examples["test"]
    assert len(reads) == 1


class StopBeforeModelWork(Exception):
    """Stop at record encoding; no training or loss evaluation is authorized."""


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("mutation", ["replace", "delete", "unchanged"])
def test_sft_encodes_the_verified_train_and_validation_snapshots(
    prepared: PreparedDataset,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    split: str,
    mutation: str,
) -> None:
    reads = change_after_read(monkeypatch, prepared.split_paths[split], mutation)
    observed: list[list[InstructionExample]] = []

    def capture_examples(
        examples: list[InstructionExample], config: sft.SFTConfig
    ) -> list[tuple[list[int], list[int]]]:
        observed.append(examples)
        if len(observed) == 2:
            raise StopBeforeModelWork
        return [([1, 2], [-100, 2])]

    monkeypatch.setattr(sft, "_encode_examples", capture_examples)
    monkeypatch.setattr(sft, "ResourceGovernor", lambda _: nullcontext())
    with pytest.raises(StopBeforeModelWork):
        sft.run_sft(prepared.path, tmp_path / "training", config=sft.SFTConfig())
    assert observed == [prepared.examples["train"], prepared.examples["validation"]]
    assert len(reads) == 1
    assert not (tmp_path / "training").exists()


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("mutation", ["replace", "delete", "unchanged"])
def test_quantization_encodes_the_verified_test_snapshot(
    prepared: PreparedDataset,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bits: quantization.QuantizationBits,
    mutation: str,
) -> None:
    config = sft.SFTConfig(model=sft.TinyModelConfig(width=16, layers=1, max_sequence_tokens=64))
    model = sft.TinyCausalLM(config.model)
    # Exercise actual quantized serialization but stop before any scoring.
    checkpoint: dict[str, Any] = {
        "training_config": config.model_dump(mode="json"),
        "dataset_manifest_sha256": prepared.manifest.manifest_sha256,
    }
    checkpoint_path = tmp_path / "untrained-fixture.pt"
    checkpoint_path.write_bytes(b"artificial-untrained-checkpoint")
    monkeypatch.setattr(quantization, "_model_from_checkpoint", lambda *_: (model, checkpoint))
    reads = change_after_read(monkeypatch, prepared.split_paths["test"], mutation)
    observed: list[list[InstructionExample]] = []

    def capture_examples(
        examples: list[InstructionExample], config: sft.SFTConfig
    ) -> list[tuple[list[int], list[int]]]:
        observed.append(examples)
        raise StopBeforeModelWork

    monkeypatch.setattr(quantization, "_encode_examples", capture_examples)
    with pytest.raises(StopBeforeModelWork):
        quantization.quantize_checkpoint(
            checkpoint_path, prepared.path, tmp_path / "quantized", bits=bits
        )
    assert observed == [prepared.examples["test"]]
    assert len(reads) == 1
    assert not (tmp_path / "quantized" / f"int{bits}-report.json").exists()
