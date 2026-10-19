"""A valid dataset cannot be substituted for a checkpoint's declared dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from olympus.foundry import quantization, sft
from olympus.foundry.data_pipeline import prepare_instruction_dataset


@pytest.mark.parametrize("identity", [None, "0" * 64])
def test_quantization_rejects_unbound_dataset_before_creating_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity: str | None,
) -> None:
    source = Path(__file__).resolve().parents[1] / "datasets/hermes-smoke/source.jsonl"
    prepared = tmp_path / "dataset"
    manifest = prepare_instruction_dataset(source, prepared)
    assert identity != manifest.manifest_sha256
    config = sft.SFTConfig(model=sft.TinyModelConfig(width=16, layers=1, heads=4))
    model = sft.TinyCausalLM(config.model)
    checkpoint: dict[str, Any] = {
        "schema_version": 2,
        "loss_normalization": config.loss_normalization,
        "training_config": config.model_dump(mode="json"),
        "model_config": config.model.model_dump(mode="json"),
        "model_state": model.state_dict(),
    }
    if identity is not None:
        checkpoint["dataset_manifest_sha256"] = identity
    checkpoint_path = tmp_path / "fixture.pt"
    torch.save(checkpoint, checkpoint_path)

    def no_weight_publication(*args: Any, **kwargs: Any) -> None:
        pytest.fail("unbound data reached weight publication before dataset admission")

    monkeypatch.setattr(quantization, "_atomic_torch_save", no_weight_publication)
    output = tmp_path / "quantized"
    with pytest.raises(ValueError, match="dataset hashes do not match"):
        quantization.quantize_checkpoint(
            checkpoint_path,
            prepared / "manifest.json",
            output,
            bits=8,
        )
    assert not output.exists()


def test_corrupt_dataset_is_rejected_before_creating_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(__file__).resolve().parents[1] / "datasets/hermes-smoke/source.jsonl"
    prepared = tmp_path / "dataset"
    manifest = prepare_instruction_dataset(source, prepared)
    descriptor = next(split for split in manifest.splits if split.name == "test")
    (prepared / descriptor.path).write_text("corrupted artificial data\n", encoding="utf-8")
    config = sft.SFTConfig(model=sft.TinyModelConfig(width=16, layers=1, heads=4))
    model = sft.TinyCausalLM(config.model)
    checkpoint: dict[str, Any] = {
        "schema_version": 2,
        "loss_normalization": config.loss_normalization,
        "training_config": config.model_dump(mode="json"),
        "model_config": config.model.model_dump(mode="json"),
        "model_state": model.state_dict(),
        "dataset_manifest_sha256": manifest.manifest_sha256,
    }
    checkpoint_path = tmp_path / "fixture.pt"
    torch.save(checkpoint, checkpoint_path)

    def no_weight_publication(*args: Any, **kwargs: Any) -> None:
        pytest.fail("corrupt data reached weight publication before dataset admission")

    monkeypatch.setattr(quantization, "_atomic_torch_save", no_weight_publication)
    output = tmp_path / "quantized"
    with pytest.raises(ValueError, match="hash"):
        quantization.quantize_checkpoint(
            checkpoint_path,
            prepared / "manifest.json",
            output,
            bits=8,
        )
    assert not output.exists()
