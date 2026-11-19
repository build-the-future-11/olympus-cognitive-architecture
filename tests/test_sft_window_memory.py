"""The real trainer must start backward without padding an entire epoch first."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
import torch

from olympus.foundry import sft
from olympus.foundry.data_pipeline import prepare_instruction_dataset


@pytest.mark.parametrize("normalization", ["legacy_batch_mean_v1", "supervised_token_mean_v2"])
@pytest.mark.parametrize("accumulation", [2, 5])
def test_actual_training_pads_only_the_current_optimizer_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    normalization: sft.LossNormalization, accumulation: int,
) -> None:
    source = Path(__file__).resolve().parents[1] / "datasets/hermes-smoke/source.jsonl"
    prepared = tmp_path / "dataset"
    prepare_instruction_dataset(source, prepared)
    original_batches = sft._batches
    original_loss = sft._loss
    original_step = torch.optim.AdamW.step
    generated_since_step = 0
    backwards_since_step = 0
    generation_at_backward: list[int] = []
    window_sizes: list[int] = []

    def traced_batches(
        rows: list[tuple[list[int], list[int]]], batch_size: int, generator: torch.Generator,
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        nonlocal generated_since_step
        for batch in original_batches(rows, batch_size, generator):
            generated_since_step += 1
            yield batch

    def traced_loss(
        model: sft.TinyCausalLM, tokens: torch.Tensor, labels: torch.Tensor, **kwargs: Any,
    ) -> torch.Tensor:
        nonlocal generated_since_step, backwards_since_step
        result = original_loss(model, tokens, labels, **kwargs)
        if model.training:
            generation_at_backward.append(generated_since_step)
            backwards_since_step += 1
        else:
            # Validation has its own batching. The final validation call resets
            # this counter before the next training epoch begins.
            generated_since_step = 0
        return result

    def traced_step(optimizer: torch.optim.AdamW, *args: Any, **kwargs: Any) -> Any:
        nonlocal generated_since_step, backwards_since_step
        window_sizes.append(backwards_since_step)
        result = original_step(optimizer, *args, **kwargs)
        generated_since_step = backwards_since_step = 0
        return result

    monkeypatch.setattr(sft, "_batches", traced_batches)
    monkeypatch.setattr(sft, "_loss", traced_loss)
    monkeypatch.setattr(torch.optim.AdamW, "step", traced_step)
    summary = sft.run_sft(
        prepared / "manifest.json", tmp_path / "training",
        config=sft.SFTConfig(
            seed=71, epochs=2, batch_size=1, gradient_accumulation_steps=accumulation,
            mixed_precision=False, pack_sequences=False, loss_normalization=normalization,
            model=sft.TinyModelConfig(width=16, layers=1, heads=4, max_sequence_tokens=160),
        ),
    )
    expected_sizes = [accumulation] * (12 // accumulation)
    if 12 % accumulation:
        expected_sizes.append(12 % accumulation)
    assert window_sizes == expected_sizes * 2
    assert summary.optimizer_steps == 2 * math.ceil(12 / accumulation)
    assert len(generation_at_backward) == 24
    assert max(generation_at_backward) <= accumulation
