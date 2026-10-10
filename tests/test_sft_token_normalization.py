"""Supervised-token objectives must not depend on microbatch partitioning."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from olympus.foundry import sft
from olympus.foundry.data_pipeline import prepare_instruction_dataset, verify_dataset_manifest
from olympus.foundry.eval_suite import HeldOutEvaluation, evaluate_checkpoint
from olympus.foundry.quantization import QuantizationReport, quantize_checkpoint

Row = tuple[list[int], list[int]]
SOURCE = Path(__file__).resolve().parents[1] / "datasets/hermes-smoke/source.jsonl"


class FixedLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([2.0, 0.0, -1.0], dtype=torch.float64))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(*tokens.shape, -1)


class Float64TinyCausalLM(sft.TinyCausalLM):
    """Same initialized model and optimizer, higher precision for the oracle."""

    def __init__(self, config: sft.TinyModelConfig) -> None:
        super().__init__(config)
        self.double()


def analytic_rows() -> list[Row]:
    # Nonmasked first-column labels still have no next-token prediction. Their
    # values must never enter either the numerator or the denominator.
    return [([0] * (count + 1), [2] + [index % 3] * count)
            for index, count in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9])]


def analytic_mean(model: FixedLogits, rows: list[Row]) -> float:
    normalizer = float(torch.logsumexp(model.bias.detach(), dim=0))
    values = [normalizer - float(model.bias.detach()[label])
              for _, labels in rows for label in labels[1:] if label != -100]
    return sum(values) / len(values)


def test_evaluation_equals_analytic_token_mean_for_unequal_answers_and_tail() -> None:
    model = FixedLogits()
    rows = analytic_rows()
    expected = analytic_mean(model, rows)
    for order in [rows, list(reversed(rows)), rows[3:] + rows[:3]]:
        observed = sft.evaluate_loss(cast(sft.TinyCausalLM, model), order,
                                     device=torch.device("cpu"))
        assert observed == pytest.approx(expected, abs=1e-12)


def test_legacy_evaluation_keeps_the_original_average_of_batch_means() -> None:
    model = FixedLogits()
    rows = analytic_rows()
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(0)).tolist()
    means = [analytic_mean(model, [rows[i] for i in order[start:start + 8]])
             for start in range(0, len(order), 8)]
    observed = sft.evaluate_loss(cast(sft.TinyCausalLM, model), rows,
                                 device=torch.device("cpu"),
                                 normalization="legacy_batch_mean_v1")
    assert observed == pytest.approx(sum(means) / len(means), abs=1e-12)
    assert observed != pytest.approx(analytic_mean(model, rows), abs=1e-3)


def test_evaluation_excludes_prompt_and_padding_targets() -> None:
    model = FixedLogits()
    rows = [([0] * 7, [1, -100, -100, 0, 1, -100, -100]),
            ([0] * 4, [2, -100, 2, 2])]
    observed = sft.evaluate_loss(cast(sft.TinyCausalLM, model), rows,
                                 device=torch.device("cpu"))
    assert observed == pytest.approx(analytic_mean(model, rows), abs=1e-12)


@pytest.mark.parametrize("rows", [[], [([0, 0], [-100, -100])], [([0], [2])]])
def test_evaluation_requires_supervised_next_token_targets(rows: list[Row]) -> None:
    with pytest.raises(ValueError, match="no batches|no supervised"):
        sft.evaluate_loss(cast(sft.TinyCausalLM, FixedLogits()), rows,
                          device=torch.device("cpu"))


def test_unknown_normalization_is_rejected() -> None:
    with pytest.raises(ValidationError):
        sft.SFTConfig.model_validate({"loss_normalization": "example_mean"})
    with pytest.raises(ValueError, match="normalization"):
        sft.evaluate_loss(cast(sft.TinyCausalLM, FixedLogits()), analytic_rows(),
                          device=torch.device("cpu"),
                          normalization=cast(sft.LossNormalization, "unknown"))


def small_config(**changes: Any) -> sft.SFTConfig:
    values: dict[str, Any] = {
        "seed": 71, "epochs": 1, "batch_size": 1, "gradient_accumulation_steps": 3,
        "learning_rate": 0.002, "gradient_clip_norm": 0.25, "weight_decay": 0.0,
        "mixed_precision": False, "pack_sequences": False,
        "model": sft.TinyModelConfig(width=16, layers=1, heads=4, max_sequence_tokens=160),
    }
    values.update(changes)
    return sft.SFTConfig.model_validate(values)


@pytest.fixture(scope="module")
def manifest_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("sft-token-dataset")
    prepare_instruction_dataset(SOURCE, root)
    return root / "manifest.json"


def checkpoint(path: str | Path) -> dict[str, Any]:
    return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=True))


def record_gradients(monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    gradients: list[torch.Tensor] = []
    original_clip = nn.utils.clip_grad_norm_

    def capture_before_clipping(parameters: list[nn.Parameter], maximum: float) -> torch.Tensor:
        pieces = []
        for parameter in parameters:
            assert parameter.grad is not None
            pieces.append(parameter.grad.detach().flatten().clone())
        gradients.append(torch.cat(pieces))
        return original_clip(parameters, maximum)

    monkeypatch.setattr(nn.utils, "clip_grad_norm_", capture_before_clipping)
    return gradients


@pytest.mark.parametrize("batch_size,accumulation", [(1, 3), (1, 5), (2, 5)])
def test_actual_training_matches_unsplit_batch_gradients_including_short_tail(
    manifest_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    batch_size: int, accumulation: int,
) -> None:
    # The exact attention key-bias gradient is zero. Float32 cancellation can
    # create tiny gradients that AdamW amplifies through its epsilon, obscuring
    # the partition identity in saved parameters. Use float64 for this strict
    # optimizer oracle; the separate tail and end-to-end tests retain float32.
    monkeypatch.setattr(sft, "TinyCausalLM", Float64TinyCausalLM)
    gradients = record_gradients(monkeypatch)
    config = small_config(batch_size=batch_size, gradient_accumulation_steps=accumulation)
    accumulated = sft.run_sft(manifest_path, tmp_path / "microbatches", config=config)
    micro_gradients = gradients.copy()
    gradients.clear()
    reference = sft.run_sft(
        manifest_path, tmp_path / "unsplit",
        config=small_config(batch_size=batch_size * accumulation, gradient_accumulation_steps=1),
    )
    assert accumulated.optimizer_steps == reference.optimizer_steps == math.ceil(
        12 / (batch_size * accumulation)
    )
    assert len(micro_gradients) == len(gradients) == accumulated.optimizer_steps
    for micro, unsplit in zip(micro_gradients, gradients, strict=True):
        assert micro.dtype == unsplit.dtype == torch.float64
        torch.testing.assert_close(micro, unsplit, rtol=1e-7, atol=1e-10)
    micro_checkpoint = checkpoint(accumulated.checkpoint_path)
    reference_checkpoint = checkpoint(reference.checkpoint_path)
    for name, value in micro_checkpoint["model_state"].items():
        torch.testing.assert_close(value, reference_checkpoint["model_state"][name],
                                   rtol=1e-8, atol=1e-9,
                                   msg=f"model state differs for {name}")
    micro_log = json.loads(Path(accumulated.log_path).read_text())
    reference_log = json.loads(Path(reference.log_path).read_text())
    assert micro_log["train_loss"] == pytest.approx(reference_log["train_loss"], rel=1e-6)
    assert micro_log["train_supervised_tokens"] == reference_log["train_supervised_tokens"] > 12
    assert micro_log["loss_normalization"] == "supervised_token_mean_v2"


def test_short_window_is_not_scaled_down_even_when_answer_lengths_are_equal(
    manifest_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Use the existing category selection control on a fixed fixture. Both
    # selected answers have 46 supervised targets, so this isolates the short
    # window defect from unequal-answer weighting without changing any data.
    examples = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    weights = {item["category"]: float(item["category"] in {"extraction", "agent_behavior"})
               for item in examples}
    config = small_config(category_mix_weights=weights, gradient_accumulation_steps=5)
    manifest = verify_dataset_manifest(manifest_path)
    rows = sft._encode_examples(sft._load_examples(manifest, "train", manifest_path.parent), config)
    assert [sum(label != -100 for label in labels[1:]) for _, labels in rows] == [46, 46]
    gradients = record_gradients(monkeypatch)
    short = sft.run_sft(manifest_path, tmp_path / "short", config=config)
    short_gradient = gradients.pop()
    reference = sft.run_sft(
        manifest_path, tmp_path / "unsplit",
        config=small_config(category_mix_weights=weights, batch_size=5,
                            gradient_accumulation_steps=1),
    )
    assert short.optimizer_steps == reference.optimizer_steps == len(gradients) == 1
    torch.testing.assert_close(short_gradient, gradients[0], rtol=1e-4, atol=2e-6)


@pytest.fixture(scope="module")
def trained_versions(
    manifest_path: Path, tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, sft.TrainingSummary]:
    root = tmp_path_factory.mktemp("sft-versioned-checkpoints")
    return {normalization: sft.run_sft(
        manifest_path, root, config=small_config(loss_normalization=normalization)
    ) for normalization in ["legacy_batch_mean_v1", "supervised_token_mean_v2"]}


def test_checkpoint_log_and_summary_bind_the_objective_and_keep_distinct_paths(
    trained_versions: dict[str, sft.TrainingSummary],
) -> None:
    legacy, current = trained_versions.values()
    assert legacy.checkpoint_path != current.checkpoint_path
    assert current.run_id == legacy.run_id + "-tokens-v2"
    for normalization, summary in trained_versions.items():
        payload = checkpoint(summary.checkpoint_path)
        assert payload["schema_version"] == 2
        assert payload["loss_normalization"] == normalization
        assert payload["training_config"]["loss_normalization"] == normalization
        assert summary.loss_normalization == normalization
        assert json.loads(Path(summary.log_path).read_text())["loss_normalization"] == normalization
        old_summary = summary.model_dump()
        del old_summary["loss_normalization"]
        assert sft.TrainingSummary.model_validate(old_summary).loss_normalization == (
            "legacy_batch_mean_v1"
        )


@pytest.mark.parametrize("normalization", ["legacy_batch_mean_v1", "supervised_token_mean_v2"])
def test_resume_refuses_an_objective_change_before_writing_training_artifacts(
    manifest_path: Path, trained_versions: dict[str, sft.TrainingSummary],
    tmp_path: Path, normalization: str,
) -> None:
    path = Path(trained_versions[normalization].checkpoint_path)
    retained = path.read_bytes()
    other = "supervised_token_mean_v2" if normalization == "legacy_batch_mean_v1" else (
        "legacy_batch_mean_v1"
    )
    with pytest.raises(ValueError, match="loss normalization does not match"):
        sft.run_sft(
            manifest_path, tmp_path, config=small_config(loss_normalization=other, epochs=2),
            resume_checkpoint=path,
        )
    assert path.read_bytes() == retained
    assert not list(tmp_path.rglob("metrics.jsonl"))
    assert not list(tmp_path.rglob("checkpoint.pt"))


def test_unversioned_checkpoint_retains_legacy_semantics_and_can_resume_explicitly(
    manifest_path: Path, trained_versions: dict[str, sft.TrainingSummary], tmp_path: Path,
) -> None:
    original = checkpoint(trained_versions["legacy_batch_mean_v1"].checkpoint_path)
    original["schema_version"] = 1
    del original["loss_normalization"]
    del original["training_config"]["loss_normalization"]
    path = tmp_path / "historical-format.pt"
    torch.save(original, path)
    retained = path.read_bytes()
    assert sft._checkpoint_training_config(original).loss_normalization == "legacy_batch_mean_v1"
    resumed = sft.run_sft(
        manifest_path, tmp_path / "resumed",
        config=small_config(loss_normalization="legacy_batch_mean_v1", epochs=2),
        resume_checkpoint=path,
    )
    assert resumed.completed_epochs == 2 and resumed.optimizer_steps == 8
    assert resumed.loss_normalization == "legacy_batch_mean_v1"
    assert path.read_bytes() == retained


@pytest.mark.parametrize("broken", ["missing", "mismatched"])
def test_versioned_checkpoint_rejects_missing_or_inconsistent_objective_metadata(
    trained_versions: dict[str, sft.TrainingSummary], tmp_path: Path, broken: str,
) -> None:
    payload = checkpoint(trained_versions["supervised_token_mean_v2"].checkpoint_path)
    if broken == "missing":
        del payload["training_config"]["loss_normalization"]
    else:
        payload["loss_normalization"] = "legacy_batch_mean_v1"
    path = tmp_path / "broken.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="loss normalization"):
        sft.load_trained_model(path)


@pytest.mark.parametrize("normalization", ["legacy_batch_mean_v1", "supervised_token_mean_v2"])
def test_held_out_and_quantization_reports_retain_checkpoint_loss_definition(
    manifest_path: Path, trained_versions: dict[str, sft.TrainingSummary],
    tmp_path: Path, normalization: str,
) -> None:
    path = Path(trained_versions[normalization].checkpoint_path)
    evaluation = evaluate_checkpoint(path, manifest_path, tmp_path / "evaluation.json",
                                     max_generation_tokens=1)
    quantization = quantize_checkpoint(path, manifest_path, tmp_path / "int8", bits=8)
    assert evaluation.loss_normalization == quantization.loss_normalization == normalization
    assert quantization.source_loss == pytest.approx(evaluation.overall_candidate_loss, abs=1e-7)
    # The artifact hashes bind the same checkpoint; legacy parsing never infers v2.
    old_evaluation = evaluation.model_dump()
    del old_evaluation["loss_normalization"]
    assert HeldOutEvaluation.model_validate(old_evaluation).loss_normalization == (
        "legacy_batch_mean_v1"
    )
    old_quantization = quantization.model_dump()
    del old_quantization["loss_normalization"]
    assert QuantizationReport.model_validate(old_quantization).loss_normalization == (
        "legacy_batch_mean_v1"
    )
    manifest = verify_dataset_manifest(manifest_path)
    assert evaluation.dataset_manifest_sha256 == manifest.manifest_sha256
