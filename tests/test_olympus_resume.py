"""Bounded numerical regressions; run only after host resource admission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from test_olympus_system import _config, _dataset_batch

from olympus.data.olympus_dataset import write_olympus_dataset_manifest
from olympus.evaluation.olympus_system import (
    evaluate_olympus_system,
    olympus_batch_inputs_targets,
)
from olympus.foundry.resources import ResourceGovernor
from olympus.models.olympus_system import OlympusReferenceSystem
from olympus.models.runtime import resolve_olympus_device
from olympus.training import olympus_system as training
from olympus.training.checkpoints import load_training_checkpoint
from olympus.training.olympus_system import (
    OlympusSystemTrainingConfig,
    TrainingInterrupted,
    train_olympus_system,
)


@pytest.fixture
def fixture(tmp_path: Path) -> tuple[Path, OlympusSystemTrainingConfig]:
    torch.manual_seed(61)
    data = tmp_path / "data"
    data.mkdir()
    torch.save(_dataset_batch(), data / "train.pt")
    torch.save(_dataset_batch(), data / "validation.pt")
    manifest = data / "manifest.json"
    write_olympus_dataset_manifest(
        manifest,
        dataset_id="resume-regression",
        version="1",
        license_name="MIT",
        train_batch=data / "train.pt",
        evaluation_batch=data / "validation.pt",
        train_source_ids=["train-0", "train-1"],
        evaluation_source_ids=["val-0", "val-1"],
    )
    model = _config()
    # Dropout makes missing Torch RNG restoration observable.
    for family in (model.hermes, model.prometheus, model.perseus, model.atlas):
        family.backbone.dropout = 0.1
    config = OlympusSystemTrainingConfig(
        model=model,
        seed=61,
        steps=4,
        batch_size=2,
        evaluation_batch_size=1,
        evaluation_interval=1,
        log_interval=1,
        device="cpu",
        resource_preflight=False,
    )
    return manifest, config


def _latest(directory: Path) -> dict[str, Any]:
    contract = json.loads((directory / "run.json").read_text())
    return load_training_checkpoint(
        directory / "checkpoints", kind="latest", binding=contract["binding"]
    )


def _assert_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_equal(first, second)
    else:
        assert left == right


def test_interruption_resume_matches_uninterrupted_parameters_optimizer_rng_and_metrics(
    fixture: tuple[Path, OlympusSystemTrainingConfig],
    tmp_path: Path,
) -> None:
    manifest, config = fixture
    full = train_olympus_system(manifest, tmp_path / "full", config=config)
    with pytest.raises(TrainingInterrupted):
        train_olympus_system(manifest, tmp_path / "split", config=config, stop_after_steps=2)
    assert _latest(tmp_path / "split")["completed_steps"] == 2
    resumed = train_olympus_system(manifest, tmp_path / "split", config=config, resume=True)
    left, right = _latest(tmp_path / "full"), _latest(tmp_path / "split")
    for key in ("model_state", "optimizer_state", "rng", "scheduler"):
        _assert_equal(left[key], right[key])
    assert full.final_evaluation.model_dump() == resumed.final_evaluation.model_dump()
    assert full.training == resumed.training
    assert full.best_step == resumed.best_step
    assert full.best_validation_loss == resumed.best_validation_loss
    assert full.best_validation_loss <= full.initial_evaluation.total_loss
    assert (tmp_path / "split/checkpoints/latest.json").read_bytes() != (
        tmp_path / "split/checkpoints/best.json"
    ).read_bytes() or resumed.best_step == 4
    events = [
        json.loads(line) for line in (tmp_path / "split/metrics.jsonl").read_text().splitlines()
    ]
    assert {"start", "training", "validation", "interrupted", "resume", "completed"} <= {
        event["phase"] for event in events
    }


def test_full_vs_uneven_batched_metrics_keep_global_denominators() -> None:
    torch.manual_seed(63)
    model = OlympusReferenceSystem(_config())
    batch = _dataset_batch()
    batch = {key: torch.cat([tensor, tensor[:1]], dim=0) for key, tensor in batch.items()}
    model.train()
    full = evaluate_olympus_system(model, batch, [])
    chunked = evaluate_olympus_system(model, batch, [], batch_size=2)
    assert model.training
    assert chunked.total_loss == pytest.approx(full.total_loss, abs=2e-6)
    assert chunked.next_state_mse == pytest.approx(full.next_state_mse, abs=2e-6)
    assert chunked.decision_accuracy == full.decision_accuracy
    assert chunked.risk.risk_coverage[-1].coverage == 1.0
    assert chunked.risk.risk_coverage[-1].population_coverage == 2 / 3


@pytest.mark.parametrize("field,value", [("seed", 62), ("steps", 5), ("batch_size", 1)])
def test_resume_refuses_changed_configuration(
    fixture: tuple[Path, OlympusSystemTrainingConfig],
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    manifest, config = fixture
    with pytest.raises(TrainingInterrupted):
        train_olympus_system(manifest, tmp_path / "run", config=config, stop_after_steps=0)
    changed = config.model_copy(update={field: value})
    with pytest.raises(ValueError, match="identity mismatch"):
        train_olympus_system(manifest, tmp_path / "run", config=changed, resume=True)


def test_resume_refuses_changed_source(
    fixture: tuple[Path, OlympusSystemTrainingConfig],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, config = fixture
    with pytest.raises(TrainingInterrupted):
        train_olympus_system(manifest, tmp_path / "run", config=config, stop_after_steps=0)
    monkeypatch.setattr(training, "source_identity", lambda: {"sha256": "f" * 64})
    with pytest.raises(ValueError, match="identity mismatch"):
        train_olympus_system(manifest, tmp_path / "run", config=config, resume=True)


def test_checkpoint_tamper_and_path_escape_are_rejected(
    fixture: tuple[Path, OlympusSystemTrainingConfig],
    tmp_path: Path,
) -> None:
    manifest, config = fixture
    run = tmp_path / "run"
    with pytest.raises(TrainingInterrupted):
        train_olympus_system(manifest, run, config=config, stop_after_steps=0)
    pointer_path = run / "checkpoints/latest.json"
    pointer = json.loads(pointer_path.read_text())
    (run / "checkpoints" / pointer["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        _latest(run)
    pointer["path"] = "../outside.pt"
    pointer_path.write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match="contained"):
        _latest(run)


def test_zero_budget_boundary_has_checkpoint_and_no_optimizer_step(
    fixture: tuple[Path, OlympusSystemTrainingConfig],
    tmp_path: Path,
) -> None:
    manifest, config = fixture
    with pytest.raises(TrainingInterrupted):
        train_olympus_system(manifest, tmp_path / "run", config=config, max_seconds=1e-12)
    payload = _latest(tmp_path / "run")
    assert payload["completed_steps"] == 0
    assert not payload["optimizer_state"]["state"]


def test_resource_denial_precedes_tensor_loading(
    fixture: tuple[Path, OlympusSystemTrainingConfig],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, config = fixture
    config.resource_preflight = True

    def denied(_self: Any) -> Any:
        raise RuntimeError("denied by test governor")

    def forbidden(_path: Path) -> Any:
        pytest.fail("tensor data loaded before admission")

    monkeypatch.setattr(ResourceGovernor, "__enter__", denied)
    monkeypatch.setattr(training, "load_olympus_dataset", forbidden)
    with pytest.raises(RuntimeError, match="denied"):
        train_olympus_system(manifest, tmp_path / "run", config=config)
    assert json.loads((tmp_path / "run/status.json").read_text())["status"] == "FAILED"


def test_completed_or_existing_directory_is_never_overwritten(
    fixture: tuple[Path, OlympusSystemTrainingConfig],
    tmp_path: Path,
) -> None:
    manifest, config = fixture
    run = tmp_path / "run"
    run.mkdir()
    marker = run / "user-work.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        train_olympus_system(manifest, run, config=config)
    assert marker.read_text() == "preserve"


def test_explicit_cuda_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_olympus_device("cuda")


def test_targets_are_not_inputs_and_kronos_resets_between_independent_calls() -> None:
    torch.manual_seed(93)
    model = OlympusReferenceSystem(_config()).eval()
    batch = _dataset_batch()
    inputs, _ = olympus_batch_inputs_targets(batch, [])
    assert not any("target" in name for name in inputs)
    states, actions, mask = batch["states"], batch["actions"], batch["temporal_mask"]
    with torch.inference_mode():
        first = model.kronos(states, actions, mask).next_state.clone()
        model.kronos(states + 100, actions - 100, mask)
        repeated = model.kronos(states, actions, mask).next_state
        altered = states.clone()
        altered[:, -1] += 100
        future_changed = model.kronos(altered, actions, mask).next_state
    assert torch.equal(first, repeated)
    assert torch.equal(first[:, :-1], future_changed[:, :-1])


def test_uniform_ablation_changes_computation_with_equal_parameter_count() -> None:
    torch.manual_seed(94)
    config = _config()
    learned = OlympusReferenceSystem(config).eval()
    uniform_config = config.model_copy(update={"fusion_strategy": "uniform"})
    uniform = OlympusReferenceSystem(uniform_config).eval()
    uniform.load_state_dict(learned.state_dict())
    assert sum(p.numel() for p in learned.parameters()) == sum(
        p.numel() for p in uniform.parameters()
    )
    inputs, _ = olympus_batch_inputs_targets(_dataset_batch(), [])
    with torch.inference_mode():
        first, second = learned(**inputs), uniform(**inputs)
    torch.testing.assert_close(second.family_gates, torch.full_like(second.family_gates, 0.2))
    assert not torch.allclose(first.family_gates, second.family_gates)


def test_evaluation_fp32_accepts_float64_bundle_without_device_or_dtype_mismatch() -> None:
    torch.manual_seed(95)
    model = OlympusReferenceSystem(_config())
    batch = _dataset_batch()
    batch = {
        name: tensor.double() if tensor.is_floating_point() else tensor
        for name, tensor in batch.items()
    }
    report = evaluate_olympus_system(model, batch, [], batch_size=1)
    assert report.examples == 2
    assert report.next_state_mse >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; not a GPU pass")
def test_tiny_cuda_forward_backward_evaluate_resume(
    fixture: tuple[Path, OlympusSystemTrainingConfig],
    tmp_path: Path,
) -> None:
    manifest, config = fixture
    config.device = "cuda"
    with pytest.raises(TrainingInterrupted):
        train_olympus_system(manifest, tmp_path / "cuda", config=config, stop_after_steps=2)
    resumed = train_olympus_system(manifest, tmp_path / "cuda", config=config, resume=True)
    full = train_olympus_system(manifest, tmp_path / "full-cuda", config=config)
    left, right = _latest(tmp_path / "cuda"), _latest(tmp_path / "full-cuda")
    for name in left["model_state"]:
        torch.testing.assert_close(
            left["model_state"][name], right["model_state"][name], rtol=0, atol=0
        )
    assert resumed.final_evaluation.total_loss == pytest.approx(
        full.final_evaluation.total_loss, abs=1e-6
    )
