from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from olympus.data.olympus_dataset import (
    _validate_bundle,
    load_olympus_dataset,
    write_olympus_dataset_manifest,
)
from olympus.models import runtime as runtime_module
from olympus.models.olympus_system import (
    OlympusObjectiveOutput,
    OlympusReferenceSystem,
    OlympusSystemConfig,
    olympus_system_loss,
)
from olympus.models.reference_families import AionReferenceOutput
from olympus.models.runtime import (
    OlympusRuntimeConfig,
    OlympusSystemRuntime,
    save_olympus_checkpoint,
)
from olympus.training.checkpoints import (
    capture_rng,
    load_training_checkpoint,
    save_training_checkpoint,
)
from olympus.training.provenance import digest_json, file_sha256


class _RuntimeProbe(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.child = nn.Dropout(0.5)
        self.events = events
        self.fail = False

    def to(self, *args: Any, **kwargs: Any) -> _RuntimeProbe:
        self.events.append("model-transfer")
        super().to(*args, **kwargs)
        return self

    def forward(self, value: torch.Tensor) -> SimpleNamespace:
        assert not self.training
        assert not self.child.training
        assert not torch.is_grad_enabled()
        if self.fail:
            raise RuntimeError("inference failure")
        return SimpleNamespace(raw_decision_logits=self.child(value))


def test_inference_restores_mixed_module_modes_and_failure_state() -> None:
    model = _RuntimeProbe([])
    model.train()
    model.child.eval()
    runtime = OlympusSystemRuntime(
        cast(OlympusReferenceSystem, model),
        OlympusRuntimeConfig(device="cpu", mixed_precision=False, resource_preflight=False),
    )
    assert model.training and not model.child.training
    inputs = {"value": torch.ones(2, 3)}
    first = runtime.infer(inputs)
    second = runtime.infer(inputs)
    assert torch.equal(first.output.raw_decision_logits, second.output.raw_decision_logits)
    assert model.training and not model.child.training
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    model.fail = True
    with pytest.raises(RuntimeError, match="inference failure"):
        runtime.infer(inputs)
    assert model.training and not model.child.training
    assert torch.are_deterministic_algorithms_enabled() is previous_determinism
    assert torch.is_deterministic_algorithms_warn_only_enabled() is previous_warn_only


def test_runtime_mixed_precision_requires_explicit_opt_in() -> None:
    assert OlympusRuntimeConfig().mixed_precision is False
    assert OlympusRuntimeConfig(mixed_precision=True).mixed_precision is True


def test_resource_admission_precedes_model_and_input_transfers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Governor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> Governor:
            events.append("admit")
            return self

        def __exit__(self, *args: Any) -> None:
            events.append("release")

    def move(inputs: Any, device: Any) -> Any:
        events.append("input-transfer")
        return inputs

    monkeypatch.setattr(runtime_module, "ResourceGovernor", Governor)
    monkeypatch.setattr(runtime_module, "_move_inputs", move)
    runtime = OlympusSystemRuntime(
        cast(OlympusReferenceSystem, _RuntimeProbe(events)),
        OlympusRuntimeConfig(device="cpu", mixed_precision=False),
    )
    assert events == ["admit", "model-transfer", "release"]
    events.clear()
    runtime.infer({"value": torch.ones(1, 2)})
    assert events == ["admit", "input-transfer", "release"]


def test_denied_admission_does_not_transfer_model(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class DeniedGovernor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> None:
            raise RuntimeError("resource admission denied")

        def __exit__(self, *args: Any) -> None:
            pass

    monkeypatch.setattr(runtime_module, "ResourceGovernor", DeniedGovernor)
    with pytest.raises(RuntimeError, match="resource admission denied"):
        OlympusSystemRuntime(
            cast(OlympusReferenceSystem, _RuntimeProbe(events)),
            OlympusRuntimeConfig(device="cpu", mixed_precision=False),
        )
    assert events == []


def test_checkpoint_publication_cannot_overwrite_a_racing_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "checkpoint.pt"
    model = SimpleNamespace(config=OlympusSystemConfig(), state_dict=lambda: {})

    def racing_save(payload: Any, temporary: Path) -> None:
        temporary.write_bytes(b"candidate checkpoint")
        destination.write_bytes(b"other writer checkpoint")

    monkeypatch.setattr(torch, "save", racing_save)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_olympus_checkpoint(cast(OlympusReferenceSystem, model), destination)
    assert destination.read_bytes() == b"other writer checkpoint"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["checkpoint.pt"]


def _batch() -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.ones((2, 3), dtype=torch.long),
        "attention_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "image": torch.zeros(2, 3, 4, 4),
        "states": torch.zeros(2, 3, 2),
        "actions": torch.zeros(2, 3, 1),
        "temporal_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "allowed_transitions": torch.ones((2, 2), dtype=torch.bool),
        "risk_score": torch.zeros(2),
        "decision_targets": torch.zeros(2, dtype=torch.long),
        "transition_targets": torch.zeros(2, dtype=torch.long),
        "next_state_targets": torch.zeros(2, 3, 2),
        "domain_ids": torch.zeros(2, dtype=torch.long),
    }


@pytest.mark.parametrize(
    ("name", "replacement", "message"),
    [
        ("tokens", lambda: torch.tensor(1), "rank-two integer"),
        ("tokens", lambda: -torch.ones((2, 3), dtype=torch.long), "non-negative IDs"),
        ("tokens", lambda: torch.empty((2, 0), dtype=torch.long), "non-empty sequence"),
        ("states", lambda: torch.zeros((2, 3, 2), dtype=torch.long), "floating-point"),
        ("actions", lambda: torch.zeros((2, 3, 1), dtype=torch.long), "floating-point"),
        (
            "next_state_targets",
            lambda: torch.zeros((2, 3, 2), dtype=torch.long),
            "floating-point",
        ),
        ("risk_score", lambda: torch.zeros(2, dtype=torch.long), "floating-point"),
        ("domain_ids", lambda: torch.tensor([0.0, 1.0]), "non-negative integers"),
        ("domain_ids", lambda: torch.tensor([0, -1]), "non-negative integers"),
        ("attention_mask", lambda: torch.zeros((2, 3), dtype=torch.bool), "active position"),
        ("temporal_mask", lambda: torch.zeros((2, 3), dtype=torch.bool), "active position"),
        (
            "attention_mask",
            lambda: torch.tensor([[True, False, True], [True, True, True]]),
            "contiguous prefix",
        ),
        (
            "temporal_mask",
            lambda: torch.tensor([[True, False, True], [True, True, True]]),
            "contiguous prefix",
        ),
    ],
)
def test_dataset_rejects_invalid_inputs_before_model_execution(
    name: str, replacement: Any, message: str
) -> None:
    batch = _batch()
    batch[name] = replacement()
    with pytest.raises(ValueError, match=message):
        _validate_bundle(batch, extensions=[], source_ids=["one", "two"])


def test_dataset_rejects_relabelled_copy_of_training_artifact(tmp_path: Path) -> None:
    train = tmp_path / "train.pt"
    evaluation = tmp_path / "evaluation.pt"
    torch.save(_batch(), train)
    evaluation.write_bytes(train.read_bytes())
    with pytest.raises(ValueError, match="same batch artifact"):
        write_olympus_dataset_manifest(
            tmp_path / "manifest.json",
            dataset_id="duplicate-test",
            version="1",
            license_name="LicenseRef-Proprietary",
            train_batch=train,
            evaluation_batch=evaluation,
            train_source_ids=["train-1", "train-2"],
            evaluation_source_ids=["eval-1", "eval-2"],
        )


def test_dataset_accepts_valid_partial_prefixes(tmp_path: Path) -> None:
    train = tmp_path / "train.pt"
    evaluation = tmp_path / "evaluation.pt"
    torch.save(_batch(), train)
    evaluation_batch = _batch()
    evaluation_batch["image"].fill_(0.5)
    torch.save(evaluation_batch, evaluation)
    manifest_path = tmp_path / "manifest.json"
    write_olympus_dataset_manifest(
        manifest_path,
        dataset_id="prefix-test",
        version="1",
        license_name="LicenseRef-Proprietary",
        train_batch=train,
        evaluation_batch=evaluation,
        train_source_ids=["train-1", "train-2"],
        evaluation_source_ids=["eval-1", "eval-2"],
    )
    loaded = load_olympus_dataset(manifest_path)
    assert torch.equal(loaded.train["temporal_mask"], _batch()["temporal_mask"])


def _objective_output(device: str = "cpu") -> OlympusObjectiveOutput:
    logits = torch.zeros((2, 2), device=device)
    inactive = torch.zeros(2, dtype=torch.bool, device=device)
    return OlympusObjectiveOutput(
        aion=AionReferenceOutput(
            transition_logits=logits,
            transition_probabilities=logits.softmax(dim=-1),
            selected_stage=torch.zeros(2, dtype=torch.long, device=device),
            forced_stop=inactive,
        ),
        next_state=torch.ones((2, 3, 2), device=device),
        component_names=("first", "second"),
        raw_decision_logits=logits,
        decision_available=~inactive,
        uncertainty=torch.ones(2, device=device),
        family_gates=torch.full((2, 2), 0.5, device=device),
        counterfactual_decision_logits=torch.zeros((2, 2, 2), device=device),
        intrinsic_risk=torch.full((2,), 0.2, device=device),
        effective_risk=torch.full((2,), 0.2, device=device),
    )


@pytest.mark.parametrize("target_name", ["decision_targets", "transition_targets"])
@pytest.mark.parametrize("invalid_dtype", [torch.float32, torch.bool, torch.complex64])
def test_direct_loss_rejects_noninteger_class_targets(
    target_name: str, invalid_dtype: torch.dtype
) -> None:
    targets = {
        "decision_targets": torch.zeros(2, dtype=torch.long),
        "transition_targets": torch.zeros(2, dtype=torch.long),
        "next_state_targets": torch.zeros(2, 3, 2),
    }
    targets[target_name] = torch.zeros(2, dtype=invalid_dtype)
    with pytest.raises(ValueError, match="targets must use an integer dtype"):
        olympus_system_loss(
            _objective_output(),
            decision_targets=targets["decision_targets"],
            transition_targets=targets["transition_targets"],
            next_state_targets=targets["next_state_targets"],
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_temporal_loss_aligns_target_device_and_precision(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA execution unavailable; this is not GPU validation")
    output = _objective_output(device)
    loss = olympus_system_loss(
        output,
        decision_targets=torch.zeros(2, dtype=torch.int32),
        transition_targets=torch.zeros(2, dtype=torch.long),
        next_state_targets=torch.zeros((2, 3, 2), dtype=torch.float64),
        temporal_mask=torch.tensor([[True, True, False], [True, False, False]]),
    )
    assert loss.temporal.device == output.next_state.device
    assert loss.temporal.dtype == output.next_state.dtype
    assert loss.temporal.item() == pytest.approx(1.0)
    assert bool(torch.isfinite(loss.total))


@pytest.mark.parametrize("tampering", ["nested_counter", "raw_config", "model_state_type"])
def test_serialized_checkpoint_rejects_rehashed_semantic_tampering(
    tmp_path: Path, tampering: str
) -> None:
    config = {"steps": 3, "precision": "fp32", "model": {"decision_classes": 2}}
    binding = {"source_sha256": "a" * 64, "training_config_sha256": digest_json(config)}
    payload: dict[str, Any] = {
        "binding": binding,
        "training_config": config,
        "source_identity": {"sha256": binding["source_sha256"]},
        "completed_steps": 1,
        "counters": {
            "completed_steps": 1,
            "best_step": 1,
            "best_validation_loss": 0.3,
            "first_loss": 0.5,
            "final_loss": 0.5,
            "maximum_gradient_norm": 0.1,
            "final_learning_rate": 0.001,
            "elapsed_seconds": 0.1,
        },
        "model_state": {"weight": torch.ones(1)},
        "optimizer_state": {"state": {}, "param_groups": [{"params": [0]}]},
        "scheduler": {"kind": "functional_warmup_cosine", "total_steps": 3},
        "precision": "fp32",
        "scaler": None,
        "initial_evaluation": {},
        "rng": capture_rng(torch.Generator().manual_seed(1), cuda=False),
    }
    pointer = save_training_checkpoint(tmp_path, payload, best=True)
    loaded = load_training_checkpoint(tmp_path, kind="latest", binding=binding)
    assert loaded["completed_steps"] == 1
    if tampering == "nested_counter":
        loaded["counters"]["completed_steps"] = 3
        expected = "nested completed-step counter"
    elif tampering == "raw_config":
        loaded["training_config"]["model"]["decision_classes"] = 4
        expected = "raw configuration digest"
    else:
        loaded["model_state"]["weight"] = "not a tensor"
        expected = "only tensors"
    checkpoint = tmp_path / pointer["path"]
    torch.save(loaded, checkpoint)
    pointer["sha256"] = file_sha256(checkpoint)
    (tmp_path / "latest.json").write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match=expected):
        load_training_checkpoint(tmp_path, kind="latest", binding=binding)
