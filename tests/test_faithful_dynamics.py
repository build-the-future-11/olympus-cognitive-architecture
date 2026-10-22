from __future__ import annotations

import ast
import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from olympus.models.faithful_dynamics import (
    DynamicsConfig,
    FaithfulDynamics,
    FaithfulDynamicsTrainer,
    FaithfulTrainingConfig,
    ResidualDynamics,
)


@pytest.fixture(autouse=True)
def deterministic_artificial_tensors() -> Iterator[None]:
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8137)
        yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


def _batch() -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.randn(3, 6, 6, dtype=torch.float64),
        torch.randn(3, 8, 2, dtype=torch.float64),
        torch.randn(3, 8, 4, dtype=torch.float64),
    )


def _models() -> tuple[ResidualDynamics, FaithfulDynamicsTrainer]:
    reference = ResidualDynamics().double()
    candidate = FaithfulDynamics(reference)
    return reference, FaithfulDynamicsTrainer(candidate)


def _assert_identical(first: Any, second: Any) -> None:
    if isinstance(first, Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert set(first) == set(second)
        for key in first:
            _assert_identical(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            _assert_identical(left, right)
    else:
        assert first == second


def _control_step(
    model: ResidualDynamics, optimizer: torch.optim.AdamW, batch: tuple[Tensor, Tensor, Tensor]
) -> None:
    history, actions, targets = batch
    optimizer.zero_grad(set_to_none=True)
    torch.autograd.backward((model(history, actions) - targets).square().mean())
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()


def test_reference_arithmetic_matches_the_untouched_frozen_v2_class() -> None:
    source = Path(__file__).parents[1] / "scripts/run_multi_horizon_dynamics_v2.py"
    definition = next(
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.ClassDef) and node.name == "Dynamics"
    )
    namespace: dict[str, Any] = {
        "torch": torch,
        "nn": nn,
        "Tensor": Tensor,
        "HIDDEN_DIM": 32,
        "STATE_DIM": 4,
        "ACTION_DIM": 2,
        "HORIZON": 8,
    }
    # Only define the retained class; never execute the frozen campaign function.
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    frozen = namespace["Dynamics"]("residual").double()
    reference = ResidualDynamics().double()
    reference.load_state_dict(frozen.state_dict())
    history, actions, _ = _batch()
    assert torch.equal(reference(history, actions), frozen(history, actions)[0])


def test_all_mean_gradients_are_identical_and_variance_is_disconnected() -> None:
    reference, trainer = _models()
    history, actions, targets = _batch()
    torch.autograd.backward((reference(history, actions) - targets).square().mean())
    mean_loss, variance_loss = trainer.model.losses(history, actions, targets)
    torch.autograd.backward(variance_loss)
    assert all(parameter.grad is None for parameter in trainer.model.mean_model.parameters())
    assert any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for parameter in trainer.model.log_variance.parameters()
    )
    torch.autograd.backward(mean_loss)
    for baseline, candidate in zip(
        reference.parameters(), trainer.model.mean_model.parameters(), strict=True
    ):
        assert baseline.data_ptr() != candidate.data_ptr()
        assert baseline.grad is not None and candidate.grad is not None
        assert torch.equal(baseline.grad, candidate.grad)


def test_multiple_adamw_updates_preserve_mean_state_while_variance_learns() -> None:
    reference, trainer = _models()
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.002, weight_decay=0.01)
    initial_variance = copy.deepcopy(trainer.model.log_variance.state_dict())
    for _ in range(4):
        batch = _batch()
        _control_step(reference, optimizer, batch)
        trainer.step(*batch)
        _assert_identical(reference.state_dict(), trainer.model.mean_model.state_dict())
        _assert_identical(optimizer.state_dict(), trainer.mean_optimizer.state_dict())
    assert any(
        not torch.equal(value, initial_variance[name])
        for name, value in trainer.model.log_variance.state_dict().items()
    )
    assert trainer.updates == 4


def test_checkpoint_roundtrip_preserves_outputs_both_optimizers_and_next_step(
    tmp_path: Path,
) -> None:
    _, trainer = _models()
    trainer.step(*_batch())
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path)
    original_bytes = path.read_bytes()
    restored = FaithfulDynamicsTrainer.load_checkpoint(path)
    batch = _batch()
    _assert_identical(trainer.model(*batch[:2]), restored.model(*batch[:2]))
    _assert_identical(trainer.mean_optimizer.state_dict(), restored.mean_optimizer.state_dict())
    _assert_identical(
        trainer.variance_optimizer.state_dict(), restored.variance_optimizer.state_dict()
    )
    assert trainer.step(*batch) == restored.step(*batch)
    _assert_identical(trainer.model.state_dict(), restored.model.state_dict())
    _assert_identical(trainer.mean_optimizer.state_dict(), restored.mean_optimizer.state_dict())
    _assert_identical(
        trainer.variance_optimizer.state_dict(), restored.variance_optimizer.state_dict()
    )
    with pytest.raises(FileExistsError):
        trainer.save_checkpoint(path)
    assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [path]


def test_joint_global_clipping_can_change_the_mean_optimizer_trajectory() -> None:
    reference, trainer = _models()
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.002, weight_decay=0.01)
    batch = _batch()
    _control_step(reference, optimizer, batch)
    mean_loss, variance_loss = trainer.model.losses(*batch)
    torch.autograd.backward(mean_loss + 1e8 * variance_loss)
    # Deliberately WRONG: coupling unrelated variance gradients into the mean norm.
    nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0, error_if_nonfinite=True)
    trainer.mean_optimizer.step()
    assert any(
        not torch.equal(left, right)
        for left, right in zip(
            reference.parameters(), trainer.model.mean_model.parameters(), strict=True
        )
    )


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_inputs_are_rejected_without_updating(index: int, invalid: float) -> None:
    _, trainer = _models()
    original = copy.deepcopy(trainer.model.state_dict())
    batch = list(_batch())
    batch[index].flatten()[0] = invalid
    with pytest.raises(ValueError, match="finite"):
        trainer.step(*batch)
    _assert_identical(original, trainer.model.state_dict())
    assert trainer.updates == 0


@pytest.mark.parametrize("index", [0, 1, 2])
def test_malformed_shapes_are_rejected(index: int) -> None:
    _, trainer = _models()
    batch = list(_batch())
    batch[index] = batch[index][:, :-1]
    with pytest.raises(ValueError, match="shape"):
        trainer.step(*batch)


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_invalid_architecture_is_rejected(invalid: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        DynamicsConfig(hidden_dim=invalid)


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("inf"), float("nan"), True])
def test_invalid_optimizer_configuration_is_rejected(invalid: Any) -> None:
    with pytest.raises(ValueError):
        FaithfulTrainingConfig(mean_clip=invalid)


@pytest.mark.parametrize("corruption", ["parameter", "moment", "step", "mapping", "config"])
def test_corrupt_checkpoint_is_rejected(tmp_path: Path, corruption: str) -> None:
    _, trainer = _models()
    trainer.step(*_batch())
    valid = tmp_path / "valid.pt"
    trainer.save_checkpoint(valid)
    payload = torch.load(valid, weights_only=True)
    if corruption == "parameter":
        next(iter(payload["model"].values())).flatten()[0] = float("nan")
    elif corruption == "moment":
        payload["mean_optimizer"]["state"][0]["exp_avg_sq"].flatten()[0] = -1.0
    elif corruption == "step":
        payload["updates"] += 1
    elif corruption == "mapping":
        payload["mean_optimizer"]["param_groups"][0]["params"].reverse()
    else:
        payload["mean_optimizer"]["param_groups"][0]["lr"] = 0.3
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)
    with pytest.raises(ValueError):
        FaithfulDynamicsTrainer.load_checkpoint(broken)


def test_fresh_checkpoint_preserves_initial_model_and_empty_optimizer_states(
    tmp_path: Path,
) -> None:
    _, trainer = _models()
    path = tmp_path / "initial.pt"
    trainer.save_checkpoint(path)
    restored = FaithfulDynamicsTrainer.load_checkpoint(path)
    assert restored.updates == 0
    for original, recovered in (
        (trainer.mean_optimizer, restored.mean_optimizer),
        (trainer.variance_optimizer, restored.variance_optimizer),
    ):
        assert recovered.state_dict()["state"] == {}
        _assert_identical(original.state_dict(), recovered.state_dict())
    _assert_identical(trainer.model.state_dict(), restored.model.state_dict())
    history, actions, _ = _batch()
    _assert_identical(trainer.model(history, actions), restored.model(history, actions))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("unexpected", True, "invalid checkpoint schema"),
        ("schema_version", 2, "unsupported checkpoint schema version"),
        ("updates", True, "invalid completed update count"),
        ("dtype", 0, "invalid checkpoint dtype"),
        ("dtype", "torch.float16", "unsupported checkpoint dtype"),
        ("architecture", {}, "invalid checkpoint architecture configuration"),
        ("training", {}, "invalid checkpoint optimizer configuration"),
        ("model", {}, "checkpoint parameter identities"),
        ("mean_optimizer", {}, "invalid optimizer state structure"),
    ],
)
def test_checkpoint_admission_rejects_malformed_metadata_without_training(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    _, trainer = _models()
    valid = tmp_path / "initial.pt"
    trainer.save_checkpoint(valid)
    original_bytes = valid.read_bytes()
    payload = torch.load(valid, weights_only=True)
    payload[key] = value
    broken = tmp_path / "malformed.pt"
    torch.save(payload, broken)
    with pytest.raises(ValueError, match=message):
        FaithfulDynamicsTrainer.load_checkpoint(broken)
    assert valid.read_bytes() == original_bytes
    assert trainer.updates == 0
