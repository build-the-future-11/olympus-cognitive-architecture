"""Mean-preserving variance fitting; an engineering application of Stirn et al. (2023)."""

from __future__ import annotations

import copy
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class DynamicsConfig:
    history: int = 6
    horizon: int = 8
    state_dim: int = 4
    action_dim: int = 2
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class FaithfulTrainingConfig:
    learning_rate: float = 0.002
    weight_decay: float = 0.01
    mean_clip: float = 1.0
    variance_clip: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
                raise ValueError(f"{name} must be finite and within its nonnegative/positive range")


def _finite(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _matching(value: Tensor, expected: Tensor, name: str) -> None:
    _finite(value, name)
    if value.shape != expected.shape or value.dtype != expected.dtype:
        raise ValueError(f"{name} has an incompatible shape or dtype")
    if value.device != expected.device:
        raise ValueError(f"{name} has an incompatible device")


class ResidualDynamics(nn.Module):
    """The frozen v2 residual arithmetic, with explicit input contracts."""

    def __init__(self, config: DynamicsConfig | None = None) -> None:
        super().__init__()
        config = DynamicsConfig() if config is None else config
        if not isinstance(config, DynamicsConfig):
            raise ValueError("config must be DynamicsConfig")
        self.config = config
        inputs = config.state_dim + config.action_dim
        self.encoder = nn.GRU(inputs, config.hidden_dim, batch_first=True)
        self.cell = nn.GRUCell(inputs, config.hidden_dim)
        self.mean = nn.Linear(config.hidden_dim, config.state_dim)

    def forward_features(self, history: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        _finite(history, "history")
        _finite(actions, "actions")
        config = self.config
        if (
            history.ndim != 3
            or history.shape[0] == 0
            or history.shape[1:] != (config.history, config.state_dim + config.action_dim)
        ):
            raise ValueError("history has an incompatible shape or empty batch")
        if actions.shape != (history.shape[0], config.horizon, config.action_dim):
            raise ValueError("actions has an incompatible shape")
        parameter = next(self.parameters())
        for name, tensor in (("history", history), ("actions", actions)):
            if tensor.dtype != parameter.dtype or tensor.device != parameter.device:
                raise ValueError(f"{name} must match model dtype and device")
        _, hidden = self.encoder(history)
        cursor = hidden[-1]
        state = history[:, -1, : config.state_dim]
        means: list[Tensor] = []
        features: list[Tensor] = []
        for step in range(config.horizon):
            cursor = self.cell(torch.cat((state, actions[:, step]), -1), cursor)
            state = state + self.mean(cursor)
            means.append(state)
            features.append(cursor)
        result = torch.stack(means, 1)
        _finite(result, "predicted mean")
        return result, torch.stack(features, 1)

    def forward(self, history: Tensor, actions: Tensor) -> Tensor:
        return self.forward_features(history, actions)[0]


class FaithfulDynamics(nn.Module):
    """Copy the mean explicitly; variance gradients cannot enter its trajectory."""

    def __init__(self, reference: ResidualDynamics) -> None:
        super().__init__()
        if not isinstance(reference, ResidualDynamics):
            raise ValueError("reference must be a ResidualDynamics model")
        self.mean_model = copy.deepcopy(reference)
        parameter = next(reference.parameters())
        if parameter.dtype not in (torch.float32, torch.float64):
            raise ValueError("mean model must use float32 or float64")
        self.log_variance = nn.Linear(reference.config.hidden_dim, reference.config.state_dim)
        self.log_variance.to(device=parameter.device, dtype=parameter.dtype)

    def forward(self, history: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        mean, features = self.mean_model.forward_features(history, actions)
        log_variance = self.log_variance(features.detach())
        _finite(log_variance, "predicted log variance")
        return mean, log_variance.clamp(-8.0, 5.0)

    def losses(self, history: Tensor, actions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_variance = self(history, actions)
        _matching(targets, mean, "targets")
        observed = targets.detach()
        mean_loss = (mean - observed).square().mean()
        variance_loss = (
            0.5
            * ((mean.detach() - observed).square() * (-log_variance).exp() + log_variance).mean()
        )
        _finite(mean_loss, "mean loss")
        _finite(variance_loss, "variance loss")
        return mean_loss, variance_loss


def _check_optimizer(optimizer: torch.optim.AdamW, state: Any, updates: int) -> None:
    expected = optimizer.state_dict()
    if not isinstance(state, dict) or set(state) != {"param_groups", "state"}:
        raise ValueError("invalid optimizer state structure")
    if state["param_groups"] != expected["param_groups"]:
        raise ValueError("optimizer parameter mapping or configuration mismatch")
    identifiers = expected["param_groups"][0]["params"]
    entries = state["state"]
    if not isinstance(entries, dict) or set(entries) != (set(identifiers) if updates else set()):
        raise ValueError("optimizer state does not cover the expected parameters")
    parameters = optimizer.param_groups[0]["params"]
    for identifier, parameter in zip(identifiers, parameters, strict=True):
        if not updates:
            continue
        row = entries[identifier]
        if not isinstance(row, dict) or set(row) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("invalid AdamW state fields")
        _finite(row["step"], "optimizer step")
        if row["step"].ndim != 0 or float(row["step"].item()) != updates:
            raise ValueError("optimizer step disagrees with completed update count")
        for key in ("exp_avg", "exp_avg_sq"):
            _matching(row[key], parameter, f"optimizer {key}")
        if bool((row["exp_avg_sq"] < 0).any()):
            raise ValueError("optimizer second moment cannot be negative")


class FaithfulDynamicsTrainer:
    """Disjoint AdamW and clipping; no data selection or scientific campaign."""

    def __init__(
        self, model: FaithfulDynamics, config: FaithfulTrainingConfig | None = None
    ) -> None:
        config = FaithfulTrainingConfig() if config is None else config
        if not isinstance(model, FaithfulDynamics) or not isinstance(
            config, FaithfulTrainingConfig
        ):
            raise ValueError("trainer requires a faithful model and training configuration")
        self.model = model
        self.config = config
        self.mean_optimizer = self._optimizer(model.mean_model.parameters())
        self.variance_optimizer = self._optimizer(model.log_variance.parameters())
        self.updates = 0

    def _optimizer(self, parameters: Any) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            parameters, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )

    def step(self, history: Tensor, actions: Tensor, targets: Tensor) -> dict[str, float]:
        self.mean_optimizer.zero_grad(set_to_none=True)
        self.variance_optimizer.zero_grad(set_to_none=True)
        mean_loss, variance_loss = self.model.losses(history, actions, targets)
        torch.autograd.backward(mean_loss)
        torch.autograd.backward(variance_loss)
        nn.utils.clip_grad_norm_(
            self.model.mean_model.parameters(), self.config.mean_clip, error_if_nonfinite=True
        )
        nn.utils.clip_grad_norm_(
            self.model.log_variance.parameters(), self.config.variance_clip, error_if_nonfinite=True
        )
        self.mean_optimizer.step()
        self.variance_optimizer.step()
        for name, value in self.model.named_parameters():
            _finite(value, f"updated parameter {name}")
        self.updates += 1
        return {"mean_loss": mean_loss.item(), "variance_loss": variance_loss.item()}

    def save_checkpoint(self, path: Path) -> None:
        """Atomically create a new file; never replace an existing checkpoint."""
        if next(self.model.parameters()).dtype not in (torch.float32, torch.float64):
            raise ValueError("checkpoint model must use float32 or float64")
        for name, value in self.model.state_dict().items():
            _finite(value, name)
        _check_optimizer(
            self._optimizer(self.model.mean_model.parameters()),
            self.mean_optimizer.state_dict(),
            self.updates,
        )
        _check_optimizer(
            self._optimizer(self.model.log_variance.parameters()),
            self.variance_optimizer.state_dict(),
            self.updates,
        )
        payload = {
            "schema_version": 1,
            "architecture": asdict(self.model.mean_model.config),
            "training": asdict(self.config),
            "dtype": str(next(self.model.parameters()).dtype),
            "model": self.model.state_dict(),
            "mean_optimizer": self.mean_optimizer.state_dict(),
            "variance_optimizer": self.variance_optimizer.state_dict(),
            "updates": self.updates,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load_checkpoint(cls, path: Path) -> FaithfulDynamicsTrainer:
        """Load validated state on CPU; callers must supply the identical next batch."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        fields = {
            "schema_version",
            "architecture",
            "training",
            "dtype",
            "model",
            "mean_optimizer",
            "variance_optimizer",
            "updates",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("invalid checkpoint schema")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("unsupported checkpoint schema version")
        updates = payload["updates"]
        if type(updates) is not int or updates < 0:
            raise ValueError("invalid completed update count")
        dtypes = {"torch.float32": torch.float32, "torch.float64": torch.float64}
        if not isinstance(payload["dtype"], str):
            raise ValueError("invalid checkpoint dtype")
        dtype = dtypes.get(payload["dtype"])
        if dtype is None:
            raise ValueError("unsupported checkpoint dtype")
        if not isinstance(payload["architecture"], dict) or set(payload["architecture"]) != set(
            asdict(DynamicsConfig())
        ):
            raise ValueError("invalid checkpoint architecture configuration")
        if not isinstance(payload["training"], dict) or set(payload["training"]) != set(
            asdict(FaithfulTrainingConfig())
        ):
            raise ValueError("invalid checkpoint optimizer configuration")
        reference = ResidualDynamics(DynamicsConfig(**payload["architecture"])).to(dtype=dtype)
        trainer = cls(FaithfulDynamics(reference), FaithfulTrainingConfig(**payload["training"]))
        expected = trainer.model.state_dict()
        if not isinstance(payload["model"], dict) or set(payload["model"]) != set(expected):
            raise ValueError("checkpoint parameter identities do not match architecture")
        for name, value in payload["model"].items():
            _matching(value, expected[name], f"checkpoint {name}")
        _check_optimizer(trainer.mean_optimizer, payload["mean_optimizer"], updates)
        _check_optimizer(trainer.variance_optimizer, payload["variance_optimizer"], updates)
        trainer.model.load_state_dict(payload["model"], strict=True)
        trainer.mean_optimizer.load_state_dict(payload["mean_optimizer"])
        trainer.variance_optimizer.load_state_dict(payload["variance_optimizer"])
        trainer.updates = updates
        return trainer
