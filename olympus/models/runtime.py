"""Resource-governed runtime and immutable checkpoints for the Olympus system."""

from __future__ import annotations

import hashlib
import io
import math
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, cast

import torch
from pydantic import Field
from torch import Tensor, nn

from olympus.core.schemas import StrictModel
from olympus.foundry.resources import ResourceGovernor
from olympus.models.olympus_system import (
    OlympusReferenceSystem,
    OlympusSystemConfig,
    OlympusSystemLoss,
    OlympusSystemLossConfig,
    OlympusSystemOutput,
    olympus_system_loss,
)

RuntimeDevice = Literal["auto", "cpu", "cuda", "mps"]
RuntimeInput = Tensor | dict[str, Tensor]


class OlympusRuntimeConfig(StrictModel):
    """Execution controls that are separate from learned model configuration."""

    device: RuntimeDevice = "auto"
    mixed_precision: bool = False
    compile_model: bool = False
    deterministic: bool = True
    resource_preflight: bool = True
    workload_lock: Path = Path("artifacts/locks/olympus-system.lock")
    min_available_bytes: int = Field(default=2 * 1024**3, ge=256 * 1024**2)
    max_swap_fraction: float = Field(default=0.90, gt=0.0, le=1.0)


class OlympusCheckpointRecord(StrictModel):
    schema_version: Literal[1] = 1
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameters: int = Field(gt=0)
    system_config: OlympusSystemConfig
    metadata: dict[str, str] = Field(default_factory=dict)


@dataclass(slots=True)
class OlympusInferenceResult:
    output: OlympusSystemOutput
    elapsed_seconds: float
    device: str
    mixed_precision_effective: bool


@dataclass(slots=True)
class OlympusTrainStepResult:
    output: OlympusSystemOutput
    loss: OlympusSystemLoss
    gradient_norm: float


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def resolve_olympus_device(requested: RuntimeDevice) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def _move_inputs(
    inputs: Mapping[str, RuntimeInput], device: torch.device
) -> dict[str, RuntimeInput]:
    moved: dict[str, RuntimeInput] = {}
    for name, value in inputs.items():
        if isinstance(value, Tensor):
            moved[name] = value.to(device=device)
        else:
            moved[name] = {key: tensor.to(device=device) for key, tensor in value.items()}
    return moved


class OlympusSystemRuntime:
    """Bounded inference runtime with explicit resource and precision controls."""

    def __init__(
        self,
        model: OlympusReferenceSystem,
        config: OlympusRuntimeConfig | None = None,
    ) -> None:
        self.config = config or OlympusRuntimeConfig()
        self.device = resolve_olympus_device(self.config.device)
        with self._resource_context():
            self.model = model.to(self.device)
            self._callable: nn.Module = self.model
            if self.config.compile_model:
                self._callable = cast(nn.Module, torch.compile(self.model, dynamic=False))

    def _resource_context(self) -> ResourceGovernor | nullcontext[None]:
        if not self.config.resource_preflight:
            return nullcontext()
        return ResourceGovernor(
            self.config.workload_lock,
            min_available_bytes=self.config.min_available_bytes,
            max_swap_fraction=self.config.max_swap_fraction,
        )

    @property
    def mixed_precision_effective(self) -> bool:
        return self.config.mixed_precision and self.device.type in {"cuda", "mps"}

    def infer(self, inputs: Mapping[str, RuntimeInput]) -> OlympusInferenceResult:
        precision = (
            torch.autocast(device_type=self.device.type, dtype=torch.float16)
            if self.mixed_precision_effective
            else nullcontext()
        )
        started = time.perf_counter()
        with (
            self._resource_context(),
            _deterministic_algorithms(self.config.deterministic),
            _evaluation_mode(self.model),
            precision,
        ):
            moved = _move_inputs(inputs, self.device)
            output = cast(OlympusSystemOutput, self._callable(**moved))
        elapsed = time.perf_counter() - started
        if not bool(torch.isfinite(output.raw_decision_logits).all()):
            raise FloatingPointError("Olympus inference produced non-finite decision logits")
        return OlympusInferenceResult(
            output=output,
            elapsed_seconds=elapsed,
            device=str(self.device),
            mixed_precision_effective=self.mixed_precision_effective,
        )


@contextmanager
def _evaluation_mode(model: nn.Module) -> Iterator[None]:
    """Restore every module's mode, including deliberately frozen submodules."""

    previous_modes = tuple((module, module.training) for module in model.modules())
    model.eval()
    try:
        yield
    finally:
        for module, training in previous_modes:
            module.training = training


@contextmanager
def _deterministic_algorithms(enabled: bool) -> Iterator[None]:
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    if enabled:
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with torch.inference_mode():
            yield
    finally:
        if enabled:
            torch.use_deterministic_algorithms(
                previous_enabled,
                warn_only=previous_warn_only,
            )


def train_olympus_step(
    model: OlympusReferenceSystem,
    optimizer: torch.optim.Optimizer,
    inputs: Mapping[str, RuntimeInput],
    targets: Mapping[str, Tensor],
    *,
    gradient_clip_norm: float = 1.0,
    loss_config: OlympusSystemLossConfig | None = None,
) -> OlympusTrainStepResult:
    """Execute one finite, clipped joint-system optimization step."""

    if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0.0:
        raise ValueError("gradient clip norm must be finite and positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = cast(OlympusSystemOutput, model(**dict(inputs)))
    required_targets = {
        "decision_targets",
        "transition_targets",
        "next_state_targets",
    }
    optional_targets = {"intrinsic_risk_targets"}
    missing_targets = sorted(required_targets - set(targets))
    unexpected_targets = sorted(set(targets) - required_targets - optional_targets)
    if missing_targets or unexpected_targets:
        raise ValueError(
            "training targets do not match the Olympus loss contract; "
            f"missing={missing_targets}, unexpected={unexpected_targets}"
        )
    temporal_mask = inputs.get("temporal_mask")
    if temporal_mask is not None and not isinstance(temporal_mask, Tensor):
        raise TypeError("temporal mask must be a tensor")
    weights = (loss_config or OlympusSystemLossConfig()).model_dump()
    loss = olympus_system_loss(
        output,
        decision_targets=targets["decision_targets"],
        transition_targets=targets["transition_targets"],
        next_state_targets=targets["next_state_targets"],
        intrinsic_risk_targets=targets.get("intrinsic_risk_targets"),
        temporal_mask=temporal_mask,
        **weights,
    )
    if not bool(torch.isfinite(loss.total)):
        raise FloatingPointError("Olympus training loss is non-finite")
    loss.total.backward()  # type: ignore[no-untyped-call]
    gradient = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    if not bool(torch.isfinite(gradient)):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Olympus gradient norm is non-finite")
    optimizer.step()
    return OlympusTrainStepResult(
        output=output,
        loss=loss,
        gradient_norm=float(gradient.detach().item()),
    )


def save_olympus_checkpoint(
    model: OlympusReferenceSystem,
    path: Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> OlympusCheckpointRecord:
    """Write a new immutable checkpoint and return its content identity."""

    if path.exists():
        raise FileExistsError("refusing to overwrite an existing Olympus checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model_config": model.config.model_dump(mode="json"),
        "model_state": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and refuses an existing destination.
        # An exists()/replace() pair can overwrite another writer's checkpoint.
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError("refusing to overwrite an existing Olympus checkpoint") from error
    finally:
        temporary.unlink(missing_ok=True)
    content = path.read_bytes()
    return OlympusCheckpointRecord(
        path=str(path.resolve()),
        sha256=_sha256(content),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        system_config=model.config,
        metadata=dict(metadata or {}),
    )


def load_olympus_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[OlympusReferenceSystem, OlympusCheckpointRecord]:
    """Load a digest-pinned checkpoint without allowing arbitrary pickle objects."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint must be a regular non-symlink file")
    content = path.read_bytes()
    observed = _sha256(content)
    if observed != expected_sha256:
        raise ValueError("checkpoint digest does not match the expected SHA-256")
    payload: Any = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("checkpoint has an unsupported schema")
    raw_config = payload.get("model_config")
    raw_state = payload.get("model_state")
    raw_metadata = payload.get("metadata", {})
    if not isinstance(raw_config, dict) or not isinstance(raw_state, dict):
        raise ValueError("checkpoint is missing its model configuration or state")
    if not isinstance(raw_metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw_metadata.items()
    ):
        raise ValueError("checkpoint metadata must map strings to strings")
    model_config = OlympusSystemConfig.model_validate(raw_config)
    model = OlympusReferenceSystem(model_config)
    model.load_state_dict(cast(dict[str, Tensor], raw_state), strict=True)
    model.eval()
    record = OlympusCheckpointRecord(
        path=str(path.resolve()),
        sha256=observed,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        system_config=model_config,
        metadata=cast(dict[str, str], raw_metadata),
    )
    return model, record
