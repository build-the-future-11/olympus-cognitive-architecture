"""Hash-bound, resource-governed training for the composed Olympus system."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import resource
import signal
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

import numpy as np
import torch
from pydantic import Field, model_validator
from torch import Tensor

from olympus.core.schemas import StrictModel
from olympus.data.olympus_dataset import LoadedOlympusDataset, load_olympus_dataset
from olympus.evaluation.decision_dossiers import (
    OlympusDecisionDossierBundle,
    OlympusDossierBinding,
    build_olympus_decision_dossiers,
    read_olympus_decision_dossier_bundle,
    verify_olympus_decision_dossier_replay,
    write_olympus_decision_dossiers,
)
from olympus.evaluation.olympus_system import (
    OlympusSystemEvaluation,
    evaluate_olympus_system,
    olympus_batch_inputs_targets,
)
from olympus.foundry.resources import ResourceGovernor
from olympus.models.olympus_system import (
    OlympusSystemConfig,
    OlympusSystemLossConfig,
    build_olympus_system,
)
from olympus.models.runtime import (
    RuntimeDevice,
    resolve_olympus_device,
    save_olympus_checkpoint,
    train_olympus_step,
)
from olympus.training.checkpoints import (
    capture_rng,
    load_training_checkpoint,
    restore_rng,
    save_training_checkpoint,
)
from olympus.training.provenance import atomic_json, digest_json, source_identity


class OlympusSystemTrainingConfig(StrictModel):
    seed: int = Field(default=20261011, ge=0, le=2**32 - 1)
    steps: int = Field(default=1_000, ge=1, le=10_000_000)
    batch_size: int = Field(default=8, ge=1, le=4_096)
    learning_rate: float = Field(default=3e-4, gt=0.0, le=0.1)
    minimum_learning_rate_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    warmup_steps: int = Field(default=0, ge=0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0, le=100.0)
    device: RuntimeDevice = "cpu"
    precision: Literal["fp32"] = "fp32"
    cpu_threads: int = Field(default=1, ge=1, le=16)
    evaluation_batch_size: int = Field(default=32, ge=1, le=4_096)
    evaluation_interval: int = Field(default=100, ge=1)
    checkpoint_interval_seconds: float = Field(default=120.0, gt=0, le=3_600)
    log_interval: int = Field(default=10, ge=1)
    tensorboard: bool = False
    deterministic: bool = True
    resource_preflight: bool = True
    workload_lock: Path = Path("artifacts/locks/olympus-system-training.lock")
    min_available_bytes: int = Field(default=2 * 1024**3, ge=256 * 1024**2)
    max_swap_fraction: float = Field(default=0.90, gt=0.0, le=1.0)
    model: OlympusSystemConfig
    loss: OlympusSystemLossConfig = Field(default_factory=OlympusSystemLossConfig)

    @model_validator(mode="after")
    def valid_schedule(self) -> OlympusSystemTrainingConfig:
        if self.warmup_steps > self.steps:
            raise ValueError("warmup steps must not exceed total steps")
        return self


class OlympusTrainingTrace(StrictModel):
    first_loss: float
    final_loss: float
    maximum_gradient_norm: float = Field(ge=0.0)
    final_learning_rate: float = Field(ge=0.0)


class OlympusSystemTrainingReport(StrictModel):
    schema_version: Literal[4] = 4
    status: Literal["COMPLETED_DEVELOPMENT_NOT_PROMOTED"]
    dataset_id: str
    dataset_version: str
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_config: OlympusSystemTrainingConfig
    completed_steps: int = Field(ge=1)
    parameters: int = Field(gt=0)
    device: str
    resource_preflight: dict[str, int | float] | None
    training: OlympusTrainingTrace
    initial_evaluation: OlympusSystemEvaluation
    final_evaluation: OlympusSystemEvaluation
    checkpoint_path: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_dossier_path: str
    decision_dossier_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_dossier_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity: dict[str, Any]
    best_step: int = Field(ge=0)
    best_validation_loss: float
    elapsed_seconds: float = Field(ge=0)
    effective_batch_size: int = Field(ge=1)
    resume_boundary: str = "completed optimizer step; independent random-subset sampler"
    promotion_authorized: Literal[False] = False
    claim_boundary: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


@contextmanager
def _deterministic_training(enabled: bool) -> Iterator[None]:
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    if enabled:
        torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        if enabled:
            torch.use_deterministic_algorithms(
                previous_enabled,
                warn_only=previous_warn_only,
            )


def _validate_split(
    split_name: str,
    batch: Mapping[str, Tensor],
    config: OlympusSystemConfig,
) -> None:
    tokens = batch["tokens"]
    if tokens.shape[1] > min(
        config.hermes.backbone.max_sequence_length,
        config.prometheus.backbone.max_sequence_length,
        config.perseus.backbone.max_sequence_length,
        config.atlas.backbone.max_sequence_length,
    ):
        raise ValueError(f"{split_name} token sequence exceeds a family context limit")
    if bool(
        (
            tokens
            >= min(
                config.hermes.backbone.vocab_size,
                config.prometheus.backbone.vocab_size,
                config.perseus.backbone.vocab_size,
                config.atlas.backbone.vocab_size,
            )
        ).any()
    ):
        raise ValueError(f"{split_name} token ids exceed a family vocabulary")
    image = batch["image"]
    if image.shape[1] != config.atlas.image_channels:
        raise ValueError(f"{split_name} image channels do not match Atlas")
    if any(size % config.atlas.patch_size for size in image.shape[-2:]):
        raise ValueError(f"{split_name} image dimensions must be divisible by the Atlas patch size")
    if batch["states"].shape[-1] != config.kronos.state_dim:
        raise ValueError(f"{split_name} state width does not match Kronos")
    if batch["actions"].shape[-1] != config.kronos.action_dim:
        raise ValueError(f"{split_name} action width does not match Kronos")
    if batch["allowed_transitions"].shape[1] != config.aion.research_stages:
        raise ValueError(f"{split_name} transition width does not match Aion")
    if bool((batch["decision_targets"] >= config.decision_classes).any()):
        raise ValueError(f"{split_name} decision targets exceed the configured classes")
    if "domain_ids" in batch and bool((batch["domain_ids"] >= config.prometheus.domains).any()):
        raise ValueError(f"{split_name} domain ids exceed the Prometheus domain count")
    for name, width in config.extension_dims.items():
        if batch[f"extension__{name}"].shape[1] != width:
            raise ValueError(f"{split_name} extension {name!r} width does not match the model")


def _validate_dataset(dataset: LoadedOlympusDataset, config: OlympusSystemTrainingConfig) -> None:
    configured_extensions = sorted(config.model.extension_dims)
    if configured_extensions != dataset.manifest.extensions:
        raise ValueError("dataset and model extension names do not match")
    if config.batch_size > dataset.train["tokens"].shape[0]:
        raise ValueError("batch size must not exceed the number of training examples")
    _validate_split("training", dataset.train, config.model)
    _validate_split("evaluation", dataset.evaluation, config.model)


def _move_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: tensor.to(device=device) for name, tensor in batch.items()}


def _learning_rate(config: OlympusSystemTrainingConfig, step: int) -> float:
    completed = step + 1
    if config.warmup_steps and completed <= config.warmup_steps:
        scale = completed / config.warmup_steps
    elif config.steps == config.warmup_steps:
        scale = 1.0
    else:
        progress = (completed - config.warmup_steps) / (config.steps - config.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = config.minimum_learning_rate_ratio
        scale = floor + (1.0 - floor) * cosine
    return config.learning_rate * scale


class TrainingInterrupted(RuntimeError):
    """A bounded run stopped with a recoverable optimizer-boundary checkpoint."""


def training_config_digest(config: OlympusSystemTrainingConfig) -> str:
    value = config.model_dump(mode="json")
    # A lock location is operational, not a scientific setting. Relocation is safe.
    value.pop("workload_lock")
    return digest_json(value)


def _configure_deterministic_cuda(enabled: bool) -> None:
    if not enabled:
        return
    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing not in {None, ":4096:8", ":16:8"}:
        raise ValueError("deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    if existing is None and torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
        raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG before initializing CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _environment(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": str(device),
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "accelerator": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


class _Metrics:
    def __init__(self, directory: Path, config: OlympusSystemTrainingConfig) -> None:
        self.path = directory / "metrics.jsonl"
        self.run_id = directory.name
        self.config = config
        self.writer: Any = None
        if config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(directory / "tensorboard"))
            except Exception as error:
                self.emit("tracking_unavailable", error_type=type(error).__name__)

    def emit(self, phase: str, **fields: Any) -> None:
        row = {
            "time_utc": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "benchmark": "olympus-composed-reference",
            "model": self.config.model.fusion_strategy,
            "seed": self.config.seed,
            "phase": phase,
            **fields,
        }
        encoded = json.dumps(row, sort_keys=True, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if self.writer is not None:
            try:
                for name, value in fields.items():
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        self.writer.add_scalar(
                            f"{phase}/{name}", value, fields.get("global_step", 0)
                        )
            except Exception as error:
                # External/optional tracking must not stop the durable local loop.
                self.writer = None
                self.emit("tracking_failed", error_type=type(error).__name__)

    def close(self) -> None:
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                self.writer = None


@contextmanager
def _run_lock(directory: Path, *, resume: bool) -> Iterator[None]:
    import fcntl

    if directory.is_symlink():
        raise ValueError("run directory must not be a symlink")
    if not resume:
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise FileExistsError("refusing to overwrite an existing Olympus system run") from error
    elif not directory.is_dir():
        raise FileNotFoundError("resume requires an existing run directory")
    lock = directory / ".training.lock"
    if lock.is_symlink():
        raise ValueError("run lock must not be a symlink")
    with lock.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("this run already has a live writer") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def train_olympus_system(
    manifest_path: Path,
    output_directory: Path,
    *,
    config: OlympusSystemTrainingConfig,
    resume: bool = False,
    max_seconds: float | None = None,
    stop_after_steps: int | None = None,
) -> OlympusSystemTrainingReport:
    """Train one source-bound candidate; resumption is at an optimizer boundary.

    The full scheduler budget stays fixed across interruptions. ``stop_after_steps``
    is an absolute completed-step boundary used for interruption tests. Wall budgets
    are cooperative: a single kernel/evaluation/checkpoint cannot be preempted.
    The best checkpoint is selected using validation total loss, including step zero.
    No protected/test split is loaded by this function.
    """
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive")
    if stop_after_steps is not None and stop_after_steps < 0:
        raise ValueError("stop_after_steps must be non-negative")
    if config.device == "mps":
        raise ValueError("resumable training supports explicit CPU or CUDA; MPS is not verified")
    directory = output_directory.resolve()
    if output_directory.is_symlink():
        raise ValueError("run directory must not be a symlink")
    with _run_lock(directory, resume=resume):
        if (directory / "report.json").exists():
            raise FileExistsError("refusing to overwrite a completed Olympus system run")
        if resume and not (directory / "checkpoints/latest.json").is_file():
            raise ValueError(
                "run has no durable checkpoint; preserve this failed attempt and use a new run root"
            )
        metrics = _Metrics(directory, config)
        try:
            governor = (
                ResourceGovernor(
                    config.workload_lock,
                    min_available_bytes=config.min_available_bytes,
                    max_swap_fraction=config.max_swap_fraction,
                )
                if config.resource_preflight
                else None
            )
            # Admission precedes decoding tensors and allocating model/device state.
            with governor if governor is not None else nullcontext():
                return _train_admitted(
                    manifest_path,
                    directory,
                    config=config,
                    resume=resume,
                    max_seconds=max_seconds,
                    stop_after_steps=stop_after_steps,
                    metrics=metrics,
                    governor=governor,
                )
        except BaseException as error:
            interrupted = isinstance(error, (TrainingInterrupted, KeyboardInterrupt))
            state = "INTERRUPTED" if interrupted else "FAILED"
            atomic_json(
                directory / "status.json",
                {
                    "status": state,
                    "error_type": type(error).__name__,
                    "resume_from": "checkpoints/latest.json"
                    if (directory / "checkpoints/latest.json").exists()
                    else None,
                },
            )
            metrics.emit(state.lower(), error_type=type(error).__name__)
            raise
        finally:
            metrics.close()


def _train_admitted(
    manifest_path: Path,
    directory: Path,
    *,
    config: OlympusSystemTrainingConfig,
    resume: bool,
    max_seconds: float | None,
    stop_after_steps: int | None,
    metrics: _Metrics,
    governor: ResourceGovernor | None,
) -> OlympusSystemTrainingReport:
    started = time.monotonic()
    _configure_deterministic_cuda(config.deterministic and config.device in {"cuda", "auto"})
    device = resolve_olympus_device(config.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("resumable training supports CPU or CUDA only")
    dataset = load_olympus_dataset(manifest_path)
    _validate_dataset(dataset, config)
    identity = source_identity()
    config_sha = training_config_digest(config)
    binding = {
        "source_sha256": identity["sha256"],
        "training_config_sha256": config_sha,
        "dataset_manifest_sha256": dataset.manifest.manifest_sha256,
        "environment": _environment(device),
    }
    contract = {
        "schema_version": 2,
        "binding": binding,
        "source_identity": identity,
        "training_config": config.model_dump(mode="json"),
        "evaluation_protocol": "validation total loss; fixed intervals; best includes initial",
    }
    contract_path = directory / "run.json"
    if resume:
        existing = json.loads(contract_path.read_text())
        if existing["binding"] != binding:
            raise ValueError("resume source, configuration, data or environment identity mismatch")
    else:
        atomic_json(contract_path, contract)

    previous_threads = torch.get_num_threads()
    previous_handler: Any = None
    signal_installed = False
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    try:
        torch.set_num_threads(config.cpu_threads)
        if threading.current_thread() is threading.main_thread():
            previous_handler = signal.signal(signal.SIGTERM, request_stop)
            signal_installed = True
        with _deterministic_training(config.deterministic):
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            generator = torch.Generator(device="cpu").manual_seed(config.seed)
            model = build_olympus_system(config.model).to(device=device, dtype=torch.float32)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            counters: dict[str, Any] = {
                "completed_steps": 0,
                "best_step": 0,
                "best_validation_loss": None,
                "first_loss": None,
                "final_loss": None,
                "maximum_gradient_norm": 0.0,
                "final_learning_rate": config.learning_rate,
                "elapsed_seconds": 0.0,
            }
            elapsed_before = 0.0
            checkpoints = directory / "checkpoints"

            def evaluate() -> OlympusSystemEvaluation:
                return evaluate_olympus_system(
                    model,
                    dataset.evaluation,
                    dataset.manifest.extensions,
                    loss_config=config.loss,
                    batch_size=config.evaluation_batch_size,
                )

            if resume:
                payload = load_training_checkpoint(checkpoints, kind="latest", binding=binding)
                model.load_state_dict(payload["model_state"], strict=True)
                optimizer.load_state_dict(payload["optimizer_state"])
                counters = payload["counters"]
                elapsed_before = counters["elapsed_seconds"]
                initial = OlympusSystemEvaluation.model_validate(payload["initial_evaluation"])
                restore_rng(payload["rng"], generator, cuda=device.type == "cuda")
                metrics.emit("resume", global_step=counters["completed_steps"])
            else:
                initial = evaluate()
                counters["best_validation_loss"] = initial.total_loss

            def save(*, best: bool = False) -> dict[str, Any]:
                counters["elapsed_seconds"] = elapsed_before + time.monotonic() - started
                return save_training_checkpoint(
                    checkpoints,
                    {
                        "binding": binding,
                        "training_config": config.model_dump(mode="json"),
                        "source_identity": identity,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler": {
                            "kind": "functional_warmup_cosine",
                            "total_steps": config.steps,
                        },
                        "scaler": None,
                        "precision": "fp32",
                        "completed_steps": counters["completed_steps"],
                        "counters": dict(counters),
                        "initial_evaluation": initial.model_dump(mode="json"),
                        "rng": capture_rng(generator, cuda=device.type == "cuda"),
                        "state_contract": (
                            "all persistent model buffers in state_dict; GRU resets per call"
                        ),
                    },
                    best=best,
                )

            if not resume:
                save(best=True)
            atomic_json(directory / "status.json", {"status": "RUNNING", "binding": binding})
            metrics.emit(
                "start",
                global_step=counters["completed_steps"],
                device=str(device),
                parameters=sum(p.numel() for p in model.parameters()),
                effective_batch_size=config.batch_size,
                precision=config.precision,
                validation_loss=initial.total_loss,
            )
            last_checkpoint = time.monotonic()
            step_times: list[float] = []
            for step in range(counters["completed_steps"], config.steps):
                remaining = (
                    None if max_seconds is None else max_seconds - (time.monotonic() - started)
                )
                estimated_step = max(step_times[-10:], default=0.0)
                if (
                    stop_requested
                    or (stop_after_steps is not None and step >= stop_after_steps)
                    or (remaining is not None and remaining <= max(1.0, 2 * estimated_step))
                ):
                    save()
                    raise TrainingInterrupted("stopped at optimizer boundary; resume from latest")
                step_started = time.monotonic()
                indices = torch.randperm(dataset.train["tokens"].shape[0], generator=generator)[
                    : config.batch_size
                ]
                selected = {
                    name: tensor.index_select(0, indices).to(
                        device=device,
                        dtype=torch.float32 if tensor.is_floating_point() else tensor.dtype,
                    )
                    for name, tensor in dataset.train.items()
                }
                inputs, targets = olympus_batch_inputs_targets(
                    selected, dataset.manifest.extensions
                )
                learning_rate = _learning_rate(config, step)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                result = train_olympus_step(
                    model,
                    optimizer,
                    inputs,
                    targets,
                    gradient_clip_norm=config.gradient_clip_norm,
                    loss_config=config.loss,
                )
                loss = float(result.loss.total.detach().item())
                if counters["first_loss"] is None:
                    counters["first_loss"] = loss
                counters.update(
                    completed_steps=step + 1,
                    final_loss=loss,
                    final_learning_rate=learning_rate,
                    maximum_gradient_norm=max(
                        counters["maximum_gradient_norm"], result.gradient_norm
                    ),
                )
                del result, selected, inputs, targets
                step_times.append(time.monotonic() - step_started)
                # Bound the timing history for arbitrarily long declared runs.
                step_times = step_times[-10:]
                if (step + 1) % config.log_interval == 0 or step == 0 or step + 1 == config.steps:
                    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    fields: dict[str, Any] = {
                        "global_step": step + 1,
                        "learning_rate": learning_rate,
                        "training_loss": loss,
                        "elapsed_seconds": elapsed_before + time.monotonic() - started,
                        "examples_per_second": config.batch_size / max(step_times[-1], 1e-9),
                        "process_peak_rss_bytes": peak_rss
                        if sys.platform == "darwin"
                        else peak_rss * 1024,
                        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else None,
                    }
                    if len(step_times) >= 5:
                        fields["estimated_training_seconds_remaining"] = (
                            sum(step_times) / len(step_times) * (config.steps - step - 1)
                        )
                    metrics.emit("training", **fields)
                if (step + 1) % config.evaluation_interval == 0 or step + 1 == config.steps:
                    validation = evaluate()
                    improved = validation.total_loss < counters["best_validation_loss"]
                    if improved:
                        counters.update(
                            best_step=step + 1, best_validation_loss=validation.total_loss
                        )
                    pointer = save(best=improved)
                    last_checkpoint = time.monotonic()
                    metrics.emit(
                        "validation",
                        global_step=step + 1,
                        validation_loss=validation.total_loss,
                        best_step=counters["best_step"],
                        checkpoint_path=pointer["path"],
                    )
                elif time.monotonic() - last_checkpoint >= config.checkpoint_interval_seconds:
                    pointer = save()
                    last_checkpoint = time.monotonic()
                    metrics.emit(
                        "checkpoint", global_step=step + 1, checkpoint_path=pointer["path"]
                    )

            # The latest snapshot remains full final-step training state; export best separately.
            best = load_training_checkpoint(checkpoints, kind="best", binding=binding)
            model.load_state_dict(best["model_state"], strict=True)
            final = evaluate()
            checkpoint_path = directory / "checkpoint.pt"
            if checkpoint_path.exists():
                # A process may have died after export but before report publication.
                # Never replace it; verify its exact weights against the selected snapshot.
                from olympus.models.runtime import load_olympus_checkpoint
                from olympus.training.provenance import file_sha256

                exported, checkpoint = load_olympus_checkpoint(
                    checkpoint_path,
                    expected_sha256=file_sha256(checkpoint_path),
                )
                expected_metadata = {
                    "dataset_manifest_sha256": str(dataset.manifest.manifest_sha256),
                    "training_config_sha256": config_sha,
                    "source_sha256": identity["sha256"],
                    "completed_steps": str(config.steps),
                    "selected_step": str(counters["best_step"]),
                    "status": "COMPLETED_DEVELOPMENT_NOT_PROMOTED",
                }
                if (
                    checkpoint.system_config != config.model
                    or checkpoint.metadata != expected_metadata
                ):
                    raise ValueError(
                        "partial final export metadata does not match the run identity"
                    )
                if any(
                    not torch.equal(value.cpu(), exported.state_dict()[name])
                    for name, value in model.state_dict().items()
                ):
                    raise ValueError("partial final export does not match selected best checkpoint")
            else:
                checkpoint = save_olympus_checkpoint(
                    model.cpu(),
                    checkpoint_path,
                    metadata={
                        "dataset_manifest_sha256": str(dataset.manifest.manifest_sha256),
                        "training_config_sha256": config_sha,
                        "source_sha256": identity["sha256"],
                        "completed_steps": str(config.steps),
                        "selected_step": str(counters["best_step"]),
                        "status": "COMPLETED_DEVELOPMENT_NOT_PROMOTED",
                    },
                )
            dossier_path = directory / "decision-dossiers.json"
            dossier_binding = OlympusDossierBinding(
                dataset_manifest_sha256=str(dataset.manifest.manifest_sha256),
                checkpoint_sha256=checkpoint.sha256,
                split="evaluation",
            )
            if dossier_path.exists():
                dossier_bundle = read_olympus_decision_dossier_bundle(
                    dossier_path,
                    expected_binding=dossier_binding,
                    expected_source_ids=dataset.manifest.evaluation.source_ids,
                )
                verify_olympus_decision_dossier_replay(
                    model,
                    dataset.evaluation,
                    dataset.manifest.extensions,
                    dataset.manifest.evaluation.source_ids,
                    binding=dossier_binding,
                    observed=dossier_bundle,
                )
            else:
                dossiers = build_olympus_decision_dossiers(
                    model,
                    dataset.evaluation,
                    dataset.manifest.extensions,
                    dataset.manifest.evaluation.source_ids,
                )
                dossier_bundle = write_olympus_decision_dossiers(
                    dossier_path,
                    binding=dossier_binding,
                    dossiers=dossiers,
                )
            if dossier_bundle.bundle_sha256 is None:
                raise ValueError("decision-dossier bundle is missing its content digest")
            report = OlympusSystemTrainingReport(
                status="COMPLETED_DEVELOPMENT_NOT_PROMOTED",
                dataset_id=dataset.manifest.dataset_id,
                dataset_version=dataset.manifest.version,
                dataset_manifest_sha256=str(dataset.manifest.manifest_sha256),
                training_config_sha256=config_sha,
                training_config=config,
                completed_steps=config.steps,
                parameters=checkpoint.parameters,
                device=str(device),
                resource_preflight=governor.preflight.as_dict()
                if governor is not None and governor.preflight is not None
                else None,
                training=OlympusTrainingTrace(
                    first_loss=counters["first_loss"],
                    final_loss=counters["final_loss"],
                    maximum_gradient_norm=counters["maximum_gradient_norm"],
                    final_learning_rate=counters["final_learning_rate"],
                ),
                initial_evaluation=initial,
                final_evaluation=final,
                checkpoint_path=checkpoint.path,
                checkpoint_sha256=checkpoint.sha256,
                decision_dossier_path=str(dossier_path),
                decision_dossier_sha256=_sha256(dossier_path.read_bytes()),
                decision_dossier_content_sha256=dossier_bundle.bundle_sha256,
                source_identity=identity,
                best_step=counters["best_step"],
                best_validation_loss=counters["best_validation_loss"],
                elapsed_seconds=elapsed_before + time.monotonic() - started,
                effective_batch_size=config.batch_size,
                claim_boundary=(
                    "Engineering/development evidence only; no protected test or promotion."
                ),
            )
            atomic_json(directory / "report.json", report.model_dump(mode="json"))
            atomic_json(directory / "status.json", {"status": "COMPLETED", "binding": binding})
            metrics.emit(
                "completed",
                global_step=config.steps,
                best_step=counters["best_step"],
                validation_loss=final.total_loss,
                checkpoint_path="checkpoint.pt",
            )
            return report
    finally:
        torch.set_num_threads(previous_threads)
        if signal_installed:
            signal.signal(signal.SIGTERM, previous_handler)


def evaluate_frozen_run(
    run_directory: Path, manifest_path: Path, *, checkpoint: str = "best"
) -> OlympusSystemEvaluation:
    """Re-evaluate a pinned checkpoint on its declared validation split, without tuning."""
    contract = json.loads((run_directory / "run.json").read_text())
    config = OlympusSystemTrainingConfig.model_validate(contract["training_config"])
    with ResourceGovernor(
        run_directory / ".decision-dossier-verification.lock",
        min_available_bytes=config.min_available_bytes,
        max_swap_fraction=config.max_swap_fraction,
    ):
        _configure_deterministic_cuda(config.deterministic and config.device in {"cuda", "auto"})
        device = resolve_olympus_device(config.device)
        dataset = load_olympus_dataset(manifest_path)
        binding = {
            "source_sha256": source_identity()["sha256"],
            "training_config_sha256": training_config_digest(config),
            "dataset_manifest_sha256": dataset.manifest.manifest_sha256,
            "environment": _environment(device),
        }
        payload = load_training_checkpoint(
            run_directory / "checkpoints", kind=checkpoint, binding=binding
        )
        model = build_olympus_system(config.model).to(device=device, dtype=torch.float32)
        model.load_state_dict(payload["model_state"], strict=True)
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(config.cpu_threads)
            with _deterministic_training(config.deterministic):
                return evaluate_olympus_system(
                    model,
                    dataset.evaluation,
                    dataset.manifest.extensions,
                    loss_config=config.loss,
                    batch_size=config.evaluation_batch_size,
                )
        finally:
            torch.set_num_threads(previous_threads)


def verify_frozen_decision_dossiers(
    run_directory: Path,
    manifest_path: Path,
) -> OlympusDecisionDossierBundle:
    """Replay a completed run's per-example dossier on its declared device."""

    report_path = run_directory / "report.json"
    if report_path.is_symlink() or not report_path.is_file():
        raise ValueError("completed run must contain a regular report.json")
    report = OlympusSystemTrainingReport.model_validate_json(report_path.read_bytes())
    dataset = load_olympus_dataset(manifest_path)
    if dataset.manifest.manifest_sha256 != report.dataset_manifest_sha256:
        raise ValueError("dataset manifest does not match the completed report")
    config = report.training_config
    if training_config_digest(config) != report.training_config_sha256:
        raise ValueError("training configuration does not match the completed report")
    if source_identity()["sha256"] != report.source_identity.get("sha256"):
        raise ValueError("current source does not match the completed report")
    dossier_path = run_directory / "decision-dossiers.json"
    checkpoint_path = run_directory / "checkpoint.pt"
    dossier_binding = OlympusDossierBinding(
        dataset_manifest_sha256=report.dataset_manifest_sha256,
        checkpoint_sha256=report.checkpoint_sha256,
        split="evaluation",
    )
    with ResourceGovernor(
        config.workload_lock,
        min_available_bytes=config.min_available_bytes,
        max_swap_fraction=config.max_swap_fraction,
    ):
        _configure_deterministic_cuda(config.deterministic and config.device in {"cuda", "auto"})
        device = resolve_olympus_device(config.device)
        if str(device) != report.device:
            raise ValueError("available device does not match the completed report")
        from olympus.models.runtime import load_olympus_checkpoint

        model, _ = load_olympus_checkpoint(
            checkpoint_path,
            expected_sha256=report.checkpoint_sha256,
        )
        model.to(device=device, dtype=torch.float32)
        observed = read_olympus_decision_dossier_bundle(
            dossier_path,
            expected_binding=dossier_binding,
            expected_source_ids=dataset.manifest.evaluation.source_ids,
            expected_file_sha256=report.decision_dossier_sha256,
        )
        if observed.bundle_sha256 != report.decision_dossier_content_sha256:
            raise ValueError("decision-dossier content digest does not match the report")
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(config.cpu_threads)
            with _deterministic_training(config.deterministic):
                verify_olympus_decision_dossier_replay(
                    model,
                    dataset.evaluation,
                    dataset.manifest.extensions,
                    dataset.manifest.evaluation.source_ids,
                    binding=dossier_binding,
                    observed=observed,
                )
        finally:
            torch.set_num_threads(previous_threads)
    return observed
