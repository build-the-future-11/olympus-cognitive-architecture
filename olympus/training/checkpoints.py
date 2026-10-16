"""Versioned optimizer-boundary checkpoints; load only pinned weights-only payloads."""

from __future__ import annotations

import io
import json
import math
import os
import random
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import torch

from olympus.training.provenance import atomic_json, digest_json, file_sha256, sync_directory

CHECKPOINT_SCHEMA = 2


def _validate_checkpoint_contract(
    payload: dict[str, Any], pointer: dict[str, Any], binding: dict[str, Any]
) -> None:
    """Validate the resume boundary using primitive metadata only, before mutation."""

    for label, record in (("payload", payload), ("pointer", pointer)):
        if (
            type(record.get("schema_version")) is not int
            or record["schema_version"] != CHECKPOINT_SCHEMA
        ):
            raise ValueError(f"unsupported checkpoint {label} schema")
        if record.get("binding") != binding:
            raise ValueError(f"checkpoint {label} identity mismatch")
    config = payload.get("training_config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("checkpoint must contain its training and model configuration")
    raw_config = dict(config)
    raw_config.pop("workload_lock", None)
    try:
        config_sha256 = digest_json(raw_config)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint configuration must contain finite JSON values") from error
    if config_sha256 != binding.get("training_config_sha256"):
        raise ValueError("checkpoint raw configuration digest mismatch")
    total_steps = config.get("steps")
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("checkpoint total step budget must be a positive integer")
    completed = payload.get("completed_steps")
    if type(completed) is not int or not 0 <= completed <= total_steps:
        raise ValueError("checkpoint completed steps fall outside the declared training budget")
    if type(pointer.get("completed_steps")) is not int or pointer["completed_steps"] != completed:
        raise ValueError("checkpoint pointer counter mismatch")
    counters = payload.get("counters")
    if not isinstance(counters, dict):
        raise ValueError("checkpoint counters must be a mapping")
    if type(counters.get("completed_steps")) is not int or counters["completed_steps"] != completed:
        raise ValueError("checkpoint nested completed-step counter mismatch")
    best_step = counters.get("best_step")
    if type(best_step) is not int or not 0 <= best_step <= completed:
        raise ValueError("checkpoint best step must identify a completed optimizer boundary")
    for name in (
        "best_validation_loss",
        "maximum_gradient_norm",
        "final_learning_rate",
        "elapsed_seconds",
    ):
        value = counters.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"checkpoint {name} must be finite")
        if name != "best_validation_loss" and value < 0:
            raise ValueError(f"checkpoint {name} must be non-negative")
    for name in ("first_loss", "final_loss"):
        value = counters.get(name)
        if completed == 0:
            if name not in counters or value is not None:
                raise ValueError(f"checkpoint {name} must be null before the first optimizer step")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"checkpoint {name} must be finite after optimizer execution")
    scheduler = payload.get("scheduler")
    if (
        not isinstance(scheduler, dict)
        or scheduler.get("kind") != "functional_warmup_cosine"
        or type(scheduler.get("total_steps")) is not int
        or scheduler["total_steps"] != total_steps
    ):
        raise ValueError("checkpoint scheduler does not match the fixed training budget")
    if (
        payload.get("precision") != "fp32"
        or config.get("precision") != "fp32"
        or "scaler" not in payload
        or payload["scaler"] is not None
    ):
        raise ValueError("checkpoint supports only the declared fp32 path without an AMP scaler")
    model_state = payload.get("model_state")
    if (
        not isinstance(model_state, dict)
        or not model_state
        or not all(isinstance(name, str) and name for name in model_state)
    ):
        raise ValueError("checkpoint model state must be a non-empty named mapping")
    optimizer = payload.get("optimizer_state")
    if (
        not isinstance(optimizer, dict)
        or not isinstance(optimizer.get("state"), dict)
        or not isinstance(optimizer.get("param_groups"), list)
        or not optimizer["param_groups"]
        or not all(
            isinstance(group, dict)
            and isinstance(group.get("params"), list)
            and all(type(index) is int and index >= 0 for index in group["params"])
            for group in optimizer["param_groups"]
        )
    ):
        raise ValueError("checkpoint optimizer state must contain state and parameter groups")
    source = payload.get("source_identity")
    if not isinstance(source, dict) or source.get("sha256") != binding.get("source_sha256"):
        raise ValueError("checkpoint source identity does not match its binding")
    if not isinstance(payload.get("initial_evaluation"), dict):
        raise ValueError("checkpoint initial evaluation must be a mapping")
    rng = payload.get("rng")
    if not isinstance(rng, dict) or not {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "sampler",
    }.issubset(rng):
        raise ValueError("checkpoint must contain every supported RNG state")


def _validate_checkpoint_tensors(payload: dict[str, Any]) -> None:
    if not all(isinstance(value, torch.Tensor) for value in payload["model_state"].values()):
        raise ValueError("checkpoint model state must contain only tensors")
    rng = payload["rng"]
    if not all(isinstance(rng[name], torch.Tensor) for name in ("torch_cpu", "sampler")):
        raise ValueError("checkpoint CPU and sampler RNG states must be tensors")
    if not isinstance(rng["torch_cuda"], list) or not all(
        isinstance(value, torch.Tensor) for value in rng["torch_cuda"]
    ):
        raise ValueError("checkpoint CUDA RNG state must be a list of tensors")


def capture_rng(generator: torch.Generator, *, cuda: bool) -> dict[str, Any]:
    algorithm, keys, position, has_gauss, cached_gauss = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [algorithm, keys.tolist(), position, has_gauss, cached_gauss],
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if cuda else [],
        "sampler": generator.get_state(),
    }


def restore_rng(state: dict[str, Any], generator: torch.Generator, *, cuda: bool) -> None:
    random.setstate(state["python"])
    algorithm, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state(
        (algorithm, np.array(keys, dtype=np.uint32), position, has_gauss, cached_gauss)
    )
    torch.set_rng_state(state["torch_cpu"])
    generator.set_state(state["sampler"])
    if cuda:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("resume CUDA device count differs from the saved RNG state")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(
    directory: Path, payload: dict[str, Any], *, best: bool
) -> dict[str, Any]:
    """Publish immutable contents, then atomically move latest/best metadata pointers.

    A killed write leaves the prior pointer intact. Unreferenced partial files never
    count as checkpoints. Old snapshots are retained for interruption provenance.
    """
    validated_payload = {**payload, "schema_version": CHECKPOINT_SCHEMA}
    contract_pointer = {
        "schema_version": CHECKPOINT_SCHEMA,
        "completed_steps": payload.get("completed_steps"),
        "binding": payload.get("binding"),
    }
    binding = payload.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("checkpoint binding must be a mapping")
    _validate_checkpoint_contract(validated_payload, contract_pointer, binding)
    _validate_checkpoint_tensors(validated_payload)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"step-{payload['completed_steps']:09d}-{uuid.uuid4().hex}.pt"
    target = directory / filename
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=directory, prefix=".checkpoint-", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(validated_payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        sync_directory(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    pointer = {
        "schema_version": CHECKPOINT_SCHEMA,
        "path": filename,
        "sha256": file_sha256(target),
        "completed_steps": payload["completed_steps"],
        "binding": payload["binding"],
    }
    # Best first: latest always records a recoverable best checkpoint.
    if best:
        atomic_json(directory / "best.json", pointer)
    atomic_json(directory / "latest.json", pointer)
    return pointer


def load_training_checkpoint(
    directory: Path, *, kind: str, binding: dict[str, Any]
) -> dict[str, Any]:
    if kind not in {"best", "latest"}:
        raise ValueError("checkpoint must be best or latest")
    pointer_path = directory / f"{kind}.json"
    if pointer_path.is_symlink():
        raise ValueError("checkpoint pointer must not be a symlink")
    pointer = json.loads(pointer_path.read_text())
    if not isinstance(pointer, dict):
        raise ValueError("checkpoint pointer must be a mapping")
    filename = pointer.get("path")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("checkpoint pointer path must be a contained filename")
    path = directory / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint must be a regular non-symlink file")
    content = path.read_bytes()
    import hashlib

    if hashlib.sha256(content).hexdigest() != pointer.get("sha256"):
        raise ValueError("training checkpoint digest mismatch")
    if pointer.get("binding") != binding or pointer.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint source, configuration, data or environment identity mismatch")
    # No custom pickle globals or unsafe fallback. Treat supplied files as untrusted.
    payload = torch.load(io.BytesIO(content), weights_only=True, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported training checkpoint schema")
    _validate_checkpoint_contract(payload, pointer, binding)
    _validate_checkpoint_tensors(payload)
    return payload
