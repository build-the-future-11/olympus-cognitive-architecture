"""Candidate full-cell profiler; invoke only through the notebook account guard.

This script performs a real, unchanged Olympus development fit when invoked. It
does not allocate a platform session or meter account quota. An external caller
must enforce the account budget, absolute deadline and subprocess timeout. A
profile is engineering evidence, not permission to launch further experiments.
Imports before ResourceGovernor admission use only the standard library and the
Torch-free Olympus control plane. Importing this file does not start any work.
"""

from __future__ import annotations

import argparse
import csv
import functools
import io
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

WARMUP_TRAINING_SAMPLES = 5
REPRESENTATIVE_STEPS = 300
REPRESENTATIVE_BATCH_SIZE = 32


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [sample["seconds"] for sample in samples if sample["succeeded"]]
    return {
        "successful_samples": len(durations),
        "failed_samples": sum(not sample["succeeded"] for sample in samples),
        "p50_seconds": _percentile(durations, 0.50),
        "p95_seconds": _percentile(durations, 0.95),
        "maximum_seconds": max(durations, default=None),
        "total_successful_seconds": sum(durations),
        "total_observed_seconds": sum(sample["seconds"] for sample in samples),
    }


def _rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _nvidia_inventory() -> dict[str, Any]:
    """Read driver-visible devices independently of Torch's selected-device mask."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        devices = [
            {"index": int(row[0]), "name": row[1].strip(), "total_vram_mib": float(row[2])}
            for row in csv.reader(io.StringIO(result.stdout))
            if row
        ]
        return {
            "status": "OBSERVED",
            "devices": devices,
            "device_count": len(devices),
            "scope": "driver-visible inventory; platform allocation count still needs observation",
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        return {"status": "UNAVAILABLE", "error_type": type(error).__name__}


class _Recorder:
    def __init__(self, torch: Any, device: Any, sample_stream: Any, started: float) -> None:
        self.torch = torch
        self.device = device
        self.sample_stream = sample_stream
        self.started = started
        self.samples: dict[str, list[dict[str, Any]]] = {}
        self.active_phases: list[str] = []
        self.models: dict[int, dict[str, Any]] = {}
        self.validation_seconds = 0.0
        self.logging_errors: list[str] = []

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def timed(self, phase: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        parents = list(self.active_phases)
        self.active_phases.append(phase)
        started = time.perf_counter()
        failure: str | None = None
        synchronization_failure: str | None = None
        try:
            self.synchronize()
            started = time.perf_counter()
            return function(*args, **kwargs)
        except BaseException as error:
            failure = type(error).__name__
            raise
        finally:
            try:
                self.synchronize()
            except BaseException as error:
                synchronization_failure = type(error).__name__
                if failure is None:
                    raise
            finally:
                elapsed = time.perf_counter() - started
                sample = {
                    "phase": phase,
                    "sample_index": len(self.samples.get(phase, [])),
                    "started_offset_seconds": started - self.started,
                    "seconds": elapsed,
                    "parent_phases": parents,
                    "model_training": phase == "forward_training"
                    if phase.startswith("forward_")
                    else None,
                    "inclusive": True,
                    "succeeded": failure is None and synchronization_failure is None,
                    "error_type": failure,
                    "synchronization_error_type": synchronization_failure,
                }
                self.samples.setdefault(phase, []).append(sample)
                self.active_phases.pop()
                try:
                    self.sample_stream.write(
                        json.dumps(sample, sort_keys=True, allow_nan=False) + "\n"
                    )
                    self.sample_stream.flush()
                except BaseException as error:
                    self.logging_errors.append(type(error).__name__)
                    if failure is None and synchronization_failure is None:
                        raise

    def validate_forward(self, model: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Observe actual tensors without moving, casting, mutating or sampling them."""
        started = time.perf_counter()
        try:
            parameters = list(model.parameters())
            if not parameters:
                raise ValueError("profiled Olympus model has no parameters")
            for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
                self._validate_tensor(tensor, f"model.{name}")
            for index, value in enumerate(args):
                self._validate_input(value, f"args[{index}]")
            for name, value in kwargs.items():
                self._validate_input(value, name)
            if id(model) not in self.models:
                self.models[id(model)] = {
                    "class": type(model).__name__,
                    "parameters": sum(parameter.numel() for parameter in parameters),
                    "trainable_parameters": sum(
                        parameter.numel() for parameter in parameters if parameter.requires_grad
                    ),
                    "parameter_bytes": sum(
                        parameter.numel() * parameter.element_size() for parameter in parameters
                    ),
                    "device": str(parameters[0].device),
                    "floating_dtype": str(self.torch.float32),
                    "model_config": model.config.model_dump(mode="json"),
                }
        finally:
            self.validation_seconds += time.perf_counter() - started

    def _validate_tensor(self, tensor: Any, name: str) -> None:
        if not isinstance(tensor, self.torch.Tensor):
            raise TypeError(f"{name} is not a tensor")
        if tensor.device != self.device:
            raise ValueError(f"{name} actually resides on {tensor.device}, expected {self.device}")
        if tensor.is_floating_point():
            if tensor.dtype != self.torch.float32:
                raise ValueError(f"{name} actually uses {tensor.dtype}; only FP32 is profiled")
        elif tensor.dtype not in {self.torch.int32, self.torch.int64, self.torch.bool}:
            raise ValueError(f"{name} has unsupported non-floating dtype {tensor.dtype}")

    def _validate_input(self, value: Any, name: str) -> None:
        if value is None:
            return
        if isinstance(value, self.torch.Tensor):
            self._validate_tensor(value, name)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("profiled input mappings require string keys")
                self._validate_input(item, f"{name}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                self._validate_input(item, f"{name}[{index}]")
        else:
            raise TypeError(f"{name} must be a tensor, nested tensor collection, or None")

    def summaries(self) -> dict[str, Any]:
        result = {}
        for phase, samples in self.samples.items():
            skip = (
                WARMUP_TRAINING_SAMPLES
                if phase in {"forward_training", "backward_training", "optimizer_training"}
                else 0
            )
            result[phase] = {
                "all": _summary(samples),
                "warmup_samples_excluded": min(skip, len(samples)),
                "warmed": _summary(samples[skip:]),
            }
        return result


@contextmanager
def _instrument(recorder: _Recorder, trainer: Any, model_class: Any) -> Iterator[None]:
    patches: list[tuple[Any, str, Any]] = []

    def patch(owner: Any, name: str, replacement: Any) -> None:
        original = getattr(owner, name)
        patches.append((owner, name, original))
        setattr(owner, name, replacement)

    def wrap(owner: Any, name: str, phase: str) -> None:
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return recorder.timed(phase, original, *args, **kwargs)

        patch(owner, name, wrapped)

    original_forward = model_class.forward

    @functools.wraps(original_forward)
    def forward(model: Any, *args: Any, **kwargs: Any) -> Any:
        recorder.validate_forward(model, args, kwargs)
        phase = "forward_training" if model.training else "forward_evaluation"
        return recorder.timed(phase, original_forward, model, *args, **kwargs)

    try:
        patch(model_class, "forward", forward)
        wrap(recorder.torch.autograd, "backward", "backward_training")
        wrap(recorder.torch.optim.AdamW, "step", "optimizer_training")
        wrap(trainer, "evaluate_olympus_system", "evaluation")
        wrap(trainer, "save_training_checkpoint", "training_checkpoint")
        wrap(trainer, "save_olympus_checkpoint", "final_export_checkpoint")
        wrap(trainer, "load_olympus_dataset", "dataset_load")
        yield
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)


def profile_run(
    manifest_path: Path,
    run_root: Path,
    cell_id: str,
    output: Path,
    requested_device: Literal["cpu", "cuda"],
) -> dict[str, Any]:
    # Direct script invocation must resolve the selected source checkout, not an
    # unrelated installed Olympus distribution.
    repository = Path(__file__).resolve().parents[2]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from olympus.foundry.resources import ResourceGovernor, memory_snapshot
    from olympus.training.campaign import execute_campaign, validate_manifest
    from olympus.training.provenance import atomic_json, digest_json, file_sha256, source_identity

    started = time.perf_counter()
    manifest_path, run_root, output = manifest_path.resolve(), run_root.resolve(), output.resolve()
    sample_path = output.with_name(output.name + ".samples.jsonl")
    if output.exists() or sample_path.exists():
        raise FileExistsError("profiling output and samples must use new paths")
    raw_manifest = json.loads(manifest_path.read_text())
    manifest = validate_manifest(raw_manifest)
    if manifest["purpose"] != "development":
        raise ValueError("full representative profiling requires a development manifest")
    cell = next((item for item in manifest["cells"] if item["id"] == cell_id), None)
    if cell is None:
        raise ValueError("unknown profiling cell")
    state_path = run_root / "campaign_state.json"
    prior_state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    if (run_root / "cells" / cell_id).exists() or cell_id in prior_state.get("cells", {}):
        raise FileExistsError("profiling requires a fresh, uncached selected cell")
    config = cell["training_config"]
    if config["device"] != requested_device or config["precision"] != "fp32":
        raise ValueError("manifest must explicitly match the requested device and FP32")
    if config["steps"] != REPRESENTATIVE_STEPS or config["batch_size"] != REPRESENTATIVE_BATCH_SIZE:
        raise ValueError("representative profiling requires all 300 steps with batch size 32")
    identity = source_identity()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNNING",
        "started_at": datetime.now(UTC).isoformat(),
        "source_identity": identity,
        "profiler_sha256": file_sha256(Path(__file__)),
        "manifest_sha256": file_sha256(manifest_path),
        "cell": cell,
        "cell_config_sha256_before_default_expansion": digest_json(config),
        "run_root": str(run_root),
        "requested_device": requested_device,
        "samples_path": str(sample_path),
        "warmup_training_samples": WARMUP_TRAINING_SAMPLES,
        "timing_contract": {
            "units": "seconds",
            "synchronization": "CUDA synchronized before and after each instrumented call",
            "nesting": "inclusive; evaluation contains forward_evaluation; never sum nested phases",
            "complete_elapsed": (
                "admission, imports, data and full cell; excludes final receipt writing"
            ),
            "overhead": (
                "synchronization, input validation and sample logging affect full-cell time"
            ),
            "extrapolation": "none; full declared 300-step, batch-32 cell must complete",
        },
        "account_guard": "required externally; this child does not stop platform billing",
        "promotion_authorized": False,
        "claim_boundary": "Candidate engineering profile; no scientific or GPU-readiness claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    recorder: _Recorder | None = None
    torch: Any = None
    selected_device: Any = None
    try:
        with sample_path.open("x", encoding="utf-8") as sample_stream:
            # Distinct from execute_campaign's lock: admission is required before
            # importing Torch, while the maintained controller keeps its own guard.
            with ResourceGovernor(run_root / "profile-admission.lock") as governor:
                receipt["resource_preflight"] = (
                    governor.preflight.as_dict() if governor.preflight is not None else None
                )
                import torch as admitted_torch

                import olympus.training.olympus_system as trainer
                from olympus.models.olympus_system import OlympusReferenceSystem
                from olympus.models.runtime import resolve_olympus_device

                torch = admitted_torch
                if (
                    Path(trainer.__file__).resolve()
                    != repository / "olympus/training/olympus_system.py"
                ):
                    raise RuntimeError("profiler imported a trainer outside its source checkout")
                trainer._configure_deterministic_cuda(
                    bool(config.get("deterministic", True)) and requested_device == "cuda"
                )
                selected_device = resolve_olympus_device(requested_device)
                if selected_device.type == "cuda":
                    selected_device = torch.device("cuda", torch.cuda.current_device())
                    torch.cuda.synchronize(selected_device)
                    torch.cuda.reset_peak_memory_stats(selected_device)
                receipt["hardware"] = {
                    "actual_device": str(selected_device),
                    "torch_version": str(torch.__version__),
                    "cuda_build": torch.version.cuda,
                    "visible_cuda_device_count": torch.cuda.device_count(),
                    "used_cuda_device_count": 1 if selected_device.type == "cuda" else 0,
                    "nvidia_smi_inventory": _nvidia_inventory(),
                    "gpus": [
                        {
                            "index": index,
                            "name": torch.cuda.get_device_properties(index).name,
                            "total_vram_bytes": torch.cuda.get_device_properties(
                                index
                            ).total_memory,
                        }
                        for index in range(torch.cuda.device_count())
                    ]
                    if selected_device.type == "cuda"
                    else [],
                    "host_memory_before": memory_snapshot().as_dict(),
                }
                recorder = _Recorder(torch, selected_device, sample_stream, started)
                cell_started = time.perf_counter()
                with _instrument(recorder, trainer, OlympusReferenceSystem):
                    state = execute_campaign(manifest_path, run_root, cell_id=cell_id)
                receipt["cell_invocation_seconds"] = time.perf_counter() - cell_started
                receipt["campaign_status"] = state.get("status")
                receipt["cell_status"] = state.get("cells", {}).get(cell_id, {})
                if receipt["cell_status"].get("status") != "COMPLETED":
                    receipt["status"] = "INCOMPLETE_PROFILE"
                    return receipt
                for phase in ("forward_training", "backward_training", "optimizer_training"):
                    actual = sum(sample["succeeded"] for sample in recorder.samples.get(phase, []))
                    if actual != REPRESENTATIVE_STEPS:
                        raise RuntimeError(
                            f"profile observed {actual} successful calls for {phase}"
                        )
                report_path = run_root / "cells" / cell_id / "report.json"
                report = json.loads(report_path.read_text())
                if report.get("completed_steps") != REPRESENTATIVE_STEPS:
                    raise RuntimeError("profiled report does not contain the complete step budget")
                if report.get("source_identity", {}).get("sha256") != identity["sha256"]:
                    raise RuntimeError("source identity changed during the profiled invocation")
                receipt["training_config_sha256"] = report["training_config_sha256"]
                receipt["dataset_manifest_sha256"] = report["dataset_manifest_sha256"]
                receipt["completed_steps"] = report["completed_steps"]
                receipt["report_sha256"] = file_sha256(report_path)
                receipt["status"] = "COMPLETED_REPRESENTATIVE_PROFILE"
                return receipt
    except BaseException as error:
        receipt["status"] = (
            "INTERRUPTED_PROFILE"
            if isinstance(error, (KeyboardInterrupt, SystemExit))
            else "FAILED_PROFILE"
        )
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        receipt["finished_at"] = datetime.now(UTC).isoformat()
        receipt["complete_elapsed_seconds"] = time.perf_counter() - started
        receipt["host_process_peak_rss_bytes"] = _rss_bytes()
        receipt["torch_imported"] = "torch" in sys.modules
        if recorder is not None:
            receipt["models"] = list(recorder.models.values())
            receipt["phase_summaries"] = recorder.summaries()
            receipt["samples"] = recorder.samples
            receipt["input_validation_overhead_seconds"] = recorder.validation_seconds
            receipt["sample_logging_errors"] = recorder.logging_errors
        if torch is not None and selected_device is not None and selected_device.type == "cuda":
            try:
                receipt["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
                    selected_device
                )
                receipt["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
                    selected_device
                )
            except Exception as error:
                receipt["cuda_memory_observation_error"] = type(error).__name__
        try:
            receipt["host_memory_after"] = memory_snapshot().as_dict()
        except Exception as error:
            receipt["host_memory_observation_error"] = type(error).__name__
        if sample_path.is_file():
            receipt["samples_sha256"] = file_sha256(sample_path)
        try:
            atomic_json(output, receipt)
        except BaseException as error:
            if "error" not in receipt:
                raise
            print(
                json.dumps(
                    {
                        "status": receipt["status"],
                        "original_error": receipt["error"]["type"],
                        "profile_receipt_write_error": type(error).__name__,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    result = profile_run(args.manifest, args.run_root, args.cell, args.output, args.device)
    print(
        json.dumps(
            {
                "status": result["status"],
                "profile": str(args.output.resolve()),
                "complete_elapsed_seconds": result["complete_elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if result["status"] != "COMPLETED_REPRESENTATIVE_PROFILE":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
