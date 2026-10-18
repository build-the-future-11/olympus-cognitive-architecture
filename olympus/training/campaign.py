"""Finite, restart-safe development campaigns; importing this module never loads Torch.

This control plane cannot authorize a confirmatory study or promote a checkpoint.
All numerical work is imported only after the resource admission lock is held.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from olympus.foundry.resources import ResourceGovernor, memory_snapshot
from olympus.training.provenance import atomic_json, digest_json, file_sha256, source_identity

DEFAULT_WALL_SECONDS = 4 * 60 * 60
MAX_WALL_SECONDS = 6 * 60 * 60
CLAIM_BOUNDARY = (
    "Synthetic or user-supplied development engineering evidence only; no confirmatory "
    "hypothesis, natural-data capability, calibration, safety, or promotion is established."
)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
_TRAINING_REQUIRED = {
    "seed",
    "steps",
    "batch_size",
    "learning_rate",
    "warmup_steps",
    "device",
    "precision",
    "cpu_threads",
    "evaluation_interval",
    "checkpoint_interval_seconds",
    "model",
}


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular non-symlink JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _contained(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or relative in {"", "."}:
        raise ValueError("artifact paths must be contained relative paths")
    candidate = root / path
    descendants = [candidate, *(part for part in candidate.parents if part.is_relative_to(root))]
    if any(part.is_symlink() for part in descendants):
        raise ValueError("artifact paths must not traverse symlinks")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes its root")
    return candidate


def _positive(value: object, name: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0 or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its finite budget")
    return result


def validate_manifest(value: dict[str, Any]) -> dict[str, Any]:
    """Validate finite controls without importing a tensor or model library."""
    allowed = {
        "schema_version",
        "campaign_id",
        "purpose",
        "max_wall_seconds",
        "stop_reserve_seconds",
        "cells",
    }
    if (
        set(value) - allowed
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
    ):
        raise ValueError("unsupported campaign schema or fields")
    if not isinstance(value.get("campaign_id"), str) or not _ID.fullmatch(value["campaign_id"]):
        raise ValueError("campaign_id must be a safe identifier")
    if value.get("purpose") not in {"smoke", "development"}:
        raise ValueError(
            "only smoke/development campaigns are authorized; "
            "confirmatory work needs a separate protocol"
        )
    maximum = _positive(
        value.get("max_wall_seconds", DEFAULT_WALL_SECONDS),
        "max_wall_seconds",
        maximum=MAX_WALL_SECONDS,
    )
    reserve = _positive(value.get("stop_reserve_seconds", 60), "stop_reserve_seconds")
    if reserve >= maximum:
        raise ValueError("stop reserve must be below the wall budget")
    cells = value.get("cells")
    if not isinstance(cells, list) or not 1 <= len(cells) <= 64:
        raise ValueError("a campaign must contain 1 through 64 explicit cells")
    identities: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) - {
            "id",
            "dataset",
            "training_config",
            "estimated_seconds",
        }:
            raise ValueError("invalid campaign cell fields")
        identifier = cell.get("id")
        if (
            not isinstance(identifier, str)
            or not _ID.fullmatch(identifier)
            or identifier in identities
        ):
            raise ValueError("cell IDs must be unique safe identifiers")
        identities.add(identifier)
        if not isinstance(cell.get("dataset"), str) or not isinstance(
            cell.get("training_config"), dict
        ):
            raise ValueError("cells require a dataset path and inline training_config object")
        if any(
            marker in Path(cell["dataset"]).name.lower()
            for marker in ("test", "confirm", "protected")
        ):
            raise ValueError("protected evaluation manifests are not development inputs")
        config = cell["training_config"]
        if _TRAINING_REQUIRED - set(config):
            raise ValueError(
                f"explicit training fields missing: {sorted(_TRAINING_REQUIRED - set(config))}"
            )
        for key in (
            "seed",
            "steps",
            "batch_size",
            "cpu_threads",
            "evaluation_interval",
            "warmup_steps",
        ):
            if isinstance(config[key], bool) or not isinstance(config[key], int):
                raise ValueError(f"{key} must be an explicit integer")
        if not 0 <= config["seed"] <= 2**32 - 1:
            raise ValueError("seed outside supported range")
        for key in (
            "steps",
            "batch_size",
            "cpu_threads",
            "evaluation_interval",
            "learning_rate",
            "checkpoint_interval_seconds",
        ):
            _positive(config[key], key)
        if not 0 <= config["warmup_steps"] <= config["steps"]:
            raise ValueError("warmup_steps must be within the run")
        if config["device"] not in {"cpu", "cuda"} or config["precision"] != "fp32":
            raise ValueError("an explicit cpu/cuda device and fp32 precision are required")
        if not isinstance(config["model"], dict) or not config["model"]:
            raise ValueError("an explicit model configuration is required")
        if config.get("resource_preflight", True) is not True:
            raise ValueError("campaign resource admission cannot be disabled")
        if config.get("min_available_bytes", 2 * 1024**3) < 2 * 1024**3:
            raise ValueError("campaign memory safety floor cannot be weakened")
        if config.get("max_swap_fraction", 0.90) > 0.90:
            raise ValueError("campaign swap ceiling cannot be weakened")
        if "estimated_seconds" in cell:
            _positive(cell["estimated_seconds"], "estimated_seconds")
    return {**value, "max_wall_seconds": maximum, "stop_reserve_seconds": reserve}


def _dataset_identity(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    artifacts: dict[str, str] = {}
    ids: list[set[str]] = []
    for split in ("train", "evaluation"):
        descriptor = manifest.get(split)
        if not isinstance(descriptor, dict):
            raise ValueError("aligned dataset requires training and development descriptors")
        artifact = _contained(path.parent, descriptor["path"])
        observed = file_sha256(artifact)
        if observed != descriptor.get("sha256"):
            raise ValueError(f"dataset digest mismatch: {split}")
        artifacts[descriptor["path"]] = observed
        source_ids = descriptor.get("source_ids")
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or len(source_ids) != len(set(source_ids))
        ):
            raise ValueError("dataset source IDs must be unique and nonempty")
        ids.append(set(source_ids))
    if ids[0] & ids[1]:
        raise ValueError("training and development source IDs overlap")
    canonical = {**manifest, "manifest_sha256": None}
    expected = hashlib.sha256(
        (
            json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
    ).hexdigest()
    if manifest.get("manifest_sha256") != expected:
        raise ValueError("dataset manifest canonical digest mismatch")
    return {"file_sha256": file_sha256(path), "manifest_sha256": expected, "artifacts": artifacts}


def bind_contract(manifest_path: Path, run_root: Path) -> dict[str, Any]:
    value = validate_manifest(_read_json(manifest_path))
    cells = []
    for cell in value["cells"]:
        dataset = _contained(manifest_path.parent, cell["dataset"])
        config = {**cell["training_config"], "resource_preflight": True}
        # Lock paths are operational and must not prevent safe checkpoint relocation.
        config.pop("workload_lock", None)
        cells.append(
            {**cell, "training_config": config, "dataset_identity": _dataset_identity(dataset)}
        )
    source = source_identity()
    return {
        "schema_version": 1,
        "manifest": {**value, "cells": cells},
        "source_sha256": source["sha256"],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def remaining_seconds(state: dict[str, Any], maximum: float, *, now: float | None = None) -> float:
    elapsed = float(state.get("elapsed_seconds", 0))
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid persisted budget accounting")
    started = state.get("active_started_at")
    if started is not None:
        current = time.time() if now is None else now
        if current < float(started):
            raise ValueError("clock moved backwards during an active campaign")
        elapsed += current - float(started)
    return max(0.0, maximum - elapsed)


def _includes(observed: object, expected: object) -> bool:
    if isinstance(expected, dict):
        return isinstance(observed, dict) and all(
            key in observed and _includes(observed[key], value) for key, value in expected.items()
        )
    return observed == expected


def _validate_report(report: dict[str, Any], cell: dict[str, Any], source_sha256: str) -> None:
    if (
        report.get("status") != "COMPLETED_DEVELOPMENT_NOT_PROMOTED"
        or report.get("promotion_authorized") is not False
    ):
        raise ValueError("training did not emit a bounded completed development report")
    if report.get("dataset_manifest_sha256") != cell["dataset_identity"]["manifest_sha256"]:
        raise ValueError("training report dataset identity mismatch")
    if report.get("source_identity", {}).get("sha256") != source_sha256:
        raise ValueError("training report source identity mismatch")
    config = report.get("training_config")
    if not isinstance(config, dict) or not _includes(config, cell["training_config"]):
        raise ValueError("training report configuration mismatch")
    scientific_config = {key: value for key, value in config.items() if key != "workload_lock"}
    if report.get("training_config_sha256") != digest_json(scientific_config):
        raise ValueError("training report configuration digest mismatch")
    if report.get("completed_steps") != cell["training_config"]["steps"]:
        raise ValueError("training report completed step count mismatch")


def _completion(run: Path, cell: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    completion = _read_json(run / "completion.json")
    if (
        completion.get("cell_sha256") != digest_json(cell)
        or completion.get("source_sha256") != source_sha256
    ):
        raise ValueError("completed cell identity differs from frozen contract")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("completed cell has no artifact identities")
    for name, expected in artifacts.items():
        if file_sha256(_contained(run, name)) != expected:
            raise ValueError(f"completed artifact changed: {name}")
    report = _read_json(run / "report.json")
    _validate_report(report, cell, source_sha256)
    return report


def _collect_completion(
    run: Path, cell: dict[str, Any], source_sha256: str, elapsed: float
) -> dict[str, Any]:
    report = _read_json(run / "report.json")
    _validate_report(report, cell, source_sha256)
    if file_sha256(run / "checkpoint.pt") != report.get("checkpoint_sha256"):
        raise ValueError("final checkpoint digest mismatch")
    artifacts = {name: file_sha256(run / name) for name in ("report.json", "checkpoint.pt")}
    for name in ("run.json", "status.json", "metrics.jsonl"):
        if (run / name).is_file():
            artifacts[name] = file_sha256(run / name)
    for kind in ("latest", "best"):
        pointer = run / "checkpoints" / f"{kind}.json"
        descriptor = _read_json(pointer)
        artifacts[str(pointer.relative_to(run))] = file_sha256(pointer)
        checkpoint_name = descriptor.get("path")
        if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
            raise ValueError("checkpoint pointer must name one contained file")
        candidate = Path("checkpoints") / checkpoint_name
        checkpoint = _contained(run, str(candidate))
        observed = file_sha256(checkpoint)
        if observed != descriptor.get("sha256"):
            raise ValueError("checkpoint pointer digest mismatch")
        expected_binding = {
            "source_sha256": source_sha256,
            "training_config_sha256": report["training_config_sha256"],
            "dataset_manifest_sha256": report["dataset_manifest_sha256"],
        }
        if not _includes(descriptor.get("binding"), expected_binding):
            raise ValueError("checkpoint pointer binding mismatch")
        step = descriptor.get("completed_steps")
        if not isinstance(step, int) or not 0 <= step <= report["completed_steps"]:
            raise ValueError("checkpoint pointer step count mismatch")
        if kind == "latest" and step != report["completed_steps"]:
            raise ValueError("latest checkpoint is incomplete")
        artifacts[str(candidate)] = observed
    result = {
        "schema_version": 1,
        "cell_sha256": digest_json(cell),
        "source_sha256": source_sha256,
        "elapsed_seconds": elapsed,
        "artifacts": artifacts,
        "promotion_authorized": False,
    }
    if (run / "completion.json").exists():
        raise FileExistsError("refusing to replace a completed cell")
    atomic_json(run / "completion.json", result)
    return result


def _admission(run_root: Path) -> ResourceGovernor:
    return ResourceGovernor(
        run_root / "campaign-admission.lock",
        min_available_bytes=2 * 1024**3,
        max_swap_fraction=0.90,
    )


def preflight(run_root: Path, device: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_imported": False,
        "promotion_authorized": False,
    }
    versions: dict[str, str | None] = {}
    problems = []
    if sys.version_info < (3, 11):  # noqa: UP036 -- Kaggle uses a separate source-run environment.
        problems.append("the source-run campaign needs Python >=3.11")
    for package, minimum in {
        "torch": (2, 4),
        "numpy": (1, 26),
        "pydantic": (2, 10),
        "Pillow": (10, 0),
    }.items():
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
            problems.append(f"missing required package: {package}")
            continue
        versions[package] = version
        match = re.match(r"^(\d+)\.(\d+)", version)
        if match is None or tuple(map(int, match.groups())) < minimum:
            problems.append(f"{package}>={minimum[0]}.{minimum[1]} is required")
    result["package_versions"] = versions
    if problems:
        result.update(status="BLOCKED_ENVIRONMENT", reasons=problems)
        return result
    try:
        result["memory"] = memory_snapshot().as_dict()
        with _admission(run_root):
            result["status"] = "ADMITTED"
            if device is not None:
                import torch

                from olympus.models.runtime import resolve_olympus_device

                result["torch_imported"] = True
                selected = resolve_olympus_device(device)  # type: ignore[arg-type]
                result["device"] = str(selected)
                result["torch_version"] = str(torch.__version__)
                result["cuda_build"] = torch.version.cuda
                result["accelerator"] = (
                    torch.cuda.get_device_name(selected) if selected.type == "cuda" else None
                )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        result["torch_imported"] = "torch" in sys.modules
        result.update(status="BLOCKED_RESOURCE_OR_DEVICE", reason=str(error))
    return result


def dry_run(manifest_path: Path, run_root: Path, *, cell_id: str | None = None) -> dict[str, Any]:
    """Enumerate the finite plan without creating files, decoding data or importing Torch."""
    manifest = validate_manifest(_read_json(manifest_path))
    if cell_id is not None and cell_id not in {cell["id"] for cell in manifest["cells"]}:
        raise ValueError("unknown cell ID")
    planned = []
    for cell in manifest["cells"]:
        if cell_id is not None and cell["id"] != cell_id:
            continue
        dataset = _contained(manifest_path.parent, cell["dataset"])
        planned.append(
            {
                "id": cell["id"],
                "dataset": str(dataset),
                "data_status": "PRESENT_UNVERIFIED" if dataset.is_file() else "NOT_PREPARED",
                "training_config": cell["training_config"],
                "config_sha256_before_default_expansion": digest_json(cell["training_config"]),
                "estimated_seconds": cell.get("estimated_seconds"),
            }
        )
    return {
        "status": "DRY_RUN_NO_EXECUTION",
        "campaign_id": manifest["campaign_id"],
        "purpose": manifest["purpose"],
        "run_root": str(run_root),
        "max_wall_seconds": manifest["max_wall_seconds"],
        "stop_reserve_seconds": manifest["stop_reserve_seconds"],
        "source_sha256": source_identity()["sha256"],
        "cells": planned,
        "promotion_authorized": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def prepare_smoke(output: Path, run_root: Path) -> dict[str, Any]:
    return prepare_development_data(output, run_root, smoke=True)


def prepare_development_data(
    output: Path, run_root: Path, *, smoke: bool = False
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("development data destination must be new")
    with _admission(run_root):
        import torch

        from olympus.data.controlled_fusion import ControlledSplitSpec, generate_controlled_split
        from olympus.data.olympus_dataset import write_olympus_dataset_manifest

        output.mkdir(parents=True)
        identifiers: dict[str, list[str]] = {}
        profile = "smoke" if smoke else "development"
        splits = (
            (("train", 8, 2026101601), ("development", 4, 2026101602))
            if smoke
            else (("train", 512, 2026101611), ("development", 128, 2026101612))
        )
        specs = []
        for name, examples, seed in splits:
            spec = ControlledSplitSpec(
                name=f"kaggle_{profile}_{name}", examples=examples, seed=seed, regime="mixed"
            )
            specs.append(spec.model_dump(mode="json"))
            batch, ids = generate_controlled_split(spec)
            torch.save(batch, output / f"{name}.pt")
            identifiers[name] = ids
        manifest = write_olympus_dataset_manifest(
            output / "manifest_development.json",
            dataset_id=f"olympus-kaggle-engineering-{profile}",
            version="1.0.0",
            license_name="CC-BY-4.0",
            train_batch=output / "train.pt",
            evaluation_batch=output / "development.pt",
            train_source_ids=identifiers["train"],
            evaluation_source_ids=identifiers["development"],
        )
        receipt = {
            "schema_version": 1,
            "status": "SMOKE_DATA_ONLY" if smoke else "DEVELOPMENT_DATA_ONLY",
            "manifest_sha256": manifest.manifest_sha256,
            "train_examples": splits[0][1],
            "development_examples": splits[1][1],
            "generator": "olympus.data.controlled_fusion.generate_controlled_split",
            "generator_source_sha256": file_sha256(
                Path(__file__).resolve().parents[1] / "data" / "controlled_fusion.py"
            ),
            "source_identity": source_identity(),
            "split_specifications": specs,
            "normalization": "generator-native tensors; image clamp [0,1]; no fitted normalization",
            "protected_splits": [],
            "claim_boundary": CLAIM_BOUNDARY,
        }
        atomic_json(output / "receipt.json", receipt)
        return receipt


def execute_campaign(
    manifest_path: Path,
    run_root: Path,
    *,
    cell_id: str | None = None,
    stop_after_steps: int | None = None,
) -> dict[str, Any]:
    """Execute each registered cell at most once; account interrupted wall time conservatively."""
    with _admission(run_root):
        contract = bind_contract(manifest_path, run_root)
        contract_path = run_root / "campaign_contract.json"
        if contract_path.exists():
            if _read_json(contract_path) != contract:
                raise ValueError(
                    "campaign source, data, config, or budget differs from frozen contract"
                )
        else:
            atomic_json(contract_path, contract)
        manifest = contract["manifest"]
        if cell_id is not None and cell_id not in {cell["id"] for cell in manifest["cells"]}:
            raise ValueError("unknown cell ID")
        state_path = run_root / "campaign_state.json"
        if not state_path.exists() and (run_root / "cells").exists():
            raise ValueError("campaign budget ledger is missing for existing run artifacts")
        state: dict[str, Any] = (
            _read_json(state_path)
            if state_path.exists()
            else {
                "schema_version": 1,
                "elapsed_seconds": 0.0,
                "cells": {},
                "contract_sha256": digest_json(contract),
            }
        )
        if state.get("contract_sha256") != digest_json(contract):
            raise ValueError("campaign budget ledger belongs to another contract")
        remaining_seconds(state, manifest["max_wall_seconds"])
        if state.get("active_started_at") is not None:
            crashed_elapsed = time.time() - float(state["active_started_at"])
            state["elapsed_seconds"] += crashed_elapsed
            active_id = state.get("active_cell_id")
            if active_id not in state["cells"]:
                raise ValueError("active budget has no corresponding cell")
            state["cells"][active_id]["elapsed_seconds"] = (
                float(state["cells"][active_id].get("elapsed_seconds", 0.0)) + crashed_elapsed
            )
        state["active_started_at"] = None
        state["active_cell_id"] = None
        atomic_json(state_path, state)
        for cell in manifest["cells"]:
            if cell_id is not None and cell["id"] != cell_id:
                continue
            run = run_root / "cells" / cell["id"]
            if (run / "completion.json").exists():
                _completion(run, cell, contract["source_sha256"])
                continue
            # A crash after the immutable report write needs no second training run.
            if (run / "report.json").exists():
                duration = float(state["cells"].get(cell["id"], {}).get("elapsed_seconds", 0.0))
                _collect_completion(run, cell, contract["source_sha256"], duration)
                state["cells"][cell["id"]] = {
                    "status": "COMPLETED",
                    "steps": cell["training_config"]["steps"],
                    "elapsed_seconds": max(duration, 0.001),
                }
                atomic_json(state_path, state)
                continue
            available = (
                remaining_seconds(state, manifest["max_wall_seconds"])
                - manifest["stop_reserve_seconds"]
            )
            completed = [
                entry for entry in state["cells"].values() if entry.get("status") == "COMPLETED"
            ]
            estimate = cell.get("estimated_seconds")
            if estimate is None and completed:
                rate = max(entry["elapsed_seconds"] / entry["steps"] for entry in completed)
                estimate = 1.25 * rate * cell["training_config"]["steps"]
            prior = state["cells"].get(cell["id"], {})
            if available <= 0 or (estimate is not None and estimate > available):
                state["status"] = "BUDGET_STOPPED"
                atomic_json(state_path, state)
                break
            if estimate is None and state["cells"] and not prior:
                state["status"] = "PILOT_REQUIRED_BEFORE_UNESTIMATED_CELL"
                atomic_json(state_path, state)
                break
            # Torch and Pydantic training schemas are imported only under admission.
            from olympus.training.olympus_system import (
                OlympusSystemTrainingConfig,
                TrainingInterrupted,
                train_olympus_system,
            )

            config = OlympusSystemTrainingConfig.model_validate(
                {
                    **cell["training_config"],
                    "workload_lock": str((run_root / "training.lock").resolve()),
                }
            )
            started = time.time()
            state["active_started_at"] = started
            state["active_cell_id"] = cell["id"]
            state["status"] = "RUNNING"
            state["cells"][cell["id"]] = {
                "status": "RUNNING",
                "steps": config.steps,
                "elapsed_seconds": float(prior.get("elapsed_seconds", 0.0)),
            }
            atomic_json(state_path, state)
            try:
                train_olympus_system(
                    _contained(manifest_path.parent, cell["dataset"]),
                    run,
                    config=config,
                    resume=bool(prior),
                    max_seconds=available,
                    stop_after_steps=stop_after_steps,
                )
                duration = time.time() - started + float(prior.get("elapsed_seconds", 0.0))
                _collect_completion(run, cell, contract["source_sha256"], duration)
                state["cells"][cell["id"]] = {
                    "status": "COMPLETED",
                    "steps": config.steps,
                    "elapsed_seconds": duration,
                }
            except TrainingInterrupted as error:
                state["cells"][cell["id"]] = {
                    "status": "INTERRUPTED",
                    "steps": config.steps,
                    "reason": str(error),
                    "elapsed_seconds": time.time()
                    - started
                    + float(prior.get("elapsed_seconds", 0.0)),
                }
                state["status"] = "INTERRUPTED"
                break
            except BaseException as error:
                state["cells"][cell["id"]] = {
                    "status": "FAILED",
                    "steps": config.steps,
                    "reason": str(error),
                    "elapsed_seconds": time.time()
                    - started
                    + float(prior.get("elapsed_seconds", 0.0)),
                }
                state["status"] = "FAILED"
                raise
            finally:
                state["elapsed_seconds"] += max(0.0, time.time() - started)
                state["active_started_at"] = None
                state["active_cell_id"] = None
                atomic_json(state_path, state)
        if all(
            (run_root / "cells" / cell["id"] / "completion.json").is_file()
            for cell in manifest["cells"]
        ):
            state["status"] = "COMPLETED_DEVELOPMENT_NOT_PROMOTED"
            atomic_json(state_path, state)
        return state


def analyze(run_root: Path) -> dict[str, Any]:
    contract = _read_json(run_root / "campaign_contract.json")
    rows = []
    for cell in contract["manifest"]["cells"]:
        run = run_root / "cells" / cell["id"]
        if not (run / "completion.json").exists():
            rows.append({"id": cell["id"], "status": "INCOMPLETE"})
            continue
        report = _completion(run, cell, contract["source_sha256"])
        rows.append(
            {
                "id": cell["id"],
                "status": report["status"],
                "completed_steps": report["completed_steps"],
                "final_evaluation": report["final_evaluation"],
                "checkpoint_sha256": report["checkpoint_sha256"],
            }
        )
    return {
        "schema_version": 1,
        "cells": rows,
        "promotion_authorized": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def export_completed(run_root: Path, destination: Path) -> dict[str, Any]:
    """Export only explicitly recorded completed artifacts, never arbitrary directories."""
    if destination.exists():
        raise FileExistsError("export destination must be new")
    contract = _read_json(run_root / "campaign_contract.json")
    allowlist = {"campaign_contract.json"}
    for cell in contract["manifest"]["cells"]:
        run = run_root / "cells" / cell["id"]
        if not (run / "completion.json").exists():
            continue
        _completion(run, cell, contract["source_sha256"])
        metadata = _read_json(run / "completion.json")
        prefix = f"cells/{cell['id']}/"
        allowlist.add(prefix + "completion.json")
        for name in metadata["artifacts"]:
            # A tampered completion file cannot smuggle arbitrary files into an export.
            if name not in {
                "report.json",
                "checkpoint.pt",
                "run.json",
                "status.json",
                "metrics.jsonl",
                "checkpoints/latest.json",
                "checkpoints/best.json",
            } and not re.fullmatch(r"checkpoints/step-[A-Za-z0-9_-]+\.pt", name):
                raise ValueError("completion contains an unsupported export artifact")
            allowlist.add(prefix + name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(destination, "x:gz") as archive:
            for name in sorted(allowlist):
                path = _contained(run_root, name)
                if not path.is_file():
                    raise ValueError("export artifact must be a regular file")
                archive.add(path, arcname=name, recursive=False)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {
        "status": "EXPORTED_COMPLETED_ARTIFACTS",
        "path": str(destination),
        "sha256": file_sha256(destination),
        "files": sorted(allowlist),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "preflight",
        "prepare-smoke",
        "prepare-development",
        "run",
        "campaign",
        "evaluate",
        "analyze",
        "export",
    ):
        sub = subparsers.add_parser(name)
        sub.add_argument("--run-root", type=Path, required=True)
        if name == "preflight":
            sub.add_argument("--device", choices=("cpu", "cuda"))
        if name in {"prepare-smoke", "prepare-development"}:
            sub.add_argument("--output", type=Path, required=True)
        if name in {"run", "campaign", "evaluate"}:
            sub.add_argument("--manifest", type=Path, required=True)
        if name in {"run", "evaluate"}:
            sub.add_argument("--cell", required=True)
        if name in {"run", "campaign"}:
            sub.add_argument("--stop-after-steps", type=int)
            sub.add_argument("--dry-run", action="store_true")
        if name == "evaluate":
            sub.add_argument("--checkpoint", choices=("best", "latest"), default="best")
        if name == "export":
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if args.command == "preflight":
        result = preflight(run_root, args.device)
    elif args.command == "prepare-smoke":
        result = prepare_smoke(args.output.resolve(), run_root)
    elif args.command == "prepare-development":
        result = prepare_development_data(args.output.resolve(), run_root)
    elif args.command in {"run", "campaign"}:
        if args.dry_run:
            result = dry_run(args.manifest.resolve(), run_root, cell_id=getattr(args, "cell", None))
        else:
            result = execute_campaign(
                args.manifest.resolve(),
                run_root,
                cell_id=getattr(args, "cell", None),
                stop_after_steps=args.stop_after_steps,
            )
    elif args.command == "analyze":
        result = analyze(run_root)
    elif args.command == "export":
        result = export_completed(run_root, args.output.resolve())
    else:
        with _admission(run_root):
            contract = bind_contract(args.manifest.resolve(), run_root)
            if _read_json(run_root / "campaign_contract.json") != contract:
                raise ValueError(
                    "evaluation source/data/config identity differs from campaign contract"
                )
            selected = next(
                (cell for cell in contract["manifest"]["cells"] if cell["id"] == args.cell), None
            )
            if selected is None:
                raise ValueError("unknown cell ID")
            from olympus.training.olympus_system import evaluate_frozen_run

            evaluation = evaluate_frozen_run(
                run_root / "cells" / args.cell,
                _contained(args.manifest.resolve().parent, selected["dataset"]),
                checkpoint=args.checkpoint,
            )
            result = {
                "status": "FROZEN_DEVELOPMENT_EVALUATION",
                "evaluation": evaluation.model_dump(mode="json"),
                "promotion_authorized": False,
                "claim_boundary": CLAIM_BOUNDARY,
            }
    print(json.dumps(result, indent=2, sort_keys=True))
    status = str(result.get("status", ""))
    if status.startswith("BLOCKED") or status == "FAILED":
        raise SystemExit(2)
    if status in {"INTERRUPTED", "BUDGET_STOPPED", "PILOT_REQUIRED_BEFORE_UNESTIMATED_CELL"}:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
