"""Execute one fresh, bounded block of the prospective Olympus prediction study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from olympus.core.forecasting import (  # noqa: E402
    ForecastInference,
    ObservablePredictor,
    representation_diagnostics,
)
from olympus.foundry.resources import ResourceGovernor  # noqa: E402
from scripts.verify_foundry_integrity import host_snapshot, source_identity  # noqa: E402

STUDY = ROOT / "research/studies/observable_prediction_v1"
NEURAL_ARMS = (
    "full", "direct", "variance", "latent", "frozen_target", "shuffled_target",
    "direct_cpu_matched",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    # Serialize before touching the destination, then publish one durable rename.
    contents = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_arrays(path: Path, values: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("raw array artifacts are immutable")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_finite_parameters(model: torch.nn.Module) -> None:
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("nonfinite final model parameters")


def array_hash(array: np.ndarray) -> str:
    identity = json.dumps({"dtype": str(array.dtype), "shape": array.shape}).encode()
    return hashlib.sha256(identity + array.tobytes(order="C")).hexdigest()


def vector_field(states: np.ndarray, family: str) -> np.ndarray:
    q, v = states.T
    if family == "pendulum":
        acceleration = -np.sin(q) - 0.15 * v
    elif family == "duffing":
        acceleration = -q - 0.5 * q**3 - 0.2 * v
    else:
        raise ValueError("unknown dynamical family")
    return np.column_stack((v, acceleration))


def trajectories(
    initial: np.ndarray, family: str, steps: int, *, substeps: int = 4
) -> np.ndarray:
    if initial.ndim != 2 or initial.shape[1] != 2 or steps < 1 or substeps < 1:
        raise ValueError("invalid trajectory dimensions or integration schedule")
    if not np.isfinite(initial).all():
        raise ValueError("nonfinite initial state")
    current = initial.astype(np.float64).copy()
    values = [current.copy()]
    dt = 0.1 / substeps
    for _ in range(steps):
        for _ in range(substeps):
            k1 = vector_field(current, family)
            k2 = vector_field(current + dt * k1 / 2, family)
            k3 = vector_field(current + dt * k2 / 2, family)
            k4 = vector_field(current + dt * k3, family)
            current += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        values.append(current.copy())
    return np.stack(values, axis=1)


def generate_split(
    seed: int, family_index: int, split_index: int, count: int, steps: int
) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, family_index, split_index]))
    initial = rng.uniform(-1.5, 1.5, (count, 2))
    if split_index == 3:
        initial[:, 0] = rng.choice([-1, 1], count) * rng.uniform(1.75, 2.5, count)
    return trajectories(initial, ("pendulum", "duffing")[family_index], steps)


def polynomial_features(states: np.ndarray, degree: int) -> np.ndarray:
    return np.column_stack([
        states[:, 0] ** i * states[:, 1] ** (power - i)
        for power in range(degree + 1) for i in range(power + 1)
    ])


def fit_ridge(inputs: np.ndarray, targets: np.ndarray, degree: int) -> np.ndarray:
    features = polynomial_features(inputs, degree)
    penalty = np.eye(features.shape[1]) * 1e-6
    penalty[0, 0] = 0
    return np.linalg.solve(
        features.T @ features / len(inputs) + penalty,
        features.T @ targets / len(inputs),
    )


def forecast(predict: Any, initial: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    current = initial.copy()
    predictions, losses = [], []
    divergent = np.zeros(len(initial), dtype=bool)
    started = time.perf_counter()
    for step in range(targets.shape[1]):
        current = predict(current)
        divergent |= ~np.isfinite(current).all(axis=1) | (np.abs(current) > 1e6).any(axis=1)
        predictions.append(current.copy())
        current[divergent] = 0
        squared_error = np.mean((current - targets[:, step]) ** 2, axis=1)
        squared_error[divergent] = 1e12
        losses.append(squared_error)
    errors = np.stack(losses, axis=1)
    return {
        "predictions": np.stack(predictions, axis=1),
        "errors": errors,
        "divergent_trajectories": int(divergent.sum()),
        "horizons": {str(h): float(errors[:, :h].mean()) for h in (1, 10, 20, 50)},
        "rollout_wall_seconds": time.perf_counter() - started,
    }


def neural_prediction(model: ForecastInference) -> Callable[[np.ndarray], np.ndarray]:
    def predict(values: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return cast(np.ndarray, model(torch.tensor(values, dtype=torch.float32)).numpy())
    return predict


def ridge_prediction(
    degree: int, coefficients: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    def predict(values: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, polynomial_features(values, degree) @ coefficients)
    return predict


def tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def preflight(output: Path) -> dict[str, Any]:
    snapshot = host_snapshot(output)
    if snapshot["effective_available_estimate_bytes"] < 1024**3:
        write_json(output / "blocked-preflight.json", snapshot)
        raise RuntimeError("estimated available memory is below the resource-v2 1GiB floor")
    if snapshot["disk_free_bytes"] < 1024**3:
        write_json(output / "blocked-preflight.json", snapshot)
        raise RuntimeError("less than1GiB disk space remains")
    return snapshot


def fit_neural(
    arm: str, initial: dict[str, torch.Tensor], inputs: torch.Tensor, targets: torch.Tensor,
    seed: int, configuration: dict[str, Any], steps: int, directory: Path,
    deadline: float, cpu_budget: float | None = None,
) -> tuple[ForecastInference, dict[str, Any]]:
    directory.mkdir()
    before = preflight(directory)
    model = ObservablePredictor()
    model.load_state_dict(initial)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=0.003, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8,
        amsgrad=False, foreach=False, fused=False,
    )
    batches = torch.Generator().manual_seed(seed + 11)
    permutations = torch.Generator().manual_seed(seed + 12)
    latent_weight = configuration["latent_weight"] if arm in {
        "latent", "full", "frozen_target", "shuffled_target"
    } else 0.0
    variance_weight = configuration["variance_weight"] if arm in {
        "variance", "full", "frozen_target", "shuffled_target"
    } else 0.0
    momentum = 1.0 if arm == "frozen_target" else 0.99
    logs: list[dict[str, Any]] = []
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    maximum = steps * 3 if cpu_budget is not None else steps
    status, error = "COMPLETED", None
    try:
        for step in range(maximum):
            if time.perf_counter() >= deadline:
                raise TimeoutError("per-block scientific budget exhausted")
            indices = torch.randint(len(inputs), (128,), generator=batches)
            permutation = (
                torch.randperm(128, generator=permutations) if arm == "shuffled_target" else None
            )
            optimizer.zero_grad(set_to_none=True)
            loss, components = model.objective(
                inputs[indices], targets[indices], latent_weight=latent_weight,
                variance_weight=variance_weight, target_permutation=permutation,
            )
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if latent_weight:
                model.update_target(momentum)
            logs.append({"step": step + 1, "loss": float(loss.detach()), **{
                name: float(value.detach()) for name, value in components.items()
            }})
            if cpu_budget is not None and time.process_time() - cpu_start >= cpu_budget:
                break
        ensure_finite_parameters(model)
    except Exception as exc:
        status, error = "FAILED", f"{type(exc).__name__}: {exc}"
    cpu_seconds, wall_seconds = time.process_time() - cpu_start, time.perf_counter() - wall_start
    checkpoint = directory / "checkpoint.pt"
    torch.save({
        "schema_version": 1, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "minibatch_rng": batches.get_state(),
        "permutation_rng": permutations.get_state(), "steps": len(logs),
        "configuration": configuration, "arm": arm, "status": status,
    }, checkpoint)
    with (directory / "losses.jsonl").open("w") as handle:
        for row in logs:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    capped = cpu_budget is not None and len(logs) >= maximum
    summary = {
        "status": status, "error": error, "optimizer_steps": len(logs),
        "training_cpu_seconds": cpu_seconds, "training_wall_seconds": wall_seconds,
        "cpu_budget_seconds": cpu_budget,
        "maximum_steps": maximum, "cap_reached": capped,
        "cpu_match_valid": cpu_budget is None or (
            not capped and cpu_budget <= cpu_seconds <= 1.05 * cpu_budget
        ),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "inference_parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
        "target_parameter_bytes": tensor_bytes(model.target_encoder.state_dict()),
        "optimizer_tensor_bytes": tensor_bytes(optimizer.state_dict()),
        "checkpoint_sha256": sha(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "losses_sha256": sha(directory / "losses.jsonl"),
        "preflight": before,
    }
    write_json(directory / "training.json", summary)
    if error is not None:
        raise RuntimeError(error)
    summary["after"] = host_snapshot(directory)
    summary["representation"] = representation_diagnostics(model, inputs[:1024])
    reloaded = ObservablePredictor()
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True)["model_state"])
    with torch.no_grad():
        if not torch.equal(model.rollout(inputs[:8], 20), reloaded.rollout(inputs[:8], 20)):
            raise RuntimeError("checkpoint reload changed the autoregressive forecast")
    summary["reload_bitwise_equal"] = True
    inference = model.inference_copy()
    exported = directory / "inference.pt"
    torch.save({"schema_version": 1, "state_dim": 2, "width": 32, "latent_dim": 8,
                "model_state": inference.state_dict()}, exported)
    restored = ObservablePredictor().inference_copy()
    restored.load_state_dict(torch.load(exported, weights_only=True)["model_state"])
    with torch.no_grad():
        if not torch.equal(inference(inputs[:8]), model(inputs[:8])):
            raise RuntimeError("inference export differs from training model")
        if not torch.equal(inference(inputs[:8]), restored(inputs[:8])):
            raise RuntimeError("inference export reload differs")
    summary["inference_checkpoint_sha256"] = sha(exported)
    summary["inference_checkpoint_bytes"] = exported.stat().st_size
    summary["inference_export_equal"] = True
    write_json(directory / "training.json", summary)
    return inference, summary


def validate_freeze(freeze_path: Path, configuration: str) -> dict[str, Any]:
    frozen: dict[str, Any] = json.loads(freeze_path.read_text())
    if frozen["status"] != "FROZEN" or frozen["selected_configuration"] != configuration:
        raise ValueError("confirmation configuration does not match frozen protocol")
    if sha(STUDY / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("protocol changed after freeze")
    required = {
        "olympus/core/forecasting.py", "olympus/foundry/resources.py",
        "scripts/run_observable_prediction.py", "scripts/analyze_observable_prediction.py",
        "scripts/verify_foundry_integrity.py",
        "scripts/execute_observable_study.py",
    }
    if not required <= frozen["source_hashes"].keys():
        raise ValueError("freeze is missing required runtime source identities")
    for path, expected in frozen["source_hashes"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"source changed after freeze: {path}")
        committed = subprocess.check_output(
            ["git", "show", frozen["code_commit"] + ":" + path], cwd=ROOT, timeout=5,
        )
        if hashlib.sha256(committed).hexdigest() != expected:
            raise ValueError("frozen source does not match the declared code commit")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", frozen["code_commit"], "HEAD"],
        cwd=ROOT, check=True, timeout=5,
    )
    return frozen


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads((STUDY / "protocol.json").read_text())
    if getattr(args, "supervisor_pid", None) != os.getppid():
        raise ValueError("resource-v2 requires the external execute_observable_study supervisor")
    configuration = next(c for c in protocol["development"]["configurations"]
                         if c["id"] == args.configuration)
    if args.phase == "confirmation":
        if args.seed not in protocol["confirmation"]["seeds"] or args.freeze is None:
            raise ValueError("confirmation requires a declared seed and frozen manifest")
        frozen = validate_freeze(args.freeze, args.configuration)
    else:
        frozen = None
        allowed = [601] if args.phase == "dry" else protocol["development"]["seeds"]
        if args.seed not in allowed:
            raise ValueError("seed is outside the declared phase")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "study_id": protocol["study_id"], "phase": args.phase, "seed": args.seed,
        "configuration": configuration, "command": sys.argv, "status": "RUNNING",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source_identity(), "protocol_sha256": sha(STUDY / "protocol.json"),
        "freeze_sha256": sha(args.freeze) if frozen is not None else None,
        "environment": {"python": sys.version, "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "families": {},
    }
    write_json(args.output / "report.json", report)
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        report["thread_settings"] = {
            "intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
        }
        with ResourceGovernor(
            ROOT / "artifacts/observable-prediction.lock", min_available_bytes=1024**3,
        ):
            report["preflight"] = preflight(args.output)
            # Fixed setup warmup makes first-use AdamW dispatch outside every measured arm.
            warmup = ObservablePredictor()
            warm_optimizer = torch.optim.AdamW(warmup.parameters())
            warmup(torch.zeros(2, 2)).square().mean().backward()
            warm_optimizer.step()
            del warmup, warm_optimizer
            for family_index, family in enumerate(protocol["families"]):
                folder = args.output / family
                folder.mkdir()
                data: dict[str, np.ndarray] = {}
                dry = args.phase == "dry"
                data["train"] = generate_split(
                    args.seed, family_index, 0, 8 if dry else 64, 32
                )
                data["development"] = generate_split(
                    args.seed, family_index, 1, 8 if dry else 24, 50
                )
                if args.phase == "confirmation":
                    data["id"] = generate_split(args.seed, family_index, 2, 64, 50)
                    data["ood"] = generate_split(args.seed, family_index, 3, 64, 50)
                initial_hashes = [array_hash(v[:, 0]) for v in data.values()]
                if len(set(initial_hashes)) != len(initial_hashes):
                    raise ValueError("split initial conditions were reused")
                seen: set[bytes] = set()
                for values in data.values():
                    starts = {row.tobytes() for row in values[:, 0]}
                    if len(starts) != len(values) or starts & seen:
                        raise ValueError("a trajectory initial condition was duplicated")
                    seen.update(starts)
                flat = data["train"].reshape(-1, 2)
                mean, std = flat.mean(axis=0), np.maximum(flat.std(axis=0), 1e-6)
                normalized = {k: (v - mean) / std for k, v in data.items()}
                inputs = normalized["train"][:, :-1].reshape(-1, 2)
                targets = normalized["train"][:, 1:].reshape(-1, 2)
                write_arrays(folder / "data.npz", {**data, "mean": mean, "std": std})
                conditions = {"development": normalized["development"]}
                if args.phase == "confirmation":
                    conditions = {"id": normalized["id"], "ood": normalized["ood"]}
                    conditions["noise"] = normalized["id"].copy()
                    rng = np.random.default_rng(
                        np.random.SeedSequence([args.seed, family_index, 4])
                    )
                    conditions["noise"][:, 0] += rng.normal(0, 0.05, (64, 2))
                    conditions["missing_velocity"] = normalized["id"].copy()
                    conditions["missing_velocity"][:, 0, 1] = (0 - mean[1]) / std[1]
                family_report: dict[str, Any] = {
                    "data_sha256": sha(folder / "data.npz"),
                    "array_hashes": {k: array_hash(v) for k, v in data.items()},
                    "initial_condition_hashes": dict(zip(data, initial_hashes, strict=True)),
                    "mean": mean.tolist(), "std": std.tolist(), "arms": {},
                }
                report["families"][family] = family_report
                arrays: dict[str, Any] = {}
                torch.manual_seed(args.seed * 100 + family_index)
                initial = {k: v.clone() for k, v in ObservablePredictor().state_dict().items()}
                initial_bytes = b"".join(v.numpy().tobytes() for v in initial.values())
                family_report["initial_state_sha256"] = hashlib.sha256(initial_bytes).hexdigest()
                cpu_budget = None
                for arm in (*NEURAL_ARMS, "persistence", "linear_ridge", "cubic_ridge"):
                    arm_arrays: dict[str, Any] = {}
                    if time.perf_counter() - started >= 95:
                        raise TimeoutError("95-second block limit reached")
                    if arm in NEURAL_ARMS:
                        model, summary = fit_neural(
                            arm, initial, torch.tensor(inputs, dtype=torch.float32),
                            torch.tensor(targets, dtype=torch.float32),
                            args.seed * 10000 + family_index * 100,
                            configuration, 10 if dry else protocol["model"]["steps"],
                            folder / arm, started + 95,
                            cpu_budget if arm == "direct_cpu_matched" else None,
                        )
                        if arm == "full":
                            cpu_budget = summary["training_cpu_seconds"]

                        predict = neural_prediction(model)
                    elif arm == "persistence":
                        summary = {"trainable_parameters": 0, "training_wall_seconds": 0.0}

                        predict = np.copy
                    else:
                        degree = 1 if arm == "linear_ridge" else 3
                        clock = time.perf_counter()
                        coefficients = fit_ridge(inputs, targets, degree)
                        summary = {"trainable_parameters": int(coefficients.size),
                                   "training_wall_seconds": time.perf_counter() - clock}
                        arm_arrays[arm + "__coefficients"] = coefficients

                        predict = ridge_prediction(degree, coefficients)
                    summary["conditions"] = {}
                    for condition, values in conditions.items():
                        result = forecast(predict, values[:, 0], values[:, 1:])
                        arm_arrays[arm + "__" + condition + "__predictions"] = result.pop(
                            "predictions"
                        )
                        arm_arrays[arm + "__" + condition + "__errors"] = result.pop("errors")
                        summary["conditions"][condition] = result
                    arm_artifact = folder / (arm + "-forecasts.npz")
                    write_arrays(arm_artifact, arm_arrays)
                    summary["forecasts_sha256"] = sha(arm_artifact)
                    arrays.update(arm_arrays)
                    family_report["arms"][arm] = summary
                    write_json(args.output / "report.json", report)
                write_arrays(folder / "forecasts.npz", arrays)
                family_report["forecasts_sha256"] = sha(folder / "forecasts.npz")
            report["status"] = "COMPLETED"
    except Exception as exc:
        report["status"] = "FAILED"
        report["failure"] = f"{type(exc).__name__}: {exc}"
    report["elapsed_seconds"] = time.perf_counter() - started
    report["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (
        1 if sys.platform == "darwin" else 1024
    )
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(args.output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["dry", "development", "confirmation"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--configuration", choices=["a", "b"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({k: report.get(k) for k in (
        "phase", "seed", "configuration", "status", "failure", "elapsed_seconds", "peak_rss_bytes"
    )}, indent=2))
    if report["status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
