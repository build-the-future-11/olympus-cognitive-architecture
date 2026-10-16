"""Checkpoint metadata checks run without importing Torch or allocating model state."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class _TensorStandIn:
    pass


def _load_validators() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    extracted: list[ast.stmt] = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ]
    for filename, names in (
        ("olympus/training/provenance.py", {"digest_json"}),
        (
            "olympus/training/checkpoints.py",
            {"_validate_checkpoint_contract", "_validate_checkpoint_tensors"},
        ),
    ):
        tree = ast.parse((root / filename).read_text())
        functions = [
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        assert {node.name for node in functions} == names
        extracted.extend(functions)
    module = ast.fix_missing_locations(ast.Module(body=extracted, type_ignores=[]))
    namespace: dict[str, Any] = {
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "CHECKPOINT_SCHEMA": 2,
        "torch": SimpleNamespace(Tensor=_TensorStandIn),
    }
    exec(compile(module, "checkpoint_contract_source", "exec"), namespace)
    return namespace


def test_contract_inspection_does_not_import_numerical_libraries() -> None:
    before = {name for name in sys.modules if name.split(".")[0] in {"torch", "numpy"}}
    _load_validators()
    after = {name for name in sys.modules if name.split(".")[0] in {"torch", "numpy"}}
    assert before == after


@pytest.fixture
def contract() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    validators = _load_validators()
    config = {
        "steps": 5,
        "precision": "fp32",
        "model": {"decision_classes": 2},
        "workload_lock": "locks/one.lock",
    }
    binding = {
        "training_config_sha256": validators["digest_json"](
            {key: value for key, value in config.items() if key != "workload_lock"}
        ),
        "source_sha256": "a" * 64,
    }
    counters = {
        "completed_steps": 2,
        "best_step": 1,
        "best_validation_loss": 0.2,
        "first_loss": 0.8,
        "final_loss": 0.3,
        "maximum_gradient_norm": 1.0,
        "final_learning_rate": 0.001,
        "elapsed_seconds": 1.5,
    }
    pointer = {"schema_version": 2, "binding": binding, "completed_steps": 2}
    payload = {
        **pointer,
        "training_config": config,
        "counters": counters,
        "scheduler": {"kind": "functional_warmup_cosine", "total_steps": 5},
        "precision": "fp32",
        "scaler": None,
        "model_state": {"weight": _TensorStandIn()},
        "optimizer_state": {"state": {}, "param_groups": [{"params": [0]}]},
        "source_identity": {"sha256": binding["source_sha256"]},
        "initial_evaluation": {},
        "rng": {
            "python": (),
            "numpy": [],
            "torch_cpu": _TensorStandIn(),
            "sampler": _TensorStandIn(),
            "torch_cuda": [],
        },
    }
    return validators, payload, pointer, binding


def _set_path(mapping: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target = mapping
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


@pytest.mark.parametrize("completed", [0, 2, 5])
def test_valid_optimizer_boundaries(contract: Any, completed: int) -> None:
    validators, payload, pointer, binding = contract
    payload["completed_steps"] = pointer["completed_steps"] = completed
    payload["counters"]["completed_steps"] = completed
    payload["counters"]["best_step"] = min(1, completed)
    if completed == 0:
        payload["counters"]["first_loss"] = None
        payload["counters"]["final_loss"] = None
    validators["_validate_checkpoint_contract"](payload, pointer, binding)
    validators["_validate_checkpoint_tensors"](payload)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("schema_version", True, "schema"),
        ("completed_steps", -1, "declared training budget"),
        ("completed_steps", 6, "declared training budget"),
        ("completed_steps", True, "declared training budget"),
        ("completed_steps", 2.0, "declared training budget"),
        ("counters.completed_steps", 1, "nested completed-step counter"),
        ("counters.completed_steps", True, "nested completed-step counter"),
        ("counters.best_step", -1, "best step"),
        ("counters.best_step", 3, "best step"),
        ("counters.best_step", 1.0, "best step"),
        ("counters.best_validation_loss", math.nan, "must be finite"),
        ("counters.best_validation_loss", math.inf, "must be finite"),
        ("counters.best_validation_loss", True, "must be finite"),
        ("counters.maximum_gradient_norm", -1.0, "non-negative"),
        ("counters.final_learning_rate", -0.1, "non-negative"),
        ("counters.elapsed_seconds", -1.0, "non-negative"),
        ("counters.first_loss", None, "finite after optimizer"),
        ("counters.final_loss", math.inf, "finite after optimizer"),
        ("scheduler.kind", "unknown", "scheduler"),
        ("scheduler.total_steps", 9, "scheduler"),
        ("scheduler.total_steps", True, "scheduler"),
        ("precision", "fp16", "fp32"),
        ("scaler", {}, "AMP scaler"),
        ("model_state", {}, "non-empty named mapping"),
        ("model_state", {1: "weight"}, "non-empty named mapping"),
        ("optimizer_state", {}, "optimizer state"),
        ("optimizer_state.param_groups", [{"params": [True]}], "optimizer state"),
        ("source_identity.sha256", "b" * 64, "source identity"),
        ("initial_evaluation", None, "initial evaluation"),
        ("rng", {}, "every supported RNG"),
    ],
)
def test_inconsistent_checkpoint_metadata_fails_closed(
    contract: Any, path: str, value: Any, message: str
) -> None:
    validators, payload, pointer, binding = contract
    _set_path(payload, path, value)
    with pytest.raises(ValueError, match=message):
        validators["_validate_checkpoint_contract"](payload, pointer, binding)


@pytest.mark.parametrize("counter", [False, 1, 3, 2.0])
def test_pointer_counter_must_match_the_payload(contract: Any, counter: Any) -> None:
    validators, payload, pointer, binding = contract
    pointer["completed_steps"] = counter
    with pytest.raises(ValueError, match="pointer counter"):
        validators["_validate_checkpoint_contract"](payload, pointer, binding)


def test_raw_configuration_is_bound_except_for_lock_relocation(contract: Any) -> None:
    validators, payload, pointer, binding = contract
    payload["training_config"]["workload_lock"] = "another/location.lock"
    validators["_validate_checkpoint_contract"](payload, pointer, binding)
    payload["training_config"]["model"]["decision_classes"] = 3
    with pytest.raises(ValueError, match="raw configuration digest"):
        validators["_validate_checkpoint_contract"](payload, pointer, binding)


def test_step_zero_requires_absent_training_losses(contract: Any) -> None:
    validators, payload, pointer, binding = contract
    payload["completed_steps"] = pointer["completed_steps"] = 0
    payload["counters"]["completed_steps"] = payload["counters"]["best_step"] = 0
    with pytest.raises(ValueError, match="null before the first optimizer"):
        validators["_validate_checkpoint_contract"](payload, pointer, binding)


@pytest.mark.parametrize("path", ["model_state.weight", "rng.torch_cpu", "rng.sampler"])
def test_tensor_state_fields_reject_arbitrary_objects(contract: Any, path: str) -> None:
    validators, payload, _pointer, _binding = contract
    _set_path(payload, path, "not a tensor")
    with pytest.raises(ValueError, match="tensor"):
        validators["_validate_checkpoint_tensors"](payload)


@pytest.mark.parametrize("value", [None, [_TensorStandIn(), "not a tensor"]])
def test_cuda_rng_requires_a_tensor_list(contract: Any, value: Any) -> None:
    validators, payload, _pointer, _binding = contract
    payload["rng"]["torch_cuda"] = copy.deepcopy(value)
    with pytest.raises(ValueError, match="CUDA RNG"):
        validators["_validate_checkpoint_tensors"](payload)
