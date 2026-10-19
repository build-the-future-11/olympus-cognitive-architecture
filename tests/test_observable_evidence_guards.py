"""Artificial evidence safeguards, independent of any scientific study outcomes."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from olympus.core.forecasting import ObservablePredictor
from scripts import analyze_observable_prediction as analyzer
from scripts import run_observable_prediction as runner


def test_inference_export_excludes_teacher_and_preserves_independent_rollouts(
    tmp_path: Path,
) -> None:
    model = ObservablePredictor(width=3, latent_dim=2).double()
    model.encoder = nn.Identity()
    predictor = nn.Linear(2, 2, bias=False).double()
    decoder = nn.Linear(2, 2, bias=False).double()
    model.predictor, model.decoder = predictor, decoder
    with torch.no_grad():
        predictor.weight.zero_()
        decoder.weight.copy_(2 * torch.eye(2, dtype=torch.double))
    initial = torch.tensor([[1.0, -1.0], [0.5, 0.25]], dtype=torch.double)
    expected = torch.tensor([
        [[2.0, -2.0], [4.0, -4.0], [8.0, -8.0]],
        [[1.0, 0.5], [2.0, 1.0], [4.0, 2.0]],
    ], dtype=torch.double)
    inference = model.inference_copy()
    assert not hasattr(inference, "target_encoder")
    assert set(inference.state_dict()) == {"predictor.weight", "decoder.weight"}
    assert all(not p.requires_grad for p in inference.parameters())
    assert all(not module.training for module in inference.modules())
    assert sum(p.numel() for p in inference.parameters()) == 8
    with torch.no_grad():
        assert torch.equal(inference(initial), model(initial))

    exported = tmp_path / "artificial-inference.pt"
    torch.save({"model_state": inference.state_dict()}, exported)
    payload = torch.load(exported, weights_only=True)
    assert set(payload) == {"model_state"}
    assert not any("target_encoder" in key or "optimizer" in key
                   for key in payload["model_state"])

    # Mutate both the source's online path and its teacher after export. The
    # independent copy and a fresh load must retain the original x -> 2*x rule.
    with torch.no_grad():
        predictor.weight.fill_(1)
        decoder.weight.zero_()
        for parameter in model.target_encoder.parameters():
            parameter.fill_(float("nan"))
    restored = model.inference_copy()
    restored.load_state_dict(payload["model_state"], strict=True)
    for candidate in (inference, restored):
        current = initial.clone()
        steps: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(3):
                current = candidate(current)
                steps.append(current)
        assert torch.equal(torch.stack(steps, dim=1), expected)
    with torch.no_grad():
        assert not torch.equal(model(initial), inference(initial))


def test_finite_final_parameter_control_is_accepted() -> None:
    runner.ensure_finite_parameters(ObservablePredictor(width=3, latent_dim=2))


@pytest.mark.parametrize("component", ["decoder", "target_encoder"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_final_parameter_check_includes_nontrainable_teacher(
    component: str, value: float,
) -> None:
    model = ObservablePredictor(width=3, latent_dim=2)
    parameter = next(getattr(model, component).parameters())
    with torch.no_grad():
        parameter.reshape(-1)[0] = value
    with pytest.raises(ValueError, match="nonfinite final model parameters"):
        runner.ensure_finite_parameters(model)


def test_nonfinite_last_update_retains_failed_checkpoint_receipt_and_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One artificial update only. Fault injection happens after a finite loss
    # and optimizer update, so a pre-update loss check cannot satisfy this test.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(417)
        initial = ObservablePredictor().state_dict()
    inputs = torch.tensor([[0.0, 0.1], [0.2, -0.3], [-0.4, 0.5], [0.6, -0.7]])
    targets = 0.9 * inputs + 0.1
    original_step = torch.optim.AdamW.step
    calls = 0

    def corrupt_final_update(optimizer: torch.optim.AdamW, closure: Any = None) -> Any:
        nonlocal calls
        result = original_step(optimizer, closure=closure)
        calls += 1
        with torch.no_grad():
            optimizer.param_groups[0]["params"][-1].reshape(-1)[0] = float("nan")
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", corrupt_final_update)
    monkeypatch.setattr(runner, "preflight", lambda _: {"fixture": "artificial guard"})
    directory = tmp_path / "injected-final-update"
    with pytest.raises(RuntimeError, match="nonfinite final model parameters"):
        runner.fit_neural(
            "direct", initial, inputs, targets, 417,
            {"id": "artificial-guard", "latent_weight": 0.0, "variance_weight": 0.0},
            1, directory, time.perf_counter() + 30,
        )
    assert calls == 1
    receipt = json.loads((directory / "training.json").read_bytes())
    checkpoint_path = directory / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    losses = [json.loads(row) for row in (directory / "losses.jsonl").read_text().splitlines()]
    assert receipt["status"] == checkpoint["status"] == "FAILED"
    assert "nonfinite final model parameters" in receipt["error"]
    assert receipt["optimizer_steps"] == checkpoint["steps"] == 1
    assert len(losses) == 1 and losses[0]["step"] == 1
    assert np.isfinite(list(losses[0].values())).all()
    assert any(not bool(torch.isfinite(value).all())
               for value in checkpoint["model_state"].values())
    assert receipt["checkpoint_sha256"] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert receipt["losses_sha256"] == hashlib.sha256(
        (directory / "losses.jsonl").read_bytes()
    ).hexdigest()
    assert receipt.get("inference_export_equal") is not True
    assert not (directory / "inference.pt").exists()


@pytest.mark.parametrize("failure", ["nonfinite", "unserializable"])
def test_json_serialization_failure_preserves_previous_evidence(
    tmp_path: Path, failure: str,
) -> None:
    destination = tmp_path / "report.json"
    previous = b'{"status":"RUNNING","completed_arms":3}\n'
    destination.write_bytes(previous)
    payload = {"value": float("nan") if failure == "nonfinite" else object()}
    with pytest.raises((TypeError, ValueError)):
        runner.write_json(destination, payload)
    assert destination.read_bytes() == previous
    assert set(tmp_path.iterdir()) == {destination}


@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_json_publish_failure_keeps_previous_bytes_and_cleans_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    destination = tmp_path / "report.json"
    previous = b'{"status":"RUNNING","completed_arms":3}\n'
    destination.write_bytes(previous)

    def fail_fsync(_: int) -> None:
        assert destination.read_bytes() == previous
        raise OSError("injected fsync failure")

    original_replace = Path.replace

    def fail_replace(source: Path, target: str | Path) -> Path:
        if Path(target) == destination:
            assert source != destination and source.is_file()
            assert destination.read_bytes() == previous
            raise OSError("injected replace failure")
        return original_replace(source, target)

    if failure == "fsync":
        monkeypatch.setattr(os, "fsync", fail_fsync)
    else:
        monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        runner.write_json(destination, {"status": "COMPLETED", "completed_arms": 10})
    assert destination.read_bytes() == previous
    assert set(tmp_path.iterdir()) == {destination}


def test_json_success_replaces_prior_record_and_cleans_temporary(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    destination.write_bytes(b'{"status":"RUNNING"}\n')
    expected = {"status": "COMPLETED", "metric": 0.25}
    runner.write_json(destination, expected)
    assert json.loads(destination.read_bytes()) == expected
    assert destination.read_bytes().endswith(b"\n")
    assert set(tmp_path.iterdir()) == {destination}


def test_raw_array_artifacts_are_fresh_and_immutable(tmp_path: Path) -> None:
    destination = tmp_path / "artificial-forecasts.npz"
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    runner.write_arrays(destination, {"fixture": expected})
    before = destination.read_bytes()
    with np.load(destination, allow_pickle=False) as arrays:
        assert set(arrays.files) == {"fixture"}
        np.testing.assert_array_equal(arrays["fixture"], expected)
    with pytest.raises(FileExistsError, match="immutable"):
        runner.write_arrays(destination, {"fixture": np.zeros((2, 2))})
    assert destination.read_bytes() == before
    assert set(tmp_path.iterdir()) == {destination}


def _report_skeleton(phase: str = "confirmation") -> dict[str, Any]:
    # Literal protocol membership, not analyzer constants: a production constant
    # accidentally dropping an arm/condition must fail the complete control.
    conditions: tuple[str, ...] = ("id", "ood", "noise", "missing_velocity")
    if phase != "confirmation":
        conditions = ("development",)
    arms = (
        "full", "direct", "variance", "latent", "frozen_target", "shuffled_target",
        "direct_cpu_matched", "persistence", "linear_ridge", "cubic_ridge",
    )
    return {
        "study_id": "olympus-observable-prediction-v1-20261019",
        "status": "COMPLETED", "phase": phase, "seed": 1701,
        "configuration": {"id": "a", "latent_weight": 0.1, "variance_weight": 0.01},
        "protocol_sha256": "a" * 64, "freeze_sha256": "b" * 64,
        "source": {"files_sha256": {
            "olympus/core/forecasting.py": "c" * 64,
            "scripts/run_observable_prediction.py": "d" * 64,
            "scripts/analyze_observable_prediction.py": "e" * 64,
        }},
        "environment": {"python": "artificial", "torch": "artificial",
                        "numpy": "artificial", "platform": "artificial"},
        "families": {family: {"arms": {arm: {
            "status": "COMPLETED",
            "conditions": {condition: {
                "horizons": {"1": 1.0, "10": 1.0, "20": 1.0, "50": 1.0},
                "divergent_trajectories": 0,
            } for condition in conditions},
        } for arm in arms}} for family in ("pendulum", "duffing")},
    }


def _frozen_skeleton(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "FROZEN", "selected_configuration": "a",
        "configuration": copy.deepcopy(report["configuration"]),
        "protocol_sha256": report["protocol_sha256"],
        "source_hashes": copy.deepcopy(report["source"]["files_sha256"]),
        "environment": copy.deepcopy(report["environment"]),
    }


@pytest.mark.parametrize("phase", ["dry", "development", "confirmation"])
def test_complete_literal_matrix_is_accepted_for_its_phase(phase: str) -> None:
    report = _report_skeleton(phase)
    analyzer.validate_report_contract(report)
    if phase == "confirmation":
        analyzer.validate_report_contract(report, _frozen_skeleton(report), "b" * 64)


@pytest.mark.parametrize("level", ["family", "arm", "condition", "horizon"])
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_analyzer_rejects_incomplete_or_extra_matrix_members(level: str, mutation: str) -> None:
    report = _report_skeleton()
    arm = report["families"]["pendulum"]["arms"]["shuffled_target"]
    mappings = {
        "family": (report["families"], "duffing"),
        "arm": (report["families"]["pendulum"]["arms"], "shuffled_target"),
        "condition": (arm["conditions"], "missing_velocity"),
        "horizon": (arm["conditions"]["ood"]["horizons"], "50"),
    }
    mapping, existing = mappings[level]
    if mutation == "missing":
        mapping.pop(existing)
    else:
        mapping["unexpected"] = copy.deepcopy(mapping[existing])
    with pytest.raises(ValueError, match="matrix"):
        analyzer.validate_report_contract(report)


@pytest.mark.parametrize("mutation", ["attempt", "arm", "study", "phase"])
def test_analyzer_rejects_failed_or_misidentified_matrix(mutation: str) -> None:
    report = _report_skeleton()
    if mutation == "attempt":
        report["status"] = "FAILED"
    elif mutation == "arm":
        report["families"]["duffing"]["arms"]["latent"]["status"] = "FAILED"
    elif mutation == "study":
        report["study_id"] = "artificial-wrong-study"
    else:
        report["phase"] = "unknown"
    with pytest.raises(ValueError):
        analyzer.validate_report_contract(report)


@pytest.mark.parametrize(
    "mutation",
    ["unfrozen", "missing_hash", "hash", "phase", "selection", "coefficients", "protocol",
     "source_missing", "source_changed", "runtime"],
)
def test_analyzer_binds_matrix_to_actual_frozen_configuration_and_runtime(mutation: str) -> None:
    report = _report_skeleton()
    frozen = _frozen_skeleton(report)
    freeze_hash: str | None = "b" * 64
    if mutation == "unfrozen":
        frozen["status"] = "DRAFT"
    elif mutation == "missing_hash":
        freeze_hash = None
    elif mutation == "hash":
        report["freeze_sha256"] = "f" * 64
    elif mutation == "phase":
        report = _report_skeleton("development")
    elif mutation == "selection":
        report["configuration"]["id"] = "b"
    elif mutation == "coefficients":
        report["configuration"]["latent_weight"] = 0.5
    elif mutation == "protocol":
        report["protocol_sha256"] = "f" * 64
    elif mutation == "source_missing":
        report["source"]["files_sha256"].pop("olympus/core/forecasting.py")
    elif mutation == "source_changed":
        report["source"]["files_sha256"]["olympus/core/forecasting.py"] = "f" * 64
    else:
        report["environment"]["torch"] = "different-artificial-runtime"
    with pytest.raises(ValueError):
        analyzer.validate_report_contract(report, frozen, freeze_hash)


def _cpu_records() -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = {"status": "COMPLETED", "training_cpu_seconds": 10.0, "optimizer_steps": 500}
    control = {
        "status": "COMPLETED", "training_cpu_seconds": 10.2, "cpu_budget_seconds": 10.0,
        "maximum_steps": 1500, "optimizer_steps": 700, "cap_reached": False,
        "cpu_match_valid": True,
    }
    return candidate, control


def test_cpu_matching_is_derived_from_measurements_not_a_supplied_flag() -> None:
    candidate, control = _cpu_records()
    control["cpu_match_valid"] = False
    assert analyzer.cpu_matching_valid(candidate, control)


@pytest.mark.parametrize(
    "updates",
    [
        {"status": "FAILED"}, {"training_cpu_seconds": 9.99},
        {"training_cpu_seconds": 10.5001}, {"training_cpu_seconds": float("nan")},
        {"cpu_budget_seconds": 9.0}, {"maximum_steps": 1499},
        {"optimizer_steps": 0}, {"optimizer_steps": 1500}, {"cap_reached": True},
    ],
)
def test_cpu_matching_rejects_invalid_measured_control_despite_positive_flag(
    updates: dict[str, Any],
) -> None:
    candidate, control = _cpu_records()
    control.update(updates)
    assert control["cpu_match_valid"] is True
    assert not analyzer.cpu_matching_valid(candidate, control)


@pytest.mark.parametrize(
    "seeds", [list(range(1701, 1708)), [*range(1701, 1708), 1707], [*range(1701, 1708), 1709]],
)
def test_analyzer_requires_the_exact_declared_seed_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seeds: list[int],
) -> None:
    template = _report_skeleton()
    frozen = _frozen_skeleton(template)
    freeze_path = tmp_path / "artificial-freeze.json"
    freeze_path.write_text(json.dumps(frozen), encoding="utf-8")
    actual_hash = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    paths = [tmp_path / f"artificial-block-{index}.json" for index in range(len(seeds))]
    by_path = dict(zip(paths, seeds, strict=True))

    def verified_skeleton(
        path: Path, frozen: dict[str, Any] | None = None, freeze_hash: str | None = None,
    ) -> dict[str, Any]:
        assert frozen == _frozen_skeleton(template)
        assert freeze_hash == actual_hash
        report = copy.deepcopy(template)
        report["seed"] = by_path[path]
        report["freeze_sha256"] = actual_hash
        return report

    # Isolate the aggregate seed contract from raw-artifact verification, which
    # is not substituted in the production analyzer. No study arrays are made.
    monkeypatch.setattr(analyzer, "verify_report", verified_skeleton)
    with pytest.raises(ValueError, match="eight declared confirmation seeds"):
        analyzer.analyze(paths, freeze_path)
