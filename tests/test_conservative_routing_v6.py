"""Artificial equation, gradient, disjointness and full lifecycle regressions."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from olympus.models.conservative_routing_v6 import (
    Correction,
    ReferenceRoutes,
    blend,
    correction_loss,
    features,
    intervention_diagnostics,
)
from scripts import run_conservative_routing_v6 as runner
from scripts import run_oracle_fusion_v4 as v4


def fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(991)
    p = torch.randn(12, 5, 4, generator=generator).softmax(-1)
    x = features(torch.randn(12, 5, 16, generator=generator), p)
    return x, p, p.mean(1)


def test_exact_endpoints_and_convex_bound() -> None:
    _, p, ref = fixture()
    alt = p[:, 0]
    assert torch.equal(blend(ref, alt, torch.zeros(12, 1)), ref)
    assert torch.equal(blend(ref, alt, torch.ones(12, 1)), alt)
    alpha = torch.full((12, 1), 0.2)
    result = blend(ref, alt, alpha)
    assert torch.allclose(result.sum(-1), torch.ones(12))
    # Total variation distance from reference cannot exceed alpha.
    assert bool(((result - ref).abs().sum(-1) / 2 <= 0.200001).all())


@pytest.mark.parametrize("kind", ["direct", "constant", "anchored"])
def test_gradient_and_reload(kind: str) -> None:
    x, p, ref = fixture()
    model = Correction(ReferenceRoutes().eval().requires_grad_(False).semantic, kind)
    pred, alpha = model(x, p, ref)
    loss = correction_loss(
        pred, torch.arange(12) % 4, torch.arange(12) % 3, alpha, robust=True, penalty=0.01
    )
    torch.autograd.backward(loss)
    assert all(
        v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in model.parameters()
    )
    other = copy.deepcopy(model)
    other.load_state_dict(model.state_dict())
    assert torch.equal(other(x, p, ref)[0], pred)


def test_constant_is_input_independent_and_adaptive_starts_at_declared_alpha() -> None:
    x, p, ref = fixture()
    for kind in ("constant", "anchored"):
        model = Correction(ReferenceRoutes().eval().requires_grad_(False).semantic, kind)
        _, alpha = model(x, p, ref)
        assert torch.allclose(alpha, torch.full_like(alpha, 0.1))


def test_group_objectives_hand_calculation() -> None:
    probabilities = torch.tensor([[0.8, 0.2], [0.5, 0.5], [0.25, 0.75]])
    labels, groups = torch.zeros(3, dtype=torch.long), torch.arange(3)
    alpha = torch.zeros(3, 1)
    expected = -probabilities[:, 0].log()
    assert torch.allclose(
        correction_loss(probabilities, labels, groups, alpha, robust=False, penalty=0),
        expected.mean(),
    )
    assert torch.allclose(
        correction_loss(probabilities, labels, groups, alpha, robust=True, penalty=0),
        expected.max(),
    )
    with pytest.raises(ValueError, match="every declared group"):
        correction_loss(
            probabilities, labels, torch.zeros_like(groups), alpha, robust=True, penalty=0
        )


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_alpha_rejected(value: float) -> None:
    _, p, ref = fixture()
    with pytest.raises(ValueError):
        blend(ref, p[:, 0], torch.full((12, 1), value))


def test_harm_rescue_identity() -> None:
    ref = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
    pred = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]])
    result = intervention_diagnostics(pred, ref, torch.zeros(3, dtype=torch.long))
    assert result == {
        "examples": 3,
        "rescues": 2,
        "harms": 1,
        "accuracy_delta": 1 / 3,
        "prediction_changes": 3,
    }


def test_disjointness_rejects_duplicates_and_overlap() -> None:
    runner.ensure_disjoint({"a": [1, 2], "b": [3]})
    for parts in ({"a": [1, 1]}, {"a": [1, 2], "b": [2, 3]}):
        with pytest.raises(ValueError, match="overlap"):
            runner.ensure_disjoint(parts)


def test_selection_tie_order_and_worst_regime() -> None:
    reports = {r: {"a": {"accuracy": 0.8}, "b": {"accuracy": 0.8}} for r in runner.REGIMES}
    assert runner.select(reports, ["b", "a"]) == "b"
    reports[runner.REGIMES[0]]["b"]["accuracy"] = 0.7
    assert runner.select(reports, ["b", "a"]) == "a"


def test_reserved_shift_preserves_targets_and_source_identity() -> None:
    from olympus.data.conservative_routing_v6 import generate

    clean = generate(20268001, 16, "clean")
    shifted = generate(20268001, 16, "composite_shift")
    for key in ("source_ids", "decision_targets", "next_state_targets"):
        assert torch.equal(clean[key], shifted[key])


def test_artificial_lifecycle_and_tamper_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One tiny fixture; not an admitted scientific campaign or result."""
    protocol = copy.deepcopy(runner.PROTOCOL)
    protocol.update(
        dataset_seeds=[20268900],
        training_seeds=[20268950],
        bank_examples=16,
        bank_steps=2,
        bank_batch_size=4,
        meta_examples_per_regime=8,
        test_examples_per_regime=8,
        reference_steps=2,
        correction_steps=2,
        batch_per_group=4,
        bootstrap_samples=20,
    )
    monkeypatch.setattr(runner, "PROTOCOL", protocol)
    bank_protocol = dict(v4.PROTOCOL, expert_steps=2, batch_size=4)
    monkeypatch.setattr(v4, "PROTOCOL", bank_protocol)
    # The host was admitted by the outer qualification command; no resource
    # thresholds are disabled, and the same guard is called inside the fixture.
    v4.write(
        tmp_path / "CONTRACT.json",
        {"sources": runner.sources(), "protocol": protocol, "scope": "artificial_fixture"},
    )
    import time

    deadline = time.monotonic() + 90
    runner.develop(tmp_path, deadline)
    runner.freeze(tmp_path)
    runner.confirm(tmp_path, deadline)
    v4.write(tmp_path / "SUMMARY.json", runner.summarize(tmp_path))
    assert runner.verify(tmp_path)["raw_reports_recomputed"] == 4
    raw_path = next((tmp_path / "protected").glob("*/clean.pt"))
    report_path = raw_path.with_suffix(".json")
    original_raw, original_report = raw_path.read_bytes(), report_path.read_bytes()
    raw = torch.load(raw_path, weights_only=True)
    raw["source_ids"] = raw["source_ids"] + 100000000000
    torch.save(raw, raw_path)
    report = v4.read(report_path)
    report["prediction_sha256"] = v4.sha(raw_path)
    import json

    report_path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="regenerated source IDs"):
        runner.verify(tmp_path)
    raw_path.write_bytes(original_raw)
    report_path.write_bytes(original_report)
    # A self-consistent changed prediction/hash/metric is still not checkpoint evidence.
    raw = torch.load(raw_path, weights_only=True)
    raw["probabilities"]["anchored_robust"] = raw["probabilities"]["anchored_robust"].roll(1, -1)
    torch.save(raw, raw_path)
    report = v4.read(report_path)
    report["prediction_sha256"] = v4.sha(raw_path)
    report["metrics"] = runner.describe(raw["probabilities"], raw["alpha"], raw["labels"])
    report_path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="checkpoint prediction replay"):
        runner.verify(tmp_path)
    raw_path.write_bytes(original_raw)
    report_path.write_bytes(original_report)
    freeze = tmp_path / "SCIENTIFIC_FREEZE.json"
    original_freeze = freeze.read_bytes()
    altered = v4.read(freeze)
    altered["files"].pop(next(iter(altered["files"])))
    freeze.write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="manifest membership"):
        runner.validate_freeze(tmp_path)
    freeze.write_bytes(original_freeze)
    contract = tmp_path / "CONTRACT.json"
    original_contract = contract.read_bytes()
    changed_contract = v4.read(contract)
    changed_contract["python"] = "tampered environment"
    contract.write_text(json.dumps(changed_contract))
    with pytest.raises(RuntimeError, match="changed frozen file"):
        runner.validate_freeze(tmp_path)
    contract.write_bytes(original_contract)
    checkpoint = next((tmp_path / "development").glob("*/checkpoint.pt"))
    with checkpoint.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(RuntimeError, match="checkpoint hash|changed frozen file"):
        runner.verify(tmp_path)
