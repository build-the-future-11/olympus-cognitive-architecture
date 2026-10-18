"""Leakage, privileged-oracle separation, matching and preservation regressions."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from olympus.data.oracle_fusion_v4 import dynamics, generate, subset
from olympus.models.oracle_fusion_v4 import (
    MODES,
    ExpertBank,
    Routers,
    expert_loss,
    probabilities,
    router_features,
)
from scripts.run_oracle_fusion_v4 import metrics, write


def test_fresh_reproducible_splits_and_full_corruption_support() -> None:
    train = generate(100, 512, "mixed")
    repeat = generate(100, 512, "mixed")
    test = generate(101, 512, "corrupted")
    for key in train:
        assert torch.equal(train[key], repeat[key])
    assert not set(train["source_ids"].tolist()) & set(test["source_ids"].tolist())
    for i in range(3):
        assert set(train["corruption_status"][:, i].tolist()) == {0, 1, 2}
    old = generate(100, 512, "missing_only")
    assert not (old["corruption_status"] == 2).any()
    assert torch.equal(train["next_state_targets"], old["next_state_targets"])
    assert torch.equal(train["decision_targets"], old["decision_targets"])


def test_metadata_does_not_encode_answers() -> None:
    batch = generate(102, 1024, "mixed")
    assert batch["allowed_transitions"].all()
    for d in range(3):
        labels = batch["decision_targets"][batch["domain_ids"] == d]
        assert set(labels.tolist()) == {0, 1, 2, 3}
    assert not torch.equal(batch["domain_ids"], batch["decision_targets"] % 3)


def test_future_target_is_clean_dynamics_not_observed_formula() -> None:
    batch = generate(103, 128, "corrupted")
    old = batch["states"] + 0.1 * torch.nn.functional.pad(batch["actions"], (0, 2))
    assert (old - batch["next_state_targets"]).square().mean() > 0.01
    # Recurrence between consecutive clean targets is exact and testable even
    # when observed state inputs were erased or corrupted.
    expected = dynamics(batch["next_state_targets"][:, :-1], batch["actions"][:, 1:])
    torch.testing.assert_close(expected, batch["next_state_targets"][:, 1:])


def test_bank_inference_never_reads_targets_or_corruption() -> None:
    torch.manual_seed(1)
    bank = ExpertBank().eval()
    batch = generate(104, 8, "mixed")
    with torch.no_grad():
        before = bank(batch)
        for key in (
            "decision_targets",
            "transition_targets",
            "next_state_targets",
            "component_quality",
            "corruption_status",
            "source_ids",
        ):
            batch[key] = torch.zeros_like(batch[key])
        after = bank(batch)
    for left, right in zip(before, after, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_oracle_is_explicit_and_nonoracle_cannot_use_privileged_fields() -> None:
    torch.manual_seed(2)
    logits = torch.randn(8, 5, 4)
    router = Routers()
    scores = router(router_features(torch.randn(8, 5, 16), logits))
    for mode in MODES:
        if mode.startswith("oracle"):
            with pytest.raises(ValueError):
                probabilities(logits, scores, mode)
        else:
            before, gates = probabilities(logits, scores, mode)
            after, _ = probabilities(
                logits, scores, mode, quality=torch.rand(8, 5), labels=torch.randint(4, (8,))
            )
            torch.testing.assert_close(before, after, atol=0, rtol=0)
            torch.testing.assert_close(gates.sum(-1), torch.ones(8))


def test_task_oracle_can_find_correct_expert_and_corruption_oracle_masks_missing() -> None:
    logits = torch.tensor(
        [
            [
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
            ]
        ]
    )
    p, _ = probabilities(logits, {}, "oracle_task", labels=torch.tensor([0]))
    assert p.argmax(-1).item() == 0
    p, gates = probabilities(
        logits, {}, "oracle_corruption", quality=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    )
    assert p.argmax(-1).item() == 0
    assert torch.equal(gates, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]))


def test_crossfit_indices_disjoint_and_exhaustive() -> None:
    batch = generate(105, 128, "mixed")
    for fold in range(2):
        fit = subset(batch, torch.arange(fold, 128, 2))["source_ids"].tolist()
        held = subset(batch, torch.arange(1 - fold, 128, 2))["source_ids"].tolist()
        assert set(fit).isdisjoint(held)
        assert len(set(fit) | set(held)) == 128


def test_counterfactual_ablation_is_matched_and_has_real_gradient_effect() -> None:
    torch.manual_seed(3)
    logits = torch.randn(16, 5, 4, requires_grad=True)
    labels = torch.randint(4, (16,))
    plain = expert_loss(logits, labels, 0)
    penalty = expert_loss(logits, labels, 0.05)
    assert penalty > plain
    a = torch.autograd.grad(plain, logits, retain_graph=True)[0]
    b = torch.autograd.grad(penalty, logits)[0]
    assert (a - b).abs().max() > 1e-6


def test_metrics_match_manual_calculation_and_outputs_are_exclusive(tmp_path: Path) -> None:
    p = torch.tensor(
        [[0.8, 0.1, 0.05, 0.05], [0.1, 0.7, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]]
    )
    result = metrics(p, torch.tensor([0, 1, 3, 2]))
    assert result["accuracy"] == 0.75
    assert result["selective_accuracy_50"] == 1
    path = tmp_path / "receipt.json"
    write(path, {"negative": True})
    with pytest.raises(FileExistsError):
        write(path, {"negative": False})
