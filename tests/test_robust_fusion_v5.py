from __future__ import annotations

import copy
from typing import cast

import pytest
import torch
from torch import nn

from olympus.data.robust_fusion_v5 import generate
from olympus.models.oracle_fusion_v4 import ExpertBank
from olympus.models.robust_fusion_v5 import Heads, routed
from scripts.run_robust_fusion_v5 import PROTOCOL, select_route


def test_coverage_pairs_identical_latents_labels_missingness() -> None:
    core, broad = (generate(20265101, 256, r) for r in ("core", "broad"))
    for key in (
        "source_ids",
        "decision_targets",
        "next_state_targets",
        "corruption_status",
        "domain_ids",
        "allowed_transitions",
    ):
        assert torch.equal(core[key], broad[key]), key
    for i, key in enumerate(("tokens", "image", "states")):
        unchanged = core["corruption_status"][:, i] != 2
        assert torch.equal(core[key][unchanged], broad[key][unchanged])
        assert not torch.equal(core[key], broad[key])


def test_novel_corruptions_are_not_development_interventions() -> None:
    novel = generate(20265114, 256, "novel_shift")
    known = generate(20265114, 256, "known_shift")
    clean = generate(20265114, 256, "clean")
    assert "novel_shift" not in PROTOCOL["development_regimes"]
    assert torch.equal(novel["next_state_targets"], known["next_state_targets"])
    for i, key in enumerate(("tokens", "image", "states")):
        damaged = novel["corruption_status"][:, i] == 2
        assert not torch.equal(novel[key][damaged], known[key][damaged])
        valid = novel["corruption_status"][:, i] == 0
        assert torch.equal(novel[key][valid], clean[key][valid])


def test_fresh_ids_and_reproducibility() -> None:
    a = generate(20265101, 32, "broad")
    b = generate(20265114, 32, "novel_shift")
    assert not set(a["source_ids"].tolist()) & set(b["source_ids"].tolist())
    for k, value in a.items():
        assert torch.equal(value, generate(20265101, 32, "broad")[k])
    with pytest.raises(ValueError):
        generate(1, 1, "undefined")


def test_residual_is_exact_persistence_at_initialization_and_pair_is_matched() -> None:
    heads = Heads()
    assert all(
        torch.equal(x, y)
        for x, y in zip(
            heads.temporal["direct"].parameters(),
            heads.temporal["residual"].parameters(),
            strict=True,
        )
    )
    z, logits, states, actions = (
        torch.randn(8, 5, 16),
        torch.randn(8, 5, 4),
        torch.randn(8, 4, 5),
        torch.randn(8, 4, 3),
    )
    _, temporal = heads(z, logits, states, actions)
    assert torch.equal(temporal["residual"], states)
    assert torch.count_nonzero(temporal["direct"]) == 0
    loss = (temporal["residual"] - torch.randn_like(states)).square().mean()
    loss.backward()
    residual_last = cast(nn.Linear, cast(nn.Sequential, heads.temporal["residual"])[-1])
    direct_last = cast(nn.Linear, cast(nn.Sequential, heads.temporal["direct"])[-1])
    assert residual_last.weight.grad is not None
    assert residual_last.weight.grad.abs().sum() > 0
    assert direct_last.weight.grad is None


def test_prediction_only_router_cannot_read_latent_features() -> None:
    heads = Heads()
    logits, states, actions = torch.randn(8, 5, 4), torch.randn(8, 4, 5), torch.randn(8, 4, 3)
    a, _ = heads(torch.randn(8, 5, 16), logits, states, actions)
    b, _ = heads(torch.randn(8, 5, 16) * 10, logits, states, actions)
    assert torch.equal(a["prediction_only"], b["prediction_only"])
    assert not torch.equal(a["semantic"], b["semantic"])
    assert sum(p.numel() for p in heads.routing["semantic"].parameters()) == sum(
        p.numel() for p in heads.routing["prediction_only"].parameters()
    )


def test_nonoracle_predictions_ignore_targets_and_quality() -> None:
    batch = generate(20265101, 8, "broad")
    bank, heads = ExpertBank().eval(), Heads().eval()
    with torch.no_grad():
        z, logits, _ = bank(batch)
        scores, temporal = heads(z, logits, batch["states"], batch["actions"])
        expected = routed(logits, scores)
        changed = copy.deepcopy(batch)
        for key in (
            "decision_targets",
            "next_state_targets",
            "component_quality",
            "corruption_status",
        ):
            changed[key].zero_()
        z2, logits2, _ = bank(changed)
        scores2, temporal2 = heads(z2, logits2, changed["states"], changed["actions"])
        actual = routed(logits2, scores2)
    assert "oracle_task" not in expected
    assert all(torch.equal(expected[k], actual[k]) for k in expected)
    assert all(torch.equal(temporal[k], temporal2[k]) for k in temporal)
    assert torch.allclose(
        expected["blend"], 0.5 * (expected["prediction_only"] + expected["utility"])
    )


def test_selection_uses_worst_development_regime_not_protected_data() -> None:
    reports = {
        r: {"routing": {k: {"accuracy": 0.7} for k in PROTOCOL["candidate_routes"]}}
        for r in PROTOCOL["development_regimes"]
    }
    reports["clean"]["routing"]["semantic"]["accuracy"] = 1
    reports["corrupted"]["routing"]["semantic"]["accuracy"] = 0.95
    reports["known_shift"]["routing"]["semantic"]["accuracy"] = 0.6
    assert select_route(reports) == "prediction_only"
