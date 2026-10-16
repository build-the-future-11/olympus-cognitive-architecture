from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from olympus.data.olympus_dataset import load_olympus_dataset
from olympus.data.reliability_fusion import (
    COMPONENT_ORDER,
    ReliabilitySplitSpec,
    generate_reliability_fusion_dataset,
    generate_reliability_split,
)
from olympus.models.olympus_system import OlympusSystemConfig
from olympus.models.reference_families import (
    AionReferenceConfig,
    AtlasReferenceConfig,
    HermesReferenceConfig,
    KronosReferenceConfig,
    PerseusReferenceConfig,
    PrometheusReferenceConfig,
    ReferenceTransformerConfig,
)
from olympus.models.reliability_fusion import (
    ReliabilityAwareOlympusSystem,
    ReliabilityFusionConfig,
    ReliabilityLossConfig,
    reliability_fusion_loss,
)


def _system_config() -> OlympusSystemConfig:
    backbone = ReferenceTransformerConfig(
        vocab_size=32,
        model_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=24,
        max_sequence_length=8,
    )
    return OlympusSystemConfig(
        hermes=HermesReferenceConfig(backbone=backbone, tool_count=3),
        prometheus=PrometheusReferenceConfig(
            backbone=backbone, domains=3, adapter_dim=8, evidence_classes=2
        ),
        perseus=PerseusReferenceConfig(
            backbone=backbone, experts=3, active_experts=2, expert_dim=24
        ),
        atlas=AtlasReferenceConfig(
            backbone=backbone, image_channels=3, patch_size=2, retrieval_dim=8
        ),
        kronos=KronosReferenceConfig(
            state_dim=5,
            action_dim=3,
            hidden_dim=8,
            discrete_actions=4,
            planning_horizon=3,
        ),
        aion=AionReferenceConfig(
            state_dim=6,
            hidden_dim=8,
            research_stages=4,
            stop_stage=3,
            kill_threshold=0.95,
        ),
        fusion_dim=12,
        decision_classes=4,
    )


def _batch() -> dict[str, torch.Tensor]:
    bundle, _ = generate_reliability_split(
        ReliabilitySplitSpec(name="unit", examples=4, seed=91, regime="corrupted")
    )
    return bundle


def _inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    names = {
        "tokens",
        "attention_mask",
        "image",
        "states",
        "actions",
        "temporal_mask",
        "allowed_transitions",
        "risk_score",
        "domain_ids",
        "timestamp",
    }
    return {name: value for name, value in batch.items() if name in names}


def test_reliability_split_is_deterministic_and_component_aligned() -> None:
    spec = ReliabilitySplitSpec(name="tiny", examples=12, seed=7, regime="corrupted")
    first, first_ids = generate_reliability_split(spec)
    second, second_ids = generate_reliability_split(spec)
    assert first_ids == second_ids
    assert all(torch.equal(first[name], second[name]) for name in first)
    targets = first["component_reliability_targets"]
    assert targets.shape == (12, len(COMPONENT_ORDER))
    assert bool(((targets == 0.0) | (targets == 0.25) | (targets == 1.0)).all())
    assert bool((targets < 1.0).any(dim=1).all())


def test_reliability_router_is_parameter_matched_bounded_and_differentiable() -> None:
    torch.manual_seed(101)
    configs = {
        mode: ReliabilityFusionConfig(
            system=_system_config(),
            routing_mode=mode,
            reliability_logit_strength=1.0,
            shrinkage_to_adaptive=0.75,
        )
        for mode in ("reliability_shrinkage", "semantic", "uniform", "reliability_only")
    }
    models = {mode: ReliabilityAwareOlympusSystem(config) for mode, config in configs.items()}
    reference_state = models["reliability_shrinkage"].state_dict()
    for model in models.values():
        model.load_state_dict(reference_state)
    parameter_counts = {
        sum(parameter.numel() for parameter in model.parameters()) for model in models.values()
    }
    assert len(parameter_counts) == 1

    batch = _batch()
    full = models["reliability_shrinkage"](**_inputs(batch))
    assert full.component_reliability.shape == (4, len(COMPONENT_ORDER))
    assert torch.allclose(full.system.family_gates.sum(dim=-1), torch.ones(4))
    assert bool((full.system.family_gates >= 0.05 - 1e-7).all())

    uniform = models["uniform"](**_inputs(batch))
    assert torch.allclose(
        uniform.system.family_gates,
        torch.full_like(uniform.system.family_gates, 1.0 / len(COMPONENT_ORDER)),
    )
    loss = reliability_fusion_loss(
        full,
        decision_targets=batch["decision_targets"],
        transition_targets=batch["transition_targets"],
        next_state_targets=batch["next_state_targets"],
        component_reliability_targets=batch["component_reliability_targets"],
        intrinsic_risk_targets=batch["intrinsic_risk_targets"],
        temporal_mask=batch["temporal_mask"],
        config=ReliabilityLossConfig(),
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()  # type: ignore[no-untyped-call]
    atlas_head = models["reliability_shrinkage"].reliability_heads["atlas"]
    assert isinstance(atlas_head, torch.nn.Sequential)
    final_head = atlas_head[-1]
    assert isinstance(final_head, torch.nn.Linear)
    assert final_head.weight.grad is not None


def test_reliability_loss_rejects_unaligned_or_invalid_targets() -> None:
    batch = _batch()
    model = ReliabilityAwareOlympusSystem(ReliabilityFusionConfig(system=_system_config()))
    output = model(**_inputs(batch))
    with pytest.raises(ValueError, match=r"\[batch, component\]"):
        reliability_fusion_loss(
            output,
            decision_targets=batch["decision_targets"],
            transition_targets=batch["transition_targets"],
            next_state_targets=batch["next_state_targets"],
            component_reliability_targets=torch.ones(4, 4),
        )
    invalid = batch["component_reliability_targets"].clone()
    invalid[0, 0] = 2.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        reliability_fusion_loss(
            output,
            decision_targets=batch["decision_targets"],
            transition_targets=batch["transition_targets"],
            next_state_targets=batch["next_state_targets"],
            component_reliability_targets=invalid,
        )


def test_reliability_dataset_is_hash_bound_and_protected(tmp_path: Path) -> None:
    root = tmp_path / "reliability-v3"
    receipt = generate_reliability_fusion_dataset(root)
    assert receipt.source_overlap_count == 0
    assert receipt.component_order == list(COMPONENT_ORDER)
    loaded = load_olympus_dataset(root / "manifest_development.json")
    assert loaded.train["component_reliability_targets"].shape == (512, 5)
    assert loaded.evaluation["component_reliability_targets"].shape == (128, 5)
    raw = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    assert raw["protected_splits"] == ["clean_test", "corrupted_test", "ood_test"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_reliability_fusion_dataset(root)
