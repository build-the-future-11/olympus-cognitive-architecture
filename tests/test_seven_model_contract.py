from __future__ import annotations

import pytest
import torch

from olympus.models.reference_families import (
    AionReferenceConfig,
    AionReferenceModel,
    AtlasReferenceConfig,
    AtlasReferenceModel,
    KronosReferenceConfig,
    KronosReferenceModel,
    PrometheusReferenceConfig,
    PrometheusReferenceModel,
    ReferenceTransformerConfig,
)
from olympus.models.seven_model_contract import (
    SEVEN_MODEL_IDS,
    audit_seven_model_contracts,
    build_seven_model_reference,
    default_seven_model_configs,
    inspect_seven_model_references,
    seven_model_id,
)
from olympus.upscaler.jepa import JEPAUpscaler, JEPAUpscalerConfig


def _backbone() -> ReferenceTransformerConfig:
    return ReferenceTransformerConfig(
        vocab_size=16,
        model_dim=8,
        layers=1,
        heads=2,
        feedforward_dim=16,
        max_sequence_length=4,
    )


def test_exact_seven_model_catalog_builds_and_passes_deterministic_contracts() -> None:
    configs = default_seven_model_configs()
    assert tuple(seven_model_id(config) for config in configs) == SEVEN_MODEL_IDS
    assert len({seven_model_id(config) for config in configs}) == 7
    assert all(
        sum(
            parameter.numel()
            for parameter in build_seven_model_reference(config).parameters()
        )
        > 0
        for config in configs
    )

    first = audit_seven_model_contracts()
    second = audit_seven_model_contracts()
    assert first == second
    assert tuple(status.model_id for status in first) == SEVEN_MODEL_IDS
    assert all(status.forward_contract == "PASS" for status in first)
    assert all(status.finite_contract == "PASS" for status in first)
    assert all(status.backward_contract == "PASS" for status in first)
    assert all(status.reference_parameters >= status.trainable_parameters > 0 for status in first)
    assert all(
        status.gradient_parameter_tensors == status.trainable_parameter_tensors
        for status in first
    )
    assert all(status.gradient_coverage == "PASS" for status in first)
    assert all(not status.target_capability_claimed for status in first)
    assert all(not status.promotion_authorized for status in first)
    assert all(not status.scientific_evaluation for status in first)


def test_fast_inventory_preserves_catalog_without_executing_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_probe(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fast inventory must not execute artificial probes")

    monkeypatch.setattr(
        "olympus.models.seven_model_contract._probe_hermes", unexpected_probe
    )
    inventory = inspect_seven_model_references()
    assert tuple(status.model_id for status in inventory) == SEVEN_MODEL_IDS
    assert all(status.contract_execution_required for status in inventory)
    assert all(
        status.reference_parameters >= status.trainable_parameters > 0 for status in inventory
    )


def test_contract_fails_closed_when_a_probe_returns_a_nonfinite_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "olympus.models.seven_model_contract._probe_hermes",
        lambda _model: torch.tensor(float("nan")),
    )
    with pytest.raises(RuntimeError, match="non-finite scalar"):
        audit_seven_model_contracts()


def test_language_and_domain_admission_reject_ambiguous_tensor_types() -> None:
    tokens = torch.ones((1, 2), dtype=torch.long)
    prometheus = PrometheusReferenceModel(PrometheusReferenceConfig(backbone=_backbone()))
    with pytest.raises(ValueError, match="Boolean dtype"):
        prometheus(tokens, torch.ones((1, 2), dtype=torch.int64))
    with pytest.raises(ValueError, match="integer dtype"):
        prometheus(tokens, domain_ids=torch.tensor([0.0]))


def test_atlas_rejects_invalid_images_and_temporal_coordinates() -> None:
    model = AtlasReferenceModel(AtlasReferenceConfig(backbone=_backbone(), patch_size=2))
    tokens = torch.ones((1, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="floating-point"):
        model(tokens, torch.ones((1, 3, 4, 4), dtype=torch.int64))
    with pytest.raises(ValueError, match="finite"):
        model(tokens, torch.full((1, 3, 4, 4), torch.nan))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model(tokens, torch.full((1, 3, 4, 4), 2.0))
    with pytest.raises(ValueError, match="timestamp.*floating-point"):
        model(tokens, torch.rand(1, 3, 4, 4), timestamp=torch.tensor([1]))
    with pytest.raises(ValueError, match="timestamp values must be finite"):
        model(tokens, torch.rand(1, 3, 4, 4), timestamp=torch.tensor([float("nan")]))


def test_kronos_zeroes_padded_outputs_and_rejects_non_boolean_masks() -> None:
    model = KronosReferenceModel(
        KronosReferenceConfig(
            state_dim=3,
            action_dim=2,
            hidden_dim=6,
            discrete_actions=3,
            planning_horizon=2,
        )
    )
    states = torch.rand(1, 3, 3)
    actions = torch.rand(1, 3, 2)
    mask = torch.tensor([[True, False, False]])
    output = model(states, actions, mask)
    assert torch.equal(output.hidden_states[:, 1:], torch.zeros(1, 2, 6))
    assert torch.equal(output.next_state[:, 1:], torch.zeros(1, 2, 3))
    assert torch.equal(output.reward[:, 1:], torch.zeros(1, 2))
    assert torch.equal(output.value[:, 1:], torch.zeros(1, 2))
    assert torch.equal(output.uncertainty[:, 1:], torch.zeros(1, 2, 3))
    with pytest.raises(ValueError, match="Boolean dtype"):
        model(states, actions, mask.to(dtype=torch.int64))
    with pytest.raises(ValueError, match="floating-point"):
        model(states.to(dtype=torch.int64), actions)


def test_aion_rejects_coerced_authority_and_risk_tensors() -> None:
    model = AionReferenceModel(
        AionReferenceConfig(state_dim=3, hidden_dim=4, research_stages=3, stop_stage=2)
    )
    states = torch.rand(1, 3)
    allowed = torch.tensor([[True, True, True]])
    with pytest.raises(ValueError, match="Boolean dtype"):
        model(states, allowed.to(dtype=torch.int64), torch.tensor([0.2]))
    with pytest.raises(ValueError, match="floating-point dtype"):
        model(states, allowed, torch.tensor([0]))


def test_jepa_rejects_ambiguous_masks_and_mismatched_target_dtype() -> None:
    model = JEPAUpscaler(
        JEPAUpscalerConfig(
            scale=2,
            feature_channels=8,
            residual_blocks=1,
            predictor_dilations=(1, 2),
        )
    )
    low = torch.rand(1, 3, 4, 4)
    high = torch.rand(1, 3, 8, 8)
    with pytest.raises(ValueError, match="Boolean dtype"):
        model(low, high, context_mask=torch.zeros(1, 4, 4))
    with pytest.raises(ValueError, match="device and dtype"):
        model(low, high.to(dtype=torch.float64))
    with pytest.raises(ValueError, match="timestamp.*floating-point"):
        model(low, high, timestamp=torch.tensor([1]))
