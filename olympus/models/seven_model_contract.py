"""Unified engineering contract for the six family references and JEPA upscaler.

The probes in this module use tiny artificial tensors.  They establish executable
and differentiable software paths only; they do not train, evaluate, or promote a
model-family checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias, cast

import torch
from pydantic import Field
from torch import Tensor, nn

from olympus.core.schemas import StrictModel
from olympus.models.reference_families import (
    AionReferenceConfig,
    AionReferenceModel,
    AtlasReferenceConfig,
    AtlasReferenceModel,
    HermesReferenceConfig,
    HermesReferenceModel,
    KronosReferenceConfig,
    KronosReferenceModel,
    PerseusReferenceConfig,
    PerseusReferenceModel,
    PrometheusReferenceConfig,
    PrometheusReferenceModel,
    ReferenceFamilyConfig,
    ReferenceTransformerConfig,
    build_reference_model,
)
from olympus.upscaler.jepa import JEPAUpscaler, JEPAUpscalerConfig, jepa_upscaler_loss

SevenModelId = Literal[
    "hermes",
    "prometheus",
    "perseus",
    "atlas",
    "kronos",
    "aion",
    "jepa_upscaler",
]
# Keep this parseable by the repository's Python-3.11 source-compatibility gate.
SevenModelConfig: TypeAlias = ReferenceFamilyConfig | JEPAUpscalerConfig  # noqa: UP040

SEVEN_MODEL_IDS: tuple[SevenModelId, ...] = (
    "hermes",
    "prometheus",
    "perseus",
    "atlas",
    "kronos",
    "aion",
    "jepa_upscaler",
)


class SevenModelEngineeringStatus(StrictModel):
    """Machine-readable result from one bounded artificial contract probe."""

    model_id: SevenModelId
    display_name: str
    model_kind: Literal["family_reference", "upscaler_reference"]
    implementation: str
    reference_parameters: int = Field(gt=0)
    trainable_parameters: int = Field(gt=0)
    trainable_parameter_tensors: int = Field(gt=0)
    gradient_parameter_tensors: int = Field(gt=0)
    gradient_coverage: Literal["PASS"] = "PASS"
    forward_contract: Literal["PASS"] = "PASS"
    finite_contract: Literal["PASS"] = "PASS"
    backward_contract: Literal["PASS"] = "PASS"
    status: Literal["ENGINEERING_REFERENCE_COMPLETE_NOT_PROMOTED"] = (
        "ENGINEERING_REFERENCE_COMPLETE_NOT_PROMOTED"
    )
    target_capability_claimed: Literal[False] = False
    promotion_authorized: Literal[False] = False
    scientific_evaluation: Literal[False] = False


class SevenModelInventoryStatus(StrictModel):
    """Fast, non-executing inventory for the bounded reference implementations."""

    model_id: SevenModelId
    display_name: str
    model_kind: Literal["family_reference", "upscaler_reference"]
    implementation: str
    reference_parameters: int = Field(gt=0)
    trainable_parameters: int = Field(gt=0)
    status: Literal["ENGINEERING_REFERENCE_LOCAL_VERIFIED_NOT_PROMOTED"] = (
        "ENGINEERING_REFERENCE_LOCAL_VERIFIED_NOT_PROMOTED"
    )
    contract_execution_required: Literal[True] = True
    target_capability_claimed: Literal[False] = False
    promotion_authorized: Literal[False] = False
    scientific_evaluation: Literal[False] = False


def default_seven_model_configs() -> tuple[SevenModelConfig, ...]:
    """Return bounded configurations for the exact seven-model engineering lineup."""

    backbone = ReferenceTransformerConfig(
        vocab_size=32,
        model_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=24,
        max_sequence_length=8,
    )
    return (
        HermesReferenceConfig(backbone=backbone, tool_count=3),
        PrometheusReferenceConfig(
            backbone=backbone,
            domains=3,
            adapter_dim=8,
            evidence_classes=2,
        ),
        PerseusReferenceConfig(
            backbone=backbone,
            experts=3,
            active_experts=2,
            expert_dim=24,
        ),
        AtlasReferenceConfig(
            backbone=backbone,
            image_channels=3,
            patch_size=2,
            retrieval_dim=6,
        ),
        KronosReferenceConfig(
            state_dim=5,
            action_dim=3,
            hidden_dim=8,
            discrete_actions=4,
            planning_horizon=3,
        ),
        AionReferenceConfig(
            state_dim=5,
            hidden_dim=8,
            research_stages=4,
            stop_stage=3,
            kill_threshold=0.75,
        ),
        JEPAUpscalerConfig(
            scale=2,
            input_channels=3,
            feature_channels=8,
            residual_blocks=1,
            target_momentum=0.9,
            predictor_dilations=(1, 2),
        ),
    )


def seven_model_id(config: SevenModelConfig) -> SevenModelId:
    """Return the stable engineering identifier for one typed model config."""

    if isinstance(config, JEPAUpscalerConfig):
        return "jepa_upscaler"
    return config.family_id


def build_seven_model_reference(config: SevenModelConfig) -> nn.Module:
    """Build one bounded implementation without implying a trained family model."""

    if isinstance(config, JEPAUpscalerConfig):
        return JEPAUpscaler(config)
    return build_reference_model(config)


def inspect_seven_model_references() -> tuple[SevenModelInventoryStatus, ...]:
    """Describe the canonical references without running their contract probes."""

    display_names: dict[SevenModelId, str] = {
        "hermes": "Hermes",
        "prometheus": "Prometheus",
        "perseus": "Perseus",
        "atlas": "Atlas",
        "kronos": "Kronos",
        "aion": "Aion",
        "jepa_upscaler": "JEPA Upscaler",
    }
    statuses: list[SevenModelInventoryStatus] = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261018)
        for config in default_seven_model_configs():
            model_id = seven_model_id(config)
            model = build_seven_model_reference(config)
            statuses.append(
                SevenModelInventoryStatus(
                    model_id=model_id,
                    display_name=display_names[model_id],
                    model_kind=(
                        "upscaler_reference"
                        if model_id == "jepa_upscaler"
                        else "family_reference"
                    ),
                    implementation=type(model).__name__,
                    reference_parameters=sum(parameter.numel() for parameter in model.parameters()),
                    trainable_parameters=sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                )
            )
    if tuple(status.model_id for status in statuses) != SEVEN_MODEL_IDS:
        raise RuntimeError("seven-model inventory did not preserve the canonical model order")
    return tuple(statuses)


def audit_seven_model_contracts() -> tuple[SevenModelEngineeringStatus, ...]:
    """Execute deterministic artificial forward/backward probes for all seven models."""

    probe_by_id: dict[SevenModelId, Callable[[nn.Module], Tensor]] = {
        "hermes": _probe_hermes,
        "prometheus": _probe_prometheus,
        "perseus": _probe_perseus,
        "atlas": _probe_atlas,
        "kronos": _probe_kronos,
        "aion": _probe_aion,
        "jepa_upscaler": _probe_jepa,
    }
    display_names: dict[SevenModelId, str] = {
        "hermes": "Hermes",
        "prometheus": "Prometheus",
        "perseus": "Perseus",
        "atlas": "Atlas",
        "kronos": "Kronos",
        "aion": "Aion",
        "jepa_upscaler": "JEPA Upscaler",
    }
    statuses: list[SevenModelEngineeringStatus] = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261018)
        for config in default_seven_model_configs():
            model_id = seven_model_id(config)
            model = build_seven_model_reference(config).train()
            model.zero_grad(set_to_none=True)
            objective = probe_by_id[model_id](model)
            if objective.ndim != 0 or not bool(torch.isfinite(objective)):
                raise RuntimeError(f"{model_id} artificial probe returned a non-finite scalar")
            objective.backward()  # type: ignore[no-untyped-call]
            trainable_parameters = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            gradients = [parameter.grad for parameter in trainable_parameters]
            if any(gradient is None for gradient in gradients):
                raise RuntimeError(
                    f"{model_id} artificial probe did not reach every trainable parameter"
                )
            finite_gradients = [cast(Tensor, gradient) for gradient in gradients]
            if not all(bool(torch.isfinite(gradient).all()) for gradient in finite_gradients):
                raise RuntimeError(f"{model_id} artificial probe produced non-finite gradients")
            trainable = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            statuses.append(
                SevenModelEngineeringStatus(
                    model_id=model_id,
                    display_name=display_names[model_id],
                    model_kind=(
                        "upscaler_reference"
                        if model_id == "jepa_upscaler"
                        else "family_reference"
                    ),
                    implementation=type(model).__name__,
                    reference_parameters=sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    trainable_parameters=trainable,
                    trainable_parameter_tensors=len(trainable_parameters),
                    gradient_parameter_tensors=len(finite_gradients),
                )
            )
    if tuple(status.model_id for status in statuses) != SEVEN_MODEL_IDS:
        raise RuntimeError("seven-model audit did not preserve the canonical model order")
    return tuple(statuses)


def _tokens() -> tuple[Tensor, Tensor]:
    return (
        torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
        torch.tensor(
            [[True, True, True, False], [True, True, False, False]], dtype=torch.bool
        ),
    )


def _finite_tensors(model_id: SevenModelId, *tensors: Tensor) -> None:
    if not tensors or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise RuntimeError(f"{model_id} artificial probe produced non-finite output")


def _probe_hermes(module: nn.Module) -> Tensor:
    model = cast(HermesReferenceModel, module)
    tokens, mask = _tokens()
    output = model(tokens, mask)
    _finite_tensors(
        "hermes",
        output.token_logits,
        output.tool_logits,
        output.abstention_logit,
        output.hidden_states,
    )
    return cast(
        Tensor,
        output.token_logits.square().mean()
        + output.tool_logits.square().mean()
        + output.abstention_logit.square().mean(),
    )


def _probe_prometheus(module: nn.Module) -> Tensor:
    model = cast(PrometheusReferenceModel, module)
    tokens, mask = _tokens()
    output = model(tokens, mask)
    _finite_tensors(
        "prometheus",
        output.token_logits,
        output.evidence_logits,
        output.domain_weights,
        output.hidden_states,
    )
    return cast(
        Tensor, output.token_logits.square().mean() + output.evidence_logits.square().mean()
    )


def _probe_perseus(module: nn.Module) -> Tensor:
    model = cast(PerseusReferenceModel, module)
    tokens, mask = _tokens()
    output = model(tokens, mask)
    _finite_tensors(
        "perseus",
        output.token_logits,
        output.routing_probabilities,
        output.load_balance_loss,
        output.hidden_states,
    )
    return cast(Tensor, output.token_logits.square().mean() + 0.01 * output.load_balance_loss)


def _probe_atlas(module: nn.Module) -> Tensor:
    model = cast(AtlasReferenceModel, module)
    tokens, mask = _tokens()
    image = torch.linspace(0.0, 1.0, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    output = model(tokens, image, mask, timestamp=torch.tensor([0.0, 1.0]))
    _finite_tensors(
        "atlas",
        output.token_logits,
        output.retrieval_embedding,
        output.grounding_weights,
        output.hidden_states,
    )
    return cast(
        Tensor,
        output.token_logits.square().mean() + output.retrieval_embedding.square().mean(),
    )


def _probe_kronos(module: nn.Module) -> Tensor:
    model = cast(KronosReferenceModel, module)
    states = torch.linspace(0.0, 1.0, 2 * 4 * 5).reshape(2, 4, 5)
    actions = torch.linspace(-1.0, 1.0, 2 * 4 * 3).reshape(2, 4, 3)
    valid = torch.tensor(
        [[True, True, True, True], [True, True, False, False]], dtype=torch.bool
    )
    output = model(states, actions, valid)
    _finite_tensors(
        "kronos",
        output.next_state,
        output.reward,
        output.value,
        output.uncertainty,
        output.plan_logits,
        output.hidden_states,
    )
    return cast(
        Tensor,
        output.next_state.square().mean()
        + output.reward.square().mean()
        + output.value.square().mean()
        + output.uncertainty.square().mean()
        + output.plan_logits.square().mean(),
    )


def _probe_aion(module: nn.Module) -> Tensor:
    model = cast(AionReferenceModel, module)
    states = torch.linspace(-1.0, 1.0, 2 * 5).reshape(2, 5)
    allowed = torch.tensor(
        [[True, True, False, True], [False, True, True, True]], dtype=torch.bool
    )
    output = model(states, allowed, torch.tensor([0.2, 0.7]))
    _finite_tensors("aion", output.transition_probabilities)
    stage_weight = torch.arange(4, dtype=output.transition_probabilities.dtype)
    return cast(Tensor, (output.transition_probabilities * stage_weight).sum(dim=-1).mean())


def _probe_jepa(module: nn.Module) -> Tensor:
    model = cast(JEPAUpscaler, module)
    low = torch.linspace(0.0, 1.0, 2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    high = torch.nn.functional.interpolate(low, scale_factor=2, mode="bilinear")
    mask = torch.zeros((2, 4, 5), dtype=torch.bool)
    mask[:, :2, :2] = True
    output = model(low, high, timestamp=torch.tensor([0.0, 1.0]), context_mask=mask)
    if output.target_latent is None:
        raise RuntimeError("jepa_upscaler artificial probe did not produce a target latent")
    _finite_tensors(
        "jepa_upscaler",
        output.upscaled,
        output.residual,
        output.predicted_latent,
        output.target_latent,
        output.geometry,
    )
    return jepa_upscaler_loss(output, high).total
