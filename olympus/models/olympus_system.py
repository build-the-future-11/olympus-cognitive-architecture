"""Trainable six-family composition for the Olympus reference architecture.

The composition fuses development-scale family representations.  It never
executes tools or grants authority: Aion receives a host-supplied legal
transition mask and risk score, and its deterministic STOP result suppresses
the downstream decision surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn

from olympus.core.schemas import StrictModel
from olympus.models.reference_families import (
    AionReferenceConfig,
    AionReferenceModel,
    AionReferenceOutput,
    AtlasReferenceConfig,
    AtlasReferenceModel,
    AtlasReferenceOutput,
    HermesReferenceConfig,
    HermesReferenceModel,
    HermesReferenceOutput,
    KronosReferenceConfig,
    KronosReferenceModel,
    KronosReferenceOutput,
    PerseusReferenceConfig,
    PerseusReferenceModel,
    PerseusReferenceOutput,
    PrometheusReferenceConfig,
    PrometheusReferenceModel,
    PrometheusReferenceOutput,
)

FAMILY_VECTOR_NAMES = ("hermes", "prometheus", "perseus", "atlas", "kronos")


class OlympusSystemConfig(StrictModel):
    """Bounded configuration for composing the six reference families."""

    hermes: HermesReferenceConfig = Field(default_factory=HermesReferenceConfig)
    architecture_revision: Literal["reference_v1", "flagship_v2"] = "reference_v1"
    prometheus: PrometheusReferenceConfig = Field(default_factory=PrometheusReferenceConfig)
    perseus: PerseusReferenceConfig = Field(default_factory=PerseusReferenceConfig)
    atlas: AtlasReferenceConfig = Field(default_factory=AtlasReferenceConfig)
    kronos: KronosReferenceConfig = Field(default_factory=KronosReferenceConfig)
    aion: AionReferenceConfig = Field(default_factory=AionReferenceConfig)
    fusion_dim: int = Field(default=32, ge=8, le=8_192)
    decision_classes: int = Field(default=8, ge=2, le=65_536)
    gate_temperature: float = Field(default=1.0, ge=0.05, le=10.0)
    fusion_strategy: Literal["learned", "uniform"] = "learned"
    disagreement_risk: bool = True
    extension_dims: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_extensions(self) -> OlympusSystemConfig:
        if len(self.extension_dims) > 16:
            raise ValueError("at most 16 Olympus extensions are supported")
        reserved = set(FAMILY_VECTOR_NAMES) | {"aion"}
        for name, width in self.extension_dims.items():
            if not name or not name.replace("_", "").isalnum() or name in reserved:
                raise ValueError(f"invalid or reserved extension name: {name!r}")
            if width < 1 or width > 8_192:
                raise ValueError("extension dimensions must fall in [1, 8192]")
        return self


class OlympusSystemLossConfig(StrictModel):
    """Validated weights for the joint Olympus coordination objective."""

    transition_weight: float = Field(default=0.5, ge=0.0)
    temporal_weight: float = Field(default=0.25, ge=0.0)
    uncertainty_weight: float = Field(default=0.1, ge=0.0)
    risk_weight: float = Field(default=0.1, ge=0.0)
    gate_balance_weight: float = Field(default=0.01, ge=0.0)
    counterfactual_weight: float = Field(default=0.05, ge=0.0)

    @model_validator(mode="after")
    def finite_weights(self) -> OlympusSystemLossConfig:
        if not all(math.isfinite(value) for value in self.model_dump().values()):
            raise ValueError("loss weights must be finite")
        return self


@dataclass(slots=True)
class OlympusSystemOutput:
    hermes: HermesReferenceOutput
    prometheus: PrometheusReferenceOutput
    perseus: PerseusReferenceOutput
    atlas: AtlasReferenceOutput
    kronos: KronosReferenceOutput
    aion: AionReferenceOutput
    component_names: tuple[str, ...]
    component_states: Tensor
    family_gates: Tensor
    fused_state: Tensor
    counterfactual_decision_logits: Tensor
    raw_decision_logits: Tensor
    decision_logits: Tensor
    decision_available: Tensor
    uncertainty: Tensor
    family_disagreement: Tensor
    intrinsic_risk: Tensor
    effective_risk: Tensor


@dataclass(slots=True)
class OlympusSystemLoss:
    total: Tensor
    decision: Tensor
    transition: Tensor
    temporal: Tensor
    uncertainty_calibration: Tensor
    risk_calibration: Tensor
    gate_balance: Tensor
    counterfactual_consistency: Tensor


@dataclass(slots=True)
class OlympusObjectiveOutput:
    """Only the sufficient predictions needed for global objectives and diagnostics.

    Evaluation can collect these on CPU without retaining vocabulary logits,
    backbone activations or per-family hidden states for the full dataset.
    """

    aion: AionReferenceOutput
    next_state: Tensor
    component_names: tuple[str, ...]
    raw_decision_logits: Tensor
    decision_available: Tensor
    uncertainty: Tensor
    family_gates: Tensor
    counterfactual_decision_logits: Tensor
    intrinsic_risk: Tensor
    effective_risk: Tensor


class OlympusReferenceSystem(nn.Module):
    """Fuse all six references while leaving execution authority outside the model."""

    hermes_type: type[HermesReferenceModel] = HermesReferenceModel
    prometheus_type: type[PrometheusReferenceModel] = PrometheusReferenceModel
    perseus_type: type[PerseusReferenceModel] = PerseusReferenceModel
    atlas_type: type[AtlasReferenceModel] = AtlasReferenceModel
    kronos_type: type[KronosReferenceModel] = KronosReferenceModel
    aion_type: type[AionReferenceModel] = AionReferenceModel

    def __init__(self, config: OlympusSystemConfig) -> None:
        super().__init__()
        self.config = config
        self.hermes = self.hermes_type(config.hermes)
        self.prometheus = self.prometheus_type(config.prometheus)
        self.perseus = self.perseus_type(config.perseus)
        self.atlas = self.atlas_type(config.atlas)
        self.kronos = self.kronos_type(config.kronos)
        self.aion = self.aion_type(config.aion)
        self.family_projections = nn.ModuleDict(
            {
                "hermes": _projection(config.hermes.backbone.model_dim, config.fusion_dim),
                "prometheus": _projection(config.prometheus.backbone.model_dim, config.fusion_dim),
                "perseus": _projection(config.perseus.backbone.model_dim, config.fusion_dim),
                "atlas": _projection(config.atlas.retrieval_dim, config.fusion_dim),
                "kronos": _projection(config.kronos.hidden_dim, config.fusion_dim),
            }
        )
        self.extension_names = tuple(sorted(config.extension_dims))
        self.extension_projections = nn.ModuleDict(
            {
                name: _projection(config.extension_dims[name], config.fusion_dim)
                for name in self.extension_names
            }
        )
        self.component_names = FAMILY_VECTOR_NAMES + self.extension_names
        family_count = len(self.component_names)
        self.gate_router = nn.Sequential(
            nn.LayerNorm(config.fusion_dim * family_count),
            nn.Linear(config.fusion_dim * family_count, config.fusion_dim),
            nn.GELU(),
            nn.Linear(config.fusion_dim, family_count),
        )
        self.fusion_norm = nn.LayerNorm(config.fusion_dim)
        self.aion_state = nn.Linear(config.fusion_dim, config.aion.state_dim)
        self.decision_head = nn.Linear(config.fusion_dim, config.decision_classes)
        self.uncertainty_head = nn.Linear(config.fusion_dim, 1)
        self.intrinsic_risk_head = nn.Sequential(
            nn.LayerNorm(config.fusion_dim + 1),
            nn.Linear(config.fusion_dim + 1, config.fusion_dim),
            nn.GELU(),
            nn.Linear(config.fusion_dim, 1),
        )
        final_risk_layer = self.intrinsic_risk_head[-1]
        assert isinstance(final_risk_layer, nn.Linear)
        nn.init.constant_(final_risk_layer.bias, -3.0)

    def forward(
        self,
        tokens: Tensor,
        image: Tensor,
        states: Tensor,
        actions: Tensor,
        allowed_transitions: Tensor,
        risk_score: Tensor,
        *,
        attention_mask: Tensor | None = None,
        domain_ids: Tensor | None = None,
        temporal_mask: Tensor | None = None,
        timestamp: Tensor | None = None,
        extension_features: dict[str, Tensor] | None = None,
    ) -> OlympusSystemOutput:
        batch = tokens.shape[0]
        if image.shape[0] != batch or states.shape[0] != batch or actions.shape[0] != batch:
            raise ValueError("all Olympus family inputs must have the same batch size")
        hermes = self.hermes(tokens, attention_mask)
        prometheus = self.prometheus(tokens, attention_mask, domain_ids=domain_ids)
        perseus = self.perseus(tokens, attention_mask)
        atlas = self.atlas(tokens, image, attention_mask, timestamp=timestamp)
        kronos = self.kronos(states, actions, temporal_mask)

        return self._compose_outputs(
            tokens,
            states,
            allowed_transitions,
            risk_score,
            hermes,
            prometheus,
            perseus,
            atlas,
            kronos,
            attention_mask=attention_mask,
            temporal_mask=temporal_mask,
            extension_features=extension_features,
        )

    def _compose_outputs(
        self,
        tokens: Tensor,
        states: Tensor,
        allowed_transitions: Tensor,
        risk_score: Tensor,
        hermes: HermesReferenceOutput,
        prometheus: PrometheusReferenceOutput,
        perseus: PerseusReferenceOutput,
        atlas: AtlasReferenceOutput,
        kronos: KronosReferenceOutput,
        *,
        attention_mask: Tensor | None,
        temporal_mask: Tensor | None,
        extension_features: dict[str, Tensor] | None,
    ) -> OlympusSystemOutput:
        batch = tokens.shape[0]

        text_indices = _last_active_indices(tokens, attention_mask)
        batch_indices = torch.arange(batch, device=tokens.device)
        temporal_indices = _last_active_indices(states[..., 0], temporal_mask)
        projected = [
            self.family_projections["hermes"](hermes.hidden_states[batch_indices, text_indices]),
            self.family_projections["prometheus"](
                prometheus.hidden_states[batch_indices, text_indices]
            ),
            self.family_projections["perseus"](perseus.hidden_states[batch_indices, text_indices]),
            self.family_projections["atlas"](atlas.retrieval_embedding),
            self.family_projections["kronos"](
                kronos.hidden_states[batch_indices, temporal_indices]
            ),
        ]
        supplied_extensions = extension_features or {}
        if set(supplied_extensions) != set(self.extension_names):
            missing = sorted(set(self.extension_names) - set(supplied_extensions))
            unexpected = sorted(set(supplied_extensions) - set(self.extension_names))
            raise ValueError(
                f"extension features do not match configuration; missing={missing}, "
                f"unexpected={unexpected}"
            )
        for name in self.extension_names:
            features = supplied_extensions[name]
            expected = (batch, self.config.extension_dims[name])
            if features.shape != expected:
                raise ValueError(f"extension {name!r} must have shape {expected}")
            if not bool(torch.isfinite(features).all()):
                raise ValueError(f"extension {name!r} contains non-finite values")
            projected.append(
                self.extension_projections[name](
                    features.to(device=tokens.device, dtype=projected[0].dtype)
                )
            )
        vectors = torch.stack(projected, dim=1)
        gate_logits = self.gate_router(vectors.flatten(1)) / self.config.gate_temperature
        learned_gates = gate_logits.softmax(dim=-1)
        if self.config.fusion_strategy == "uniform":
            # The router is deliberately executed above and its result is retained in
            # the graph with a zero multiplier. This keeps the control parameter- and
            # forward-compute-matched without allowing it to affect the fused state.
            family_gates = learned_gates * 0.0 + torch.full_like(
                learned_gates, 1.0 / learned_gates.shape[1]
            )
        else:
            family_gates = learned_gates
        weighted = vectors * family_gates.unsqueeze(-1)
        pre_normalized_fused = weighted.sum(dim=1)
        fused = self.fusion_norm(pre_normalized_fused)

        normalized_vectors = nn.functional.normalize(vectors, dim=-1)
        normalized_centroid = nn.functional.normalize(
            (normalized_vectors * family_gates.unsqueeze(-1)).sum(dim=1), dim=-1
        )
        cosine_to_centroid = (normalized_vectors * normalized_centroid.unsqueeze(1)).sum(dim=-1)
        family_disagreement = ((1.0 - cosine_to_centroid).clamp(0.0, 2.0) * family_gates).sum(
            dim=-1
        ) / 2.0
        risk_disagreement = (
            family_disagreement if self.config.disagreement_risk else family_disagreement * 0.0
        )
        risk_features = torch.cat((fused, risk_disagreement.unsqueeze(-1)), dim=-1)
        intrinsic_risk = self.intrinsic_risk_head(risk_features).squeeze(-1).sigmoid()
        host_risk = risk_score.to(device=fused.device, dtype=fused.dtype)
        if host_risk.shape != (batch,) or not bool(torch.isfinite(host_risk).all()):
            raise ValueError("risk score must contain one finite value per batch item")
        if bool(((host_risk < 0.0) | (host_risk > 1.0)).any()):
            raise ValueError("risk scores must fall in [0, 1]")
        effective_risk = 1.0 - (1.0 - host_risk) * (1.0 - intrinsic_risk)
        aion = self.aion(self.aion_state(fused), allowed_transitions, effective_risk)
        raw_decision = self.decision_head(fused)
        leave_one_out: list[Tensor] = []
        for index in range(vectors.shape[1]):
            remaining_mass = (1.0 - family_gates[:, index]).clamp_min(1e-6).unsqueeze(-1)
            counterfactual = self.fusion_norm(
                (pre_normalized_fused - weighted[:, index]) / remaining_mass
            )
            leave_one_out.append(self.decision_head(counterfactual))
        counterfactual_logits = torch.stack(leave_one_out, dim=1)
        decision_available = ~aion.forced_stop
        decision = torch.where(
            decision_available.unsqueeze(-1), raw_decision, torch.zeros_like(raw_decision)
        )
        uncertainty = nn.functional.softplus(self.uncertainty_head(fused).squeeze(-1)) + 1e-6
        return OlympusSystemOutput(
            hermes=hermes,
            prometheus=prometheus,
            perseus=perseus,
            atlas=atlas,
            kronos=kronos,
            aion=aion,
            component_names=self.component_names,
            component_states=vectors,
            family_gates=family_gates,
            fused_state=fused,
            counterfactual_decision_logits=counterfactual_logits,
            raw_decision_logits=raw_decision,
            decision_logits=decision,
            decision_available=decision_available,
            uncertainty=uncertainty,
            family_disagreement=family_disagreement,
            intrinsic_risk=intrinsic_risk,
            effective_risk=effective_risk,
        )


def olympus_system_loss(
    output: OlympusSystemOutput | OlympusObjectiveOutput,
    *,
    decision_targets: Tensor,
    transition_targets: Tensor,
    next_state_targets: Tensor,
    intrinsic_risk_targets: Tensor | None = None,
    temporal_mask: Tensor | None = None,
    transition_weight: float = 0.5,
    temporal_weight: float = 0.25,
    uncertainty_weight: float = 0.1,
    risk_weight: float = 0.1,
    gate_balance_weight: float = 0.01,
    counterfactual_weight: float = 0.05,
) -> OlympusSystemLoss:
    """Train coordination without treating a model decision as execution authority."""

    batch = output.raw_decision_logits.shape[0]
    next_state = (
        output.next_state
        if isinstance(output, OlympusObjectiveOutput)
        else output.kronos.next_state
    )
    _validate_loss_weights(
        transition_weight=transition_weight,
        temporal_weight=temporal_weight,
        uncertainty_weight=uncertainty_weight,
        risk_weight=risk_weight,
        gate_balance_weight=gate_balance_weight,
        counterfactual_weight=counterfactual_weight,
    )
    if decision_targets.shape != (batch,):
        raise ValueError("decision targets must have shape [batch]")
    if transition_targets.shape != (batch,):
        raise ValueError("transition targets must have shape [batch]")
    if next_state_targets.shape != next_state.shape:
        raise ValueError("next-state targets must match the Kronos prediction shape")
    if not bool(torch.isfinite(next_state_targets).all()):
        raise ValueError("next-state targets must be finite")
    for name, labels in (
        ("decision", decision_targets),
        ("transition", transition_targets),
    ):
        if labels.dtype not in {torch.int32, torch.int64}:
            raise ValueError(f"{name} targets must use an integer dtype")
    next_state_targets = next_state_targets.to(device=next_state.device, dtype=next_state.dtype)
    decision_labels = decision_targets.to(
        device=output.raw_decision_logits.device, dtype=torch.long
    )
    transition_labels = transition_targets.to(
        device=output.aion.transition_logits.device, dtype=torch.long
    )
    invalid_decisions = (decision_labels < 0) | (
        decision_labels >= output.raw_decision_logits.shape[1]
    )
    if bool(invalid_decisions.any()):
        raise ValueError("decision targets fall outside the configured class range")
    invalid_transitions = (transition_labels < 0) | (
        transition_labels >= output.aion.transition_logits.shape[1]
    )
    if bool(invalid_transitions.any()):
        raise ValueError("transition targets fall outside the configured stage range")
    target_transition_logits = output.aion.transition_logits.gather(
        1, transition_labels.unsqueeze(1)
    )
    if not bool(torch.isfinite(target_transition_logits).all()):
        raise ValueError("transition target is forbidden by the host-supplied policy")

    per_example_decision = nn.functional.cross_entropy(
        output.raw_decision_logits,
        decision_labels,
        reduction="none",
    )
    active = output.decision_available
    if bool(active.any()):
        decision = per_example_decision[active].mean()
        uncertainty = output.uncertainty[active]
        uncertainty_calibration = (
            per_example_decision[active] / uncertainty + uncertainty.log()
        ).mean()
    else:
        decision = output.raw_decision_logits.sum() * 0.0
        uncertainty_calibration = output.uncertainty.sum() * 0.0
    if intrinsic_risk_targets is None:
        risk_calibration = output.intrinsic_risk.sum() * 0.0
    else:
        if intrinsic_risk_targets.shape != (batch,):
            raise ValueError("intrinsic-risk targets must have shape [batch]")
        risk_targets = intrinsic_risk_targets.to(
            device=output.intrinsic_risk.device, dtype=output.intrinsic_risk.dtype
        )
        if not bool(torch.isfinite(risk_targets).all()) or bool(
            ((risk_targets < 0.0) | (risk_targets > 1.0)).any()
        ):
            raise ValueError("intrinsic-risk targets must be finite values in [0, 1]")
        risk_calibration = nn.functional.binary_cross_entropy(output.intrinsic_risk, risk_targets)
    transition = nn.functional.cross_entropy(
        output.aion.transition_logits,
        transition_labels,
    )
    temporal_error = (next_state - next_state_targets).square().mean(dim=-1)
    if temporal_mask is None:
        temporal = temporal_error.mean()
    else:
        if temporal_mask.shape != temporal_error.shape:
            raise ValueError("temporal mask must match the Kronos batch and time dimensions")
        mask = temporal_mask.to(device=temporal_error.device, dtype=torch.bool)
        if not bool(mask.any()):
            raise ValueError("temporal mask must contain at least one active step")
        temporal = temporal_error[mask].mean()
    mean_gate = output.family_gates.mean(dim=0).clamp_min(
        torch.finfo(output.family_gates.dtype).tiny
    )
    uniform = torch.full_like(mean_gate, 1.0 / mean_gate.numel())
    gate_balance = (mean_gate * (mean_gate / uniform).log()).sum()
    full_probabilities = output.raw_decision_logits.softmax(dim=-1).detach()
    counterfactual_log_probabilities = output.counterfactual_decision_logits.log_softmax(dim=-1)
    per_component_counterfactual = nn.functional.kl_div(
        counterfactual_log_probabilities,
        full_probabilities.unsqueeze(1).expand_as(counterfactual_log_probabilities),
        reduction="none",
    ).sum(dim=-1)
    per_example_counterfactual = (per_component_counterfactual * output.family_gates.detach()).sum(
        dim=-1
    )
    if bool(active.any()):
        counterfactual_consistency = per_example_counterfactual[active].mean()
    else:
        counterfactual_consistency = output.counterfactual_decision_logits.sum() * 0.0
    total = (
        decision
        + transition_weight * transition
        + temporal_weight * temporal
        + uncertainty_weight * uncertainty_calibration
        + risk_weight * risk_calibration
        + gate_balance_weight * gate_balance
        + counterfactual_weight * counterfactual_consistency
    )
    return OlympusSystemLoss(
        total=total,
        decision=decision,
        transition=transition,
        temporal=temporal,
        uncertainty_calibration=uncertainty_calibration,
        risk_calibration=risk_calibration,
        gate_balance=gate_balance,
        counterfactual_consistency=counterfactual_consistency,
    )


def _last_active_indices(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return torch.full(
            (values.shape[0],), values.shape[1] - 1, dtype=torch.long, device=values.device
        )
    if mask.shape != values.shape[:2]:
        raise ValueError("activity mask must match the batch and sequence dimensions")
    active = mask.to(device=values.device, dtype=torch.bool)
    if not bool(active.any(dim=1).all()):
        raise ValueError("every sequence must contain at least one active position")
    indices = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
    return torch.where(active, indices, -1).max(dim=1).values


def _projection(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim))


def _validate_loss_weights(**weights: float) -> None:
    for name, value in weights.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")


def build_olympus_system(config: OlympusSystemConfig) -> OlympusReferenceSystem:
    """Construct the explicitly selected architecture revision."""
    if config.architecture_revision == "flagship_v2":
        from olympus.models.flagship import OlympusFlagshipSystem

        return OlympusFlagshipSystem(config)
    return OlympusReferenceSystem(config)
