"""Reliability-supervised successor to the frozen Olympus v2 fusion mechanism.

The module is additive: it leaves the v2 reference implementation intact and
reuses its family encoders.  Every routing control instantiates and executes
the same reliability heads so parameter count and forward compute remain
matched.  Reliability estimates are predictions from model state; protected
targets are never inputs to the forward pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn

from olympus.core.schemas import StrictModel
from olympus.models.olympus_system import (
    OlympusReferenceSystem,
    OlympusSystemConfig,
    OlympusSystemLoss,
    OlympusSystemOutput,
    olympus_system_loss,
)

ReliabilityRoutingMode = Literal[
    "reliability_shrinkage",
    "reliability_only",
    "semantic",
    "uniform",
]


class ReliabilityFusionConfig(StrictModel):
    """Frozen, bounded configuration for the v3 successor mechanism."""

    system: OlympusSystemConfig
    routing_mode: ReliabilityRoutingMode = "reliability_shrinkage"
    reliability_logit_strength: float = Field(default=1.0, ge=0.0, le=16.0)
    shrinkage_to_adaptive: float = Field(default=0.75, ge=0.0, le=1.0)
    probability_floor: float = Field(default=1e-6, gt=0.0, le=1e-2)

    @model_validator(mode="after")
    def valid_mechanism(self) -> ReliabilityFusionConfig:
        if self.routing_mode in {"reliability_shrinkage", "reliability_only"}:
            if self.reliability_logit_strength <= 0.0:
                raise ValueError("reliability-aware routing requires positive logit strength")
            if self.shrinkage_to_adaptive <= 0.0:
                raise ValueError("reliability-aware routing requires positive adaptive mass")
        return self


class ReliabilityLossConfig(StrictModel):
    """Joint v3 objective with explicit reliability calibration."""

    transition_weight: float = Field(default=0.5, ge=0.0)
    temporal_weight: float = Field(default=0.25, ge=0.0)
    uncertainty_weight: float = Field(default=0.1, ge=0.0)
    risk_weight: float = Field(default=0.1, ge=0.0)
    gate_balance_weight: float = Field(default=0.01, ge=0.0)
    counterfactual_weight: float = Field(default=0.05, ge=0.0)
    reliability_weight: float = Field(default=0.25, ge=0.0)

    @model_validator(mode="after")
    def finite_weights(self) -> ReliabilityLossConfig:
        if not all(math.isfinite(value) for value in self.model_dump().values()):
            raise ValueError("loss weights must be finite")
        return self

    def base_weights(self) -> dict[str, float]:
        values = self.model_dump()
        values.pop("reliability_weight")
        return values


@dataclass(slots=True)
class ReliabilityFusionOutput:
    system: OlympusSystemOutput
    semantic_gates: Tensor
    component_reliability_logits: Tensor
    component_reliability: Tensor


@dataclass(slots=True)
class ReliabilityFusionLoss:
    total: Tensor
    system: OlympusSystemLoss
    reliability_calibration: Tensor


class ReliabilityAwareOlympusSystem(nn.Module):
    """Route family states using calibrated reliability with uniform shrinkage."""

    def __init__(self, config: ReliabilityFusionConfig) -> None:
        super().__init__()
        self.config = config
        self.system = OlympusReferenceSystem(config.system)
        self.reliability_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(config.system.fusion_dim),
                    nn.Linear(config.system.fusion_dim, config.system.fusion_dim),
                    nn.GELU(),
                    nn.Linear(config.system.fusion_dim, 1),
                )
                for name in self.system.component_names
            }
        )

    @property
    def component_names(self) -> tuple[str, ...]:
        return self.system.component_names

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
    ) -> ReliabilityFusionOutput:
        base = self.system(
            tokens,
            image,
            states,
            actions,
            allowed_transitions,
            risk_score,
            attention_mask=attention_mask,
            domain_ids=domain_ids,
            temporal_mask=temporal_mask,
            timestamp=timestamp,
            extension_features=extension_features,
        )
        vectors = base.component_states
        reliability_logits = torch.cat(
            [
                self.reliability_heads[name](vectors[:, index])
                for index, name in enumerate(self.component_names)
            ],
            dim=-1,
        )
        reliability = reliability_logits.sigmoid()
        gates = self._route(base.family_gates, reliability, reliability_logits)
        system_output = self._recompute_system_output(
            base,
            gates,
            allowed_transitions,
            risk_score,
        )
        return ReliabilityFusionOutput(
            system=system_output,
            semantic_gates=base.family_gates,
            component_reliability_logits=reliability_logits,
            component_reliability=reliability,
        )

    def _route(
        self,
        semantic_gates: Tensor,
        reliability: Tensor,
        reliability_logits: Tensor,
    ) -> Tensor:
        uniform = torch.full_like(semantic_gates, 1.0 / semantic_gates.shape[-1])
        mode = self.config.routing_mode
        if mode == "uniform":
            # Preserve the full matched graph while disabling both learned signals.
            return uniform + 0.0 * (semantic_gates + reliability_logits)
        if mode == "semantic":
            return semantic_gates + 0.0 * reliability_logits

        floor = self.config.probability_floor
        reliability_term = (
            self.config.reliability_logit_strength * reliability.clamp_min(floor).log()
        )
        if mode == "reliability_only":
            adaptive = reliability_term.softmax(dim=-1)
        else:
            semantic_term = semantic_gates.clamp_min(floor).log()
            adaptive = (semantic_term + reliability_term).softmax(dim=-1)
        mass = self.config.shrinkage_to_adaptive
        return (1.0 - mass) * uniform + mass * adaptive

    def _recompute_system_output(
        self,
        base: OlympusSystemOutput,
        gates: Tensor,
        allowed_transitions: Tensor,
        risk_score: Tensor,
    ) -> OlympusSystemOutput:
        vectors = base.component_states
        weighted = vectors * gates.unsqueeze(-1)
        pre_normalized = weighted.sum(dim=1)
        fused = self.system.fusion_norm(pre_normalized)

        normalized_vectors = nn.functional.normalize(vectors, dim=-1)
        normalized_centroid = nn.functional.normalize(
            (normalized_vectors * gates.unsqueeze(-1)).sum(dim=1), dim=-1
        )
        cosine_to_centroid = (normalized_vectors * normalized_centroid.unsqueeze(1)).sum(dim=-1)
        disagreement = ((1.0 - cosine_to_centroid).clamp(0.0, 2.0) * gates).sum(dim=-1) / 2.0
        risk_disagreement = (
            disagreement if self.config.system.disagreement_risk else disagreement * 0.0
        )
        risk_features = torch.cat((fused, risk_disagreement.unsqueeze(-1)), dim=-1)
        intrinsic_risk = self.system.intrinsic_risk_head(risk_features).squeeze(-1).sigmoid()

        batch = fused.shape[0]
        host_risk = risk_score.to(device=fused.device, dtype=fused.dtype)
        if host_risk.shape != (batch,) or not bool(torch.isfinite(host_risk).all()):
            raise ValueError("risk score must contain one finite value per batch item")
        if bool(((host_risk < 0.0) | (host_risk > 1.0)).any()):
            raise ValueError("risk scores must fall in [0, 1]")
        effective_risk = 1.0 - (1.0 - host_risk) * (1.0 - intrinsic_risk)
        aion = self.system.aion(self.system.aion_state(fused), allowed_transitions, effective_risk)
        raw_decision = self.system.decision_head(fused)

        leave_one_out: list[Tensor] = []
        for index in range(vectors.shape[1]):
            remaining_mass = (1.0 - gates[:, index]).clamp_min(1e-6).unsqueeze(-1)
            counterfactual = self.system.fusion_norm(
                (pre_normalized - weighted[:, index]) / remaining_mass
            )
            leave_one_out.append(self.system.decision_head(counterfactual))
        counterfactual_logits = torch.stack(leave_one_out, dim=1)
        decision_available = ~aion.forced_stop
        decision = torch.where(
            decision_available.unsqueeze(-1), raw_decision, torch.zeros_like(raw_decision)
        )
        uncertainty = nn.functional.softplus(self.system.uncertainty_head(fused).squeeze(-1)) + 1e-6
        return replace(
            base,
            aion=aion,
            family_gates=gates,
            fused_state=fused,
            counterfactual_decision_logits=counterfactual_logits,
            raw_decision_logits=raw_decision,
            decision_logits=decision,
            decision_available=decision_available,
            uncertainty=uncertainty,
            family_disagreement=disagreement,
            intrinsic_risk=intrinsic_risk,
            effective_risk=effective_risk,
        )


def reliability_fusion_loss(
    output: ReliabilityFusionOutput,
    *,
    decision_targets: Tensor,
    transition_targets: Tensor,
    next_state_targets: Tensor,
    component_reliability_targets: Tensor,
    intrinsic_risk_targets: Tensor | None = None,
    temporal_mask: Tensor | None = None,
    config: ReliabilityLossConfig | None = None,
) -> ReliabilityFusionLoss:
    """Optimize task behavior and independently observable reliability targets."""

    weights = config or ReliabilityLossConfig()
    targets = component_reliability_targets.to(
        device=output.component_reliability.device,
        dtype=output.component_reliability.dtype,
    )
    if targets.shape != output.component_reliability.shape:
        raise ValueError("reliability targets must match [batch, component]")
    if not bool(torch.isfinite(targets).all()) or bool(((targets < 0.0) | (targets > 1.0)).any()):
        raise ValueError("reliability targets must be finite values in [0, 1]")
    system_loss = olympus_system_loss(
        output.system,
        decision_targets=decision_targets,
        transition_targets=transition_targets,
        next_state_targets=next_state_targets,
        intrinsic_risk_targets=intrinsic_risk_targets,
        temporal_mask=temporal_mask,
        **weights.base_weights(),
    )
    reliability_calibration = nn.functional.binary_cross_entropy_with_logits(
        output.component_reliability_logits,
        targets,
    )
    total = system_loss.total + weights.reliability_weight * reliability_calibration
    return ReliabilityFusionLoss(
        total=total,
        system=system_loss,
        reliability_calibration=reliability_calibration,
    )
