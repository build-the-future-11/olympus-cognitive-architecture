"""Versioned six-family engineering features; target-scale efficacy is unmeasured."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from olympus.models.olympus_system import (
    OlympusReferenceSystem,
    OlympusSystemConfig,
    OlympusSystemOutput,
    _last_active_indices,
)
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
    ReferenceTransformerConfig,
    _patch_coordinates,
)


def default_flagship_config() -> OlympusSystemConfig:
    """Tiny artificial development configuration, independent of flagship targets."""
    backbone = ReferenceTransformerConfig(
        vocab_size=16,
        model_dim=8,
        layers=1,
        heads=2,
        feedforward_dim=16,
        max_sequence_length=8,
    )
    return OlympusSystemConfig(
        architecture_revision="flagship_v2",
        hermes=HermesReferenceConfig(backbone=backbone, tool_count=3),
        prometheus=PrometheusReferenceConfig(backbone=backbone, domains=2, adapter_dim=4),
        perseus=PerseusReferenceConfig(
            backbone=backbone, experts=3, active_experts=2, expert_dim=8
        ),
        atlas=AtlasReferenceConfig(backbone=backbone, patch_size=2, retrieval_dim=4),
        kronos=KronosReferenceConfig(
            state_dim=3,
            action_dim=2,
            hidden_dim=4,
            discrete_actions=3,
            planning_horizon=3,
        ),
        aion=AionReferenceConfig(state_dim=4, hidden_dim=4, research_stages=3, stop_stage=2),
        fusion_dim=8,
        decision_classes=3,
    )


def _mask(value: Tensor, shape: tuple[int, ...], device: torch.device, name: str) -> Tensor:
    if value.shape != shape or value.dtype != torch.bool or value.device != device:
        raise ValueError(f"{name} must be a Boolean tensor with shape {shape} on {device}")
    return value


def _continuous(value: Tensor, reference: Tensor, name: str) -> None:
    if (
        not value.is_floating_point()
        or value.device != reference.device
        or value.dtype != reference.dtype
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{name} must be finite and match the reference dtype and device")


def _active(tokens: Tensor, mask: Tensor | None) -> Tensor:
    return torch.ones_like(tokens, dtype=torch.bool) if mask is None else mask


@dataclass(slots=True)
class HermesFlagshipOutput(HermesReferenceOutput):
    tool_probabilities: Tensor
    tool_disagreement: Tensor
    tool_available: Tensor


class HermesFlagshipModel(HermesReferenceModel):
    """Two token views vote on tools; disagreement can only raise abstention."""

    def __init__(self, config: HermesReferenceConfig) -> None:
        super().__init__(config)
        self.context_tool_head = nn.Linear(config.backbone.model_dim, config.tool_count)

    def forward(
        self,
        tokens: Tensor,
        attention_mask: Tensor | None = None,
        *,
        allowed_tools: Tensor | None = None,
    ) -> HermesFlagshipOutput:
        base = super().forward(tokens, attention_mask)
        active = _active(tokens, attention_mask)
        mean = base.hidden_states.sum(1) / active.sum(1, keepdim=True)
        first = base.tool_logits.softmax(-1)
        second = self.context_tool_head(mean).softmax(-1)
        # Total variation is bounded and has no arbitrary log-epsilon calibration.
        disagreement = (first - second).abs().sum(-1) / 2
        allowed = (
            torch.ones_like(first, dtype=torch.bool)
            if allowed_tools is None
            else _mask(allowed_tools, tuple(first.shape), first.device, "allowed_tools")
        )
        probabilities = ((first + second) / 2).masked_fill(~allowed, 0)
        available = allowed.any(-1)
        probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(
            torch.finfo(first.dtype).tiny
        )
        abstention = base.abstention_logit + disagreement
        # All denied tools make abstention certain without infinite training logits.
        abstention = torch.where(
            available, abstention, torch.maximum(abstention, torch.full_like(abstention, 30))
        )
        return HermesFlagshipOutput(
            base.token_logits,
            base.tool_logits.masked_fill(~allowed, -torch.inf),
            abstention,
            base.hidden_states,
            probabilities,
            disagreement,
            available,
        )


@dataclass(slots=True)
class PrometheusFlagshipOutput(PrometheusReferenceOutput):
    token_domain_weights: Tensor
    adapter_disagreement: Tensor


class PrometheusFlagshipModel(PrometheusReferenceModel):
    """Causal domain routing with a disagreement-dependent residual norm budget."""

    def forward(
        self,
        tokens: Tensor,
        attention_mask: Tensor | None = None,
        *,
        domain_ids: Tensor | None = None,
    ) -> PrometheusFlagshipOutput:
        hidden = self.backbone(tokens, attention_mask)
        active = _active(tokens, attention_mask)
        if domain_ids is None:
            weights = self.domain_router(hidden).softmax(-1)
        else:
            if (
                domain_ids.shape != (tokens.shape[0],)
                or domain_ids.dtype not in {torch.int32, torch.int64}
                or domain_ids.device != tokens.device
                or bool(((domain_ids < 0) | (domain_ids >= self.config.domains)).any())
            ):
                raise ValueError("domain_ids must be in-range integer batch identifiers")
            weights = nn.functional.one_hot(domain_ids.long(), self.config.domains).to(hidden)
            weights = weights[:, None].expand(-1, tokens.shape[1], -1)
        adapters = torch.stack([adapter(hidden) for adapter in self.adapters], dim=2)
        residual = (adapters * weights[..., None]).sum(2)
        disagreement = ((adapters - residual.unsqueeze(2)).square().mean(-1) * weights).sum(-1)
        # Per-token, differentiable trust radius; no future-token pooling.
        norm = residual.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        radius = (1 + disagreement).reciprocal().unsqueeze(-1)
        adapted = (hidden + residual * (radius / norm).clamp(max=1)).masked_fill(
            ~active[..., None], 0
        )
        indices = _last_active_indices(tokens, attention_mask)
        batch = torch.arange(tokens.shape[0], device=tokens.device)
        return PrometheusFlagshipOutput(
            self.language_head(adapted),
            self.evidence_head(adapted),
            weights[batch, indices],
            adapted,
            weights.masked_fill(~active[..., None], 0),
            disagreement.masked_fill(~active, 0),
        )


class PerseusFlagshipModel(PerseusReferenceModel):
    """Top-k execution only among resident experts, excluding every padded token."""

    def forward(
        self,
        tokens: Tensor,
        attention_mask: Tensor | None = None,
        *,
        resident_experts: Tensor | None = None,
    ) -> PerseusReferenceOutput:
        hidden = self.backbone(tokens, attention_mask)
        active = _active(tokens, attention_mask)
        allowed = torch.ones(self.config.experts, dtype=torch.bool, device=tokens.device)
        if resident_experts is not None:
            allowed = _mask(
                resident_experts, (self.config.experts,), tokens.device, "resident_experts"
            )
        if int(allowed.sum()) < self.config.active_experts:
            raise ValueError("resident expert set cannot satisfy the configured top-k")
        logits = self.router(hidden).masked_fill(~allowed, -torch.inf)
        routing = logits.softmax(-1)
        values, indices = routing.topk(self.config.active_experts, dim=-1)
        weights = values / values.sum(-1, keepdim=True)
        flat_hidden, flat_active = hidden.reshape(-1, hidden.shape[-1]), active.flatten()
        flat_indices = indices.reshape(-1, self.config.active_experts)
        flat_weights = weights.reshape_as(flat_indices)
        mixed = torch.zeros_like(flat_hidden)
        for index, expert in enumerate(self.experts):
            rows, slots = torch.where((flat_indices == index) & flat_active[:, None])
            if rows.numel():
                mixed.index_add_(
                    0, rows, expert(flat_hidden[rows]) * flat_weights[rows, slots, None]
                )
        result = (hidden + mixed.reshape_as(hidden)).masked_fill(~active[..., None], 0)
        assignments = nn.functional.one_hot(indices, self.config.experts).to(hidden)
        fractions = (assignments * active[..., None, None]).sum((0, 1, 2))
        fractions = fractions / (active.sum() * self.config.active_experts)
        probabilities = (routing * active[..., None]).sum((0, 1)) / active.sum()
        balance = allowed.sum() * (fractions * probabilities).sum()
        return PerseusReferenceOutput(
            self.language_head(result),
            routing.masked_fill(~active[..., None], 0),
            indices.masked_fill(~active[..., None], -1),
            balance,
            result,
        )


class AtlasFlagshipModel(AtlasReferenceModel):
    """Patch evidence firewall: excluded patches never enter attention or retrieval."""

    def forward(
        self,
        tokens: Tensor,
        image: Tensor,
        attention_mask: Tensor | None = None,
        *,
        timestamp: Tensor | None = None,
        patch_mask: Tensor | None = None,
    ) -> AtlasReferenceOutput:
        _continuous(image, self.patch_embedding.weight, "image")
        patch = self.config.patch_size
        if (
            image.ndim != 4
            or image.shape[:2] != (tokens.shape[0], self.config.image_channels)
            or min(image.shape[-2:]) < patch
            or image.shape[-2] % patch
            or image.shape[-1] % patch
            or bool(((image < 0) | (image > 1)).any())
        ):
            raise ValueError(
                "image must be normalized and exactly divisible into configured patches"
            )
        text = self.text_backbone(tokens, attention_mask)
        patches_2d = self.patch_embedding(image)
        height, width = patches_2d.shape[-2:]
        patches = patches_2d.flatten(2).transpose(1, 2)
        allowed = torch.ones(patches.shape[:2], dtype=torch.bool, device=patches.device)
        if patch_mask is not None:
            allowed = _mask(patch_mask, tuple(allowed.shape), patches.device, "patch_mask")
        if not bool(allowed.any(-1).all()):
            raise ValueError("every image needs at least one admitted evidence patch")
        coordinates = _patch_coordinates(
            height,
            width,
            dtype=patches.dtype,
            device=patches.device,
            batch=patches.shape[0],
            timestamp=timestamp,
        )
        patches = (patches + self.coordinate_embedding(coordinates)).masked_fill(
            ~allowed[..., None], 0
        )
        grounded, weights = self.cross_attention(
            text,
            patches,
            patches,
            key_padding_mask=~allowed,
            need_weights=True,
        )
        active = _active(tokens, attention_mask)
        fused = self.fusion_norm(text + grounded).masked_fill(~active[..., None], 0)
        pooled = fused.sum(1) / active.sum(1, keepdim=True)
        return AtlasReferenceOutput(
            self.language_head(fused),
            nn.functional.normalize(self.retrieval_head(pooled), dim=-1),
            weights.masked_fill(~active[..., None], 0),
            fused,
        )


@dataclass(slots=True)
class KronosFlagshipOutput(KronosReferenceOutput):
    plan_states: Tensor
    cumulative_plan_uncertainty: Tensor


class KronosFlagshipModel(KronosReferenceModel):
    """Residual dynamics roll forward the predicted state at every planning step."""

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        valid_mask: Tensor | None = None,
    ) -> KronosFlagshipOutput:
        base = super().forward(states, actions, valid_mask)
        indices = _last_active_indices(states[..., 0], valid_mask)
        batch = torch.arange(states.shape[0], device=states.device)
        cursor, state = base.hidden_states[batch, indices], states[batch, indices]
        plans, trajectory, uncertainties = [], [], []
        accumulated = torch.zeros_like(state)
        for _ in range(self.config.planning_horizon):
            logits = self.plan_head(cursor)
            plans.append(logits)
            action = logits.softmax(-1) @ self.action_dictionary
            cursor = self.plan_cell(torch.cat((state, action), -1), cursor)
            state = state + self.next_state_head(cursor)
            accumulated = accumulated + nn.functional.softplus(self.uncertainty_head(cursor))
            trajectory.append(state)
            uncertainties.append(accumulated)
        active = torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)
        if valid_mask is not None:
            active = valid_mask
        return KronosFlagshipOutput(
            (states + base.next_state).masked_fill(~active[..., None], 0),
            base.reward,
            base.value,
            base.uncertainty,
            torch.stack(plans, 1),
            base.hidden_states,
            torch.stack(trajectory, 1),
            torch.stack(uncertainties, 1),
        )


class AionFlagshipModel(AionReferenceModel):
    """Host authority intersected with exact per-transition affordability."""

    def forward(
        self,
        state_features: Tensor,
        allowed_transitions: Tensor,
        risk_score: Tensor,
        *,
        transition_costs: Tensor | None = None,
        remaining_budget: Tensor | None = None,
    ) -> AionReferenceOutput:
        if (transition_costs is None) != (remaining_budget is None):
            raise ValueError("transition costs and remaining budget must be supplied together")
        allowed = _mask(
            allowed_transitions,
            (state_features.shape[0], self.config.research_stages),
            state_features.device,
            "allowed_transitions",
        )
        risk = risk_score
        if transition_costs is not None and remaining_budget is not None:
            _continuous(transition_costs, state_features, "transition_costs")
            _continuous(remaining_budget, state_features, "remaining_budget")
            if (
                transition_costs.shape != allowed.shape
                or remaining_budget.shape != (state_features.shape[0],)
                or bool((transition_costs < 0).any())
                or bool((remaining_budget < 0).any())
                or bool((transition_costs[:, self.config.stop_stage] != 0).any())
            ):
                raise ValueError("costs and budgets must be nonnegative, aligned; STOP costs zero")
            allowed = allowed & (transition_costs <= remaining_budget[:, None])
            operational = allowed.clone()
            operational[:, self.config.stop_stage] = False
            exhausted = ~operational.any(-1)
            _continuous(risk_score, state_features, "risk_score")
            if risk_score.shape != exhausted.shape or bool(
                ((risk_score < 0) | (risk_score > 1)).any()
            ):
                raise ValueError("risk_score must contain a [0, 1] value per example")
            risk = torch.where(exhausted, torch.ones_like(risk_score), risk_score)
        return super().forward(state_features, allowed, risk)


@dataclass(slots=True)
class FlagshipSystemOutput(OlympusSystemOutput):
    evidence_risk_terms: Tensor


class OlympusFlagshipSystem(OlympusReferenceSystem):
    """V2 composition exposes a monotone risk contribution from each specialist."""

    hermes_type = HermesFlagshipModel
    prometheus_type = PrometheusFlagshipModel
    perseus_type = PerseusFlagshipModel
    atlas_type = AtlasFlagshipModel
    kronos_type = KronosFlagshipModel
    aion_type = AionFlagshipModel
    hermes: HermesFlagshipModel
    prometheus: PrometheusFlagshipModel
    perseus: PerseusFlagshipModel
    atlas: AtlasFlagshipModel
    kronos: KronosFlagshipModel
    aion: AionFlagshipModel

    def __init__(self, config: OlympusSystemConfig) -> None:
        if config.architecture_revision != "flagship_v2":
            raise ValueError("flagship system requires architecture_revision='flagship_v2'")
        # Factories avoid allocating and then discarding six predecessor models.
        super().__init__(config)

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
        allowed_tools: Tensor | None = None,
        patch_mask: Tensor | None = None,
        resident_experts: Tensor | None = None,
        transition_costs: Tensor | None = None,
        remaining_budget: Tensor | None = None,
    ) -> FlagshipSystemOutput:
        _continuous(risk_score, image, "risk_score")
        if any(value.shape[0] != tokens.shape[0] for value in (image, states, actions)):
            raise ValueError("all Olympus family inputs must have the same batch size")
        hermes = self.hermes(tokens, attention_mask, allowed_tools=allowed_tools)
        prometheus = self.prometheus(tokens, attention_mask, domain_ids=domain_ids)
        perseus = self.perseus(tokens, attention_mask, resident_experts=resident_experts)
        atlas = self.atlas(
            tokens, image, attention_mask, timestamp=timestamp, patch_mask=patch_mask
        )
        kronos = self.kronos(states, actions, temporal_mask)
        base = self._compose_outputs(
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
        hermes = cast(HermesFlagshipOutput, base.hermes)
        prometheus = cast(PrometheusFlagshipOutput, base.prometheus)
        kronos = cast(KronosFlagshipOutput, base.kronos)
        active = _active(tokens, attention_mask)
        adapter_risk = (prometheus.adapter_disagreement * active).sum(1) / active.sum(1)
        adapter_risk = adapter_risk / (1 + adapter_risk)
        temporal_risk = kronos.cumulative_plan_uncertainty[:, -1].mean(-1)
        temporal_risk = temporal_risk / (1 + temporal_risk)
        terms = torch.stack((hermes.tool_disagreement, adapter_risk, temporal_risk), -1)
        # Terms are engineering diagnostics, not calibrated probabilities.
        effective = 1 - (1 - base.effective_risk) * (1 - terms).prod(-1)
        aion = self.aion(
            self.aion_state(base.fused_state),
            allowed_transitions,
            effective,
            transition_costs=transition_costs,
            remaining_budget=remaining_budget,
        )
        available = ~aion.forced_stop
        values = {name: getattr(base, name) for name in OlympusSystemOutput.__dataclass_fields__}
        values.update(
            aion=aion,
            effective_risk=effective,
            decision_available=available,
            decision_logits=base.raw_decision_logits.masked_fill(~available[:, None], 0),
        )
        return FlagshipSystemOutput(**values, evidence_risk_terms=terms)
