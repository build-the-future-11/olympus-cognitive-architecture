"""Scale-safe reference architectures for every active BU1LD model family.

These modules make each family contract executable at development scale.  They
are deliberately named ``Reference`` models: constructing one does not create a
checkpoint at the target parameter count or satisfy a family promotion gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn

from olympus.core.schemas import StrictModel


class ReferenceTransformerConfig(StrictModel):
    vocab_size: int = Field(default=256, ge=16, le=1_000_000)
    model_dim: int = Field(default=32, ge=8, le=8_192)
    layers: int = Field(default=2, ge=1, le=128)
    heads: int = Field(default=4, ge=1, le=128)
    feedforward_dim: int = Field(default=64, ge=16, le=65_536)
    max_sequence_length: int = Field(default=128, ge=2, le=1_000_000)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def valid_attention_shape(self) -> ReferenceTransformerConfig:
        if self.model_dim % self.heads != 0:
            raise ValueError("model dimension must be divisible by attention heads")
        return self


class _CausalBlock(nn.Module):
    def __init__(self, config: ReferenceTransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.model_dim)
        self.attention = nn.MultiheadAttention(
            config.model_dim,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(config.model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(config.model_dim, config.feedforward_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.model_dim),
        )

    def forward(self, inputs: Tensor, attention_mask: Tensor) -> Tensor:
        length = inputs.shape[1]
        causal_mask = torch.ones((length, length), dtype=torch.bool, device=inputs.device).triu(1)
        normalized = self.attention_norm(inputs)
        attended, _weights = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=~attention_mask,
            need_weights=False,
        )
        hidden = inputs + attended
        hidden = hidden + self.feedforward(self.feedforward_norm(hidden))
        return cast(Tensor, hidden.masked_fill(~attention_mask.unsqueeze(-1), 0.0))


class CausalReferenceBackbone(nn.Module):
    """A bounded causal backbone shared by language-oriented references."""

    def __init__(self, config: ReferenceTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.model_dim)
        self.blocks = nn.ModuleList(_CausalBlock(config) for _ in range(config.layers))
        self.output_norm = nn.LayerNorm(config.model_dim)

    def forward(self, tokens: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        if tokens.shape[1] > self.config.max_sequence_length:
            raise ValueError("token sequence exceeds the configured maximum")
        if tokens.dtype not in {torch.int32, torch.int64}:
            raise ValueError("tokens must use an integer dtype")
        if tokens.numel() == 0 or tokens.min() < 0 or tokens.max() >= self.config.vocab_size:
            raise ValueError("tokens fall outside the configured vocabulary")
        if attention_mask is None:
            mask = torch.ones_like(tokens, dtype=torch.bool)
        else:
            if attention_mask.shape != tokens.shape:
                raise ValueError("attention mask must match token shape")
            if attention_mask.dtype != torch.bool:
                raise ValueError("attention mask must use Boolean dtype")
            if attention_mask.device != tokens.device:
                raise ValueError("attention mask must share the token device")
            mask = attention_mask
            if not bool(mask.any(dim=1).all()):
                raise ValueError("every sequence must contain at least one active token")
            if bool(((~mask).cummax(dim=1).values & mask).any()):
                raise ValueError("active tokens must form a contiguous prefix")
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            hidden = block(hidden, mask)
        return cast(Tensor, self.output_norm(hidden)).masked_fill(~mask.unsqueeze(-1), 0.0)


class HermesReferenceConfig(StrictModel):
    family_id: Literal["hermes"] = "hermes"
    backbone: ReferenceTransformerConfig = Field(default_factory=ReferenceTransformerConfig)
    tool_count: int = Field(default=8, ge=1, le=65_536)


@dataclass(slots=True)
class HermesReferenceOutput:
    token_logits: Tensor
    tool_logits: Tensor
    abstention_logit: Tensor
    hidden_states: Tensor


class HermesReferenceModel(nn.Module):
    """Local-assistant reference with explicit tool and abstention heads."""

    def __init__(self, config: HermesReferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = CausalReferenceBackbone(config.backbone)
        self.language_head = nn.Linear(config.backbone.model_dim, config.backbone.vocab_size)
        self.tool_head = nn.Linear(config.backbone.model_dim, config.tool_count)
        self.abstention_head = nn.Linear(config.backbone.model_dim, 1)

    def forward(
        self, tokens: Tensor, attention_mask: Tensor | None = None
    ) -> HermesReferenceOutput:
        hidden = self.backbone(tokens, attention_mask)
        pooled = _last_active(hidden, attention_mask)
        return HermesReferenceOutput(
            token_logits=self.language_head(hidden),
            tool_logits=self.tool_head(pooled),
            abstention_logit=self.abstention_head(pooled).squeeze(-1),
            hidden_states=hidden,
        )


class PrometheusReferenceConfig(StrictModel):
    family_id: Literal["prometheus"] = "prometheus"
    backbone: ReferenceTransformerConfig = Field(default_factory=ReferenceTransformerConfig)
    domains: int = Field(default=4, ge=2, le=256)
    adapter_dim: int = Field(default=16, ge=2, le=8_192)
    evidence_classes: int = Field(default=3, ge=2, le=256)


@dataclass(slots=True)
class PrometheusReferenceOutput:
    token_logits: Tensor
    evidence_logits: Tensor
    domain_weights: Tensor
    hidden_states: Tensor


class PrometheusReferenceModel(nn.Module):
    """Domain-adaptable reference with auditable adapter routing and evidence heads."""

    def __init__(self, config: PrometheusReferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = CausalReferenceBackbone(config.backbone)
        self.domain_router = nn.Linear(config.backbone.model_dim, config.domains)
        self.adapters = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(config.backbone.model_dim),
                nn.Linear(config.backbone.model_dim, config.adapter_dim),
                nn.GELU(),
                nn.Linear(config.adapter_dim, config.backbone.model_dim),
            )
            for _ in range(config.domains)
        )
        self.language_head = nn.Linear(config.backbone.model_dim, config.backbone.vocab_size)
        self.evidence_head = nn.Linear(config.backbone.model_dim, config.evidence_classes)

    def forward(
        self,
        tokens: Tensor,
        attention_mask: Tensor | None = None,
        *,
        domain_ids: Tensor | None = None,
    ) -> PrometheusReferenceOutput:
        hidden = self.backbone(tokens, attention_mask)
        pooled = _last_active(hidden, attention_mask)
        if domain_ids is None:
            domain_weights = self.domain_router(pooled).softmax(dim=-1)
        else:
            if domain_ids.shape != (tokens.shape[0],):
                raise ValueError("domain identifiers must have shape [batch]")
            if domain_ids.dtype not in {torch.int32, torch.int64}:
                raise ValueError("domain identifiers must use an integer dtype")
            if domain_ids.device != hidden.device:
                raise ValueError("domain identifiers must share the token device")
            if domain_ids.min() < 0 or domain_ids.max() >= self.config.domains:
                raise ValueError("domain identifier is outside the configured domain set")
            domain_weights = nn.functional.one_hot(
                domain_ids.to(dtype=torch.long), num_classes=self.config.domains
            ).to(dtype=hidden.dtype)
        adapter_outputs = torch.stack([adapter(hidden) for adapter in self.adapters], dim=2)
        adapted = hidden + (adapter_outputs * domain_weights[:, None, :, None]).sum(dim=2)
        return PrometheusReferenceOutput(
            token_logits=self.language_head(adapted),
            evidence_logits=self.evidence_head(adapted),
            domain_weights=domain_weights,
            hidden_states=adapted,
        )


class PerseusReferenceConfig(StrictModel):
    family_id: Literal["perseus"] = "perseus"
    backbone: ReferenceTransformerConfig = Field(default_factory=ReferenceTransformerConfig)
    experts: int = Field(default=4, ge=2, le=256)
    active_experts: int = Field(default=2, ge=1, le=32)
    expert_dim: int = Field(default=64, ge=8, le=65_536)

    @model_validator(mode="after")
    def valid_expert_count(self) -> PerseusReferenceConfig:
        if self.active_experts > self.experts:
            raise ValueError("active experts must not exceed total experts")
        return self


@dataclass(slots=True)
class PerseusReferenceOutput:
    token_logits: Tensor
    routing_probabilities: Tensor
    active_expert_indices: Tensor
    load_balance_loss: Tensor
    hidden_states: Tensor


class PerseusReferenceModel(nn.Module):
    """Sparse top-k expert reference with observable routing and balance loss."""

    def __init__(self, config: PerseusReferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = CausalReferenceBackbone(config.backbone)
        self.router = nn.Linear(config.backbone.model_dim, config.experts)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(config.backbone.model_dim, config.expert_dim),
                nn.SiLU(),
                nn.Linear(config.expert_dim, config.backbone.model_dim),
            )
            for _ in range(config.experts)
        )
        self.language_head = nn.Linear(config.backbone.model_dim, config.backbone.vocab_size)

    def forward(
        self, tokens: Tensor, attention_mask: Tensor | None = None
    ) -> PerseusReferenceOutput:
        hidden = self.backbone(tokens, attention_mask)
        routing = self.router(hidden).softmax(dim=-1)
        top_values, top_indices = routing.topk(self.config.active_experts, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True)
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_indices = top_indices.reshape(-1, self.config.active_experts)
        flat_weights = top_weights.reshape(-1, self.config.active_experts)
        mixed = torch.zeros_like(flat_hidden)
        for expert_index, expert in enumerate(self.experts):
            token_indices, slots = torch.where(flat_indices == expert_index)
            if token_indices.numel() == 0:
                continue
            contribution = expert(flat_hidden[token_indices])
            contribution = contribution * flat_weights[token_indices, slots].unsqueeze(-1)
            mixed.index_add_(0, token_indices, contribution)
        mixed = (hidden + mixed.reshape_as(hidden)).masked_fill(
            ~_attention_mask(tokens, attention_mask).unsqueeze(-1), 0.0
        )
        mask = _attention_mask(tokens, attention_mask)
        assignments = nn.functional.one_hot(top_indices, num_classes=self.config.experts).to(
            dtype=hidden.dtype
        )
        active_routes = mask.sum().to(dtype=hidden.dtype) * self.config.active_experts
        assignment_fraction = (assignments * mask[..., None, None]).sum(
            dim=(0, 1, 2)
        ) / active_routes
        mean_probability = (routing * mask.unsqueeze(-1)).sum(dim=(0, 1))
        mean_probability = mean_probability / mask.sum().to(dtype=hidden.dtype)
        balance = self.config.experts * (assignment_fraction * mean_probability).sum()
        return PerseusReferenceOutput(
            token_logits=self.language_head(mixed),
            routing_probabilities=routing,
            active_expert_indices=top_indices,
            load_balance_loss=balance,
            hidden_states=mixed,
        )


class AtlasReferenceConfig(StrictModel):
    family_id: Literal["atlas"] = "atlas"
    backbone: ReferenceTransformerConfig = Field(default_factory=ReferenceTransformerConfig)
    image_channels: int = Field(default=3, ge=1, le=16)
    patch_size: int = Field(default=4, ge=1, le=64)
    retrieval_dim: int = Field(default=16, ge=2, le=8_192)


@dataclass(slots=True)
class AtlasReferenceOutput:
    token_logits: Tensor
    retrieval_embedding: Tensor
    grounding_weights: Tensor
    hidden_states: Tensor


class AtlasReferenceModel(nn.Module):
    """Multimodal grounding reference with patch-level attribution."""

    def __init__(self, config: AtlasReferenceConfig) -> None:
        super().__init__()
        self.config = config
        dimension = config.backbone.model_dim
        self.text_backbone = CausalReferenceBackbone(config.backbone)
        self.patch_embedding = nn.Conv2d(
            config.image_channels,
            dimension,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.coordinate_embedding = nn.Linear(3, dimension)
        self.cross_attention = nn.MultiheadAttention(
            dimension, config.backbone.heads, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(dimension)
        self.language_head = nn.Linear(dimension, config.backbone.vocab_size)
        self.retrieval_head = nn.Linear(dimension, config.retrieval_dim)

    def forward(
        self,
        tokens: Tensor,
        image: Tensor,
        attention_mask: Tensor | None = None,
        *,
        timestamp: Tensor | None = None,
    ) -> AtlasReferenceOutput:
        if image.ndim != 4 or image.shape[0] != tokens.shape[0]:
            raise ValueError("image must have shape [batch, channels, height, width]")
        if image.shape[1] != self.config.image_channels:
            raise ValueError("image channels do not match the Atlas configuration")
        if image.shape[-2] < self.config.patch_size or image.shape[-1] < self.config.patch_size:
            raise ValueError("image is smaller than one configured patch")
        if not image.is_floating_point():
            raise ValueError("image must use a floating-point dtype")
        if image.device != tokens.device:
            raise ValueError("image must share the token device")
        if image.dtype != self.patch_embedding.weight.dtype:
            raise ValueError("image dtype must match the Atlas parameter dtype")
        if not bool(torch.isfinite(image).all()):
            raise ValueError("image must contain only finite values")
        if bool(((image < 0.0) | (image > 1.0)).any()):
            raise ValueError("image values must fall in [0, 1]")
        text = self.text_backbone(tokens, attention_mask)
        patches_2d = self.patch_embedding(image)
        patch_height, patch_width = patches_2d.shape[-2:]
        patches = patches_2d.flatten(2).transpose(1, 2)
        coordinates = _patch_coordinates(
            patch_height,
            patch_width,
            dtype=patches.dtype,
            device=patches.device,
            batch=patches.shape[0],
            timestamp=timestamp,
        )
        patches = patches + self.coordinate_embedding(coordinates)
        grounded, weights = self.cross_attention(
            text,
            patches,
            patches,
            need_weights=True,
            average_attn_weights=True,
        )
        fused = self.fusion_norm(text + grounded)
        mask = _attention_mask(tokens, attention_mask)
        fused = fused.masked_fill(~mask.unsqueeze(-1), 0.0)
        pooled = (fused * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        retrieval = nn.functional.normalize(self.retrieval_head(pooled), dim=-1)
        return AtlasReferenceOutput(
            token_logits=self.language_head(fused),
            retrieval_embedding=retrieval,
            grounding_weights=weights.masked_fill(~mask.unsqueeze(-1), 0.0),
            hidden_states=fused,
        )


class KronosReferenceConfig(StrictModel):
    family_id: Literal["kronos"] = "kronos"
    state_dim: int = Field(default=16, ge=2, le=65_536)
    action_dim: int = Field(default=8, ge=1, le=65_536)
    hidden_dim: int = Field(default=32, ge=4, le=65_536)
    discrete_actions: int = Field(default=6, ge=2, le=65_536)
    planning_horizon: int = Field(default=4, ge=1, le=1_024)


@dataclass(slots=True)
class KronosReferenceOutput:
    next_state: Tensor
    reward: Tensor
    value: Tensor
    uncertainty: Tensor
    plan_logits: Tensor
    hidden_states: Tensor


class KronosReferenceModel(nn.Module):
    """Temporal world-model reference with explicit uncertainty and planning."""

    def __init__(self, config: KronosReferenceConfig) -> None:
        super().__init__()
        self.config = config
        transition_input = config.state_dim + config.action_dim
        self.transition = nn.GRU(transition_input, config.hidden_dim, batch_first=True)
        self.next_state_head = nn.Linear(config.hidden_dim, config.state_dim)
        self.reward_head = nn.Linear(config.hidden_dim, 1)
        self.value_head = nn.Linear(config.hidden_dim, 1)
        self.uncertainty_head = nn.Linear(config.hidden_dim, config.state_dim)
        self.plan_head = nn.Linear(config.hidden_dim, config.discrete_actions)
        self.action_dictionary = nn.Parameter(
            torch.empty(config.discrete_actions, config.action_dim)
        )
        self.plan_cell = nn.GRUCell(transition_input, config.hidden_dim)
        nn.init.normal_(self.action_dictionary, std=1.0 / math.sqrt(config.action_dim))

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        valid_mask: Tensor | None = None,
    ) -> KronosReferenceOutput:
        if states.ndim != 3 or states.shape[-1] != self.config.state_dim:
            raise ValueError("states must have shape [batch, time, state_dim]")
        if actions.shape != (*states.shape[:2], self.config.action_dim):
            raise ValueError("actions must align with states and the configured action dimension")
        if not states.is_floating_point() or not actions.is_floating_point():
            raise ValueError("states and actions must use floating-point dtypes")
        if states.device != actions.device or states.device != self.transition.weight_ih_l0.device:
            raise ValueError("states, actions, and Kronos parameters must share one device")
        if states.dtype != actions.dtype or states.dtype != self.transition.weight_ih_l0.dtype:
            raise ValueError("states, actions, and Kronos parameters must share one dtype")
        if not bool(torch.isfinite(states).all()) or not bool(torch.isfinite(actions).all()):
            raise ValueError("states and actions must contain only finite values")
        if valid_mask is None:
            mask = torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)
        else:
            if valid_mask.shape != states.shape[:2]:
                raise ValueError("valid mask must have shape [batch, time]")
            if valid_mask.dtype != torch.bool:
                raise ValueError("valid mask must use Boolean dtype")
            if valid_mask.device != states.device:
                raise ValueError("valid mask must share the trajectory device")
            mask = valid_mask
            if not bool(mask.any(dim=1).all()):
                raise ValueError("every trajectory must contain at least one valid transition")
            if bool(((~mask).cummax(dim=1).values & mask).any()):
                raise ValueError("valid trajectory steps must form a contiguous prefix")
        sequence, _hidden = self.transition(torch.cat((states, actions), dim=-1))
        sequence = sequence.masked_fill(~mask.unsqueeze(-1), 0.0)
        lengths = mask.sum(dim=1).to(dtype=torch.long)
        batch_indices = torch.arange(states.shape[0], device=states.device)
        final_hidden = sequence[batch_indices, lengths - 1]
        final_state = states[batch_indices, lengths - 1]
        plan_steps: list[Tensor] = []
        cursor = final_hidden
        for _step in range(self.config.planning_horizon):
            logits = self.plan_head(cursor)
            plan_steps.append(logits)
            soft_action = logits.softmax(dim=-1) @ self.action_dictionary
            cursor = self.plan_cell(torch.cat((final_state, soft_action), dim=-1), cursor)
        next_state = self.next_state_head(sequence).masked_fill(~mask.unsqueeze(-1), 0.0)
        reward = self.reward_head(sequence).squeeze(-1).masked_fill(~mask, 0.0)
        value = self.value_head(sequence).squeeze(-1).masked_fill(~mask, 0.0)
        uncertainty = (nn.functional.softplus(self.uncertainty_head(sequence)) + 1e-6).masked_fill(
            ~mask.unsqueeze(-1), 0.0
        )
        return KronosReferenceOutput(
            next_state=next_state,
            reward=reward,
            value=value,
            uncertainty=uncertainty,
            plan_logits=torch.stack(plan_steps, dim=1),
            hidden_states=sequence,
        )


class AionReferenceConfig(StrictModel):
    family_id: Literal["aion"] = "aion"
    state_dim: int = Field(default=16, ge=2, le=65_536)
    hidden_dim: int = Field(default=32, ge=4, le=65_536)
    research_stages: int = Field(default=8, ge=2, le=256)
    stop_stage: int = Field(default=7, ge=0, le=255)
    kill_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def valid_stop_stage(self) -> AionReferenceConfig:
        if self.stop_stage >= self.research_stages:
            raise ValueError("stop stage must be inside the configured research stages")
        return self


@dataclass(slots=True)
class AionReferenceOutput:
    transition_logits: Tensor
    transition_probabilities: Tensor
    selected_stage: Tensor
    forced_stop: Tensor


class AionReferenceModel(nn.Module):
    """Governed research router constrained by host-approved transitions and a kill rule."""

    def __init__(self, config: AionReferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.router = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.research_stages),
        )

    def forward(
        self,
        state_features: Tensor,
        allowed_transitions: Tensor,
        risk_score: Tensor,
    ) -> AionReferenceOutput:
        if state_features.ndim != 2 or state_features.shape[-1] != self.config.state_dim:
            raise ValueError("state features must have shape [batch, state_dim]")
        if not state_features.is_floating_point():
            raise ValueError("state features must use a floating-point dtype")
        first_parameter = cast(nn.Linear, self.router[1]).weight
        if (
            state_features.device != first_parameter.device
            or state_features.dtype != first_parameter.dtype
        ):
            raise ValueError("state features must match the Aion parameter device and dtype")
        expected = (state_features.shape[0], self.config.research_stages)
        if allowed_transitions.shape != expected:
            raise ValueError("allowed-transition mask does not match batch and stage counts")
        if allowed_transitions.dtype != torch.bool:
            raise ValueError("allowed-transition mask must use Boolean dtype")
        if allowed_transitions.device != state_features.device:
            raise ValueError("allowed-transition mask must share the state device")
        if risk_score.shape != (state_features.shape[0],):
            raise ValueError("risk score must have shape [batch]")
        if not risk_score.is_floating_point():
            raise ValueError("risk score must use a floating-point dtype")
        if risk_score.device != state_features.device or risk_score.dtype != state_features.dtype:
            raise ValueError("risk score must share the state device and dtype")
        risk = risk_score
        if not bool(torch.isfinite(state_features).all()) or not bool(torch.isfinite(risk).all()):
            raise ValueError("state features and risk scores must be finite")
        if bool(((risk < 0.0) | (risk > 1.0)).any()):
            raise ValueError("risk scores must fall in [0, 1]")
        allowed = allowed_transitions
        if not bool(allowed.any(dim=1).all()):
            raise ValueError("every state must permit at least one transition")
        forced_stop = risk >= self.config.kill_threshold
        if bool((forced_stop & ~allowed[:, self.config.stop_stage]).any()):
            raise ValueError("the kill rule requires STOP to be an allowed transition")
        logits = self.router(state_features).masked_fill(~allowed, -torch.inf)
        if bool(forced_stop.any()):
            forced_logits = torch.full_like(logits, -torch.inf)
            forced_logits[:, self.config.stop_stage] = 0.0
            logits = torch.where(forced_stop.unsqueeze(-1), forced_logits, logits)
        probabilities = logits.softmax(dim=-1)
        return AionReferenceOutput(
            transition_logits=logits,
            transition_probabilities=probabilities,
            selected_stage=probabilities.argmax(dim=-1),
            forced_stop=forced_stop,
        )


ReferenceFamilyConfig = (
    HermesReferenceConfig
    | PrometheusReferenceConfig
    | PerseusReferenceConfig
    | AtlasReferenceConfig
    | KronosReferenceConfig
    | AionReferenceConfig
)


def build_reference_model(config: ReferenceFamilyConfig) -> nn.Module:
    """Construct a development-scale family implementation from a typed config."""

    if isinstance(config, HermesReferenceConfig):
        return HermesReferenceModel(config)
    if isinstance(config, PrometheusReferenceConfig):
        return PrometheusReferenceModel(config)
    if isinstance(config, PerseusReferenceConfig):
        return PerseusReferenceModel(config)
    if isinstance(config, AtlasReferenceConfig):
        return AtlasReferenceModel(config)
    if isinstance(config, KronosReferenceConfig):
        return KronosReferenceModel(config)
    if isinstance(config, AionReferenceConfig):
        return AionReferenceModel(config)
    raise TypeError(f"unsupported family config: {type(config).__name__}")


def default_reference_configs() -> tuple[ReferenceFamilyConfig, ...]:
    """Return bounded defaults for the complete six-family lineup."""

    return (
        HermesReferenceConfig(),
        PrometheusReferenceConfig(),
        PerseusReferenceConfig(),
        AtlasReferenceConfig(),
        KronosReferenceConfig(),
        AionReferenceConfig(),
    )


def _attention_mask(tokens: Tensor, attention_mask: Tensor | None) -> Tensor:
    if attention_mask is None:
        return torch.ones_like(tokens, dtype=torch.bool)
    return attention_mask


def _last_active(hidden: Tensor, attention_mask: Tensor | None) -> Tensor:
    if attention_mask is None:
        return hidden[:, -1]
    mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
    indices = mask.sum(dim=1).to(dtype=torch.long) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]


def _patch_coordinates(
    height: int,
    width: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    batch: int,
    timestamp: Tensor | None,
) -> Tensor:
    y = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
    x = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    spatial = torch.stack((grid_x, grid_y), dim=-1).reshape(1, height * width, 2)
    if timestamp is None:
        time = torch.zeros((batch, height * width, 1), dtype=dtype, device=device)
    else:
        if timestamp.shape != (batch,):
            raise ValueError("timestamp must have shape [batch]")
        if not timestamp.is_floating_point():
            raise ValueError("timestamp must use a floating-point dtype")
        if timestamp.device != device or timestamp.dtype != dtype:
            raise ValueError("timestamp must match the image feature device and dtype")
        if not bool(torch.isfinite(timestamp).all()):
            raise ValueError("timestamp values must be finite")
        time = timestamp.tanh()[:, None, None]
        time = time.expand(-1, height * width, -1)
    return torch.cat((spatial.expand(batch, -1, -1), time), dim=-1)
