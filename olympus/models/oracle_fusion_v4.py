"""V4: frozen Olympus expert banks with common, cross-fitted routing controls."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from olympus.models.olympus_system import OlympusReferenceSystem, OlympusSystemConfig
from olympus.models.reference_families import (
    AionReferenceConfig,
    AtlasReferenceConfig,
    HermesReferenceConfig,
    KronosReferenceConfig,
    PerseusReferenceConfig,
    PrometheusReferenceConfig,
    ReferenceTransformerConfig,
)

MODES = (
    "uniform",
    "fixed",
    "semantic",
    "utility",
    "corruption",
    "oracle_corruption",
    "oracle_task",
    "best_single",
    "semantic_train_mean",
)
INPUTS = ("tokens", "image", "states", "actions", "allowed_transitions", "risk_score")


def system_config() -> OlympusSystemConfig:
    backbone = ReferenceTransformerConfig(
        vocab_size=32,
        model_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=32,
        max_sequence_length=8,
    )
    return OlympusSystemConfig(
        hermes=HermesReferenceConfig(backbone=backbone, tool_count=4),
        prometheus=PrometheusReferenceConfig(
            backbone=backbone, domains=3, adapter_dim=8, evidence_classes=4
        ),
        perseus=PerseusReferenceConfig(
            backbone=backbone, experts=4, active_experts=2, expert_dim=32
        ),
        atlas=AtlasReferenceConfig(
            backbone=backbone, image_channels=3, patch_size=2, retrieval_dim=12
        ),
        kronos=KronosReferenceConfig(
            state_dim=5, action_dim=3, hidden_dim=12, discrete_actions=4, planning_horizon=4
        ),
        aion=AionReferenceConfig(state_dim=8, hidden_dim=12, research_stages=4, stop_stage=3),
        fusion_dim=16,
        decision_classes=4,
        fusion_strategy="uniform",
    )


class ExpertBank(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.system = OlympusReferenceSystem(system_config())
        self.heads = nn.ModuleList(nn.Linear(16, 4) for _ in range(5))

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        output = self.system(**{key: batch[key] for key in INPUTS}, domain_ids=batch["domain_ids"])
        z = output.component_states
        logits = torch.stack([head(z[:, i]) for i, head in enumerate(self.heads)], 1)
        return z, logits, output.kronos.next_state


def expert_loss(logits: Tensor, target: Tensor, weight: float) -> Tensor:
    # All experts have explicit task supervision; no head is rewarded for a label
    # it only saw through target-derived metadata. CF is the ONLY paired change.
    ce = nn.functional.cross_entropy(logits.flatten(0, 1), target.repeat_interleave(5))
    p = logits.softmax(-1)
    full = p.mean(1).detach()
    leave = (p.sum(1, keepdim=True) - p) / 4
    consistency = (
        nn.functional.kl_div(
            leave.clamp_min(1e-8).log(), full[:, None].expand_as(leave), reduction="none"
        )
        .sum(-1)
        .mean()
    )
    return ce + weight * consistency


def router_features(z: Tensor, logits: Tensor) -> Tensor:
    p = logits.softmax(-1)
    entropy = -(p * p.clamp_min(1e-8).log()).sum(-1)
    return torch.cat([z.flatten(1), p.flatten(1), entropy], 1).detach()


class Routers(nn.Module):
    """Every control shares this exact trained graph and frozen expert predictions.

    Heads do not share trainable hidden layers, avoiding auxiliary-loss transfer
    between semantic, utility, and corruption controls. All are always executed.
    """

    training_mean_gates: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.fixed = nn.Parameter(torch.zeros(5))
        self.register_buffer("training_mean_gates", torch.ones(5) / 5)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(105), nn.Linear(105, 32), nn.GELU(), nn.Linear(32, 5)
                )
                for name in ("semantic", "utility", "corruption")
            }
        )

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        return {
            **{name: head(features) for name, head in self.heads.items()},
            "fixed": self.fixed.expand(features.shape[0], -1),
            "semantic_train_mean": self.training_mean_gates.expand(features.shape[0], -1),
        }


def normalize_quality(quality: Tensor) -> Tensor:
    mass = quality.sum(-1, keepdim=True)
    return torch.where(mass > 0, quality / mass.clamp_min(1e-8), torch.ones_like(quality) / 5)


def probabilities(
    logits: Tensor,
    scores: dict[str, Tensor],
    mode: str,
    *,
    quality: Tensor | None = None,
    labels: Tensor | None = None,
    best_single: int = 0,
) -> tuple[Tensor, Tensor]:
    p = logits.softmax(-1)
    if mode == "uniform":
        gates = torch.ones_like(p[..., 0]) / 5
    elif mode in {"fixed", "semantic"}:
        gates = scores[mode].softmax(-1)
    elif mode == "semantic_train_mean":
        gates = scores[mode]
    elif mode in {"utility", "corruption"}:
        gates = normalize_quality(scores[mode].sigmoid())
    elif mode == "oracle_corruption":
        if quality is None:
            raise ValueError("privileged corruption oracle requires quality metadata")
        gates = normalize_quality(quality)
    elif mode == "oracle_task":
        if labels is None:
            raise ValueError("privileged task oracle requires true labels")
        utility = p.gather(2, labels[:, None, None].expand(-1, 5, 1)).squeeze(-1)
        gates = nn.functional.one_hot(utility.argmax(-1), 5).to(p)
    elif mode == "best_single":
        gates = torch.zeros_like(p[..., 0])
        gates[:, best_single] = 1
    else:
        raise ValueError(mode)
    return (gates[..., None] * p).sum(1), gates
