"""Matched routing-feature and temporal-skip ablations on frozen v4 banks."""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from olympus.models.oracle_fusion_v4 import normalize_quality, router_features

ROUTES = (
    "uniform",
    "fixed",
    "semantic",
    "prediction_only",
    "utility",
    "corruption",
    "blend",
    "oracle_task",
)
TEMPORAL = ("v4", "direct", "residual", "persistence")


class Heads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fixed = nn.Parameter(torch.zeros(5))
        net = nn.Sequential(nn.LayerNorm(105), nn.Linear(105, 32), nn.GELU(), nn.Linear(32, 5))
        self.routing = nn.ModuleDict(
            {
                name: copy.deepcopy(net)
                for name in ("semantic", "prediction_only", "utility", "corruption")
            }
        )
        # Identical initialization/parameters/updates for the temporal pair.
        temporal = nn.Sequential(nn.Linear(88, 64), nn.GELU(), nn.Linear(64, 5))
        assert isinstance(temporal[-1], nn.Linear)
        nn.init.zeros_(temporal[-1].weight)
        nn.init.zeros_(temporal[-1].bias)
        self.temporal = nn.ModuleDict(
            {name: copy.deepcopy(temporal) for name in ("direct", "residual")}
        )

    def forward(
        self, z: Tensor, logits: Tensor, states: Tensor, actions: Tensor
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        features = router_features(z, logits)
        # Same 105-dimensional network; zeroing ONLY the latent features isolates
        # representation dependence without changing parameter counts.
        prediction_features = torch.cat((torch.zeros_like(features[:, :80]), features[:, 80:]), -1)
        scores = {
            name: net(prediction_features if name == "prediction_only" else features)
            for name, net in self.routing.items()
        }
        scores["fixed"] = self.fixed.expand(len(z), -1)
        context = z.flatten(1).detach()[:, None].expand(-1, states.shape[1], -1)
        raw = torch.cat((states, actions, context), -1)
        temporal = {
            name: net(raw) + (states if name == "residual" else 0)
            for name, net in self.temporal.items()
        }
        return scores, temporal


def routed(
    logits: Tensor, scores: dict[str, Tensor], labels: Tensor | None = None
) -> dict[str, Tensor]:
    p = logits.softmax(-1)
    gates = {name: scores[name].softmax(-1) for name in ("fixed", "semantic", "prediction_only")}
    gates.update(
        {name: normalize_quality(scores[name].sigmoid()) for name in ("utility", "corruption")}
    )
    gates["uniform"] = torch.ones_like(p[..., 0]) / 5
    gates["blend"] = (gates["prediction_only"] + gates["utility"]) / 2
    values = {name: (gate[..., None] * p).sum(1) for name, gate in gates.items()}
    if labels is not None:
        best = p.gather(2, labels[:, None, None].expand(-1, 5, 1)).squeeze(-1).argmax(-1)
        values["oracle_task"] = p[torch.arange(len(p)), best]
    return values
