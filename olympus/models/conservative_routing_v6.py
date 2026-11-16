"""Disjoint-stacked routing with explicit reference intervention (v6).

This is a bounded reference model, not a safety or improvement guarantee.
Group identifiers are used only by the training objective, never inference.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn


def validate_probabilities(p: Tensor) -> None:
    if p.ndim < 2 or p.shape[0] == 0 or p.shape[-1] < 2:
        raise ValueError("probabilities require nonempty examples and >=2 classes")
    if not bool(torch.isfinite(p).all()) or bool((p < 0).any()):
        raise ValueError("probabilities must be finite and nonnegative")
    if not torch.allclose(p.sum(-1), torch.ones_like(p[..., 0]), atol=1e-5, rtol=0):
        raise ValueError("probabilities must sum to one")


def blend(reference: Tensor, alternative: Tensor, alpha: Tensor) -> Tensor:
    validate_probabilities(reference)
    validate_probabilities(alternative)
    if reference.shape != alternative.shape or alpha.shape != reference.shape[:-1] + (1,):
        raise ValueError("reference/alternative/alpha shapes disagree")
    if not bool(torch.isfinite(alpha).all()) or bool(((alpha < 0) | (alpha > 1)).any()):
        raise ValueError("alpha must be finite in [0,1]")
    return (1 - alpha) * reference + alpha * alternative


def features(z: Tensor, probabilities: Tensor) -> Tensor:
    if z.ndim != 3 or z.shape[1:] != (5, 16) or probabilities.shape != (len(z), 5, 4):
        raise ValueError("expected [N,5,16] states and [N,5,4] probabilities")
    validate_probabilities(probabilities)
    if not bool(torch.isfinite(z).all()):
        raise ValueError("nonfinite expert states")
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
    return torch.cat((z.flatten(1), probabilities.flatten(1), entropy), -1).detach()


def routing_network() -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(105), nn.Linear(105, 32), nn.GELU(), nn.Linear(32, 5))


class ReferenceRoutes(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.semantic = routing_network()
        self.fixed = nn.Parameter(torch.zeros(5))

    def forward(self, x: Tensor, experts: Tensor) -> dict[str, Tensor]:
        return {
            **{f"expert_{i}": experts[:, i] for i in range(5)},
            "uniform": experts.mean(1),
            "fixed": (self.fixed.softmax(-1)[None, :, None] * experts).sum(1),
            "semantic": (self.semantic(x).softmax(-1)[..., None] * experts).sum(1),
        }


class Correction(nn.Module):
    """Shared alternative initialization; only the declared correction differs.

    Direct variants have no intervention parameters. Constant variants have one;
    adaptive variants have 106. Counts are reported rather than padded with dead
    parameters. A constant blend is a trained input-independent alpha control.
    """

    def __init__(self, initial: nn.Sequential, kind: str, alpha_initial: float = 0.1) -> None:
        super().__init__()
        if kind not in {"direct", "anchored", "constant"} or not 0 < alpha_initial < 1:
            raise ValueError("invalid correction kind or alpha initialization")
        self.kind = kind
        self.alternative = copy.deepcopy(initial).requires_grad_(True)
        if kind == "anchored":
            self.intervention: nn.Module = nn.Linear(105, 1)
            nn.init.zeros_(self.intervention.weight)
            nn.init.constant_(
                self.intervention.bias, float(torch.logit(torch.tensor(alpha_initial)))
            )
        elif kind == "constant":
            self.intervention = nn.Linear(1, 1, bias=False)
            nn.init.constant_(
                self.intervention.weight, float(torch.logit(torch.tensor(alpha_initial)))
            )
        else:
            self.intervention = nn.Identity()

    def forward(self, x: Tensor, experts: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
        alternative = (self.alternative(x).softmax(-1)[..., None] * experts).sum(1)
        if self.kind == "direct":
            alpha = torch.ones_like(reference[:, :1])
        elif self.kind == "constant":
            alpha = self.intervention(torch.ones_like(reference[:, :1])).sigmoid()
        else:
            alpha = self.intervention(x).sigmoid()
        return blend(reference, alternative, alpha), alpha


def correction_loss(
    predictions: Tensor,
    labels: Tensor,
    groups: Tensor,
    alpha: Tensor,
    *,
    robust: bool,
    penalty: float,
    group_count: int = 3,
) -> Tensor:
    validate_probabilities(predictions)
    if labels.shape != (len(predictions),) or groups.shape != labels.shape:
        raise ValueError("labels/groups must have one entry per prediction")
    if labels.dtype != torch.long or groups.dtype != torch.long:
        raise ValueError("integer labels/groups required")
    if bool(((labels < 0) | (labels >= predictions.shape[-1])).any()):
        raise ValueError("class out of range")
    if group_count < 1 or bool(((groups < 0) | (groups >= group_count)).any()):
        raise ValueError("group out of range")
    if penalty < 0 or not __import__("math").isfinite(penalty):
        raise ValueError("invalid anchor penalty")
    if alpha.shape != (len(predictions), 1) or not bool(torch.isfinite(alpha).all()):
        raise ValueError("invalid alpha")
    if bool(((alpha < 0) | (alpha > 1)).any()):
        raise ValueError("alpha out of range")
    losses = -predictions.gather(1, labels[:, None]).clamp_min(1e-8).log().squeeze(1)
    per_group = []
    for group in range(group_count):
        mask = groups == group
        if not bool(mask.any()):
            raise ValueError("every declared group must be present")
        per_group.append(losses[mask].mean())
    values = torch.stack(per_group)
    objective = values.max() if robust else values.mean()
    return objective + penalty * alpha.mean()


def intervention_diagnostics(p: Tensor, reference: Tensor, y: Tensor) -> dict[str, int | float]:
    good, ref_good = p.argmax(-1).eq(y), reference.argmax(-1).eq(y)
    rescues = int((good & ~ref_good).sum())
    harms = int((~good & ref_good).sum())
    return {
        "examples": len(y),
        "rescues": rescues,
        "harms": harms,
        "accuracy_delta": (rescues - harms) / len(y),
        "prediction_changes": int(p.argmax(-1).ne(reference.argmax(-1)).sum()),
    }
