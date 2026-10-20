"""Function-preserving FFN growth and masked latent learning for added capacity.

Widening follows Net2Net's replication/count compensation. JEPA supplies a training
objective for a new residual adapter; neither operation creates pretrained knowledge.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GrowthReceipt:
    original_parameters: int
    expanded_parameters: int
    original_width: int
    expanded_width: int
    optimizer_rebuild_required: bool = True
    trained_target_scale: bool = False


def widen_feedforward(
    block: nn.Sequential,
    width: int,
    *,
    max_parameter_bytes: int,
) -> tuple[nn.Sequential, GrowthReceipt]:
    """Copy a Linear/activation[/zero-dropout]/Linear block into a wider equivalent.

    The old module and optimizer are never mutated. The caller must build a new
    optimizer; preserving outputs at initialization does not preserve training dynamics.
    """
    layers = list(block.children())
    if (
        len(layers) not in {3, 4}
        or not isinstance(layers[0], nn.Linear)
        or not isinstance(layers[-1], nn.Linear)
        or not isinstance(layers[1], (nn.GELU, nn.SiLU, nn.ReLU))
        or (len(layers) == 4 and (not isinstance(layers[2], nn.Dropout) or layers[2].p != 0))
    ):
        raise ValueError("growth supports Linear/activation[/zero-dropout]/Linear only")
    first, last = layers[0], layers[-1]
    assert isinstance(first, nn.Linear) and isinstance(last, nn.Linear)
    old = first.out_features
    if type(width) is not int or width <= old or last.in_features != old:
        raise ValueError("growth width must exceed the aligned original hidden width")
    parameters = list(block.parameters())
    if any(
        p.is_meta or not p.is_floating_point() or not bool(torch.isfinite(p).all())
        for p in parameters
    ):
        raise ValueError("growth requires materialized finite floating-point parameters")
    if any(p.dtype != first.weight.dtype or p.device != first.weight.device for p in parameters):
        raise ValueError("growth parameters must share dtype and device")
    count = width * (first.in_features + last.out_features + int(first.bias is not None))
    count += last.out_features if last.bias is not None else 0
    old_count = sum(p.numel() for p in parameters)
    if (
        type(max_parameter_bytes) is not int
        or (count + old_count) * first.weight.element_size() > max_parameter_bytes
    ):
        raise ValueError("growth exceeds the combined old/new parameter byte budget")
    device, dtype = first.weight.device, first.weight.dtype
    # Allocate uninitialized tensors without consuming any application RNG state.
    left = nn.Linear(first.in_features, width, bias=first.bias is not None, device="meta")
    right = nn.Linear(width, last.out_features, bias=last.bias is not None, device="meta")
    mapping = torch.arange(width, device=device) % old
    counts = torch.bincount(mapping, minlength=old).to(dtype)
    left.weight = nn.Parameter(first.weight.detach()[mapping].clone(), first.weight.requires_grad)
    right.weight = nn.Parameter(
        (last.weight.detach()[:, mapping] / counts[mapping]).clone(), last.weight.requires_grad
    )
    if first.bias is not None:
        left.bias = nn.Parameter(first.bias.detach()[mapping].clone(), first.bias.requires_grad)
    if last.bias is not None:
        right.bias = nn.Parameter(last.bias.detach().clone(), last.bias.requires_grad)
    result = nn.Sequential(left, *(copy.deepcopy(layer) for layer in layers[1:-1]), right)
    result.train(block.training)
    return result, GrowthReceipt(old_count, count, old, width)


class JEPAParameterGrowth(nn.Module):
    """Identity-initialized residual capacity trained with masked latent prediction."""

    def __init__(self, width: int, added_width: int, *, momentum: float = 0.99) -> None:
        super().__init__()
        if type(width) is not int or type(added_width) is not int or min(width, added_width) < 2:
            raise ValueError("widths must be integers >= 2")
        if max(width, added_width) > 8192 or not math.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("growth dimensions or EMA momentum exceed the bounded contract")
        self.width, self.momentum = width, momentum
        self.context_encoder = nn.Linear(width, added_width)
        self.target_encoder = copy.deepcopy(self.context_encoder).requires_grad_(False)
        self.predictor = nn.Sequential(nn.Linear(added_width, added_width), nn.GELU())
        self.residual_head = nn.Linear(added_width, width)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _validate(self, hidden: Tensor) -> None:
        reference = self.context_encoder.weight
        if (
            hidden.ndim != 3
            or hidden.shape[-1] != self.width
            or min(hidden.shape[:2]) < 1
            or hidden.dtype != reference.dtype
            or hidden.device != reference.device
            or not bool(torch.isfinite(hidden).all())
        ):
            raise ValueError(
                "hidden states must be finite [batch, tokens, width] matching parameters"
            )

    def forward(self, hidden: Tensor) -> Tensor:
        self._validate(hidden)
        return cast(
            Tensor, hidden + self.residual_head(self.predictor(self.context_encoder(hidden)))
        )

    def masked_objective(self, hidden: Tensor, target_mask: Tensor) -> Tensor:
        """Exclude target positions from this adapter's latent prediction context.

        This cannot undo information already mixed by an upstream encoder. For
        raw-input masking studies the caller must encode context from separately
        masked inputs; this bounded objective alone does not certify that boundary.
        """
        self._validate(hidden)
        if (
            target_mask.shape != hidden.shape[:2]
            or target_mask.dtype != torch.bool
            or target_mask.device != hidden.device
            or not bool(target_mask.any(-1).all())
            or not bool((~target_mask).any(-1).all())
        ):
            raise ValueError("each sequence needs Boolean visible and target positions")
        visible = ~target_mask
        context = self.context_encoder(hidden.masked_fill(target_mask[..., None], 0))
        pooled = (context * visible[..., None]).sum(1) / visible.sum(1, keepdim=True)
        predicted = self.predictor(pooled)[:, None].expand(-1, hidden.shape[1], -1)
        with torch.no_grad():
            target = self.target_encoder(hidden.detach())
        return cast(Tensor, (predicted[target_mask] - target[target_mask]).square().mean())

    @torch.no_grad()
    def update_target(self) -> None:
        online = tuple(self.context_encoder.parameters())
        target = tuple(self.target_encoder.parameters())
        if any(not bool(torch.isfinite(p).all()) for p in (*online, *target)):
            raise ValueError("EMA update requires finite online and target parameters")
        for teacher, student in zip(target, online, strict=True):
            teacher.lerp_(student, 1 - self.momentum)
