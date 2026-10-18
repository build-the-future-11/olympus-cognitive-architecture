"""Paired-image Olympus model for the EditCLEVR intervention benchmark.

This model is intentionally narrow: it predicts which object factor changed in
a before/after image pair. It does not fabricate language, temporal, risk, or
governance inputs that EditCLEVR does not contain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

EDIT_FACTORS = ("none", "color", "material", "size", "shape")
FUSION_MODES = ("learned", "uniform", "pair_only", "delta_only")


class _ImageEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, width * 2),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, width * 4),
            nn.GELU(),
            nn.Conv2d(width * 4, width * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, width * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, image: Tensor) -> Tensor:
        return cast(Tensor, self.network(image))


@dataclass(slots=True)
class EditCLEVROutput:
    probabilities: dict[str, Tensor]
    gates: Tensor
    expert_logits: Tensor
    pair_embedding: Tensor


class OlympusEditCLEVRModel(nn.Module):
    """Shared encoders plus explicit experts and an observable router.

    The direct pair expert is a strong control. The learned router must improve
    on it under held-out hard and CoGenT shifts; beating a deliberately weak
    uniform average is insufficient for the registered primary.
    """

    def __init__(self, *, width: int = 32, embedding_dim: int = 128) -> None:
        super().__init__()
        if width < 8 or width % 4:
            raise ValueError("width must be at least 8 and divisible by 4")
        if embedding_dim < 32:
            raise ValueError("embedding_dim must be at least 32")
        encoded = width * 4
        self.image_encoder = _ImageEncoder(width)
        self.delta_encoder = _ImageEncoder(width)
        self.project = nn.Sequential(nn.LayerNorm(encoded), nn.Linear(encoded, embedding_dim))
        self.expert_heads = nn.ModuleList(
            nn.Linear(embedding_dim, len(EDIT_FACTORS)) for _ in range(4)
        )
        pair_width = embedding_dim * 4
        self.pair_network = nn.Sequential(
            nn.LayerNorm(pair_width),
            nn.Linear(pair_width, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        router_width = embedding_dim * 4 + len(EDIT_FACTORS) * 4 + 4
        self.router = nn.Sequential(
            nn.LayerNorm(router_width),
            nn.Linear(router_width, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 4),
        )

    def forward(self, before: Tensor, after: Tensor) -> EditCLEVROutput:
        if before.shape != after.shape or before.ndim != 4 or before.shape[1] != 3:
            raise ValueError("before and after must share shape [batch, 3, height, width]")
        if before.shape[-2] < 32 or before.shape[-1] < 32:
            raise ValueError("images must be at least 32 by 32")
        if not bool(torch.isfinite(before).all() and torch.isfinite(after).all()):
            raise ValueError("images contain non-finite values")

        before_z = self.project(self.image_encoder(before))
        after_z = self.project(self.image_encoder(after))
        # Inputs are normalized to [-1, 1]. Keep both derived views in a
        # bounded range so their encoder receives the same scale as RGB.
        signed_z = self.project(self.delta_encoder((after - before) / 2.0))
        absolute_z = self.project(self.delta_encoder((after - before).abs() / 2.0))
        pair_z = self.pair_network(torch.cat((before_z, after_z, signed_z, absolute_z), dim=-1))
        embeddings = torch.stack((before_z, after_z, absolute_z, pair_z), dim=1)
        logits = torch.stack(
            [head(embeddings[:, index]) for index, head in enumerate(self.expert_heads)], dim=1
        )
        expert_probabilities = logits.softmax(dim=-1)
        entropy = -(
            expert_probabilities * expert_probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1)
        features = torch.cat(
            (embeddings.flatten(1), expert_probabilities.flatten(1), entropy), dim=-1
        )
        gates = self.router(features).softmax(dim=-1)
        probabilities = {
            "learned": (gates[..., None] * expert_probabilities).sum(dim=1),
            "uniform": expert_probabilities.mean(dim=1),
            "pair_only": expert_probabilities[:, 3],
            "delta_only": expert_probabilities[:, 2],
        }
        return EditCLEVROutput(
            probabilities=probabilities,
            gates=gates,
            expert_logits=logits,
            pair_embedding=pair_z,
        )


def editclevr_loss(output: EditCLEVROutput, labels: Tensor) -> Tensor:
    if labels.ndim != 1 or labels.shape[0] != output.expert_logits.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if labels.dtype != torch.long:
        raise ValueError("labels must use torch.long")
    if labels.numel() == 0 or labels.min() < 0 or labels.max() >= len(EDIT_FACTORS):
        raise ValueError("labels fall outside EditCLEVR classes")
    # Every expert is independently useful; the router cannot manufacture gains
    # by starving a control head of supervision.
    expert = nn.functional.cross_entropy(
        output.expert_logits.flatten(0, 1), labels.repeat_interleave(4)
    )
    learned = nn.functional.nll_loss(output.probabilities["learned"].clamp_min(1e-8).log(), labels)
    balance = (output.gates.mean(dim=0) - 0.25).square().mean()
    return expert + learned + 0.01 * balance
