"""V5 fresh examples, expanded training coverage, and reserved OOD families."""

from __future__ import annotations

import torch
from torch import Tensor

from olympus.data.oracle_fusion_v4 import generate as previous_generate


def generate(seed: int, count: int, regime: str) -> dict[str, Tensor]:
    if regime not in {"core", "broad", "clean", "corrupted", "known_shift", "novel_shift"}:
        raise ValueError(regime)
    source = {
        "core": "mixed",
        "broad": "mixed",
        "known_shift": "ood",
        "novel_shift": "corrupted",
    }.get(regime, regime)
    batch = previous_generate(seed, count, source)
    g = torch.Generator().manual_seed(seed + 910000)
    if regime == "broad":
        # Same latent examples and missingness as core. Only damaged observations
        # change; clean and missing-only examples remain exactly matched.
        clean = previous_generate(seed, count, "clean")
        choose = torch.rand(count, generator=g) < 0.5
        for index, key in enumerate(("tokens", "image", "states")):
            damaged = (batch["corruption_status"][:, index] == 2) & choose
            x = clean[key].clone()
            if index == 0:
                x[:, :3] = x[:, :3].flip(-1)
            elif index == 1:
                x = (1 - x).clamp(0, 1)
            else:
                x += 1.5
            batch[key][damaged] = x[damaged]
    elif regime == "novel_shift":
        clean = previous_generate(seed, count, "clean")
        for index, key in enumerate(("tokens", "image", "states")):
            damaged = batch["corruption_status"][:, index] == 2
            x = clean[key].clone()
            if index == 0:
                # Relocation is absent from core/broad/development corruption.
                x = x.roll(4, -1)
            elif index == 1:
                # Spatially structured occlusion, not inversion or random noise.
                x[:, :, ::2, ::2] = 0.8
                x[:, :, 1::2, 1::2] = 0.8
            else:
                # Negative time-varying drift, not the known constant positive bias.
                x -= torch.linspace(0.4, 1.6, 4)[None, :, None]
            batch[key][damaged] = x[damaged]
    return batch
