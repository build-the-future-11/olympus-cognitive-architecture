"""Fresh source identities with a prospectively reserved composite shift."""

from __future__ import annotations

import torch
from torch import Tensor

from olympus.data.robust_fusion_v5 import generate as previous_generate


def generate(seed: int, count: int, regime: str) -> dict[str, Tensor]:
    if regime != "composite_shift":
        return previous_generate(seed, count, regime)
    batch = previous_generate(seed, count, "corrupted")
    clean = previous_generate(seed, count, "clean")
    # Reserved configuration declared before execution; no target-dependent transform.
    # Known perturbation types, not a novel family or natural-data OOD claim.
    for index, key in enumerate(("tokens", "image", "states")):
        mask = batch["corruption_status"][:, index] == 2
        x = clean[key].clone()
        if key == "tokens":
            x = x.roll(2, -1)
        elif key == "image":
            x[:, :, 1::2, :] = 0.5
        else:
            x = 0.5 * x + torch.linspace(-0.75, 0.75, x.shape[1])[None, :, None]
        batch[key][mask] = x[mask]
    return batch
