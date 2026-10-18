"""Fresh v4 mechanism data; target-free domain metadata and clean dynamics.

This is a controlled synthetic diagnostic, not a natural-data benchmark.
Only the explicitly privileged oracle consumes corruption metadata or labels.
"""

from __future__ import annotations

import torch
from torch import Tensor


def dynamics(state: Tensor, action: Tensor) -> Tensor:
    padded = torch.nn.functional.pad(action, (0, 2))
    return 0.82 * state + 0.18 * state.roll(1, -1).tanh() + 0.12 * padded


def generate(seed: int, count: int, regime: str) -> dict[str, Tensor]:
    if regime not in {"mixed", "missing_only", "clean", "corrupted", "ood"} or count < 1:
        raise ValueError("invalid v4 split")
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 4, (count,), generator=g)
    # Independent metadata: the target is never used to construct domain or legality.
    domains = torch.randint(0, 3, (count,), generator=g)
    clean_initial = torch.randn(count, 5, generator=g) * 0.15
    clean_initial[torch.arange(count), labels] += 1.5
    actions = torch.rand(count, 4, 3, generator=g) * 2 - 1
    clean_states = [clean_initial]
    future = []
    for t in range(4):
        next_state = dynamics(clean_states[-1], actions[:, t])
        future.append(next_state)
        if t < 3:
            clean_states.append(next_state)
    clean = torch.stack(clean_states, 1)
    next_states = torch.stack(future, 1)

    # Independent modality corruption. Mixed train includes all test intervention
    # categories; OOD reserves bias/reversal interventions, never semantic remapping.
    status = torch.zeros(count, 3, dtype=torch.long)
    branch = torch.randint(0, 4, (count,), generator=g)
    first = torch.randint(0, 3, (count,), generator=g)
    second = (first + torch.randint(1, 3, (count,), generator=g)) % 3
    if regime == "missing_only":
        missing_branch = (branch == 1) | (branch == 3)
        status[missing_branch, first[missing_branch]] = 1
    elif regime == "mixed":
        status[branch == 1, first[branch == 1]] = 1
        status[branch == 2, first[branch == 2]] = 2
        status[branch == 3, first[branch == 3]] = 1
        status[branch == 3, second[branch == 3]] = 2
    elif regime in {"corrupted", "ood"}:
        status[torch.arange(count), first] = 1
        status[torch.arange(count), second] = 2

    tokens = torch.randint(8, 31, (count, 8), generator=g)
    tokens[:, 0] = labels + 1
    tokens[:, 1] = 5 + labels % 2
    tokens[:, 2] = 7 + labels // 2
    image = torch.rand(count, 3, 8, 8, generator=g) * 0.15
    for i in range(count):
        y = int(labels[i])
        image[i, y % 3, (y // 2) * 4 : (y // 2 + 1) * 4, (y % 2) * 4 : (y % 2 + 1) * 4] += 0.75
    states = clean + torch.randn(clean.shape, generator=g) * 0.12
    for modality, x in enumerate((tokens, image, states)):
        missing = status[:, modality] == 1
        damaged = status[:, modality] == 2
        x[missing] = 0
        n = int(damaged.sum())
        if not n:
            continue
        if modality == 0:
            if regime == "ood":
                x[damaged, :3] = x[damaged, :3].flip(-1)
            else:
                x[damaged, :3] = torch.randint(1, 12, (n, 3), generator=g)
        elif modality == 1:
            if regime == "ood":
                x[damaged] = (1.0 - x[damaged]).clamp(0, 1)
            else:
                x[damaged] = torch.rand(n, 3, 8, 8, generator=g)
        elif regime == "ood":
            x[damaged] += 1.5
        else:
            x[damaged] = torch.randn(n, 4, 5, generator=g)
    quality = torch.tensor([1.0, 0.0, 0.25])[status]
    # Atlas consumes BOTH text and image. Its corruption proxy explicitly reflects
    # both, rather than incorrectly labeling it an image-only expert.
    component_quality = torch.stack(
        [
            quality[:, 0],
            quality[:, 0],
            quality[:, 0],
            (quality[:, 0] + quality[:, 1]) / 2,
            quality[:, 2],
        ],
        1,
    )
    return {
        "tokens": tokens,
        "image": image,
        "states": states,
        "actions": actions,
        "domain_ids": domains,
        "allowed_transitions": torch.ones(count, 4, dtype=torch.bool),
        "risk_score": torch.zeros(count),
        "decision_targets": labels,
        "transition_targets": labels % 3,
        "next_state_targets": next_states,
        "component_quality": component_quality,
        "corruption_status": status,
        "source_ids": torch.arange(count, dtype=torch.int64) + seed * 100000,
    }


def subset(batch: dict[str, Tensor], indices: Tensor) -> dict[str, Tensor]:
    return {key: value[indices] for key, value in batch.items()}
