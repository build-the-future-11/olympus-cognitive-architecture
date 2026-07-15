from __future__ import annotations

import torch


def ambiguous_interpretation_dataset() -> list[tuple[str, str]]:
    return [
        ("The cloud was a dragon above the city.", "analogy"),
        ("The log line means the service restarted.", "literal"),
        ("The market breathed before the crash.", "symbolic"),
    ]


def transform_signals() -> torch.Tensor:
    time = torch.linspace(0.0, 6.28, steps=64)
    stacked = torch.stack(
        [
            torch.sin(time),
            torch.cos(time),
            torch.sin(time * 2.0) * 0.5,
        ],
        dim=1,
    )
    return stacked
