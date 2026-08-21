from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class TinyJEPA(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, context: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        context_latent = self.context_encoder(context)
        with torch.no_grad():
            target_latent = self.target_encoder(target)
        predicted = self.predictor(context_latent)
        return predicted, target_latent


@dataclass(slots=True)
class JEPATrainingResult:
    final_loss: float
    variance_regularization: float
    rollout_error: float


def synthetic_sequence_batch(batch_size: int = 32, sequence_dim: int = 4) -> tuple[Tensor, Tensor]:
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    if sequence_dim < 1:
        raise ValueError("sequence_dim must be positive")
    base = torch.linspace(0.0, 1.0, steps=batch_size).unsqueeze(1)
    context = torch.cat(
        [base + offset for offset in torch.linspace(0.0, 0.3, steps=sequence_dim)],
        dim=1,
    )
    target = context + 0.15
    return context, target


def train_jepa(epochs: int = 160, latent_dim: int = 8, seed: int = 7) -> JEPATrainingResult:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if latent_dim < 1:
        raise ValueError("latent_dim must be positive")
    torch.manual_seed(seed)
    context, target = synthetic_sequence_batch()
    model = TinyJEPA(input_dim=context.shape[1], latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    mse = nn.MSELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        prediction, target_latent = model(context, target)
        variance_penalty = torch.relu(0.1 - prediction.std(dim=0)).mean()
        loss = mse(prediction, target_latent) + (0.1 * variance_penalty)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        prediction, target_latent = model(context, target)
        final_loss = mse(prediction, target_latent).item()
        variance_regularization = torch.relu(0.1 - prediction.std(dim=0)).mean().item()
        future_prediction, future_target = model(target, target + 0.15)
        rollout_error = mse(future_prediction, future_target).item()

    return JEPATrainingResult(
        final_loss=round(final_loss, 6),
        variance_regularization=round(variance_regularization, 6),
        rollout_error=round(rollout_error, 6),
    )
