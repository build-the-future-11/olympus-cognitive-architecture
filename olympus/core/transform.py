from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class TransformMetrics:
    reconstruction_loss: float
    sparsity_score: float
    independence_score: float
    predictive_utility: float


class LearnedTransformNetwork(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        latent = torch.tanh(self.encoder(batch))
        reconstruction = self.decoder(latent)
        return latent, reconstruction


def train_learned_transform(
    signals: Tensor,
    latent_dim: int = 3,
    epochs: int = 120,
    learning_rate: float = 0.03,
    seed: int = 7,
) -> TransformMetrics:
    if signals.ndim != 2 or signals.shape[0] < 2 or signals.shape[1] < 1:
        raise ValueError("signals must be a two-dimensional batch with at least two rows")
    if latent_dim < 1 or epochs < 1 or learning_rate <= 0:
        raise ValueError("latent_dim, epochs, and learning_rate must be positive")
    torch.manual_seed(seed)
    model = LearnedTransformNetwork(signals.shape[1], latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        latent, reconstruction = model(signals)
        reconstruction_loss = loss_fn(reconstruction, signals)
        sparsity_penalty = latent.abs().mean()
        decorrelation_penalty = torch.cov(latent.T).abs().mean()
        loss = reconstruction_loss + (0.02 * sparsity_penalty) + (0.01 * decorrelation_penalty)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        latent, reconstruction = model(signals)
        reconstruction_loss = loss_fn(reconstruction, signals).item()
        sparsity_score = float(1.0 - torch.clamp(latent.abs().mean(), 0.0, 1.0).item())
        independence_score = float(
            1.0 - torch.clamp(torch.cov(latent.T).abs().mean(), 0.0, 1.0).item()
        )
        predictive_utility = float(1.0 - min(1.0, reconstruction_loss))
    return TransformMetrics(
        reconstruction_loss=round(reconstruction_loss, 6),
        sparsity_score=round(sparsity_score, 6),
        independence_score=round(independence_score, 6),
        predictive_utility=round(predictive_utility, 6),
    )
