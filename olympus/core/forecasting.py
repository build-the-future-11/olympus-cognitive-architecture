"""Observable forecasting contract for the separately versioned prediction study.

The historical fixed-feature demo in prediction.py retains its original semantics.
This module makes no JEPA novelty or general model-capability claim.
"""

from __future__ import annotations

import copy
import math
from typing import cast

import torch
from torch import Tensor, nn


class ObservablePredictor(nn.Module):
    def __init__(self, state_dim: int = 2, width: int = 32, latent_dim: int = 8) -> None:
        super().__init__()
        if min(state_dim, width, latent_dim) < 1:
            raise ValueError("all model dimensions must be positive")
        self.state_dim = state_dim
        self.encoder: nn.Module = nn.Sequential(
            nn.Linear(state_dim, width), nn.Tanh(), nn.Linear(width, latent_dim)
        )
        self.predictor: nn.Module = nn.Sequential(
            nn.Linear(latent_dim, width), nn.Tanh(), nn.Linear(width, latent_dim)
        )
        self.decoder: nn.Module = nn.Sequential(
            nn.Linear(latent_dim, width), nn.Tanh(), nn.Linear(width, state_dim)
        )
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)

    def _check_state(self, inputs: Tensor) -> None:
        if inputs.ndim != 2 or inputs.shape[1] != self.state_dim or inputs.shape[0] < 1:
            raise ValueError("state must be a nonempty batch with the declared state dimension")
        if not inputs.is_floating_point() or not bool(torch.isfinite(inputs).all()):
            raise ValueError("state must contain finite floating-point values")

    def representations(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        self._check_state(inputs)
        current = cast(Tensor, self.encoder(inputs))
        future = current + self.predictor(current)
        return current, future

    def forward(self, inputs: Tensor) -> Tensor:
        _, future = self.representations(inputs)
        return cast(Tensor, self.decoder(future))

    def objective(
        self,
        inputs: Tensor,
        targets: Tensor,
        *,
        latent_weight: float,
        variance_weight: float,
        target_permutation: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        self._check_state(targets)
        if inputs.shape != targets.shape or inputs.shape[0] < 2:
            raise ValueError("training needs matching batches with at least two states")
        if any(not math.isfinite(v) or v < 0 for v in (latent_weight, variance_weight)):
            raise ValueError("loss weights must be finite and nonnegative")
        current, future = self.representations(inputs)
        state = (self.decoder(future) - targets).square().mean()
        reconstruction = (self.decoder(current) - inputs).square().mean()
        latent = current.new_zeros(())
        variance = current.new_zeros(())
        if latent_weight:
            with torch.no_grad():
                target_latent = self.target_encoder(targets)
                if target_permutation is not None:
                    expected = torch.arange(inputs.shape[0], device=inputs.device)
                    if (
                        target_permutation.dtype != torch.long
                        or target_permutation.shape != expected.shape
                        or not torch.equal(target_permutation.sort().values, expected)
                    ):
                        raise ValueError("target permutation must contain each batch index once")
                    target_latent = target_latent[target_permutation]
            latent = (future - target_latent).square().mean()
        if variance_weight:
            std = torch.sqrt(current.var(dim=0, unbiased=False) + 1e-4)
            variance = torch.relu(0.2 - std).mean()
        loss = state + 0.1 * reconstruction + latent_weight * latent + variance_weight * variance
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite forecasting loss")
        return loss, {
            "state": state,
            "reconstruction": reconstruction,
            "latent": latent,
            "variance": variance,
        }

    @torch.no_grad()
    def update_target(self, momentum: float = 0.99) -> None:
        if not math.isfinite(momentum) or not 0 <= momentum <= 1:
            raise ValueError("EMA momentum must be in [0,1]")
        for target, online in zip(
            self.target_encoder.parameters(), self.encoder.parameters(), strict=True
        ):
            target.mul_(momentum).add_(online, alpha=1 - momentum)

    @torch.no_grad()
    def rollout(self, initial: Tensor, horizon: int) -> Tensor:
        self._check_state(initial)
        if horizon < 1:
            raise ValueError("horizon must be positive")
        current = initial
        predictions: list[Tensor] = []
        for _ in range(horizon):
            current = self(current)
            predictions.append(current)
        return torch.stack(predictions, dim=1)

    def inference_copy(self) -> ForecastInference:
        return ForecastInference(self)


class ForecastInference(nn.Module):
    """Independent E/P/D export; contains neither target encoder nor optimizer."""

    def __init__(self, source: ObservablePredictor) -> None:
        super().__init__()
        self.state_dim = source.state_dim
        self.encoder = copy.deepcopy(source.encoder)
        self.predictor = copy.deepcopy(source.predictor)
        self.decoder = copy.deepcopy(source.decoder)
        self.requires_grad_(False).eval()

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.state_dim or inputs.shape[0] < 1:
            raise ValueError("invalid inference state shape")
        if not inputs.is_floating_point() or not bool(torch.isfinite(inputs).all()):
            raise ValueError("inference state must be finite floating point")
        latent = self.encoder(inputs)
        return cast(Tensor, self.decoder(latent + self.predictor(latent)))


@torch.no_grad()
def representation_diagnostics(model: ObservablePredictor, states: Tensor) -> dict[str, object]:
    model._check_state(states)
    latent = model.encoder(states).double()
    centered = latent - latent.mean(dim=0)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered / len(states)).clamp_min(0)
    total = float(eigenvalues.sum())
    if total > 0:
        probabilities = eigenvalues[eigenvalues > 0] / total
        effective_rank = float(torch.exp(-(probabilities * probabilities.log()).sum()))
    else:
        effective_rank = 0.0
    return {
        "coordinate_std": latent.std(dim=0, unbiased=False).tolist(),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "effective_rank": effective_rank,
    }
