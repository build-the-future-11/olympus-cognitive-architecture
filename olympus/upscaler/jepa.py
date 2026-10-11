"""Geometry-conditioned joint-embedding predictive super-resolution model."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import cast

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn

from olympus.core.schemas import StrictModel


class JEPAUpscalerConfig(StrictModel):
    scale: int = Field(default=2, ge=2, le=4)
    input_channels: int = Field(default=3, ge=1, le=4)
    feature_channels: int = Field(default=48, ge=8, le=512)
    residual_blocks: int = Field(default=4, ge=1, le=64)
    target_momentum: float = Field(default=0.996, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def supported_scale(self) -> JEPAUpscalerConfig:
        if self.scale not in {2, 3, 4}:
            raise ValueError("scale must be exactly 2, 3, or 4")
        return self


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.gain.tanh() * cast(Tensor, self.body(inputs))


class _ImageEncoder(nn.Module):
    def __init__(self, input_channels: int, feature_channels: int, blocks: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels, 3, padding=1),
            *(_ResidualBlock(feature_channels) for _ in range(blocks)),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.layers(inputs))


@dataclass(slots=True)
class JEPAUpscalerOutput:
    upscaled: Tensor
    predicted_latent: Tensor
    target_latent: Tensor | None
    geometry: Tensor


@dataclass(slots=True)
class JEPAUpscalerLoss:
    total: Tensor
    reconstruction: Tensor
    edge: Tensor
    joint_embedding: Tensor
    latent_geometry: Tensor
    variance: Tensor


def _coordinate_features(inputs: Tensor, timestamp: Tensor | None) -> Tensor:
    batch, _channels, height, width = inputs.shape
    dtype = inputs.dtype
    device = inputs.device
    y = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
    x = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(grid_x.square() + grid_y.square()).clamp_max(1.5)
    if timestamp is None:
        time = torch.zeros((batch, 1, 1), dtype=dtype, device=device)
    else:
        if timestamp.shape != (batch,):
            raise ValueError("timestamp must have shape [batch]")
        time = timestamp.to(device=device, dtype=dtype).tanh().reshape(batch, 1, 1)
    spatial = torch.stack((grid_x, grid_y, radius), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
    temporal = time.unsqueeze(1).expand(-1, 1, height, width)
    return torch.cat((spatial, temporal), dim=1)


class JEPAUpscaler(nn.Module):
    """Predict target embeddings before reconstructing a bicubic residual."""

    def __init__(self, config: JEPAUpscalerConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = _ImageEncoder(
            config.input_channels, config.feature_channels, config.residual_blocks
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Conv2d(config.feature_channels + 4, config.feature_channels, 3, padding=1),
            nn.GELU(),
            _ResidualBlock(config.feature_channels),
            nn.Conv2d(config.feature_channels, config.feature_channels, 1),
        )
        self.reconstruction = nn.Conv2d(
            config.feature_channels,
            config.input_channels * config.scale * config.scale,
            3,
            padding=1,
        )
        nn.init.zeros_(self.reconstruction.weight)
        if self.reconstruction.bias is not None:
            nn.init.zeros_(self.reconstruction.bias)

    def forward(
        self,
        low_resolution: Tensor,
        target_high_resolution: Tensor | None = None,
        *,
        timestamp: Tensor | None = None,
    ) -> JEPAUpscalerOutput:
        if low_resolution.ndim != 4 or low_resolution.shape[1] != self.config.input_channels:
            raise ValueError(f"expected NCHW input with {self.config.input_channels} channels")
        geometry = _coordinate_features(low_resolution, timestamp)
        context = self.context_encoder(low_resolution)
        predicted = cast(Tensor, self.predictor(torch.cat((context, geometry), dim=1)))
        residual = nn.functional.pixel_shuffle(self.reconstruction(predicted), self.config.scale)
        baseline = nn.functional.interpolate(
            low_resolution,
            scale_factor=self.config.scale,
            mode="bicubic",
            align_corners=False,
        )
        target_latent = None
        if target_high_resolution is not None:
            expected = (
                low_resolution.shape[-2] * self.config.scale,
                low_resolution.shape[-1] * self.config.scale,
            )
            if (
                target_high_resolution.ndim != 4
                or target_high_resolution.shape[:2] != low_resolution.shape[:2]
                or target_high_resolution.shape[-2:] != expected
            ):
                raise ValueError("target resolution must exactly match the configured scale")
            target_low_resolution = nn.functional.interpolate(
                target_high_resolution,
                size=low_resolution.shape[-2:],
                mode="area",
            )
            with torch.no_grad():
                target_latent = self.target_encoder(target_low_resolution)
        return JEPAUpscalerOutput(
            upscaled=(baseline + residual).clamp(0.0, 1.0),
            predicted_latent=predicted,
            target_latent=target_latent,
            geometry=geometry,
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        value = self.config.target_momentum if momentum is None else momentum
        if not 0.0 <= value < 1.0:
            raise ValueError("target momentum must be in [0, 1)")
        for target, online in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters(), strict=True
        ):
            target.mul_(value).add_(online, alpha=1.0 - value)


def _spatial_gradients(values: Tensor) -> tuple[Tensor, Tensor]:
    return values[..., :, 1:] - values[..., :, :-1], values[..., 1:, :] - values[..., :-1, :]


def jepa_upscaler_loss(
    output: JEPAUpscalerOutput,
    target: Tensor,
    *,
    reconstruction_weight: float = 1.0,
    edge_weight: float = 0.1,
    embedding_weight: float = 0.25,
    geometry_weight: float = 0.05,
    variance_weight: float = 0.02,
) -> JEPAUpscalerLoss:
    """Combine pixel fidelity with non-collapsing predictive representation losses."""

    if output.target_latent is None:
        raise ValueError("a target image is required to compute JEPA loss")
    if output.upscaled.shape != target.shape:
        raise ValueError("prediction and target image shapes must match")
    reconstruction = nn.functional.l1_loss(output.upscaled, target)
    pred_x, pred_y = _spatial_gradients(output.upscaled)
    target_x, target_y = _spatial_gradients(target)
    edge = nn.functional.l1_loss(pred_x, target_x) + nn.functional.l1_loss(pred_y, target_y)
    predicted_normalized = nn.functional.normalize(output.predicted_latent, dim=1)
    target_normalized = nn.functional.normalize(output.target_latent, dim=1)
    joint_embedding = (1.0 - (predicted_normalized * target_normalized).sum(dim=1)).mean()
    latent_pred_x, latent_pred_y = _spatial_gradients(output.predicted_latent)
    latent_target_x, latent_target_y = _spatial_gradients(output.target_latent)
    latent_geometry = nn.functional.smooth_l1_loss(
        latent_pred_x, latent_target_x
    ) + nn.functional.smooth_l1_loss(latent_pred_y, latent_target_y)
    channel_std = output.predicted_latent.flatten(0, 1).std(dim=(-2, -1), unbiased=False)
    variance = torch.relu(0.1 - channel_std).mean()
    total = (
        reconstruction_weight * reconstruction
        + edge_weight * edge
        + embedding_weight * joint_embedding
        + geometry_weight * latent_geometry
        + variance_weight * variance
    )
    return JEPAUpscalerLoss(
        total=total,
        reconstruction=reconstruction,
        edge=edge,
        joint_embedding=joint_embedding,
        latent_geometry=latent_geometry,
        variance=variance,
    )
