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
    predictor_dilations: tuple[int, ...] = (1, 2, 4)

    @model_validator(mode="after")
    def supported_scale(self) -> JEPAUpscalerConfig:
        if self.scale not in {2, 3, 4}:
            raise ValueError("scale must be exactly 2, 3, or 4")
        if not self.predictor_dilations or any(
            dilation < 1 or dilation > 16 for dilation in self.predictor_dilations
        ):
            raise ValueError("predictor dilations must contain values from 1 through 16")
        if len(set(self.predictor_dilations)) != len(self.predictor_dilations):
            raise ValueError("predictor dilations must be unique")
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


class _MultiScalePredictor(nn.Module):
    def __init__(self, channels: int, geometry_channels: int, dilations: tuple[int, ...]) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv2d(channels + geometry_channels, channels, 1),
            nn.GELU(),
        )
        self.branches = nn.ModuleList(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
            for dilation in dilations
        )
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels * len(dilations), channels, 1),
            _ResidualBlock(channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        projected = cast(Tensor, self.input_projection(inputs))
        features = torch.cat([branch(projected) for branch in self.branches], dim=1)
        return cast(Tensor, self.output(features))


@dataclass(slots=True)
class JEPAUpscalerOutput:
    upscaled: Tensor
    residual: Tensor
    predicted_latent: Tensor
    target_latent: Tensor | None
    geometry: Tensor
    context_mask: Tensor


@dataclass(slots=True)
class JEPAUpscalerLoss:
    total: Tensor
    reconstruction: Tensor
    edge: Tensor
    joint_embedding: Tensor
    latent_geometry: Tensor
    variance: Tensor
    frequency: Tensor
    structure: Tensor
    pyramid: Tensor
    spectral_coherence: Tensor


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
        time_values = timestamp.to(device=device, dtype=dtype)
        if not bool(torch.isfinite(time_values).all()):
            raise ValueError("timestamp values must be finite")
        time = time_values.tanh().reshape(batch, 1, 1)
    spatial = torch.stack((grid_x, grid_y, radius), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
    temporal = time.unsqueeze(1).expand(-1, 1, height, width)
    return torch.cat((spatial, temporal), dim=1)


class JEPAUpscaler(nn.Module):
    """Predict masked high-resolution detail embeddings above a bicubic baseline."""

    def __init__(self, config: JEPAUpscalerConfig) -> None:
        super().__init__()
        self.config = config
        patch_channels = config.input_channels * config.scale * config.scale
        self.context_encoder = _ImageEncoder(
            patch_channels, config.feature_channels, config.residual_blocks
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.mask_token = nn.Parameter(torch.zeros(1, config.feature_channels, 1, 1))
        self.predictor = _MultiScalePredictor(
            config.feature_channels,
            geometry_channels=4,
            dilations=config.predictor_dilations,
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
        context_mask: Tensor | None = None,
    ) -> JEPAUpscalerOutput:
        if low_resolution.ndim != 4 or low_resolution.shape[1] != self.config.input_channels:
            raise ValueError(f"expected NCHW input with {self.config.input_channels} channels")
        if not low_resolution.is_floating_point():
            raise ValueError("low-resolution input must use a floating-point dtype")
        if not bool(torch.isfinite(low_resolution).all()) or bool(
            ((low_resolution < 0.0) | (low_resolution > 1.0)).any()
        ):
            raise ValueError("low-resolution input must contain finite values in [0, 1]")
        if min(low_resolution.shape[-2:]) < 2:
            raise ValueError("low-resolution spatial dimensions must both be at least two")
        geometry = _coordinate_features(low_resolution, timestamp)
        baseline = nn.functional.interpolate(
            low_resolution,
            scale_factor=self.config.scale,
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
        context_patches = nn.functional.pixel_unshuffle(baseline, self.config.scale)
        context = self.context_encoder(context_patches)
        if context_mask is None:
            mask = torch.zeros(
                low_resolution.shape[0],
                *low_resolution.shape[-2:],
                dtype=torch.bool,
                device=low_resolution.device,
            )
        else:
            if context_mask.shape != (
                low_resolution.shape[0],
                low_resolution.shape[2],
                low_resolution.shape[3],
            ):
                raise ValueError("context mask must have shape [batch, height, width]")
            mask = context_mask.to(device=low_resolution.device, dtype=torch.bool)
            if bool(mask.flatten(1).all(dim=1).any()):
                raise ValueError("context mask must leave at least one visible position")
        context = torch.where(mask.unsqueeze(1), self.mask_token.to(dtype=context.dtype), context)
        predicted = cast(Tensor, self.predictor(torch.cat((context, geometry), dim=1)))
        residual = nn.functional.pixel_shuffle(self.reconstruction(predicted), self.config.scale)
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
            if not target_high_resolution.is_floating_point() or not bool(
                torch.isfinite(target_high_resolution).all()
            ):
                raise ValueError("target image must contain finite floating-point values")
            if bool(((target_high_resolution < 0.0) | (target_high_resolution > 1.0)).any()):
                raise ValueError("target image values must fall in [0, 1]")
            with torch.no_grad():
                target_patches = nn.functional.pixel_unshuffle(
                    target_high_resolution, self.config.scale
                )
                target_latent = self.target_encoder(target_patches)
        return JEPAUpscalerOutput(
            upscaled=(baseline + residual).clamp(0.0, 1.0),
            residual=residual,
            predicted_latent=predicted,
            target_latent=target_latent,
            geometry=geometry,
            context_mask=mask,
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
    frequency_weight: float = 0.05,
    structure_weight: float = 0.1,
    pyramid_weight: float = 0.1,
    spectral_coherence_weight: float = 0.05,
) -> JEPAUpscalerLoss:
    """Combine pixel fidelity with non-collapsing predictive representation losses."""

    if output.target_latent is None:
        raise ValueError("a target image is required to compute JEPA loss")
    if output.upscaled.shape != target.shape:
        raise ValueError("prediction and target image shapes must match")
    difference = output.upscaled - target
    reconstruction = torch.sqrt(difference.square() + 1e-6).mean()
    pred_x, pred_y = _spatial_gradients(output.upscaled)
    target_x, target_y = _spatial_gradients(target)
    edge = nn.functional.l1_loss(pred_x, target_x) + nn.functional.l1_loss(pred_y, target_y)
    predicted_normalized = nn.functional.normalize(output.predicted_latent, dim=1)
    target_normalized = nn.functional.normalize(output.target_latent, dim=1)
    embedding_distance = 1.0 - (predicted_normalized * target_normalized).sum(dim=1)
    if bool(output.context_mask.any()):
        joint_embedding = embedding_distance[output.context_mask].mean()
    else:
        joint_embedding = embedding_distance.mean()
    latent_pred_x, latent_pred_y = _spatial_gradients(output.predicted_latent)
    latent_target_x, latent_target_y = _spatial_gradients(output.target_latent)
    latent_geometry = nn.functional.smooth_l1_loss(
        latent_pred_x, latent_target_x
    ) + nn.functional.smooth_l1_loss(latent_pred_y, latent_target_y)
    channel_samples = output.predicted_latent.permute(1, 0, 2, 3).flatten(1)
    channel_std = channel_samples.std(dim=1, unbiased=False)
    variance = torch.relu(0.1 - channel_std).mean()
    prediction_spectrum = torch.fft.rfft2(output.upscaled.float(), norm="ortho").abs()
    target_spectrum = torch.fft.rfft2(target.float(), norm="ortho").abs()
    frequency = nn.functional.l1_loss(
        torch.log1p(prediction_spectrum), torch.log1p(target_spectrum)
    ).to(dtype=output.upscaled.dtype)
    structure = 1.0 - _structural_similarity(output.upscaled, target)
    pyramid = _pyramid_consistency(output.upscaled, target)
    spectral_coherence = _spectral_coherence(output.upscaled, target)
    total = (
        reconstruction_weight * reconstruction
        + edge_weight * edge
        + embedding_weight * joint_embedding
        + geometry_weight * latent_geometry
        + variance_weight * variance
        + frequency_weight * frequency
        + structure_weight * structure
        + pyramid_weight * pyramid
        + spectral_coherence_weight * spectral_coherence
    )
    return JEPAUpscalerLoss(
        total=total,
        reconstruction=reconstruction,
        edge=edge,
        joint_embedding=joint_embedding,
        latent_geometry=latent_geometry,
        variance=variance,
        frequency=frequency,
        structure=structure,
        pyramid=pyramid,
        spectral_coherence=spectral_coherence,
    )


def _pyramid_consistency(prediction: Tensor, target: Tensor) -> Tensor:
    losses: list[Tensor] = []
    for factor in (2, 4):
        if min(prediction.shape[-2:]) < factor:
            continue
        predicted_scale = nn.functional.avg_pool2d(prediction, factor, stride=factor)
        target_scale = nn.functional.avg_pool2d(target, factor, stride=factor)
        losses.append(torch.sqrt((predicted_scale - target_scale).square() + 1e-6).mean())
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def _spectral_coherence(prediction: Tensor, target: Tensor) -> Tensor:
    prediction_spectrum = torch.fft.rfft2(prediction.float(), norm="ortho")
    target_spectrum = torch.fft.rfft2(target.float(), norm="ortho")
    denominator = prediction_spectrum.abs() * target_spectrum.abs()
    coherence = (
        (prediction_spectrum * target_spectrum.conj()).real / denominator.clamp_min(1e-8)
    ).clamp(-1.0, 1.0)
    weights = torch.log1p(target_spectrum.abs())
    loss = ((1.0 - coherence) * weights).sum() / weights.sum().clamp_min(1e-8)
    return cast(Tensor, loss.to(dtype=prediction.dtype))


def _structural_similarity(prediction: Tensor, target: Tensor) -> Tensor:
    """Differentiable local SSIM for finite images normalized to [0, 1]."""

    mean_prediction = nn.functional.avg_pool2d(prediction, 3, stride=1, padding=1)
    mean_target = nn.functional.avg_pool2d(target, 3, stride=1, padding=1)
    variance_prediction = nn.functional.avg_pool2d(prediction.square(), 3, 1, 1)
    variance_prediction = (variance_prediction - mean_prediction.square()).clamp_min(0.0)
    variance_target = nn.functional.avg_pool2d(target.square(), 3, 1, 1)
    variance_target = (variance_target - mean_target.square()).clamp_min(0.0)
    covariance = nn.functional.avg_pool2d(prediction * target, 3, 1, 1)
    covariance = covariance - mean_prediction * mean_target
    numerator = (2.0 * mean_prediction * mean_target + 0.01**2) * (2.0 * covariance + 0.03**2)
    denominator = (mean_prediction.square() + mean_target.square() + 0.01**2) * (
        variance_prediction + variance_target + 0.03**2
    )
    return (numerator / denominator.clamp_min(1e-12)).mean().clamp(-1.0, 1.0)


def upscale_tiled(
    model: JEPAUpscaler,
    low_resolution: Tensor,
    *,
    tile_size: int = 128,
    overlap: int = 16,
    timestamp: Tensor | None = None,
) -> Tensor:
    """Upscale large inputs using seam-suppressing overlap blends and bounded memory."""

    if tile_size < 2:
        raise ValueError("tile size must be at least two")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile overlap must fall in [0, tile_size)")
    height, width = low_resolution.shape[-2:]
    if height <= tile_size and width <= tile_size:
        with torch.inference_mode():
            return cast(Tensor, model(low_resolution, timestamp=timestamp).upscaled)
    stride = tile_size - overlap
    y_positions = _tile_positions(height, tile_size, stride)
    x_positions = _tile_positions(width, tile_size, stride)
    scale = model.config.scale
    output = torch.zeros(
        low_resolution.shape[0],
        model.config.input_channels,
        height * scale,
        width * scale,
        dtype=low_resolution.dtype,
        device=low_resolution.device,
    )
    counts = torch.zeros_like(output)
    with torch.inference_mode():
        for y in y_positions:
            for x in x_positions:
                tile = low_resolution[..., y : y + tile_size, x : x + tile_size]
                prediction = model(tile, timestamp=timestamp).upscaled
                blend = _tile_blend_weights(
                    prediction,
                    overlap=overlap * scale,
                    top=y > 0,
                    bottom=y + tile.shape[-2] < height,
                    left=x > 0,
                    right=x + tile.shape[-1] < width,
                ).expand_as(prediction)
                output[
                    ...,
                    y * scale : (y + tile.shape[-2]) * scale,
                    x * scale : (x + tile.shape[-1]) * scale,
                ].add_(prediction * blend)
                counts[
                    ...,
                    y * scale : (y + tile.shape[-2]) * scale,
                    x * scale : (x + tile.shape[-1]) * scale,
                ].add_(blend)
    if bool((counts == 0).any()):
        raise RuntimeError("tiled upscaling left uncovered output pixels")
    return output / counts


def _tile_blend_weights(
    tile: Tensor,
    *,
    overlap: int,
    top: bool,
    bottom: bool,
    left: bool,
    right: bool,
) -> Tensor:
    """Create separable squared-sine ramps without attenuating exterior image edges."""

    height, width = tile.shape[-2:]
    y_weights = torch.ones(height, dtype=tile.dtype, device=tile.device)
    x_weights = torch.ones(width, dtype=tile.dtype, device=tile.device)
    y_blend = min(overlap, height)
    x_blend = min(overlap, width)
    if y_blend:
        ramp = (
            torch.linspace(0.0, torch.pi / 2.0, y_blend, dtype=tile.dtype, device=tile.device)
            .sin()
            .square()
            .clamp_min(1e-3)
        )
        if top:
            y_weights[:y_blend] = ramp
        if bottom:
            y_weights[-y_blend:] = ramp.flip(0)
    if x_blend:
        ramp = (
            torch.linspace(0.0, torch.pi / 2.0, x_blend, dtype=tile.dtype, device=tile.device)
            .sin()
            .square()
            .clamp_min(1e-3)
        )
        if left:
            x_weights[:x_blend] = ramp
        if right:
            x_weights[-x_blend:] = ramp.flip(0)
    return y_weights[:, None].mul(x_weights[None, :]).unsqueeze(0).unsqueeze(0)


def upscale_self_ensemble(
    model: JEPAUpscaler,
    low_resolution: Tensor,
    *,
    timestamp: Tensor | None = None,
) -> Tensor:
    """Average the inverse-mapped predictions of the eight square symmetries."""

    predictions: list[Tensor] = []
    with torch.inference_mode():
        for rotation in range(4):
            rotated = torch.rot90(low_resolution, rotation, dims=(-2, -1))
            for reflected in (False, True):
                transformed = torch.flip(rotated, dims=(-1,)) if reflected else rotated
                prediction = model(transformed, timestamp=timestamp).upscaled
                if reflected:
                    prediction = torch.flip(prediction, dims=(-1,))
                predictions.append(torch.rot90(prediction, -rotation, dims=(-2, -1)))
    return torch.stack(predictions).mean(dim=0)


def _tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if positions[-1] != final:
        positions.append(final)
    return positions
