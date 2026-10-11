"""Differentiable encoder-to-residual-vector-quantization pathway."""

from __future__ import annotations

from typing import NamedTuple, cast

import torch
from pydantic import Field
from torch import Tensor, nn

from olympus.core.schemas import StrictModel


class EncoderQuantizationConfig(StrictModel):
    input_dim: int = Field(ge=1)
    latent_dim: int = Field(ge=2)
    hidden_dim: int = Field(default=256, ge=2)
    codebook_size: int = Field(default=256, ge=2, le=65_536)
    residual_codebooks: int = Field(default=2, ge=1, le=16)
    commitment_weight: float = Field(default=0.25, ge=0.0, le=10.0)


class QuantizationOutput(NamedTuple):
    quantized: Tensor
    indices: Tensor
    loss: Tensor
    commitment_loss: Tensor
    codebook_loss: Tensor
    quantization_error: Tensor
    utilization: Tensor


class ResidualVectorQuantizer(nn.Module):
    """Residual VQ with masked tokens and a straight-through encoder gradient."""

    def __init__(self, config: EncoderQuantizationConfig) -> None:
        super().__init__()
        self.config = config
        self.codebooks = nn.Parameter(
            torch.empty(config.residual_codebooks, config.codebook_size, config.latent_dim)
        )
        nn.init.uniform_(
            self.codebooks,
            -1.0 / config.codebook_size,
            1.0 / config.codebook_size,
        )

    def forward(self, latent: Tensor, mask: Tensor | None = None) -> QuantizationOutput:
        if latent.ndim < 2 or latent.shape[-1] != self.config.latent_dim:
            raise ValueError(f"expected [..., {self.config.latent_dim}] latent tensor")
        token_shape = latent.shape[:-1]
        flat = latent.reshape(-1, self.config.latent_dim)
        if mask is None:
            active = torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
        else:
            if tuple(mask.shape) != tuple(token_shape):
                raise ValueError("mask shape must equal the latent token dimensions")
            active = mask.to(device=flat.device, dtype=torch.bool).reshape(-1)
        if not bool(active.any()):
            raise ValueError("at least one latent token must be active")

        residual = flat
        quantized_sum = torch.zeros_like(flat)
        indices: list[Tensor] = []
        commitment_terms: list[Tensor] = []
        codebook_terms: list[Tensor] = []
        utilization_terms: list[Tensor] = []

        for codebook in self.codebooks:
            active_residual = residual[active]
            distances = (
                active_residual.square().sum(dim=1, keepdim=True)
                - 2.0 * active_residual @ codebook.transpose(0, 1)
                + codebook.square().sum(dim=1).unsqueeze(0)
            )
            active_indices = distances.argmin(dim=1)
            selected_active = codebook[active_indices]
            selected = torch.zeros_like(flat)
            selected[active] = selected_active
            full_indices = torch.full((flat.shape[0],), -1, dtype=torch.long, device=flat.device)
            full_indices[active] = active_indices
            indices.append(full_indices)
            commitment_terms.append(
                nn.functional.mse_loss(active_residual, selected_active.detach())
            )
            codebook_terms.append(nn.functional.mse_loss(selected_active, active_residual.detach()))
            utilization_terms.append(
                active_indices.unique().numel()
                / torch.tensor(
                    self.config.codebook_size,
                    dtype=flat.dtype,
                    device=flat.device,
                )
            )
            quantized_sum = quantized_sum + selected
            residual = residual - selected.detach()

        commitment = torch.stack(commitment_terms).mean()
        codebook_loss = torch.stack(codebook_terms).mean()
        loss = codebook_loss + self.config.commitment_weight * commitment
        straight_through = flat + (quantized_sum - flat).detach()
        straight_through = torch.where(active.unsqueeze(1), straight_through, flat)
        error = nn.functional.mse_loss(quantized_sum[active], flat[active])
        return QuantizationOutput(
            quantized=straight_through.reshape(*token_shape, self.config.latent_dim),
            indices=torch.stack(indices, dim=-1).reshape(
                *token_shape, self.config.residual_codebooks
            ),
            loss=loss,
            commitment_loss=commitment,
            codebook_loss=codebook_loss,
            quantization_error=error,
            utilization=torch.stack(utilization_terms).mean(),
        )


class QuantizedEncoder(nn.Module):
    """Trainable encoder whose continuous states pass through residual VQ."""

    def __init__(self, config: EncoderQuantizationConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )
        self.quantizer = ResidualVectorQuantizer(config)

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> QuantizationOutput:
        if inputs.ndim < 2 or inputs.shape[-1] != self.config.input_dim:
            raise ValueError(f"expected [..., {self.config.input_dim}] encoder input")
        return cast(QuantizationOutput, self.quantizer(self.encoder(inputs), mask=mask))


class QuantizedAutoencoderOutput(NamedTuple):
    reconstruction: Tensor
    quantization: QuantizationOutput


class QuantizedAutoencoder(nn.Module):
    """Rate-distortion training wrapper around the encoder-quantization pathway."""

    def __init__(self, config: EncoderQuantizationConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = QuantizedEncoder(config)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> QuantizedAutoencoderOutput:
        quantization = self.encoder(inputs, mask=mask)
        reconstruction = cast(Tensor, self.decoder(quantization.quantized))
        return QuantizedAutoencoderOutput(
            reconstruction=reconstruction,
            quantization=quantization,
        )
