"""JEPA-based super-resolution research components."""

from olympus.upscaler.jepa import (
    JEPAUpscaler,
    JEPAUpscalerConfig,
    JEPAUpscalerLoss,
    JEPAUpscalerOutput,
    jepa_upscaler_loss,
    upscale_self_ensemble,
    upscale_tiled,
)
from olympus.upscaler.training import load_jepa_upscaler_checkpoint

__all__ = [
    "JEPAUpscaler",
    "JEPAUpscalerConfig",
    "JEPAUpscalerLoss",
    "JEPAUpscalerOutput",
    "jepa_upscaler_loss",
    "load_jepa_upscaler_checkpoint",
    "upscale_self_ensemble",
    "upscale_tiled",
]
