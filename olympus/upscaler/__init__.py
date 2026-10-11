"""JEPA-based super-resolution research components."""

from olympus.upscaler.jepa import (
    JEPAUpscaler,
    JEPAUpscalerConfig,
    JEPAUpscalerLoss,
    JEPAUpscalerOutput,
    jepa_upscaler_loss,
)

__all__ = [
    "JEPAUpscaler",
    "JEPAUpscalerConfig",
    "JEPAUpscalerLoss",
    "JEPAUpscalerOutput",
    "jepa_upscaler_loss",
]
