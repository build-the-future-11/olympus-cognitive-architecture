"""Olympus cognitive architecture."""

from __future__ import annotations

from typing import Any

__version__ = "0.2.0"

__all__ = ["__version__", "app"]


def __getattr__(name: str) -> Any:
    if name == "app":
        from olympus.api import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
