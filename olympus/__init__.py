"""Olympus cognitive architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

    from olympus.api import app

__version__ = "0.2.0"


def __getattr__(name: str) -> FastAPI:
    # Importing a data/worker module must not initialize the entire web/model stack.
    if name == "app":
        from olympus.api import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["__version__", "app"]
