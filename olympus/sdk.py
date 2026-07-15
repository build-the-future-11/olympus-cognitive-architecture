from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import httpx


@dataclass(slots=True)
class OlympusSDK:
    base_url: str

    def health(self) -> dict[str, str]:
        response = httpx.get(f"{self.base_url}/health", timeout=5)
        return cast(dict[str, str], response.json())

    def compile_behavior(self, behavior: str) -> dict[str, object]:
        response = httpx.post(
            f"{self.base_url}/forge/compile",
            json={"behavior": behavior},
            timeout=10,
        )
        return cast(dict[str, object], response.json())
