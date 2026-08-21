from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass(slots=True)
class OlympusSDK:
    base_url: str
    timeout_seconds: float = 10.0
    transport: httpx.BaseTransport | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP or HTTPS URL")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        with httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Olympus API returned a non-object JSON response")
        return payload

    def health(self) -> dict[str, str]:
        payload = self._request("GET", "/health")
        string_fields = all(
            isinstance(key, str) and isinstance(value, str) for key, value in payload.items()
        )
        if not string_fields:
            raise ValueError("Olympus health response must contain only string fields")
        return payload

    def compile_behavior(self, behavior: str) -> dict[str, object]:
        return self._request(
            "POST",
            "/forge/compile",
            json={"behavior": behavior},
        )

    def foundry_status(self) -> dict[str, object]:
        return self._request("GET", "/foundry/status")

    def list_models(self) -> list[dict[str, object]]:
        payload = self._request("GET", "/v1/models")
        models = payload.get("data")
        if not isinstance(models, list) or any(not isinstance(model, dict) for model in models):
            raise ValueError("Olympus model registry response is malformed")
        return models

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int = 160,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> dict[str, object]:
        if not model.strip():
            raise ValueError("model is required")
        if not prompt.strip():
            raise ValueError("prompt is required")
        return self._request(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed,
                "stream": False,
            },
        )
