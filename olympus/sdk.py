from __future__ import annotations

import ipaddress
import math
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx


@dataclass(slots=True)
class OlympusSDK:
    base_url: str
    timeout_seconds: float = 10.0
    api_token: str | None = field(default=None, repr=False)
    transport: httpx.BaseTransport | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        try:
            parsed = urlsplit(self.base_url)
            _ = parsed.port
        except ValueError as error:
            raise ValueError(
                "base_url must be a credential-free HTTP or HTTPS origin"
            ) from error
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be a credential-free HTTP or HTTPS origin")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if self.api_token is not None:
            if (
                not self.api_token
                or len(self.api_token) > 4_096
                or self.api_token.strip() != self.api_token
                or any(character.isspace() for character in self.api_token)
            ):
                raise ValueError("api_token must be a non-empty token without whitespace")
            try:
                is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
            except ValueError:
                is_loopback = parsed.hostname.casefold().rstrip(".") == "localhost"
            if parsed.scheme != "https" and not is_loopback:
                raise ValueError("api_token requires HTTPS for non-loopback APIs")

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        with httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
            trust_env=False,
            headers=(
                {"Authorization": f"Bearer {self.api_token}"}
                if self.api_token is not None
                else None
            ),
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
        if not model.strip() or len(model) > 200:
            raise ValueError("model must contain between 1 and 200 characters")
        if not prompt.strip() or len(prompt) > 50_000:
            raise ValueError("prompt must contain between 1 and 50000 characters")
        if not 1 <= max_tokens <= 4_096:
            raise ValueError("max_tokens must be between 1 and 4096")
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be finite and between 0 and 2")
        if seed is not None and not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be between 0 and 2^32-1")
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
