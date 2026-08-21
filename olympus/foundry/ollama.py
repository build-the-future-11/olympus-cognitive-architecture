from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from olympus.foundry.schemas import GenerationResult


@dataclass(slots=True)
class OllamaClient:
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 120.0
    transport: httpx.BaseTransport | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("Ollama base_url must be an HTTP or HTTPS URL")
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
            raise ValueError("Ollama returned a non-object response")
        return payload

    def list_models(self) -> list[str]:
        payload = self._request("GET", "/api/tags")
        models = payload.get("models")
        if not isinstance(models, list):
            raise ValueError("Ollama model list is malformed")
        names: list[str] = []
        for model in models:
            if not isinstance(model, dict) or not isinstance(model.get("name"), str):
                raise ValueError("Ollama model entry is malformed")
            names.append(model["name"])
        return names

    def health(self) -> dict[str, object]:
        models = self.list_models()
        return {"status": "ok", "provider": "ollama", "models": models}

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 256,
        context_tokens: int = 2_048,
        temperature: float = 0.0,
    ) -> GenerationResult:
        if not model.strip():
            raise ValueError("model is required")
        if not prompt.strip():
            raise ValueError("prompt is required")
        if not 1 <= max_tokens <= 4_096:
            raise ValueError("max_tokens must be between 1 and 4096")
        if not 256 <= context_tokens <= 131_072:
            raise ValueError("context_tokens must be between 256 and 131072")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = self._request(
            "POST",
            "/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {
                    "num_predict": max_tokens,
                    "num_ctx": context_tokens,
                    "temperature": temperature,
                },
            },
        )
        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Ollama chat response is malformed")
        content = message["content"]
        return GenerationResult(
            model=model,
            content=content,
            finish_reason="stop" if payload.get("done") is True else "length",
            prompt_characters=len(prompt),
            generated_characters=len(content),
            evidence={
                "provider": "ollama",
                "done_reason": payload.get("done_reason"),
                "total_duration_ns": payload.get("total_duration"),
                "eval_count": payload.get("eval_count"),
                "load_duration_ns": payload.get("load_duration"),
            },
        )
