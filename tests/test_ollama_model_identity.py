from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from olympus.api import app, get_ollama_client
from olympus.foundry.ollama import OllamaClient


def _client(requested_model: str, response_fields: dict[str, object]) -> OllamaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        assert json.loads(request.content)["model"] == requested_model
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "fixture response"},
                "done": True,
                "done_reason": "stop",
                **response_fields,
            },
        )

    return OllamaClient(transport=httpx.MockTransport(handler))


INVALID_MODEL_FIELDS: list[dict[str, object]] = [
    {},
    {"model": None},
    {"model": ""},
    {"model": " "},
    {"model": True},
    {"model": 1},
    {"model": []},
    {"model": {}},
    {"model": "different-model:latest"},
    {"model": "local-test-model:other-tag"},
    {"model": "another-namespace/local-test-model:latest"},
    {"model": " local-test-model:latest"},
]


@pytest.mark.parametrize("response_fields", INVALID_MODEL_FIELDS)
def test_generation_rejects_missing_malformed_or_different_model_identity(
    response_fields: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="Ollama chat response model"):
        _client("local-test-model", response_fields).generate(
            model="local-test-model", prompt="Verify the provider receipt"
        )


@pytest.mark.parametrize(
    ("requested_model", "provider_model"),
    [
        ("local-test-model", "local-test-model"),
        ("local-test-model", "local-test-model:latest"),
        ("local-test-model:latest", "local-test-model"),
        ("namespace/local-test-model:v1", "namespace/local-test-model:v1"),
        ("localhost:5000/team/model", "localhost:5000/team/model:latest"),
    ],
)
def test_generation_retains_requested_and_provider_model_identity(
    requested_model: str, provider_model: str
) -> None:
    result = _client(requested_model, {"model": provider_model}).generate(
        model=requested_model, prompt="Verify the provider receipt"
    )
    assert result.model == requested_model
    assert result.evidence["requested_model"] == requested_model
    assert result.evidence["provider_model"] == provider_model
    assert result.content == "fixture response"
    assert result.finish_reason == "stop"


@pytest.mark.parametrize("response_fields", INVALID_MODEL_FIELDS)
def test_api_returns_provider_error_without_relabeling_a_response(
    monkeypatch: pytest.MonkeyPatch,
    response_fields: dict[str, object],
) -> None:
    client = _client("local-test-model", response_fields)
    overrides = app.dependency_overrides.copy()
    overrides[get_ollama_client] = lambda: client
    monkeypatch.setattr(app, "dependency_overrides", overrides)

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as api:
            return await api.post(
                "/foundry/providers/ollama/generate",
                json={"model": "local-test-model", "prompt": "Verify the provider receipt"},
            )

    response = asyncio.run(request())
    assert response.status_code == 503
    assert response.json()["detail"].startswith("Ollama chat response model")
    assert "content" not in response.json()
    assert "model" not in response.json()


def test_generation_does_not_equate_distinct_registry_hosts() -> None:
    requested_model = "registry-a:5000/team/model:latest"
    with pytest.raises(ValueError, match="Ollama chat response model"):
        _client(requested_model, {"model": "registry-b:5000/team/model:latest"}).generate(
            model=requested_model, prompt="Verify the provider receipt"
        )
