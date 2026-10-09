from __future__ import annotations

import asyncio

import httpx
import pytest

from olympus.api import app, get_ollama_client
from olympus.foundry.ollama import OllamaClient


def _client(completion: dict[str, object]) -> OllamaClient:
    payload = {
        "message": {"role": "assistant", "content": "Partial answer: café"},
        "eval_count": 8,
        **completion,
    }
    return OllamaClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    )


@pytest.mark.parametrize(
    ("completion", "expected_reason"),
    [
        pytest.param({"done": True, "done_reason": "length"}, "length", id="token-limit"),
        pytest.param({"done": True, "done_reason": "stop"}, "stop", id="normal-stop"),
    ],
)
def test_generation_preserves_completed_response_reason(
    completion: dict[str, object], expected_reason: str
) -> None:
    result = _client(completion).generate(model="local-test-model", prompt="Explain café")

    assert result.finish_reason == expected_reason
    assert result.content == "Partial answer: café"
    assert result.prompt_characters == len("Explain café")
    assert result.generated_characters == len("Partial answer: café")
    assert result.evidence["done_reason"] == completion.get("done_reason")
    assert result.evidence["eval_count"] == 8
    assert result.model_dump(mode="json")["finish_reason"] == expected_reason


@pytest.mark.parametrize(
    "completion",
    [
        pytest.param({"done": True}, id="omitted-reason"),
        pytest.param({"done": True, "done_reason": ""}, id="empty-reason"),
    ],
)
def test_generation_rejects_responses_without_explicit_reason(
    completion: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="Ollama chat response has an unsupported done_reason"):
        _client(completion).generate(model="local-test-model", prompt="Explain café")


@pytest.mark.parametrize(
    "completion",
    [
        pytest.param({"done": False, "done_reason": "length"}, id="incomplete"),
        pytest.param({"done_reason": "stop"}, id="missing-completion-flag"),
        pytest.param({"done": None, "done_reason": "stop"}, id="null-completion-flag"),
        pytest.param({"done": 1, "done_reason": "stop"}, id="integer-completion-flag"),
        pytest.param({"done": "true", "done_reason": "stop"}, id="string-completion-flag"),
    ],
)
def test_generation_rejects_responses_without_explicit_completion(
    completion: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="Ollama chat response is not complete"):
        _client(completion).generate(model="local-test-model", prompt="Explain café")


@pytest.mark.parametrize(
    "reason",
    [
        pytest.param("load", id="model-loaded-without-generating"),
        pytest.param("unload", id="model-unloaded-without-generating"),
        pytest.param("unknown", id="unknown-reason"),
        pytest.param(None, id="null-reason"),
        pytest.param(True, id="boolean-reason"),
        pytest.param(1, id="integer-reason"),
        pytest.param([], id="list-reason"),
        pytest.param({}, id="object-reason"),
    ],
)
def test_generation_rejects_unsupported_completion_reasons(reason: object) -> None:
    with pytest.raises(ValueError, match="Ollama chat response has an unsupported done_reason"):
        _client({"done": True, "done_reason": reason}).generate(
            model="local-test-model", prompt="Explain café"
        )


@pytest.mark.parametrize(
    ("completion", "status", "body_key", "expected_value"),
    [
        pytest.param(
            {"done": True, "done_reason": "length"},
            200,
            "finish_reason",
            "length",
            id="token-limit-reaches-caller",
        ),
        pytest.param(
            {"done": True, "done_reason": "stop"},
            200,
            "finish_reason",
            "stop",
            id="normal-stop-reaches-caller",
        ),
        pytest.param(
            {"done": True},
            503,
            "detail",
            "Ollama chat response has an unsupported done_reason",
            id="omitted-reason-is-provider-error",
        ),
        pytest.param(
            {"done": True, "done_reason": ""},
            503,
            "detail",
            "Ollama chat response has an unsupported done_reason",
            id="empty-reason-is-provider-error",
        ),
        pytest.param(
            {"done": False},
            503,
            "detail",
            "Ollama chat response is not complete",
            id="partial-response-is-provider-error",
        ),
        pytest.param(
            {"done": True, "done_reason": "load"},
            503,
            "detail",
            "Ollama chat response has an unsupported done_reason",
            id="non-generation-is-provider-error",
        ),
    ],
)
def test_ollama_api_preserves_completion_or_reports_provider_error(
    monkeypatch: pytest.MonkeyPatch,
    completion: dict[str, object],
    status: int,
    body_key: str,
    expected_value: str,
) -> None:
    client = _client(completion)
    overrides = app.dependency_overrides.copy()
    overrides[get_ollama_client] = lambda: client
    monkeypatch.setattr(app, "dependency_overrides", overrides)

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as api:
            return await api.post(
                "/foundry/providers/ollama/generate",
                json={"model": "local-test-model", "prompt": "Explain café"},
            )

    response = asyncio.run(request())
    assert response.status_code == status
    assert response.json()[body_key] == expected_value
    if status == 503:
        assert "finish_reason" not in response.json()
        assert "content" not in response.json()
