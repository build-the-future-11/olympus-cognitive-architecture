import json

import httpx
import pytest
from typer.testing import CliRunner

from olympus.cli import app
from olympus.foundry.ollama import OllamaClient
from olympus.models.hermes_chat import HermesChat


@pytest.mark.parametrize("names, code", [([], 1), (["base"], 0)])
def test_doctor_does_not_claim_generation(
    monkeypatch: pytest.MonkeyPatch, names: list[str], code: int,
) -> None:
    monkeypatch.setattr(OllamaClient, "version", lambda self: "test")
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: names)
    result = CliRunner().invoke(app, ["hermes", "doctor"])
    assert result.exit_code == code
    report = json.loads(result.output)
    assert report["generation_verified"] is False
    assert report["models"] == names


def test_doctor_refuses_missing_selected_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OllamaClient, "version", lambda self: "test")
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: ["base"])

    def missing(self: OllamaClient, name: str) -> str:
        raise ValueError("private response")

    monkeypatch.setattr(OllamaClient, "model_digest", missing)
    result = CliRunner().invoke(app, ["hermes", "doctor", "--model", "missing"])
    assert result.exit_code == 1
    assert "private response" not in result.output
    assert json.loads(result.output)["status"] == "NOT_READY"


def test_chat_retains_turns_and_clears_and_reports_length() -> None:
    requests: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "base", "digest": "a" * 64}]})
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "base", "message": {"content": "hello"},
                                        "done": True, "done_reason": "length"})

    session = HermesChat(OllamaClient(transport=httpx.MockTransport(handle)), "base")
    assert session.answer("first").finish_reason == "length"
    session.answer("second")
    assert requests[1]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "second"},
    ]
    session.clear()
    assert session.history == []


def test_chat_bounds_history_and_rejects_changed_model() -> None:
    digest = "a" * 64

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "base", "digest": digest}]})
        return httpx.Response(200, json={"model": "base", "message": {"content": "x" * 80},
                                        "done": True})

    session = HermesChat(OllamaClient(transport=httpx.MockTransport(handle)), "base",
                         context_tokens=256, max_tokens=16)
    session.answer("hello")
    session.answer("again")
    assert session.truncated
    assert len(session.history) == 2
    before = list(session.history)
    with pytest.raises(ValueError, match="budget"):
        session.answer("x" * 500)
    digest = "b" * 64
    with pytest.raises(ValueError, match="digest changed"):
        session.answer("hello")
    assert session.history == before


def test_cli_backend_failure_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: OllamaClient, name: str) -> str:
        raise httpx.ConnectError("private backend detail")

    monkeypatch.setattr(OllamaClient, "model_digest", fail)
    result = CliRunner().invoke(app, ["hermes", "chat", "--model", "base"])
    assert result.exit_code == 1
    assert "ollama list" in result.output
    assert "private backend detail" not in result.output


@pytest.mark.parametrize("complete", [True, False])
def test_streaming_commits_only_complete_turns(complete: bool) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "base", "digest": "a" * 64}]})
        assert json.loads(request.content)["stream"] is True
        events = [{"model": "base", "message": {"content": "Hello"}, "done": False}]
        if complete:
            events.append({"model": "base", "message": {"content": " world"}, "done": True})
        return httpx.Response(200, text="\n".join(json.dumps(event) for event in events))

    session = HermesChat(OllamaClient(transport=httpx.MockTransport(handle)), "base")
    tokens: list[str] = []
    if complete:
        result = session.answer("hi", on_token=tokens.append)
        assert result.content == "Hello world"
        assert tokens == ["Hello", " world"]
        assert len(session.history) == 2
    else:
        with pytest.raises(ValueError, match="without completion"):
            session.answer("hi", on_token=tokens.append)
        assert tokens == ["Hello"]
        assert session.history == []


def test_interrupt_does_not_commit_partial_turn() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "base", "digest": "a" * 64}]})
        return httpx.Response(200, text=json.dumps(
            {"model": "base", "message": {"content": "partial"}, "done": False}))

    def interrupt(text: str) -> None:
        raise KeyboardInterrupt

    session = HermesChat(OllamaClient(transport=httpx.MockTransport(handle)), "base")
    with pytest.raises(KeyboardInterrupt):
        session.answer("hello", on_token=interrupt)
    assert session.history == []
