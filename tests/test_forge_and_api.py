import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from olympus.api import app, demos, get_foundry_service, get_ollama_client
from olympus.demos import run_division_demo, run_forge_demo, run_hermes_demo
from olympus.forge.compiler import NaturalLanguageBehaviorCompiler
from olympus.forge.language import BehaviorLanguage
from olympus.forge.runtime import ForgeRuntime
from olympus.foundry.resources import WorkloadUnavailable
from olympus.foundry.schemas import GenerationResult
from olympus.foundry.service import FoundryService
from olympus.memory.store import MemoryStore


async def _async_request(
    method: str,
    path: str,
    json: dict[str, object] | None = None,
    *,
    headers: dict[str, str] | None = None,
    client_address: tuple[str, int] = ("127.0.0.1", 123),
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, client=client_address)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        return await client.request(method, path, json=json, headers=headers)


def _request(
    method: str,
    path: str,
    json: dict[str, object] | None = None,
    *,
    headers: dict[str, str] | None = None,
    client_address: tuple[str, int] = ("127.0.0.1", 123),
) -> httpx.Response:
    return asyncio.run(
        _async_request(method, path, json, headers=headers, client_address=client_address)
    )


def test_behavior_language_round_trip() -> None:
    compiler = NaturalLanguageBehaviorCompiler()
    language = BehaviorLanguage()
    compilation = compiler.compile("Maintain several interpretations, verify them, and merge them.")
    payload = language.format(compilation.spec)
    parsed = language.parse(payload)
    assert parsed.name == compilation.spec.name
    assert language.semantic_check(parsed) == []


def test_forge_runtime_executes_behavior() -> None:
    compiler = NaturalLanguageBehaviorCompiler()
    runtime = ForgeRuntime()
    spec = compiler.compile(
        "Maintain interpretations, use tools to test predictions, and merge."
    ).spec
    result = runtime.execute(spec, "The cloud was a dragon above the city.")
    assert "output" in result.outputs
    assert len(result.events) >= 4
    assert result.outputs["tool_result"] == {
        "status": "skipped",
        "reason": "no tool handler configured",
    }


def test_forge_runtime_only_reports_effects_it_performs(tmp_path: Path) -> None:
    compiler = NaturalLanguageBehaviorCompiler()
    spec = compiler.compile("Interpret, use a tool, write memory, and merge.").spec
    calls: list[str] = []

    def tool(prompt: str, outputs: dict[str, Any]) -> dict[str, Any]:
        calls.append(prompt)
        return {"status": "executed", "interpretations_seen": len(outputs["interpretations"])}

    with MemoryStore(tmp_path / "forge.sqlite3") as memory:
        result = ForgeRuntime(tool_handler=tool, memory=memory).execute(spec, "Evidence first.")
        records = memory.query("forge")

    assert calls == ["Evidence first."]
    assert result.outputs["tool_result"] == {
        "status": "executed",
        "interpretations_seen": 6,
    }
    assert result.outputs["memory_write"]["status"] == "persisted"
    assert len(records) == 1


def test_public_package_app_export_is_compatible() -> None:
    import olympus

    assert olympus.app is app
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = olympus.missing_export


def test_foreground_verification_reports_admission_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(self: FoundryService, sample: Path) -> None:
        raise WorkloadUnavailable("private host details")

    monkeypatch.setattr(FoundryService, "run_verification_pipeline", unavailable)
    with FoundryService(tmp_path) as service:
        app.dependency_overrides[get_foundry_service] = lambda: service
        try:
            response = _request("POST", "/foundry/verify")
            assert response.status_code == 409
            assert "private host details" not in response.text
        finally:
            app.dependency_overrides.pop(get_foundry_service, None)


def test_api_endpoints() -> None:
    assert _request("GET", "/health").json() == {"status": "ok"}
    assert _request("POST", "/forge/compile", {"behavior": "   "}).status_code == 422
    assert (
        _request(
            "POST",
            "/forge/compile",
            {"behavior": "Interpret and merge.", "unexpected": True},
        ).status_code
        == 422
    )
    compile_response = _request(
        "POST",
        "/forge/compile",
        {"behavior": "Maintain several interpretations, verify them, and merge them."},
    )
    assert compile_response.status_code == 200
    run_response = _request(
        "POST",
        "/forge/run",
        {
            "behavior": (
                "Maintain several interpretations, use tools to test predictions, and merge them."
            ),
            "prompt": "The dependency graph looked like a forest canopy.",
        },
    )
    assert run_response.status_code == 200
    assert "output" in run_response.json()["outputs"]


def test_mutating_api_endpoints_are_local_or_token_authenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = _request(
        "POST",
        "/foundry/verify",
        client_address=("203.0.113.10", 123),
    )
    assert remote.status_code == 403

    monkeypatch.setenv("OLYMPUS_API_TOKEN", "test-secret")
    missing = _request("POST", "/foundry/verify")
    assert missing.status_code == 401
    private_read = _request("GET", "/foundry/jobs/current")
    assert private_read.status_code == 401
    assert _request("GET", "/foundry/jobs/history").status_code == 401
    wrong = _request(
        "POST",
        "/foundry/verify",
        headers={"Authorization": "Bearer wrong"},
    )
    assert wrong.status_code == 401
    raw_token = _request(
        "POST",
        "/foundry/verify",
        headers={"Authorization": "test-secret"},
    )
    assert raw_token.status_code == 401
    wrong_scheme = _request(
        "POST",
        "/foundry/verify",
        headers={"Authorization": "Basic test-secret"},
    )
    assert wrong_scheme.status_code == 401
    authorized = _request(
        "POST",
        "/foundry/jobs/cancel",
        headers={"Authorization": "Bearer test-secret"},
    )
    assert authorized.status_code == 409
    authorized_read = _request(
        "GET",
        "/foundry/jobs/current",
        headers={"Authorization": "Bearer test-secret"},
    )
    assert authorized_read.status_code == 200
    assert _request("GET", "/foundry/jobs/history",
                    headers={"Authorization": "Bearer test-secret"}).status_code == 200
    assert _request("GET", "/foundry/jobs/history?limit=101",
                    headers={"Authorization": "Bearer test-secret"}).status_code == 422


def test_authentication_runs_before_stateful_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def construct_service() -> object:
        calls.append("service")
        raise AssertionError("stateful dependency must not run before authentication")

    monkeypatch.setenv("OLYMPUS_API_TOKEN", "test-secret")
    app.dependency_overrides[get_foundry_service] = construct_service
    try:
        response = _request("POST", "/foundry/verify")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401
    assert calls == []


@pytest.mark.parametrize("configured_token", ["", "   ", "bad token"])
def test_invalid_authentication_configuration_fails_closed(
    configured_token: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLYMPUS_API_TOKEN", configured_token)
    response = _request("GET", "/foundry/jobs/current")
    assert response.status_code == 503
    assert response.json() == {"detail": "API authentication configuration is invalid"}


def test_local_mutations_reject_attacker_controlled_host_headers() -> None:
    for hostile_host in ("attacker.example", "attacker.example@127.0.0.1", "["):
        rebound = _request(
            "POST",
            "/foundry/verify",
            headers={"Host": hostile_host},
        )
        assert rebound.status_code == 403
        assert "loopback Host" in rebound.json()["detail"]

    localhost = _request(
        "POST",
        "/foundry/jobs/cancel",
        headers={"Host": "localhost:8000"},
    )
    assert localhost.status_code == 409

    proxied = _request(
        "POST",
        "/foundry/verify",
        headers={"X-Forwarded-Proto": "https"},
    )
    assert proxied.status_code == 403
    assert "proxied mutations" in proxied.json()["detail"]


def test_demo_endpoint_returns_real_cached_results() -> None:
    demos.cache_clear()
    first = _request("GET", "/demos")
    second = _request("GET", "/demos")

    assert first.status_code == 200
    assert first.json() == second.json()
    assert first.json()["forge"]["passed"] is True
    assert first.json()["hermes"]["confidence"] > 0
    assert demos.cache_info().hits >= 1


def test_demos_and_hermes(tmp_path: Path) -> None:
    forge_demo = run_forge_demo()
    division_demo = run_division_demo()
    hermes = run_hermes_demo(tmp_path / "memory.sqlite3")
    assert forge_demo["forge"]["passed"] is True
    assert division_demo["division_reassembly"]["passed"] is True
    assert hermes["confidence"] > 0.0


def test_api_errors_do_not_expose_internal_paths_or_upstream_urls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingService:
        def run_verification_pipeline(self, _: Path) -> None:
            raise OSError("/private/secret/checkpoint.bin")

    class FailingOllama:
        def health(self) -> None:
            request = httpx.Request("GET", "https://user:secret@example.invalid/api/tags")
            raise httpx.ConnectError("connection exposed user:secret", request=request)

    app.dependency_overrides[get_foundry_service] = FailingService
    app.dependency_overrides[get_ollama_client] = FailingOllama
    try:
        verification = _request("POST", "/foundry/verify")
        provider = _request("GET", "/foundry/providers/ollama")
    finally:
        app.dependency_overrides.clear()

    assert verification.status_code == 422
    assert verification.json() == {"detail": "Foundry verification failed"}
    assert "secret" not in verification.text
    assert provider.status_code == 503
    assert provider.json() == {"detail": "Ollama provider unavailable"}
    assert "secret" not in provider.text
    assert "secret" not in caplog.text
    assert "Traceback" not in caplog.text


def test_ollama_api_forwards_the_reproducibility_seed() -> None:
    observed: dict[str, object] = {}

    class CapturingOllama:
        def generate(self, **kwargs: object) -> GenerationResult:
            observed.update(kwargs)
            return GenerationResult(
                model=str(kwargs["model"]),
                content="bounded",
                finish_reason="stop",
                prompt_characters=6,
                generated_characters=7,
                evidence={"provider": "test"},
            )

    app.dependency_overrides[get_ollama_client] = CapturingOllama
    try:
        response = _request(
            "POST",
            "/foundry/providers/ollama/generate",
            {
                "model": "qwen3:8b",
                "prompt": "verify",
                "temperature": 0,
                "seed": 41,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert observed["seed"] == 41
