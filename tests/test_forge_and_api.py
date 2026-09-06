import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from olympus.api import app, demos
from olympus.demos import run_division_demo, run_forge_demo, run_hermes_demo
from olympus.forge.compiler import NaturalLanguageBehaviorCompiler
from olympus.forge.language import BehaviorLanguage
from olympus.forge.runtime import ForgeRuntime
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
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
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
    wrong = _request(
        "POST",
        "/foundry/verify",
        headers={"Authorization": "Bearer wrong"},
    )
    assert wrong.status_code == 401
    authorized = _request(
        "POST",
        "/foundry/jobs/cancel",
        headers={"Authorization": "Bearer test-secret"},
    )
    assert authorized.status_code == 409


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
