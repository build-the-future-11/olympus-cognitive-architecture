from pathlib import Path

from fastapi.testclient import TestClient

from olympus.api import app
from olympus.demos import run_division_demo, run_forge_demo, run_hermes_demo
from olympus.forge.compiler import NaturalLanguageBehaviorCompiler
from olympus.forge.language import BehaviorLanguage
from olympus.forge.runtime import ForgeRuntime


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


def test_api_endpoints() -> None:
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    compile_response = client.post(
        "/forge/compile",
        json={"behavior": "Maintain several interpretations, verify them, and merge them."},
    )
    assert compile_response.status_code == 200
    run_response = client.post(
        "/forge/run",
        json={
            "behavior": (
                "Maintain several interpretations, use tools to test predictions, and merge them."
            ),
            "prompt": "The dependency graph looked like a forest canopy.",
        },
    )
    assert run_response.status_code == 200
    assert "output" in run_response.json()["outputs"]


def test_demos_and_hermes(tmp_path: Path) -> None:
    forge_demo = run_forge_demo()
    division_demo = run_division_demo()
    hermes = run_hermes_demo(tmp_path / "memory.sqlite3")
    assert forge_demo["forge"]["passed"] is True
    assert division_demo["division_reassembly"]["passed"] is True
    assert hermes["confidence"] > 0.0
