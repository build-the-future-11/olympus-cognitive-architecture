from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from olympus.core.interpretation import InterpretiveSuperpositionNetwork
from olympus.core.workspace import LatentWorkspace
from olympus.demos import (
    run_ambiguous_interpretation_demo,
    run_division_demo,
    run_forge_demo,
    run_hermes_demo,
    run_representation_demo,
    run_retrodiction_demo,
    run_transform_demo,
)
from olympus.forge.compiler import NaturalLanguageBehaviorCompiler
from olympus.forge.runtime import ForgeRuntime


class WorkspaceRequest(BaseModel):
    objective: str
    prompt: str


class CompileRequest(BaseModel):
    behavior: str


class RunRequest(BaseModel):
    behavior: str
    prompt: str


app = FastAPI(title="Olympus Forge API", version="0.1.0")
compiler = NaturalLanguageBehaviorCompiler()
runtime = ForgeRuntime()
interpreter = InterpretiveSuperpositionNetwork()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/workspace/analyze")
def analyze_workspace(request: WorkspaceRequest) -> dict[str, object]:
    branches = interpreter.analyze(request.prompt)
    workspace = LatentWorkspace(
        objective=request.objective,
        hypotheses=interpreter.summarize_hypotheses(branches),
        active_plan=["interpret", "verify", "merge"],
        tools_available=["filesystem", "shell", "calculation"],
        branch_activations={branch.category.value: branch.confidence for branch in branches},
    )
    return {
        "workspace": workspace.model_dump(mode="json"),
        "branches": [branch.model_dump(mode="json") for branch in branches],
    }


@app.post("/forge/compile")
def compile_behavior(request: CompileRequest) -> dict[str, object]:
    compilation = compiler.compile(request.behavior)
    return {
        "spec": compilation.spec.model_dump(mode="json"),
        "issues": [issue.model_dump(mode="json") for issue in compilation.issues],
        "mermaid": compilation.spec.to_mermaid(),
    }


@app.post("/forge/run")
def run_behavior(request: RunRequest) -> dict[str, object]:
    compilation = compiler.compile(request.behavior)
    result = runtime.execute(compilation.spec, request.prompt)
    return {
        "spec": compilation.spec.model_dump(mode="json"),
        "outputs": result.outputs,
        "events": [
            {
                "node_id": event.node_id,
                "kind": event.kind,
                "payload": event.payload,
            }
            for event in result.events
        ],
    }


@app.get("/demos")
def demos() -> dict[str, object]:
    return {
        **run_ambiguous_interpretation_demo(),
        **run_retrodiction_demo(),
        **run_transform_demo(),
        **run_representation_demo(),
        **run_division_demo(),
        **run_forge_demo(),
        "hermes": run_hermes_demo(Path("/tmp/olympus-hermes-demo.sqlite3")),
    }
