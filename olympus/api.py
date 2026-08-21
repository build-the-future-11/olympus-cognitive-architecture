from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from olympus import __version__
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
from olympus.foundry.ollama import OllamaClient
from olympus.foundry.service import FoundryService


class ApiRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class WorkspaceRequest(ApiRequest):
    objective: str = Field(min_length=1, max_length=2_000)
    prompt: str = Field(min_length=1, max_length=50_000)


class CompileRequest(ApiRequest):
    behavior: str = Field(min_length=1, max_length=20_000)


class RunRequest(CompileRequest):
    prompt: str = Field(min_length=1, max_length=50_000)


class ChatMessage(ApiRequest):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=50_000)


class ChatCompletionRequest(ApiRequest):
    model: str = Field(min_length=1, max_length=200)
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    max_tokens: int = Field(default=160, ge=1, le=4_096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    stream: Literal[False] = False


class OllamaGenerateRequest(ApiRequest):
    model: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=50_000)
    system: str | None = Field(default=None, max_length=10_000)
    max_tokens: int = Field(default=256, ge=1, le=4_096)
    context_tokens: int = Field(default=2_048, ge=256, le=131_072)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


compiler = NaturalLanguageBehaviorCompiler()
runtime = ForgeRuntime()
interpreter = InterpretiveSuperpositionNetwork()


@lru_cache(maxsize=1)
def get_foundry_service() -> FoundryService:
    root = Path(os.environ.get("OLYMPUS_FOUNDRY_ROOT", "artifacts/foundry"))
    return FoundryService(root)


@lru_cache(maxsize=1)
def get_ollama_client() -> OllamaClient:
    return OllamaClient(os.environ.get("OLYMPUS_OLLAMA_URL", "http://127.0.0.1:11434"))


FoundryDependency = Annotated[FoundryService, Depends(get_foundry_service)]
OllamaDependency = Annotated[OllamaClient, Depends(get_ollama_client)]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    if get_foundry_service.cache_info().currsize:
        get_foundry_service().close()
        get_foundry_service.cache_clear()
    get_ollama_client.cache_clear()


app = FastAPI(title="Olympus Forge API", version=__version__, lifespan=lifespan)


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


@lru_cache(maxsize=1)
def demos() -> dict[str, object]:
    with TemporaryDirectory(prefix="olympus-demos-") as directory:
        return {
            **run_ambiguous_interpretation_demo(),
            **run_retrodiction_demo(),
            **run_transform_demo(),
            **run_representation_demo(),
            **run_division_demo(),
            **run_forge_demo(),
            "hermes": run_hermes_demo(Path(directory) / "hermes.sqlite3"),
        }


app.get("/demos")(demos)


@app.get("/foundry/status")
def foundry_status(
    service: FoundryDependency,
) -> dict[str, object]:
    return service.status()


@app.post("/foundry/verify")
def verify_foundry(
    service: FoundryDependency,
) -> dict[str, object]:
    sample = Path(__file__).resolve().parent / "foundry/foundry_verification.txt"
    try:
        return service.run_verification_pipeline(sample).model_dump(mode="json")
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/v1/models")
def list_foundry_models(
    service: FoundryDependency,
) -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": model.model_id,
                "object": "model",
                "owned_by": "bu1ld-olympus",
                "status": model.status.value,
                "capabilities": model.capabilities,
                "limitations": model.limitations,
            }
            for model in service.list_models()
        ],
    }


@app.post("/v1/chat/completions")
def create_chat_completion(
    request: ChatCompletionRequest,
    service: FoundryDependency,
) -> dict[str, object]:
    prompt = "\n".join(f"{message.role}: {message.content}" for message in request.messages)
    try:
        result = service.generate(
            request.model,
            prompt,
            max_characters=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    checkpoint_sha = str(result.evidence["checkpoint_sha256"])
    return {
        "id": f"chatcmpl-{checkpoint_sha[:16]}",
        "object": "chat.completion",
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.content},
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_characters,
            "completion_tokens": result.generated_characters,
            "total_tokens": result.prompt_characters + result.generated_characters,
            "unit": "characters",
        },
        "evidence": result.evidence,
    }


@app.get("/foundry/providers/ollama")
def ollama_health(
    client: OllamaDependency,
) -> dict[str, object]:
    try:
        return client.health()
    except (httpx.HTTPError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/foundry/providers/ollama/generate")
def ollama_generate(
    request: OllamaGenerateRequest,
    client: OllamaDependency,
) -> dict[str, object]:
    try:
        return client.generate(
            model=request.model,
            prompt=request.prompt,
            system=request.system,
            max_tokens=request.max_tokens,
            context_tokens=request.context_tokens,
            temperature=request.temperature,
        ).model_dump(mode="json")
    except httpx.HTTPStatusError as error:
        status = 404 if error.response.status_code == 404 else 503
        raise HTTPException(status_code=status, detail=str(error)) from error
    except (httpx.HTTPError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
