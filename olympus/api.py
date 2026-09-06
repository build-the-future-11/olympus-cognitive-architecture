from __future__ import annotations

import ipaddress
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
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
from olympus.foundry.jobs import FoundryJobManager
from olympus.foundry.ollama import OllamaClient
from olympus.foundry.resources import WorkloadTier, assess_workload, memory_snapshot
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
    configured = os.environ.get("OLYMPUS_FOUNDRY_ROOT")
    root = Path(configured).expanduser() if configured else Path.cwd() / "artifacts/foundry"
    return FoundryService(root)


@lru_cache(maxsize=1)
def get_ollama_client() -> OllamaClient:
    return OllamaClient(os.environ.get("OLYMPUS_OLLAMA_URL", "http://127.0.0.1:11434"))


@lru_cache(maxsize=1)
def get_foundry_job_manager() -> FoundryJobManager:
    sample = Path(__file__).resolve().parent / "foundry/foundry_verification.txt"
    return FoundryJobManager(get_foundry_service(), sample)


FoundryDependency = Annotated[FoundryService, Depends(get_foundry_service)]
OllamaDependency = Annotated[OllamaClient, Depends(get_ollama_client)]
FoundryJobsDependency = Annotated[FoundryJobManager, Depends(get_foundry_job_manager)]


def require_mutation_access(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Permit local mutations, or require the configured bearer token.

    The default remains convenient for a directly bound loopback development
    server. Requests arriving from another host or through a forwarding proxy
    are denied unless OLYMPUS_API_TOKEN is configured and supplied.
    """

    expected = os.environ.get("OLYMPUS_API_TOKEN")
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if expected:
        if supplied and secrets.compare_digest(supplied, expected):
            return
        raise HTTPException(status_code=401, detail="valid bearer token required")

    if request.headers.get("forwarded") or request.headers.get("x-forwarded-for"):
        raise HTTPException(
            status_code=403,
            detail="proxied mutations require OLYMPUS_API_TOKEN",
        )
    host = request.client.host if request.client is not None else ""
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = host == "testclient"
    if not is_loopback:
        raise HTTPException(
            status_code=403,
            detail="mutating endpoints are loopback-only without OLYMPUS_API_TOKEN",
        )


MutationAccess = Annotated[None, Depends(require_mutation_access)]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    if get_foundry_service.cache_info().currsize:
        get_foundry_service().close()
        get_foundry_service.cache_clear()
    get_ollama_client.cache_clear()
    get_foundry_job_manager.cache_clear()


app = FastAPI(title="Olympus Forge API", version=__version__, lifespan=lifespan)


@app.get("/health")
def health(service: FoundryDependency) -> dict[str, str]:
    try:
        integrity = service.store.integrity_check()
    except Exception as error:
        raise HTTPException(status_code=503, detail="foundry registry unavailable") from error
    if integrity != "ok":
        raise HTTPException(status_code=503, detail=f"foundry registry integrity: {integrity}")
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


@app.get("/foundry/overview")
def foundry_overview(service: FoundryDependency) -> dict[str, object]:
    snapshot = memory_snapshot()
    datasets = service.store.datasets()
    experiments = service.store.experiments()
    checkpoints = service.store.checkpoints()
    evaluations = service.store.evaluations()
    models = service.store.models()
    return {
        "status": service.status(),
        "latest_dataset": datasets[-1].model_dump(mode="json") if datasets else None,
        "latest_experiment": experiments[-1].model_dump(mode="json") if experiments else None,
        "latest_checkpoint": checkpoints[-1].model_dump(mode="json") if checkpoints else None,
        "latest_evaluation": evaluations[-1].model_dump(mode="json") if evaluations else None,
        "latest_model": models[-1].model_dump(mode="json") if models else None,
        "promotion_boundary": (
            "VERIFIED_INFRASTRUCTURE_ONLY" if models else "NO_PROMOTED_ARTIFACT"
        ),
        "resources": {
            "snapshot": snapshot.as_dict(),
            "small": assess_workload(WorkloadTier.SMALL, snapshot).admitted,
            "medium": assess_workload(WorkloadTier.MEDIUM, snapshot).admitted,
        },
    }


@app.get("/foundry/jobs/current")
def foundry_job_status(manager: FoundryJobsDependency) -> dict[str, object]:
    return manager.snapshot().model_dump(mode="json")


@app.post("/foundry/jobs/start")
def start_foundry_job(manager: FoundryJobsDependency, _access: MutationAccess) -> dict[str, object]:
    try:
        return manager.start().model_dump(mode="json")
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/foundry/jobs/cancel")
def cancel_foundry_job(
    manager: FoundryJobsDependency, _access: MutationAccess
) -> dict[str, object]:
    try:
        return manager.cancel().model_dump(mode="json")
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/foundry/jobs/retry")
def retry_foundry_job(manager: FoundryJobsDependency, _access: MutationAccess) -> dict[str, object]:
    try:
        return manager.retry().model_dump(mode="json")
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/foundry/verify")
def verify_foundry(
    service: FoundryDependency,
    _access: MutationAccess,
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
    _access: MutationAccess,
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
    _access: MutationAccess,
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
