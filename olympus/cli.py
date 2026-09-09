from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.table import Table

from olympus.api import demos
from olympus.core.workspace import LatentWorkspace
from olympus.forge.compiler import NaturalLanguageBehaviorCompiler
from olympus.forge.runtime import ForgeRuntime
from olympus.foundry.attestation import load_strict_json_object
from olympus.foundry.data_pipeline import prepare_instruction_dataset
from olympus.foundry.eval_suite import evaluate_checkpoint
from olympus.foundry.ollama import OllamaClient
from olympus.foundry.promotion import evaluate_promotion, record_invalid_promotion_attempt
from olympus.foundry.quantization import quantize_checkpoint
from olympus.foundry.resources import memory_snapshot
from olympus.foundry.service import FoundryService
from olympus.foundry.sft import SFTConfig, TinyModelConfig, run_sft
from olympus.labos.manifest import ProjectRecord, ResourceProfile
from olympus.labos.portfolio import PortfolioService
from olympus.models.hermes_chat import HermesChat
from olympus.models.registry import model_family_status

app = typer.Typer(help="Olympus command line interface.")
demo_app = typer.Typer(help="Run built-in Olympus demos.")
forge_app = typer.Typer(help="Compile and execute behaviors.")
labos_app = typer.Typer(help="Discover, validate, and run portfolio projects.")
foundry_app = typer.Typer(help="Operate the durable Olympus Model Foundry.")
models_app = typer.Typer(help="Inspect and smoke-test executable model-family components.")
app.add_typer(demo_app, name="demo")
app.add_typer(forge_app, name="forge")
app.add_typer(labos_app, name="labos")
app.add_typer(foundry_app, name="foundry")
app.add_typer(models_app, name="models")
hermes_app = typer.Typer(help="Experimental Hermes with an explicitly selected base model.")
app.add_typer(hermes_app, name="hermes")
console = Console()


@hermes_app.command("doctor")
def hermes_doctor(
    base_url: str = "http://127.0.0.1:11434",
    model: str | None = None,
) -> None:
    """Check backend and installed identity without downloading or generating."""
    try:
        client = OllamaClient(base_url=base_url, timeout_seconds=5)
        version = client.version()
        names = client.list_models()
        digest = client.model_digest(model) if model is not None else None
    except (httpx.HTTPError, ValueError):
        typer.echo(json.dumps({"status": "NOT_READY", "reason":
            "Backend unavailable, malformed response, or requested model missing. "
            "Check Ollama service, endpoint, and 'ollama list'."}))
        raise typer.Exit(1) from None
    ready = bool(names)
    typer.echo(json.dumps({"status": "AVAILABLE_UNQUALIFIED" if ready else "NO_MODELS",
                           "backend_version": version, "models": names,
                           "selected_model": model, "digest": digest,
                           "generation_verified": False}))
    if not ready:
        raise typer.Exit(1)


@hermes_app.command("chat")
def hermes_chat(
    model: str = typer.Option(..., help="Exact installed Ollama model name; no fallback."),
    base_url: str = "http://127.0.0.1:11434",
    context_tokens: int = typer.Option(2048, min=256, max=131072),
    max_tokens: int = typer.Option(256, min=1, max=4096),
) -> None:
    """Ephemeral multi-turn chat. /clear resets history; /exit exits."""
    try:
        session = HermesChat(OllamaClient(base_url=base_url), model,
                             context_tokens=context_tokens, max_tokens=max_tokens)
    except (httpx.HTTPError, ValueError):
        typer.echo("Cannot initialize Hermes. Start Ollama and check the exact installed model "
                   "with 'ollama list'; also check endpoint and context settings.", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Experimental Hermes powered by {model}\nDigest: {session.digest}\n"
               "No chat is saved. /clear, /exit. Responses are not document-verified.\n"
               "Streamed text is provisional until completion; failed turns are not retained.")
    while True:
        try:
            prompt = input("You> ")
            if prompt.strip() == "/exit":
                return
            if prompt.strip() == "/clear":
                session.clear()
                typer.echo("Conversation cleared.")
                continue
            typer.echo("Hermes> ", nl=False)
            result = session.answer(prompt, on_token=lambda text: typer.echo(text, nl=False))
            typer.echo()
            if session.truncated:
                typer.echo("[Older turns omitted to fit the input budget.]")
            if result.finish_reason == "length":
                typer.echo("[Output budget reached; answer may be incomplete.]")
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nSession closed.")
            return
        except (httpx.HTTPError, ValueError):
            typer.echo("\nGeneration failed; discard any partial output. "
                       "Check backend availability, model identity and "
                       "input length. Failed turn was not saved.", err=True)


def _default_instruction_source() -> Path:
    repository_source = (
        Path(__file__).resolve().parents[1] / "datasets/hermes-smoke/source.jsonl"
    )
    if repository_source.is_file():
        return repository_source
    return Path(sys.prefix) / "share/olympus/datasets/source.jsonl"


_DEFAULT_INSTRUCTION_SOURCE = _default_instruction_source()


@demo_app.command("run-all")
def run_all_demos() -> None:
    console.print_json(json.dumps(demos()))


@forge_app.command("compile")
def compile_behavior(behavior: str) -> None:
    compiler = NaturalLanguageBehaviorCompiler()
    compilation = compiler.compile(behavior)
    console.print(compilation.spec.to_mermaid())


@forge_app.command("run")
def run_behavior(behavior: str, prompt: str) -> None:
    compiler = NaturalLanguageBehaviorCompiler()
    runtime = ForgeRuntime()
    compilation = compiler.compile(behavior)
    result = runtime.execute(compilation.spec, prompt)
    console.print_json(json.dumps(result.outputs))


@app.command("workspace")
def workspace(objective: str, prompt: str, output: Path | None = None) -> None:
    compiler = NaturalLanguageBehaviorCompiler()
    runtime = ForgeRuntime()
    spec = compiler.compile(
        "Maintain several possible interpretations, verify them, and merge them."
    ).spec
    result = runtime.execute(spec, prompt)
    workspace_model = LatentWorkspace(
        objective=objective,
        task_state={"prompt": prompt},
        hypotheses=[],
        active_plan=["interpret", "verify", "merge"],
        predicted_consequences=[result.outputs["merged"]["statement"]],
    )
    if output is not None:
        workspace_model.checkpoint(output)
        console.print(f"Saved workspace to {output}")
        return
    console.print_json(workspace_model.serialize())


@models_app.command("status")
def models_status() -> None:
    """Report implementation evidence separately from checkpoint promotion."""

    console.print_json(json.dumps(model_family_status()))


@models_app.command("smoke")
def models_smoke(
    output_dir: Path = Path("artifacts/family-model-smoke/smoke"),
    seed: int = 20_260_906,
    steps: int = typer.Option(24, min=1, max=10_000),
) -> None:
    """Run bounded component training smokes; this never promotes a model."""

    from olympus.models.smoke import run_family_smoke

    manifest = run_family_smoke(output_dir.resolve(), seed=seed, steps=steps)
    console.print_json(json.dumps(manifest.model_dump(mode="json")))
    if not manifest.passed:
        raise typer.Exit(code=1)


@models_app.command("composition-smoke")
def models_composition_smoke(
    output_dir: Path = Path("artifacts/model-composition-smoke"),
    seed: int = 20_260_906,
) -> None:
    """Run the fixed non-material six-role contract-composition replay."""

    from olympus.models.composition_smoke import run_reference_replay_smoke

    manifest = run_reference_replay_smoke(output_dir.resolve(), seed=seed)
    console.print_json(manifest.model_dump_json())
    if not manifest.passed:
        raise typer.Exit(code=1)


def _foundry_service(root: Path) -> FoundryService:
    return FoundryService(root.resolve())


@foundry_app.command("verify-pipeline")
def verify_foundry_pipeline(
    root: Path = Path("artifacts/foundry"),
    sample: Path | None = None,
) -> None:
    sample_path = sample or Path(__file__).resolve().parent / "foundry/foundry_verification.txt"
    with _foundry_service(root) as service:
        result = service.run_verification_pipeline(sample_path.resolve())
        console.print_json(result.model_dump_json())


@foundry_app.command("status")
def foundry_status(root: Path = Path("artifacts/foundry")) -> None:
    with _foundry_service(root) as service:
        console.print_json(json.dumps(service.status()))


@foundry_app.command("list-models")
def list_foundry_models(root: Path = Path("artifacts/foundry")) -> None:
    with _foundry_service(root) as service:
        console.print_json(
            json.dumps([model.model_dump(mode="json") for model in service.list_models()])
        )


@foundry_app.command("generate")
def foundry_generate(
    model: str,
    prompt: str,
    root: Path = Path("artifacts/foundry"),
    max_characters: int = 160,
    temperature: float = 0.7,
    seed: int | None = None,
) -> None:
    with _foundry_service(root) as service:
        result = service.generate(
            model,
            prompt,
            max_characters=max_characters,
            temperature=temperature,
            seed=seed,
        )
        console.print_json(result.model_dump_json())


@foundry_app.command("ollama-smoke")
def ollama_smoke(
    model: str,
    prompt: str = "Reply with exactly: OLYMPUS_LOCAL_MODEL_OK",
    base_url: str = "http://127.0.0.1:11434",
    max_tokens: int = 16,
    context_tokens: int = 1_024,
) -> None:
    client = OllamaClient(base_url=base_url)
    available = client.list_models()
    if model not in available:
        raise typer.BadParameter(
            f"model {model!r} is not installed; available models: {', '.join(available) or 'none'}"
        )
    result = client.generate(
        model=model,
        prompt=prompt,
        system="Follow the user's formatting instruction exactly and do not add commentary.",
        max_tokens=max_tokens,
        context_tokens=context_tokens,
        temperature=0,
    )
    console.print_json(result.model_dump_json())


@foundry_app.command("resource-status")
def foundry_resource_status() -> None:
    console.print_json(json.dumps(memory_snapshot().as_dict()))


@foundry_app.command("prepare-dataset")
def foundry_prepare_dataset(
    source: Path = _DEFAULT_INSTRUCTION_SOURCE,
    output: Path = Path("artifacts/foundry/deep/dataset"),
    dataset_id: str = "olympus-foundry-instructions",
    version: str = "1.0.0",
    source_uri: str = "repository://datasets/hermes-smoke/source.jsonl",
) -> None:
    manifest = prepare_instruction_dataset(
        source.resolve(),
        output.resolve(),
        dataset_id=dataset_id,
        version=version,
        source_uri=source_uri,
    )
    console.print_json(manifest.model_dump_json())


@foundry_app.command("train-sft")
def foundry_train_sft(
    manifest: Path,
    output: Path = Path("artifacts/foundry/deep/training"),
    mode: str = "full",
    base_checkpoint: Path | None = None,
    resume_checkpoint: Path | None = None,
    seed: int = 7,
    epochs: int = 2,
    width: int = 64,
    layers: int = 2,
    heads: int = 4,
    sequence_tokens: int = 192,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
) -> None:
    if mode not in {"full", "lora", "qlora"}:
        raise typer.BadParameter("mode must be full, lora, or qlora")
    model = TinyModelConfig(
        width=width,
        layers=layers,
        heads=heads,
        max_sequence_tokens=sequence_tokens,
    )
    config = SFTConfig(
        mode=mode,  # type: ignore[arg-type]
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        model=model,
    )
    summary = run_sft(
        manifest.resolve(),
        output.resolve(),
        config=config,
        base_checkpoint=base_checkpoint.resolve() if base_checkpoint else None,
        resume_checkpoint=resume_checkpoint.resolve() if resume_checkpoint else None,
    )
    console.print_json(summary.model_dump_json())


@foundry_app.command("evaluate-checkpoint")
def foundry_evaluate_checkpoint(
    checkpoint: Path,
    manifest: Path,
    output: Path,
    base_checkpoint: Path | None = None,
    max_generation_tokens: int = 48,
) -> None:
    report = evaluate_checkpoint(
        checkpoint.resolve(),
        manifest.resolve(),
        output.resolve(),
        base_checkpoint=base_checkpoint.resolve() if base_checkpoint else None,
        max_generation_tokens=max_generation_tokens,
    )
    console.print_json(report.model_dump_json())


@foundry_app.command("quantize-checkpoint")
def foundry_quantize_checkpoint(
    checkpoint: Path,
    manifest: Path,
    output: Path,
    bits: int = 8,
) -> None:
    if bits not in {4, 8}:
        raise typer.BadParameter("bits must be 4 or 8")
    report = quantize_checkpoint(
        checkpoint.resolve(),
        manifest.resolve(),
        output.resolve(),
        bits=bits,  # type: ignore[arg-type]
    )
    console.print_json(report.model_dump_json())


@foundry_app.command("promotion-check")
def foundry_promotion_check(
    requested_model_id: str,
    checkpoint: Path,
    manifest: Path,
    evaluation: Path,
    quantization: Path,
    model_card: Path,
    output: Path,
    approved_base_license: str,
    serving_verification: Path | None = None,
    attestation_bundle: Path | None = None,
    attestation_trust_store: Path | None = None,
) -> None:
    def record_invalid(error: Exception) -> None:
        try:
            attempt = record_invalid_promotion_attempt(
                output.resolve(), requested_model_id=requested_model_id,
                error_type=type(error).__name__,
            )
            console.print_json(json.dumps({"status": "INVALID_INPUT", "attempt": str(attempt)}))
        except OSError:
            console.print("Invalid-input diagnostic could not be persisted; check output access.")

    serving_payload = None
    if serving_verification is not None:
        try:
            serving_payload = load_strict_json_object(serving_verification.resolve())
        except (OSError, ValueError) as exc:
            record_invalid(exc)
            raise typer.BadParameter(
                f"invalid serving-verification JSON: {exc}",
                param_hint="serving_verification",
            ) from exc
    try:
        report = evaluate_promotion(
            requested_model_id=requested_model_id,
            checkpoint_path=checkpoint.resolve(),
            dataset_manifest_path=manifest.resolve(),
            evaluation_path=evaluation.resolve(),
            quantization_report_path=quantization.resolve(),
            model_card_path=model_card.resolve(),
            output_path=output.resolve(),
            approved_base_license=approved_base_license,
            serving_verification=serving_payload,
            attestation_bundle_path=(
                attestation_bundle.resolve() if attestation_bundle is not None else None
            ),
            attestation_trust_store_path=(
                attestation_trust_store.resolve()
                if attestation_trust_store is not None
                else None
            ),
        )
    except (OSError, ValueError) as exc:
        record_invalid(exc)
        raise typer.BadParameter(
            "Promotion evidence is invalid or unreadable. Check input paths, byte limits, "
            "schema/metric consistency and dataset hashes. No new decision was produced; "
            "do not use an older output as this run's result.",
            param_hint="promotion evidence",
        ) from exc
    console.print_json(report.model_dump_json())
    if not report.passed:
        raise typer.Exit(code=1)


def _portfolio_service(workspace: Path, artifacts: Path) -> PortfolioService:
    return PortfolioService(workspace_root=workspace.resolve(), artifact_root=artifacts.resolve())


def _print_records_table(records: Sequence[ProjectRecord]) -> None:
    table = Table(title="LabOS Portfolio")
    table.add_column("Project")
    table.add_column("Status")
    table.add_column("Domain")
    table.add_column("Source")
    for record in records:
        table.add_row(
            record.manifest.project_id,
            record.status.value,
            record.manifest.domain,
            record.manifest.manifest_source.value,
        )
    console.print(table)


@labos_app.command("discover")
def discover(
    workspace: Path = Path(".."),
    artifacts: Path = Path("artifacts"),
) -> None:
    service = _portfolio_service(workspace, artifacts)
    records = service.discover()
    _print_records_table(records)


@labos_app.command("list")
def list_projects(
    workspace: Path = Path(".."),
    artifacts: Path = Path("artifacts"),
) -> None:
    service = _portfolio_service(workspace, artifacts)
    _print_records_table(service.discover())


@labos_app.command("validate")
def validate(
    workspace: Path = Path(".."),
    artifacts: Path = Path("artifacts"),
) -> None:
    service = _portfolio_service(workspace, artifacts)
    records = service.validate()
    _print_records_table(records)
    for record in records:
        for issue in record.validation.issues:
            console.print(
                f"[{issue.severity}] {record.manifest.project_id}: {issue.message}"
            )


@labos_app.command("run-smoke")
def run_smoke(
    workspace: Path = Path(".."),
    artifacts: Path = Path("artifacts"),
    project: list[str] | None = None,
) -> None:
    service = _portfolio_service(workspace, artifacts)
    records = service.run_smoke(
        project_ids=set(project) if project else None,
        profile=ResourceProfile.SMOKE,
    )
    _print_records_table(records)


@labos_app.command("run-all")
def run_all(
    workspace: Path = Path(".."),
    artifacts: Path = Path("artifacts"),
    profile: ResourceProfile = ResourceProfile.SMOKE,
) -> None:
    service = _portfolio_service(workspace, artifacts)
    records = service.run_smoke(profile=profile)
    _print_records_table(records)


@labos_app.command("run-group")
def run_group(
    projects: list[str],
    workspace: Path = Path(".."),
    artifacts: Path = Path("artifacts"),
    profile: ResourceProfile = ResourceProfile.SMOKE,
) -> None:
    service = _portfolio_service(workspace, artifacts)
    records = service.run_smoke(project_ids=set(projects), profile=profile)
    _print_records_table(records)


@labos_app.command("run-project")
def run_project(
    project_id: str,
    workspace: Path = Path(".."),
    artifacts: Path = Path("artifacts"),
    profile: ResourceProfile = ResourceProfile.SMOKE,
) -> None:
    service = _portfolio_service(workspace, artifacts)
    records = service.run_smoke(project_ids={project_id}, profile=profile)
    _print_records_table(records)


@labos_app.command("retry-transient")
def retry_transient(
    workspace: Path = Path(".."),
    artifacts: Path = Path("artifacts"),
) -> None:
    service = _portfolio_service(workspace, artifacts)
    transient = {
        project_id
        for project_id, command in service.runner.run_db.failed_runs()
        if "timeout" in command.lower() or "pytest" in command.lower()
    }
    if not transient:
        console.print("No transient-looking failed runs recorded.")
        return
    records = service.run_smoke(project_ids=transient, profile=ResourceProfile.SMOKE)
    _print_records_table(records)


@labos_app.command("resume-failed")
def resume_failed(
    workspace: Path = Path(".."),
    artifacts: Path = Path("artifacts"),
) -> None:
    service = _portfolio_service(workspace, artifacts)
    failed = service.runner.run_db.failed_runs()
    if not failed:
        console.print("No failed runs recorded.")
        return
    project_ids = {project_id for project_id, _ in failed}
    records = service.run_smoke(project_ids=project_ids, profile=ResourceProfile.SMOKE)
    _print_records_table(records)


@labos_app.command("list-blocked")
def list_blocked(
    workspace: Path = Path(".."),
    artifacts: Path = Path("artifacts"),
) -> None:
    service = _portfolio_service(workspace, artifacts)
    records = service.discover()
    blocked = [record for record in records if record.blocked_reason]
    if not blocked:
        console.print("No blocked dependencies detected.")
        return
    _print_records_table(blocked)
    for record in blocked:
        console.print(f"{record.manifest.project_id}: {record.blocked_reason}")


@labos_app.command("aggregate-metrics")
def aggregate_metrics(
    workspace: Path = Path(".."),
    artifacts: Path = Path("artifacts"),
) -> None:
    service = _portfolio_service(workspace, artifacts)
    records = service.discover()
    summary = service.reporter.summarize(records)
    console.print_json(
        json.dumps(
            {
                "total_projects": summary.total_projects,
                "by_status": summary.by_status,
                "run_history": service.runner.run_db.rows(),
            }
        )
    )


@labos_app.command("compare-runs")
def compare_runs(
    workspace: Path = Path(".."),
    artifacts: Path = Path("artifacts"),
) -> None:
    service = _portfolio_service(workspace, artifacts)
    rows = service.runner.run_db.rows()
    by_project: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_project.setdefault(str(row["project_id"]), []).append(row)
    console.print_json(
        json.dumps(
            {
                project_id: {
                    "runs": len(items),
                    "latest_status": items[-1]["status"],
                    "latest_command": items[-1]["command"],
                    "latest_finished_at": items[-1]["finished_at"],
                }
                for project_id, items in sorted(by_project.items())
            }
        )
    )


@labos_app.command("generate-report")
def generate_report(
    workspace: Path = Path(".."),
    artifacts: Path = Path("artifacts"),
    output_root: Path = Path("artifacts/reports"),
) -> None:
    service = _portfolio_service(workspace, artifacts)
    records = service.discover()
    service.write_reports(records, output_root.resolve())
    console.print(f"Wrote reports to {output_root.resolve()}")


if __name__ == "__main__":
    app()
