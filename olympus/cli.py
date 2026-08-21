from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from olympus.api import demos
from olympus.core.workspace import LatentWorkspace
from olympus.forge.compiler import NaturalLanguageBehaviorCompiler
from olympus.forge.runtime import ForgeRuntime
from olympus.foundry.ollama import OllamaClient
from olympus.foundry.service import FoundryService
from olympus.labos.manifest import ProjectRecord, ResourceProfile
from olympus.labos.portfolio import PortfolioService

app = typer.Typer(help="Olympus command line interface.")
demo_app = typer.Typer(help="Run built-in Olympus demos.")
forge_app = typer.Typer(help="Compile and execute behaviors.")
labos_app = typer.Typer(help="Discover, validate, and run portfolio projects.")
foundry_app = typer.Typer(help="Operate the durable Olympus Model Foundry.")
app.add_typer(demo_app, name="demo")
app.add_typer(forge_app, name="forge")
app.add_typer(labos_app, name="labos")
app.add_typer(foundry_app, name="foundry")
console = Console()


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
