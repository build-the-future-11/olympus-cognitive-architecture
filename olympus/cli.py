from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from olympus.api import demos
from olympus.core.workspace import LatentWorkspace
from olympus.data.olympus_dataset import load_olympus_dataset
from olympus.forge.compiler import NaturalLanguageBehaviorCompiler
from olympus.forge.runtime import ForgeRuntime
from olympus.foundry.data_pipeline import prepare_instruction_dataset
from olympus.foundry.eval_suite import evaluate_checkpoint
from olympus.foundry.load_probe import probe_asgi_app, write_load_smoke_report
from olympus.foundry.ollama import OllamaClient
from olympus.foundry.promotion import evaluate_promotion
from olympus.foundry.quantization import quantize_checkpoint
from olympus.foundry.readiness import (
    audit_instruction_dataset,
    evaluate_portfolio_readiness,
    write_readiness_evidence_schema,
)
from olympus.foundry.resources import memory_snapshot
from olympus.foundry.service import FoundryService
from olympus.foundry.sft import SFTConfig, TinyModelConfig, run_sft
from olympus.labos.manifest import ProjectRecord, ResourceProfile
from olympus.labos.portfolio import PortfolioService
from olympus.models.encoder_quantization import EncoderQuantizationConfig
from olympus.models.olympus_system import OlympusSystemConfig
from olympus.models.reference_families import build_reference_model, default_reference_configs
from olympus.models.runtime import (
    OlympusRuntimeConfig,
    OlympusSystemRuntime,
    load_olympus_checkpoint,
)
from olympus.models.spatiotemporal_lazy import (
    SpatiotemporalParameterStore,
    SpatiotemporalQuery,
)
from olympus.models.specifications import BU1LD_MODEL_SPECS
from olympus.training.encoder_quantization import (
    EncoderQuantizationTrainingConfig,
    train_encoder_quantizer,
)
from olympus.training.olympus_system import (
    OlympusSystemTrainingConfig,
    train_olympus_system,
)
from olympus.upscaler.jepa import JEPAUpscalerConfig
from olympus.upscaler.training import JEPAUpscalerTrainingConfig, train_jepa_upscaler

app = typer.Typer(help="Olympus command line interface.")
demo_app = typer.Typer(help="Run built-in Olympus demos.")
forge_app = typer.Typer(help="Compile and execute behaviors.")
labos_app = typer.Typer(help="Discover, validate, and run portfolio projects.")
foundry_app = typer.Typer(help="Operate the durable Olympus Model Foundry.")
bu1ld_app = typer.Typer(help="Operate evidence-bound BU1LD model pathways.")
app.add_typer(demo_app, name="demo")
app.add_typer(forge_app, name="forge")
app.add_typer(labos_app, name="labos")
app.add_typer(foundry_app, name="foundry")
app.add_typer(bu1ld_app, name="bu1ld")
console = Console()


@bu1ld_app.command("specs")
def bu1ld_specs() -> None:
    """Print target specifications without asserting that target models exist."""

    console.print_json(json.dumps([spec.model_dump(mode="json") for spec in BU1LD_MODEL_SPECS]))


@bu1ld_app.command("reference-models")
def bu1ld_reference_models() -> None:
    """Inspect bounded executable references without claiming family checkpoints."""

    summaries: list[dict[str, object]] = []
    for config in default_reference_configs():
        model = build_reference_model(config)
        summaries.append(
            {
                "family_id": config.family_id,
                "implementation": type(model).__name__,
                "reference_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "status": "REFERENCE_IMPLEMENTATION_NOT_PROMOTED",
                "target_capability_claimed": False,
            }
        )
    console.print_json(json.dumps(summaries))


@bu1ld_app.command("train-encoder-quantizer")
def bu1ld_train_encoder_quantizer(
    manifest: Path,
    output: Path,
    steps: int = 1_000,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 20261011,
    input_dim: int = 768,
    hidden_dim: int = 1_024,
    latent_dim: int = 256,
    codebook_size: int = 1_024,
    residual_codebooks: int = 4,
) -> None:
    """Train one hash-bound encoder-quantizer development candidate."""

    encoder = EncoderQuantizationConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        codebook_size=codebook_size,
        residual_codebooks=residual_codebooks,
    )
    config = EncoderQuantizationTrainingConfig(
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        encoder=encoder,
    )
    report = train_encoder_quantizer(
        manifest.resolve(),
        output.resolve(),
        config=config,
    )
    console.print_json(report.model_dump_json())


@bu1ld_app.command("train-jepa-upscaler")
def bu1ld_train_jepa_upscaler(
    manifest: Path,
    output: Path,
    steps: int = 1_000,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    seed: int = 20261011,
    scale: int = 2,
    feature_channels: int = 48,
    residual_blocks: int = 4,
) -> None:
    """Train one source-disjoint JEPA upscaler development candidate."""

    model = JEPAUpscalerConfig(
        scale=scale,
        feature_channels=feature_channels,
        residual_blocks=residual_blocks,
    )
    config = JEPAUpscalerTrainingConfig(
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model=model,
    )
    report = train_jepa_upscaler(
        manifest.resolve(),
        output.resolve(),
        config=config,
    )
    console.print_json(report.model_dump_json())


@bu1ld_app.command("train-olympus-system")
def bu1ld_train_olympus_system(
    manifest: Path,
    model_config: Path,
    output: Path,
    steps: int = 1_000,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    warmup_steps: int = 0,
    seed: int = 20261011,
    device: str = "auto",
    workload_lock: Path = Path("artifacts/locks/olympus-system-training.lock"),
) -> None:
    """Train and diagnose one hash-bound composed Olympus development candidate."""

    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise typer.BadParameter("device must be auto, cpu, cuda, or mps")
    resolved_config = model_config.resolve()
    if model_config.is_symlink() or not resolved_config.is_file():
        raise typer.BadParameter("model config must be a regular non-symlink JSON file")
    model = OlympusSystemConfig.model_validate_json(resolved_config.read_bytes())
    config = OlympusSystemTrainingConfig(
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        device=device,  # type: ignore[arg-type]
        workload_lock=workload_lock.resolve(),
        model=model,
    )
    report = train_olympus_system(
        manifest.resolve(),
        output.resolve(),
        config=config,
    )
    console.print_json(report.model_dump_json())


@bu1ld_app.command("verify-olympus-dataset")
def bu1ld_verify_olympus_dataset(manifest: Path) -> None:
    """Verify every aligned tensor, digest, split identity, and schema before a run."""

    loaded = load_olympus_dataset(manifest.resolve())
    console.print_json(
        json.dumps(
            {
                "status": "VERIFIED_REFERENCE_DATASET_NOT_SCIENTIFICALLY_QUALIFIED",
                "dataset_id": loaded.manifest.dataset_id,
                "version": loaded.manifest.version,
                "manifest_sha256": loaded.manifest.manifest_sha256,
                "train_examples": len(loaded.manifest.train.source_ids),
                "evaluation_examples": len(loaded.manifest.evaluation.source_ids),
                "extensions": loaded.manifest.extensions,
                "train_shapes": loaded.manifest.train.shapes,
                "evaluation_shapes": loaded.manifest.evaluation.shapes,
            }
        )
    )


@bu1ld_app.command("run-olympus-dataset")
def bu1ld_run_olympus_dataset(
    manifest: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    device: str = "auto",
    mixed_precision: bool = True,
) -> None:
    """Run digest-pinned reference inference over a verified evaluation split."""

    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise typer.BadParameter("device must be auto, cpu, cuda, or mps")
    loaded = load_olympus_dataset(manifest.resolve())
    model, record = load_olympus_checkpoint(checkpoint.resolve(), expected_sha256=checkpoint_sha256)
    if set(model.extension_names) != set(loaded.manifest.extensions):
        raise typer.BadParameter("checkpoint and dataset extension names do not match")
    batch = loaded.evaluation
    input_names = {
        "tokens",
        "image",
        "states",
        "actions",
        "allowed_transitions",
        "risk_score",
        "attention_mask",
        "domain_ids",
        "temporal_mask",
        "timestamp",
    }
    inputs: dict[str, object] = {
        name: value for name, value in batch.items() if name in input_names
    }
    inputs["extension_features"] = {
        name: batch[f"extension__{name}"] for name in loaded.manifest.extensions
    }
    runtime = OlympusSystemRuntime(
        model,
        OlympusRuntimeConfig(
            device=device,  # type: ignore[arg-type]
            mixed_precision=mixed_precision,
        ),
    )
    result = runtime.infer(inputs)  # type: ignore[arg-type]
    output = result.output
    console.print_json(
        json.dumps(
            {
                "status": "COMPLETED_REFERENCE_INFERENCE_NOT_PROMOTED",
                "dataset_manifest_sha256": loaded.manifest.manifest_sha256,
                "checkpoint_sha256": record.sha256,
                "examples": output.raw_decision_logits.shape[0],
                "device": result.device,
                "mixed_precision_effective": result.mixed_precision_effective,
                "elapsed_seconds": result.elapsed_seconds,
                "forced_stop_examples": int(output.aion.forced_stop.sum().item()),
                "mean_effective_risk": float(output.effective_risk.mean().item()),
                "mean_family_gates": {
                    name: float(weight)
                    for name, weight in zip(
                        output.component_names,
                        output.family_gates.mean(dim=0).tolist(),
                        strict=True,
                    )
                },
                "promotion_authorized": False,
            }
        )
    )


@bu1ld_app.command("lazy-plan")
def bu1ld_lazy_plan(
    manifest: Path,
    budget_bytes: int,
    x: float,
    y: float,
    z: float,
    timestamp: float,
    resolution: float,
    velocity_x: float = 0.0,
    velocity_y: float = 0.0,
    velocity_z: float = 0.0,
    lookahead_seconds: float = 0.0,
) -> None:
    """Plan parameter residency without reading any unselected shard."""

    store = SpatiotemporalParameterStore(manifest.resolve(), cache_budget_bytes=budget_bytes)
    plan = store.plan(
        SpatiotemporalQuery(
            position=(x, y, z),
            timestamp=timestamp,
            resolution=resolution,
            velocity=(velocity_x, velocity_y, velocity_z),
            lookahead_seconds=lookahead_seconds,
        ),
        budget_bytes=budget_bytes,
    )
    console.print_json(plan.model_dump_json())


def _default_instruction_source() -> Path:
    repository_source = Path(__file__).resolve().parents[1] / "datasets/hermes-smoke/source.jsonl"
    if repository_source.is_file():
        return repository_source
    return Path(sys.prefix) / "share/olympus/datasets/source.jsonl"


_DEFAULT_INSTRUCTION_SOURCE = _default_instruction_source()


def _default_readiness_thresholds() -> Path:
    repository_config = Path(__file__).resolve().parents[1] / "configs/llm-readiness-v1.json"
    if repository_config.is_file():
        return repository_config
    return Path(sys.prefix) / "share/olympus/configs/llm-readiness-v1.json"


_DEFAULT_READINESS_THRESHOLDS = _default_readiness_thresholds()


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
    learning_rate: float = 3e-3,
    weight_decay: float = 0.01,
    loss_normalization: str = "supervised_token_mean_v2",
) -> None:
    if mode not in {"full", "lora", "qlora"}:
        raise typer.BadParameter("mode must be full, lora, or qlora")
    if loss_normalization not in {"legacy_batch_mean_v1", "supervised_token_mean_v2"}:
        raise typer.BadParameter(
            "loss normalization must be legacy_batch_mean_v1 or supervised_token_mean_v2"
        )
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
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        loss_normalization=loss_normalization,  # type: ignore[arg-type]
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
) -> None:
    report = evaluate_promotion(
        requested_model_id=requested_model_id,
        checkpoint_path=checkpoint.resolve(),
        dataset_manifest_path=manifest.resolve(),
        evaluation_path=evaluation.resolve(),
        quantization_report_path=quantization.resolve(),
        model_card_path=model_card.resolve(),
        output_path=output.resolve(),
        approved_base_license=approved_base_license,
    )
    console.print_json(report.model_dump_json())


@foundry_app.command("audit-dataset-readiness")
def foundry_audit_dataset_readiness(
    manifest: Path,
    output: Path,
) -> None:
    """Verify a dataset and write its concrete quality and scale evidence."""

    evidence = audit_instruction_dataset(manifest.resolve(), output.resolve())
    console.print_json(evidence.model_dump_json())


@foundry_app.command("readiness-check")
def foundry_readiness_check(
    evidence: Path,
    thresholds: Path = _DEFAULT_READINESS_THRESHOLDS,
    output: Path = Path("artifacts/readiness/portfolio-report.json"),
    require_ready: bool = typer.Option(
        False,
        "--require-ready",
        help="Exit with status 1 unless every canonical family is ready.",
    ),
) -> None:
    """Evaluate all six families against immutable large-scale readiness evidence."""

    report = evaluate_portfolio_readiness(
        evidence.resolve(), thresholds.resolve(), output.resolve()
    )
    console.print_json(report.model_dump_json())
    if require_ready and not report.all_families_ready:
        raise typer.Exit(code=1)


@foundry_app.command("readiness-schema")
def foundry_readiness_schema(output: Path) -> None:
    """Write the canonical JSON Schema for six-family evidence bundles."""

    write_readiness_evidence_schema(output.resolve())
    console.print(f"Wrote readiness evidence schema to {output.resolve()}")


@foundry_app.command("load-smoke")
def foundry_load_smoke(
    output: Path,
    endpoint: str = "/health",
    requests: int = 200,
    concurrency: int = 16,
) -> None:
    """Run a bounded in-process ASGI load smoke and retain percentile evidence."""

    from olympus.api import app as api_app

    report = asyncio.run(
        probe_asgi_app(
            api_app,
            endpoint=endpoint,
            requests=requests,
            concurrency=concurrency,
        )
    )
    write_load_smoke_report(report, output.resolve())
    console.print_json(report.model_dump_json())


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
            console.print(f"[{issue.severity}] {record.manifest.project_id}: {issue.message}")


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
