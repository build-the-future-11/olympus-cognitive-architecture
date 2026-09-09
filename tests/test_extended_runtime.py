from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import cast

import httpx
import pytest
import torch
from pydantic import ValidationError

from olympus.core.modalities import (
    Artifact,
    BaseArtifact,
    ImageArtifact,
    ModalityKind,
    ReferenceEncoder,
)
from olympus.core.observability import TraceRecorder
from olympus.core.prediction import synthetic_sequence_batch, train_jepa
from olympus.core.representation import (
    DynamicRepresentationTheory,
    RepresentationDescriptor,
    RepresentationFamily,
)
from olympus.core.schemas import StrictModel
from olympus.core.transform import train_learned_transform
from olympus.core.verification import CalibrationVerifier, CodeVerifier, SchemaVerifier
from olympus.data.manifests import DatasetManifest, DatasetSplit
from olympus.data.registry import DataSourceRecord, DataSourceRegistry
from olympus.evaluation.harness import EvaluationHarness
from olympus.forge.experiments import ExperimentController
from olympus.labos.manifest import ProjectStatus, ResourceProfile
from olympus.labos.runner import (
    EventLogger,
    PortfolioRunner,
    RunDatabase,
    RunResult,
    classify_run,
    status_from_run,
)
from olympus.sdk import OlympusSDK
from olympus.training.synthetic import transform_signals
from olympus.training.trainer import OlympusTrainer


class _ExamplePayload(StrictModel):
    value: int


def _source(name: str = "source") -> DataSourceRecord:
    return DataSourceRecord(
        name=name,
        source_url="https://data.example.org/source.json",
        organization="Example Research",
        license="CC-BY-4.0",
        permitted_uses=["research"],
        retrieval_date="2026-07-23",
        sha256="a" * 64,
        provenance="Downloaded from the publisher endpoint.",
        quality_score=0.95,
        language="en",
        domain="testing",
        personal_data=False,
        training_eligible=True,
    )


def test_trace_recorder_captures_result_timing_and_metadata() -> None:
    recorder = TraceRecorder()
    result = recorder.record("calculation", lambda: 42, project="olympus")

    assert result == 42
    assert recorder.events[0].name == "calculation"
    assert recorder.events[0].finished_at >= recorder.events[0].started_at
    assert recorder.events[0].metadata == {"project": "olympus"}
    assert recorder.events[0].succeeded is True
    assert recorder.events[0].error_type is None

    def fail() -> None:
        raise RuntimeError("test failure")

    with pytest.raises(RuntimeError, match="test failure"):
        recorder.record("failure", fail, project="olympus")
    assert recorder.events[1].succeeded is False
    assert recorder.events[1].error_type == "RuntimeError"


def test_reference_encoder_handles_structured_and_unknown_content() -> None:
    encoder = ReferenceEncoder()
    structured = encoder.encode(ImageArtifact(content={"width": 640, "height": 480}))
    unknown = encoder.encode(
        cast(Artifact, BaseArtifact(modality=ModalityKind.ENVIRONMENT_STATE, content=["ready"]))
    )

    assert len(structured.vector) == 2
    assert structured.summary == "width=640, height=480"
    assert unknown.vector == [0.0]
    assert unknown.summary == "['ready']"


def test_representation_selection_and_projection_cover_every_runtime_path() -> None:
    theory = DynamicRepresentationTheory()
    graph = theory.select({"edges": [("a", "b")]})
    grid = theory.select({"grid": [[1, 2], [3, 4]]})
    sequence = theory.select({"sequence": [1, 2, 3]})
    default = theory.select({"x": 1.0, "y": 2.0})

    assert graph.family == RepresentationFamily.GRAPH
    assert grid.family == RepresentationFamily.GRID
    assert sequence.family == RepresentationFamily.SEQUENCE
    assert default.family == RepresentationFamily.EUCLIDEAN
    assert theory.project([], default) == []
    assert theory.project([2.0, -1.0], default) == [1.0, -0.5]
    assert theory.project(
        [2.0, -2.0],
        RepresentationDescriptor(
            family=RepresentationFamily.SPHERICAL,
            reasons=["test"],
            quality_score=1.0,
        ),
    ) == [0.5, -0.5]
    assert theory.project(
        [1.0, 3.0],
        RepresentationDescriptor(
            family=RepresentationFamily.LOW_RANK,
            reasons=["test"],
            quality_score=1.0,
        ),
    ) == [-1.0, 1.0]
    assert theory.project([1.2345678], graph) == [1.234568]


def test_verifiers_report_success_and_failure_evidence() -> None:
    schema = SchemaVerifier()
    code = CodeVerifier()

    assert schema.verify(_ExamplePayload, {"value": 3}).passed is True
    assert schema.verify(_ExamplePayload, {"value": "invalid"}).passed is False
    assert code.verify("def valid():\n    return 1\n").payload["functions"] == ["valid"]
    assert code.verify("def invalid(:\n").passed is False
    assert CalibrationVerifier().verify(0.95, False).passed is False
    assert CalibrationVerifier().verify(0.1, True).passed is False


def test_dataset_manifest_and_registry_enforce_provenance(tmp_path: Path) -> None:
    split = DatasetSplit(train=0.8, validation=0.1, test=0.1)
    manifest = DatasetManifest(
        name="olympus-synthetic",
        version="1.0.0",
        split=split,
        sources=["local://datasets/samples/notes.txt"],
        licenses=["proprietary-test-fixture"],
    )
    assert manifest.split.train == 0.8

    with pytest.raises(ValidationError, match="sum to 1.0"):
        DatasetSplit(train=0.8, validation=0.2, test=0.2)
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        DataSourceRecord.model_validate({**_source().model_dump(), "sha256": "invalid"})

    registry = DataSourceRegistry()
    registry.add(_source())
    with pytest.raises(ValueError, match="already exists"):
        registry.add(_source())
    path = tmp_path / "nested" / "registry.json"
    registry.save(path)
    loaded = DataSourceRegistry.load(path)
    assert loaded == registry
    assert json.loads(path.read_text(encoding="utf-8"))["sources"][0]["name"] == "source"


def test_evaluation_experiments_and_training_are_deterministic() -> None:
    evaluations = EvaluationHarness().run()
    assert len(evaluations) == 5
    assert all(result.passed for result in evaluations)

    experiment = ExperimentController().run_trials(
        "Maintain interpretations, verify them, and merge.",
        "The dependency graph looked like a canopy.",
        trials=2,
    )
    assert experiment.trial_count == 2
    with pytest.raises(ValueError, match="trials"):
        ExperimentController().run_trials("Interpret and merge.", "prompt", trials=0)

    first = OlympusTrainer().run(seed=19)
    second = OlympusTrainer().run(seed=19)
    assert first == second


def test_training_validates_shapes_and_parameters() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        synthetic_sequence_batch(batch_size=1)
    with pytest.raises(ValueError, match="sequence_dim"):
        synthetic_sequence_batch(sequence_dim=0)
    with pytest.raises(ValueError, match="epochs"):
        train_jepa(epochs=0)
    with pytest.raises(ValueError, match="latent_dim"):
        train_jepa(latent_dim=0)
    with pytest.raises(ValueError, match="signals"):
        train_learned_transform(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="positive"):
        train_learned_transform(transform_signals(), learning_rate=0)


def test_sdk_validates_configuration_status_and_payloads() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/forge/compile":
            return httpx.Response(200, json={"spec": {"name": "compiled"}})
        return httpx.Response(404, json={"detail": "not found"})

    sdk = OlympusSDK("https://olympus.example/", transport=httpx.MockTransport(handler))
    assert sdk.health() == {"status": "ok"}
    assert sdk.compile_behavior("Interpret and merge.")["spec"] == {"name": "compiled"}

    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        OlympusSDK("olympus.example")
    with pytest.raises(ValueError, match="timeout_seconds"):
        OlympusSDK("https://olympus.example", timeout_seconds=0)

    failing = OlympusSDK(
        "https://olympus.example",
        transport=httpx.MockTransport(lambda request: httpx.Response(503, json={})),
    )
    with pytest.raises(httpx.HTTPStatusError):
        failing.health()

    non_object = OlympusSDK(
        "https://olympus.example",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=["invalid"])),
    )
    with pytest.raises(ValueError, match="non-object"):
        non_object.health()


def test_run_database_event_log_and_runner_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "runs.sqlite3"
    result = RunResult(
        project_id="demo",
        command="python -c pass",
        return_code=0,
        stdout="ok",
        stderr="",
        status=ProjectStatus.SMOKE_TESTED,
        started_at="2026-07-23T00:00:00+00:00",
        finished_at="2026-07-23T00:00:01+00:00",
        profile="smoke",
        cwd=str(tmp_path),
        classification="ok",
    )
    with RunDatabase(database_path) as database:
        database.insert(result)
        assert database.rows()[0]["project_id"] == "demo"
        assert database.detailed_rows()[0]["stdout"] == "ok"
        assert database.failed_runs() == []
    with pytest.raises(sqlite3.ProgrammingError):
        database.rows()

    event_path = tmp_path / "events.jsonl"
    EventLogger(event_path).log({"status": "ok"})
    assert json.loads(event_path.read_text(encoding="utf-8")) == {"status": "ok"}

    runner = PortfolioRunner(tmp_path / "runner")
    success = runner.run_command(
        project_id="success",
        command=f"{sys.executable} -c pass",
        cwd=tmp_path,
        profile=ResourceProfile.SMOKE,
    )
    missing_command = runner.run_command(
        project_id="missing-command",
        command="olympus-command-that-does-not-exist",
        cwd=tmp_path,
        profile=ResourceProfile.SMOKE,
    )
    timeout = runner.run_command(
        project_id="timeout",
        command=f"{sys.executable} -c 'import time; time.sleep(1)'",
        cwd=tmp_path,
        profile=ResourceProfile.BENCHMARK,
        timeout_seconds=0.01,
    )
    secret = "never-write-this-secret"
    inherited_secret = "never-persist-inherited-secret"
    monkeypatch.setenv("LABOS_TEST_API_KEY", inherited_secret)
    redacted = runner.run_command(
        project_id="redaction",
        command=(
            f"{sys.executable} -c 'import os; "
            'print(os.environ["OPENAI_API_KEY"]); '
            'print(os.environ["LABOS_TEST_API_KEY"]); '
            'print(os.environ["PUBLIC_MODE"])\''
        ),
        cwd=tmp_path,
        profile=ResourceProfile.SMOKE,
        extra_env={"OPENAI_API_KEY": secret, "PUBLIC_MODE": "local"},
    )
    runner.close()
    assert success.status == ProjectStatus.SMOKE_TESTED
    assert missing_command.return_code == 127
    assert missing_command.classification == "dependency"
    assert missing_command.status == ProjectStatus.DISCOVERED_ONLY
    assert timeout.classification == "timeout"
    assert timeout.status == ProjectStatus.FULL_BENCHMARK_PENDING_COMPUTE
    assert secret not in redacted.stdout
    assert inherited_secret not in redacted.stdout
    assert "<redacted>\n<redacted>\nlocal" in redacted.stdout
    persisted = (tmp_path / "runner" / "labos_events.jsonl").read_text(encoding="utf-8")
    assert secret not in persisted
    assert inherited_secret not in persisted
    event = json.loads(persisted.splitlines()[-1])
    assert event["env_overrides"] == {
        "OPENAI_API_KEY": "<redacted>",
        "PUBLIC_MODE": "local",
    }
    with pytest.raises(ValueError, match="finite positive"):
        runner.run_command(
            project_id="invalid-timeout",
            command=f"{sys.executable} -c pass",
            cwd=tmp_path,
            profile=ResourceProfile.SMOKE,
            timeout_seconds=float("nan"),
        )


@pytest.mark.parametrize(
    ("return_code", "stderr", "classification"),
    [
        (0, "", "ok"),
        (124, "", "timeout"),
        (1, "operation not permitted", "permission"),
        (1, "No module named optional", "dependency"),
        (1, "dataset not found", "external_dataset"),
        (1, "API key credential missing", "credential"),
        (1, "assertion failed", "failed_command"),
    ],
)
def test_run_classification_is_explicit(
    return_code: int,
    stderr: str,
    classification: str,
) -> None:
    assert classify_run(return_code, stderr) == classification


def test_status_mapping_covers_release_states() -> None:
    assert status_from_run(0, "ok", ResourceProfile.BENCHMARK) == (
        ProjectStatus.SCIENTIFICALLY_RUNNABLE
    )
    assert status_from_run(1, "external_dataset", ResourceProfile.BENCHMARK) == (
        ProjectStatus.BLOCKED_BY_EXTERNAL_DATASET
    )
    assert status_from_run(1, "credential", ResourceProfile.SMOKE) == (
        ProjectStatus.BLOCKED_BY_CREDENTIAL
    )
    assert status_from_run(1, "permission", ResourceProfile.SMOKE) == (
        ProjectStatus.BLOCKED_BY_PERMISSION
    )
    assert status_from_run(1, "failed_command", ResourceProfile.SMOKE) == (
        ProjectStatus.DISCOVERED_ONLY
    )
