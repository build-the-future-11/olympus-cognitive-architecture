"""Promotion and serving reject invalid evidence in disposable registries."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from olympus.api import app, get_foundry_service
from olympus.cli import app as cli_app
from olympus.foundry.schemas import ArtifactStatus, FoundryPipelineResult
from olympus.foundry.service import FoundryService


@pytest.fixture
def registered(tmp_path: Path) -> Iterator[tuple[FoundryService, FoundryPipelineResult]]:
    sample = Path(__file__).resolve().parents[1] / "olympus/foundry/foundry_verification.txt"
    with FoundryService(tmp_path / "registry") as service:
        result = service.run_verification_pipeline(sample)
        yield service, result


def _remove_model(service: FoundryService) -> None:
    with service.store.transaction() as connection:
        connection.execute("DELETE FROM models")


def _change_evaluation(
    service: FoundryService, result: FoundryPipelineResult, **changes: Any
) -> None:
    evaluation = result.evaluation.model_copy(update=changes)
    with service.store.transaction() as connection:
        connection.execute(
            "UPDATE evaluations SET record_json = ? WHERE evaluation_id = ?",
            (evaluation.model_dump_json(), evaluation.evaluation_id),
        )


def test_register_model_requires_an_evaluation(
    registered: tuple[FoundryService, FoundryPipelineResult],
) -> None:
    service, result = registered
    _remove_model(service)
    with service.store.transaction() as connection:
        connection.execute("DELETE FROM evaluations")
    with pytest.raises(ValueError, match="evaluation"):
        service.store.register_model(result.model)
    assert service.store.models() == []


@pytest.mark.parametrize(
    "state",
    [
        ArtifactStatus.RUNNING_EXPERIMENT,
        ArtifactStatus.FAILED,
        ArtifactStatus.NEGATIVE_RESULT,
        ArtifactStatus.INCONCLUSIVE,
    ],
)
def test_register_model_rejects_ineligible_experiment(
    registered: tuple[FoundryService, FoundryPipelineResult],
    state: ArtifactStatus,
) -> None:
    service, result = registered
    _remove_model(service)
    result.experiment.status = state
    service.store.update_experiment(result.experiment)
    with pytest.raises(ValueError, match="experiment"):
        service.store.register_model(result.model)


@pytest.mark.parametrize(
    "changes",
    [
        {"family": "Hermes"},
        {"model_id": "hermes-alpha"},
        {"runtime": "remote-provider"},
        {"version": "unrelated"},
    ],
)
def test_register_model_cannot_relabel_a_verification_checkpoint(
    registered: tuple[FoundryService, FoundryPipelineResult],
    changes: dict[str, Any],
) -> None:
    service, result = registered
    _remove_model(service)
    with pytest.raises(ValueError, match="(identity|runtime|family)"):
        service.store.register_model(result.model.model_copy(update=changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"passed": False},
        {"suite": "training-only"},
        {"dataset_sha256": "0" * 64},
        {"checkpoint_id": "ckpt_" + "0" * 16},
        {"candidate_metrics": {"negative_log_likelihood": 1.0, "perplexity": 2.0}},
        {
            "candidate_metrics": {
                "negative_log_likelihood": 999.0,
                "perplexity": 999.0,
                "evaluated_transitions": 135.0,
            }
        },
        {
            "candidate_metrics": {
                "negative_log_likelihood": 1.0,
                "perplexity": 2.0,
                "evaluated_transitions": 0.0,
            }
        },
        {
            "candidate_metrics": {
                "negative_log_likelihood": 1.0,
                "perplexity": 2.0,
                "evaluated_transitions": 1.0,
            }
        },
    ],
)
def test_verified_label_does_not_override_invalid_evaluation(
    registered: tuple[FoundryService, FoundryPipelineResult],
    changes: dict[str, Any],
) -> None:
    service, result = registered
    _remove_model(service)
    _change_evaluation(service, result, **changes)
    with pytest.raises(ValueError, match="evaluation"):
        service.store.register_model(result.model)


def test_experiment_dataset_digest_must_match_registered_data(
    registered: tuple[FoundryService, FoundryPipelineResult],
) -> None:
    service, result = registered
    _remove_model(service)
    result.experiment.dataset_sha256 = "0" * 64
    service.store.update_experiment(result.experiment)
    with pytest.raises(ValueError, match="dataset"):
        service.store.register_model(result.model)


def test_matching_passing_evidence_can_admit_the_checkpoint_after_restart(
    registered: tuple[FoundryService, FoundryPipelineResult],
) -> None:
    service, result = registered
    _remove_model(service)
    service.store.register_model(result.model)
    with FoundryService(service.root) as recovered:
        assert recovered.model_health(result.model.model_id)["status"] == "ok"
        generated = recovered.generate(result.model.model_id, "request", max_characters=12)
        assert generated.evidence["checkpoint_sha256"] == result.checkpoint.sha256


@pytest.mark.parametrize("operation", ["health", "generate", "export"])
def test_preexisting_invalid_model_is_refused_at_the_serving_boundary(
    registered: tuple[FoundryService, FoundryPipelineResult],
    tmp_path: Path,
    operation: str,
) -> None:
    service, result = registered
    _change_evaluation(service, result, passed=False)
    destination = tmp_path / "retained.json"
    destination.write_bytes(b"retained previous export")
    if operation == "health":
        health = service.model_health(result.model.model_id)
        assert health["status"] == "corrupt"
        assert "evaluation" in str(health["reason"])
    else:
        with pytest.raises((RuntimeError, ValueError), match="evaluation"):
            if operation == "generate":
                service.generate(result.model.model_id, "request", max_characters=8)
            else:
                service.export_checkpoint(result.model.model_id, destination)
    assert destination.read_bytes() == b"retained previous export"


@pytest.mark.parametrize("source_state", ["corrupt", "missing"])
def test_failed_export_preserves_existing_destination(
    registered: tuple[FoundryService, FoundryPipelineResult],
    tmp_path: Path,
    source_state: str,
) -> None:
    service, result = registered
    source = Path(result.checkpoint.path)
    if source_state == "corrupt":
        source.write_bytes(b"corrupted source")
    else:
        source.unlink()
    destination = tmp_path / "retained.json"
    destination.write_bytes(b"retained previous export")
    with pytest.raises((RuntimeError, OSError)):
        service.export_checkpoint(result.model.model_id, destination)
    assert destination.read_bytes() == b"retained previous export"


def test_export_publication_failure_preserves_destination_and_cleans_temporary(
    registered: tuple[FoundryService, FoundryPipelineResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, result = registered
    destination = tmp_path / "export" / "retained.json"
    destination.parent.mkdir()
    destination.write_bytes(b"retained previous export")
    replace = Path.replace

    def fail_destination(path: Path, target: Path) -> Path:
        if target == destination:
            raise OSError("simulated atomic publication failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_destination)
    with pytest.raises(OSError, match="publication failure"):
        service.export_checkpoint(result.model.model_id, destination)
    assert destination.read_bytes() == b"retained previous export"
    assert list(destination.parent.iterdir()) == [destination]


def test_export_uses_one_verified_source_snapshot(
    registered: tuple[FoundryService, FoundryPipelineResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, result = registered
    source = Path(result.checkpoint.path)
    original = source.read_bytes()
    read = Path.read_bytes
    reads: list[bytes] = []

    def replace_after_read(path: Path) -> bytes:
        payload = read(path)
        if path == source:
            reads.append(payload)
            source.write_bytes(b"changed after source read")
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    destination = service.export_checkpoint(result.model.model_id, tmp_path / "export.json")
    assert destination.read_bytes() == original
    assert reads == [original]


async def _listing(service: FoundryService) -> httpx.Response:
    app.dependency_overrides[get_foundry_service] = lambda: service
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.get("/v1/models")
    finally:
        app.dependency_overrides.pop(get_foundry_service, None)


@pytest.mark.parametrize("failure", ["evaluation", "artifact"])
def test_api_model_list_exposes_live_failure(
    registered: tuple[FoundryService, FoundryPipelineResult],
    failure: str,
) -> None:
    service, result = registered
    if failure == "evaluation":
        _change_evaluation(service, result, passed=False)
    else:
        Path(result.checkpoint.path).unlink()
    response = asyncio.run(_listing(service))
    assert response.status_code == 200
    model = response.json()["data"][0]
    assert model["status"] == "UNAVAILABLE"
    assert model["health"]["status"] == "corrupt"
    assert model["health"]["reason"]
    assert service.store.models()[0].status is ArtifactStatus.VERIFIED


def test_health_reports_unreadable_checkpoint_without_crashing(
    registered: tuple[FoundryService, FoundryPipelineResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, result = registered
    source = Path(result.checkpoint.path)
    read = Path.read_bytes

    def deny_source(path: Path) -> bytes:
        if path == source:
            raise PermissionError("simulated unreadable artifact")
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", deny_source)
    health = service.model_health(result.model.model_id)
    assert health["status"] == "corrupt"
    assert "unreadable" in str(health["reason"])


@pytest.mark.parametrize("metric_source", ["baseline_metrics", "candidate_metrics"])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_nonfinite_metric_receipts_cannot_admit_models(
    registered: tuple[FoundryService, FoundryPipelineResult],
    metric_source: str,
    value: float,
) -> None:
    service, result = registered
    _remove_model(service)
    receipt = result.evaluation.model_dump()
    receipt[metric_source]["negative_log_likelihood"] = value
    with service.store.transaction() as connection:
        connection.execute("UPDATE evaluations SET record_json = ?", (json.dumps(receipt),))
    with pytest.raises(ValueError, match="evaluation"):
        service.store.register_model(result.model)


def test_malformed_evaluation_is_reported_as_failed_health(
    registered: tuple[FoundryService, FoundryPipelineResult],
) -> None:
    service, result = registered
    with service.store.transaction() as connection:
        connection.execute("UPDATE evaluations SET record_json = ?", ('{"broken":true}',))
    health = service.model_health(result.model.model_id)
    assert health["status"] == "corrupt"
    assert "evaluation" in str(health["reason"])


def test_wrong_checkpoint_byte_count_is_rejected(
    registered: tuple[FoundryService, FoundryPipelineResult],
) -> None:
    service, result = registered
    _remove_model(service)
    checkpoint = result.checkpoint.model_copy(update={"byte_count": 1})
    with service.store.transaction() as connection:
        connection.execute(
            "UPDATE checkpoints SET record_json = ?", (checkpoint.model_dump_json(),)
        )
    with pytest.raises(RuntimeError, match="byte count"):
        service.store.register_model(result.model)


@pytest.mark.parametrize("defect", ["provenance", "count_matrix", "schema"])
def test_matching_hash_is_insufficient_for_a_malformed_checkpoint(
    registered: tuple[FoundryService, FoundryPipelineResult],
    tmp_path: Path,
    defect: str,
) -> None:
    service, result = registered
    _remove_model(service)
    checkpoint_data = json.loads(Path(result.checkpoint.path).read_bytes())
    if defect == "provenance":
        checkpoint_data["training_provenance"]["dataset_sha256"] = "0" * 64
    elif defect == "count_matrix":
        checkpoint_data["counts"] = []
    else:
        checkpoint_data["schema_version"] = 999
    payload = json.dumps(checkpoint_data).encode()
    digest = hashlib.sha256(payload).hexdigest()
    source = tmp_path / "new-checkpoint.json"
    source.write_bytes(payload)
    checkpoint = result.checkpoint.model_copy(
        update={
            "checkpoint_id": f"ckpt_{digest[:16]}",
            "sha256": digest,
            "path": str(source),
            "byte_count": len(payload),
        }
    )
    service.store.register_checkpoint(checkpoint)
    service.store.register_evaluation(
        result.evaluation.model_copy(
            update={
                "checkpoint_id": checkpoint.checkpoint_id,
                "evaluation_id": f"eval_{digest[:16]}",
            }
        )
    )
    model = result.model.model_copy(
        update={
            "model_id": f"foundry-verification-bigram-{digest[:8]}",
            "version": digest[:12],
            "checkpoint_id": checkpoint.checkpoint_id,
        }
    )
    with pytest.raises(ValueError, match="(provenance|matrix|schema)"):
        service.store.register_model(model)
    assert service.store.models() == []


def test_cli_displays_live_failure_without_rewriting_recorded_status(
    registered: tuple[FoundryService, FoundryPipelineResult],
) -> None:
    service, result = registered
    _change_evaluation(service, result, passed=False)
    command = CliRunner().invoke(cli_app, ["foundry", "list-models", "--root", str(service.root)])
    assert command.exit_code == 0, command.output
    model = json.loads(command.output)[0]
    assert model["status"] == "UNAVAILABLE"
    assert model["registered_status"] == "VERIFIED"
    assert model["health"]["status"] == "corrupt"
