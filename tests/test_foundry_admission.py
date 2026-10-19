from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from olympus.api import app, get_foundry_service
from olympus.foundry.bigram import BigramCheckpoint, CharacterBigramModel
from olympus.foundry.schemas import (
    ArtifactStatus,
    DatasetRecord,
    EvaluationRecord,
    FoundryPipelineResult,
)
from olympus.foundry.service import FoundryService


async def _api_request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.request(method, path, **kwargs)


def _register(service: FoundryService, path: Path) -> DatasetRecord:
    path.write_text("request verify evidence\nresponse retain evidence\n" * 16)
    return service.register_text_dataset(
        path,
        dataset_id="admission-fixture",
        version="1",
        source="test://artificial-admission-fixture-v1",
        owner="engineering-test",
        license_name="test-only",
        provenance="Fresh artificial infrastructure fixture, not capability evidence.",
        privacy_classification="internal",
        synthetic=True,
        generator="deterministic test literal v1",
    )


@pytest.fixture
def admitted(tmp_path: Path) -> Iterator[tuple[FoundryService, FoundryPipelineResult]]:
    with FoundryService(tmp_path / "foundry") as service:
        dataset = _register(service, tmp_path / "source.txt")
        result = service.run_bigram_experiment(dataset, hypothesis="Engineering admission control.")
        yield service, result


@pytest.mark.parametrize("field", ["payload", "path", "license", "counts"])
def test_experiment_rejects_caller_substitution_before_writes(tmp_path: Path, field: str) -> None:
    with FoundryService(tmp_path / "foundry") as service:
        dataset = _register(service, tmp_path / "source.txt")
        changed = dataset.model_dump()
        if field in {"payload", "path"}:
            alternate = tmp_path / "alternate.txt"
            raw = Path(dataset.materialized_path).read_bytes()
            if field == "payload":
                raw = b"alpha beta gamma\n" * 45
            alternate.write_bytes(raw)
            changed.update(
                materialized_path=str(alternate),
                sha256=hashlib.sha256(raw).hexdigest(),
                byte_count=len(raw),
                character_count=len(raw.decode()),
            )
        elif field == "license":
            changed["license"] = "substituted-license"
        else:
            changed["character_count"] += 1
        with pytest.raises(ValueError, match="registered dataset"):
            service.run_bigram_experiment(
                type(dataset).model_validate(changed),
                hypothesis="Must reject a substituted identity.",
            )
        assert service.store.experiments() == []
        assert service.store.checkpoints() == []
        assert not (service.root / "checkpoints").exists()


@pytest.mark.parametrize(
    ("dataset_id", "version"),
    [("../escape", "1"), ("safe-dataset", "../../escape"), ("safe-dataset", "/absolute")],
)
def test_dataset_path_components_rejected_before_materialization(
    tmp_path: Path, dataset_id: str, version: str
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("alpha beta\n" * 20)
    with FoundryService(tmp_path / "foundry") as service:
        with pytest.raises(ValueError):
            service.register_text_dataset(
                source,
                dataset_id=dataset_id,
                version=version,
                source="test://path-preflight",
                owner="test",
                license_name="test-only",
                provenance="Artificial path validation test.",
                privacy_classification="internal",
                synthetic=False,
            )
        assert list(service.root.rglob("*.txt")) == []
        assert service.store.datasets() == []


def test_store_rejects_experiment_dataset_digest_mismatch(
    admitted: tuple[FoundryService, FoundryPipelineResult],
) -> None:
    service, result = admitted
    invalid = result.experiment.model_copy(
        update={"experiment_id": "FND_FORGED", "dataset_sha256": "a" * 64}
    )
    with pytest.raises(ValueError, match="dataset"):
        service.store.register_experiment(invalid)
    assert len(service.store.experiments()) == 1


@pytest.mark.parametrize("case", ["missing", "failed", "wrong_dataset", "false_metrics", "nan"])
def test_status_alone_cannot_admit_a_model(
    admitted: tuple[FoundryService, FoundryPipelineResult], case: str
) -> None:
    service, result = admitted
    with service.store.transaction() as connection:
        connection.execute("DELETE FROM models")
        connection.execute("DELETE FROM evaluations")
        if case != "missing":
            evaluation = result.evaluation.model_dump(mode="json")
            if case == "failed":
                evaluation["passed"] = False
            elif case == "wrong_dataset":
                evaluation["dataset_sha256"] = "b" * 64
            elif case == "false_metrics":
                evaluation["candidate_metrics"] = evaluation["baseline_metrics"]
            elif case == "nan":
                evaluation["candidate_metrics"]["negative_log_likelihood"] = float("nan")
            connection.execute(
                "INSERT INTO evaluations VALUES (?, ?, ?)",
                (
                    result.evaluation.evaluation_id,
                    result.checkpoint.checkpoint_id,
                    json.dumps(evaluation),
                ),
            )
    with pytest.raises(ValueError):
        service.store.register_model(result.model)
    assert service.store.models() == []


@pytest.mark.parametrize(
    "change",
    [
        {"family": "Hermes"},
        {"model_id": "hermes-alpha"},
        {"runtime": "external-provider"},
        {"version": "unbound-version"},
    ],
)
def test_bigram_cannot_be_relabelled_as_another_runtime_or_identity(
    admitted: tuple[FoundryService, FoundryPipelineResult], change: dict[str, str]
) -> None:
    service, result = admitted
    with service.store.transaction() as connection:
        connection.execute("DELETE FROM models")
    with pytest.raises(ValueError):
        service.store.register_model(result.model.model_copy(update=change))
    assert service.store.models() == []


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 99},
        {"model_type": "other-runtime"},
        {"training_provenance": {}},
        {"training_characters": 3},
    ],
)
def test_checkpoint_admission_checks_format_and_provenance(
    admitted: tuple[FoundryService, FoundryPipelineResult], change: dict[str, Any]
) -> None:
    service, result = admitted
    values = json.loads(Path(result.checkpoint.path).read_bytes())
    values.update(change)
    raw = json.dumps(values, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    path = service.root / "checkpoints" / "invalid.json"
    path.write_bytes(raw)
    checkpoint = result.checkpoint.model_copy(
        update={
            "checkpoint_id": f"ckpt_{digest[:16]}",
            "sha256": digest,
            "path": str(path),
            "byte_count": len(raw),
        }
    )
    with pytest.raises(ValueError):
        service.store.register_checkpoint(checkpoint)
    assert len(service.store.checkpoints()) == 1


@pytest.mark.parametrize("case", ["missing", "corrupt", "unsupported_runtime", "unverified"])
def test_unavailable_models_remain_visible_but_cannot_be_served(
    admitted: tuple[FoundryService, FoundryPipelineResult], case: str
) -> None:
    service, result = admitted
    if case == "missing":
        Path(result.checkpoint.path).unlink()
    elif case == "corrupt":
        Path(result.checkpoint.path).write_bytes(b"{}")
    else:
        changed = result.model.model_dump(mode="json")
        if case == "unsupported_runtime":
            changed["runtime"] = "external-provider"
        else:
            changed["status"] = ArtifactStatus.FAILED.value
        with service.store.transaction() as connection:
            connection.execute(
                "UPDATE models SET status = ?, record_json = ?",
                (changed["status"], json.dumps(changed)),
            )
    assert service.model_health(result.model.model_id)["status"] == "corrupt"
    app.dependency_overrides[get_foundry_service] = lambda: service
    try:
        listing = asyncio.run(_api_request("GET", "/v1/models"))
        assert listing.status_code == 200
        record = listing.json()["data"][0]
        assert record["id"] == result.model.model_id
        assert record["available"] is False
        assert record["health"]["status"] == "corrupt"
        response = asyncio.run(
            _api_request(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": result.model.model_id,
                    "messages": [{"role": "user", "content": "verify"}],
                },
            )
        )
        assert response.status_code == 422
        assert "choices" not in response.json()
    finally:
        app.dependency_overrides.clear()


def test_export_rejection_preserves_existing_destination(
    admitted: tuple[FoundryService, FoundryPipelineResult], tmp_path: Path
) -> None:
    service, result = admitted
    destination = tmp_path / "portable.json"
    destination.write_bytes(b"previous valid artifact must survive")
    Path(result.checkpoint.path).write_bytes(b"{}")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        service.export_checkpoint(result.model.model_id, destination)
    assert destination.read_bytes() == b"previous valid artifact must survive"


def test_atomic_export_failure_preserves_destination_and_cleans_temporary(
    admitted: tuple[FoundryService, FoundryPipelineResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, result = admitted
    destination = tmp_path / "portable.json"
    destination.write_bytes(b"prior export")
    original_replace = Path.replace

    def refuse_destination(path: Path, target: Any) -> Path:
        if Path(target) == destination:
            raise OSError("artificial replace failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", refuse_destination)
    with pytest.raises(OSError, match="artificial replace failure"):
        service.export_checkpoint(result.model.model_id, destination)
    assert destination.read_bytes() == b"prior export"
    assert list(tmp_path.glob(".portable.json.*")) == []


def test_repeated_registration_and_restart_preserve_admitted_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    with FoundryService(tmp_path / "foundry") as service:
        _register(service, source)
        registered_again = _register(service, source)
        result = service.run_bigram_experiment(
            registered_again, hypothesis="Timestamp-only repeat registration stays valid."
        )
        expected = service.generate(result.model.model_id, "verify", seed=7, max_characters=20)
    with FoundryService(tmp_path / "foundry") as recovered:
        actual = recovered.generate(result.model.model_id, "verify", seed=7, max_characters=20)
        assert actual == expected
        assert recovered.model_health(result.model.model_id)["status"] == "ok"


@pytest.mark.parametrize("target", ["checkpoint", "database", "dataset", "symlink"])
def test_export_cannot_replace_another_managed_artifact(
    admitted: tuple[FoundryService, FoundryPipelineResult], tmp_path: Path, target: str
) -> None:
    service, first = admitted
    second = service.run_bigram_experiment(first.dataset, hypothesis="Second independent attempt.")
    destination = {
        "checkpoint": Path(second.checkpoint.path),
        "database": service.store.path,
        "dataset": Path(first.dataset.materialized_path),
        "symlink": tmp_path / "alias.json",
    }[target]
    if target == "symlink":
        destination.symlink_to(second.checkpoint.path)
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="outside the Foundry"):
        service.export_checkpoint(first.model.model_id, destination)
    assert destination.read_bytes() == before
    assert service.model_health(second.model.model_id)["status"] == "ok"
    assert service.store.integrity_check() == "ok"


@pytest.mark.parametrize("external_registration", [False, True])
def test_export_preserves_outward_symlinks_and_externally_registered_checkpoints(
    admitted: tuple[FoundryService, FoundryPipelineResult],
    tmp_path: Path,
    external_registration: bool,
) -> None:
    service, first = admitted
    second = service.run_bigram_experiment(first.dataset, hypothesis="Independent export witness.")
    registered_path = Path(second.checkpoint.path)
    external = tmp_path / "external-checkpoint.json"
    registered_path.rename(external)
    if external_registration:
        # Exercise an existing registry with a valid checkpoint outside the managed directory.
        record = second.checkpoint.model_copy(update={"path": str(external)})
        with service.store.transaction() as connection:
            connection.execute(
                "UPDATE checkpoints SET record_json = ? WHERE checkpoint_id = ?",
                (record.model_dump_json(), record.checkpoint_id),
            )
        destination = external
    else:
        registered_path.symlink_to(external)
        destination = registered_path
    before = external.read_bytes()
    assert service.model_health(second.model.model_id)["status"] == "ok"
    with pytest.raises(ValueError, match="outside the Foundry|distinct from input artifacts"):
        service.export_checkpoint(first.model.model_id, destination)
    assert external.read_bytes() == before
    if not external_registration:
        assert registered_path.is_symlink()
    assert service.model_health(second.model.model_id)["status"] == "ok"


def test_requested_model_identity_cannot_alias_a_different_registry_record(
    admitted: tuple[FoundryService, FoundryPipelineResult], tmp_path: Path
) -> None:
    service, first = admitted
    second = service.run_bigram_experiment(first.dataset, hypothesis="Distinct admitted identity.")
    with service.store.transaction() as connection:
        connection.execute(
            "UPDATE models SET record_json = ? WHERE model_id = ?",
            (second.model.model_dump_json(), first.model.model_id),
        )
    assert service.model_health(first.model.model_id)["status"] == "corrupt"
    with pytest.raises(RuntimeError, match="registry columns"):
        service.generate(first.model.model_id, "must not serve another model")
    with pytest.raises(RuntimeError, match="registry columns"):
        service.export_checkpoint(first.model.model_id, tmp_path / "invalid-export.json")
    app.dependency_overrides[get_foundry_service] = lambda: service
    try:
        response = asyncio.run(_api_request("GET", "/v1/models"))
        assert response.status_code == 503
        assert "data" not in response.json()
    finally:
        app.dependency_overrides.clear()


def _rehashed_evaluation(
    prior: EvaluationRecord, checkpoint_id: str, candidate: dict[str, float]
) -> EvaluationRecord:
    values = {
        "checkpoint_id": checkpoint_id,
        "dataset_sha256": prior.dataset_sha256,
        "baseline": prior.baseline_metrics,
        "candidate": candidate,
        "passed": True,
    }
    digest = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return prior.model_copy(
        update={
            "evaluation_id": f"eval_{digest[:16]}",
            "checkpoint_id": checkpoint_id,
            "candidate_metrics": candidate,
            "passed": True,
        }
    )


def test_promotion_rejects_counts_that_were_not_fitted_to_the_training_split(
    admitted: tuple[FoundryService, FoundryPipelineResult],
) -> None:
    service, result = admitted
    values = json.loads(Path(result.checkpoint.path).read_bytes())
    row = next(row for row in values["counts"] if max(row) > 1)
    index = max(range(len(row)), key=row.__getitem__)
    row[index] -= 1
    row[(index + 1) % len(row)] += 1
    model = CharacterBigramModel(BigramCheckpoint.model_validate(values))
    raw = model.checkpoint.canonical_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    path = service.root / "checkpoints" / "substituted-counts.json"
    path.write_bytes(raw)
    checkpoint = result.checkpoint.model_copy(
        update={
            "checkpoint_id": f"ckpt_{digest[:16]}",
            "sha256": digest,
            "byte_count": len(raw),
            "path": str(path),
        }
    )
    service.store.register_checkpoint(checkpoint)
    text = Path(result.dataset.materialized_path).read_text()
    split = result.experiment.config["training_characters"]
    candidate = model.evaluate(text[split - 1 :]).as_dict()
    assert candidate["perplexity"] < result.evaluation.baseline_metrics["perplexity"]
    evaluation = _rehashed_evaluation(result.evaluation, checkpoint.checkpoint_id, candidate)
    service.store.register_evaluation(evaluation)
    record = result.model.model_copy(
        update={
            "model_id": f"foundry-verification-bigram-{digest[:8]}",
            "checkpoint_id": checkpoint.checkpoint_id,
            "version": digest[:12],
        }
    )
    with pytest.raises(ValueError, match="registered training split"):
        service.store.register_model(record)


def test_metric_tolerance_cannot_promote_an_actual_baseline_tie(
    admitted: tuple[FoundryService, FoundryPipelineResult], tmp_path: Path
) -> None:
    service, control = admitted
    source = tmp_path / "uniform.txt"
    source.write_text(("aabb" * 18) + "a" + "aaaa")
    dataset = service.register_text_dataset(
        source,
        dataset_id="uniform-fixture",
        version="1",
        source="test://uniform-counts",
        owner="test",
        license_name="test-only",
        provenance="Artificial exact baseline tie.",
        privacy_classification="internal",
        synthetic=True,
        generator="uniform counts v1",
    )
    with pytest.raises(RuntimeError, match="did not beat"):
        service.run_bigram_experiment(
            dataset,
            hypothesis="Uniform counts must fail strict improvement.",
            train_fraction=73 / 77,
        )
    experiment = service.store.experiments()[-1]
    checkpoint = service.store.checkpoints()[-1]
    prior = service.store.evaluations()[-1]
    # An attacker may rehash a nearly passing receipt and change a mutable status.
    # Admission must still enforce the actual strict comparison, independently of tolerance.
    service.store.update_experiment(
        experiment.model_copy(update={"status": ArtifactStatus.CHECKPOINTED})
    )
    candidate = dict(prior.candidate_metrics)
    assert candidate["perplexity"] == 2.0
    candidate["negative_log_likelihood"] = math.log(2) - 1e-13
    candidate["perplexity"] = math.exp(candidate["negative_log_likelihood"])
    service.store.register_evaluation(
        _rehashed_evaluation(prior, checkpoint.checkpoint_id, candidate)
    )
    record = control.model.model_copy(
        update={
            "model_id": f"foundry-verification-bigram-{checkpoint.sha256[:8]}",
            "checkpoint_id": checkpoint.checkpoint_id,
            "version": checkpoint.sha256[:12],
        }
    )
    with pytest.raises(ValueError, match="does not beat"):
        service.store.register_model(record)
