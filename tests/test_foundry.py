from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from olympus.api import app, get_foundry_service, get_ollama_client
from olympus.foundry.bigram import CharacterBigramModel
from olympus.foundry.ollama import OllamaClient
from olympus.foundry.schemas import ArtifactStatus, DatasetRecord, ExperimentRecord
from olympus.foundry.service import FoundryService
from olympus.foundry.store import FoundryStore


def _sample_path() -> Path:
    return Path(__file__).resolve().parents[1] / "olympus/foundry/foundry_verification.txt"


def test_bigram_training_evaluation_and_generation_are_real_and_deterministic() -> None:
    corpus = "request verify evidence\nresponse verify evidence\n" * 4
    model = CharacterBigramModel.train(corpus, seed=11)
    metrics = model.evaluate(corpus)

    assert metrics.perplexity < len(model.checkpoint.alphabet)
    assert metrics.top1_accuracy > 0
    assert model.generate("request", max_characters=40, seed=3) == model.generate(
        "request", max_characters=40, seed=3
    )
    assert len(model.generate("", max_characters=7, temperature=0)) == 7
    with pytest.raises(ValueError, match="at least two characters"):
        CharacterBigramModel.train("x")
    with pytest.raises(ValueError, match="distinct characters"):
        CharacterBigramModel.train("xxxx")
    with pytest.raises(ValueError, match="between 1 and 4096"):
        model.generate("test", max_characters=0)
    with pytest.raises(ValueError, match="between 0 and 2"):
        model.generate("test", temperature=3)


def test_dataset_schema_requires_generator_for_synthetic_data() -> None:
    with pytest.raises(ValueError, match="identify their generator"):
        DatasetRecord(
            dataset_id="fixture-data",
            version="1",
            sha256="a" * 64,
            materialized_path="/tmp/data.txt",
            source="repository://fixture",
            owner="test",
            license="test-only",
            provenance="test fixture",
            privacy_classification="internal",
            synthetic=True,
            byte_count=10,
            character_count=10,
            created_at="2026-08-21T00:00:00+00:00",
        )


def test_store_rejects_dataset_identity_rewrite(tmp_path: Path) -> None:
    first = DatasetRecord(
        dataset_id="fixture-data",
        version="1",
        sha256="a" * 64,
        materialized_path=str(tmp_path / "a.txt"),
        source="repository://fixture",
        owner="test",
        license="test-only",
        provenance="test fixture",
        privacy_classification="internal",
        synthetic=False,
        byte_count=10,
        character_count=10,
        created_at="2026-08-21T00:00:00+00:00",
    )
    rewritten = first.model_copy(update={"sha256": "b" * 64})
    with FoundryStore(tmp_path / "registry.sqlite3") as store:
        assert store.register_dataset(first) is True
        assert store.register_dataset(first) is False
        with pytest.raises(ValueError, match="different content"):
            store.register_dataset(rewritten)
        changed_license = first.model_copy(update={"license": "different-license"})
        with pytest.raises(ValueError, match="different metadata"):
            store.register_dataset(changed_license)
        assert store.integrity_check() == "ok"


def test_store_preserves_experiment_identity_and_append_only_finite_evidence(
    tmp_path: Path,
) -> None:
    dataset = DatasetRecord(
        dataset_id="fixture-data",
        version="1",
        sha256="a" * 64,
        materialized_path=str(tmp_path / "fixture.txt"),
        source="repository://fixture",
        owner="test",
        license="test-only",
        provenance="test fixture",
        privacy_classification="internal",
        synthetic=False,
        byte_count=10,
        character_count=10,
        created_at="2026-08-21T00:00:00+00:00",
    )
    experiment = ExperimentRecord(
        experiment_id="FND_FIXTURE",
        hypothesis="The immutable experiment linkage remains coherent.",
        model_family="fixture",
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        dataset_sha256=dataset.sha256,
        seed=7,
        config={"test": True},
        code_commit="a" * 40,
        hardware={"machine": "test"},
        status=ArtifactStatus.RUNNING_EXPERIMENT,
        created_at="2026-08-21T00:00:00+00:00",
        updated_at="2026-08-21T00:00:00+00:00",
    )
    with FoundryStore(tmp_path / "registry.sqlite3") as store:
        store.register_dataset(dataset)
        store.register_experiment(experiment)
        updated = experiment.model_copy(
            update={
                "updated_at": "2026-08-21T00:00:01+00:00",
            }
        )
        store.update_experiment(updated)
        forged = updated.model_copy(
            update={
                "dataset_id": "other-dataset",
                "dataset_version": "9",
                "dataset_sha256": "f" * 64,
            }
        )
        with pytest.raises(ValueError, match="immutable metadata"):
            store.update_experiment(forged)
        unsupported_transition = updated.model_copy(
            update={
                "status": ArtifactStatus.VERIFIED,
                "updated_at": "2026-08-21T00:00:02+00:00",
            }
        )
        with pytest.raises(ValueError, match="invalid experiment status transition"):
            store.update_experiment(unsupported_transition)
        assert store.experiments() == [updated]

        with pytest.raises(ValueError, match="finite JSON"):
            store.append_evidence(
                timestamp="2026-08-21T00:00:02+00:00",
                entity_type="experiment",
                entity_id=experiment.experiment_id,
                action="invalid",
                payload={"metric": float("nan")},
            )
        nonfinite = experiment.model_copy(
            update={"experiment_id": "FND_NONFINITE", "config": {"metric": float("nan")}}
        )
        with pytest.raises(ValueError, match="finite JSON"):
            store.register_experiment(nonfinite)
        store.append_evidence(
            timestamp="2026-08-21T00:00:02+00:00",
            entity_type="experiment",
            entity_id=experiment.experiment_id,
            action="checkpointed",
            payload={"metric": 1.0},
        )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store._connection.execute("DELETE FROM evidence")
        assert store.integrity_check() == "ok"


def test_store_integrity_detects_column_and_json_linkage_divergence(tmp_path: Path) -> None:
    database = tmp_path / "registry.sqlite3"
    with FoundryStore(database) as store:
        dataset = DatasetRecord(
            dataset_id="fixture-data",
            version="1",
            sha256="a" * 64,
            materialized_path=str(tmp_path / "fixture.txt"),
            source="repository://fixture",
            owner="test",
            license="test-only",
            provenance="test fixture",
            privacy_classification="internal",
            synthetic=False,
            byte_count=10,
            character_count=10,
            created_at="2026-08-21T00:00:00+00:00",
        )
        store.register_dataset(dataset)
        divergent = dataset.model_copy(update={"sha256": "b" * 64})
        store._connection.execute(
            "UPDATE datasets SET record_json = ? WHERE dataset_id = ? AND version = ?",
            (divergent.model_dump_json(), dataset.dataset_id, dataset.version),
        )
        store._connection.commit()
        assert store.integrity_check() == "datasets column/record mismatch: sha256"


def test_store_integrity_detects_cross_record_dataset_divergence(tmp_path: Path) -> None:
    database = tmp_path / "registry.sqlite3"
    dataset = DatasetRecord(
        dataset_id="fixture-data",
        version="1",
        sha256="a" * 64,
        materialized_path=str(tmp_path / "fixture.txt"),
        source="repository://fixture",
        owner="test",
        license="test-only",
        provenance="test fixture",
        privacy_classification="internal",
        synthetic=False,
        byte_count=10,
        character_count=10,
        created_at="2026-08-21T00:00:00+00:00",
    )
    experiment = ExperimentRecord(
        experiment_id="FND_FIXTURE",
        hypothesis="Registry lineage remains coherent.",
        model_family="fixture",
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        dataset_sha256=dataset.sha256,
        seed=7,
        config={"test": True},
        code_commit="a" * 40,
        hardware={"machine": "test"},
        status=ArtifactStatus.RUNNING_EXPERIMENT,
        created_at="2026-08-21T00:00:00+00:00",
        updated_at="2026-08-21T00:00:00+00:00",
    )
    with FoundryStore(database) as store:
        store.register_dataset(dataset)
        store.register_experiment(experiment)
        divergent = experiment.model_copy(update={"dataset_sha256": "b" * 64})
        store._connection.execute(
            "UPDATE experiments SET record_json = ? WHERE experiment_id = ?",
            (divergent.model_dump_json(), experiment.experiment_id),
        )
        store._connection.commit()
        assert store.integrity_check() == "experiment dataset content identity mismatch"


def test_store_migrates_the_exact_legacy_verification_source(tmp_path: Path) -> None:
    database = tmp_path / "registry.sqlite3"
    record = DatasetRecord(
        dataset_id="foundry-verification-corpus",
        version="1.0.0",
        sha256="6f4823e0503426bda11aa4bffe8357963f3bc97a62ea40c2b69700af166a8eab",
        materialized_path=str(tmp_path / "verification.txt"),
        source="repository://datasets/samples/foundry_verification.txt",
        owner="BU1LD Olympus",
        license="LicenseRef-Proprietary",
        provenance="legacy registry fixture",
        privacy_classification="internal",
        synthetic=True,
        generator="legacy fixture",
        byte_count=674,
        character_count=674,
        created_at="2026-08-21T00:00:00+00:00",
    )
    with FoundryStore(database) as store:
        store.register_dataset(record)
    connection = sqlite3.connect(database)
    try:
        connection.execute("DELETE FROM schema_migrations WHERE version = 2")
        connection.commit()
    finally:
        connection.close()
    with FoundryStore(database) as migrated:
        assert migrated.datasets()[0].source == (
            "repository://olympus/foundry/foundry_verification.txt"
        )
        assert migrated.evidence()[-1].action == "metadata_migrated"


def test_foundry_golden_path_persists_provenance_and_serves(tmp_path: Path) -> None:
    with FoundryService(tmp_path / "foundry") as service:
        result = service.run_verification_pipeline(_sample_path())

        assert result.experiment.status is ArtifactStatus.VERIFIED
        assert result.evaluation.passed is True
        assert (
            result.evaluation.candidate_metrics["negative_log_likelihood"]
            < result.evaluation.baseline_metrics["negative_log_likelihood"]
        )
        assert Path(result.checkpoint.path).is_file()
        assert Path(result.export_path).is_file()
        copied_export = service.export_checkpoint(
            result.model.model_id, tmp_path / "portable" / "model.json"
        )
        assert copied_export.read_bytes() == Path(result.checkpoint.path).read_bytes()
        assert service.model_health(result.model.model_id)["status"] == "ok"
        generated = service.generate(
            result.model.model_id,
            "request verify",
            max_characters=32,
            seed=4,
        )
        assert generated.generated_characters == 32
        assert generated.evidence["checkpoint_id"] == result.checkpoint.checkpoint_id
        assert service.status() == {
            "integrity": "ok",
            "datasets": 1,
            "experiments": 1,
            "checkpoints": 1,
            "evaluations": 1,
            "models": 1,
            "evidence_events": 6,
        }

    with FoundryService(tmp_path / "foundry") as recovered:
        assert recovered.list_models()[0].model_id == result.model.model_id
        assert recovered.model_health(result.model.model_id)["status"] == "ok"


def test_negative_experiment_stays_visible(tmp_path: Path) -> None:
    corpus = tmp_path / "negative.txt"
    corpus.write_text("ab" * 200 + "xyz" * 33 + "x", encoding="utf-8")
    with FoundryService(tmp_path / "foundry") as service:
        dataset = service.register_text_dataset(
            corpus,
            dataset_id="negative-result-fixture",
            version="1",
            source="test://negative",
            owner="test",
            license_name="test-only",
            provenance="deliberately shifted evaluation suffix",
            privacy_classification="internal",
            synthetic=False,
        )
        with pytest.raises(RuntimeError, match="did not beat"):
            service.run_bigram_experiment(
                dataset,
                hypothesis="A distribution shift should reject promotion.",
            )
        experiments = service.store.experiments()
        assert experiments[0].status is ArtifactStatus.NEGATIVE_RESULT
        assert service.store.models() == []
        assert service.store.evidence()[-1].payload["status"] == "NEGATIVE_RESULT"


def test_foundry_detects_checkpoint_tampering(tmp_path: Path) -> None:
    with FoundryService(tmp_path / "foundry") as service:
        result = service.run_verification_pipeline(_sample_path())
        Path(result.checkpoint.path).write_text("{}", encoding="utf-8")

        health = service.model_health(result.model.model_id)
        assert health["status"] == "corrupt"
        assert health["reason"] in {"checkpoint byte count mismatch", "checkpoint hash mismatch"}
        with pytest.raises(RuntimeError, match="checkpoint (?:byte count|hash) mismatch"):
            service.generate(result.model.model_id, "test")


def test_foundry_rejects_noncanonical_checkpoint_registry_paths(tmp_path: Path) -> None:
    with FoundryService(tmp_path / "foundry") as service:
        result = service.run_verification_pipeline(_sample_path())
        outside = tmp_path / "outside.json"
        outside.write_bytes(Path(result.checkpoint.path).read_bytes())
        forged = result.checkpoint.model_copy(update={"path": str(outside)})
        service.store._connection.execute(
            "UPDATE checkpoints SET record_json = ? WHERE checkpoint_id = ?",
            (forged.model_dump_json(), result.checkpoint.checkpoint_id),
        )
        service.store._connection.commit()

        health = service.model_health(result.model.model_id)
        assert health["status"] == "corrupt"
        assert health["reason"] == "checkpoint path is not canonical"
        with pytest.raises(RuntimeError, match="path is not canonical"):
            service.generate(result.model.model_id, "test")


def test_foundry_code_provenance_fails_closed_when_git_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git is unavailable")

    monkeypatch.setattr("olympus.foundry.service.subprocess.run", unavailable)
    assert FoundryService._code_commit(tmp_path) == "unavailable"


def test_foundry_persists_secret_safe_unexpected_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "PRIVATE_TRAINING_FAILURE_DETAIL"

    def fail(*_: object, **__: object) -> None:
        raise OSError(secret)

    with FoundryService(tmp_path / "foundry") as service:
        dataset = service.register_text_dataset(
            _sample_path(),
            dataset_id="failure-fixture",
            version="1",
            source="repository://fixture",
            owner="test",
            license_name="test-only",
            provenance="failure redaction fixture",
            privacy_classification="internal",
            synthetic=False,
        )
        monkeypatch.setattr("olympus.foundry.service.CharacterBigramModel.train", fail)
        with pytest.raises(OSError, match=secret):
            service.run_bigram_experiment(
                dataset,
                hypothesis="Unexpected failure details must remain private.",
            )
        experiment = service.store.experiments()[0]
        failure_event = service.store.evidence()[-1]

    assert experiment.status is ArtifactStatus.FAILED
    assert secret not in (experiment.failure_reason or "")
    assert secret not in str(failure_event.payload)
    assert secret not in caplog.text
    assert "OSError" in caplog.text


def test_foundry_rejects_changed_materialized_dataset(tmp_path: Path) -> None:
    with FoundryService(tmp_path / "foundry") as service:
        dataset = service.register_text_dataset(
            _sample_path(),
            dataset_id="integrity-fixture",
            version="1",
            source="repository://fixture",
            owner="test",
            license_name="test-only",
            provenance="copied test fixture",
            privacy_classification="internal",
            synthetic=False,
        )
        Path(dataset.materialized_path).write_text("tampered" * 20, encoding="utf-8")
        with pytest.raises(ValueError, match="no longer matches"):
            service.run_bigram_experiment(dataset, hypothesis="Integrity must be checked.")


def test_foundry_uses_only_registered_canonical_dataset_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    alternate = tmp_path / "alternate.txt"
    source.write_text("registered evidence " * 20, encoding="utf-8")
    alternate.write_text("different unregistered content " * 20, encoding="utf-8")
    with FoundryService(tmp_path / "foundry") as service:
        dataset = service.register_text_dataset(
            source,
            dataset_id="lineage-fixture",
            version="1",
            source="repository://fixture",
            owner="test",
            license_name="test-only",
            provenance="registered fixture",
            privacy_classification="internal",
            synthetic=False,
        )
        forged = dataset.model_copy(
            update={
                "sha256": hashlib.sha256(alternate.read_bytes()).hexdigest(),
                "materialized_path": str(alternate),
                "byte_count": alternate.stat().st_size,
                "character_count": len(alternate.read_text(encoding="utf-8")),
            }
        )
        with pytest.raises(ValueError, match="does not match the registered dataset"):
            service.run_bigram_experiment(forged, hypothesis="Forged lineage must fail.")
        assert service.store.experiments() == []


def test_foundry_rejects_path_traversal_before_materializing_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("bounded dataset materialization " * 20, encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    escaped = tmp_path / f"escape-{digest}.txt"
    with FoundryService(tmp_path / "foundry") as service:
        with pytest.raises(ValueError, match="string_pattern_mismatch"):
            service.register_text_dataset(
                source,
                dataset_id="path-fixture",
                version="../../../escape",
                source="repository://fixture",
                owner="test",
                license_name="test-only",
                provenance="path traversal fixture",
                privacy_classification="internal",
                synthetic=False,
            )
    assert not escaped.exists()


def test_duplicate_dataset_registration_is_idempotent_and_emits_one_event(
    tmp_path: Path,
) -> None:
    with FoundryService(tmp_path / "foundry") as service:
        first = service.register_text_dataset(
            _sample_path(),
            dataset_id="idempotent-fixture",
            version="1",
            source="repository://fixture",
            owner="test",
            license_name="test-only",
            provenance="idempotency fixture",
            privacy_classification="internal",
            synthetic=False,
        )
        second = service.register_text_dataset(
            _sample_path(),
            dataset_id="idempotent-fixture",
            version="1",
            source="repository://fixture",
            owner="test",
            license_name="test-only",
            provenance="idempotency fixture",
            privacy_classification="internal",
            synthetic=False,
        )
        assert second == first
        assert [event.action for event in service.store.evidence()] == ["registered"]


def test_ollama_client_validates_and_generates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "qwen3:8b", "digest": "sha256:" + "a" * 64}
                    ]
                },
            )
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.11.10"})
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["think"] is False
        assert body["options"] == {
            "num_predict": 256,
            "num_ctx": 2_048,
            "temperature": 0.0,
        }
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "message": {"role": "assistant", "content": "OLYMPUS_LOCAL_MODEL_OK"},
                "done": True,
                "done_reason": "stop",
                "eval_count": 4,
            },
        )

    client = OllamaClient(transport=httpx.MockTransport(handler))
    assert client.list_models() == ["qwen3:8b"]
    assert client.version() == "0.11.10"
    assert client.model_digest("qwen3:8b") == "sha256:" + "a" * 64
    result = client.generate(model="qwen3:8b", prompt="verify")
    assert result.content == "OLYMPUS_LOCAL_MODEL_OK"
    assert result.evidence["provider"] == "ollama"
    with pytest.raises(ValueError, match="max_tokens"):
        client.generate(model="qwen3:8b", prompt="verify", max_tokens=0)
    with pytest.raises(ValueError, match="context_tokens"):
        client.generate(model="qwen3:8b", prompt="verify", context_tokens=128)

    for invalid_url in (
        "ftp://127.0.0.1:11434",
        "http://user:secret@127.0.0.1:11434",
        "http://127.0.0.1:11434/api",
        "http://127.0.0.1:11434?token=secret",
    ):
        with pytest.raises(ValueError, match="HTTP or HTTPS origin"):
            OllamaClient(base_url=invalid_url)
    with pytest.raises(ValueError, match="require HTTPS"):
        OllamaClient(base_url="http://ollama.example:11434")
    with pytest.raises(ValueError, match="finite and positive"):
        OllamaClient(timeout_seconds=float("nan"))


def test_ollama_client_rejects_mismatched_models_and_malformed_metrics() -> None:
    def mismatched(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "other:latest",
                "message": {"role": "assistant", "content": "unexpected"},
                "done": True,
            },
        )

    client = OllamaClient(transport=httpx.MockTransport(mismatched))
    with pytest.raises(ValueError, match="model does not match"):
        client.generate(model="qwen3:8b", prompt="verify")

    def malformed_metric(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "message": {"role": "assistant", "content": "unexpected"},
                "done": True,
                "eval_count": -1,
            },
        )

    malformed = OllamaClient(transport=httpx.MockTransport(malformed_metric))
    with pytest.raises(ValueError, match="malformed eval_count"):
        malformed.generate(model="qwen3:8b", prompt="verify")


async def _api_request(
    method: str,
    path: str,
    *,
    json_body: dict[str, object] | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        return await client.request(method, path, json=json_body)


def test_foundry_openai_compatible_api(tmp_path: Path) -> None:
    service = FoundryService(tmp_path / "foundry")

    def ollama_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

    ollama = OllamaClient(transport=httpx.MockTransport(ollama_handler))
    app.dependency_overrides[get_foundry_service] = lambda: service
    app.dependency_overrides[get_ollama_client] = lambda: ollama
    try:
        verification = asyncio.run(_api_request("POST", "/foundry/verify"))
        assert verification.status_code == 200
        model_id = verification.json()["model"]["model_id"]

        models = asyncio.run(_api_request("GET", "/v1/models"))
        assert models.status_code == 200
        assert models.json()["data"][0]["id"] == model_id

        completion = asyncio.run(
            _api_request(
                "POST",
                "/v1/chat/completions",
                json_body={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "verify evidence"}],
                    "max_tokens": 24,
                    "temperature": 0,
                },
            )
        )
        assert completion.status_code == 200
        assert completion.json()["object"] == "chat.completion"
        assert completion.json()["usage"]["completion_tokens"] == 24
        assert completion.json()["usage"]["unit"] == "characters"

        missing = asyncio.run(
            _api_request(
                "POST",
                "/v1/chat/completions",
                json_body={
                    "model": "hermes-does-not-exist",
                    "messages": [{"role": "user", "content": "test"}],
                },
            )
        )
        assert missing.status_code == 404
        provider = asyncio.run(_api_request("GET", "/foundry/providers/ollama"))
        assert provider.json() == {
            "status": "ok",
            "provider": "ollama",
            "models": ["qwen3:8b"],
        }
    finally:
        app.dependency_overrides.clear()
        service.close()
