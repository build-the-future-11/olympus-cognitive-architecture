from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from olympus.api import app, get_foundry_service, get_ollama_client
from olympus.foundry.bigram import CharacterBigramModel
from olympus.foundry.ollama import OllamaClient
from olympus.foundry.schemas import ArtifactStatus, DatasetRecord
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
        store.register_dataset(first)
        store.register_dataset(first)
        with pytest.raises(ValueError, match="different content"):
            store.register_dataset(rewritten)
        changed_license = first.model_copy(update={"license": "different-license"})
        with pytest.raises(ValueError, match="different metadata"):
            store.register_dataset(changed_license)
        assert store.integrity_check() == "ok"


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
        assert health["reason"] == "checkpoint hash mismatch"
        with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
            service.generate(result.model.model_id, "test")


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


def test_ollama_client_validates_and_generates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
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
                "message": {"role": "assistant", "content": "OLYMPUS_LOCAL_MODEL_OK"},
                "done": True,
                "done_reason": "stop",
                "eval_count": 4,
            },
        )

    client = OllamaClient(transport=httpx.MockTransport(handler))
    assert client.list_models() == ["qwen3:8b"]
    result = client.generate(model="qwen3:8b", prompt="verify")
    assert result.content == "OLYMPUS_LOCAL_MODEL_OK"
    assert result.evidence["provider"] == "ollama"
    with pytest.raises(ValueError, match="max_tokens"):
        client.generate(model="qwen3:8b", prompt="verify", max_tokens=0)
    with pytest.raises(ValueError, match="context_tokens"):
        client.generate(model="qwen3:8b", prompt="verify", context_tokens=128)


async def _api_request(
    method: str,
    path: str,
    *,
    json_body: dict[str, object] | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
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
