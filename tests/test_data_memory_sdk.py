from pathlib import Path

import httpx

from olympus.data.ingestion import IngestionPipeline
from olympus.data.retrieval import RetrievalPlatform
from olympus.memory.store import MemoryRecord, MemoryStore
from olympus.models.families import HermesNano
from olympus.sdk import OlympusSDK


def test_memory_store_round_trip(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.put(
            MemoryRecord(kind="episodic", key="k1", value="value", salience=0.9, tags=["demo"])
        )
        record = store.get("k1")
        assert record is not None
        assert record.value == "value"


def test_ingestion_and_retrieval(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("The cloud looked like a dragon above the city.", encoding="utf-8")
    pipeline = IngestionPipeline()
    document = pipeline.ingest_file(sample)
    retrieval = RetrievalPlatform()
    lexical = retrieval.lexical_search("dragon cloud", [document])
    semantic = retrieval.semantic_search("dragon cloud", [document])
    assert lexical[0][1] > 0
    assert semantic[0][1] > 0


def test_hermes_nano_uses_memory(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "hermes.sqlite3") as store:
        hermes = HermesNano(store)
        response = hermes.respond("Keep the project local and private.")
        records = store.query("episodic")
        assert response["confidence"] > 0
        assert len(records) == 1


def test_sdk_exposes_stable_foundry_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/foundry/status":
            return httpx.Response(200, json={"integrity": "ok", "models": 1})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "model-1"}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "object": "chat.completion",
                    "model": "model-1",
                    "choices": [
                        {"message": {"role": "assistant", "content": "evidence"}}
                    ],
                },
            )
        raise AssertionError(f"unexpected request: {request.url}")

    sdk = OlympusSDK(
        "http://olympus.test",
        transport=httpx.MockTransport(handler),
    )
    assert sdk.foundry_status()["integrity"] == "ok"
    assert sdk.list_models() == [{"id": "model-1"}]
    assert sdk.generate(model="model-1", prompt="verify")["model"] == "model-1"
    assert [request.url.path for request in requests] == [
        "/foundry/status",
        "/v1/models",
        "/v1/chat/completions",
    ]
