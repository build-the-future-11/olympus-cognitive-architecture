import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from olympus.data.ingestion import IngestionPipeline
from olympus.data.retrieval import RetrievalPlatform
from olympus.memory.store import MemoryRecord, MemoryStore
from olympus.models.families import HermesNano
from olympus.sdk import OlympusSDK


def test_memory_store_round_trip(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "nested" / "memory.sqlite3") as store:
        store.put(
            MemoryRecord(
                kind="episodic",
                key="k1",
                value="value",
                salience=0.9,
                tags=["policy,security", "demo"],
            )
        )
        record = store.get("k1")
        assert record is not None
        assert record.value == "value"
        assert record.tags == ["policy,security", "demo"]

        store.connection.execute(
            "REPLACE INTO memory (kind, key, value, salience, tags) VALUES (?, ?, ?, ?, ?)",
            ("episodic", "legacy", "value", 0.5, "old,format"),
        )
        legacy = store.get("legacy")
        assert legacy is not None
        assert legacy.tags == ["old", "format"]


def test_memory_store_serializes_shared_connection_writes(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        records = [
            MemoryRecord(kind="episodic", key=f"key-{index}", value=f"value-{index}")
            for index in range(32)
        ]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(store.put, records))
        assert len(store.query("episodic")) == len(records)

    with pytest.raises(ValueError, match="at most 200 characters"):
        MemoryRecord(kind="episodic", key="bounded", value="value", tags=["x" * 201])


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
        expected = hashlib.sha256(b"Keep the project local and private.").hexdigest()
        assert records[0].key == f"prompt:{expected}"


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
                    "choices": [{"message": {"role": "assistant", "content": "evidence"}}],
                },
            )
        raise AssertionError(f"unexpected request: {request.url}")

    sdk = OlympusSDK(
        "http://olympus.test",
        api_token=None,
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


def test_sdk_supports_authenticated_remote_deployments_without_plaintext_tokens() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "ok"})

    sdk = OlympusSDK(
        "https://olympus.example",
        api_token="test-token",
        transport=httpx.MockTransport(handler),
    )
    assert sdk.health() == {"status": "ok"}
    assert requests[0].headers["Authorization"] == "Bearer test-token"

    with pytest.raises(ValueError, match="requires HTTPS"):
        OlympusSDK("http://olympus.example", api_token="test-token")
    with pytest.raises(ValueError, match="without whitespace"):
        OlympusSDK("https://olympus.example", api_token="bad token")
    with pytest.raises(ValueError, match="credential-free"):
        OlympusSDK("https://user:secret@olympus.example")
    with pytest.raises(ValueError, match="origin"):
        OlympusSDK("https://olympus.example/private/base")
    with pytest.raises(ValueError, match="finite and positive"):
        OlympusSDK("https://olympus.example", timeout_seconds=float("nan"))
    with pytest.raises(ValueError, match="origin"):
        OlympusSDK("https://olympus.example:not-a-port")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_tokens": 0}, "max_tokens"),
        ({"temperature": float("nan")}, "temperature"),
        ({"seed": -1}, "seed"),
        ({"prompt": "x" * 50_001}, "prompt"),
    ],
)
def test_sdk_rejects_invalid_generation_requests_locally(
    kwargs: dict[str, object],
    message: str,
) -> None:
    sdk = OlympusSDK(
        "https://olympus.example",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={})),
    )
    arguments: dict[str, object] = {"model": "model-1", "prompt": "verify", **kwargs}
    with pytest.raises(ValueError, match=message):
        sdk.generate(**arguments)  # type: ignore[arg-type]
