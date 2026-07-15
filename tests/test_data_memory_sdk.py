from pathlib import Path

from olympus.data.ingestion import IngestionPipeline
from olympus.data.retrieval import RetrievalPlatform
from olympus.memory.store import MemoryRecord, MemoryStore
from olympus.models.families import HermesNano


def test_memory_store_round_trip(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.put(MemoryRecord(kind="episodic", key="k1", value="value", salience=0.9, tags=["demo"]))
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
    store = MemoryStore(tmp_path / "hermes.sqlite3")
    hermes = HermesNano(store)
    response = hermes.respond("Keep the project local and private.")
    records = store.query("episodic")
    assert response["confidence"] > 0
    assert len(records) == 1
