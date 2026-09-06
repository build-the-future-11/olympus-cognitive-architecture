import json
from pathlib import Path

from olympus.models.smoke import FamilySmokeManifest, _source_digest

ROOT = Path(__file__).resolve().parents[1]


def test_all_twelve_stage_records_are_unique_and_evidenced() -> None:
    status = json.loads((ROOT / "evidence" / "status.json").read_text())
    stages = status["stages"]
    assert [record["stage"] for record in stages] == list(range(1, 13))
    assert len({record["name"] for record in stages}) == 12
    for record in stages:
        assert (ROOT / record["evidence"]).is_file()


def test_model_family_truth_is_fail_closed() -> None:
    status = json.loads((ROOT / "evidence" / "status.json").read_text())
    assert (ROOT / status["final_verification"]).is_file()
    assert status["promotion"] == "NOT_PROMOTED"
    assert set(status["families"].values()) == {"EXPERIMENTAL_SMOKE_NOT_PROMOTED"}
    assert set(status["families"]) == {
        "Hermes",
        "Prometheus",
        "Perseus",
        "Olympus-Atlas",
        "Kronos",
        "Aion",
    }
    assert status["verified_test_floor"] >= 150


def test_model_family_smoke_evidence_is_present_and_non_promoting() -> None:
    status = json.loads((ROOT / "evidence" / "status.json").read_text())
    evidence_path = ROOT / status["family_component_evidence"]
    evidence = json.loads(evidence_path.read_text())
    manifest = FamilySmokeManifest.model_validate(evidence)

    assert evidence["passed"] is True
    assert evidence["promotion_authorized"] is False
    assert evidence["scientific_claim"] == "execution_smoke_only"
    assert manifest.source_sha256 == _source_digest()
    assert manifest.substrate.passed
    assert manifest.substrate_checkpoint.qualifying_checkpoint is False
    assert manifest.substrate_checkpoint.promoted is False
    assert [record["family"] for record in evidence["families"]] == [
        "Hermes",
        "Prometheus",
        "Perseus",
        "Olympus-Atlas",
        "Kronos",
        "Aion",
    ]
    for record in evidence["families"]:
        assert record["status"] == "EXPERIMENTAL_SMOKE_NOT_PROMOTED"
        assert record["passed"] is True
        assert record["checkpoint"]["qualifying_checkpoint"] is False
        assert record["checkpoint"]["promoted"] is False
