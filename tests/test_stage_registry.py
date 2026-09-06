import json
from pathlib import Path

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
    assert status["families"]["Hermes"] == "NOT_PROMOTED"
    assert set(status["families"].values()) == {"NOT_PROMOTED", "SPECIFIED"}
    assert status["verified_test_floor"] >= 64
