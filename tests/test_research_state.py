from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

import pytest

from olympus.research_state import ResearchStateError, audit_research_state

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_root_research_state_matches_version_and_successor_states() -> None:
    state = _read(ROOT / "RESEARCH_STATE.json")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert state["software"]["version"] == project["version"]
    assert state["software"]["release_ready"] is False
    assert state["software"]["model_promoted"] is False

    studies = {study["id"]: study for study in state["studies"]}
    v4 = _read(ROOT / studies["olympus-oracle-fusion-v4"]["path"])
    v5 = _read(ROOT / studies["olympus-robust-fusion-v5"]["path"])
    assert studies["olympus-oracle-fusion-v4"]["phase"] == v4["phase"]
    assert v4["scientific_result"].startswith("PRIMARY_NOT_SUPPORTED")
    assert studies["olympus-robust-fusion-v5"]["phase"] == v5["phase"]
    assert v5["promotion_authorized"] is False


def test_security_policy_tracks_current_minor_version() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    supported = ".".join(version.split(".")[:2]) + ".x"
    assert f"`{supported}`" in (ROOT / "SECURITY.md").read_text(encoding="utf-8")


def test_source_archive_includes_canonical_research_status_graph() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    required = {
        "include RESEARCH_STATE.json",
        "include research/llm_readiness_v1/CURRENT_REPORT.json",
        "include research/olympus_editclevr_v1/RESEARCH_STATE.json",
        "include research/olympus_irp_successor_v2/RESEARCH_STATE.json",
        "include research/olympus_irp_successor_v3/RESEARCH_STATE.json",
        "include research/olympus_irp_successor_v4/RESEARCH_STATE.json",
        "include research/olympus_irp_successor_v5/RESEARCH_STATE.json",
        "include research/ENGINEERING_RESEARCH_STATUS_HARDENING_20261018.json",
        "include research/olympus_consolidated_2026/RESEARCH_STATE.json",
        "include research/olympus_consolidated_2026/RELEASE_RECEIPT_20261018.json",
        "include research/seven_model_engineering_v1/RESEARCH_STATE.json",
        "include research/seven_model_engineering_v1/RECEIPT.json",
        "include research/flagship_engineering_v2/RESEARCH_STATE.json",
        "include research/flagship_engineering_v2/RECEIPT.json",
        "include research/decision_dossier_replay_v1/RESEARCH_STATE.json",
        "include research/DECISION_DOSSIER_REPLAY_HARDENING_20261020.json",
    }
    assert required <= set(manifest)


def _copy_status_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "repository"
    state = _read(ROOT / "RESEARCH_STATE.json")
    paths = [
        Path("research/REALITY_LEDGER.md"),
        Path(state["large_scale_readiness_v1"]["current_report"]),
        *(Path(study["path"]) for study in state["studies"]),
    ]
    for study in state["engineering_studies"]:
        paths.append(Path(study["path"]))
        receipt_path = Path(study["receipt"])
        paths.append(receipt_path)
        paths.extend(Path(path) for path in _read(ROOT / receipt_path)["source_sha256"])
    for relative in paths:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    fixture_state = dict(state)
    fixture_state["artifacts"] = []
    (target / "RESEARCH_STATE.json").write_text(json.dumps(fixture_state), encoding="utf-8")
    # The fixture deliberately strips artifact inventory. Bind that new fixture
    # root, leaving immutable historical receipt pins untouched.
    for study in state["engineering_studies"]:
        if study.get("source_verification", "current_files") != "current_files":
            continue
        receipt_path = target / study["receipt"]
        receipt = _read(receipt_path)
        if "RESEARCH_STATE.json" in receipt["source_sha256"]:
            receipt["source_sha256"]["RESEARCH_STATE.json"] = hashlib.sha256(
                (target / "RESEARCH_STATE.json").read_bytes()
            ).hexdigest()
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return target


def test_canonical_research_status_audit_is_fail_closed_and_non_promotional() -> None:
    status = audit_research_state(ROOT)
    assert status.families_ready == 0
    assert status.families_total == 6
    assert status.all_families_ready is False
    assert status.model_promoted is False
    assert status.release_ready is False
    assert status.production_ready is False
    assert status.pending_archive_studies == (
        "olympus-robust-fusion-v5",
        "olympus-editclevr-v1",
    )
    studies = {study.study_id: study for study in status.studies}
    assert studies["olympus-robust-fusion-v5"].routing_primary_supported is False
    assert studies["olympus-robust-fusion-v5"].temporal_primary_supported is True
    assert studies["olympus-editclevr-v1"].primary_supported is False
    assert status.engineering_studies[0].study_id == "seven-model-engineering-v1"
    assert status.engineering_studies[0].phase == (
        "LOCAL_ENGINEERING_VERIFIED_HOSTED_EXACT_SOURCE_PENDING"
    )
    assert status.engineering_studies[0].source_verification == "historical_receipt"
    assert status.engineering_studies[1].study_id == "flagship-engineering-v2"
    assert status.engineering_studies[1].source_verification == "historical_receipt"
    assert status.engineering_studies[2].study_id == "decision-dossier-replay-v1"
    assert status.engineering_studies[2].source_verification == "historical_receipt"
    assert status.engineering_studies[3].study_id == "aion-executor-v1"
    assert status.engineering_studies[3].source_verification == "current_files"


def test_research_status_rejects_current_aion_source_drift(tmp_path: Path) -> None:
    target = _copy_status_fixture(tmp_path)
    audit_research_state(target)
    source = target / "olympus/evaluation/aion_executor.py"
    source.write_text(source.read_text() + "\n# changed after receipt\n")
    with pytest.raises(
        ResearchStateError,
        match="source hash mismatch: olympus/evaluation/aion_executor.py",
    ):
        audit_research_state(target)


def test_research_status_pins_historical_dossier_receipt_without_claiming_current_source(
    tmp_path: Path,
) -> None:
    target = _copy_status_fixture(tmp_path)
    source = target / "olympus/evaluation/decision_dossiers.py"
    source.write_text(source.read_text() + "\n# later implementation revision\n")
    status = audit_research_state(target)
    dossier = next(
        study for study in status.engineering_studies
        if study.study_id == "decision-dossier-replay-v1"
    )
    assert dossier.source_verification == "historical_receipt"

    receipt_path = target / dossier.receipt
    receipt = _read(receipt_path)
    receipt["unexpected_post_hoc_edit"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ResearchStateError, match="historical receipt hash mismatch"):
        audit_research_state(target)


def test_research_status_rejects_root_child_phase_drift(tmp_path: Path) -> None:
    target = _copy_status_fixture(tmp_path)
    child_path = target / "research/olympus_irp_successor_v5/RESEARCH_STATE.json"
    child = _read(child_path)
    child["phase"] = "CONTRADICTORY_PHASE"
    child_path.write_text(json.dumps(child), encoding="utf-8")

    with pytest.raises(ResearchStateError, match="phase mismatch"):
        audit_research_state(target)


def test_research_status_rejects_false_archive_retrieval_claim(tmp_path: Path) -> None:
    target = _copy_status_fixture(tmp_path)
    child_path = target / "research/olympus_editclevr_v1/RESEARCH_STATE.json"
    child = _read(child_path)
    child["remote_archive"]["retrieved_to_repository"] = True
    child_path.write_text(json.dumps(child), encoding="utf-8")

    with pytest.raises(ResearchStateError, match="pending-retrieval phase"):
        audit_research_state(target)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("promotion_flag", "model_promoted must remain false"),
        ("duplicate_study", "duplicate study id"),
        ("path_escape", "escapes the repository"),
        ("child_promotion", "unexpectedly authorizes promotion"),
        ("temporal_mismatch", "temporal_primary_supported mismatch"),
        ("archive_hash", "archive requires a lowercase SHA-256"),
        ("verifier_failure", "pending archive requires verifier PASS"),
        ("duplicate_family", "duplicate readiness family id"),
        ("all_ready_drift", "all_families_ready contradicts"),
        ("root_readiness_drift", "root readiness status must be"),
        ("missing_archive_gate", "missing from the root blocking gates"),
        ("missing_ledger_correction", "without the required append-only"),
        ("engineering_phase", "engineering phase mismatch"),
        ("engineering_hash", "receipt.*hash mismatch"),
    ],
)
def test_research_status_rejects_adversarial_drift(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    target = _copy_status_fixture(tmp_path)
    root_path = target / "RESEARCH_STATE.json"
    root = _read(root_path)
    v5_path = target / "research/olympus_irp_successor_v5/RESEARCH_STATE.json"
    v5 = _read(v5_path)
    report_path = target / root["large_scale_readiness_v1"]["current_report"]
    report = _read(report_path)

    if case == "promotion_flag":
        root["software"]["model_promoted"] = True
    elif case == "duplicate_study":
        root["studies"].append(dict(root["studies"][0]))
    elif case == "path_escape":
        root["studies"][0]["path"] = "../outside.json"
    elif case == "child_promotion":
        v5["promotion_authorized"] = True
    elif case == "temporal_mismatch":
        root["studies"][3]["temporal_primary_supported"] = False
    elif case == "archive_hash":
        v5["remote_archive"]["sha256"] = "not-a-digest"
    elif case == "verifier_failure":
        v5["verifier"]["status"] = "FAIL"
    elif case == "duplicate_family":
        report["family_reports"][1]["family_id"] = report["family_reports"][0]["family_id"]
    elif case == "all_ready_drift":
        report["all_families_ready"] = True
    elif case == "root_readiness_drift":
        root["large_scale_readiness_v1"]["status"] = "6_OF_6_FAMILIES_READY"
    elif case == "missing_archive_gate":
        root["blocking_gates"] = [gate for gate in root["blocking_gates"] if "archives" not in gate]
    elif case == "missing_ledger_correction":
        ledger_path = target / "research/REALITY_LEDGER.md"
        ledger = ledger_path.read_text(encoding="utf-8")
        ledger_path.write_text(
            ledger.split("## 2026-10-18 canonical completion correction", 1)[0],
            encoding="utf-8",
        )
    elif case == "engineering_phase":
        root["engineering_studies"][0]["phase"] = "WRONG"
    elif case == "engineering_hash":
        receipt_path = target / root["engineering_studies"][0]["receipt"]
        receipt = _read(receipt_path)
        first_path = next(iter(receipt["source_sha256"]))
        receipt["source_sha256"][first_path] = "0" * 64
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    else:
        raise AssertionError(f"unhandled case: {case}")

    root_path.write_text(json.dumps(root), encoding="utf-8")
    v5_path.write_text(json.dumps(v5), encoding="utf-8")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ResearchStateError, match=message):
        audit_research_state(target)
