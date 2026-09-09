import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _module(relative_path: str, name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_v4_machine_evidence_is_byte_pinned():
    expected = {
        "PUBLICATION_READINESS.json": (
            "57b9ea892c03f7b2edf18dfd653bad715b1698150784af650fdb3d3e3abe07ae"
        ),
        "configs/local_ood_protocol.json": (
            "3200a1162b677c0b71bea5eb995b72969b7dfec88a99e32bf14883d83c15e420"
        ),
        "audit/FINAL_INTEGRITY_MANIFEST.json": (
            "c5daa5e7e05e778a0f05b7f8a7e47bca5464f7bb0955670bdfb418c67fd73767"
        ),
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest

    status = (ROOT / "audit" / "FINAL_INTEGRITY_MANIFEST_STATUS.md").read_text()
    assert "historical and superseded" in status
    assert "must not be used as" in status
    assert "current whole-tree integrity result" in status


def test_v4_verifier_reports_source_drift_without_mutating_frozen_protocol():
    protocol = ROOT / "configs" / "local_ood_protocol.json"
    before = protocol.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "write_local_ood_protocol.py"),
            "--manifest",
            "external/corebench/manifest.jsonl",
            "--protocol",
            "configs/local_ood_protocol.json",
            "--verify",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 4
    assert json.loads(result.stderr)["status"] == "V4_SOURCE_DRIFT_REQUIRES_VERSIONED_PROTOCOL"
    assert protocol.read_bytes() == before


def test_v5_source_scope_includes_analysis_and_poststudy_code():
    module = _module("scripts/write_local_ood_protocol.py", "protocol_source_scope")
    relative = {str(path.relative_to(ROOT)) for path in module.source_files()}
    assert "analysis/analyze_external_rigor.py" in relative
    assert "poststudy/scientific_readiness_gate.py" in relative
    assert "tests/test_publication_workflow.py" in relative


def test_structural_writer_cannot_overwrite_historical_readiness(tmp_path):
    module = _module("scripts/publication_gate.py", "publication_gate")
    historical_json = tmp_path / "PUBLICATION_READINESS.json"
    historical_md = tmp_path / "PUBLICATION_READINESS.md"
    historical_json.write_text("frozen-json\n")
    historical_md.write_text("current-wrapper\n")
    result = {
        "gate_scope": "artifact_completeness_only",
        "verdict": "ARTIFACT_COMPLETE",
        "artifact_complete": True,
        "scientific_readiness_evaluated": False,
        "publication_ready": False,
        "checks": [{"check": "fixture", "pass": True, "detail": "complete"}],
        "interpretation": "Scientific evaluation is separate.",
    }

    module.write_assessment(result, tmp_path / "poststudy" / "STRUCTURAL_READINESS.json")

    written = json.loads((tmp_path / "poststudy" / "STRUCTURAL_READINESS.json").read_text())
    assert written["verdict"] == "ARTIFACT_COMPLETE"
    assert written["publication_ready"] is False
    assert historical_json.read_text() == "frozen-json\n"
    assert historical_md.read_text() == "current-wrapper\n"


def test_structural_output_is_byte_identical_across_runtime_variation(tmp_path):
    module = _module("scripts/publication_gate.py", "publication_gate_determinism")
    first = module.stable_detail("........................ [100%]\n26 passed in 1.23s")
    second = module.stable_detail("........................ [100%]\n26 passed in 19.87s")
    assert first == second == "26 passed"
    result = {
        "gate_scope": "artifact_completeness_only",
        "verdict": "ARTIFACT_COMPLETE",
        "artifact_complete": True,
        "scientific_readiness_evaluated": False,
        "publication_ready": False,
        "checks": [{"check": "tests", "pass": True, "detail": first}],
        "interpretation": "Scientific evaluation is separate.",
    }
    output = tmp_path / "poststudy" / "STRUCTURAL_READINESS.json"
    module.write_assessment(result, output)
    first_bytes = output.read_bytes()
    module.write_assessment(result, output)
    assert output.read_bytes() == first_bytes


def test_scientific_publication_mode_blocks_failed_capability_floor(tmp_path):
    questions = tmp_path / "questions.csv"
    tasks = tmp_path / "tasks.csv"
    structural = tmp_path / "structural.json"
    output = tmp_path / "scientific.json"
    pd.DataFrame({"agent_a_answered": [0] * 20, "agent_b_answered": [0] * 20}).to_csv(
        questions, index=False
    )
    pd.DataFrame(
        {
            "agent_a_submitted_report": [False] * 17,
            "agent_b_submitted_report": [False] * 17,
        }
    ).to_csv(tasks, index=False)
    structural.write_text(json.dumps({"artifact_complete": True, "verdict": "ARTIFACT_COMPLETE"}))

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "poststudy" / "scientific_readiness_gate.py"),
            "--questions",
            str(questions),
            "--tasks",
            str(tasks),
            "--legacy-gate",
            str(structural),
            "--output",
            str(output),
            "--require-capability-floor",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 4
    assessment = json.loads(output.read_text())
    assert assessment["verdict"] == "NEGATIVE_RESULT_ARTIFACT_COMPLETE_CAPABILITY_FLOOR_FAILED"
    assert assessment["capability_floor_passed"] is False


def test_scoped_integrity_validator_detects_mutation(tmp_path):
    module = _module("scripts/scoped_integrity.py", "scoped_integrity")
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence" / "a.json").write_text("a\n")
    (tmp_path / "evidence" / "b.json").write_text("b\n")
    specs = {"fixture": (("evidence/*.json",), 2)}
    manifest = module.build_manifest(tmp_path, specs)
    assert module.validate_manifest(manifest, tmp_path, specs)["status"] == "PASS"

    (tmp_path / "evidence" / "b.json").write_text("changed\n")
    failed = module.validate_manifest(manifest, tmp_path, specs)
    assert failed["status"] == "FAIL"
    assert failed["mismatched_files"] == ["evidence/b.json"]


def test_execution_workflows_require_protocol_and_scientific_gate():
    for relative in ["scripts/run_local_ood_study.sh", "scripts/run_final_external.sh"]:
        source = (ROOT / relative).read_text()
        protocol = source.index("scripts/write_local_ood_protocol.py")
        execution = source.index("scripts/run_external_study.py")
        rigorous = source.index("analysis/analyze_external_rigor.py")
        structural = source.index("scripts/publication_gate.py")
        scientific = source.index("poststudy/scientific_readiness_gate.py")
        assert protocol < execution < rigorous < structural < scientific
        assert "--require-capability-floor" in source
        assert "publication_gate.py || true" not in source
        assert "PANTHEON_PROTOCOL" not in source

    makefile = (ROOT / "Makefile").read_text()
    assert "readiness" in makefile
    assert "integrity" in makefile
