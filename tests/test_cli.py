from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from typer.testing import CliRunner

from olympus.cli import app
from olympus.data.olympus_dataset import write_olympus_dataset_manifest
from olympus.foundry.resources import MemorySnapshot
from olympus.models.olympus_system import OlympusSystemConfig

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[1]


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    project = workspace / "demo"
    project.mkdir(parents=True)
    (project / ".git").mkdir()
    (project / "README.md").write_text("# Demo\n\nExecutable test project.\n", encoding="utf-8")
    (project / "labos.project.yaml").write_text(
        f"""
project_id: demo
title: Demo
description: Executable test project.
hypothesis: The command completes.
root_path: .
domain: testing
language: python
entry_points:
  - name: smoke
    command: {sys.executable} -c pass
    profile: smoke
    expected_outputs: []
datasets: []
expected_outputs: []
validation_commands: []
tags: []
""".strip(),
        encoding="utf-8",
    )
    return workspace, tmp_path / "artifacts"


def _invoke(*arguments: str) -> str:
    result = runner.invoke(app, list(arguments))
    assert result.exit_code == 0, result.output
    return result.output


def test_research_status_command_reports_evidence_bound_portfolio() -> None:
    status = json.loads(_invoke("research-status", "--root", str(ROOT)))
    assert status["families_ready"] == 0
    assert status["families_total"] == 6
    assert status["model_promoted"] is False
    assert status["pending_archive_studies"] == [
        "olympus-robust-fusion-v5",
        "olympus-editclevr-v1",
    ]


def test_resource_status_fails_closed_without_traceback(monkeypatch: object) -> None:
    def unavailable() -> MemorySnapshot:
        raise RuntimeError("resource inspection command failed: sysctl")

    monkeypatch.setattr("olympus.cli.memory_snapshot", unavailable)  # type: ignore[attr-defined]
    result = runner.invoke(app, ["foundry", "resource-status"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload == {
        "status": "RESOURCE_INSPECTION_UNAVAILABLE",
        "workload_admitted": False,
        "reason": "resource inspection command failed: sysctl",
    }
    assert "Traceback" not in result.output


def test_resource_status_reports_snapshot_without_admitting_workload(monkeypatch: object) -> None:
    snapshot = MemorySnapshot(
        total_bytes=16 * 1024**3,
        available_bytes=4 * 1024**3,
        swap_total_bytes=8 * 1024**3,
        swap_used_bytes=4 * 1024**3,
        process_peak_rss_bytes=256 * 1024**2,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "olympus.cli.memory_snapshot",
        lambda: snapshot,
    )
    status = json.loads(_invoke("foundry", "resource-status"))
    assert status["status"] == "RESOURCE_INSPECTION_AVAILABLE"
    assert status["workload_admitted"] is None
    assert status["swap_fraction"] == 0.5


def test_readiness_check_can_enforce_unready_exit_status(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    output = tmp_path / "report.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": "olympus-six-family-readiness-v1",
                "candidates": [],
            }
        ),
        encoding="utf-8",
    )

    report_only = runner.invoke(
        app,
        ["foundry", "readiness-check", str(evidence), "--output", str(output)],
    )
    assert report_only.exit_code == 0, report_only.output
    assert json.loads(output.read_text(encoding="utf-8"))["all_families_ready"] is False

    enforced = runner.invoke(
        app,
        [
            "foundry",
            "readiness-check",
            str(evidence),
            "--output",
            str(output),
            "--require-ready",
        ],
    )
    assert enforced.exit_code == 1


def test_readiness_schema_commands_write_valid_schemas(tmp_path: Path) -> None:
    evidence_schema = tmp_path / "evidence.schema.json"
    report_schema = tmp_path / "report.schema.json"
    receipt_schema = tmp_path / "receipt.schema.json"
    trust_schema = tmp_path / "trust.schema.json"

    evidence_result = runner.invoke(
        app,
        ["foundry", "readiness-schema", str(evidence_schema)],
    )
    assert evidence_result.exit_code == 0, evidence_result.output
    assert json.loads(evidence_schema.read_text(encoding="utf-8"))["title"] == (
        "ReadinessEvidenceSet"
    )

    report_result = runner.invoke(
        app,
        ["foundry", "readiness-report-schema", str(report_schema)],
    )
    assert report_result.exit_code == 0, report_result.output
    assert json.loads(report_schema.read_text(encoding="utf-8"))["title"] == (
        "PortfolioReadinessReport"
    )

    receipt_result = runner.invoke(
        app,
        ["foundry", "readiness-receipt-schema", str(receipt_schema)],
    )
    assert receipt_result.exit_code == 0, receipt_result.output
    assert "anyOf" in json.loads(receipt_schema.read_text(encoding="utf-8"))

    trust_result = runner.invoke(
        app,
        ["foundry", "readiness-trust-schema", str(trust_schema)],
    )
    assert trust_result.exit_code == 0, trust_result.output
    assert json.loads(trust_schema.read_text(encoding="utf-8"))["title"] == (
        "ReceiptTrustPolicy"
    )


def test_forge_workspace_and_demo_commands(tmp_path: Path) -> None:
    compile_output = _invoke(
        "forge",
        "compile",
        "Maintain interpretations, verify them, and merge.",
    )
    assert "graph TD" in compile_output

    run_output = _invoke(
        "forge",
        "run",
        "Maintain interpretations, verify them, and merge.",
        "The graph looked like a forest.",
    )
    assert "statement" in json.loads(run_output)["output"]

    checkpoint = tmp_path / "workspace.json"
    saved = _invoke(
        "workspace",
        "Verify the interpretation",
        "The graph looked like a forest.",
        "--output",
        str(checkpoint),
    )
    assert "Saved workspace" in saved
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["objective"] == (
        "Verify the interpretation"
    )

    inline = _invoke(
        "workspace",
        "Verify the interpretation",
        "The graph looked like a forest.",
    )
    assert json.loads(inline)["objective"] == "Verify the interpretation"

    demos = json.loads(_invoke("demo", "run-all"))
    assert demos["forge"]["passed"] is True


def test_bu1ld_specifications_command_is_explicitly_non_promotional() -> None:
    specifications = json.loads(_invoke("bu1ld", "specs"))
    assert [item["family_id"] for item in specifications] == [
        "hermes",
        "prometheus",
        "perseus",
        "atlas",
        "kronos",
        "aion",
    ]
    assert all(item["capability_claimed"] is False for item in specifications)

    references = json.loads(_invoke("bu1ld", "reference-models"))
    assert [item["family_id"] for item in references] == [
        "hermes",
        "prometheus",
        "perseus",
        "atlas",
        "kronos",
        "aion",
    ]
    assert all(item["reference_parameters"] > 0 for item in references)
    assert all(item["status"] == "REFERENCE_IMPLEMENTATION_NOT_PROMOTED" for item in references)
    assert all(item["target_capability_claimed"] is False for item in references)

    seven_models = json.loads(_invoke("bu1ld", "seven-model-status"))
    assert [item["model_id"] for item in seven_models] == [
        "hermes",
        "prometheus",
        "perseus",
        "atlas",
        "kronos",
        "aion",
        "jepa_upscaler",
    ]
    assert all(
        item["status"] == "ENGINEERING_REFERENCE_LOCAL_VERIFIED_NOT_PROMOTED"
        for item in seven_models
    )
    assert all(item["contract_execution_required"] is True for item in seven_models)
    assert all(item["promotion_authorized"] is False for item in seven_models)

    contracts = json.loads(_invoke("bu1ld", "seven-model-contract"))
    assert all(item["forward_contract"] == "PASS" for item in contracts)
    assert all(item["backward_contract"] == "PASS" for item in contracts)
    assert all(item["gradient_coverage"] == "PASS" for item in contracts)


def test_aion_executor_demo_enforces_host_policy() -> None:
    executed = json.loads(_invoke("bu1ld", "aion-executor-demo"))
    assert executed["status"] == "AION_HOST_GOVERNED_EXECUTION_FIXTURE_NOT_PROMOTED"
    assert executed["trace"]["status"] == "completed"
    assert executed["tool_authority_granted"] is False

    stopped = json.loads(
        _invoke("bu1ld", "aion-executor-demo", "--selected-stage", "1")
    )
    assert stopped["trace"]["status"] == "stopped"
    assert stopped["trace"]["stop_reason"] == "selected_transition_illegal"


def test_bu1ld_composed_training_command_runs_end_to_end(
    tmp_path: Path, monkeypatch: object
) -> None:
    config = OlympusSystemConfig()
    batch_size = 2
    batch = {
        "tokens": torch.ones((batch_size, 4), dtype=torch.long),
        "attention_mask": torch.ones((batch_size, 4), dtype=torch.bool),
        "image": torch.rand(batch_size, 3, 8, 8),
        "states": torch.rand(batch_size, 3, 16),
        "actions": torch.rand(batch_size, 3, 8),
        "temporal_mask": torch.ones((batch_size, 3), dtype=torch.bool),
        "allowed_transitions": torch.ones((batch_size, 8), dtype=torch.bool),
        "risk_score": torch.tensor([0.1, 0.9]),
        "decision_targets": torch.tensor([0, 1]),
        "transition_targets": torch.tensor([0, 7]),
        "next_state_targets": torch.rand(batch_size, 3, 16),
        "intrinsic_risk_targets": torch.tensor([0.1, 0.9]),
        "domain_ids": torch.tensor([0, 1]),
        "timestamp": torch.tensor([0.0, 1.0]),
    }
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    torch.save(batch, dataset / "train.pt")
    torch.save(batch, dataset / "evaluation.pt")
    manifest = dataset / "manifest.json"
    write_olympus_dataset_manifest(
        manifest,
        dataset_id="cli-test",
        version="1.0.0",
        license_name="LicenseRef-Proprietary",
        train_batch=dataset / "train.pt",
        evaluation_batch=dataset / "evaluation.pt",
        train_source_ids=["train-0", "train-1"],
        evaluation_source_ids=["evaluation-0", "evaluation-1"],
    )
    model_config = tmp_path / "model.json"
    model_config.write_text(config.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "olympus.foundry.resources.memory_snapshot",
        lambda: MemorySnapshot(
            total_bytes=16 * 1024**3,
            available_bytes=8 * 1024**3,
            swap_total_bytes=1024**3,
            swap_used_bytes=0,
            process_peak_rss_bytes=1024**2,
        ),
    )
    output = tmp_path / "run"
    result = json.loads(
        _invoke(
            "bu1ld",
            "train-olympus-system",
            str(manifest),
            str(model_config),
            str(output),
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--workload-lock",
            str(tmp_path / "training.lock"),
        )
    )
    assert result["status"] == "COMPLETED_DEVELOPMENT_NOT_PROMOTED"
    assert result["promotion_authorized"] is False
    assert (output / "checkpoint.pt").is_file()
    assert (output / "decision-dossiers.json").is_file()
    assert (output / "report.json").is_file()
    verified = json.loads(
        _invoke(
            "bu1ld",
            "verify-olympus-dossiers",
            str(output),
            str(manifest),
        )
    )
    assert verified["status"] == "VERIFIED_DECISION_DOSSIER_REPLAY_NOT_PROMOTED"
    assert verified["examples"] == 2
    assert verified["promotion_authorized"] is False


def test_labos_read_run_and_report_commands(tmp_path: Path) -> None:
    workspace, artifacts = _workspace(tmp_path)
    common = ("--workspace", str(workspace), "--artifacts", str(artifacts))

    assert "demo" in _invoke("labos", "discover", *common)
    assert "demo" in _invoke("labos", "list", *common)
    assert "demo" in _invoke("labos", "validate", *common)
    assert "smoke_tested" in _invoke("labos", "run-project", "demo", *common)
    assert "smoke_tested" in _invoke("labos", "run-group", "demo", *common)
    assert "smoke_tested" in _invoke("labos", "run-smoke", *common, "--project", "demo")
    assert "smoke_tested" in _invoke("labos", "run-all", *common, "--profile", "smoke")
    assert "total_projects" in _invoke("labos", "aggregate-metrics", *common)
    assert '"demo"' in _invoke("labos", "compare-runs", *common)
    assert "No transient-looking" in _invoke("labos", "retry-transient", *common)
    assert "No blocked dependencies" in _invoke("labos", "list-blocked", *common)

    output_root = tmp_path / "reports"
    generated = _invoke(
        "labos",
        "generate-report",
        *common,
        "--output-root",
        str(output_root),
    )
    assert "Wrote reports to" in generated
    assert (output_root / "portfolio_status.json").is_file()


def test_labos_resume_failed_reports_empty_history(tmp_path: Path) -> None:
    workspace, artifacts = _workspace(tmp_path)
    output = _invoke(
        "labos",
        "resume-failed",
        "--workspace",
        str(workspace),
        "--artifacts",
        str(artifacts),
    )
    assert "No failed runs recorded" in output
