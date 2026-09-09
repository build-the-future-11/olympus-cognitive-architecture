from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from olympus.cli import app
from olympus.foundry.promotion import PromotionReport

runner = CliRunner()


@pytest.mark.parametrize("passed, exit_code", [(True, 0), (False, 1)])
def test_promotion_cli_exit_codes_follow_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passed: bool, exit_code: int
) -> None:
    report = PromotionReport(
        requested_model_id="fixture-model", checkpoint_sha256="a" * 64,
        gates=[], passed=passed, status="PROMOTED" if passed else "NOT_PROMOTED",
        release_manifest_path=None, blockers=[] if passed else ["task_quality"],
    )
    monkeypatch.setattr("olympus.cli.evaluate_promotion", lambda **kwargs: report)
    result = runner.invoke(app, [
        "foundry", "promotion-check", "fixture-model",
        *[str(tmp_path / name) for name in (
            "checkpoint.pt", "manifest.json", "evaluation.json", "quantization.json",
            "MODEL_CARD.md", "promotion.json",
        )], "Apache-2.0",
    ])
    assert result.exit_code == exit_code
    assert json.loads(result.output)["passed"] is passed


def test_promotion_cli_reports_unreadable_evidence_without_traceback(tmp_path: Path) -> None:
    result = runner.invoke(app, [
        "foundry", "promotion-check", "fixture-model",
        *[str(tmp_path / name) for name in (
            "checkpoint.pt", "manifest.json", "evaluation.json", "quantization.json",
            "MODEL_CARD.md", "promotion.json",
        )],
        "Apache-2.0",
    ])
    assert result.exit_code == 2
    assert "unreadable" in result.output
    assert "No new decision was produced" in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "promotion.json").exists()
    attempts = list((tmp_path / "promotion.json.runs").glob("*/invalid-input.json"))
    assert len(attempts) == 1
    assert json.loads(attempts[0].read_text())["status"] == "INVALID_INPUT"


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


def test_promotion_cli_rejects_ambiguous_serving_evidence(tmp_path: Path) -> None:
    serving = tmp_path / "serving.json"
    serving.write_text(
        '{"passed":true,"passed":false,"checkpoint_sha256":"' + "a" * 64 + '"}',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "foundry",
            "promotion-check",
            "fixture-model",
            str(tmp_path / "checkpoint.pt"),
            str(tmp_path / "manifest.json"),
            str(tmp_path / "evaluation.json"),
            str(tmp_path / "quantization.json"),
            str(tmp_path / "MODEL_CARD.md"),
            str(tmp_path / "promotion.json"),
            "Apache-2.0",
            "--serving-verification",
            str(serving),
        ],
    )

    assert result.exit_code == 2
    assert "duplicate JSON key is forbidden: passed" in result.output
