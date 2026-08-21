from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from olympus.cli import app

runner = CliRunner()


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
