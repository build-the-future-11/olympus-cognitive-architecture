from __future__ import annotations

from pathlib import Path

from olympus.labos.discovery import discover_projects, infer_manifest
from olympus.labos.manifest import (
    DatasetDependency,
    EntryPoint,
    ManifestSource,
    ProjectLanguage,
    ProjectManifest,
    ProjectRecord,
    ProjectStatus,
    ResourceProfile,
    ResourceRequirements,
    ValidationResult,
)
from olympus.labos.portfolio import PortfolioService
from olympus.labos.reports import PortfolioReporter
from olympus.labos.runner import RunResult, classify_run, status_from_run
from olympus.labos.scheduler import ResourceAwareScheduler
from olympus.labos.validation import validate_project


def test_infer_manifest_from_python_repo(tmp_path: Path) -> None:
    repo = tmp_path / "SampleRepo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "README.md").write_text(
        "# Sample Repo\n\nA compact research system.\n\nHypothesis paragraph.\n",
        encoding="utf-8",
    )
    (repo / "pyproject.toml").write_text("[project]\nname='sample-repo'\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / ".venv").mkdir()

    manifest = infer_manifest(repo)

    assert manifest.project_id == "sample-repo"
    assert manifest.language == ProjectLanguage.PYTHON
    assert manifest.manifest_source == ManifestSource.INFERRED
    assert any(entry.profile == ResourceProfile.SMOKE for entry in manifest.entry_points)


def test_infer_manifest_for_src_only_repo_adds_pythonpath_smoke_command(tmp_path: Path) -> None:
    repo = tmp_path / "Ascension"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "README.md").write_text(
        "# Project Ascension\n\nScientific computing kernels.\n",
        encoding="utf-8",
    )
    (repo / "pyproject.toml").write_text("[project]\nname='project-ascension'\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "tests").mkdir()

    manifest = infer_manifest(repo)

    assert manifest.entry_points
    assert manifest.entry_points[0].env["PYTHONPATH"] == "src"
    assert "pytest" in manifest.entry_points[0].command


def test_discover_projects_reads_declared_manifest(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo = workspace / "demo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "labos.project.yaml").write_text(
        """
project_id: demo
title: Demo
description: Demo description
hypothesis: Demo hypothesis
root_path: ignored
domain: testing
language: python
manifest_source: declared
entry_points: []
datasets: []
expected_outputs: []
validation_commands: []
tags: []
resource_requirements:
  profile: smoke
  cpu_only: true
  gpu_optional: false
  memory_gb: 1
  notes: test
metadata: {}
""".strip(),
        encoding="utf-8",
    )
    (repo / "README.md").write_text("# Demo\n", encoding="utf-8")

    manifests = discover_projects(workspace)

    assert manifests[0].project_id == "demo"
    assert manifests[0].manifest_source == ManifestSource.DECLARED


def test_validation_detects_missing_readme(tmp_path: Path) -> None:
    repo = tmp_path / "no_readme"
    repo.mkdir()
    manifest = ProjectManifest(
        project_id="missing-readme",
        title="Missing README",
        description="desc",
        hypothesis="hyp",
        root_path=str(repo),
        domain="testing",
        language=ProjectLanguage.PYTHON,
        entry_points=[],
        validation_commands=[],
    )
    result = validate_project(manifest)
    assert result.valid is True
    assert any("README" in issue.message for issue in result.issues)


def test_scheduler_prioritizes_declared_olympus_like_projects() -> None:
    manifest = ProjectManifest(
        project_id="olympus",
        title="Olympus",
        description="desc",
        hypothesis="hyp",
        root_path="/tmp/olympus",
        domain="agentic-ml",
        language=ProjectLanguage.PYTHON,
        manifest_source=ManifestSource.DECLARED,
        entry_points=[
            EntryPoint(
                name="smoke",
                command="pytest",
                profile=ResourceProfile.SMOKE,
            )
        ],
        resource_requirements=ResourceRequirements(),
    )
    tasks = ResourceAwareScheduler().schedule([manifest], ResourceProfile.SMOKE)
    assert tasks[0].project_id == "olympus"
    assert tasks[0].priority >= 5


def test_reporter_writes_files(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    reporter = PortfolioReporter(artifact_root)
    record = ProjectRecord(
        manifest=ProjectManifest(
            project_id="demo",
            title="Demo",
            description="desc",
            hypothesis="hyp",
            root_path=str(tmp_path / "demo"),
            domain="testing",
            language=ProjectLanguage.PYTHON,
        ),
        validation=ValidationResult(project_id="demo", valid=True, issues=[]),
        status=ProjectStatus.DISCOVERED_ONLY,
        summary="summary",
    )
    reporter.write_status_json([record], tmp_path / "portfolio_status.json")
    reporter.write_completion_report([record], tmp_path / "PORTFOLIO_COMPLETION_REPORT.md")
    reporter.write_remaining_actions([record], tmp_path / "REMAINING_EXTERNAL_ACTIONS.md")
    assert (tmp_path / "portfolio_status.json").exists()
    assert (tmp_path / "PORTFOLIO_COMPLETION_REPORT.md").exists()
    assert (tmp_path / "REMAINING_EXTERNAL_ACTIONS.md").exists()


def test_portfolio_service_discovers_and_reports(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo = workspace / "olympus"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "README.md").write_text("# Olympus\n\nTest repo.\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='olympus'\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / ".venv").mkdir()
    service = PortfolioService(workspace, tmp_path / "artifacts")
    records = service.discover()
    assert len(records) == 1
    service.write_reports(records, tmp_path)
    assert (tmp_path / "portfolio_status.json").exists()


def test_smoke_result_with_dataset_dependency_is_upgraded(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    service = PortfolioService(tmp_path, artifact_root)
    record = ProjectRecord(
        manifest=ProjectManifest(
            project_id="demo",
            title="Demo",
            description="desc",
            hypothesis="hyp",
            root_path=str(tmp_path),
            domain="testing",
            language=ProjectLanguage.PYTHON,
            datasets=[
                DatasetDependency(
                    name="manual-data",
                    required=True,
                    notes="manual",
                    blocked_by_user_action=True,
                )
            ],
        ),
        validation=ValidationResult(project_id="demo", valid=True, issues=[]),
        status=ProjectStatus.SMOKE_TESTED,
        summary="done",
    )
    service.runner.run_db.insert(
        RunResult(
            project_id="demo",
            command="pytest",
            return_code=0,
            stdout="ok",
            stderr="",
            status=ProjectStatus.SMOKE_TESTED,
            started_at="2026-07-15T00:00:00+00:00",
            finished_at="2026-07-15T00:00:01+00:00",
            profile=ResourceProfile.SMOKE.value,
            cwd=str(tmp_path),
            classification="ok",
        )
    )
    updated = service._attach_run_history([record])
    assert updated[0].status == ProjectStatus.FULL_BENCHMARK_PENDING_COMPUTE


def test_failed_run_history_records_precise_blocked_reason(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    service = PortfolioService(tmp_path, artifact_root)
    record = ProjectRecord(
        manifest=ProjectManifest(
            project_id="demo",
            title="Demo",
            description="desc",
            hypothesis="hyp",
            root_path=str(tmp_path),
            domain="testing",
            language=ProjectLanguage.PYTHON,
        ),
        validation=ValidationResult(project_id="demo", valid=True, issues=[]),
        status=ProjectStatus.DISCOVERED_ONLY,
        summary="done",
    )
    service.runner.run_db.insert(
        RunResult(
            project_id="demo",
            command="pytest",
            return_code=1,
            stdout=(
                "FAILED tests/api/test_app.py::test_log_path\n"
                ">       assert canonical_log.is_file()\n"
                "E       AssertionError: assert False\n"
            ),
            stderr="",
            status=ProjectStatus.DISCOVERED_ONLY,
            started_at="2026-07-15T00:00:00+00:00",
            finished_at="2026-07-15T00:00:01+00:00",
            profile=ResourceProfile.SMOKE.value,
            cwd=str(tmp_path),
            classification="failed_command",
        )
    )
    updated = service._attach_run_history([record])
    assert (
        updated[0].blocked_reason
        == "tests/api/test_app.py::test_log_path: assert canonical_log.is_file()"
    )


def test_run_classification_and_status_mapping() -> None:
    assert classify_run(0, "") == "ok"
    assert classify_run(1, "Permission denied") == "permission"
    assert status_from_run(0, "ok", ResourceProfile.SMOKE) == ProjectStatus.SMOKE_TESTED
    assert (
        status_from_run(1, "credential", ResourceProfile.SMOKE)
        == ProjectStatus.BLOCKED_BY_CREDENTIAL
    )
