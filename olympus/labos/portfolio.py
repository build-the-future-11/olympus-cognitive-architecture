from __future__ import annotations

from pathlib import Path

from olympus.labos.discovery import discover_projects
from olympus.labos.manifest import (
    ProjectManifest,
    ProjectRecord,
    ProjectStatus,
    ResourceProfile,
)
from olympus.labos.reports import PortfolioReporter
from olympus.labos.runner import PortfolioRunner
from olympus.labos.validation import validate_project


class PortfolioService:
    def __init__(self, workspace_root: Path, artifact_root: Path) -> None:
        self.workspace_root = workspace_root
        self.artifact_root = artifact_root
        self.runner = PortfolioRunner(artifact_root)
        self.reporter = PortfolioReporter(artifact_root)

    def discover(self) -> list[ProjectRecord]:
        manifests = discover_projects(self.workspace_root)
        records = [self._record_from_manifest(manifest) for manifest in manifests]
        return self._attach_run_history(records)

    def validate(self) -> list[ProjectRecord]:
        return self.discover()

    def run_smoke(
        self,
        project_ids: set[str] | None = None,
        profile: ResourceProfile = ResourceProfile.SMOKE,
    ) -> list[ProjectRecord]:
        records = self.discover()
        for record in records:
            if project_ids is not None and record.manifest.project_id not in project_ids:
                continue
            results = self.runner.run_manifest(record.manifest, profile=profile)
            if results:
                record.status = results[-1].status
                if (
                    record.status == ProjectStatus.SMOKE_TESTED
                    and any(dataset.blocked_by_user_action for dataset in record.manifest.datasets)
                ):
                    record.status = ProjectStatus.FULL_BENCHMARK_PENDING_COMPUTE
                record.summary = (
                    f"Executed {len(results)} {profile.value} command(s); "
                    f"last classification={results[-1].classification}."
                )
                if results[-1].classification != "ok":
                    record.blocked_reason = _describe_run_failure(
                        results[-1].classification,
                        results[-1].stdout,
                        results[-1].stderr,
                    )
                elif any(
                    dataset.blocked_by_user_action for dataset in record.manifest.datasets
                ):
                    record.blocked_reason = (
                        "One or more datasets require manual download or acceptance."
                    )
                else:
                    record.blocked_reason = ""
            else:
                record.summary = "No matching entry points for the requested profile."
        return records

    def write_reports(
        self,
        records: list[ProjectRecord],
        output_root: Path,
    ) -> None:
        self.reporter.write_status_json(records, output_root / "portfolio_status.json")
        self.reporter.write_completion_report(
            records,
            output_root / "PORTFOLIO_COMPLETION_REPORT.md",
        )
        self.reporter.write_runs_json(output_root / "portfolio_runs.json")
        self.reporter.write_scientific_validation_report(
            records,
            output_root / "SCIENTIFIC_VALIDATION_REPORT.md",
        )
        self.reporter.write_reproducibility_report(
            records,
            output_root / "REPRODUCIBILITY.md",
        )
        self.reporter.write_remaining_actions(
            records,
            output_root / "REMAINING_EXTERNAL_ACTIONS.md",
        )

    def _record_from_manifest(self, manifest: ProjectManifest) -> ProjectRecord:
        validation = validate_project(manifest)
        status = ProjectStatus.DISCOVERED_ONLY
        blocked_reason = ""
        if manifest.manifest_source.value == "declared" and validation.valid:
            status = ProjectStatus.SCIENTIFICALLY_RUNNABLE
        if any(dataset.blocked_by_user_action for dataset in manifest.datasets):
            blocked_reason = "One or more datasets require manual download or acceptance."
            if status == ProjectStatus.DISCOVERED_ONLY:
                status = ProjectStatus.BLOCKED_BY_EXTERNAL_DATASET
        if (
            any(issue.severity == "warning" for issue in validation.issues)
            and status == ProjectStatus.COMPLETE
        ):
            status = ProjectStatus.SCIENTIFICALLY_RUNNABLE
        summary = manifest.description
        return ProjectRecord(
            manifest=manifest,
            validation=validation,
            status=status,
            summary=summary,
            blocked_reason=blocked_reason,
        )

    def _attach_run_history(self, records: list[ProjectRecord]) -> list[ProjectRecord]:
        latest_by_project: dict[str, dict[str, object]] = {}
        for history_row in self.runner.run_db.detailed_rows():
            latest_by_project[str(history_row["project_id"])] = history_row
        for record in records:
            latest_row = latest_by_project.get(record.manifest.project_id)
            if latest_row is None:
                continue
            status_name = str(latest_row["status"])
            record.status = ProjectStatus(status_name)
            if (
                record.status == ProjectStatus.SMOKE_TESTED
                and any(dataset.blocked_by_user_action for dataset in record.manifest.datasets)
            ):
                record.status = ProjectStatus.FULL_BENCHMARK_PENDING_COMPUTE
            record.summary = (
                f"Last run `{latest_row['command']}` completed with status "
                f"`{record.status.value}`."
            )
            classification = str(latest_row["classification"])
            if classification != "ok":
                record.blocked_reason = _describe_run_failure(
                    classification,
                    str(latest_row.get("stdout", "")),
                    str(latest_row.get("stderr", "")),
                )
        return records


def _describe_run_failure(classification: str, stdout: str, stderr: str) -> str:
    if classification == "ok":
        return ""

    text = "\n".join(part for part in (stdout, stderr) if part)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if classification == "external_dataset":
        return "One or more required datasets are missing or require manual access."
    if classification == "credential":
        return "A required credential or API key is missing."
    if classification == "dependency":
        return "A required module or command is missing from the runtime environment."

    permission_line = next(
        (
            line
            for line in lines
            if "PermissionError:" in line
            or "Operation not permitted" in line
            or "Permission denied" in line
        ),
        "",
    )
    if classification == "permission" and permission_line:
        return permission_line

    failed_line = next(
        (
            line.removeprefix("FAILED ").strip()
            for line in reversed(lines)
            if line.startswith("FAILED ")
        ),
        "",
    )
    error_line = next(
        (
            line.removeprefix("E").strip()
            for line in reversed(lines)
            if "AssertionError:" in line
            or "PermissionError:" in line
            or "ModuleNotFoundError:" in line
            or "ImportError:" in line
            or "FileNotFoundError:" in line
        ),
        "",
    )
    assert_line = next(
        (
            line.lstrip("> ").strip()
            for line in reversed(lines)
            if line.lstrip("> ").strip().startswith("assert ")
        ),
        "",
    )

    if failed_line and assert_line:
        return f"{failed_line}: {assert_line}"
    if failed_line and error_line:
        return f"{failed_line}: {error_line}"
    if failed_line:
        return failed_line
    if permission_line:
        return permission_line
    if error_line:
        return error_line
    return classification
