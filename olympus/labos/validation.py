from __future__ import annotations

from pathlib import Path

from olympus.labos.manifest import ProjectManifest, ValidationIssue, ValidationResult


def validate_project(manifest: ProjectManifest) -> ValidationResult:
    issues: list[ValidationIssue] = []
    project_root = Path(manifest.root_path)
    if not project_root.exists():
        issues.append(ValidationIssue(severity="error", message="Project root does not exist."))
    if not _has_readme(project_root):
        issues.append(ValidationIssue(severity="warning", message="README is missing."))
    if not manifest.entry_points:
        issues.append(
            ValidationIssue(
                severity="warning",
                message="No runnable entry points detected.",
            )
        )
    if not manifest.validation_commands:
        issues.append(
            ValidationIssue(
                severity="warning",
                message="No validation commands detected.",
            )
        )
    for dataset in manifest.datasets:
        if dataset.blocked_by_user_action:
            issues.append(
                ValidationIssue(
                    severity="info",
                    message=f"Dataset '{dataset.name}' may require manual download or acceptance.",
                )
            )
    return ValidationResult(
        project_id=manifest.project_id,
        valid=not any(issue.severity == "error" for issue in issues),
        issues=issues,
    )


def _has_readme(project_root: Path) -> bool:
    return any((project_root / name).exists() for name in ("README.md", "README.MD", "readme.md"))
