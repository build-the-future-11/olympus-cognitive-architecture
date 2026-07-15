from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from olympus.labos.manifest import ProjectRecord, ProjectStatus
from olympus.labos.runner import RunDatabase


@dataclass(slots=True)
class PortfolioSummary:
    total_projects: int
    by_status: dict[str, int]


class PortfolioReporter:
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root
        self.run_db = RunDatabase(artifact_root / "labos_runs.sqlite3")

    def summarize(self, records: list[ProjectRecord]) -> PortfolioSummary:
        counts: dict[str, int] = {}
        for record in records:
            counts[record.status.value] = counts.get(record.status.value, 0) + 1
        return PortfolioSummary(total_projects=len(records), by_status=counts)

    def write_status_json(self, records: list[ProjectRecord], output_path: Path) -> None:
        summary = self.summarize(records)
        exact_counts = _exact_status_counts(records)
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "generated_at": datetime.now(UTC).isoformat(),
                    "summary": {
                        "total_projects": summary.total_projects,
                        "by_status": summary.by_status,
                        "by_final_status": exact_counts,
                        "all_ready": _all_ready(records),
                    },
                    "projects": [
                        {
                            "project_id": record.manifest.project_id,
                            "title": record.manifest.title,
                            "description": record.manifest.description,
                            "hypothesis": record.manifest.hypothesis,
                            "domain": record.manifest.domain,
                            "classification": record.manifest.classification.value,
                            "root_path": record.manifest.root_path,
                            "manifest_source": record.manifest.manifest_source.value,
                            "status": record.status.value,
                            "final_status": _exact_status(record),
                            "summary": record.summary,
                            "blocked_reason": record.blocked_reason,
                            "target_user": record.manifest.target_user,
                            "current_alternative": record.manifest.current_alternative,
                            "proposed_solution": record.manifest.proposed_solution,
                            "differentiation": record.manifest.differentiation,
                            "market": record.manifest.market,
                            "business_model": record.manifest.business_model,
                            "distribution_strategy": record.manifest.distribution_strategy,
                            "technical_requirements": record.manifest.technical_requirements,
                            "data_requirements": record.manifest.data_requirements,
                            "regulatory_considerations": record.manifest.regulatory_considerations,
                            "security_risks": record.manifest.security_risks,
                            "capital_intensity": record.manifest.capital_intensity,
                            "time_to_mvp": record.manifest.time_to_mvp,
                            "evidence_available": record.manifest.evidence_available,
                            "fatal_assumptions": record.manifest.fatal_assumptions,
                            "validation_experiments": record.manifest.validation_experiments,
                            "go_no_go_criteria": record.manifest.go_no_go_criteria,
                            "benchmark_plan": record.manifest.benchmark_plan,
                            "ablation_plan": record.manifest.ablation_plan,
                            "release_gates": record.manifest.release_gates,
                            "scientific_caveat": record.manifest.scientific_caveat,
                            "entry_points": [
                                {
                                    "name": item.name,
                                    "command": item.command,
                                    "profile": item.profile.value,
                                    "expected_outputs": item.expected_outputs,
                                }
                                for item in record.manifest.entry_points
                            ],
                            "datasets": [
                                {
                                    "name": item.name,
                                    "required": item.required,
                                    "notes": item.notes,
                                    "blocked_by_user_action": item.blocked_by_user_action,
                                }
                                for item in record.manifest.datasets
                            ],
                            "validation": [
                                {
                                    "severity": issue.severity,
                                    "message": issue.message,
                                }
                                for issue in record.validation.issues
                            ],
                        }
                        for record in records
                    ],
                    "run_history": self.run_db.rows(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def write_completion_report(self, records: list[ProjectRecord], output_path: Path) -> None:
        summary = self.summarize(records)
        lines = [
            "# Portfolio Completion Report",
            "",
            f"Generated: {datetime.now(UTC).isoformat()}",
            "",
            f"Total discovered projects: {summary.total_projects}",
            f"All ready: {'yes' if _all_ready(records) else 'no'}",
            "",
            "## Status Summary",
            "",
        ]
        for status, count in sorted(summary.by_status.items()):
            lines.append(f"- `{status}`: {count}")
        lines.extend(["", "## Final Status Categories", ""])
        for status, count in sorted(_exact_status_counts(records).items()):
            lines.append(f"- {status}: {count}")
        lines.extend(["", "## Project Details", ""])
        for record in records:
            lines.extend(
                [
                    f"### {record.manifest.title}",
                    "",
                    f"- Project ID: `{record.manifest.project_id}`",
                    f"- Domain: `{record.manifest.domain}`",
                    f"- Root: `{record.manifest.root_path}`",
                    f"- Classification: `{record.manifest.classification.value}`",
                    f"- Manifest source: `{record.manifest.manifest_source.value}`",
                    f"- Status: `{record.status.value}`",
                    f"- Final status: {_exact_status(record)}",
                    f"- Summary: {record.summary or record.manifest.description}",
                    f"- Target user: {record.manifest.target_user}",
                    f"- Differentiation: {record.manifest.differentiation}",
                    f"- Business model: {record.manifest.business_model}",
                    f"- Data requirements: {_join_items(record.manifest.data_requirements)}",
                    f"- Security risks: {_join_items(record.manifest.security_risks)}",
                    (
                        "- Regulatory considerations: "
                        f"{_join_items(record.manifest.regulatory_considerations)}"
                    ),
                    f"- Benchmark plan: {_join_items(record.manifest.benchmark_plan)}",
                    f"- Ablation plan: {_join_items(record.manifest.ablation_plan)}",
                    f"- Release gates: {_join_items(record.manifest.release_gates)}",
                    f"- Caveat: {record.manifest.scientific_caveat}",
                ]
            )
            if record.validation.issues:
                lines.append("- Validation issues:")
                for issue in record.validation.issues:
                    lines.append(f"  - [{issue.severity}] {issue.message}")
            if record.blocked_reason:
                lines.append(f"- Blocked reason: {record.blocked_reason}")
            lines.append("")
        lines.extend(["## Commands Executed", ""])
        for row in self.run_db.rows():
            lines.append(
                f"- `{row['project_id']}`: `{row['command']}` -> "
                f"`{row['status']}` (rc={row['return_code']})"
            )
        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def write_runs_json(self, output_path: Path) -> None:
        rows = self.run_db.detailed_rows()
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "generated_at": datetime.now(UTC).isoformat(),
                    "total_runs": len(rows),
                    "runs": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def write_scientific_validation_report(
        self,
        records: list[ProjectRecord],
        output_path: Path,
    ) -> None:
        lines = [
            "# Scientific Validation Report",
            "",
            f"Generated: {datetime.now(UTC).isoformat()}",
            "",
            (
                "This report separates executable smoke evidence from scientific "
                "benchmark evidence. It does not fabricate benchmark results."
            ),
            "",
        ]
        for record in records:
            lines.extend(
                [
                    f"## {record.manifest.title}",
                    "",
                    f"- Final status: {_exact_status(record)}",
                    f"- Hypothesis: {record.manifest.hypothesis}",
                    f"- Evidence available: {_join_items(record.manifest.evidence_available)}",
                    f"- Benchmark status: {_benchmark_status(record)}",
                    f"- Benchmark plan: {_join_items(record.manifest.benchmark_plan)}",
                    f"- Ablation plan: {_join_items(record.manifest.ablation_plan)}",
                    f"- Fatal assumptions: {_join_items(record.manifest.fatal_assumptions)}",
                    f"- Scientific caveat: {record.manifest.scientific_caveat}",
                    "",
                ]
            )
            if record.validation.issues:
                lines.append("Validation notes:")
                for issue in record.validation.issues:
                    lines.append(f"- [{issue.severity}] {issue.message}")
                lines.append("")
        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def write_reproducibility_report(
        self,
        records: list[ProjectRecord],
        output_path: Path,
    ) -> None:
        lines = [
            "# Reproducibility",
            "",
            f"Generated: {datetime.now(UTC).isoformat()}",
            "",
            "## LabOS commands",
            "",
            (
                "- Discover: `.venv/bin/python -m olympus.cli labos discover "
                "--workspace /Users/ryan/Documents --artifacts artifacts`"
            ),
            (
                "- Validate: `.venv/bin/python -m olympus.cli labos validate "
                "--workspace /Users/ryan/Documents --artifacts artifacts`"
            ),
            (
                "- Run all smoke tests: `.venv/bin/python -m olympus.cli labos "
                "run-all --workspace /Users/ryan/Documents --artifacts artifacts "
                "--profile smoke`"
            ),
            (
                "- Generate reports: `.venv/bin/python -m olympus.cli labos "
                "generate-report --workspace /Users/ryan/Documents --artifacts "
                "artifacts --output-root /Users/ryan/Documents/Olympus`"
            ),
            "",
            "## Project commands",
            "",
        ]
        for record in records:
            lines.extend([f"### {record.manifest.project_id}", ""])
            if not record.manifest.entry_points:
                lines.append("- No runnable entry points are currently declared.")
            for entry in record.manifest.entry_points:
                lines.append(f"- `{entry.profile.value}` `{entry.name}`: `{entry.command}`")
            lines.extend(
                [
                    f"- Expected artifacts: {_join_items(record.manifest.expected_outputs)}",
                    f"- Compute: CPU-only={record.manifest.resource_requirements.cpu_only}; "
                    f"GPU optional={record.manifest.resource_requirements.gpu_optional}; "
                    f"memory={record.manifest.resource_requirements.memory_gb}GB",
                    "",
                ]
            )
        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def write_remaining_actions(
        self,
        records: list[ProjectRecord],
        output_path: Path,
    ) -> None:
        lines = [
            "# Remaining External Actions",
            "",
            (
                "These are actions LabOS cannot honestly complete without external data, "
                "credentials, legal/security review, or permission changes."
            ),
            "",
        ]
        actionable = False
        for record in records:
            status_blocked = record.status in {
                ProjectStatus.BLOCKED_BY_EXTERNAL_DATASET,
                ProjectStatus.BLOCKED_BY_CREDENTIAL,
                ProjectStatus.BLOCKED_BY_PERMISSION,
            }
            has_dataset_action = any(
                dataset.blocked_by_user_action for dataset in record.manifest.datasets
            )
            has_fintech_review = any(
                record.manifest.domain in {"economics-finance", "governed-fintech-systems"}
                and ("legal" in item.lower() or "licens" in item.lower())
                for item in record.manifest.regulatory_considerations
            )
            if status_blocked or has_dataset_action or has_fintech_review:
                actionable = True
                lines.extend(
                    [
                        f"## {record.manifest.title}",
                        "",
                        f"- Status: `{_exact_status(record)}`",
                        f"- Why: {record.blocked_reason or 'Further external input is required.'}",
                        f"- Path: `{record.manifest.root_path}`",
                        f"- Data actions: {_precise_data_action(record)}",
                        (
                            "- Review actions: "
                            f"{_join_items(record.manifest.regulatory_considerations)}"
                        ),
                        f"- Go/no-go criteria: {_join_items(record.manifest.go_no_go_criteria)}",
                        *_external_action_details(record),
                        "",
                    ]
                )
        if not actionable:
            lines.append(
                "No user-only external actions are currently required for the "
                "verified Olympus workflows."
            )
        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _join_items(items: list[str]) -> str:
    if not items:
        return "none recorded"
    return "; ".join(items)


def _all_ready(records: list[ProjectRecord]) -> bool:
    ready_statuses = {
        ProjectStatus.COMPLETE,
        ProjectStatus.SCIENTIFICALLY_RUNNABLE,
        ProjectStatus.SMOKE_TESTED,
    }
    return bool(records) and all(
        record.status in ready_statuses
        and not record.blocked_reason
        and not any(issue.severity == "error" for issue in record.validation.issues)
        for record in records
    )


def _exact_status(record: ProjectRecord) -> str:
    if any(dataset.blocked_by_user_action for dataset in record.manifest.datasets):
        return "Complete except external dataset access"
    if record.status in {ProjectStatus.COMPLETE, ProjectStatus.SMOKE_TESTED}:
        return "Complete and smoke-tested"
    if record.status == ProjectStatus.BLOCKED_BY_EXTERNAL_DATASET:
        return "Complete except external dataset access"
    if record.status == ProjectStatus.BLOCKED_BY_CREDENTIAL:
        return "Complete except external credential"
    if record.status in {
        ProjectStatus.SCIENTIFICALLY_RUNNABLE,
        ProjectStatus.FULL_BENCHMARK_PENDING_COMPUTE,
    }:
        return "Scientifically runnable; full benchmark pending compute"
    return "Blocked by a precisely documented technical reason"


def _exact_status_counts(records: list[ProjectRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        status = _exact_status(record)
        counts[status] = counts.get(status, 0) + 1
    return counts


def _benchmark_status(record: ProjectRecord) -> str:
    if record.status == ProjectStatus.SMOKE_TESTED:
        return "Smoke path executed; full scientific benchmark not claimed."
    if record.status == ProjectStatus.FULL_BENCHMARK_PENDING_COMPUTE:
        return (
            "Benchmark path exists or prior smoke passed, but full benchmark "
            "requires compute or data."
        )
    if record.status == ProjectStatus.BLOCKED_BY_EXTERNAL_DATASET:
        return "Benchmark blocked by external dataset access."
    if record.status == ProjectStatus.BLOCKED_BY_CREDENTIAL:
        return "Benchmark blocked by external credential."
    return "Not yet benchmark-runnable from current LabOS metadata."


def _precise_data_action(record: ProjectRecord) -> str:
    if record.manifest.project_id == "project-atlas-portfolio":
        return (
            "Place `train_FD00*.txt`, `test_FD00*.txt`, and `RUL_FD00*.txt` in "
            "`/Users/ryan/Documents/ATLAS/data/raw`, then run "
            "`atlas-preprocess --data-dir data/raw --output-dir data/processed` "
            "from `/Users/ryan/Documents/ATLAS`. Verify with "
            "`test -d /Users/ryan/Documents/ATLAS/data/processed`."
        )
    if record.manifest.project_id == "project-genesis":
        return (
            "From `/Users/ryan/Documents/Genesis`, run "
            "`python -m genesis.train --config configs/split_mnist.json --download-data` "
            "after approving the benchmark dataset download. Verify with "
            "`test -d /Users/ryan/Documents/Genesis/runs`."
        )
    return _join_items(record.manifest.data_requirements)


def _external_action_details(record: ProjectRecord) -> list[str]:
    root = record.manifest.root_path
    labos_prefix = (
        "cd /Users/ryan/Documents/Olympus && .venv/bin/python -m olympus.cli labos "
        "run-project"
    )
    if record.manifest.project_id == "project-atlas-portfolio":
        return [
            (
                "- Exact external action: download NASA C-MAPSS from the NASA "
                "Prognostics Data Repository after accepting the current terms."
            ),
            (
                "- Required files: `train_FD001.txt`..`train_FD004.txt`, "
                "`test_FD001.txt`..`test_FD004.txt`, and "
                "`RUL_FD001.txt`..`RUL_FD004.txt`."
            ),
            f"- Destination path: `{root}/data/raw/`.",
            "- Environment variables: none required for the local preprocessing command.",
            (
                f"- Verification command: `cd {root} && atlas-preprocess "
                "--data-dir data/raw --output-dir data/processed`."
            ),
            "- Expected output: processed C-MAPSS artifacts under `data/processed/`.",
            (
                f"- LabOS verification: `{labos_prefix} project-atlas-portfolio "
                "--workspace /Users/ryan/Documents --artifacts artifacts "
                "--profile benchmark`."
            ),
        ]
    if record.manifest.project_id == "project-genesis":
        return [
            (
                "- Exact external action: permit the first public benchmark dataset "
                "download for Split-MNIST/MNIST-family experiments."
            ),
            f"- Destination path: `{root}/data/`.",
            "- Environment variables: none required for the documented local command.",
            (
                f"- Download command: `cd {root} && python -m genesis.train "
                "--config configs/split_mnist.json --download-data`."
            ),
            (
                "- Expected output: MNIST-family files under the project data "
                "directory and a training run artifact under `runs/`."
            ),
            (
                f"- LabOS verification: `{labos_prefix} project-genesis "
                "--workspace /Users/ryan/Documents --artifacts artifacts "
                "--profile benchmark`."
            ),
        ]
    if record.manifest.domain in {"economics-finance", "governed-fintech-systems"}:
        return [
            (
                "- Exact external action: obtain qualified legal/security review "
                "before any regulated or real-customer financial use."
            ),
            (
                "- Environment variables: none; do not add credentials or real "
                "customer data until review is complete."
            ),
            "- SQL: none required by the current local smoke workflow.",
            (
                f"- Smoke verification: `{labos_prefix} {record.manifest.project_id} "
                "--workspace /Users/ryan/Documents --artifacts artifacts "
                "--profile smoke`."
            ),
            (
                "- Expected output: LabOS run status remains smoke-tested; "
                "production claims remain blocked until review is complete."
            ),
        ]
    return [
        "- Exact external action: see project-specific review notes above.",
        "- Environment variables: none recorded.",
        "- SQL: none recorded.",
    ]
