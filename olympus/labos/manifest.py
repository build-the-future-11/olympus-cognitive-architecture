from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field

from olympus.core.schemas import StrictModel


class ManifestSource(StrEnum):
    DECLARED = "declared"
    INFERRED = "inferred"


class ProjectLanguage(StrEnum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ProjectStatus(StrEnum):
    COMPLETE = "complete"
    SCIENTIFICALLY_RUNNABLE = "scientifically_runnable"
    SMOKE_TESTED = "smoke_tested"
    FULL_BENCHMARK_PENDING_COMPUTE = "full_benchmark_pending_compute"
    BLOCKED_BY_EXTERNAL_DATASET = "blocked_by_external_dataset"
    BLOCKED_BY_CREDENTIAL = "blocked_by_credential"
    BLOCKED_BY_PERMISSION = "blocked_by_permission"
    DISCOVERED_ONLY = "discovered_only"


class ProjectClassification(StrEnum):
    FLAGSHIP_COMPANY = "flagship_company"
    HIGH_POTENTIAL_MVP = "high_potential_mvp"
    USEFUL_FEATURE = "useful_feature_for_another_company"
    RESEARCH_PROJECT = "research_project"
    REQUIRES_VALIDATION = "requires_validation"
    WEAK_DIFFERENTIATION = "weak_differentiation"
    MERGE_WITH_ANOTHER_IDEA = "merge_with_another_idea"
    ARCHIVE = "archive"


class ResourceProfile(StrEnum):
    SMOKE = "smoke"
    DEVELOPMENT = "development"
    BENCHMARK = "benchmark"
    FULL = "full"


class DatasetDependency(StrictModel):
    name: str
    required: bool = True
    notes: str = ""
    blocked_by_user_action: bool = False


class EntryPoint(StrictModel):
    name: str
    command: str
    profile: ResourceProfile
    expected_outputs: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = 120


class ResourceRequirements(StrictModel):
    profile: ResourceProfile = ResourceProfile.SMOKE
    cpu_only: bool = True
    gpu_optional: bool = False
    memory_gb: int = 2
    notes: str = ""


class ProjectManifest(StrictModel):
    project_id: str
    title: str
    description: str
    hypothesis: str
    root_path: str
    domain: str
    language: ProjectLanguage = ProjectLanguage.UNKNOWN
    manifest_source: ManifestSource = ManifestSource.INFERRED
    entry_points: list[EntryPoint] = Field(default_factory=list)
    datasets: list[DatasetDependency] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    validation_commands: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    classification: ProjectClassification = ProjectClassification.RESEARCH_PROJECT
    target_user: str = "Research engineer or technical founder evaluating the project."
    current_alternative: str = (
        "Manual notebooks, one-off scripts, or ungoverned prototype repositories."
    )
    proposed_solution: str = (
        "A reproducible local implementation with manifests, tests, reports, "
        "and explicit release gates."
    )
    differentiation: str = (
        "Evidence-first execution with machine-readable status and blocked-action tracking."
    )
    market: str = (
        "Specialized research, developer, and fintech/ML infrastructure markets "
        "depending on the project domain."
    )
    business_model: str = (
        "Requires project-specific validation before pricing; likely starts as "
        "consulting, SaaS, or infrastructure licensing."
    )
    distribution_strategy: str = (
        "Technical content, founder-led design partnerships, open demos, and targeted pilots."
    )
    technical_requirements: list[str] = Field(default_factory=list)
    data_requirements: list[str] = Field(default_factory=list)
    regulatory_considerations: list[str] = Field(default_factory=list)
    security_risks: list[str] = Field(default_factory=list)
    capital_intensity: str = "low-to-medium before full benchmark scale-up"
    time_to_mvp: str = "1-4 weeks for a narrow smoke-tested MVP when dependencies are local"
    evidence_available: list[str] = Field(default_factory=list)
    fatal_assumptions: list[str] = Field(default_factory=list)
    validation_experiments: list[str] = Field(default_factory=list)
    go_no_go_criteria: list[str] = Field(default_factory=list)
    benchmark_plan: list[str] = Field(default_factory=list)
    ablation_plan: list[str] = Field(default_factory=list)
    release_gates: list[str] = Field(default_factory=list)
    scientific_caveat: str = (
        "Smoke tests and synthetic experiments are engineering evidence, not proof of "
        "scientific superiority or regulated production readiness."
    )
    resource_requirements: ResourceRequirements = Field(default_factory=ResourceRequirements)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def path(self) -> Path:
        return Path(self.root_path)

    def smoke_entry_points(self) -> list[EntryPoint]:
        return [item for item in self.entry_points if item.profile == ResourceProfile.SMOKE]


class ValidationIssue(StrictModel):
    severity: str
    message: str


class ValidationResult(StrictModel):
    project_id: str
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class ProjectRecord(StrictModel):
    manifest: ProjectManifest
    validation: ValidationResult
    status: ProjectStatus = ProjectStatus.DISCOVERED_ONLY
    summary: str = ""
    blocked_reason: str = ""
