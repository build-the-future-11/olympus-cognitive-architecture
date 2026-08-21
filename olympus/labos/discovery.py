from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path
from typing import cast

import yaml

from olympus.labos.manifest import (
    DatasetDependency,
    EntryPoint,
    ManifestSource,
    ProjectClassification,
    ProjectLanguage,
    ProjectManifest,
    ResourceProfile,
    ResourceRequirements,
)

README_NAMES = ("README.md", "README.MD", "readme.md")


def discover_projects(workspace_root: Path) -> list[ProjectManifest]:
    projects: list[ProjectManifest] = []
    for candidate in sorted(path for path in workspace_root.iterdir() if path.is_dir()):
        if candidate.name.startswith("."):
            continue
        if not (candidate / ".git").exists():
            continue
        projects.append(load_or_infer_manifest(candidate))
    if (workspace_root / ".git").exists():
        root_manifest = load_or_infer_manifest(workspace_root)
        if root_manifest.project_id not in {item.project_id for item in projects}:
            projects.append(root_manifest)
    return sorted(projects, key=lambda item: item.project_id)


def load_or_infer_manifest(project_root: Path) -> ProjectManifest:
    for filename in ("labos.project.yaml", "labos.project.yml", "labos.project.json"):
        manifest_path = project_root / filename
        if manifest_path.exists():
            payload = _read_manifest(manifest_path)
            payload["root_path"] = str(project_root.resolve())
            payload["manifest_source"] = ManifestSource.DECLARED
            return ProjectManifest.model_validate(payload)
    return infer_manifest(project_root)


def infer_manifest(project_root: Path) -> ProjectManifest:
    readme = _read_readme(project_root)
    title = _extract_title(project_root.name, readme)
    description, hypothesis = _extract_summary(readme, project_root.name)
    language = _infer_language(project_root)
    inferred_name = infer_python_project_name(project_root)
    entry_points = _infer_entry_points(project_root, language)
    validation_commands = _infer_validation_commands(project_root, language)
    datasets = _infer_datasets(readme)
    domain = _infer_domain(readme, title)
    return ProjectManifest(
        project_id=_slugify(inferred_name or project_root.name),
        title=title,
        description=description,
        hypothesis=hypothesis,
        root_path=str(project_root.resolve()),
        domain=domain,
        language=language,
        manifest_source=ManifestSource.INFERRED,
        entry_points=entry_points,
        datasets=datasets,
        expected_outputs=["results", "artifacts", "reports"],
        validation_commands=validation_commands,
        tags=_infer_tags(readme, project_root),
        classification=_infer_classification(readme, title, domain),
        target_user=_infer_target_user(domain),
        current_alternative=_infer_current_alternative(domain),
        proposed_solution=description,
        differentiation=_infer_differentiation(readme, domain),
        market=_infer_market(domain),
        business_model=_infer_business_model(domain),
        distribution_strategy=_infer_distribution_strategy(domain),
        technical_requirements=_infer_technical_requirements(project_root, language),
        data_requirements=[item.name for item in datasets]
        or ["local synthetic or repository-bundled data"],
        regulatory_considerations=_infer_regulatory_considerations(domain),
        security_risks=_infer_security_risks(domain),
        capital_intensity=_infer_capital_intensity(readme, domain),
        time_to_mvp=_infer_time_to_mvp(entry_points, datasets),
        evidence_available=_infer_evidence(project_root, readme),
        fatal_assumptions=_infer_fatal_assumptions(domain, datasets),
        validation_experiments=_infer_validation_experiments(domain),
        go_no_go_criteria=_infer_go_no_go_criteria(domain),
        benchmark_plan=_infer_benchmark_plan(domain),
        ablation_plan=_infer_ablation_plan(domain),
        release_gates=_infer_release_gates(domain),
        scientific_caveat=_infer_scientific_caveat(domain),
        resource_requirements=ResourceRequirements(
            profile=ResourceProfile.SMOKE,
            cpu_only=True,
            gpu_optional="cuda" in readme.lower() or "gpu" in readme.lower(),
            memory_gb=4,
            notes="Automatically inferred from repository metadata.",
        ),
        metadata={"readme_present": bool(readme), "inferred": True},
    )


def _read_manifest(path: Path) -> dict[str, object]:
    if path.suffix == ".json":
        return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    return cast(
        dict[str, object],
        yaml.safe_load(path.read_text(encoding="utf-8")),
    )


def _read_readme(project_root: Path) -> str:
    for name in README_NAMES:
        path = project_root / name
        if path.exists():
            return path.read_text(encoding="utf-8")
    return ""


def _extract_title(default_name: str, readme: str) -> str:
    for line in readme.splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return default_name


def _extract_summary(readme: str, default_name: str) -> tuple[str, str]:
    paragraphs = [
        line.strip()
        for line in readme.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and not line.startswith("```")
    ]
    description = paragraphs[0] if paragraphs else f"{default_name} research project."
    hypothesis = paragraphs[1] if len(paragraphs) > 1 else description
    return description, hypothesis


def _infer_language(project_root: Path) -> ProjectLanguage:
    has_pyproject = (project_root / "pyproject.toml").exists()
    has_package = (project_root / "package.json").exists()
    if has_pyproject and has_package:
        return ProjectLanguage.MIXED
    if has_pyproject:
        return ProjectLanguage.PYTHON
    if has_package:
        return ProjectLanguage.TYPESCRIPT
    return ProjectLanguage.UNKNOWN


def _infer_entry_points(
    project_root: Path,
    language: ProjectLanguage,
) -> list[EntryPoint]:
    entry_points: list[EntryPoint] = []
    olympus_python = sys.executable
    if (
        language in {ProjectLanguage.PYTHON, ProjectLanguage.MIXED}
        and (project_root / ".venv").exists()
    ):
        command = ".venv/bin/python -m pytest -q -p no:cacheprovider"
        env: dict[str, str] = {}
        timeout_seconds = 120
        entry_points.append(
            EntryPoint(
                name="smoke-test",
                command=command,
                profile=ResourceProfile.SMOKE,
                expected_outputs=["test-report"],
                env=env,
                timeout_seconds=timeout_seconds,
            )
        )
    elif language == ProjectLanguage.PYTHON and (project_root / "src").exists():
        env = {"PYTHONPATH": "src"}
        if (project_root / "tests").exists():
            entry_points.append(
                EntryPoint(
                    name="smoke-test",
                    command=f"{olympus_python} -m pytest tests -q -p no:cacheprovider",
                    profile=ResourceProfile.SMOKE,
                    expected_outputs=["test-report"],
                    env=env,
                    timeout_seconds=180,
                )
            )
    if (
        language in {ProjectLanguage.TYPESCRIPT, ProjectLanguage.MIXED}
        and (project_root / "package.json").exists()
    ):
        entry_points.append(
            EntryPoint(
                name="node-test",
                command="npm test",
                profile=ResourceProfile.SMOKE,
                expected_outputs=["test-report"],
                timeout_seconds=180,
            )
        )
    return entry_points


def _infer_validation_commands(
    project_root: Path,
    language: ProjectLanguage,
) -> list[str]:
    commands: list[str] = []
    if language in {ProjectLanguage.PYTHON, ProjectLanguage.MIXED}:
        commands.extend(
            [
                f"test -f {project_root / 'pyproject.toml'}",
                f"test -d {project_root / 'tests'} || test -d {project_root / 'test'}",
            ]
        )
    if language in {ProjectLanguage.TYPESCRIPT, ProjectLanguage.MIXED}:
        commands.append(f"test -f {project_root / 'package.json'}")
    commands.append(f"test -f {project_root / 'README.md'}")
    return commands


def _infer_datasets(readme: str) -> list[DatasetDependency]:
    lowered = readme.lower()
    dataset_names: list[DatasetDependency] = []
    if "nasa" in readme.lower():
        dataset_names.append(
            DatasetDependency(
                name="NASA data",
                notes="Mentioned in README; may require manual download.",
                blocked_by_user_action=True,
            )
        )
    if "split-mnist" in lowered or "mnist" in lowered:
        dataset_names.append(
            DatasetDependency(
                name="MNIST-family dataset",
                notes="README references optional benchmark download.",
                blocked_by_user_action=True,
            )
        )
    if "manual download" in lowered and not dataset_names:
        dataset_names.append(
            DatasetDependency(
                name="external dataset",
                notes="Repository README references manual dataset download.",
                blocked_by_user_action=True,
            )
        )
    return dataset_names


def _infer_domain(readme: str, title: str) -> str:
    lowered = f"{title}\n{readme}".lower()
    if "proxy" in lowered or "codex" in lowered or "claude" in lowered:
        return "developer-tooling"
    if "ledger foundry" in lowered or ("ledger" in lowered and "audit" in lowered):
        return "governed-fintech-systems"
    if "economic" in lowered or "econom" in lowered or "market" in lowered or "finance" in lowered:
        return "economics-finance"
    if (
        "developmental" in lowered
        or "continual learning" in lowered
        or "modular learning" in lowered
    ):
        return "developmental-learning"
    if (
        "mathematical" in lowered
        or "scientific-computing" in lowered
        or "topology" in lowered
        or "quantum" in lowered
    ):
        return "mathematical-scientific-research"
    if "weather" in lowered or "climate" in lowered or "wildfire" in lowered:
        return "climate-forecasting"
    if "compression" in lowered:
        return "representation-learning"
    if "cognitive" in lowered or "agent" in lowered:
        return "agentic-ml-systems"
    return "general-ml-research"
def _infer_classification(
    readme: str,
    title: str,
    domain: str,
) -> ProjectClassification:
    lowered = f"{title}\n{readme}".lower()
    if "portfolio" in lowered or "operating" in lowered or "platform" in lowered:
        return ProjectClassification.FLAGSHIP_COMPANY
    if domain in {"economics-finance", "governed-fintech-systems", "developer-tooling"}:
        return ProjectClassification.HIGH_POTENTIAL_MVP
    if "experiment" in lowered or "paper" in lowered or "research" in lowered:
        return ProjectClassification.RESEARCH_PROJECT
    return ProjectClassification.REQUIRES_VALIDATION


def _infer_target_user(domain: str) -> str:
    if domain == "economics-finance":
        return "Quant researcher, fintech founder, or policy/economic intelligence analyst."
    if domain == "governed-fintech-systems":
        return "Fintech operator needing auditable human-approved decision support."
    if domain == "developer-tooling":
        return "AI developer who needs local provider routing and workflow continuity."
    if domain == "agentic-ml-systems":
        return "Research engineer building reproducible agent and cognitive-runtime experiments."
    if domain == "climate-forecasting":
        return "Applied ML researcher or climate-risk operator validating forecasting models."
    if domain == "developmental-learning":
        return "ML researcher testing continual, modular, or developmental-learning methods."
    if domain == "mathematical-scientific-research":
        return (
            "Scientific-computing researcher who needs typed executable kernels "
            "and validation artifacts."
        )
    return "Research engineer validating a narrow technical thesis."


def _infer_current_alternative(domain: str) -> str:
    if domain in {"economics-finance", "governed-fintech-systems"}:
        return "Spreadsheets, ad hoc notebooks, vendor dashboards, or un-audited decision scripts."
    if domain == "developer-tooling":
        return (
            "Manual provider switching, brittle shell scripts, or direct single-provider clients."
        )
    return "One-off notebooks, isolated benchmark scripts, and manually curated experiment notes."


def _infer_differentiation(readme: str, domain: str) -> str:
    if "audit" in readme.lower() or domain == "governed-fintech-systems":
        return "Auditability, governance boundaries, and explicit release/compliance gates."
    if "benchmark" in readme.lower():
        return "Benchmark-first implementation with reproducible experiment outputs."
    if domain == "agentic-ml-systems":
        return (
            "Behavior-graph execution and typed latent-workspace runtime rather "
            "than prompt-only orchestration."
        )
    return "Local-first reproducibility, manifest-driven execution, and evidence-linked reporting."


def _infer_market(domain: str) -> str:
    return {
        "economics-finance": (
            "Fintech analytics, quant tooling, financial research infrastructure, "
            "and economic intelligence."
        ),
        "governed-fintech-systems": (
            "Regulated fintech operations, audit tooling, and decision-support "
            "infrastructure."
        ),
        "developer-tooling": "AI developer tools and local model-provider orchestration.",
        "agentic-ml-systems": (
            "AI research tooling, agent evaluation infrastructure, and internal "
            "R&D platforms."
        ),
        "climate-forecasting": "Climate risk, industrial forecasting, and applied scientific ML.",
        "developmental-learning": (
            "Continual-learning research infrastructure and evaluation tooling."
        ),
        "mathematical-scientific-research": (
            "Scientific-computing tools, research infrastructure, and technical "
            "education."
        ),
    }.get(domain, "Applied ML research tooling and technical validation services.")


def _infer_business_model(domain: str) -> str:
    if domain in {"economics-finance", "governed-fintech-systems"}:
        return (
            "Design-partner pilots, enterprise SaaS, usage-based analytics, or "
            "licensed internal tooling after legal review."
        )
    if domain == "developer-tooling":
        return "Open-core developer tool with paid team governance, hosted relay, or support tiers."
    return "Research platform licensing, consulting-backed pilots, or internal-product incubation."


def _infer_distribution_strategy(domain: str) -> str:
    if domain == "developer-tooling":
        return "GitHub demos, developer tutorials, integration guides, and community-led adoption."
    if domain in {"economics-finance", "governed-fintech-systems"}:
        return (
            "Founder-led outreach to fintech teams, compliance-friendly pilots, "
            "and benchmark case studies."
        )
    return (
        "Research writeups, reproducible demos, conference-style artifacts, and "
        "targeted technical pilots."
    )


def _infer_technical_requirements(
    project_root: Path,
    language: ProjectLanguage,
) -> list[str]:
    requirements = ["README and machine-readable project metadata"]
    if language in {ProjectLanguage.PYTHON, ProjectLanguage.MIXED}:
        requirements.append("Python runtime with project dependencies installed")
    if language in {ProjectLanguage.TYPESCRIPT, ProjectLanguage.MIXED}:
        requirements.append("Node.js package manager and test script")
    if (project_root / "tests").exists() or (project_root / "test").exists():
        requirements.append("Automated test suite")
    if (project_root / "docs").exists():
        requirements.append("Documentation or release notes")
    return requirements


def _infer_regulatory_considerations(domain: str) -> list[str]:
    if domain in {"economics-finance", "governed-fintech-systems"}:
        return [
            "Financial-data licensing and permitted use must be reviewed.",
            (
                "Suitability, disclosures, payment/custody boundaries, and "
                "jurisdiction limits require qualified legal review."
            ),
            "Do not claim legal compliance without counsel-approved production controls.",
        ]
    return [
        (
            "No regulated-production claim is made; project-specific legal "
            "review is required before deployment."
        )
    ]


def _infer_security_risks(domain: str) -> list[str]:
    risks = [
        "Secrets must not be committed or logged.",
        "Experiment outputs need provenance and tamper-evident run records.",
    ]
    if domain in {"economics-finance", "governed-fintech-systems"}:
        risks.extend(
            [
                (
                    "Financial or identity data requires least-privilege access "
                    "control and audit trails."
                ),
                "Fraud, abuse, and model-output misuse need explicit human review boundaries.",
            ]
        )
    if domain == "developer-tooling":
        risks.append(
            
                "Provider credentials and local proxy traffic require careful "
                "redaction and permission boundaries."
            
        )
    return risks


def _infer_capital_intensity(readme: str, domain: str) -> str:
    lowered = readme.lower()
    if "gpu" in lowered or "cuda" in lowered or domain == "climate-forecasting":
        return "medium before full benchmark campaigns; GPU or larger datasets may increase cost"
    if domain in {"economics-finance", "governed-fintech-systems"}:
        return "medium once licensed data, compliance review, and production security are included"
    return "low for smoke validation; medium for full multi-seed benchmark campaigns"


def _infer_time_to_mvp(
    entry_points: list[EntryPoint],
    datasets: list[DatasetDependency],
) -> str:
    if any(dataset.blocked_by_user_action for dataset in datasets):
        return "2-6 weeks after external data access is approved"
    if entry_points:
        return "1-2 weeks to package a narrow validated MVP"
    return "2-4 weeks to add runnable entry points and MVP surface"


def _infer_evidence(project_root: Path, readme: str) -> list[str]:
    evidence = []
    if readme:
        evidence.append("README/product thesis is present")
    if (project_root / "tests").exists() or (project_root / "test").exists():
        evidence.append("test directory exists")
    if (project_root / "docs").exists():
        evidence.append("documentation directory exists")
    if "benchmark" in readme.lower():
        evidence.append("README references benchmarks")
    return evidence or ["repository discovered; deeper evidence still required"]


def _infer_fatal_assumptions(
    domain: str,
    datasets: list[DatasetDependency],
) -> list[str]:
    assumptions = [
        "Target users experience the stated problem often enough to adopt a new workflow."
    ]
    if datasets:
        assumptions.append("Required datasets can be licensed, downloaded, and reproduced.")
    if domain in {"economics-finance", "governed-fintech-systems"}:
        assumptions.append(
            
                "Regulated use can be bounded to decision support with "
                "acceptable legal and compliance controls."
            
        )
    return assumptions


def _infer_validation_experiments(domain: str) -> list[str]:
    common = [
        "Run smoke tests and record artifacts through LabOS.",
        "Run at least three seeds or scenario variants for benchmark claims.",
    ]
    if domain in {"economics-finance", "governed-fintech-systems"}:
        common.append(
            "Interview five target users and validate willingness to run a governed pilot."
        )
    elif domain == "developer-tooling":
        common.append(
            "Run a real provider-compatibility matrix with credentials supplied by the user."
        )
    else:
        common.append("Compare against a simple baseline and document failure cases.")
    return common


def _infer_go_no_go_criteria(domain: str) -> list[str]:
    criteria = [
        "Go: deterministic smoke workflow passes locally.",
        "Go: benchmark output is reproducible and stored with provenance.",
        "No-go: core claim depends on unavailable private data or unstated manual steps.",
    ]
    if domain in {"economics-finance", "governed-fintech-systems"}:
        criteria.append("No-go: legal/compliance review rejects the proposed operating boundary.")
    return criteria


def _infer_benchmark_plan(domain: str) -> list[str]:
    if domain in {"economics-finance", "governed-fintech-systems"}:
        return [
            "Synthetic smoke benchmark for correctness.",
            "Licensed/consented historical-data replay before any real-world claim.",
            "Stress tests for fraud, missing data, outliers, and audit-log integrity.",
        ]
    return [
        "Smoke benchmark on bundled or synthetic fixtures.",
        "Baseline comparison with simple heuristic/model.",
        "Multi-seed run with stored metrics before claiming scientific improvement.",
    ]


def _infer_ablation_plan(domain: str) -> list[str]:
    if domain == "agentic-ml-systems":
        return [
            "Disable verification passes.",
            "Disable memory/retrieval.",
            "Compare behavior-graph execution against direct single-step execution.",
        ]
    return [
        "Remove the main proposed mechanism.",
        "Compare against a simple baseline.",
        "Vary data scale, random seed, and failure/noise conditions.",
    ]


def _infer_release_gates(domain: str) -> list[str]:
    gates = [
        "All tests pass in a clean local run.",
        "LabOS portfolio status and completion report are regenerated.",
        "No placeholders, pseudocode, or fabricated metrics are presented as results.",
    ]
    if domain in {"economics-finance", "governed-fintech-systems"}:
        gates.append("Qualified legal/security review completed before regulated production use.")
    return gates


def _infer_scientific_caveat(domain: str) -> str:
    if domain in {"economics-finance", "governed-fintech-systems"}:
        return (
            "Synthetic or smoke-tested fintech outputs are not investment, credit, legal, "
            "tax, custody, payment, or suitability advice."
        )
    return (
        "Smoke tests establish executable engineering behavior only; full scientific claims "
        "require documented baselines, ablations, external data rights, and multi-seed benchmarks."
    )


def _infer_tags(readme: str, project_root: Path) -> list[str]:
    tags = {project_root.name.lower()}
    lowered = readme.lower()
    for token in (
        "benchmark",
        "paper",
        "portfolio",
        "world model",
        "transformer",
        "economics",
        "research",
        "cli",
        "api",
    ):
        if token in lowered:
            tags.add(token.replace(" ", "-"))
    return sorted(tags)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def infer_python_project_name(project_root: Path) -> str | None:
    pyproject = project_root / "pyproject.toml"
    if not pyproject.exists():
        return None
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    name = project.get("name")
    if isinstance(name, str):
        return name
    return None
