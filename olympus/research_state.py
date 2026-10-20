from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ResearchStateError(ValueError):
    """Raised when the canonical research state is internally inconsistent."""


@dataclass(frozen=True)
class StudyStatus:
    study_id: str
    phase: str
    primary_supported: bool | None
    routing_primary_supported: bool | None
    temporal_primary_supported: bool | None
    archive_retrieved: bool | None


@dataclass(frozen=True)
class EngineeringStudyStatus:
    study_id: str
    phase: str
    receipt: str
    source_verification: str = "current_files"


@dataclass(frozen=True)
class ResearchPortfolioStatus:
    project: str
    updated_utc: str
    software_version: str
    release_ready: bool
    production_ready: bool
    model_promoted: bool
    portfolio_scientific_disposition: str
    families_ready: int
    families_total: int
    all_families_ready: bool
    studies: tuple[StudyStatus, ...]
    engineering_studies: tuple[EngineeringStudyStatus, ...]
    pending_archive_studies: tuple[str, ...]
    blocking_gates: tuple[str, ...]
    claims_forbidden: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchStateError(f"cannot read valid JSON object from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ResearchStateError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _repository_file(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ResearchStateError(f"{label} must be a non-empty repository-relative path")
    candidate = root / relative
    if candidate.is_symlink():
        raise ResearchStateError(f"{label} must not be a symlink: {relative}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ResearchStateError(f"{label} escapes the repository: {relative}") from error
    if not resolved.is_file():
        raise ResearchStateError(f"{label} is not a regular file: {relative}")
    return resolved


def _optional_bool(value: object, *, label: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ResearchStateError(f"{label} must be a boolean when present")
    return value


def _child_primary_supported(child: dict[str, Any]) -> bool | None:
    if "primary_supported" in child:
        return _optional_bool(child["primary_supported"], label="child primary_supported")
    scientific_result = child.get("scientific_result")
    if isinstance(scientific_result, dict) and "primary_supported" in scientific_result:
        return _optional_bool(
            scientific_result["primary_supported"],
            label="child scientific_result.primary_supported",
        )
    if isinstance(scientific_result, str) and scientific_result.startswith(
        "PRIMARY_NOT_SUPPORTED"
    ):
        return False
    disposition = child.get("final_disposition", child.get("terminal_disposition"))
    if disposition == "NEGATIVE":
        return False
    routing_primary = child.get("routing_primary")
    if isinstance(routing_primary, dict) and routing_primary.get("supported") is False:
        return False
    return None


def _phase_matches(root_phase: str, child: dict[str, Any]) -> bool:
    child_phase = child.get("phase", child.get("current_phase"))
    if root_phase == child_phase:
        return True
    if root_phase == "COMPLETED_NEGATIVE":
        disposition = child.get("final_disposition", child.get("terminal_disposition"))
        return disposition == "NEGATIVE" and isinstance(child_phase, str) and (
            "TERMINAL_NEGATIVE" in child_phase
        )
    return False


def _require_matching_bool(
    study_id: str,
    root_study: dict[str, Any],
    key: str,
    child_value: bool | None,
) -> bool | None:
    root_value = _optional_bool(root_study.get(key), label=f"{study_id}.{key}")
    if root_value != child_value:
        raise ResearchStateError(
            f"{study_id} {key} mismatch: root={root_value!r}, child={child_value!r}"
        )
    return root_value


def _validate_ledger_correction(root: Path) -> None:
    ledger = _repository_file(root, "research/REALITY_LEDGER.md", label="reality ledger")
    text = ledger.read_text(encoding="utf-8")
    stale_v5 = "It has not run and has no result."
    stale_editclevr = "An executed EditCLEVR benchmark or JEPA efficacy result."
    correction = "## 2026-10-18 canonical completion correction"
    if (stale_v5 in text or stale_editclevr in text) and correction not in text:
        raise ResearchStateError(
            "REALITY_LEDGER.md retains launch-time v5/EditCLEVR statements without the "
            "required append-only canonical completion correction"
        )


def _validate_artifacts(root: Path, state: dict[str, Any]) -> None:
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, list) or not all(isinstance(item, str) for item in artifacts):
        raise ResearchStateError("artifacts must be a list of repository-relative paths")
    for artifact in artifacts:
        _repository_file(root, artifact, label="artifact")


def _validate_engineering_studies(
    root: Path, state: dict[str, Any]
) -> tuple[EngineeringStudyStatus, ...]:
    raw_studies = state.get("engineering_studies")
    if not isinstance(raw_studies, list) or not raw_studies:
        raise ResearchStateError("engineering_studies must be a non-empty list")
    seen_ids: set[str] = set()
    statuses: list[EngineeringStudyStatus] = []
    for raw_study in raw_studies:
        if not isinstance(raw_study, dict):
            raise ResearchStateError("each engineering study entry must be an object")
        study_id = raw_study.get("id")
        phase = raw_study.get("phase")
        if not isinstance(study_id, str) or not isinstance(phase, str):
            raise ResearchStateError("every engineering study requires string id and phase")
        if study_id in seen_ids:
            raise ResearchStateError(f"duplicate engineering study id: {study_id}")
        seen_ids.add(study_id)
        child = _read_object(
            _repository_file(root, raw_study.get("path"), label=f"{study_id} state")
        )
        if child.get("phase") != phase:
            raise ResearchStateError(f"{study_id} engineering phase mismatch")
        if child.get("promotion_authorized") is not False:
            raise ResearchStateError(f"{study_id} must not authorize promotion")
        for budget in (
            "scientific_training_epochs",
            "protected_evaluations",
            "external_downloads",
            "paid_jobs",
        ):
            if child.get(budget) != 0:
                raise ResearchStateError(f"{study_id} must retain a zero {budget} budget")
        receipt_relative = raw_study.get("receipt")
        receipt_path = _repository_file(root, receipt_relative, label=f"{study_id} receipt")
        receipt = _read_object(receipt_path)
        if receipt.get("study_id") != study_id or receipt.get("phase") != phase:
            raise ResearchStateError(f"{study_id} receipt identity or phase mismatch")
        source_hashes = receipt.get("source_sha256")
        if not isinstance(source_hashes, dict) or not source_hashes:
            raise ResearchStateError(f"{study_id} receipt must contain source_sha256")
        verification_scope = raw_study.get("source_verification", "current_files")
        if verification_scope not in {"current_files", "historical_receipt"}:
            raise ResearchStateError(f"{study_id} has an unknown source verification scope")
        if verification_scope == "historical_receipt":
            pinned = raw_study.get("receipt_sha256")
            if not isinstance(pinned, str) or _SHA256.fullmatch(pinned) is None:
                raise ResearchStateError(f"{study_id} historical receipt requires a SHA-256")
            if _sha256(receipt_path) != pinned:
                raise ResearchStateError(f"{study_id} historical receipt hash mismatch")
        for relative, expected in source_hashes.items():
            if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
                raise ResearchStateError(f"{study_id} receipt has invalid source SHA-256")
            if verification_scope == "current_files":
                source = _repository_file(root, relative, label=f"{study_id} receipt source")
                if _sha256(source) != expected:
                    raise ResearchStateError(f"{study_id} receipt source hash mismatch: {relative}")
        statuses.append(
            EngineeringStudyStatus(
                study_id=study_id,
                phase=phase,
                receipt=str(receipt_relative),
                source_verification=verification_scope,
            )
        )
    return tuple(statuses)


def audit_research_state(root: Path) -> ResearchPortfolioStatus:
    """Validate and summarize the repository's canonical research evidence state."""

    resolved_root = root.resolve()
    state = _read_object(_repository_file(resolved_root, "RESEARCH_STATE.json", label="state"))
    _validate_artifacts(resolved_root, state)
    software = state.get("software")
    if not isinstance(software, dict):
        raise ResearchStateError("software must be an object")
    for key in ("release_ready", "production_ready", "model_promoted"):
        if software.get(key) is not False:
            raise ResearchStateError(f"software.{key} must remain false until its gate passes")

    raw_studies = state.get("studies")
    if not isinstance(raw_studies, list) or not raw_studies:
        raise ResearchStateError("studies must be a non-empty list")
    seen_ids: set[str] = set()
    studies: list[StudyStatus] = []
    pending_archives: list[str] = []
    for raw_study in raw_studies:
        if not isinstance(raw_study, dict):
            raise ResearchStateError("each study entry must be an object")
        study_id = raw_study.get("id")
        phase = raw_study.get("phase")
        if not isinstance(study_id, str) or not isinstance(phase, str):
            raise ResearchStateError("every study requires string id and phase fields")
        if study_id in seen_ids:
            raise ResearchStateError(f"duplicate study id: {study_id}")
        seen_ids.add(study_id)
        child_path = _repository_file(
            resolved_root,
            raw_study.get("path"),
            label=f"{study_id} state",
        )
        child = _read_object(child_path)
        child_phase = child.get("phase", child.get("current_phase"))
        if not _phase_matches(phase, child):
            raise ResearchStateError(
                f"{study_id} phase mismatch: root={phase!r}, child={child_phase!r}"
            )
        if child.get("promotion_authorized") is True:
            raise ResearchStateError(f"{study_id} unexpectedly authorizes promotion")

        primary = _child_primary_supported(child)
        if "primary_supported" in raw_study:
            primary = _require_matching_bool(
                study_id,
                raw_study,
                "primary_supported",
                primary,
            )
        routing = None
        if "routing_primary_supported" in raw_study:
            routing_child = child.get("routing_primary")
            routing_value = (
                routing_child.get("supported") if isinstance(routing_child, dict) else None
            )
            routing = _require_matching_bool(
                study_id,
                raw_study,
                "routing_primary_supported",
                _optional_bool(routing_value, label=f"{study_id} child routing supported"),
            )
        temporal = None
        if "temporal_primary_supported" in raw_study:
            temporal_child = child.get("temporal_primary")
            temporal_value = (
                temporal_child.get("supported") if isinstance(temporal_child, dict) else None
            )
            temporal = _require_matching_bool(
                study_id,
                raw_study,
                "temporal_primary_supported",
                _optional_bool(temporal_value, label=f"{study_id} child temporal supported"),
            )

        archive_retrieved = None
        if phase.endswith("ARCHIVE_PENDING_RETRIEVAL"):
            remote_archive = child.get("remote_archive")
            verifier = child.get("verifier")
            if not isinstance(remote_archive, dict) or not isinstance(verifier, dict):
                raise ResearchStateError(
                    f"{study_id} pending archive requires archive and verifier"
                )
            archive_retrieved = _optional_bool(
                remote_archive.get("retrieved_to_repository"),
                label=f"{study_id} archive retrieval",
            )
            if archive_retrieved is not False:
                raise ResearchStateError(
                    f"{study_id} pending-retrieval phase requires retrieved_to_repository=false"
                )
            archive_sha256 = remote_archive.get("sha256")
            if not isinstance(archive_sha256, str) or _SHA256.fullmatch(archive_sha256) is None:
                raise ResearchStateError(f"{study_id} archive requires a lowercase SHA-256")
            if verifier.get("status") != "PASS":
                raise ResearchStateError(f"{study_id} pending archive requires verifier PASS")
            pending_archives.append(study_id)

        studies.append(
            StudyStatus(
                study_id=study_id,
                phase=phase,
                primary_supported=primary,
                routing_primary_supported=routing,
                temporal_primary_supported=temporal,
                archive_retrieved=archive_retrieved,
            )
        )

    readiness = state.get("large_scale_readiness_v1")
    if not isinstance(readiness, dict):
        raise ResearchStateError("large_scale_readiness_v1 must be an object")
    report = _read_object(
        _repository_file(
            resolved_root,
            readiness.get("current_report"),
            label="current readiness report",
        )
    )
    family_reports = report.get("family_reports")
    if not isinstance(family_reports, list) or not family_reports:
        raise ResearchStateError("readiness report must contain family reports")
    family_ids: set[str] = set()
    ready_count = 0
    for family in family_reports:
        if not isinstance(family, dict) or not isinstance(family.get("family_id"), str):
            raise ResearchStateError("every readiness family report requires a family_id")
        family_id = family["family_id"]
        if family_id in family_ids:
            raise ResearchStateError(f"duplicate readiness family id: {family_id}")
        family_ids.add(family_id)
        ready = family.get("ready_for_large_scale")
        if not isinstance(ready, bool):
            raise ResearchStateError(f"{family_id} readiness must be boolean")
        ready_count += int(ready)
    total_families = len(family_reports)
    all_ready = report.get("all_families_ready")
    if not isinstance(all_ready, bool) or all_ready != (ready_count == total_families):
        raise ResearchStateError("all_families_ready contradicts the family reports")
    expected_readiness = f"{ready_count}_OF_{total_families}_FAMILIES_READY"
    if readiness.get("status") != expected_readiness:
        raise ResearchStateError(
            f"root readiness status must be {expected_readiness}, got {readiness.get('status')!r}"
        )

    blocking_gates = state.get("blocking_gates")
    claims_forbidden = state.get("claims_forbidden")
    if not isinstance(blocking_gates, list) or not all(
        isinstance(item, str) for item in blocking_gates
    ):
        raise ResearchStateError("blocking_gates must be a list of strings")
    if not isinstance(claims_forbidden, list) or not all(
        isinstance(item, str) for item in claims_forbidden
    ):
        raise ResearchStateError("claims_forbidden must be a list of strings")
    if pending_archives and not any(
        "archive" in gate.lower() and "not yet retrieved" in gate.lower()
        for gate in blocking_gates
    ):
        raise ResearchStateError("pending remote archives are missing from the root blocking gates")

    _validate_ledger_correction(resolved_root)
    engineering_studies = _validate_engineering_studies(resolved_root, state)
    return ResearchPortfolioStatus(
        project=str(state.get("project")),
        updated_utc=str(state.get("updated_utc")),
        software_version=str(software.get("version")),
        release_ready=False,
        production_ready=False,
        model_promoted=False,
        portfolio_scientific_disposition=str(state.get("portfolio_scientific_disposition")),
        families_ready=ready_count,
        families_total=total_families,
        all_families_ready=bool(all_ready),
        studies=tuple(studies),
        engineering_studies=engineering_studies,
        pending_archive_studies=tuple(pending_archives),
        blocking_gates=tuple(blocking_gates),
        claims_forbidden=tuple(claims_forbidden),
    )
