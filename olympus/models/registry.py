"""Truthful registry for the six proposed Olympus model families.

The registry describes executable components, not released models.  In
particular, ``EXPERIMENTAL_SMOKE`` records component-level execution only; it
does not imply that a family checkpoint passed a promotion evaluation.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from olympus.core.schemas import StrictModel


class ImplementationStatus(StrEnum):
    """Whether an executable family implementation exists in this package."""

    IMPLEMENTED = "IMPLEMENTED"


class ValidationStatus(StrEnum):
    """Strongest evidence currently attached to an implementation."""

    EXPERIMENTAL_SMOKE = "EXPERIMENTAL_SMOKE"


class PromotionStatus(StrEnum):
    """Release status under the family-specific promotion gates."""

    NOT_PROMOTED = "NOT_PROMOTED"


class FamilyDefinition(StrictModel):
    """Static, inspectable identity and claim boundary for one family."""

    family_id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    public_name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    module: str = Field(pattern=r"^olympus\.models\.[a-z_]+$")
    adapter_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    learned_classes: tuple[str, ...]
    deterministic_classes: tuple[str, ...]
    promotion_dependencies: tuple[str, ...] = ()
    implementation: ImplementationStatus = ImplementationStatus.IMPLEMENTED
    validation: ValidationStatus = ValidationStatus.EXPERIMENTAL_SMOKE
    promotion: PromotionStatus = PromotionStatus.NOT_PROMOTED
    qualifying_checkpoint: None = None
    claim_boundary: str = (
        "Executable reference components and synthetic/component tests only; "
        "no qualifying family checkpoint or capability claim."
    )

    @property
    def qualified_class_names(self) -> tuple[str, ...]:
        """Return importable class paths without importing heavyweight modules."""

        classes = self.learned_classes + self.deterministic_classes
        return tuple(f"{self.module}.{class_name}" for class_name in classes)


MODEL_FAMILIES: tuple[FamilyDefinition, ...] = (
    FamilyDefinition(
        family_id="hermes",
        public_name="Hermes",
        purpose="Grounded, cited, human-facing response with explicit abstention.",
        module="olympus.models.hermes",
        adapter_name="hermes",
        learned_classes=("HermesGroundingModule",),
        deterministic_classes=("HermesGroundedService", "HermesWorkspaceRuntime"),
    ),
    FamilyDefinition(
        family_id="prometheus",
        public_name="Prometheus",
        purpose="Bounded hypothesis proposal, process verification, and scientific synthesis.",
        module="olympus.models.prometheus",
        adapter_name="prometheus",
        learned_classes=("TrainableBranchProposer", "TrainableProcessVerifier"),
        deterministic_classes=("PrometheusSelector",),
        promotion_dependencies=("Hermes",),
    ),
    FamilyDefinition(
        family_id="perseus",
        public_name="Perseus",
        purpose="Typed tool selection, bounded execution, recovery, and transaction control.",
        module="olympus.models.perseus",
        adapter_name="perseus",
        learned_classes=("TrainableActionPolicy", "TrainableRecoveryPolicy"),
        deterministic_classes=(
            "DeterministicActionValidator",
            "CapabilityKernel",
            "TransactionManager",
        ),
        promotion_dependencies=("Hermes",),
    ),
    FamilyDefinition(
        family_id="olympus-atlas",
        public_name="Olympus-Atlas",
        purpose="Access-controlled hybrid retrieval with versioned provenance and attribution.",
        module="olympus.models.atlas",
        adapter_name="olympus_atlas",
        learned_classes=("OlympusAtlasRetriever",),
        deterministic_classes=("OlympusAtlasService",),
        promotion_dependencies=("Hermes", "Perseus"),
    ),
    FamilyDefinition(
        family_id="kronos",
        public_name="Kronos",
        purpose="Temporal forecasting, planning, and guarded offline adaptation.",
        module="olympus.models.kronos",
        adapter_name="kronos",
        learned_classes=("KronosTemporalPlanner", "SelectiveStateEncoder"),
        deterministic_classes=("TemporalAdapterManager",),
        promotion_dependencies=("Olympus-Atlas", "Perseus"),
    ),
    FamilyDefinition(
        family_id="aion",
        public_name="Aion",
        purpose="Governed research protocol orchestration with an optional learned router.",
        module="olympus.models.aion",
        adapter_name="aion",
        learned_classes=("AionRouter",),
        deterministic_classes=("AionController",),
        promotion_dependencies=(
            "Hermes",
            "Prometheus",
            "Perseus",
            "Olympus-Atlas",
            "Kronos",
        ),
    ),
)


def get_family(family_id: str) -> FamilyDefinition:
    """Resolve a family by stable id or public name, case-insensitively."""

    normalized = family_id.casefold()
    for family in MODEL_FAMILIES:
        if normalized in {family.family_id.casefold(), family.public_name.casefold()}:
            return family
    raise KeyError(f"unknown Olympus model family: {family_id}")


def model_family_status() -> dict[str, object]:
    """Return a JSON-compatible snapshot with an explicit scientific boundary."""

    return {
        "schema_version": "1.0",
        "summary": {
            "implementations": "IMPLEMENTED",
            "strongest_validation": "EXPERIMENTAL_SMOKE",
            "promotion": "NOT_PROMOTED",
            "qualifying_checkpoints": 0,
        },
        "composition": {
            "status": "TWO_PHASE_REFERENCE_REPLAY",
            "module": "olympus.models.composition",
            "scientific_claim": "contract_composition_only",
            "state_scope": "PROCESS_LOCAL",
            "action_bound_approval": True,
            "executor_profile": "HOST_ASSERTED",
            "material_tools_allowed": False,
            "sandbox_enforced": False,
            "promotion_authorized": False,
            "qualifying_result": False,
        },
        "claim_boundary": (
            "All six families have executable reference components. Synthetic training and "
            "component smokes are not qualifying evaluations, promoted checkpoints, or evidence "
            "of family-level capability."
        ),
        "families": [family.model_dump(mode="json") for family in MODEL_FAMILIES],
    }
