"""Canonical, evidence-bound specifications for the BU1LD model programme."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from olympus.core.schemas import StrictModel

ModelFamilyId = Literal["hermes", "prometheus", "perseus", "atlas", "kronos", "aion"]


class ModelTargetSpecification(StrictModel):
    """An intended model target, never evidence that the target already exists."""

    family_id: ModelFamilyId
    display_name: str
    target_parameters: int | None = Field(default=None, gt=0)
    target_parameters_min: int | None = Field(default=None, gt=0)
    target_parameters_max: int | None = Field(default=None, gt=0)
    intended_role: str
    capability_gates: list[str] = Field(min_length=1)
    status: Literal["TARGET_SPECIFICATION"] = "TARGET_SPECIFICATION"
    capability_claimed: Literal[False] = False

    @model_validator(mode="after")
    def valid_target_range(self) -> ModelTargetSpecification:
        low, high = self.target_parameters_min, self.target_parameters_max
        if (low is None) != (high is None) or (low is not None and high is not None and low > high):
            raise ValueError("target ranges require ordered minimum and maximum")
        if low is not None and high is not None and self.target_parameters is not None:
            if not low <= self.target_parameters <= high:
                raise ValueError("point target must lie inside the target range")
        return self


BU1LD_MODEL_SPECS: tuple[ModelTargetSpecification, ...] = (
    ModelTargetSpecification(
        family_id="hermes",
        display_name="Hermes",
        target_parameters=12_000_000_000,
        intended_role="Private, local-first assistant with grounded tool use and recovery.",
        capability_gates=[
            "held-out exact task completion",
            "typed tool arguments and denied-call handling",
            "grounded citation and calibrated abstention",
            "bounded local memory, latency, and privacy",
        ],
    ),
    ModelTargetSpecification(
        family_id="prometheus",
        display_name="Prometheus",
        target_parameters_min=12_000_000_000,
        target_parameters_max=20_000_000_000,
        intended_role=(
            "Domain specialist adapted for research, software engineering, and design without "
            "erasing general safety or calibration."
        ),
        capability_gates=[
            "domain-disjoint specialist evaluations",
            "base-versus-adapter regression controls",
            "source-grounded research and executable verification",
            "adapter isolation, rollback, and provenance",
        ],
    ),
    ModelTargetSpecification(
        family_id="perseus",
        display_name="Perseus",
        target_parameters=1_024_000_000_000,
        intended_role=(
            "Sparse generalist designed for globally accessible inference with reliable routing."
        ),
        capability_gates=[
            "active-parameter and memory budget compliance",
            "multilingual and low-resource evaluations",
            "tool-use reliability and side-effect containment",
            "quality parity against dense and smaller sparse baselines",
        ],
    ),
    ModelTargetSpecification(
        family_id="atlas",
        display_name="Atlas",
        target_parameters=None,
        intended_role=(
            "Deep multimodal intelligence with long-context grounding and explicit uncertainty."
        ),
        capability_gates=[
            "long-context retrieval and citation correctness",
            "spatiotemporal world-state consistency",
            "novel held-out reasoning with strong external baselines",
            "behavioral capability reporting without sentience claims",
        ],
    ),
    ModelTargetSpecification(
        family_id="kronos",
        display_name="Kronos",
        target_parameters=8_000_000_000_000,
        intended_role=(
            "Long-horizon general intelligence research system with planning, adaptation, and "
            "fail-closed control."
        ),
        capability_gates=[
            "long-horizon plan execution and recovery",
            "continual-learning retention and rollback",
            "independent safety and autonomy evaluation",
            "distributed-training, serving, and cost evidence",
        ],
    ),
    ModelTargetSpecification(
        family_id="aion",
        display_name="Aion",
        target_parameters=None,
        intended_role=(
            "Governed research-protocol orchestration with deterministic authority boundaries "
            "and an optional learned transition router."
        ),
        capability_gates=[
            "frozen-protocol and evidence-lineage integrity",
            "human authority for execution and promotion transitions",
            "budget, kill-rule, and illegal-transition enforcement",
            "advantage over the deterministic controller without false promotion",
        ],
    ),
)


def model_spec(family_id: ModelFamilyId) -> ModelTargetSpecification:
    """Return one immutable target specification by family identifier."""

    return next(spec for spec in BU1LD_MODEL_SPECS if spec.family_id == family_id)
