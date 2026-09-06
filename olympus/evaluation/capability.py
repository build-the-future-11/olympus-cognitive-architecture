from __future__ import annotations

import json
import os
from collections import Counter
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from pydantic import Field, model_validator

from olympus.core.schemas import StrictModel


class FailureOrigin(StrEnum):
    NONE = "none"
    MODEL = "model"
    ORCHESTRATION = "orchestration"
    TOOL = "tool"


class ToolCall(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    arguments: dict[str, object]


class CapabilityTask(StrictModel):
    task_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,127}$")
    category: str = Field(min_length=1, max_length=100)
    workflow: Literal["direct", "tool", "percy"]
    prompt: str = Field(min_length=1, max_length=50_000)
    expected_text: str | None = None
    expected_tool: ToolCall | None = None
    should_refuse: bool = False
    should_express_uncertainty: bool = False

    @model_validator(mode="after")
    def require_an_objective(self) -> CapabilityTask:
        if self.expected_text is None and self.expected_tool is None and not self.should_refuse:
            raise ValueError("task must define text, a tool call, or a refusal boundary")
        return self


class CandidateOutcome(StrictModel):
    task_id: str
    text: str = ""
    tool_call: ToolCall | None = None
    refused: bool = False
    expressed_uncertainty: bool = False
    orchestration_error: str | None = None
    tool_error: str | None = None


class TaskScore(StrictModel):
    task_id: str
    category: str
    workflow: str
    task_exact: bool
    tool_selection_exact: bool | None
    tool_arguments_exact: bool | None
    refusal_correct: bool
    uncertainty_correct: bool
    failure_origin: FailureOrigin
    detail: str


class CapabilityReport(StrictModel):
    schema_version: int = 1
    suite: str = "olympus-capability-contract-v1"
    candidate_id: str
    baseline_id: str
    task_count: int = Field(gt=0)
    task_exact_rate: float = Field(ge=0.0, le=1.0)
    tool_selection_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    tool_arguments_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    refusal_accuracy: float = Field(ge=0.0, le=1.0)
    uncertainty_accuracy: float = Field(ge=0.0, le=1.0)
    percy_workflow_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    failure_counts: dict[str, int]
    scores: list[TaskScore]


def _normalize(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def _canonical_arguments(arguments: dict[str, object]) -> str:
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _score(task: CapabilityTask, outcome: CandidateOutcome) -> TaskScore:
    if outcome.orchestration_error:
        return TaskScore(
            task_id=task.task_id,
            category=task.category,
            workflow=task.workflow,
            task_exact=False,
            tool_selection_exact=None,
            tool_arguments_exact=None,
            refusal_correct=False,
            uncertainty_correct=False,
            failure_origin=FailureOrigin.ORCHESTRATION,
            detail=outcome.orchestration_error,
        )
    if outcome.tool_error:
        origin = FailureOrigin.TOOL
        detail = outcome.tool_error
    else:
        origin = FailureOrigin.NONE
        detail = "all required fields matched"

    selection: bool | None = None
    arguments: bool | None = None
    if task.expected_tool is not None:
        selection = (
            outcome.tool_call is not None and outcome.tool_call.name == task.expected_tool.name
        )
        arguments = (
            selection is True
            and outcome.tool_call is not None
            and _canonical_arguments(outcome.tool_call.arguments)
            == _canonical_arguments(task.expected_tool.arguments)
        )
    text_exact = task.expected_text is None or _normalize(outcome.text) == _normalize(
        task.expected_text
    )
    refusal_correct = outcome.refused == task.should_refuse
    uncertainty_correct = (
        outcome.expressed_uncertainty == task.should_express_uncertainty
    )
    exact = (
        not outcome.tool_error
        and text_exact
        and selection is not False
        and arguments is not False
        and refusal_correct
        and uncertainty_correct
    )
    if not exact and origin is FailureOrigin.NONE:
        origin = FailureOrigin.MODEL
        detail = "candidate response did not match the frozen task contract"
    return TaskScore(
        task_id=task.task_id,
        category=task.category,
        workflow=task.workflow,
        task_exact=exact,
        tool_selection_exact=selection,
        tool_arguments_exact=arguments,
        refusal_correct=refusal_correct,
        uncertainty_correct=uncertainty_correct,
        failure_origin=origin,
        detail=detail,
    )


def _rate(values: list[bool]) -> float | None:
    return None if not values else sum(values) / len(values)


def evaluate_capabilities(
    tasks: list[CapabilityTask],
    outcomes: list[CandidateOutcome],
    *,
    candidate_id: str,
    baseline_id: str,
    output_path: Path | None = None,
) -> CapabilityReport:
    if not tasks:
        raise ValueError("capability suite must contain at least one task")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("capability suite contains duplicate task IDs")
    outcome_map = {outcome.task_id: outcome for outcome in outcomes}
    if len(outcome_map) != len(outcomes):
        raise ValueError("candidate outcomes contain duplicate task IDs")
    if set(outcome_map) != set(task_ids):
        missing = sorted(set(task_ids) - set(outcome_map))
        extra = sorted(set(outcome_map) - set(task_ids))
        raise ValueError(f"outcomes do not match tasks: missing={missing}, extra={extra}")
    scores = [_score(task, outcome_map[task.task_id]) for task in tasks]
    tool_selection = [
        score.tool_selection_exact
        for score in scores
        if score.tool_selection_exact is not None
    ]
    tool_arguments = [
        score.tool_arguments_exact
        for score in scores
        if score.tool_arguments_exact is not None
    ]
    percy = [score.task_exact for score in scores if score.workflow == "percy"]
    report = CapabilityReport(
        candidate_id=candidate_id,
        baseline_id=baseline_id,
        task_count=len(tasks),
        task_exact_rate=sum(score.task_exact for score in scores) / len(scores),
        tool_selection_rate=_rate(tool_selection),
        tool_arguments_rate=_rate(tool_arguments),
        refusal_accuracy=sum(score.refusal_correct for score in scores) / len(scores),
        uncertainty_accuracy=sum(score.uncertainty_correct for score in scores) / len(scores),
        percy_workflow_rate=_rate(percy),
        failure_counts=dict(
            sorted(Counter(score.failure_origin.value for score in scores).items())
        ),
        scores=scores,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (report.model_dump_json(indent=2) + "\n").encode()
        with NamedTemporaryFile(
            dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(output_path)
    return report
