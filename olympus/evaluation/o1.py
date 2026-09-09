from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from olympus.core.schemas import StrictModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class O1Task(StrictModel):
    task_id: str = Field(pattern=r"^o1\.[a-z_]+\.[0-9]{3}$")
    category: Literal[
        "compositionality",
        "pragmatics",
        "counterfactual",
        "ambiguity",
        "long_tail",
        "perturbation",
    ]
    prompt: str = Field(min_length=8, max_length=4_000)
    answer: str = Field(min_length=1, max_length=200)
    retrieval_context: str = Field(min_length=1, max_length=4_000)
    inference_required: bool
    source: str = Field(min_length=3, max_length=500)
    perturbation_group: str = Field(min_length=1, max_length=100)


class O1Arm(StrictModel):
    arm_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    kind: Literal[
        "general",
        "oracle_specialist",
        "routed_specialist",
        "retrieval",
        "retrieval_only",
        "shuffled_retrieval",
    ]
    system_template: str = Field(min_length=20, max_length=4_000)
    retrieval_policy: Literal["none", "relevant", "shuffled"]


class O1Protocol(StrictModel):
    schema_version: Literal[1] = 1
    protocol_id: str = Field(pattern=r"^olympus-o1-v[0-9]+$")
    state: Literal["FROZEN_PRE_OUTCOME"]
    frozen_at_utc: str
    task_manifest: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.jsonl$")
    task_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arm_manifest: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.jsonl$")
    arm_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: Literal["ollama"]
    provider_version: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    context_tokens: int = Field(ge=256, le=131_072)
    maximum_output_tokens: int = Field(ge=8, le=512)
    temperature: float = Field(ge=0.0, le=2.0)
    seeds: list[int] = Field(min_length=3)
    primary_metric: Literal["exact_match"]
    secondary_metrics: list[Literal["category_accuracy", "brier", "consistency"]]
    bootstrap_resamples: int = Field(ge=10_000)
    promotion_threshold: float = Field(ge=0.0, le=1.0)
    failure_threshold: float = Field(ge=0.0, le=1.0)
    raw_output_directory: str = Field(pattern=r"^runs/[A-Za-z0-9][A-Za-z0-9_.-]*$")
    scorer_version: Literal["o1-exact-json-v1"]
    no_outcome_access_attestation: bool

    @model_validator(mode="after")
    def integrity(self) -> O1Protocol:
        if not self.no_outcome_access_attestation:
            raise ValueError("protocol cannot freeze without no-outcome-access attestation")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("protocol seeds must be unique")
        if self.failure_threshold > self.promotion_threshold:
            raise ValueError("failure threshold cannot exceed promotion threshold")
        return self


class ParsedResponse(StrictModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)


class O1Score(StrictModel):
    task_id: str
    arm_id: str
    seed: int
    exact: bool
    confidence: float
    brier: float
    parse_error: str | None = None


def validate_score_keys(
    scores: list[O1Score],
    tasks: list[O1Task],
    arms: list[O1Arm],
    seeds: list[int],
    *,
    require_complete: bool = False,
) -> set[tuple[str, str, int]]:
    expected = {
        (task.task_id, arm.arm_id, seed) for seed in seeds for task in tasks for arm in arms
    }
    observed: set[tuple[str, str, int]] = set()
    for score in scores:
        key = (score.task_id, score.arm_id, score.seed)
        if key not in expected:
            raise ValueError(f"score is outside the frozen protocol: {key}")
        if key in observed:
            raise ValueError(f"duplicate score: {key}")
        observed.add(key)
    if require_complete and observed != expected:
        raise ValueError(
            f"incomplete score set: missing {len(expected - observed)} frozen observations; "
            f"found {len(observed - expected)} unexpected observations"
        )
    return observed


def _load_jsonl(path: Path, model_type: type[StrictModel]) -> list[StrictModel]:
    rows: list[StrictModel] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(model_type.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid {path} line {line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def load_frozen_protocol(path: Path) -> tuple[O1Protocol, list[O1Task], list[O1Arm]]:
    protocol = O1Protocol.model_validate_json(path.read_text())
    root = path.parent
    tasks_path = root / protocol.task_manifest
    arms_path = root / protocol.arm_manifest
    if sha256_file(tasks_path) != protocol.task_manifest_sha256:
        raise ValueError("task manifest hash does not match frozen protocol")
    if sha256_file(arms_path) != protocol.arm_manifest_sha256:
        raise ValueError("arm manifest hash does not match frozen protocol")
    tasks = [row for row in _load_jsonl(tasks_path, O1Task) if isinstance(row, O1Task)]
    arms = [row for row in _load_jsonl(arms_path, O1Arm) if isinstance(row, O1Arm)]
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("task manifest contains duplicate IDs")
    if len({arm.arm_id for arm in arms}) != len(arms):
        raise ValueError("arm manifest contains duplicate IDs")
    if len(tasks) < 60:
        raise ValueError("O1 requires at least 60 frozen tasks")
    required_categories = {
        "compositionality",
        "pragmatics",
        "counterfactual",
        "ambiguity",
        "long_tail",
        "perturbation",
    }
    if {task.category for task in tasks} != required_categories:
        raise ValueError("task manifest does not cover every required category")
    required_kinds = {
        "general",
        "oracle_specialist",
        "routed_specialist",
        "retrieval",
        "retrieval_only",
        "shuffled_retrieval",
    }
    if {arm.kind for arm in arms} != required_kinds:
        raise ValueError("arm manifest does not contain the complete six-arm comparison")
    return protocol, tasks, arms


def route_category(prompt: str) -> str:
    rules = [
        ("start with '", "compositionality"),
        ("convention:", "pragmatics"),
        ("deterministic circuit", "counterfactual"),
        ("sentence:", "ambiguity"),
        ("supplied registry", "long_tail"),
        ("sealed samples", "perturbation"),
    ]
    lowered = prompt.casefold()
    matches = [category for marker, category in rules if marker in lowered]
    if len(matches) != 1:
        raise ValueError(f"router expected exactly one category marker, found {matches}")
    return matches[0]


def render_prompt(
    task: O1Task, arm: O1Arm, *, shuffled_context: str | None = None
) -> tuple[str, str]:
    routed_category = (
        route_category(task.prompt) if arm.kind == "routed_specialist" else task.category
    )
    system = arm.system_template.replace("{category}", routed_category)
    context = ""
    if arm.retrieval_policy == "relevant":
        context = task.retrieval_context
    elif arm.retrieval_policy == "shuffled":
        if shuffled_context is None:
            raise ValueError("shuffled retrieval arm requires a control context")
        context = shuffled_context
    instruction = (
        "Return one JSON object only with keys answer and confidence. "
        "The answer must be short; confidence must be a number from 0 to 1."
    )
    prompt = f"TASK:\n{task.prompt}\n\n{instruction}"
    if context:
        prompt = f"REFERENCE CONTEXT:\n{context}\n\n{prompt}"
    return system, prompt


def parse_and_score(raw: str, task: O1Task, arm_id: str, seed: int) -> O1Score:
    try:
        parsed = ParsedResponse.model_validate(json.loads(raw))
        normalized = " ".join(parsed.answer.casefold().strip().split())
        expected = " ".join(task.answer.casefold().strip().split())
        exact = normalized == expected
        target = 1.0 if exact else 0.0
        return O1Score(
            task_id=task.task_id,
            arm_id=arm_id,
            seed=seed,
            exact=exact,
            confidence=parsed.confidence,
            brier=(parsed.confidence - target) ** 2,
        )
    except Exception as exc:
        return O1Score(
            task_id=task.task_id,
            arm_id=arm_id,
            seed=seed,
            exact=False,
            confidence=0.0,
            brier=1.0,
            parse_error=str(exc),
        )


def summarize_scores(
    scores: list[O1Score], tasks: list[O1Task], *, resamples: int
) -> dict[str, dict[str, object]]:
    if not scores:
        raise ValueError("cannot summarize an empty score set")
    category_by_task = {task.task_id: task.category for task in tasks}
    by_arm: dict[str, list[O1Score]] = defaultdict(list)
    for score in scores:
        if score.task_id not in category_by_task:
            raise ValueError(f"score references unknown task {score.task_id}")
        by_arm[score.arm_id].append(score)
    rng = random.Random(20260906)
    result: dict[str, dict[str, object]] = {}
    for arm_id, arm_scores in sorted(by_arm.items()):
        values = [float(score.exact) for score in arm_scores]
        scores_by_task: dict[str, list[O1Score]] = defaultdict(list)
        for score in arm_scores:
            scores_by_task[score.task_id].append(score)
        task_accuracies = [
            sum(float(score.exact) for score in task_scores) / len(task_scores)
            for _, task_scores in sorted(scores_by_task.items())
        ]
        boot = sorted(
            sum(rng.choice(task_accuracies) for _ in task_accuracies) / len(task_accuracies)
            for _ in range(resamples)
        )
        category = {}
        for name in sorted(set(category_by_task.values())):
            subset = [
                accuracy
                for task_id, accuracy in zip(
                    sorted(scores_by_task), task_accuracies, strict=True
                )
                if category_by_task[task_id] == name
            ]
            category[name] = sum(subset) / len(subset)
        consistent = [
            len({score.exact for score in task_scores}) == 1
            for task_scores in scores_by_task.values()
        ]
        result[arm_id] = {
            "n": len(values),
            "tasks_n": len(task_accuracies),
            "experimental_unit": "task",
            "exact_match": sum(values) / len(values),
            "exact_match_ci95": [
                boot[math.floor(0.025 * (len(boot) - 1))],
                boot[math.floor(0.975 * (len(boot) - 1))],
            ],
            "mean_brier": sum(score.brier for score in arm_scores) / len(arm_scores),
            "consistency": sum(consistent) / len(consistent),
            "parse_errors": sum(score.parse_error is not None for score in arm_scores),
            "category_accuracy": category,
        }
    return result


def paired_arm_effects(
    scores: list[O1Score], *, reference_arm: str, resamples: int
) -> dict[str, object]:
    by_arm: dict[str, dict[tuple[str, int], float]] = defaultdict(dict)
    for score in scores:
        key = (score.task_id, score.seed)
        if key in by_arm[score.arm_id]:
            raise ValueError(f"duplicate score for {score.arm_id} {key}")
        by_arm[score.arm_id][key] = float(score.exact)
    if reference_arm not in by_arm:
        raise ValueError(f"missing reference arm {reference_arm}")
    reference = by_arm[reference_arm]
    rng = random.Random(20260907)
    effects: dict[str, object] = {}
    for arm_id, observations in sorted(by_arm.items()):
        if arm_id == reference_arm:
            continue
        if set(observations) != set(reference):
            raise ValueError(f"arm {arm_id} is not paired with {reference_arm}")
        task_differences: dict[str, list[float]] = defaultdict(list)
        for task_id, seed in sorted(reference):
            task_differences[task_id].append(
                observations[(task_id, seed)] - reference[(task_id, seed)]
            )
        differences = [
            sum(seed_differences) / len(seed_differences)
            for _, seed_differences in sorted(task_differences.items())
        ]
        boot = sorted(
            sum(rng.choice(differences) for _ in differences) / len(differences)
            for _ in range(resamples)
        )
        effects[arm_id] = {
            "reference_arm": reference_arm,
            "paired_n": len(reference),
            "paired_tasks_n": len(differences),
            "experimental_unit": "task",
            "accuracy_difference": sum(differences) / len(differences),
            "difference_ci95": [
                boot[math.floor(0.025 * (len(boot) - 1))],
                boot[math.floor(0.975 * (len(boot) - 1))],
            ],
            "wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
        }
    return effects
