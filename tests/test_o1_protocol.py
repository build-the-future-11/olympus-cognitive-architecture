import json
from pathlib import Path
from typing import Literal, cast

import pytest

from olympus.evaluation.o1 import (
    O1Arm,
    O1Protocol,
    O1Score,
    O1Task,
    _load_jsonl,
    load_frozen_protocol,
    paired_arm_effects,
    parse_and_score,
    render_prompt,
    route_category,
    sha256_file,
    summarize_scores,
    validate_score_keys,
)


def _task() -> O1Task:
    return O1Task(
        task_id="o1.compositionality.001",
        category="compositionality",
        prompt="Apply reverse, then append Q, to the string abc.",
        answer="cbaQ",
        retrieval_context="reverse(abc)=cba; append Q adds Q at the end.",
        inference_required=True,
        source="repository-authored deterministic transformation",
        perturbation_group="composition-1",
    )


ArmKind = Literal[
    "general",
    "oracle_specialist",
    "routed_specialist",
    "retrieval",
    "retrieval_only",
    "shuffled_retrieval",
]
RetrievalPolicy = Literal["none", "relevant", "shuffled"]


def _arm(kind: ArmKind, index: int = 0) -> O1Arm:
    policies: dict[ArmKind, RetrievalPolicy] = {
        "general": "none",
        "oracle_specialist": "none",
        "routed_specialist": "none",
        "retrieval": "relevant",
        "retrieval_only": "relevant",
        "shuffled_retrieval": "shuffled",
    }
    return O1Arm(
        arm_id=f"{kind}_{index}",
        kind=kind,
        system_template="Solve this category carefully: {category}. Return exact JSON only.",
        retrieval_policy=policies[kind],
    )


def _protocol(tasks_path: Path, arms_path: Path, **updates: object) -> O1Protocol:
    values: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": "olympus-o1-v1",
        "state": "FROZEN_PRE_OUTCOME",
        "frozen_at_utc": "2026-09-06T00:00:00Z",
        "task_manifest": tasks_path.name,
        "task_manifest_sha256": sha256_file(tasks_path),
        "arm_manifest": arms_path.name,
        "arm_manifest_sha256": sha256_file(arms_path),
        "provider": "ollama",
        "provider_version": "0.11.10",
        "model": "qwen3:0.6b",
        "model_digest": "sha256:" + "1" * 64,
        "context_tokens": 2048,
        "maximum_output_tokens": 64,
        "temperature": 0.2,
        "seeds": [17, 31, 47],
        "primary_metric": "exact_match",
        "secondary_metrics": ["category_accuracy", "brier", "consistency"],
        "bootstrap_resamples": 10_000,
        "promotion_threshold": 0.05,
        "failure_threshold": 0.0,
        "raw_output_directory": "runs/o1_v1",
        "scorer_version": "o1-exact-json-v1",
        "no_outcome_access_attestation": True,
    }
    values.update(updates)
    return O1Protocol.model_validate(values)


def _write_complete_manifests(tmp_path: Path) -> tuple[Path, Path]:
    categories = [
        "compositionality",
        "pragmatics",
        "counterfactual",
        "ambiguity",
        "long_tail",
        "perturbation",
    ]
    tasks = []
    for index in range(60):
        category = categories[index % len(categories)]
        tasks.append(
            _task().model_copy(
                update={"task_id": f"o1.{category}.{index + 1:03d}", "category": category}
            )
        )
    arms = [_arm(kind) for kind in (
        "general",
        "oracle_specialist",
        "routed_specialist",
        "retrieval",
        "retrieval_only",
        "shuffled_retrieval",
    )]
    tasks_path = tmp_path / "tasks.jsonl"
    arms_path = tmp_path / "arms.jsonl"
    tasks_path.write_text("\n".join(task.model_dump_json() for task in tasks) + "\n")
    arms_path.write_text("\n".join(arm.model_dump_json() for arm in arms) + "\n")
    return tasks_path, arms_path


def test_parser_scores_exact_json_and_calibration() -> None:
    score = parse_and_score('{"answer":" cbaQ ","confidence":0.8}', _task(), "general", 17)
    assert score.exact is True
    assert score.brier == pytest.approx(0.04)
    malformed = parse_and_score("cbaQ", _task(), "general", 17)
    assert malformed.exact is False
    assert malformed.parse_error


def test_prompt_policies_do_not_leak_context() -> None:
    direct = O1Arm(
        arm_id="general_direct",
        kind="general",
        system_template="Solve the task carefully and return only the requested structure.",
        retrieval_policy="none",
    )
    _, prompt = render_prompt(_task(), direct)
    assert "REFERENCE CONTEXT" not in prompt
    shuffled = direct.model_copy(
        update={"arm_id": "shuffled", "kind": "shuffled_retrieval", "retrieval_policy": "shuffled"}
    )
    _, shuffled_prompt = render_prompt(_task(), shuffled, shuffled_context="irrelevant fact")
    assert "irrelevant fact" in shuffled_prompt
    with pytest.raises(ValueError, match="requires a control context"):
        render_prompt(_task(), shuffled)


def test_summary_reports_uncertainty_and_consistency() -> None:
    task = _task()
    scores = [
        parse_and_score('{"answer":"cbaQ","confidence":0.9}', task, "general", 17),
        parse_and_score('{"answer":"wrong","confidence":0.7}', task, "general", 31),
        parse_and_score('{"answer":"cbaQ","confidence":0.6}', task, "general", 47),
    ]
    summary = summarize_scores(scores, [task], resamples=10_000)["general"]
    assert summary["exact_match"] == pytest.approx(2 / 3)
    assert summary["consistency"] == 0.0
    assert summary["tasks_n"] == 1
    assert summary["experimental_unit"] == "task"


def test_frozen_protocol_rejects_manifest_tampering(tmp_path: Path) -> None:
    (tmp_path / "tasks.jsonl").write_text(json.dumps(_task().model_dump()) + "\n")
    arm = O1Arm(
        arm_id="general_direct",
        kind="general",
        system_template="Solve the task carefully and return only the requested structure.",
        retrieval_policy="none",
    )
    (tmp_path / "arms.jsonl").write_text(json.dumps(arm.model_dump()) + "\n")
    protocol = {
        "schema_version": 1,
        "protocol_id": "olympus-o1-v1",
        "state": "FROZEN_PRE_OUTCOME",
        "frozen_at_utc": "2026-09-06T00:00:00Z",
        "task_manifest": "tasks.jsonl",
        "task_manifest_sha256": "0" * 64,
        "arm_manifest": "arms.jsonl",
        "arm_manifest_sha256": "0" * 64,
        "provider": "ollama",
        "provider_version": "0.11.10",
        "model": "qwen3:0.6b",
        "model_digest": "sha256:" + "1" * 64,
        "context_tokens": 2048,
        "maximum_output_tokens": 64,
        "temperature": 0.2,
        "seeds": [17, 31, 47],
        "primary_metric": "exact_match",
        "secondary_metrics": ["category_accuracy", "brier", "consistency"],
        "bootstrap_resamples": 10000,
        "promotion_threshold": 0.05,
        "failure_threshold": 0.0,
        "raw_output_directory": "runs/o1_v1",
        "scorer_version": "o1-exact-json-v1",
        "no_outcome_access_attestation": True,
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="task manifest hash"):
        load_frozen_protocol(path)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"no_outcome_access_attestation": False}, "no-outcome-access"),
        ({"seeds": [17, 17, 47]}, "seeds must be unique"),
        ({"failure_threshold": 0.6, "promotion_threshold": 0.5}, "failure threshold"),
    ],
)
def test_protocol_rejects_invalid_freeze_conditions(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    tasks_path, arms_path = _write_complete_manifests(tmp_path)
    with pytest.raises(ValueError, match=message):
        _protocol(tasks_path, arms_path, **updates)


@pytest.mark.parametrize(
    ("updates", "field"),
    [
        ({"task_manifest": "../tasks.jsonl"}, "task_manifest"),
        ({"arm_manifest": "/tmp/arms.jsonl"}, "arm_manifest"),
        ({"raw_output_directory": "runs/../escaped"}, "raw_output_directory"),
        ({"raw_output_directory": "/tmp/o1"}, "raw_output_directory"),
    ],
)
def test_protocol_rejects_paths_outside_its_frozen_layout(
    tmp_path: Path, updates: dict[str, object], field: str
) -> None:
    tasks_path, arms_path = _write_complete_manifests(tmp_path)
    with pytest.raises(ValueError, match=field):
        _protocol(tasks_path, arms_path, **updates)


def test_frozen_protocol_loads_complete_manifests(tmp_path: Path) -> None:
    tasks_path, arms_path = _write_complete_manifests(tmp_path)
    protocol = _protocol(tasks_path, arms_path)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(protocol.model_dump_json())

    loaded, tasks, arms = load_frozen_protocol(protocol_path)

    assert loaded == protocol
    assert len(tasks) == 60
    assert len(arms) == 6


def test_jsonl_loader_reports_empty_and_malformed_inputs(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="contains no rows"):
        _load_jsonl(empty, O1Task)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{}\n")
    with pytest.raises(ValueError, match="line 1"):
        _load_jsonl(malformed, O1Task)


@pytest.mark.parametrize(
    ("prompt", "category"),
    [
        ("Start with 'abc' and reverse it.", "compositionality"),
        ("Use this convention: green means yes.", "pragmatics"),
        ("Evaluate this deterministic circuit after changing x.", "counterfactual"),
        ("Resolve this sentence: I saw her duck.", "ambiguity"),
        ("Answer using the supplied registry entry.", "long_tail"),
        ("Compare the sealed samples after one edit.", "perturbation"),
    ],
)
def test_router_recognizes_each_frozen_category(prompt: str, category: str) -> None:
    assert route_category(prompt) == category


@pytest.mark.parametrize(
    "prompt",
    ["There is no routing marker here.", "Convention: start with 'abc' and reverse it."],
)
def test_router_rejects_missing_or_ambiguous_markers(prompt: str) -> None:
    with pytest.raises(ValueError, match="exactly one category marker"):
        route_category(prompt)


def test_retrieval_and_routed_prompts_bind_expected_context() -> None:
    retrieval = _arm("retrieval")
    system, prompt = render_prompt(_task(), retrieval)
    assert "compositionality" in system
    assert _task().retrieval_context in prompt

    routed = _arm("routed_specialist")
    routed_task = _task().model_copy(update={"prompt": "Start with 'abc', reverse, then append Q."})
    routed_system, _ = render_prompt(routed_task, routed)
    assert "compositionality" in routed_system


def test_summary_rejects_empty_and_unknown_task_scores() -> None:
    with pytest.raises(ValueError, match="empty score set"):
        summarize_scores([], [_task()], resamples=10)
    unknown = O1Score(
        task_id="missing",
        arm_id="general",
        seed=17,
        exact=False,
        confidence=0.0,
        brier=1.0,
    )
    with pytest.raises(ValueError, match="unknown task"):
        summarize_scores([unknown], [_task()], resamples=10)


def test_paired_effects_report_wins_ties_and_losses() -> None:
    scores = []
    comparisons = {
        "o1.compositionality.001": [(False, True), (False, True), (True, True)],
        "o1.compositionality.002": [(True, True), (False, False), (True, True)],
        "o1.compositionality.003": [(True, False), (True, False), (False, False)],
    }
    for task_id, outcomes in comparisons.items():
        for seed, (reference, candidate) in zip((17, 31, 47), outcomes, strict=True):
            scores.extend(
                [
                    O1Score(task_id=task_id, arm_id="general", seed=seed,
                            exact=reference, confidence=0.5, brier=0.25),
                    O1Score(task_id=task_id, arm_id="candidate", seed=seed,
                            exact=candidate, confidence=0.5, brier=0.25),
                ]
            )
    effect = cast(
        dict[str, object],
        paired_arm_effects(scores, reference_arm="general", resamples=100)["candidate"],
    )
    assert effect["accuracy_difference"] == 0.0
    assert (effect["wins"], effect["ties"], effect["losses"]) == (1, 1, 1)
    assert effect["paired_n"] == 9
    assert effect["paired_tasks_n"] == 3
    assert effect["experimental_unit"] == "task"


def test_paired_effects_reject_duplicates_missing_reference_and_unpaired_arms() -> None:
    score = O1Score(
        task_id=_task().task_id,
        arm_id="candidate",
        seed=17,
        exact=True,
        confidence=1.0,
        brier=0.0,
    )
    with pytest.raises(ValueError, match="duplicate score"):
        paired_arm_effects([score, score], reference_arm="general", resamples=10)
    with pytest.raises(ValueError, match="missing reference arm"):
        paired_arm_effects([score], reference_arm="general", resamples=10)
    reference = score.model_copy(update={"arm_id": "general"})
    extra = score.model_copy(update={"seed": 31})
    with pytest.raises(ValueError, match="is not paired"):
        paired_arm_effects([reference, score, extra], reference_arm="general", resamples=10)


def test_score_key_validation_binds_resumed_rows_to_frozen_protocol() -> None:
    task = _task()
    arm = _arm("general")
    score = O1Score(
        task_id=task.task_id,
        arm_id=arm.arm_id,
        seed=17,
        exact=True,
        confidence=1.0,
        brier=0.0,
    )
    assert validate_score_keys([score], [task], [arm], [17]) == {
        (task.task_id, arm.arm_id, 17)
    }
    with pytest.raises(ValueError, match="duplicate score"):
        validate_score_keys([score, score], [task], [arm], [17])
    with pytest.raises(ValueError, match="outside the frozen protocol"):
        validate_score_keys([score.model_copy(update={"seed": 31})], [task], [arm], [17])
    with pytest.raises(ValueError, match="incomplete score set"):
        validate_score_keys([], [task], [arm], [17], require_complete=True)
