from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, Self

import torch
from pydantic import Field, model_validator

from olympus.core.schemas import StrictModel
from olympus.foundry.data_pipeline import (
    REQUIRED_CATEGORIES,
    InstructionExample,
    verify_dataset_manifest,
)
from olympus.foundry.resources import memory_snapshot
from olympus.foundry.sft import (
    SFTConfig,
    TinyCausalLM,
    TinyModelConfig,
    _encode_examples,
    _model_from_checkpoint,
    evaluate_loss,
    generate_text,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


WORKFLOW_CATEGORIES: dict[str, tuple[str, ...]] = {
    "tool_workflow": ("tool_use", "agent_behavior"),
    "result_comparison": ("reasoning", "self_correction", "research"),
    "multi_step_planning": ("planning", "safety"),
}


class CategoryEvaluation(StrictModel):
    category: str
    records: int = Field(gt=0)
    baseline_loss: float = Field(ge=0.0)
    candidate_loss: float = Field(ge=0.0)
    loss_change_fraction: float
    regressed: bool

    @model_validator(mode="after")
    def consistent_metrics(self) -> Self:
        change = (self.candidate_loss - self.baseline_loss) / max(self.baseline_loss, 1e-12)
        if not math.isclose(change, self.loss_change_fraction, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("category loss change does not match losses")
        if self.regressed != (change > 0.05):
            raise ValueError("category regression flag does not match losses")
        return self


class WorkflowScore(StrictModel):
    name: str
    categories: list[str]
    examples: int = Field(gt=0)
    exact_match_rate: float = Field(ge=0.0, le=1.0)
    nonempty_rate: float = Field(ge=0.0, le=1.0)


class HeldOutEvaluation(StrictModel):
    schema_version: Literal[3] = 3
    suite: Literal["olympus-held-out-capabilities-v3"] = "olympus-held-out-capabilities-v3"
    sampling_policy: Literal["all-records-once-unpacked-v1"] = "all-records-once-unpacked-v1"
    loss_aggregation: Literal["supervised-token-mean-v2"] = "supervised-token-mean-v2"
    checkpoint_path: str
    checkpoint_sha256: str
    dataset_manifest_sha256: str
    baseline_identity: str
    test_records: int = Field(gt=0)
    overall_baseline_loss: float = Field(ge=0)
    overall_candidate_loss: float = Field(ge=0)
    overall_baseline_perplexity: float = Field(ge=1)
    overall_candidate_perplexity: float = Field(ge=1)
    category_results: list[CategoryEvaluation]
    regression_count: int = Field(ge=0)
    exact_match_rate: float = Field(ge=0.0, le=1.0)
    format_compliance_rate: float = Field(ge=0.0, le=1.0)
    workflow_scores: list[WorkflowScore]
    median_generation_latency_ms: float = Field(ge=0.0)
    generated_bytes_per_second: float = Field(ge=0.0)
    peak_rss_bytes: int = Field(ge=0)
    passed_smoke_quality_gate: bool
    decision: str

    @model_validator(mode="after")
    def consistent_report(self) -> Self:
        counts = {item.category: item.records for item in self.category_results}
        if len(counts) != len(self.category_results) or set(counts) != set(REQUIRED_CATEGORIES):
            raise ValueError("evaluation must contain each required category exactly once")
        if sum(counts.values()) != self.test_records:
            raise ValueError("category counts do not match test record count")
        if self.regression_count != sum(item.regressed for item in self.category_results):
            raise ValueError("regression count does not match category results")
        names = [item.name for item in self.workflow_scores]
        if len(names) != len(set(names)) or set(names) != set(WORKFLOW_CATEGORIES):
            raise ValueError("evaluation must contain each required workflow exactly once")
        for score in self.workflow_scores:
            expected = WORKFLOW_CATEGORIES[score.name]
            if len(score.categories) != len(expected) or set(score.categories) != set(expected):
                raise ValueError("workflow category membership is incorrect")
            if score.examples != sum(counts[category] for category in expected):
                raise ValueError("workflow example count is incorrect")
            if score.nonempty_rate < score.exact_match_rate:
                raise ValueError("exact matches cannot exceed nonempty outputs")
        for loss, perplexity in (
            (self.overall_baseline_loss, self.overall_baseline_perplexity),
            (self.overall_candidate_loss, self.overall_candidate_perplexity),
        ):
            if not math.isclose(math.exp(min(loss, 20)), perplexity, rel_tol=1e-6):
                raise ValueError("perplexity does not match loss")
        expected_gate = (
            self.overall_candidate_loss < self.overall_baseline_loss
            and self.regression_count == 0
            and self.exact_match_rate >= 0.5
            and self.format_compliance_rate >= 0.8
            and min(score.exact_match_rate for score in self.workflow_scores) >= 0.5
        )
        if self.passed_smoke_quality_gate != expected_gate:
            raise ValueError("smoke quality decision does not match metrics")
        return self


def _load_test_examples(manifest_path: Path) -> tuple[str, list[InstructionExample]]:
    manifest = verify_dataset_manifest(manifest_path)
    if manifest.manifest_sha256 is None:
        raise ValueError("dataset manifest has no hash")
    descriptor = next(split for split in manifest.splits if split.name == "test")
    split_path = Path(descriptor.path)
    if not split_path.is_absolute():
        split_path = manifest_path.parent / split_path
    examples = [
        InstructionExample.model_validate_json(line)
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return manifest.manifest_sha256, examples


def _baseline_model(
    checkpoint: dict[str, Any], base_checkpoint: Path | None
) -> tuple[TinyCausalLM, str]:
    if base_checkpoint is not None:
        model, _ = _model_from_checkpoint(base_checkpoint, torch.device("cpu"))
        return model, f"checkpoint:{_sha256(base_checkpoint)}"
    training = SFTConfig.model_validate(checkpoint["training_config"])
    torch.manual_seed(training.seed)
    return TinyCausalLM(TinyModelConfig.model_validate(checkpoint["model_config"])), (
        f"deterministic-untrained-seed-{training.seed}"
    )


def _normalize(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def _format_compliant(expected: str, generated: str) -> bool:
    if expected.lstrip().startswith("{"):
        try:
            parsed = json.loads(generated)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, dict)
    return bool(generated.strip()) and "\ufffd" not in generated


def _workflow_scores(
    examples: list[InstructionExample], generated: dict[str, str]
) -> list[WorkflowScore]:
    results: list[WorkflowScore] = []
    for name, categories in WORKFLOW_CATEGORIES.items():
        selected = [example for example in examples if example.category in categories]
        exact = sum(
            _normalize(generated[item.id]) == _normalize(item.response) for item in selected
        )
        nonempty = sum(bool(generated[item.id].strip()) for item in selected)
        results.append(
            WorkflowScore(
                name=name,
                categories=list(categories),
                examples=len(selected),
                exact_match_rate=exact / len(selected),
                nonempty_rate=nonempty / len(selected),
            )
        )
    return results


def evaluate_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    base_checkpoint: Path | None = None,
    max_generation_tokens: int = 48,
) -> HeldOutEvaluation:
    device = torch.device("cpu")
    candidate, checkpoint = _model_from_checkpoint(checkpoint_path, device)
    baseline, baseline_identity = _baseline_model(checkpoint, base_checkpoint)
    manifest_sha, examples = _load_test_examples(manifest_path)
    if checkpoint["dataset_manifest_sha256"] != manifest_sha:
        raise ValueError("checkpoint and evaluation dataset hashes do not match")
    training = SFTConfig.model_validate(checkpoint["training_config"])
    evaluation_config = training.for_evaluation()
    all_rows = _encode_examples(examples, evaluation_config)
    baseline_loss = evaluate_loss(baseline, all_rows, device=device)
    candidate_loss = evaluate_loss(candidate, all_rows, device=device)

    category_results: list[CategoryEvaluation] = []
    for category in sorted({example.category for example in examples}):
        selected = [example for example in examples if example.category == category]
        rows = _encode_examples(selected, evaluation_config)
        before = evaluate_loss(baseline, rows, device=device)
        after = evaluate_loss(candidate, rows, device=device)
        change = (after - before) / max(before, 1e-12)
        category_results.append(
            CategoryEvaluation(
                category=category,
                records=len(selected),
                baseline_loss=before,
                candidate_loss=after,
                loss_change_fraction=change,
                regressed=change > 0.05,
            )
        )

    generated: dict[str, str] = {}
    latencies: list[float] = []
    generated_bytes = 0
    for example in examples:
        start = time.perf_counter()
        answer = generate_text(
            candidate,
            example.prompt,
            device=device,
            max_new_tokens=max_generation_tokens,
        )
        latencies.append(time.perf_counter() - start)
        generated_bytes += len(answer.encode())
        generated[example.id] = answer
    exact = sum(_normalize(generated[item.id]) == _normalize(item.response) for item in examples)
    compliant = sum(_format_compliant(item.response, generated[item.id]) for item in examples)
    regressions = sum(result.regressed for result in category_results)
    exact_rate = exact / len(examples)
    format_rate = compliant / len(examples)
    workflow_floor = min(score.exact_match_rate for score in _workflow_scores(examples, generated))
    passed = (
        candidate_loss < baseline_loss
        and regressions == 0
        and exact_rate >= 0.5
        and format_rate >= 0.8
        and workflow_floor >= 0.5
    )
    decision = (
        "Passed the bounded capability smoke gate; this does not satisfy Hermes promotion gates."
        if passed
        else "Failed the bounded capability smoke gate and is ineligible for model promotion."
    )
    report = HeldOutEvaluation(
        checkpoint_path=str(checkpoint_path.resolve()),
        checkpoint_sha256=_sha256(checkpoint_path),
        dataset_manifest_sha256=manifest_sha,
        baseline_identity=baseline_identity,
        test_records=len(examples),
        overall_baseline_loss=baseline_loss,
        overall_candidate_loss=candidate_loss,
        overall_baseline_perplexity=math.exp(min(baseline_loss, 20.0)),
        overall_candidate_perplexity=math.exp(min(candidate_loss, 20.0)),
        category_results=category_results,
        regression_count=regressions,
        exact_match_rate=exact_rate,
        format_compliance_rate=format_rate,
        workflow_scores=_workflow_scores(examples, generated),
        median_generation_latency_ms=statistics.median(latencies) * 1_000,
        generated_bytes_per_second=(
            generated_bytes / sum(latencies) if sum(latencies) else 0.0
        ),
        peak_rss_bytes=memory_snapshot().process_peak_rss_bytes,
        passed_smoke_quality_gate=passed,
        decision=decision,
    )
    _atomic_json(output_path, report.model_dump(mode="json"))
    return report
