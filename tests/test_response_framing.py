"""Artificial response-boundary oracles; no training or capability measurements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from olympus.foundry import eval_suite, promotion, quantization, sft
from olympus.foundry.data_pipeline import REQUIRED_CATEGORIES, InstructionExample


def test_generation_source_is_bound_to_this_worktree() -> None:
    expected = Path(__file__).resolve().parents[1] / "olympus/foundry/sft.py"
    assert Path(sft.__file__).resolve() == expected


def _oracle_tokens(text: str, *, eos: bool = False) -> list[int]:
    # Independent byte specification: do not use the serializer or tokenizer
    # under test to manufacture the target whose correctness we then assert.
    return [byte + 4 for byte in text.encode("utf-8")] + ([2] if eos else [])


class _ScriptedLM(nn.Module):
    """Prescribed next-token logits exercised through the real generation loop."""

    def __init__(self, script: list[int]) -> None:
        super().__init__()
        self.config = sft.TinyModelConfig(
            width=16, layers=1, heads=2, max_sequence_tokens=256
        )
        self.script = script
        self.inputs: list[list[int]] = []

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        index = len(self.inputs)
        assert index < len(self.script), "generation continued beyond the oracle script"
        self.inputs.append(tokens[0].tolist())
        logits = torch.full((*tokens.shape, 260), -1_000.0, device=tokens.device)
        logits[:, -1, self.script[index]] = 1_000.0
        return logits


@pytest.mark.parametrize(
    "answer",
    [
        "Ready.",
        '{"answer":7}',
        '{"text":"literal <|end|> inside","tail":"preserved"}',
        '{"text":"literal\\n<|end|>"}',
        "first\n<|end|>\nlast",
        "literal at the end\n<|end|>",
        "two literals\n<|end|>\n<|end|>",
        "café λ 🌿",
    ],
)
def test_perfect_serialized_target_decodes_to_exact_answer(answer: str) -> None:
    target = _oracle_tokens(answer + "\n<|end|>", eos=True)
    model = _ScriptedLM(target + _oracle_tokens("unreachable"))
    generated = sft.generate_text(
        cast(sft.TinyCausalLM, model),
        "  Return the requested value.\n",
        device=torch.device("cpu"),
        max_new_tokens=len(model.script),
    )
    assert generated == answer
    assert len(model.inputs) == len(target)
    assert model.inputs[0] == [1] + _oracle_tokens(
        "<|user|>\nReturn the requested value.\n<|assistant|>\n"
    )
    assert eval_suite._normalize(generated) == eval_suite._normalize(answer)
    assert eval_suite._format_compliant(answer, generated)
    if answer.startswith("{"):
        assert json.loads(generated) == json.loads(answer)


def test_training_target_and_prompt_mask_keep_the_original_byte_contract() -> None:
    example = InstructionExample(
        id="oracle-json",
        category="instruction",
        prompt="  Return the requested value.\n",
        response='  {"answer":7}\n',
        license="MIT",
        source="artificial://response-framing-oracle",
        split="train",
    )
    config = sft.SFTConfig(
        pack_sequences=False,
        model=sft.TinyModelConfig(width=16, layers=1, heads=2, max_sequence_tokens=128),
    )
    prefix = [1] + _oracle_tokens("<|user|>\nReturn the requested value.\n<|assistant|>\n")
    target = _oracle_tokens('{"answer":7}\n<|end|>', eos=True)
    tokens, labels = sft._encode_examples([example], config)[0]
    assert tokens == prefix + target
    assert labels == [-100] * len(prefix) + target


@pytest.mark.parametrize("cut", ["Ready.\n<|", "Ready.\n<|end|", "Ready.\n<|end|>"])
def test_output_budget_preserves_partial_or_complete_marker_without_eos(cut: str) -> None:
    target = _oracle_tokens("Ready.\n<|end|>", eos=True)
    model = _ScriptedLM(target)
    generated = sft.generate_text(
        cast(sft.TinyCausalLM, model), "Return a value.", device=torch.device("cpu"),
        max_new_tokens=len(_oracle_tokens(cut)),
    )
    assert generated == cut
    assert len(model.inputs) < len(target)


@pytest.mark.parametrize("answer", ["", "Ready.", "Ready.\n<|end", "Ready.\n<|end|> extra"])
def test_eos_without_the_exact_terminal_frame_does_not_trim_content(answer: str) -> None:
    target = _oracle_tokens(answer, eos=True)
    model = _ScriptedLM(target + _oracle_tokens("unreachable"))
    assert sft.generate_text(
        cast(sft.TinyCausalLM, model), "Return a value.", device=torch.device("cpu"),
        max_new_tokens=len(model.script),
    ) == answer
    assert len(model.inputs) == len(target)


def test_eos_must_immediately_follow_marker_tokens_before_frame_removal() -> None:
    # The tokenizer has always ignored PAD. A PAD between marker and EOS must
    # not turn a nonterminal token sequence into an apparently valid frame.
    raw = "Ready.\n<|end|>"
    model = _ScriptedLM(_oracle_tokens(raw) + [0, 2])
    assert sft.generate_text(
        cast(sft.TinyCausalLM, model), "Return a value.", device=torch.device("cpu"),
        max_new_tokens=len(model.script),
    ) == raw


def test_partial_utf8_at_output_budget_is_not_repaired() -> None:
    model = _ScriptedLM(_oracle_tokens("café\n<|end|>", eos=True))
    generated = sft.generate_text(
        cast(sft.TinyCausalLM, model), "Return a value.", device=torch.device("cpu"),
        max_new_tokens=4,
    )
    assert generated == "caf\ufffd"
    assert not eval_suite._format_compliant("café", generated)


def test_legacy_oracle_preserves_unstripped_prompt_and_raw_json_failure() -> None:
    answer = '{"answer":7}'
    raw = answer + "\n<|end|>"
    model = _ScriptedLM(_oracle_tokens(raw, eos=True))
    prompt = "  Return the requested value.\n"
    generated = sft.generate_text(
        cast(sft.TinyCausalLM, model), prompt, device=torch.device("cpu"),
        max_new_tokens=len(model.script), output_protocol="legacy_raw_bytes_v1",
    )
    assert generated == raw
    assert model.inputs[0] == [1] + _oracle_tokens(
        "<|user|>\n  Return the requested value.\n\n<|assistant|>\n"
    )
    assert eval_suite._normalize(generated) != eval_suite._normalize(answer)
    assert not eval_suite._format_compliant(answer, generated)
    assert sft.ByteTokenizer.decode(_oracle_tokens(raw, eos=True)) == raw


def test_unknown_output_protocol_is_rejected_before_generation() -> None:
    model = _ScriptedLM([2])
    with pytest.raises(ValueError, match="output protocol"):
        sft.generate_text(
            cast(sft.TinyCausalLM, model), "Return a value.", device=torch.device("cpu"),
            output_protocol=cast(sft.OutputProtocol, "unknown"),
        )
    assert model.inputs == []


@pytest.mark.parametrize(
    "output_protocol,version,expected_exact",
    [("legacy_raw_bytes_v1", 1, 0.0), ("framed_answer_eos_v2", 2, 1.0)],
)
def test_report_writers_use_the_declared_protocol_with_generation_oracles(
    deep_artifact_dataset: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_protocol: sft.OutputProtocol,
    version: int,
    expected_exact: float,
) -> None:
    # Real artificial checkpoint/data admission, losses and quantization; only
    # next-token logits are replaced for an independent scoring-boundary oracle.
    answers = {
        f"Question test_{category}?": f"Answer test_{category}"
        for category in REQUIRED_CATEGORIES
    }
    protocols: list[sft.OutputProtocol] = []

    def oracle_generation(
        model: sft.TinyCausalLM,
        prompt: str,
        *,
        device: torch.device,
        max_new_tokens: int,
        output_protocol: sft.OutputProtocol = "framed_answer_eos_v2",
    ) -> str:
        del model
        protocols.append(output_protocol)
        answer = answers.get(prompt, "Ready.")
        scripted = _ScriptedLM(_oracle_tokens(answer + "\n<|end|>", eos=True))
        return sft.generate_text(
            cast(sft.TinyCausalLM, scripted), prompt, device=device,
            max_new_tokens=max_new_tokens, output_protocol=output_protocol,
        )

    monkeypatch.setattr(eval_suite, "generate_text", oracle_generation)
    monkeypatch.setattr(quantization, "generate_text", oracle_generation)
    checkpoint: Path = deep_artifact_dataset["checkpoint"]
    checkpoint_before = checkpoint.read_bytes()
    evaluation = eval_suite.evaluate_checkpoint(
        checkpoint, deep_artifact_dataset["manifest_path"], tmp_path / "evaluation.json",
        max_generation_tokens=48, output_protocol=output_protocol,
    )
    quantized = quantization.quantize_checkpoint(
        checkpoint, deep_artifact_dataset["manifest_path"], tmp_path / "quantized", bits=8,
        max_generation_tokens=48, output_protocol=output_protocol,
    )
    assert protocols and set(protocols) == {output_protocol}
    assert evaluation.schema_version == quantized.schema_version == version
    assert evaluation.suite == f"olympus-held-out-capabilities-v{version}"
    assert quantized.suite == f"olympus-quantization-evaluation-v{version}"
    assert evaluation.output_protocol == quantized.output_protocol == output_protocol
    assert evaluation.max_generation_tokens == quantized.max_generation_tokens == 48
    assert evaluation.exact_match_rate == quantized.tool_exact_match_rate == expected_exact
    assert all(item.exact_match_rate == expected_exact for item in evaluation.workflow_scores)
    assert evaluation.loss_normalization == quantized.loss_normalization == "legacy_batch_mean_v1"
    assert quantized.source_loss == pytest.approx(evaluation.overall_candidate_loss, abs=1e-7)
    artifact, _, _ = quantization._read_quantized_payload(Path(quantized.artifact_path))
    assert artifact["schema_version"] == 1
    assert artifact["format"] == "olympus-symmetric-int8-v1"
    assert checkpoint.read_bytes() == checkpoint_before


def _rewrite_report(path: Path, updates: dict[str, Any], remove: tuple[str, ...] = ()) -> None:
    report = json.loads(path.read_bytes())
    report.update(updates)
    for field in remove:
        report.pop(field, None)
    path.write_text(json.dumps(report), encoding="utf-8")


def _use_v2_reports(evidence: dict[str, Any]) -> None:
    for key, suite in (
        ("evaluation_path", "olympus-held-out-capabilities-v2"),
        ("quantization_report_path", "olympus-quantization-evaluation-v2"),
    ):
        _rewrite_report(evidence[key], {
            "schema_version": 2, "suite": suite,
            "output_protocol": "framed_answer_eos_v2", "max_generation_tokens": 48,
        })


def test_v2_evidence_is_admitted_without_changing_numerical_gates(
    deep_promotion_evidence: dict[str, Any],
) -> None:
    _use_v2_reports(deep_promotion_evidence)
    report = promotion.evaluate_promotion(**deep_promotion_evidence)
    # The fixture supplies artificial gate numbers and substitutes scale; this
    # asserts report compatibility, never a trained model's actual promotion.
    assert report.status == "PROMOTED"


def test_historical_reports_without_output_metadata_keep_v1_interpretation(
    deep_promotion_evidence: dict[str, Any],
) -> None:
    for key in ("evaluation_path", "quantization_report_path"):
        _rewrite_report(deep_promotion_evidence[key], {}, (
            "suite", "output_protocol", "max_generation_tokens",
        ))
    evaluation = eval_suite.HeldOutEvaluation.model_validate_json(
        deep_promotion_evidence["evaluation_path"].read_bytes()
    )
    quantized = quantization.QuantizationReport.model_validate_json(
        deep_promotion_evidence["quantization_report_path"].read_bytes()
    )
    assert evaluation.output_protocol == quantized.output_protocol == "legacy_raw_bytes_v1"
    assert evaluation.schema_version == quantized.schema_version == 1
    assert evaluation.max_generation_tokens is quantized.max_generation_tokens is None
    assert evaluation.exact_match_rate == 0.9
    assert quantized.tool_exact_match_rate == 0.8
    assert promotion.evaluate_promotion(**deep_promotion_evidence).status == "PROMOTED"


@pytest.mark.parametrize(
    "key,updates,remove,gate",
    [
        ("evaluation_path", {
            "schema_version": 1, "suite": "olympus-held-out-capabilities-v1",
            "output_protocol": "legacy_raw_bytes_v1",
        }, (), "quantization_evidence"),
        ("quantization_report_path", {
            "schema_version": 1, "suite": "olympus-quantization-evaluation-v1",
            "output_protocol": "legacy_raw_bytes_v1",
        }, (), "quantization_evidence"),
        ("evaluation_path", {"schema_version": 1}, (), "evaluation_evidence"),
        ("evaluation_path", {"suite": "olympus-held-out-capabilities-v1"}, (),
         "evaluation_evidence"),
        ("quantization_report_path", {"schema_version": 1}, (), "quantization_evidence"),
        ("quantization_report_path", {"suite": "olympus-quantization-evaluation-v1"}, (),
         "quantization_evidence"),
        ("evaluation_path", {}, ("output_protocol",), "evaluation_evidence"),
        ("quantization_report_path", {}, ("output_protocol",), "quantization_evidence"),
        ("evaluation_path", {}, ("max_generation_tokens",), "evaluation_evidence"),
        ("quantization_report_path", {}, ("max_generation_tokens",), "quantization_evidence"),
        ("quantization_report_path", {"max_generation_tokens": 47}, (), "quantization_evidence"),
    ],
)
def test_promotion_rejects_mixed_or_incomplete_output_protocol_evidence(
    deep_promotion_evidence: dict[str, Any],
    key: str,
    updates: dict[str, Any],
    remove: tuple[str, ...],
    gate: str,
) -> None:
    _use_v2_reports(deep_promotion_evidence)
    _rewrite_report(deep_promotion_evidence[key], updates, remove)
    report = promotion.evaluate_promotion(**deep_promotion_evidence)
    assert report.status == "NOT_PROMOTED"
    assert any(item.gate == gate and not item.passed for item in report.gates)
    assert report.release_manifest_path is None


@pytest.mark.parametrize("key", ["evaluation_path", "quantization_report_path"])
def test_unknown_report_output_protocol_is_rejected(
    deep_promotion_evidence: dict[str, Any], key: str,
) -> None:
    _rewrite_report(deep_promotion_evidence[key], {"output_protocol": "unknown"})
    with pytest.raises(ValidationError, match="output_protocol"):
        promotion.evaluate_promotion(**deep_promotion_evidence)


@pytest.mark.parametrize("operation", ["evaluation", "quantization"])
def test_nonpositive_report_generation_budget_is_rejected_before_io(
    tmp_path: Path, operation: str,
) -> None:
    missing = tmp_path / "missing-checkpoint.pt"
    with pytest.raises(ValueError, match="max_generation_tokens"):
        if operation == "evaluation":
            eval_suite.evaluate_checkpoint(missing, missing, tmp_path / "report.json",
                                           max_generation_tokens=0)
        else:
            quantization.quantize_checkpoint(missing, missing, tmp_path / "quantized", bits=8,
                                             max_generation_tokens=0)
