from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, cast

import torch
from pydantic import Field

from olympus.core.schemas import StrictModel
from olympus.foundry.data_pipeline import InstructionExample, verify_dataset_manifest
from olympus.foundry.resources import memory_snapshot
from olympus.foundry.sft import (
    SFTConfig,
    TinyCausalLM,
    TinyModelConfig,
    _encode_examples,
    _model_from_checkpoint,
    _pack_nibbles,
    _unpack_nibbles,
    evaluate_loss,
    generate_text,
)

QuantizationBits = Literal[4, 8]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: object) -> None:
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


class QuantizedTensor(StrictModel):
    name: str
    shape: list[int]
    floating: bool
    axis: int | None


class QuantizationReport(StrictModel):
    schema_version: int = 1
    format: str
    bits: QuantizationBits
    source_checkpoint_sha256: str
    artifact_path: str
    artifact_sha256: str
    float_weights_bytes: int = Field(gt=0)
    quantized_bytes: int = Field(gt=0)
    size_reduction_fraction: float
    source_loss: float
    quantized_loss: float
    loss_change_fraction: float
    source_latency_ms: float
    quantized_latency_ms: float
    startup_latency_ms: float
    max_context_tokens: int
    context_limit_enforced: bool
    tool_exact_match_rate: float = Field(ge=0.0, le=1.0)
    peak_rss_bytes: int = Field(ge=0)
    passed_quality_gate: bool


def _quantize_tensor(tensor: torch.Tensor, bits: QuantizationBits) -> dict[str, Any]:
    value = tensor.detach().cpu()
    if not value.is_floating_point():
        return {"floating": False, "shape": list(value.shape), "value": value}
    floating = value.float()
    axis = 0 if floating.ndim >= 2 else None
    limit = 127 if bits == 8 else 7
    if axis == 0:
        scale = floating.abs().reshape(floating.shape[0], -1).amax(dim=1).clamp_min(1e-12) / limit
        shaped_scale = scale.reshape(floating.shape[0], *([1] * (floating.ndim - 1)))
    else:
        scale = floating.abs().amax().reshape(1).clamp_min(1e-12) / limit
        shaped_scale = scale
    signed = torch.round(floating / shaped_scale).clamp(-limit, limit).to(torch.int8)
    if bits == 8:
        quantized = signed
        count = signed.numel()
    else:
        quantized = _pack_nibbles(signed + 8)
        count = signed.numel()
    return {
        "floating": True,
        "shape": list(value.shape),
        "axis": axis,
        "scale": scale,
        "quantized": quantized,
        "count": count,
    }


def _dequantize_tensor(payload: dict[str, Any], bits: QuantizationBits) -> torch.Tensor:
    if not payload["floating"]:
        return cast(torch.Tensor, payload["value"])
    shape = cast(list[int], payload["shape"])
    if bits == 8:
        signed = cast(torch.Tensor, payload["quantized"]).to(torch.int8)
    else:
        packed = cast(torch.Tensor, payload["quantized"])
        signed = _unpack_nibbles(packed, int(payload["count"])).to(torch.int8) - 8
    signed = signed.reshape(shape).float()
    scale = cast(torch.Tensor, payload["scale"]).float()
    if payload["axis"] == 0:
        scale = scale.reshape(shape[0], *([1] * (len(shape) - 1)))
    return signed * scale


def load_quantized_model(path: Path) -> TinyCausalLM:
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=True))
    bits = cast(QuantizationBits, int(payload["bits"]))
    model = TinyCausalLM(TinyModelConfig.model_validate(payload["model_config"]))
    state = {
        name: _dequantize_tensor(tensor_payload, bits)
        for name, tensor_payload in cast(dict[str, dict[str, Any]], payload["tensors"]).items()
    }
    model.load_state_dict(state)
    model.eval()
    return model


def _latency(model: TinyCausalLM, prompt: str) -> float:
    start = time.perf_counter()
    generate_text(model, prompt, device=torch.device("cpu"), max_new_tokens=16)
    return (time.perf_counter() - start) * 1_000


def quantize_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    output_root: Path,
    *,
    bits: QuantizationBits,
) -> QuantizationReport:
    source, checkpoint = _model_from_checkpoint(checkpoint_path, torch.device("cpu"))
    training = SFTConfig.model_validate(checkpoint["training_config"])
    if training.mode != "full":
        raise ValueError("post-training quantization currently requires a merged full checkpoint")
    output_root.mkdir(parents=True, exist_ok=True)
    float_path = output_root / "float32-model.pt"
    _atomic_torch_save(float_path, source.state_dict())
    artifact = output_root / f"model-int{bits}.pt"
    tensor_payloads = {
        name: _quantize_tensor(tensor, bits) for name, tensor in source.state_dict().items()
    }
    _atomic_torch_save(
        artifact,
        {
            "schema_version": 1,
            "format": f"olympus-symmetric-int{bits}-v1",
            "bits": bits,
            "model_config": training.model.model_dump(mode="json"),
            "source_checkpoint_sha256": _sha256(checkpoint_path),
            "tensors": tensor_payloads,
        },
    )
    start = time.perf_counter()
    quantized = load_quantized_model(artifact)
    startup_ms = (time.perf_counter() - start) * 1_000

    manifest = verify_dataset_manifest(manifest_path)
    test_descriptor = next(split for split in manifest.splits if split.name == "test")
    test_path = Path(test_descriptor.path)
    if not test_path.is_absolute():
        test_path = manifest_path.parent / test_path
    examples = [
        InstructionExample.model_validate_json(line)
        for line in test_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_config = training.model_copy(update={"pack_sequences": False})
    rows = _encode_examples(examples, eval_config)
    source_loss = evaluate_loss(source, rows, device=torch.device("cpu"))
    quantized_loss = evaluate_loss(quantized, rows, device=torch.device("cpu"))
    prompt = "List the safe steps before starting a memory-heavy local model workload."
    source_latency = _latency(source, prompt)
    quantized_latency = _latency(quantized, prompt)
    tool_examples = [item for item in examples if item.category in {"tool_use", "agent_behavior"}]
    tool_exact = sum(
        generate_text(quantized, item.prompt, device=torch.device("cpu"), max_new_tokens=48)
        .strip()
        .casefold()
        == item.response.strip().casefold()
        for item in tool_examples
    ) / len(tool_examples)
    context_limit_enforced = False
    try:
        quantized(torch.zeros((1, training.model.max_sequence_tokens + 1), dtype=torch.long))
    except ValueError:
        context_limit_enforced = True
    float_size = float_path.stat().st_size
    quantized_size = artifact.stat().st_size
    loss_change = (quantized_loss - source_loss) / max(source_loss, 1e-12)
    report = QuantizationReport(
        format=f"olympus-symmetric-int{bits}-v1",
        bits=bits,
        source_checkpoint_sha256=_sha256(checkpoint_path),
        artifact_path=str(artifact.resolve()),
        artifact_sha256=_sha256(artifact),
        float_weights_bytes=float_size,
        quantized_bytes=quantized_size,
        size_reduction_fraction=1.0 - quantized_size / float_size,
        source_loss=source_loss,
        quantized_loss=quantized_loss,
        loss_change_fraction=loss_change,
        source_latency_ms=source_latency,
        quantized_latency_ms=quantized_latency,
        startup_latency_ms=startup_ms,
        max_context_tokens=training.model.max_sequence_tokens,
        context_limit_enforced=context_limit_enforced,
        tool_exact_match_rate=tool_exact,
        peak_rss_bytes=memory_snapshot().process_peak_rss_bytes,
        passed_quality_gate=(
            abs(loss_change) <= 0.02
            and context_limit_enforced
            and tool_exact >= 0.75
        ),
    )
    report_path = output_root / f"int{bits}-report.json"
    _atomic_json(report_path, report.model_dump(mode="json"))
    return report
