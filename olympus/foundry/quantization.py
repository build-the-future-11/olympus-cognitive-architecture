from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, cast

import torch
from pydantic import Field

from olympus.core.schemas import StrictModel
from olympus.foundry.artifacts import validate_output_paths
from olympus.foundry.data_pipeline import load_verified_dataset
from olympus.foundry.resources import memory_snapshot
from olympus.foundry.sft import (
    LossNormalization,
    OutputProtocol,
    TinyCausalLM,
    TinyModelConfig,
    _checkpoint_training_config,
    _encode_examples,
    _model_from_payload,
    _output_protocol_version,
    _pack_nibbles,
    _read_checkpoint,
    _require_finite,
    _unpack_nibbles,
    evaluate_loss,
    generate_text,
)

QuantizationBits = Literal[4, 8]


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
    suite: str = "olympus-quantization-evaluation-v1"
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
    loss_normalization: LossNormalization = "legacy_batch_mean_v1"
    output_protocol: OutputProtocol = "legacy_raw_bytes_v1"
    max_generation_tokens: int | None = Field(default=None, gt=0)


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


def _read_quantized_payload(path: Path) -> tuple[dict[str, Any], str, int]:
    """Bind all artifact metadata and weights to one deserialized byte snapshot."""
    raw = path.read_bytes()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("quantized artifact must contain a mapping")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported quantized artifact schema")
    bits = payload.get("bits")
    if type(bits) is not int or bits not in {4, 8}:
        raise ValueError("unsupported quantized artifact bits")
    if payload.get("format") != f"olympus-symmetric-int{bits}-v1":
        raise ValueError("quantized artifact format does not match its bits")
    return payload, hashlib.sha256(raw).hexdigest(), len(raw)


def _model_from_quantized_payload(payload: dict[str, Any]) -> TinyCausalLM:
    bits = cast(QuantizationBits, payload["bits"])
    model = TinyCausalLM(TinyModelConfig.model_validate(payload["model_config"]))
    expected = model.state_dict()
    tensors = payload.get("tensors")
    if not isinstance(tensors, dict) or tensors.keys() != expected.keys():
        raise ValueError("quantized artifact tensor names do not match its model")
    state: dict[str, torch.Tensor] = {}
    for name, reference in expected.items():
        tensor = tensors[name]
        if (
            not isinstance(tensor, dict)
            or tensor.get("shape") != list(reference.shape)
            or tensor.get("floating") is not reference.is_floating_point()
        ):
            raise ValueError("quantized artifact tensor metadata does not match its model")
        if reference.is_floating_point():
            axis = 0 if reference.ndim >= 2 else None
            encoded = tensor.get("quantized")
            scale = tensor.get("scale")
            scale_shape = [reference.shape[0]] if axis == 0 else [1]
            encoded_count = (reference.numel() + 1) // 2 if bits == 4 else reference.numel()
            if (
                tensor.get("axis") != axis
                or type(tensor.get("count")) is not int
                or tensor["count"] != reference.numel()
                or not isinstance(encoded, torch.Tensor)
                or encoded.dtype != (torch.uint8 if bits == 4 else torch.int8)
                or encoded.numel() != encoded_count
                or not isinstance(scale, torch.Tensor)
                or not scale.is_floating_point()
                or list(scale.shape) != scale_shape
            ):
                raise ValueError("quantized artifact encoding does not match its declared bits")
            _require_finite(scale, "quantized artifact scales")
            if not bool((scale > 0).all()):
                raise ValueError("quantized artifact scales must be positive")
        else:
            value = tensor.get("value")
            if not isinstance(value, torch.Tensor) or value.dtype != reference.dtype:
                raise ValueError("quantized artifact nonfloating tensor dtype does not match")
        state[name] = _dequantize_tensor(tensor, bits)
    _require_finite(state, "quantized artifact model state")
    model.load_state_dict(state)
    model.eval()
    return model


def load_quantized_model(path: Path) -> TinyCausalLM:
    payload, _, _ = _read_quantized_payload(path)
    return _model_from_quantized_payload(payload)


def _latency(model: TinyCausalLM, prompt: str, output_protocol: OutputProtocol) -> float:
    start = time.perf_counter()
    generate_text(
        model, prompt, device=torch.device("cpu"), max_new_tokens=16,
        output_protocol=output_protocol,
    )
    return (time.perf_counter() - start) * 1_000


def quantize_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    output_root: Path,
    *,
    bits: QuantizationBits,
    max_generation_tokens: int = 48,
    output_protocol: OutputProtocol = "framed_answer_eos_v2",
) -> QuantizationReport:
    protocol_version = _output_protocol_version(output_protocol)
    if type(bits) is not int or bits not in {4, 8}:
        raise ValueError("bits must be 4 or 8")
    if max_generation_tokens < 1:
        raise ValueError("max_generation_tokens must be positive")
    float_path = output_root / "float32-model.pt"
    artifact = output_root / f"model-int{bits}.pt"
    report_path = output_root / f"int{bits}-report.json"
    validate_output_paths([float_path, artifact, report_path], [checkpoint_path, manifest_path])
    checkpoint, source_sha = _read_checkpoint(checkpoint_path)
    source = _model_from_payload(checkpoint, torch.device("cpu"))
    training = _checkpoint_training_config(checkpoint)
    if training.mode != "full":
        raise ValueError("post-training quantization currently requires a merged full checkpoint")
    manifest, examples_by_split = load_verified_dataset(manifest_path)
    if checkpoint.get("dataset_manifest_sha256") != manifest.manifest_sha256:
        raise ValueError("checkpoint and quantization dataset hashes do not match")
    examples = examples_by_split["test"]
    del examples_by_split
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(float_path, source.state_dict())
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
            "source_checkpoint_sha256": source_sha,
            "tensors": tensor_payloads,
        },
    )
    start = time.perf_counter()
    artifact_payload, artifact_sha, quantized_size = _read_quantized_payload(artifact)
    if (
        artifact_payload.get("source_checkpoint_sha256") != source_sha
        or artifact_payload["bits"] != bits
        or TinyModelConfig.model_validate(artifact_payload["model_config"]) != training.model
    ):
        raise ValueError("quantized artifact changed before evaluation admission")
    quantized = _model_from_quantized_payload(artifact_payload)
    startup_ms = (time.perf_counter() - start) * 1_000

    eval_config = training.model_copy(update={"pack_sequences": False})
    rows = _encode_examples(examples, eval_config)
    source_loss = evaluate_loss(
        source, rows, device=torch.device("cpu"), normalization=training.loss_normalization
    )
    quantized_loss = evaluate_loss(
        quantized, rows, device=torch.device("cpu"), normalization=training.loss_normalization
    )
    prompt = "List the safe steps before starting a memory-heavy local model workload."
    source_latency = _latency(source, prompt, output_protocol)
    quantized_latency = _latency(quantized, prompt, output_protocol)
    tool_examples = [item for item in examples if item.category in {"tool_use", "agent_behavior"}]
    tool_exact = sum(
        generate_text(
            quantized, item.prompt, device=torch.device("cpu"),
            max_new_tokens=max_generation_tokens, output_protocol=output_protocol,
        )
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
    loss_change = (quantized_loss - source_loss) / max(source_loss, 1e-12)
    report = QuantizationReport(
        schema_version=protocol_version,
        suite=f"olympus-quantization-evaluation-v{protocol_version}",
        output_protocol=output_protocol,
        max_generation_tokens=max_generation_tokens,
        loss_normalization=training.loss_normalization,
        format=f"olympus-symmetric-int{bits}-v1",
        bits=bits,
        source_checkpoint_sha256=source_sha,
        artifact_path=str(artifact.resolve()),
        artifact_sha256=artifact_sha,
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
        passed_quality_gate=(abs(loss_change) <= 0.02 and context_limit_enforced),
    )
    _atomic_json(report_path, report.model_dump(mode="json"))
    return report
