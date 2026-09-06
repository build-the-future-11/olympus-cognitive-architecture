from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, cast

import torch
from pydantic import Field, model_validator
from torch import nn
from torch.nn import functional as F

from olympus.core.schemas import StrictModel
from olympus.foundry.data_pipeline import (
    DatasetManifestV2,
    InstructionExample,
    verify_dataset_manifest,
)
from olympus.foundry.resources import ResourceGovernor, memory_snapshot

AdapterMode = Literal["full", "lora", "qlora"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
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


class ByteTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    vocab_size = 260

    @classmethod
    def encode(cls, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        tokens = [byte + 4 for byte in text.encode("utf-8")]
        return ([cls.bos_id] if bos else []) + tokens + ([cls.eos_id] if eos else [])

    @classmethod
    def decode(cls, tokens: Iterable[int]) -> str:
        values = bytes(token - 4 for token in tokens if 4 <= token < cls.vocab_size)
        return values.decode("utf-8", errors="replace")


class TinyModelConfig(StrictModel):
    vocab_size: int = ByteTokenizer.vocab_size
    width: int = Field(default=64, ge=16, le=512)
    layers: int = Field(default=2, ge=1, le=12)
    heads: int = Field(default=4, ge=1, le=16)
    max_sequence_tokens: int = Field(default=192, ge=32, le=2_048)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_heads(self) -> TinyModelConfig:
        if self.width % self.heads:
            raise ValueError("model width must be divisible by attention heads")
        return self


class SFTConfig(StrictModel):
    mode: AdapterMode = "full"
    seed: int = Field(default=7, ge=0, le=2**32 - 1)
    epochs: int = Field(default=2, ge=1, le=100)
    batch_size: int = Field(default=4, ge=1, le=128)
    learning_rate: float = Field(default=3e-3, gt=0.0, le=1.0)
    gradient_accumulation_steps: int = Field(default=2, ge=1, le=128)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0, le=100.0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    mixed_precision: bool = True
    pack_sequences: bool = True
    lora_rank: int = Field(default=4, ge=1, le=128)
    lora_alpha: float = Field(default=8.0, gt=0.0, le=1_024.0)
    category_mix_weights: dict[str, float] = Field(default_factory=dict)
    device: Literal["auto", "cpu", "cuda", "mps"] = "cpu"
    model: TinyModelConfig = Field(default_factory=TinyModelConfig)


class TrainingSummary(StrictModel):
    run_id: str
    mode: AdapterMode
    dataset_manifest_sha256: str
    base_checkpoint_sha256: str | None
    checkpoint_path: str
    checkpoint_sha256: str
    adapter_path: str | None
    adapter_sha256: str | None
    log_path: str
    completed_epochs: int
    optimizer_steps: int
    trainable_parameters: int
    total_parameters: int
    initial_validation_loss: float
    final_validation_loss: float
    final_validation_perplexity: float
    mixed_precision_requested: bool
    mixed_precision_effective: bool
    elapsed_seconds: float
    preflight_memory: dict[str, int | float]
    final_memory: dict[str, int | float]


class LoRALinear(nn.Module):
    def __init__(self, source: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.rank = rank
        self.scale = alpha / rank
        self.base_weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.base_bias = (
            nn.Parameter(source.bias.detach().clone(), requires_grad=False)
            if source.bias is not None
            else None
        )
        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.base_weight, self.base_bias)
        adapter = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return base + adapter * self.scale


def _pack_nibbles(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten().to(torch.uint8)
    if flat.numel() % 2:
        flat = torch.cat((flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)))
    return flat[0::2] | (flat[1::2] << 4)


def _unpack_nibbles(packed: torch.Tensor, count: int) -> torch.Tensor:
    unpacked = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
    unpacked[0::2] = packed & 0x0F
    unpacked[1::2] = packed >> 4
    return unpacked[:count]


class QLoRALinear(nn.Module):
    """Trainable LoRA over a frozen, genuinely nibble-packed symmetric 4-bit base."""

    def __init__(self, source: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.rank = rank
        self.scale = alpha / rank
        weight = source.weight.detach().float()
        row_scale = weight.abs().amax(dim=1).clamp_min(1e-8) / 7.0
        quantized = torch.round(weight / row_scale[:, None]).clamp(-7, 7).to(torch.int8) + 8
        self.packed_weight: torch.Tensor
        self.row_scale: torch.Tensor
        self.base_bias: torch.Tensor | None
        self.register_buffer("packed_weight", _pack_nibbles(quantized))
        self.register_buffer("row_scale", row_scale)
        if source.bias is None:
            self.register_buffer("base_bias", None)
        else:
            self.register_buffer("base_bias", source.bias.detach().clone())
        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def _weight(self, dtype: torch.dtype) -> torch.Tensor:
        values = _unpack_nibbles(
            self.packed_weight, self.in_features * self.out_features
        ).reshape(self.out_features, self.in_features)
        signed = values.to(torch.int8) - 8
        return signed.to(dtype) * self.row_scale.to(dtype)[:, None]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self._weight(inputs.dtype), self.base_bias)
        adapter = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return base + adapter * self.scale


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinyModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.qkv = nn.Linear(config.width, config.width * 3)
        self.output = nn.Linear(config.width, config.width)
        self.dropout = config.dropout

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, length, width = inputs.shape
        qkv = self.qkv(inputs).reshape(batch, length, 3, self.heads, self.head_width)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        projected = self.output(
            attended.transpose(1, 2).contiguous().reshape(batch, length, width)
        )
        return cast(torch.Tensor, projected)


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, config.width * 4),
            nn.GELU(),
            nn.Linear(config.width * 4, config.width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        return cast(torch.Tensor, inputs + self.mlp(self.mlp_norm(inputs)))


class TinyCausalLM(nn.Module):
    def __init__(self, config: TinyModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.max_sequence_tokens, config.width)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.layers))
        self.output_norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.config.max_sequence_tokens:
            raise ValueError("input exceeds model sequence limit")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            hidden = block(hidden)
        return cast(torch.Tensor, self.lm_head(self.output_norm(hidden)))


def apply_adapters(model: nn.Module, mode: AdapterMode, rank: int, alpha: float) -> None:
    if mode == "full":
        return
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            replacement: nn.Module
            if mode == "lora":
                replacement = LoRALinear(child, rank, alpha)
            else:
                replacement = QLoRALinear(child, rank, alpha)
            setattr(model, name, replacement)
        else:
            apply_adapters(child, mode, rank, alpha)
    for parameter in model.parameters():
        if "lora_" not in next(
            (name for name, value in model.named_parameters() if value is parameter), ""
        ):
            parameter.requires_grad = False


def _load_examples(
    manifest: DatasetManifestV2, split_name: str, manifest_root: Path
) -> list[InstructionExample]:
    descriptor = next(split for split in manifest.splits if split.name == split_name)
    path = Path(descriptor.path)
    if not path.is_absolute():
        path = manifest_root / path
    return [
        InstructionExample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _encode_examples(
    examples: list[InstructionExample], config: SFTConfig
) -> list[tuple[list[int], list[int]]]:
    expanded: list[InstructionExample] = []
    for example in examples:
        weight = config.category_mix_weights.get(example.category, 1.0)
        if weight <= 0:
            continue
        expanded.extend([example] * max(1, int(round(weight))))
    rows: list[tuple[list[int], list[int]]] = []
    for example in expanded:
        prefix = f"<|user|>\n{example.prompt.strip()}\n<|assistant|>\n"
        prefix_tokens = ByteTokenizer.encode(prefix, bos=True)
        answer_tokens = ByteTokenizer.encode(f"{example.response.strip()}\n<|end|>", eos=True)
        tokens = (prefix_tokens + answer_tokens)[: config.model.max_sequence_tokens]
        labels = [-100] * min(len(prefix_tokens), len(tokens)) + tokens[len(prefix_tokens) :]
        if not any(label >= 0 for label in labels):
            raise ValueError(f"sequence limit removes the entire answer for {example.id}")
        rows.append((tokens, labels))
    if not rows:
        raise ValueError("dataset mixing removed every training example")
    if not config.pack_sequences:
        return rows
    packed: list[tuple[list[int], list[int]]] = []
    current_tokens: list[int] = []
    current_labels: list[int] = []
    for tokens, labels in rows:
        if current_tokens and len(current_tokens) + len(tokens) > config.model.max_sequence_tokens:
            packed.append((current_tokens, current_labels))
            current_tokens, current_labels = [], []
        current_tokens.extend(tokens)
        current_labels.extend(labels)
    if current_tokens:
        packed.append((current_tokens, current_labels))
    return packed


def _batches(
    rows: list[tuple[list[int], list[int]]], batch_size: int, generator: torch.Generator
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    order = torch.randperm(len(rows), generator=generator).tolist()
    for start in range(0, len(order), batch_size):
        selected = [rows[index] for index in order[start : start + batch_size]]
        width = max(len(tokens) for tokens, _ in selected)
        tokens = torch.full((len(selected), width), ByteTokenizer.pad_id, dtype=torch.long)
        labels = torch.full((len(selected), width), -100, dtype=torch.long)
        for row, (item_tokens, item_labels) in enumerate(selected):
            tokens[row, : len(item_tokens)] = torch.tensor(item_tokens)
            labels[row, : len(item_labels)] = torch.tensor(item_labels)
        yield tokens, labels


def _loss(model: TinyCausalLM, tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(tokens[:, :-1])
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100
    )


def evaluate_loss(
    model: TinyCausalLM,
    rows: list[tuple[list[int], list[int]]],
    *,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    generator = torch.Generator().manual_seed(0)
    with torch.inference_mode():
        for tokens, labels in _batches(rows, 8, generator):
            losses.append(float(_loss(model, tokens.to(device), labels.to(device)).item()))
    if not losses:
        raise ValueError("evaluation contains no batches")
    return sum(losses) / len(losses)


def generate_text(
    model: TinyCausalLM,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int = 64,
) -> str:
    prefix = ByteTokenizer.encode(f"<|user|>\n{prompt}\n<|assistant|>\n", bos=True)
    window = prefix[-model.config.max_sequence_tokens :]
    tokens = torch.tensor([window], device=device)
    generated_start = tokens.shape[1]
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(tokens[:, -model.config.max_sequence_tokens :])[:, -1]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=1)
            if int(next_token.item()) == ByteTokenizer.eos_id:
                break
    return ByteTokenizer.decode(tokens[0, generated_start:].tolist())


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)


def _model_from_checkpoint(path: Path, device: torch.device) -> tuple[TinyCausalLM, dict[str, Any]]:
    checkpoint = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=True))
    model_config = TinyModelConfig.model_validate(checkpoint["model_config"])
    training_config = SFTConfig.model_validate(checkpoint["training_config"])
    model = TinyCausalLM(model_config)
    apply_adapters(
        model,
        training_config.mode,
        training_config.lora_rank,
        training_config.lora_alpha,
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device), checkpoint


def load_trained_model(path: Path, *, device: str = "cpu") -> TinyCausalLM:
    model, _ = _model_from_checkpoint(path, _resolve_device(device))
    model.eval()
    return model


def _validated_resume_config(
    requested: SFTConfig, checkpoint: dict[str, Any]
) -> SFTConfig:
    """Permit extending epochs while keeping every artifact-defining setting frozen."""
    loaded = SFTConfig.model_validate(checkpoint["training_config"])
    requested_frozen = requested.model_dump(mode="json", exclude={"epochs"})
    loaded_frozen = loaded.model_dump(mode="json", exclude={"epochs"})
    if requested_frozen != loaded_frozen:
        raise ValueError(
            "resume checkpoint configuration mismatch; only the epoch target may change"
        )
    completed_epochs = int(checkpoint["completed_epochs"])
    if requested.epochs < completed_epochs:
        raise ValueError("resume epoch target cannot be below completed epochs")
    return loaded.model_copy(update={"epochs": requested.epochs})


def run_sft(
    manifest_path: Path,
    output_root: Path,
    *,
    config: SFTConfig,
    base_checkpoint: Path | None = None,
    resume_checkpoint: Path | None = None,
) -> TrainingSummary:
    if config.mode != "full" and base_checkpoint is None and resume_checkpoint is None:
        raise ValueError("LoRA and QLoRA require a base checkpoint or resume checkpoint")
    if config.mode == "full" and base_checkpoint is not None:
        raise ValueError("full SFT does not accept a base checkpoint")
    manifest = verify_dataset_manifest(manifest_path)
    if manifest.manifest_sha256 is None:
        raise ValueError("dataset manifest has no immutable hash")
    output_root = output_root.resolve()
    governor = ResourceGovernor(output_root / ".workload.lock")
    start = time.perf_counter()
    with governor:
        effective_config = config
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        generator = torch.Generator().manual_seed(config.seed)
        device = _resolve_device(config.device)
        base_sha: str | None = None
        completed_epochs = 0
        optimizer_steps = 0

        if resume_checkpoint is not None:
            model, resumed = _model_from_checkpoint(resume_checkpoint, device)
            if resumed["dataset_manifest_sha256"] != manifest.manifest_sha256:
                raise ValueError("resume checkpoint dataset hash does not match")
            effective_config = _validated_resume_config(config, resumed)
            completed_epochs = int(resumed["completed_epochs"])
            optimizer_steps = int(resumed["optimizer_steps"])
            base_sha = cast(str | None, resumed.get("base_checkpoint_sha256"))
        elif base_checkpoint is not None:
            base_model, _ = _model_from_checkpoint(base_checkpoint, device)
            model = TinyCausalLM(config.model).to(device)
            model.load_state_dict(base_model.state_dict())
            apply_adapters(model, config.mode, config.lora_rank, config.lora_alpha)
            model.to(device)
            base_sha = _sha256(base_checkpoint)
        else:
            model = TinyCausalLM(config.model).to(device)

        train_rows = _encode_examples(
            _load_examples(manifest, "train", manifest_path.parent), effective_config
        )
        validation_rows = _encode_examples(
            _load_examples(manifest, "validation", manifest_path.parent), effective_config
        )

        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("training configuration has no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=effective_config.learning_rate,
            weight_decay=effective_config.weight_decay,
        )
        if resume_checkpoint is not None:
            optimizer.load_state_dict(resumed["optimizer_state"])
            generator.set_state(resumed["generator_state"])

        initial_validation = evaluate_loss(model, validation_rows, device=device)
        mixed_effective = effective_config.mixed_precision and device.type == "cuda"
        run_id = (
            f"sft-{effective_config.mode}-{manifest.manifest_sha256[:12]}"
            f"-s{effective_config.seed}"
        )
        log_path = output_root / run_id / "metrics.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_mode = "a" if resume_checkpoint is not None else "w"
        optimizer.zero_grad(set_to_none=True)
        with log_path.open(log_mode, encoding="utf-8") as log:
            for epoch in range(completed_epochs, effective_config.epochs):
                model.train()
                epoch_losses: list[float] = []
                batches = list(_batches(train_rows, effective_config.batch_size, generator))
                for batch_index, (tokens, labels) in enumerate(batches):
                    autocast = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if mixed_effective
                        else nullcontext()
                    )
                    with autocast:
                        loss = _loss(model, tokens.to(device), labels.to(device))
                        scaled_loss = loss / effective_config.gradient_accumulation_steps
                    scaled_loss.backward()  # type: ignore[no-untyped-call]
                    epoch_losses.append(float(loss.detach().cpu().item()))
                    should_step = (
                        (batch_index + 1)
                        % effective_config.gradient_accumulation_steps
                        == 0
                        or batch_index + 1 == len(batches)
                    )
                    if should_step:
                        nn.utils.clip_grad_norm_(
                            trainable, effective_config.gradient_clip_norm
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_steps += 1
                completed_epochs = epoch + 1
                validation_loss = evaluate_loss(model, validation_rows, device=device)
                record = {
                    "epoch": completed_epochs,
                    "optimizer_steps": optimizer_steps,
                    "train_loss": sum(epoch_losses) / len(epoch_losses),
                    "validation_loss": validation_loss,
                    "validation_perplexity": math.exp(min(validation_loss, 20.0)),
                }
                log.write(json.dumps(record, sort_keys=True) + "\n")
                log.flush()
                os.fsync(log.fileno())

        final_validation = evaluate_loss(model, validation_rows, device=device)
        checkpoint_path = output_root / run_id / "checkpoint.pt"
        checkpoint_payload: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "model_config": effective_config.model.model_dump(mode="json"),
            "training_config": effective_config.model_dump(mode="json"),
            "dataset_manifest_sha256": manifest.manifest_sha256,
            "base_checkpoint_sha256": base_sha,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "completed_epochs": completed_epochs,
            "optimizer_steps": optimizer_steps,
            "generator_state": generator.get_state(),
            "validation_loss": final_validation,
        }
        _atomic_torch_save(checkpoint_path, checkpoint_payload)
        adapter_path: Path | None = None
        if effective_config.mode != "full":
            adapter_path = output_root / run_id / "adapter.pt"
            adapter_state = {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
                if "lora_" in name
            }
            _atomic_torch_save(
                adapter_path,
                {
                    "schema_version": 1,
                    "mode": effective_config.mode,
                    "rank": effective_config.lora_rank,
                    "alpha": effective_config.lora_alpha,
                    "base_checkpoint_sha256": base_sha,
                    "dataset_manifest_sha256": manifest.manifest_sha256,
                    "adapter_state": adapter_state,
                },
            )
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in trainable)
        return TrainingSummary(
            run_id=run_id,
            mode=effective_config.mode,
            dataset_manifest_sha256=manifest.manifest_sha256,
            base_checkpoint_sha256=base_sha,
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=_sha256(checkpoint_path),
            adapter_path=str(adapter_path) if adapter_path else None,
            adapter_sha256=_sha256(adapter_path) if adapter_path else None,
            log_path=str(log_path),
            completed_epochs=completed_epochs,
            optimizer_steps=optimizer_steps,
            trainable_parameters=trainable_parameters,
            total_parameters=total_parameters,
            initial_validation_loss=initial_validation,
            final_validation_loss=final_validation,
            final_validation_perplexity=math.exp(min(final_validation, 20.0)),
            mixed_precision_requested=effective_config.mixed_precision,
            mixed_precision_effective=mixed_effective,
            elapsed_seconds=time.perf_counter() - start,
            preflight_memory=governor.preflight.as_dict() if governor.preflight else {},
            final_memory=memory_snapshot().as_dict(),
        )
