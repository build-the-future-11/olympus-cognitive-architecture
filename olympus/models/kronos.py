"""Executable temporal forecasting, planning, and guarded adaptation for Kronos.

This module contains trainable components and deterministic promotion controls.
It does not contain or imply a promoted Kronos checkpoint.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import math
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, cast

import torch
from pydantic import Field, model_validator
from torch import nn
from torch.nn import functional as F

from olympus.core.schemas import StrictModel


class EventKind(StrEnum):
    OBSERVATION = "observation"
    INTERVENTION = "intervention"
    OUTCOME = "outcome"
    MISSING = "missing"


class TemporalEvent(StrictModel):
    event_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    occurred_at: datetime
    kind: EventKind
    features: list[float] = Field(min_length=1, max_length=4_096)
    environment_version: str = Field(min_length=1, max_length=128)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    intervention: str | None = Field(default=None, max_length=1_000)
    outcome_observed: bool = False

    @model_validator(mode="after")
    def intervention_matches_kind(self) -> TemporalEvent:
        if self.kind is EventKind.INTERVENTION and not self.intervention:
            raise ValueError("intervention events must identify the intervention")
        if self.intervention and self.kind is not EventKind.INTERVENTION:
            raise ValueError("only intervention events may include an intervention")
        if self.occurred_at.utcoffset() is None:
            raise ValueError("event timestamps must be timezone-aware")
        if not all(math.isfinite(value) for value in self.features):
            raise ValueError("event features must be finite")
        return self


class TemporalBatch(StrictModel):
    event_ids: list[list[str]]
    environment_versions: list[str]
    feature_dim: int = Field(gt=0)


class KronosConfig(StrictModel):
    feature_dim: int = Field(default=8, ge=1, le=4_096)
    hidden_dim: int = Field(default=32, ge=8, le=2_048)
    output_dim: int = Field(default=1, ge=1, le=1_024)
    horizons: int = Field(default=3, ge=1, le=128)
    plan_steps: int = Field(default=4, ge=1, le=128)
    action_count: int = Field(default=5, ge=2, le=10_000)
    attention_heads: int = Field(default=4, ge=1, le=32)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_attention_width(self) -> KronosConfig:
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        return self


@dataclass(slots=True)
class KronosOutput:
    forecast_mean: torch.Tensor
    forecast_log_variance: torch.Tensor
    action_logits: torch.Tensor
    hidden_states: torch.Tensor
    pooled_state: torch.Tensor


@dataclass(slots=True)
class KronosLoss:
    total: torch.Tensor
    forecast_nll: torch.Tensor
    action_cross_entropy: torch.Tensor


class SelectiveStateEncoder(nn.Module):
    """Small input-selective diagonal state-space recurrence.

    The recurrence is intentionally explicit so its state transition and time
    decay are testable. It is an experimental SSM, not a claim of Mamba parity.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim * 2)
        self.delta_projection = nn.Linear(input_dim, hidden_dim)
        self.log_decay = nn.Parameter(torch.zeros(hidden_dim))
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, sequence, features]")
        if inputs.shape[0] == 0 or inputs.shape[1] == 0:
            raise ValueError("inputs must contain at least one event per batch")
        if time_deltas.shape != inputs.shape[:2]:
            raise ValueError("time_deltas must have shape [batch, sequence]")
        if torch.any(time_deltas < 0):
            raise ValueError("time deltas cannot be negative")
        if mask is not None and mask.shape != inputs.shape[:2]:
            raise ValueError("mask must have shape [batch, sequence]")
        if not torch.isfinite(inputs).all() or not torch.isfinite(time_deltas).all():
            raise ValueError("inputs and time deltas must be finite")
        if mask is not None and torch.any(mask.to(torch.bool).sum(dim=1) == 0):
            raise ValueError("each stream requires at least one unmasked event")

        state = torch.zeros(
            inputs.shape[0], self.hidden_dim, device=inputs.device, dtype=inputs.dtype
        )
        outputs: list[torch.Tensor] = []
        decay_rate = F.softplus(self.log_decay).to(inputs.dtype) + 1e-4
        for position in range(inputs.shape[1]):
            current = inputs[:, position]
            candidate, gate_logits = self.input_projection(current).chunk(2, dim=-1)
            gate = torch.sigmoid(gate_logits)
            learned_delta = F.softplus(self.delta_projection(current))
            elapsed = time_deltas[:, position, None].to(inputs.dtype)
            decay = torch.exp(-decay_rate[None, :] * learned_delta * elapsed)
            proposed = decay * state + gate * torch.tanh(candidate)
            if mask is not None:
                active = mask[:, position, None].to(torch.bool)
                state = torch.where(active, proposed, state)
            else:
                state = proposed
            outputs.append(self.output_norm(state))
        return torch.stack(outputs, dim=1)


class KronosTemporalPlanner(nn.Module):
    """Trainable event-stream forecaster and receding-horizon action planner."""

    def __init__(self, config: KronosConfig) -> None:
        super().__init__()
        self.config = config
        self.event_projection = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.temporal = SelectiveStateEncoder(config.hidden_dim, config.hidden_dim)
        self.readout_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.forecast_head = nn.Linear(
            config.hidden_dim, config.horizons * config.output_dim * 2
        )
        self.plan_queries = nn.Parameter(
            torch.empty(config.plan_steps, config.hidden_dim).normal_(std=0.02)
        )
        self.plan_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.action_head = nn.Linear(config.hidden_dim, config.action_count)

    def forward(
        self,
        events: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> KronosOutput:
        if events.ndim != 3:
            raise ValueError("events must have shape [batch, sequence, features]")
        if events.shape[0] == 0 or events.shape[1] == 0:
            raise ValueError("events must contain at least one event per batch")
        if events.shape[-1] != self.config.feature_dim:
            raise ValueError("event feature width does not match KronosConfig")
        if time_deltas.shape != events.shape[:2]:
            raise ValueError("time_deltas must match the event batch and sequence")
        if mask is not None and mask.shape != events.shape[:2]:
            raise ValueError("mask must match the event batch and sequence")
        if not torch.isfinite(events).all() or not torch.isfinite(time_deltas).all():
            raise ValueError("events and time deltas must be finite")
        if mask is not None and torch.any(mask.to(torch.bool).sum(dim=1) == 0):
            raise ValueError("each stream requires at least one unmasked event")
        projected = self.event_projection(events)
        hidden = self.temporal(projected, time_deltas, mask)
        key_padding_mask = None if mask is None else ~mask.to(torch.bool)
        summary_query = hidden[:, -1:, :]
        pooled, _ = self.readout_attention(
            summary_query,
            hidden,
            hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled_state = pooled[:, 0]
        forecast = self.forecast_head(pooled_state).reshape(
            events.shape[0], self.config.horizons, self.config.output_dim, 2
        )
        forecast_mean = forecast[..., 0]
        forecast_log_variance = forecast[..., 1].clamp(-10.0, 8.0)

        plan_queries = self.plan_queries[None, :, :].expand(events.shape[0], -1, -1)
        plan_states, _ = self.plan_attention(
            plan_queries,
            hidden,
            hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return KronosOutput(
            forecast_mean=forecast_mean,
            forecast_log_variance=forecast_log_variance,
            action_logits=self.action_head(plan_states),
            hidden_states=hidden,
            pooled_state=pooled_state,
        )


def kronos_loss(
    output: KronosOutput,
    forecast_targets: torch.Tensor,
    action_targets: torch.Tensor,
    *,
    forecast_weight: float = 1.0,
    action_weight: float = 1.0,
) -> KronosLoss:
    if not all(
        math.isfinite(weight) and weight >= 0.0
        for weight in (forecast_weight, action_weight)
    ):
        raise ValueError("Kronos loss weights must be finite and non-negative")
    if forecast_targets.shape != output.forecast_mean.shape:
        raise ValueError("forecast target shape does not match model output")
    if action_targets.shape != output.action_logits.shape[:2]:
        raise ValueError("action target shape does not match model output")
    if action_targets.dtype not in {torch.int32, torch.int64}:
        raise ValueError("action targets must use an integer tensor dtype")
    if torch.any(action_targets < 0) or torch.any(
        action_targets >= output.action_logits.shape[-1]
    ):
        raise ValueError("action targets contain an index outside the action logits")
    if not torch.isfinite(forecast_targets).all():
        raise ValueError("forecast targets must be finite")
    inverse_variance = torch.exp(-output.forecast_log_variance)
    forecast_nll = 0.5 * (
        output.forecast_log_variance
        + (forecast_targets - output.forecast_mean).square() * inverse_variance
    ).mean()
    action_ce = F.cross_entropy(
        output.action_logits.reshape(-1, output.action_logits.shape[-1]),
        action_targets.reshape(-1),
    )
    total = forecast_weight * forecast_nll + action_weight * action_ce
    return KronosLoss(total=total, forecast_nll=forecast_nll, action_cross_entropy=action_ce)


def tensorize_event_stream(
    streams: list[list[TemporalEvent]], *, feature_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TemporalBatch]:
    if not streams or any(not stream for stream in streams):
        raise ValueError("at least one non-empty event stream is required")
    event_ids = [event.event_id for stream in streams for event in stream]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("temporal event IDs must be unique across a batch")
    ordered: list[list[TemporalEvent]] = []
    environment_versions: list[str] = []
    for stream in streams:
        sorted_stream = sorted(stream, key=lambda event: event.occurred_at)
        if stream != sorted_stream:
            raise ValueError("event streams must already be in chronological order")
        versions = {event.environment_version for event in stream}
        if len(versions) != 1:
            raise ValueError("one batch stream cannot mix environment versions")
        if any(len(event.features) != feature_dim for event in stream):
            raise ValueError("event feature width mismatch")
        ordered.append(stream)
        environment_versions.append(next(iter(versions)))

    max_length = max(len(stream) for stream in ordered)
    features = torch.zeros(len(ordered), max_length, feature_dim, dtype=torch.float32)
    time_deltas = torch.zeros(len(ordered), max_length, dtype=torch.float32)
    mask = torch.zeros(len(ordered), max_length, dtype=torch.bool)
    ids: list[list[str]] = []
    for batch_index, stream in enumerate(ordered):
        ids.append([event.event_id for event in stream])
        previous = stream[0].occurred_at
        for position, event in enumerate(stream):
            features[batch_index, position] = torch.tensor(event.features)
            elapsed = max((event.occurred_at - previous).total_seconds(), 0.0)
            time_deltas[batch_index, position] = max(elapsed / 3600.0, 1e-3)
            mask[batch_index, position] = True
            previous = event.occurred_at
    metadata = TemporalBatch(
        event_ids=ids,
        environment_versions=environment_versions,
        feature_dim=feature_dim,
    )
    return features, time_deltas, mask, metadata


class AdaptationMetrics(StrictModel):
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_loss: float
    baseline_loss: float
    forgetting: float = Field(ge=0.0)
    evaluation_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rollback_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deletion_replay_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def finite_metrics(self) -> AdaptationMetrics:
        if not all(
            math.isfinite(value)
            for value in (self.candidate_loss, self.baseline_loss, self.forgetting)
        ):
            raise ValueError("adaptation metrics must be finite")
        return self


AdaptationEvidenceKind = Literal["evaluation", "rollback", "deletion_replay"]


def adaptation_evidence_binding_sha256(
    kind: AdaptationEvidenceKind,
    *,
    candidate_sha256: str,
    candidate_loss: float,
    baseline_loss: float,
    forgetting: float,
) -> str:
    """Identify trusted evidence for one exact checkpoint and metric tuple."""

    if len(candidate_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in candidate_sha256
    ):
        raise ValueError("candidate_sha256 must be a lowercase SHA-256 value")
    if not all(
        math.isfinite(value) for value in (candidate_loss, baseline_loss, forgetting)
    ):
        raise ValueError("adaptation evidence metrics must be finite")
    payload = {
        "schema_version": 1,
        "kind": kind,
        "candidate_sha256": candidate_sha256,
        "candidate_loss": candidate_loss,
        "baseline_loss": baseline_loss,
        "forgetting": forgetting,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AdapterPromotionDecision(StrictModel):
    promoted: bool
    reasons: list[str]
    checkpoint_sha256: str | None = None


class TemporalAdapterManager:
    """Hash-bound, locked staging for experimental Kronos planner checkpoints.

    The historical class name is retained for API compatibility. The current
    reference artifact contains the complete ``KronosTemporalPlanner`` state,
    not a production backbone adapter.
    """

    def __init__(
        self,
        root: Path,
        *,
        maximum_forgetting: float = 0.02,
        trusted_evidence_sha256: frozenset[str] | set[str] = frozenset(),
    ) -> None:
        if not 0.0 <= maximum_forgetting <= 1.0:
            raise ValueError("maximum_forgetting must be in [0, 1]")
        self.root = root.resolve()
        self.maximum_forgetting = maximum_forgetting
        if any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for digest in trusted_evidence_sha256
        ):
            raise ValueError("trusted evidence hashes must be lowercase SHA-256 values")
        self.trusted_evidence_sha256 = frozenset(trusted_evidence_sha256)
        self.root.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        with self._thread_lock, (self.root / ".kronos-adapter.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def stage(self, model: KronosTemporalPlanner, metadata: dict[str, Any]) -> Path:
        if not isinstance(model, KronosTemporalPlanner):
            raise TypeError("only KronosTemporalPlanner instances can be staged")
        safe_metadata = _safe_json_metadata(metadata)
        destination = self.root / "candidate.pt"
        payload = {
            "format_version": 1,
            "model_class": "KronosTemporalPlanner",
            "config": model.config.model_dump(mode="json"),
            "state_dict": model.state_dict(),
            "metadata": safe_metadata,
        }
        with self._exclusive():
            with NamedTemporaryFile(
                dir=self.root, prefix=".candidate.", delete=False
            ) as handle:
                temporary = Path(handle.name)
            try:
                torch.save(payload, temporary)
                _validate_kronos_payload(temporary.read_bytes())
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        return destination

    def promote(
        self, candidate: Path, metrics: AdaptationMetrics
    ) -> AdapterPromotionDecision:
        reasons: list[str] = []
        candidate_path = candidate.absolute()
        expected_candidate = self.root / "candidate.pt"
        if metrics.candidate_loss >= metrics.baseline_loss:
            reasons.append("candidate does not improve the frozen baseline")
        if metrics.forgetting > self.maximum_forgetting:
            reasons.append("candidate exceeds the forgetting bound")
        evidence = (
            ("evaluation", "evaluation report", metrics.evaluation_report_sha256),
            ("rollback", "rollback receipt", metrics.rollback_receipt_sha256),
            (
                "deletion_replay",
                "deletion-replay receipt",
                metrics.deletion_replay_receipt_sha256,
            ),
        )
        for kind, label, digest in evidence:
            expected = adaptation_evidence_binding_sha256(
                cast(AdaptationEvidenceKind, kind),
                candidate_sha256=metrics.candidate_sha256,
                candidate_loss=metrics.candidate_loss,
                baseline_loss=metrics.baseline_loss,
                forgetting=metrics.forgetting,
            )
            if digest != expected:
                reasons.append(f"{label} does not bind the candidate and metric tuple")
            elif digest not in self.trusted_evidence_sha256:
                reasons.append(f"{label} is not trusted by this manager")

        with self._exclusive():
            if (
                candidate_path != expected_candidate
                or candidate.is_symlink()
                or not candidate.is_file()
            ):
                reasons.append("candidate checkpoint is not the manager's staged candidate")
                return AdapterPromotionDecision(promoted=False, reasons=reasons)
            try:
                checkpoint_bytes = candidate.read_bytes()
                _validate_kronos_payload(checkpoint_bytes)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                reasons.append(f"candidate checkpoint is invalid: {type(error).__name__}")
                return AdapterPromotionDecision(promoted=False, reasons=reasons)
            digest = hashlib.sha256(checkpoint_bytes).hexdigest()
            if digest != metrics.candidate_sha256:
                reasons.append("candidate checkpoint is not bound to the supplied metrics")
            if reasons:
                return AdapterPromotionDecision(promoted=False, reasons=reasons)
            promoted = self.root / "promoted.pt"
            with NamedTemporaryFile(
                dir=self.root, prefix=".promoted.", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(checkpoint_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                temporary.replace(promoted)
                if self._sha256(promoted) != digest:
                    raise OSError("promoted checkpoint hash differs from staged bytes")
            finally:
                temporary.unlink(missing_ok=True)
        return AdapterPromotionDecision(
            promoted=True,
            reasons=[],
            checkpoint_sha256=digest,
        )


def load_staged_kronos(
    path: Path, config: KronosConfig, *, expected_sha256: str
) -> tuple[KronosTemporalPlanner, dict[str, Any]]:
    checkpoint_bytes = path.read_bytes()
    observed_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("Kronos checkpoint does not match the expected SHA-256")
    payload = _validate_kronos_payload(checkpoint_bytes)
    stored_config = KronosConfig.model_validate(payload["config"])
    if stored_config != config:
        raise ValueError("Kronos checkpoint configuration does not match the requested model")
    model = KronosTemporalPlanner(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    metadata = cast(dict[str, Any], payload["metadata"])
    return model, metadata


def _validate_kronos_payload(checkpoint_bytes: bytes) -> dict[str, Any]:
    payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    expected_keys = {"format_version", "model_class", "config", "state_dict", "metadata"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("invalid Kronos checkpoint payload")
    if payload["format_version"] != 1 or payload["model_class"] != "KronosTemporalPlanner":
        raise ValueError("unsupported Kronos checkpoint format")
    config = KronosConfig.model_validate(payload["config"])
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError("Kronos checkpoint state_dict must contain only named tensors")
    try:
        KronosTemporalPlanner(config).load_state_dict(state_dict, strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Kronos checkpoint state_dict does not match its configuration") from error
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Kronos checkpoint metadata must be an object")
    _safe_json_metadata(metadata)
    return cast(dict[str, Any], payload)


def _safe_json_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Kronos metadata must contain finite JSON values") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("Kronos metadata must be an object")
    return cast(dict[str, Any], decoded)
