"""Held-out diagnostics for the composed Olympus reference system."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from pydantic import Field
from torch import Tensor, nn

from olympus.core.schemas import StrictModel
from olympus.models.olympus_system import (
    OlympusObjectiveOutput,
    OlympusReferenceSystem,
    OlympusSystemLossConfig,
    OlympusSystemOutput,
    olympus_system_loss,
)
from olympus.models.reference_families import AionReferenceOutput
from olympus.models.runtime import RuntimeInput, _evaluation_mode

_INPUT_NAMES = frozenset(
    {
        "tokens",
        "image",
        "states",
        "actions",
        "allowed_transitions",
        "risk_score",
        "attention_mask",
        "domain_ids",
        "temporal_mask",
        "timestamp",
    }
)
_REQUIRED_TARGETS = frozenset({"decision_targets", "transition_targets", "next_state_targets"})


class RiskCoveragePoint(StrictModel):
    coverage: float = Field(gt=0.0, le=1.0)
    retained_examples: int = Field(gt=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    maximum_effective_risk: float = Field(ge=0.0, le=1.0)
    # Historical schema-1 reports omitted this denominator; do not invent it on load.
    population_coverage: float | None = Field(default=None, gt=0.0, le=1.0)


class OlympusRoutingDiagnostics(StrictModel):
    mean_gates: dict[str, float]
    normalized_gate_entropy: float = Field(ge=0.0, le=1.0)
    effective_components: float = Field(ge=1.0)
    counterfactual_kl: dict[str, float]
    counterfactual_flip_rate: dict[str, float]


class OlympusRiskDiagnostics(StrictModel):
    mean_intrinsic_risk: float = Field(ge=0.0, le=1.0)
    mean_effective_risk: float = Field(ge=0.0, le=1.0)
    intrinsic_brier: float | None = Field(default=None, ge=0.0)
    intrinsic_ece: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_coverage: list[RiskCoveragePoint]


class OlympusSystemEvaluation(StrictModel):
    examples: int = Field(gt=0)
    decision_available_examples: int = Field(ge=0)
    forced_stop_examples: int = Field(ge=0)
    total_loss: float
    decision_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    transition_accuracy: float = Field(ge=0.0, le=1.0)
    next_state_mse: float = Field(ge=0.0)
    mean_uncertainty: float = Field(gt=0.0)
    routing: OlympusRoutingDiagnostics
    risk: OlympusRiskDiagnostics


def olympus_batch_inputs_targets(
    batch: Mapping[str, Tensor], extensions: Sequence[str]
) -> tuple[dict[str, RuntimeInput], dict[str, Tensor]]:
    """Split one verified tensor bundle into the model and objective contracts."""

    missing_targets = sorted(_REQUIRED_TARGETS - set(batch))
    if missing_targets:
        raise ValueError(f"Olympus batch is missing targets: {missing_targets}")
    inputs: dict[str, RuntimeInput] = {
        name: value for name, value in batch.items() if name in _INPUT_NAMES
    }
    inputs["extension_features"] = {name: batch[f"extension__{name}"] for name in extensions}
    targets = {name: batch[name] for name in _REQUIRED_TARGETS}
    if "intrinsic_risk_targets" in batch:
        targets["intrinsic_risk_targets"] = batch["intrinsic_risk_targets"]
    return inputs, targets


def _expected_calibration_error(predictions: Tensor, targets: Tensor, *, bins: int = 10) -> float:
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=predictions.device)
    error = predictions.new_zeros(())
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        selected = (predictions >= lower) & (
            predictions <= upper if index == bins - 1 else predictions < upper
        )
        if bool(selected.any()):
            gap = (predictions[selected].mean() - targets[selected].mean()).abs()
            error = error + selected.float().mean() * gap
    return float(error.item())


def _risk_coverage(
    output: OlympusSystemOutput | OlympusObjectiveOutput, decision_targets: Tensor
) -> list[RiskCoveragePoint]:
    active = output.decision_available
    if not bool(active.any()):
        return []
    correct = output.raw_decision_logits.argmax(dim=-1).eq(decision_targets)[active]
    risks = output.effective_risk[active]
    order = risks.argsort(stable=True)
    ordered_correct = correct[order].float()
    ordered_risk = risks[order]
    result: list[RiskCoveragePoint] = []
    for coverage in (0.25, 0.5, 0.75, 1.0):
        retained = max(1, math.ceil(coverage * ordered_correct.numel()))
        result.append(
            RiskCoveragePoint(
                coverage=retained / ordered_correct.numel(),
                retained_examples=retained,
                accuracy=float(ordered_correct[:retained].mean().item()),
                maximum_effective_risk=float(ordered_risk[:retained].max().item()),
                population_coverage=retained / active.numel(),
            )
        )
    return result


def evaluate_olympus_system(
    model: OlympusReferenceSystem,
    batch: Mapping[str, Tensor],
    extensions: Sequence[str],
    *,
    loss_config: OlympusSystemLossConfig | None = None,
    batch_size: int | None = None,
) -> OlympusSystemEvaluation:
    """Batched forward passes, followed by exact global diagnostic reductions.

    This does not average minibatch losses: active-example and valid-timestep
    denominators differ, and gate balance depends on the global mean gates.
    Only compact detached predictions are retained on CPU. Caller modes are restored.
    ``coverage`` is conditional on decision availability; ``population_coverage``
    includes examples forced to STOP in its denominator.
    """

    count = batch["tokens"].shape[0]
    if batch_size is not None and batch_size < 1:
        raise ValueError("evaluation batch_size must be positive")
    if count < 1:
        raise ValueError("evaluation requires at least one example")
    width = count if batch_size is None else batch_size
    parameter = next(model.parameters())
    device = parameter.device
    chunks: list[OlympusObjectiveOutput] = []
    with _evaluation_mode(model), torch.inference_mode():
        for start in range(0, count, width):
            selected = {
                name: tensor[start : start + width].to(
                    device=device,
                    dtype=parameter.dtype if tensor.is_floating_point() else tensor.dtype,
                )
                for name, tensor in batch.items()
            }
            chunk_inputs, _ = olympus_batch_inputs_targets(selected, extensions)
            prediction = model(**chunk_inputs)
            chunks.append(_compact(prediction))
            del prediction, selected, chunk_inputs
    output = _join(chunks)
    # Global diagnostics stay in reference precision and on the CPU.
    batch = {
        name: tensor.detach().to(
            device="cpu", dtype=torch.float32 if tensor.is_floating_point() else tensor.dtype
        )
        for name, tensor in batch.items()
    }
    inputs, targets = olympus_batch_inputs_targets(batch, extensions)
    with torch.inference_mode():
        temporal_mask = inputs.get("temporal_mask")
        assert temporal_mask is None or isinstance(temporal_mask, Tensor)
        loss = olympus_system_loss(
            output,
            decision_targets=targets["decision_targets"],
            transition_targets=targets["transition_targets"],
            next_state_targets=targets["next_state_targets"],
            intrinsic_risk_targets=targets.get("intrinsic_risk_targets"),
            temporal_mask=temporal_mask,
            **(loss_config or OlympusSystemLossConfig()).model_dump(),
        )
    finite_outputs = (
        output.raw_decision_logits,
        output.aion.transition_logits.masked_fill(output.aion.transition_logits.isneginf(), 0.0),
        output.next_state,
        output.uncertainty,
        output.family_gates,
        output.counterfactual_decision_logits,
        output.intrinsic_risk,
        output.effective_risk,
        loss.total.unsqueeze(0),
    )
    if not all(bool(torch.isfinite(value).all()) for value in finite_outputs):
        raise FloatingPointError("Olympus evaluation produced non-finite diagnostics")

    decision_targets = targets["decision_targets"].to(
        device=output.raw_decision_logits.device, dtype=torch.long
    )
    transition_targets = targets["transition_targets"].to(
        device=output.aion.transition_logits.device, dtype=torch.long
    )
    active = output.decision_available
    decision_accuracy = None
    if bool(active.any()):
        decision_accuracy = float(
            output.raw_decision_logits.argmax(dim=-1)[active]
            .eq(decision_targets[active])
            .float()
            .mean()
            .item()
        )

    temporal_error = (output.next_state - targets["next_state_targets"]).square().mean(dim=-1)
    temporal_mask = inputs.get("temporal_mask")
    if isinstance(temporal_mask, Tensor):
        next_state_mse = float(temporal_error[temporal_mask].mean().item())
    else:
        next_state_mse = float(temporal_error.mean().item())

    gates = output.family_gates.clamp_min(torch.finfo(output.family_gates.dtype).tiny)
    entropy = -(gates * gates.log()).sum(dim=-1)
    normalized_entropy = float((entropy / math.log(gates.shape[1])).mean().item())
    full_probabilities = output.raw_decision_logits.softmax(dim=-1)
    counterfactual_log_probabilities = output.counterfactual_decision_logits.log_softmax(dim=-1)
    counterfactual_kl_tensor = nn.functional.kl_div(
        counterfactual_log_probabilities,
        full_probabilities.unsqueeze(1).expand_as(counterfactual_log_probabilities),
        reduction="none",
    ).sum(dim=-1)
    full_classes = output.raw_decision_logits.argmax(dim=-1).unsqueeze(1)
    counterfactual_flips = output.counterfactual_decision_logits.argmax(dim=-1).ne(full_classes)

    risk_targets = targets.get("intrinsic_risk_targets")
    brier = None
    ece = None
    if risk_targets is not None:
        risk_targets = risk_targets.to(
            device=output.intrinsic_risk.device, dtype=output.intrinsic_risk.dtype
        )
        brier = float((output.intrinsic_risk - risk_targets).square().mean().item())
        ece = _expected_calibration_error(output.intrinsic_risk, risk_targets)

    return OlympusSystemEvaluation(
        examples=output.raw_decision_logits.shape[0],
        decision_available_examples=int(active.sum().item()),
        forced_stop_examples=int(output.aion.forced_stop.sum().item()),
        total_loss=float(loss.total.item()),
        decision_accuracy=decision_accuracy,
        transition_accuracy=float(
            output.aion.selected_stage.eq(transition_targets).float().mean().item()
        ),
        next_state_mse=next_state_mse,
        mean_uncertainty=float(output.uncertainty.mean().item()),
        routing=OlympusRoutingDiagnostics(
            mean_gates={
                name: float(value)
                for name, value in zip(
                    output.component_names,
                    output.family_gates.mean(dim=0).tolist(),
                    strict=True,
                )
            },
            normalized_gate_entropy=normalized_entropy,
            effective_components=float(entropy.exp().mean().item()),
            counterfactual_kl={
                name: float(value)
                for name, value in zip(
                    output.component_names,
                    counterfactual_kl_tensor.mean(dim=0).tolist(),
                    strict=True,
                )
            },
            counterfactual_flip_rate={
                name: float(value)
                for name, value in zip(
                    output.component_names,
                    counterfactual_flips.float().mean(dim=0).tolist(),
                    strict=True,
                )
            },
        ),
        risk=OlympusRiskDiagnostics(
            mean_intrinsic_risk=float(output.intrinsic_risk.mean().item()),
            mean_effective_risk=float(output.effective_risk.mean().item()),
            intrinsic_brier=brier,
            intrinsic_ece=ece,
            risk_coverage=_risk_coverage(output, decision_targets),
        ),
    )


def _compact(output: OlympusSystemOutput) -> OlympusObjectiveOutput:
    def cpu(tensor: Tensor) -> Tensor:
        value = tensor.detach().cpu()
        return value.float() if value.is_floating_point() else value

    return OlympusObjectiveOutput(
        aion=AionReferenceOutput(
            transition_logits=cpu(output.aion.transition_logits),
            transition_probabilities=cpu(output.aion.transition_probabilities),
            selected_stage=cpu(output.aion.selected_stage),
            forced_stop=cpu(output.aion.forced_stop),
        ),
        next_state=cpu(output.kronos.next_state),
        component_names=output.component_names,
        raw_decision_logits=cpu(output.raw_decision_logits),
        decision_available=cpu(output.decision_available),
        uncertainty=cpu(output.uncertainty),
        family_gates=cpu(output.family_gates),
        counterfactual_decision_logits=cpu(output.counterfactual_decision_logits),
        intrinsic_risk=cpu(output.intrinsic_risk),
        effective_risk=cpu(output.effective_risk),
    )


def _join(chunks: list[OlympusObjectiveOutput]) -> OlympusObjectiveOutput:
    def cat(name: str) -> Tensor:
        return torch.cat([getattr(chunk, name) for chunk in chunks], dim=0)

    def aion_cat(name: str) -> Tensor:
        return torch.cat([getattr(chunk.aion, name) for chunk in chunks], dim=0)

    return OlympusObjectiveOutput(
        aion=AionReferenceOutput(
            transition_logits=aion_cat("transition_logits"),
            transition_probabilities=aion_cat("transition_probabilities"),
            selected_stage=aion_cat("selected_stage"),
            forced_stop=aion_cat("forced_stop"),
        ),
        next_state=cat("next_state"),
        component_names=chunks[0].component_names,
        raw_decision_logits=cat("raw_decision_logits"),
        decision_available=cat("decision_available"),
        uncertainty=cat("uncertainty"),
        family_gates=cat("family_gates"),
        counterfactual_decision_logits=cat("counterfactual_decision_logits"),
        intrinsic_risk=cat("intrinsic_risk"),
        effective_risk=cat("effective_risk"),
    )
