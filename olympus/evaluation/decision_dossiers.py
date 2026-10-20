"""Source-bound, per-example audits for the composed Olympus reference system.

The aggregate evaluator deliberately retains only compact reductions.  This
module exports an inspectable companion artifact for a declared split without
serializing raw tokens, images, or hidden states.  A dossier reports what the
model exposed, what the host allowed, and how leave-one-family-out logits differ;
it does not claim that a family gate or counterfactual is a causal explanation.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from pydantic import Field, model_validator
from torch import Tensor

from olympus.core.schemas import StrictModel
from olympus.evaluation.olympus_system import olympus_batch_inputs_targets
from olympus.models.olympus_system import OlympusReferenceSystem
from olympus.models.runtime import _evaluation_mode


class OlympusDossierBinding(StrictModel):
    """Immutable identities needed to interpret a decision-dossier bundle."""

    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")


class OlympusCounterfactualDecision(StrictModel):
    """Raw decision behavior after removing one projected component."""

    raw_predicted_class: int = Field(ge=0)
    raw_predicted_probability: float = Field(ge=0.0, le=1.0)
    kl_from_full: float = Field(ge=0.0)
    changes_raw_prediction: bool


class OlympusDecisionDossier(StrictModel):
    """A compact audit record for one source-bound example."""

    source_id: str = Field(min_length=1)
    decision_available: bool
    raw_predicted_class: int = Field(ge=0)
    raw_predicted_probability: float = Field(ge=0.0, le=1.0)
    exposed_predicted_class: int | None = Field(default=None, ge=0)
    exposed_predicted_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    allowed_transitions: list[bool] = Field(min_length=1)
    selected_transition: int = Field(ge=0)
    selected_transition_probability: float = Field(ge=0.0, le=1.0)
    forced_stop: bool
    host_risk: float = Field(ge=0.0, le=1.0)
    intrinsic_risk: float = Field(ge=0.0, le=1.0)
    effective_risk: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(gt=0.0)
    family_disagreement: float = Field(ge=0.0, le=1.0)
    component_gates: dict[str, float] = Field(min_length=1)
    counterfactuals: dict[str, OlympusCounterfactualDecision] = Field(min_length=1)

    @model_validator(mode="after")
    def internally_consistent(self) -> OlympusDecisionDossier:
        if set(self.component_gates) != set(self.counterfactuals):
            raise ValueError("component gates and counterfactuals must name the same components")
        if any(value < 0.0 or value > 1.0 for value in self.component_gates.values()):
            raise ValueError("component gates must fall in [0, 1]")
        if not abs(sum(self.component_gates.values()) - 1.0) <= 1e-5:
            raise ValueError("component gates must sum to one")
        if self.forced_stop == self.decision_available:
            raise ValueError("decision availability must be the inverse of forced STOP")
        if self.forced_stop and (
            self.exposed_predicted_class is not None
            or self.exposed_predicted_probability is not None
        ):
            raise ValueError("forced STOP must not expose a decision")
        if not self.forced_stop and (
            self.exposed_predicted_class != self.raw_predicted_class
            or self.exposed_predicted_probability != self.raw_predicted_probability
        ):
            raise ValueError("available decisions must exactly expose the raw prediction")
        if self.selected_transition >= len(self.allowed_transitions):
            raise ValueError("selected transition falls outside the host transition mask")
        if not self.allowed_transitions[self.selected_transition]:
            raise ValueError("selected transition is forbidden by the host transition mask")
        if self.effective_risk + 1e-7 < self.host_risk:
            raise ValueError("effective risk must not fall below the host risk")
        return self


class OlympusDecisionDossierBundle(StrictModel):
    schema_version: int = Field(default=1, frozen=True)
    binding: OlympusDossierBinding
    dossiers: list[OlympusDecisionDossier] = Field(min_length=1)
    bundle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def unique_sources(self) -> OlympusDecisionDossierBundle:
        sources = [dossier.source_id for dossier in self.dossiers]
        if len(sources) != len(set(sources)):
            raise ValueError("decision dossiers must have unique source identities")
        if self.bundle_sha256 is not None:
            expected = hashlib.sha256(self.canonical_bytes(include_hash=False)).hexdigest()
            if self.bundle_sha256 != expected:
                raise ValueError("decision-dossier bundle digest does not match its contents")
        return self

    def canonical_bytes(self, *, include_hash: bool = True) -> bytes:
        payload = self.model_dump(mode="json")
        if not include_hash:
            payload["bundle_sha256"] = None
        return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def read_olympus_decision_dossier_bundle(
    path: Path,
    *,
    expected_binding: OlympusDossierBinding,
    expected_source_ids: Sequence[str],
    expected_file_sha256: str | None = None,
) -> OlympusDecisionDossierBundle:
    """Read one immutable bundle and verify its external and semantic bindings."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("decision-dossier artifact must be a regular non-symlink file")
    payload = path.read_bytes()
    actual_file_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_file_sha256 is not None and actual_file_sha256 != expected_file_sha256:
        raise ValueError("decision-dossier file digest does not match the report")
    bundle = OlympusDecisionDossierBundle.model_validate_json(payload)
    if bundle.binding != expected_binding:
        raise ValueError("decision-dossier binding does not match the expected run")
    if [dossier.source_id for dossier in bundle.dossiers] != list(expected_source_ids):
        raise ValueError("decision-dossier sources do not match the expected split")
    return bundle


def verify_olympus_decision_dossier_replay(
    model: OlympusReferenceSystem,
    batch: Mapping[str, Tensor],
    extensions: Sequence[str],
    source_ids: Sequence[str],
    *,
    binding: OlympusDossierBinding,
    observed: OlympusDecisionDossierBundle,
) -> None:
    """Recompute every attribution from the declared checkpoint and split.

    This deliberately requires byte-identical canonical values.  A verifier must
    replay on the run's declared device/runtime rather than silently accepting a
    numerically similar bundle produced by different software or hardware.
    """

    replayed = OlympusDecisionDossierBundle(
        binding=binding,
        dossiers=build_olympus_decision_dossiers(
            model,
            batch,
            extensions,
            source_ids,
        ),
    )
    replayed.bundle_sha256 = hashlib.sha256(
        replayed.canonical_bytes(include_hash=False)
    ).hexdigest()
    if observed.canonical_bytes() != replayed.canonical_bytes():
        raise ValueError(
            "decision-dossier replay does not match the selected checkpoint and split"
        )


def build_olympus_decision_dossiers(
    model: OlympusReferenceSystem,
    batch: Mapping[str, Tensor],
    extensions: Sequence[str],
    source_ids: Sequence[str],
) -> list[OlympusDecisionDossier]:
    """Produce CPU-native dossiers without retaining model activations.

    ``source_ids`` must be the ordered source identities from the exact split
    manifest.  This function intentionally does not accept arbitrary labels or
    raw provenance text, preventing an inspection artifact from silently being
    disconnected from the evaluated tensor bundle.
    """

    count = batch["tokens"].shape[0]
    if (
        len(source_ids) != count
        or len(set(source_ids)) != count
        or any(not value for value in source_ids)
    ):
        raise ValueError("source identities must be unique and align with the tensor batch")
    parameter = next(model.parameters())
    device = parameter.device
    selected = {
        name: tensor.to(
            device=device,
            dtype=parameter.dtype if tensor.is_floating_point() else tensor.dtype,
        )
        for name, tensor in batch.items()
    }
    inputs, _ = olympus_batch_inputs_targets(selected, extensions)
    with _evaluation_mode(model), torch.inference_mode():
        output = model(**inputs)

    full_probabilities = output.raw_decision_logits.softmax(dim=-1).detach().cpu()
    counterfactual_probabilities = (
        output.counterfactual_decision_logits.softmax(dim=-1).detach().cpu()
    )
    counterfactual_log_probabilities = (
        output.counterfactual_decision_logits.log_softmax(dim=-1).detach().cpu()
    )
    full_cpu = full_probabilities
    counterfactual_kl = torch.nn.functional.kl_div(
        counterfactual_log_probabilities,
        full_cpu.unsqueeze(1).expand_as(counterfactual_log_probabilities),
        reduction="none",
    ).sum(dim=-1)
    gates = output.family_gates.detach().cpu()
    decision_available = output.decision_available.detach().cpu()
    forced_stop = output.aion.forced_stop.detach().cpu()
    selected_transition = output.aion.selected_stage.detach().cpu()
    transition_probabilities = output.aion.transition_probabilities.detach().cpu()
    allowed_transitions = batch["allowed_transitions"].detach().cpu()
    host_risk = batch["risk_score"].detach().cpu()
    intrinsic_risk = output.intrinsic_risk.detach().cpu()
    effective_risk = output.effective_risk.detach().cpu()
    uncertainty = output.uncertainty.detach().cpu()
    disagreement = output.family_disagreement.detach().cpu()

    dossiers: list[OlympusDecisionDossier] = []
    for index, source_id in enumerate(source_ids):
        raw_class = int(full_probabilities[index].argmax().item())
        raw_probability = float(full_probabilities[index, raw_class].item())
        transition = int(selected_transition[index].item())
        available = bool(decision_available[index].item())
        component_gates = {
            name: float(gates[index, component_index].item())
            for component_index, name in enumerate(output.component_names)
        }
        counterfactuals = {}
        for component_index, name in enumerate(output.component_names):
            probabilities = counterfactual_probabilities[index, component_index]
            counterfactual_class = int(probabilities.argmax().item())
            counterfactuals[name] = OlympusCounterfactualDecision(
                raw_predicted_class=counterfactual_class,
                raw_predicted_probability=float(probabilities[counterfactual_class].item()),
                kl_from_full=float(counterfactual_kl[index, component_index].item()),
                changes_raw_prediction=counterfactual_class != raw_class,
            )
        dossiers.append(
            OlympusDecisionDossier(
                source_id=str(source_id),
                decision_available=available,
                raw_predicted_class=raw_class,
                raw_predicted_probability=raw_probability,
                exposed_predicted_class=raw_class if available else None,
                exposed_predicted_probability=raw_probability if available else None,
                allowed_transitions=[bool(value) for value in allowed_transitions[index].tolist()],
                selected_transition=transition,
                selected_transition_probability=float(
                    transition_probabilities[index, transition].item()
                ),
                forced_stop=bool(forced_stop[index].item()),
                host_risk=float(host_risk[index].item()),
                intrinsic_risk=float(intrinsic_risk[index].item()),
                effective_risk=float(effective_risk[index].item()),
                uncertainty=float(uncertainty[index].item()),
                family_disagreement=float(disagreement[index].item()),
                component_gates=component_gates,
                counterfactuals=counterfactuals,
            )
        )
    return dossiers


def write_olympus_decision_dossiers(
    path: Path,
    *,
    binding: OlympusDossierBinding,
    dossiers: Sequence[OlympusDecisionDossier],
) -> OlympusDecisionDossierBundle:
    """Atomically export a hash-bound dossier bundle without overwriting one."""

    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite an existing decision-dossier bundle")
    bundle = OlympusDecisionDossierBundle(binding=binding, dossiers=list(dossiers))
    bundle.bundle_sha256 = hashlib.sha256(bundle.canonical_bytes(include_hash=False)).hexdigest()
    payload = bundle.canonical_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)
    return bundle
