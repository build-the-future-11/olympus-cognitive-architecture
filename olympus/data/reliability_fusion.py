"""Fresh, source-disjoint data for the preregistered Olympus v3 successor."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import torch
from pydantic import Field
from torch import Tensor

from olympus.core.schemas import StrictModel
from olympus.data.olympus_dataset import write_olympus_dataset_manifest

ReliabilityRegime = Literal["mixed", "clean", "corrupted", "ood"]


class ReliabilitySplitSpec(StrictModel):
    name: str = Field(min_length=1)
    examples: int = Field(ge=1)
    seed: int = Field(ge=0, le=2**63 - 1)
    regime: ReliabilityRegime


class ReliabilityDatasetReceipt(StrictModel):
    schema_version: Literal[1] = 1
    dataset_id: Literal["olympus-reliability-fusion"] = "olympus-reliability-fusion"
    version: Literal["3.0.0"] = "3.0.0"
    license: Literal["CC-BY-4.0"] = "CC-BY-4.0"
    generator: Literal["olympus.data.reliability_fusion"] = "olympus.data.reliability_fusion"
    splits: list[ReliabilitySplitSpec]
    bundle_sha256: dict[str, str]
    manifest_sha256: dict[str, str]
    source_id_sha256: dict[str, str]
    source_overlap_count: int = Field(ge=0)
    protected_splits: list[str]
    component_order: list[str]
    reliability_semantics: dict[str, float]
    claim_boundary: str


REGISTERED_RELIABILITY_SPLITS = (
    ReliabilitySplitSpec(name="train", examples=512, seed=20261501, regime="mixed"),
    ReliabilitySplitSpec(name="development", examples=128, seed=20261502, regime="mixed"),
    ReliabilitySplitSpec(name="clean_test", examples=256, seed=20261503, regime="clean"),
    ReliabilitySplitSpec(name="corrupted_test", examples=256, seed=20261504, regime="corrupted"),
    ReliabilitySplitSpec(name="ood_test", examples=256, seed=20261505, regime="ood"),
)
COMPONENT_ORDER = ("hermes", "prometheus", "perseus", "atlas", "kronos")
TEXT_COMPONENTS = (0, 1, 2)
RELIABILITY_VALUES = {"clean": 1.0, "perturbed": 0.25, "missing": 0.0}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _source_ids(spec: ReliabilitySplitSpec) -> list[str]:
    return [
        f"v3-{spec.name}-{_sha256(f'{spec.seed}:{index}'.encode())[:20]}"
        for index in range(spec.examples)
    ]


def _corruption_plan(
    spec: ReliabilitySplitSpec, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    missing = torch.full((spec.examples,), -1, dtype=torch.long)
    perturbed = torch.full((spec.examples,), -1, dtype=torch.long)
    if spec.regime == "mixed":
        selected = torch.rand(spec.examples, generator=generator) < 0.70
        missing[selected] = torch.randint(0, 3, (int(selected.sum()),), generator=generator)
    elif spec.regime in {"corrupted", "ood"}:
        missing = torch.randint(0, 3, (spec.examples,), generator=generator)
        offset = torch.randint(1, 3, (spec.examples,), generator=generator)
        perturbed = (missing + offset) % 3
    return missing, perturbed


def _reliability_targets(missing: Tensor, perturbed: Tensor) -> Tensor:
    targets = torch.ones((missing.shape[0], len(COMPONENT_ORDER)), dtype=torch.float32)
    modality_components = {0: TEXT_COMPONENTS, 1: (3,), 2: (4,)}
    for modality, components in modality_components.items():
        for component in components:
            targets[missing == modality, component] = RELIABILITY_VALUES["missing"]
            targets[perturbed == modality, component] = RELIABILITY_VALUES["perturbed"]
    return targets


def generate_reliability_split(
    spec: ReliabilitySplitSpec,
) -> tuple[dict[str, Tensor], list[str]]:
    """Generate a deterministic split with independently specified reliability labels."""

    generator = torch.Generator(device="cpu").manual_seed(spec.seed)
    count = spec.examples
    labels = torch.randint(0, 4, (count,), generator=generator)
    missing, perturbed = _corruption_plan(spec, generator)
    reliability_targets = _reliability_targets(missing, perturbed)
    noise_scale = 0.12 if spec.regime != "ood" else 0.24

    tokens = torch.randint(8, 31, (count, 8), generator=generator, dtype=torch.long)
    class_tokens = labels + 1
    if spec.regime == "ood":
        class_tokens = torch.tensor([3, 1, 4, 2], dtype=torch.long)[labels]
    tokens[:, 0] = class_tokens
    tokens[:, 1] = 5 + labels.remainder(2)
    tokens[:, 2] = 7 + labels.div(2, rounding_mode="floor")
    attention_mask = torch.ones_like(tokens, dtype=torch.bool)

    image = torch.rand((count, 3, 8, 8), generator=generator) * noise_scale
    for index in range(count):
        label = int(labels[index])
        channel = label % 3
        top = 0 if label < 2 else 4
        left = 0 if label % 2 == 0 else 4
        if spec.regime == "ood":
            channel = (channel + 1) % 3
            left = 4 - left
        image[index, channel, top : top + 4, left : left + 4] += 0.75
    image.clamp_(0.0, 1.0)

    actions = torch.rand((count, 4, 3), generator=generator) * 2.0 - 1.0
    time = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    label_value = labels.float().view(count, 1, 1) / 3.0
    state_noise = torch.randn((count, 4, 5), generator=generator) * noise_scale
    state_template = torch.cat(
        (
            label_value.expand(-1, 4, 1),
            (label_value + 0.08 * time).expand(-1, -1, 1),
            labels.remainder(2).float().view(count, 1, 1).expand(-1, 4, 1),
            labels.div(2, rounding_mode="floor").float().view(count, 1, 1).expand(-1, 4, 1),
            torch.sin(label_value * torch.pi + time * 0.3),
        ),
        dim=-1,
    )
    states = state_template + state_noise

    for modality, value in enumerate((tokens, image, states)):
        selected_missing = missing == modality
        selected_perturbed = perturbed == modality
        if bool(selected_missing.any()):
            value[selected_missing] = 0
        if bool(selected_perturbed.any()):
            amount = int(selected_perturbed.sum())
            if modality == 0:
                tokens[selected_perturbed, :3] = torch.randint(
                    1, 12, (amount, 3), generator=generator
                )
            elif modality == 1:
                image[selected_perturbed] = torch.rand((amount, 3, 8, 8), generator=generator)
            else:
                states[selected_perturbed] = torch.randn((amount, 4, 5), generator=generator)

    temporal_mask = torch.ones((count, 4), dtype=torch.bool)
    allowed_transitions = torch.ones((count, 4), dtype=torch.bool)
    transition_targets = labels.remainder(3)
    forbidden = (transition_targets + 1).remainder(3)
    allowed_transitions[torch.arange(count), forbidden] = False
    allowed_transitions[:, 3] = True
    corruption_count = missing.ge(0).float() + perturbed.ge(0).float()
    risk_score = 0.05 + corruption_count * 0.15
    padded_actions = torch.nn.functional.pad(actions, (0, 2))

    bundle = {
        "tokens": tokens,
        "attention_mask": attention_mask,
        "image": image,
        "states": states,
        "actions": actions,
        "temporal_mask": temporal_mask,
        "allowed_transitions": allowed_transitions,
        "risk_score": risk_score,
        "domain_ids": labels.remainder(3),
        "timestamp": torch.arange(count, dtype=torch.float32) / max(1, count - 1),
        "decision_targets": labels,
        "transition_targets": transition_targets,
        "next_state_targets": states + 0.1 * padded_actions,
        "intrinsic_risk_targets": corruption_count / 2.0,
        "component_reliability_targets": reliability_targets,
    }
    return bundle, _source_ids(spec)


def _serialize_bundle(bundle: dict[str, Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save(bundle, buffer)
    return buffer.getvalue()


def generate_reliability_fusion_dataset(output: Path) -> ReliabilityDatasetReceipt:
    """Materialize v3 splits once and bind every split to content hashes."""

    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite a reliability dataset directory")
    output.mkdir(parents=True, exist_ok=True)

    source_ids: dict[str, list[str]] = {}
    bundle_hashes: dict[str, str] = {}
    for spec in REGISTERED_RELIABILITY_SPLITS:
        bundle, identifiers = generate_reliability_split(spec)
        payload = _serialize_bundle(bundle)
        _atomic_write(output / f"{spec.name}.pt", payload)
        source_ids[spec.name] = identifiers
        bundle_hashes[spec.name] = _sha256(payload)

    all_ids = [identifier for values in source_ids.values() for identifier in values]
    overlap_count = len(all_ids) - len(set(all_ids))
    if overlap_count:
        raise RuntimeError("v3 source identities overlap")

    manifest_hashes: dict[str, str] = {}
    for evaluation_name in ("development", "clean_test", "corrupted_test", "ood_test"):
        manifest = write_olympus_dataset_manifest(
            output / f"manifest_{evaluation_name}.json",
            dataset_id="olympus-reliability-fusion",
            version="3.0.0",
            license_name="CC-BY-4.0",
            train_batch=output / "train.pt",
            evaluation_batch=output / f"{evaluation_name}.pt",
            train_source_ids=source_ids["train"],
            evaluation_source_ids=source_ids[evaluation_name],
        )
        assert manifest.manifest_sha256 is not None
        manifest_hashes[evaluation_name] = manifest.manifest_sha256

    receipt = ReliabilityDatasetReceipt(
        splits=list(REGISTERED_RELIABILITY_SPLITS),
        bundle_sha256=bundle_hashes,
        manifest_sha256=manifest_hashes,
        source_id_sha256={
            name: _sha256(("\n".join(values) + "\n").encode())
            for name, values in source_ids.items()
        },
        source_overlap_count=overlap_count,
        protected_splits=["clean_test", "corrupted_test", "ood_test"],
        component_order=list(COMPONENT_ORDER),
        reliability_semantics=RELIABILITY_VALUES,
        claim_boundary=(
            "Controlled synthetic reliability-mechanism evidence only; no natural-data, "
            "target-scale, deployment, safety, or JEPA claim is licensed."
        ),
    )
    _atomic_write(
        output / "receipt.json",
        (json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode(),
    )
    return receipt
