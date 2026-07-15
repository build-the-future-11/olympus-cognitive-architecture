from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

from pydantic import Field

from olympus.core.schemas import StrictModel


class ModalityKind(StrEnum):
    TEXT = "text"
    CODE = "code"
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"
    TOOL_OBSERVATION = "tool_observation"
    ENVIRONMENT_STATE = "environment_state"
    REPOSITORY_STATE = "repository_state"


class BaseArtifact(StrictModel):
    modality: ModalityKind
    content: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextArtifact(BaseArtifact):
    modality: ModalityKind = ModalityKind.TEXT
    content: str


class CodeArtifact(BaseArtifact):
    modality: ModalityKind = ModalityKind.CODE
    content: str
    language: str = "python"


class DocumentArtifact(BaseArtifact):
    modality: ModalityKind = ModalityKind.DOCUMENT
    content: str
    title: str = "untitled"


class ImageArtifact(BaseArtifact):
    modality: ModalityKind = ModalityKind.IMAGE
    content: dict[str, float | int | str]


class AudioArtifact(BaseArtifact):
    modality: ModalityKind = ModalityKind.AUDIO
    content: dict[str, float | int | str]


class VideoArtifact(BaseArtifact):
    modality: ModalityKind = ModalityKind.VIDEO
    content: dict[str, float | int | str]


class StructuredArtifact(BaseArtifact):
    modality: ModalityKind = ModalityKind.STRUCTURED
    content: dict[str, Any]


class ToolObservationArtifact(BaseArtifact):
    modality: ModalityKind = ModalityKind.TOOL_OBSERVATION
    content: dict[str, Any]


class EnvironmentStateArtifact(BaseArtifact):
    modality: ModalityKind = ModalityKind.ENVIRONMENT_STATE
    content: dict[str, Any]


class RepositoryStateArtifact(BaseArtifact):
    modality: ModalityKind = ModalityKind.REPOSITORY_STATE
    content: dict[str, Any]


Artifact = (
    TextArtifact
    | CodeArtifact
    | DocumentArtifact
    | ImageArtifact
    | AudioArtifact
    | VideoArtifact
    | StructuredArtifact
    | ToolObservationArtifact
    | EnvironmentStateArtifact
    | RepositoryStateArtifact
)


class EncodedArtifact(StrictModel):
    modality: ModalityKind
    vector: list[float]
    summary: str


class ReferenceEncoder:
    """A deterministic local encoder used when no pretrained model is present."""

    def encode(self, artifact: Artifact) -> EncodedArtifact:
        if isinstance(artifact.content, str):
            tokens = artifact.content.split()
            length = len(artifact.content)
            avg_word = sum(len(token) for token in tokens) / max(1, len(tokens))
            vector = [
                float(len(tokens)),
                float(length),
                float(avg_word),
                float(sum(char.isdigit() for char in artifact.content)),
                float(sum(char.isupper() for char in artifact.content)),
            ]
            summary = artifact.content[:120]
        elif isinstance(artifact.content, dict):
            numeric_values = [
                float(value)
                for value in artifact.content.values()
                if isinstance(value, int | float)
            ]
            vector = numeric_values[:8]
            summary = ", ".join(f"{key}={value}" for key, value in artifact.content.items())
        else:
            vector = [0.0]
            summary = str(artifact.content)

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        normalized = [round(value / norm, 6) for value in vector]
        return EncodedArtifact(modality=artifact.modality, vector=normalized, summary=summary)
