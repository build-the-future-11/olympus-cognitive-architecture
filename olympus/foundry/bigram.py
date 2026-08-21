from __future__ import annotations

import json
import math
from dataclasses import dataclass
from random import Random
from typing import Any, Self

from pydantic import Field

from olympus.core.schemas import StrictModel


class BigramCheckpoint(StrictModel):
    schema_version: int = 1
    model_type: str = "character-bigram"
    alphabet: list[str] = Field(min_length=2)
    counts: list[list[int]]
    smoothing: float = Field(gt=0.0, le=10.0)
    seed: int = Field(ge=0, le=2**32 - 1)
    training_characters: int = Field(gt=1)
    training_provenance: dict[str, Any] = Field(default_factory=dict)

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> Self:
        return cls.model_validate_json(payload)


@dataclass(frozen=True, slots=True)
class LanguageModelMetrics:
    negative_log_likelihood: float
    perplexity: float
    top1_accuracy: float
    evaluated_transitions: int

    def as_dict(self) -> dict[str, float]:
        return {
            "negative_log_likelihood": self.negative_log_likelihood,
            "perplexity": self.perplexity,
            "top1_accuracy": self.top1_accuracy,
            "evaluated_transitions": float(self.evaluated_transitions),
        }


class CharacterBigramModel:
    """A small real language model used to verify the Foundry lifecycle cheaply."""

    def __init__(self, checkpoint: BigramCheckpoint) -> None:
        size = len(checkpoint.alphabet)
        if len(checkpoint.counts) != size or any(len(row) != size for row in checkpoint.counts):
            raise ValueError("checkpoint count matrix does not match alphabet")
        if any(value < 0 for row in checkpoint.counts for value in row):
            raise ValueError("checkpoint counts cannot be negative")
        self.checkpoint = checkpoint
        self._index = {character: index for index, character in enumerate(checkpoint.alphabet)}

    @classmethod
    def train(cls, text: str, *, smoothing: float = 0.25, seed: int = 7) -> Self:
        if len(text) < 2:
            raise ValueError("training text must contain at least two characters")
        if not math.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be a finite positive value")
        alphabet = sorted(set(text))
        if len(alphabet) < 2:
            raise ValueError("training text must contain at least two distinct characters")
        index = {character: position for position, character in enumerate(alphabet)}
        counts = [[0 for _ in alphabet] for _ in alphabet]
        for current, following in zip(text, text[1:], strict=False):
            counts[index[current]][index[following]] += 1
        return cls(
            BigramCheckpoint(
                alphabet=alphabet,
                counts=counts,
                smoothing=smoothing,
                seed=seed,
                training_characters=len(text),
            )
        )

    def _probabilities(self, character: str) -> list[float]:
        row_index = self._index.get(character)
        size = len(self.checkpoint.alphabet)
        if row_index is None:
            return [1.0 / size] * size
        row = self.checkpoint.counts[row_index]
        denominator = sum(row) + self.checkpoint.smoothing * size
        return [(value + self.checkpoint.smoothing) / denominator for value in row]

    def evaluate(self, text: str) -> LanguageModelMetrics:
        transitions = list(zip(text, text[1:], strict=False))
        if not transitions:
            raise ValueError("evaluation text must contain at least two characters")
        negative_log_likelihood = 0.0
        correct = 0
        for current, expected in transitions:
            probabilities = self._probabilities(current)
            expected_index = self._index.get(expected)
            probability = (
                probabilities[expected_index]
                if expected_index is not None
                else 1.0 / len(self.checkpoint.alphabet)
            )
            negative_log_likelihood -= math.log(max(probability, 1e-12))
            predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
            correct += int(expected_index == predicted)
        average_nll = negative_log_likelihood / len(transitions)
        return LanguageModelMetrics(
            negative_log_likelihood=average_nll,
            perplexity=math.exp(average_nll),
            top1_accuracy=correct / len(transitions),
            evaluated_transitions=len(transitions),
        )

    def generate(
        self,
        prompt: str,
        *,
        max_characters: int = 160,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> str:
        if max_characters <= 0 or max_characters > 4_096:
            raise ValueError("max_characters must be between 1 and 4096")
        if not math.isfinite(temperature) or temperature < 0 or temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        current = prompt[-1] if prompt else "\n"
        if current not in self._index:
            current = "\n" if "\n" in self._index else self.checkpoint.alphabet[0]
        random = Random(self.checkpoint.seed if seed is None else seed)
        generated: list[str] = []
        for _ in range(max_characters):
            probabilities = self._probabilities(current)
            if temperature == 0:
                next_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            else:
                weights = [probability ** (1.0 / temperature) for probability in probabilities]
                next_index = random.choices(range(len(weights)), weights=weights, k=1)[0]
            current = self.checkpoint.alphabet[next_index]
            generated.append(current)
        return "".join(generated)


def uniform_baseline_metrics(text: str, alphabet_size: int) -> LanguageModelMetrics:
    if len(text) < 2:
        raise ValueError("evaluation text must contain at least two characters")
    if alphabet_size < 2:
        raise ValueError("alphabet_size must be at least two")
    transitions = len(text) - 1
    nll = math.log(alphabet_size)
    return LanguageModelMetrics(
        negative_log_likelihood=nll,
        perplexity=float(alphabet_size),
        top1_accuracy=1.0 / alphabet_size,
        evaluated_transitions=transitions,
    )
