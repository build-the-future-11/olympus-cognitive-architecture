from __future__ import annotations

import math

import pytest

from olympus.foundry.bigram import BigramCheckpoint, CharacterBigramModel


@pytest.mark.parametrize("temperature", [1e-4, 1e-10, math.ulp(0.0)])
def test_low_positive_temperature_remains_sampleable(temperature: float) -> None:
    model = CharacterBigramModel.train("abcd" * 8)
    expected = model.generate("a", max_characters=32, temperature=0)
    assert model.generate("a", max_characters=32, temperature=temperature) == expected


@pytest.mark.parametrize("temperature", [0.0, math.ulp(0.0), 0.7, 2.0])
def test_generation_is_repeatable_and_stays_in_the_character_alphabet(temperature: float) -> None:
    model = CharacterBigramModel.train("\u03b1\u03b2\u03b3\n" * 10)
    first = model.generate("unknown", max_characters=40, temperature=temperature, seed=9)
    second = model.generate("unknown", max_characters=40, temperature=temperature, seed=9)
    assert first == second
    assert len(first) == 40
    assert set(first) <= set(model.checkpoint.alphabet)


def test_low_temperature_preserves_random_choice_between_tied_maxima() -> None:
    checkpoint = BigramCheckpoint(
        alphabet=["a", "b"],
        counts=[[0, 0], [0, 0]],
        smoothing=0.25,
        seed=7,
        training_characters=2,
    )
    model = CharacterBigramModel(checkpoint)
    generated = model.generate("a", max_characters=64, temperature=math.ulp(0.0))
    assert set(generated) == {"a", "b"}
    assert generated == model.generate("a", max_characters=64, temperature=1.0)


@pytest.mark.parametrize("alphabet", [["a", "a"], ["a", "bc"], ["", "a"]])
def test_checkpoint_requires_unique_single_character_symbols(alphabet: list[str]) -> None:
    checkpoint = BigramCheckpoint(
        alphabet=alphabet,
        counts=[[1, 1], [1, 1]],
        smoothing=0.25,
        seed=7,
        training_characters=5,
    )
    with pytest.raises(ValueError, match="unique single characters"):
        CharacterBigramModel(checkpoint)
