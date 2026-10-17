"""Numerical sampling controls on constructed reference-model distributions."""

from random import Random

import pytest

from olympus.foundry.bigram import BigramCheckpoint, CharacterBigramModel


@pytest.mark.parametrize("temperature", [1e-3, 1e-5, 1e-100, 5e-324])
def test_small_positive_temperatures_do_not_zero_every_sampling_weight(
    temperature: float,
) -> None:
    model = CharacterBigramModel.train("ab" * 20)
    assert model.generate("a", max_characters=16, temperature=temperature) == "ba" * 8


@pytest.mark.parametrize("temperature", [1e-3, 1e-5, 1e-100, 5e-324])
def test_uniform_distribution_keeps_ties_random_at_small_temperatures(
    temperature: float,
) -> None:
    model = CharacterBigramModel.train("ab")
    # The unobserved 'b' row is exactly uniform after smoothing. Lowering the
    # temperature must preserve that distribution, including its seeded draws.
    for seed in range(8):
        expected = model.generate("b", max_characters=1, temperature=1.0, seed=seed)
        assert model.generate(
            "b", max_characters=1, temperature=temperature, seed=seed
        ) == expected


@pytest.mark.parametrize("temperature", [0.1, 0.7, 1.0, 2.0])
def test_ordinary_temperature_seeded_outputs_match_retained_sampler(
    temperature: float,
) -> None:
    model = CharacterBigramModel.train("aababbba" * 12)
    random = Random(123)
    current = "a"
    expected = []
    for _ in range(64):
        probabilities = model._probabilities(current)
        weights = [value ** (1.0 / temperature) for value in probabilities]
        index = random.choices(range(len(weights)), weights=weights, k=1)[0]
        current = model.checkpoint.alphabet[index]
        expected.append(current)
    assert model.generate(
        "a", max_characters=64, temperature=temperature, seed=123
    ) == "".join(expected)


def test_zero_temperature_retains_greedy_path() -> None:
    model = CharacterBigramModel.train("ab" * 20)
    assert model.generate("a", max_characters=16, temperature=0.0) == "ba" * 8


@pytest.mark.parametrize("temperature", [0.7, 1e-5, 5e-324])
def test_underflowed_zero_probability_remains_a_valid_zero_weight(temperature: float) -> None:
    model = CharacterBigramModel.train("ab" * 20, smoothing=5e-324)
    assert 0.0 in model._probabilities("a")
    assert model.generate("a", max_characters=16, temperature=temperature) == "ba" * 8


@pytest.mark.parametrize("temperature", [1e-100, 5e-324])
def test_tiny_temperature_preserves_distinct_near_maximum_probabilities(
    temperature: float,
) -> None:
    alphabet = list("abcdefghijklmnop")
    row = [2**52 + 1] + [2**52] * 15
    model = CharacterBigramModel(
        BigramCheckpoint(
            alphabet=alphabet,
            counts=[row.copy() for _ in alphabet],
            smoothing=0.25,
            seed=7,
            training_characters=2,
        )
    )
    probabilities = model._probabilities("a")
    assert probabilities[0] > max(probabilities[1:])
    assert model.generate("a", max_characters=8, temperature=temperature) == "a" * 8
