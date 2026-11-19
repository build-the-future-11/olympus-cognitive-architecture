from __future__ import annotations

import pytest

from olympus.foundry.bigram import BigramCheckpoint, CharacterBigramModel


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


@pytest.mark.parametrize("text", ["abba" * 8, "\u03b1\u03b2\u03b3\n" * 8])
def test_valid_character_checkpoint_roundtrip_preserves_generation(text: str) -> None:
    model = CharacterBigramModel.train(text)
    payload = model.checkpoint.canonical_bytes()
    restored = CharacterBigramModel(BigramCheckpoint.from_bytes(payload))
    assert restored.checkpoint.canonical_bytes() == payload
    expected = model.generate("unknown", max_characters=40, temperature=0.7, seed=9)
    assert restored.generate("unknown", max_characters=40, temperature=0.7, seed=9) == expected
    assert len(expected) == 40
    assert set(expected) <= set(model.checkpoint.alphabet)
