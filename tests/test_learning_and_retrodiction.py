from olympus.core.prediction import train_jepa
from olympus.core.retrodiction import RetrodictiveIdentityNetwork, code_bug_generators
from olympus.core.transform import train_learned_transform
from olympus.training.synthetic import transform_signals


def test_transform_training_produces_low_loss() -> None:
    metrics = train_learned_transform(transform_signals(), epochs=80)
    assert metrics.reconstruction_loss < 0.4


def test_jepa_training_converges() -> None:
    result = train_jepa(epochs=120)
    assert result.final_loss < 0.5
    assert result.rollout_error < 0.7


def test_retrodiction_finds_forward_compatible_candidate() -> None:
    current = "def average(total, count):\n    return total / count\n"
    network = RetrodictiveIdentityNetwork()
    results = network.infer(
        current,
        generators=code_bug_generators(),
        forward_simulator=lambda candidate: (
            candidate.replace("return total / (count - 1)", "return total / count")
            .replace("return total / count if count else 0", "return total / count")
            .replace("total =+ value", "total += value")
        ),
        invariant_checker=lambda candidate: "return" in candidate,
    )
    assert results[0].forward_match
