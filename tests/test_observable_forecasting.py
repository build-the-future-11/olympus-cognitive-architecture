from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from olympus.core.forecasting import ObservablePredictor, representation_diagnostics
from scripts.analyze_observable_prediction import paired_analysis
from scripts.run_observable_prediction import (
    fit_ridge,
    forecast,
    generate_split,
    polynomial_features,
    trajectories,
)


def test_target_is_a_frozen_copy_and_ema_has_exact_update_order() -> None:
    model = ObservablePredictor()
    previous = [p.clone() for p in model.encoder.parameters()]
    assert all(not p.requires_grad for p in model.target_encoder.parameters())
    for target, online in zip(model.target_encoder.parameters(), previous, strict=True):
        assert torch.equal(target, online)
    with torch.no_grad():
        for parameter in model.encoder.parameters():
            parameter.add_(2)
    model.update_target(0.75)
    for target, before in zip(model.target_encoder.parameters(), previous, strict=True):
        torch.testing.assert_close(target, before + 0.5)


def test_objective_only_shuffles_auxiliary_targets_and_blocks_target_gradient() -> None:
    torch.manual_seed(11)
    model = ObservablePredictor().double()
    inputs, targets = torch.randn(5, 2, dtype=torch.double), torch.randn(5, 2, dtype=torch.double)
    loss, aligned = model.objective(inputs, targets, latent_weight=0.1, variance_weight=0.01)
    _, shuffled = model.objective(
        inputs, targets, latent_weight=0.1, variance_weight=0.01,
        target_permutation=torch.tensor([1, 2, 3, 4, 0]),
    )
    assert torch.equal(aligned["state"], shuffled["state"])
    assert torch.equal(aligned["reconstruction"], shuffled["reconstruction"])
    assert not torch.equal(aligned["latent"], shuffled["latent"])
    torch.autograd.backward(loss)
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
               for p in model.encoder.parameters())


def test_forward_gradient_matches_finite_differences() -> None:
    torch.manual_seed(3)
    model = ObservablePredictor(width=3, latent_dim=2).double()
    inputs = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(model, (inputs,))


def test_rollout_feeds_its_predictions_back_and_checkpoint_is_exact() -> None:
    model = ObservablePredictor(width=2, latent_dim=2)
    model.encoder = nn.Identity()
    predictor = nn.Linear(2, 2, bias=False)
    decoder = nn.Linear(2, 2, bias=False)
    model.predictor, model.decoder = predictor, decoder
    with torch.no_grad():
        predictor.weight.zero_()
        decoder.weight.copy_(torch.eye(2) * 2)
    initial = torch.tensor([[1.0, -1.0]])
    expected = torch.tensor([[[2.0, -2.0], [4.0, -4.0], [8.0, -8.0]]])
    assert torch.equal(model.rollout(initial, 3), expected)
    cloned = copy.deepcopy(model)
    assert torch.equal(cloned.rollout(initial, 3), expected)


@pytest.mark.parametrize("momentum", [-0.1, 1.1, float("nan")])
def test_invalid_ema_is_rejected(momentum: float) -> None:
    with pytest.raises(ValueError, match="momentum"):
        ObservablePredictor().update_target(momentum)


@pytest.mark.parametrize("value", [torch.zeros(2), torch.zeros(2, 3),
                                   torch.zeros(0, 2), torch.ones(2, 2, dtype=torch.long),
                                   torch.full((2, 2), float("nan"))])
def test_invalid_state_rejected(value: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        ObservablePredictor()(value)


def test_shape_coefficient_and_permutation_contracts() -> None:
    model, x = ObservablePredictor(), torch.zeros(3, 2)
    for invalid in [-0.1, float("nan")]:
        with pytest.raises(ValueError, match="weights"):
            model.objective(x, x, latent_weight=invalid, variance_weight=0)
    with pytest.raises(ValueError, match="matching"):
        model.objective(x, x[:1], latent_weight=0, variance_weight=0)
    with pytest.raises(ValueError, match="permutation"):
        model.objective(x, x, latent_weight=1, variance_weight=0,
                        target_permutation=torch.tensor([0, 0, 2]))
    with pytest.raises(ValueError, match="horizon"):
        model.rollout(x, 0)
    with pytest.raises(ValueError, match="dimensions"):
        ObservablePredictor(width=0)


def test_effective_rank_does_not_mistake_copied_coordinates_for_independence() -> None:
    model = ObservablePredictor()
    encoder = nn.Linear(2, 8, bias=False)
    model.encoder = encoder
    with torch.no_grad():
        encoder.weight.fill_(1)
    inputs = torch.linspace(-1, 1, 32)[:, None].repeat(1, 2)
    report = representation_diagnostics(model, inputs)
    assert float(report["effective_rank"]) < 1.00001  # type: ignore[arg-type]
    zero = representation_diagnostics(model, torch.zeros(5, 2))
    assert zero["effective_rank"] == 0


def test_generator_has_independent_splits_and_numerical_refinement_agreement() -> None:
    first = generate_split(601, 0, 0, 8, 32)
    second = generate_split(601, 0, 1, 8, 32)
    assert np.array_equal(first, generate_split(601, 0, 0, 8, 32))
    assert not np.array_equal(first[:, 0], second[:, 0])
    refined = trajectories(first[:, 0], "pendulum", 32, substeps=8)
    np.testing.assert_allclose(first, refined, rtol=0, atol=2e-7)
    np.testing.assert_array_equal(trajectories(np.zeros((3, 2)), "duffing", 10), 0)


def test_affine_baseline_recovers_a_hand_defined_transition() -> None:
    rng = np.random.default_rng(51)
    train, test = rng.normal(size=(100, 2)), rng.normal(size=(20, 2))
    matrix = np.array([[0.8, -0.1], [0.2, 0.95]])
    expected = test @ matrix + 0.15
    weights = fit_ridge(train, train @ matrix + 0.15, 1)
    np.testing.assert_allclose(polynomial_features(test, 1) @ weights, expected, atol=1e-5)


def test_divergence_is_preserved_and_not_excluded_from_error() -> None:
    initial = np.array([[2.0, 1.0], [0.0, 0.0]])
    result = forecast(lambda x: x * 1e7, initial, np.zeros((2, 50, 2)))
    assert result["divergent_trajectories"] == 1
    assert result["horizons"]["20"] == 5e11
    assert result["predictions"][0, 0, 0] == 2e7


def test_sign_test_and_bootstrap_match_hand_cases() -> None:
    result = paired_analysis(np.full(8, np.log(2)))
    assert result["effect_fraction"] == pytest.approx(0.5)
    assert result["effect_ci95"] == pytest.approx([0.5, 0.5])
    assert result["one_sided_sign_p"] == 1 / 256
    seven = paired_analysis(np.array([1.0] * 7 + [-1.0]))
    assert seven["one_sided_sign_p"] == 9 / 256
    with pytest.raises(ValueError, match="eight"):
        paired_analysis(np.zeros(7))
