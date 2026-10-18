from __future__ import annotations

import pytest
import torch

from olympus.models.editclevr import EDIT_FACTORS, OlympusEditCLEVRModel, editclevr_loss


def test_editclevr_model_outputs_normalized_controls_and_gradients() -> None:
    torch.manual_seed(7)
    model = OlympusEditCLEVRModel(width=8, embedding_dim=32)
    before = torch.rand(4, 3, 64, 64)
    after = before.clone()
    after[1:, :, 16:32, 16:32] = 1 - after[1:, :, 16:32, 16:32]
    output = model(before, after)
    assert set(output.probabilities) == {"learned", "uniform", "pair_only", "delta_only"}
    assert output.gates.shape == (4, 4)
    torch.testing.assert_close(output.gates.sum(-1), torch.ones(4))
    for probabilities in output.probabilities.values():
        assert probabilities.shape == (4, len(EDIT_FACTORS))
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(4))
    loss = editclevr_loss(output, torch.tensor([0, 1, 2, 3]))
    torch.autograd.backward(loss)
    assert model.router[-1].weight.grad is not None
    assert model.image_encoder.network[0].weight.grad is not None
    assert model.delta_encoder.network[0].weight.grad is not None


def test_editclevr_model_rejects_invalid_inputs_and_labels() -> None:
    model = OlympusEditCLEVRModel(width=8, embedding_dim=32)
    image = torch.rand(2, 3, 64, 64)
    with pytest.raises(ValueError, match="share shape"):
        model(image, image[:1])
    with pytest.raises(ValueError, match="at least 32"):
        model(image[..., :16, :16], image[..., :16, :16])
    output = model(image, image)
    with pytest.raises(ValueError, match="torch.long"):
        editclevr_loss(output, torch.zeros(2))
