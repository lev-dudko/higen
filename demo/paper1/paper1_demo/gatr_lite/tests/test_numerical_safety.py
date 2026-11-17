"""Protection of attention and training against overflow without changing the formulas."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from paper1_demo.gatr_lite.gatr.layers import MultiHeadScalarAttention, ScalarAttention
from paper1_demo.gatr_lite.gatr.model import GATrLite, GATrLiteConfig
from paper1_demo.gatr_lite.training.eval import load_model
from paper1_demo.gatr_lite.training.train import _epoch_eval, _epoch_train, _roc_auc


@pytest.mark.parametrize("heads", [1, 2])
def test_large_finite_attention_has_finite_output_and_gradients(heads: int) -> None:
    """Scores of order 10⁴⁶ must give a finite softmax that selects the second token."""
    layer = (ScalarAttention(1, attn_channels=1, scores_fp64=True) if heads == 1 else
             MultiHeadScalarAttention(2, n_heads=2, attn_channels=1, scores_fp64=True))
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.fill_(1.0)
    x = torch.zeros(1, 2, heads, 16)
    x[:, 0, :, 0], x[:, 1, :, 0] = 1e23, 2e23
    x.requires_grad_()
    result = layer(x)
    expected = torch.zeros_like(x)
    expected[..., 0] = 2e23
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected)
    (result / 1e23).sum().backward()
    assert torch.isfinite(x.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


class _ScalarClassifier(nn.Module):
    """Minimal classifier for testing a real AdamW step."""
    def __init__(self, singular_gradient: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0 if singular_gradient else 1.0))
        self.singular_gradient = singular_gradient

    def forward(self, p4: torch.Tensor, pdg: torch.Tensor) -> torch.Tensor:
        coefficient = self.weight.sqrt() if self.singular_gradient else self.weight
        return coefficient * p4[:, :1, 0]


def _batch(value: float) -> dict[str, torch.Tensor]:
    return {"p4": torch.full((1, 6, 4), value),
            "pdg": torch.zeros(1, 6, dtype=torch.long), "label": torch.zeros(1)}


@pytest.mark.parametrize("clip", [0.0, 1.0])
@pytest.mark.parametrize("singular_gradient", [False, True])
def test_nonfinite_step_does_not_update_weights(clip: float, singular_gradient: bool) -> None:
    """A NaN in forward or an Inf in backward must not reach the AdamW state."""
    model = _ScalarClassifier(singular_gradient)
    before = model.weight.detach().clone()
    optim = torch.optim.AdamW(model.parameters(), lr=0.1)
    batch = _batch(1.0 if singular_gradient else float("nan"))
    with pytest.raises((FloatingPointError, RuntimeError), match="[Nn]on.?finite"):
        _epoch_train(model, [batch], optim, nn.BCEWithLogitsLoss(), torch.device("cpu"), clip)
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    assert not optim.state


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_validation_rejects_nonfinite_logits(bad: float) -> None:
    """The sigmoid must not hide a non-finite logit when the model is evaluated."""
    with pytest.raises(FloatingPointError, match="[Nn]on.?finite"):
        _epoch_eval(_ScalarClassifier(), [_batch(bad)], nn.BCEWithLogitsLoss(), torch.device("cpu"))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_auc_rejects_nonfinite_scores(bad: float) -> None:
    """Ranks of NaN/Inf do not give a valid AUC."""
    with pytest.raises(ValueError, match="[Nn]on.?finite"):
        _roc_auc(np.array([0.1, bad]), np.array([0, 1]))


def test_finite_training_and_auc_still_work() -> None:
    """The protection lets normal training through and handles ties in the AUC correctly."""
    model = _ScalarClassifier()
    optim = torch.optim.AdamW(model.parameters(), lr=0.1)
    loss = _epoch_train(model, [_batch(1.0)], optim, nn.BCEWithLogitsLoss(), torch.device("cpu"))
    assert np.isfinite(loss) and model.weight.item() < 1.0
    assert _roc_auc(np.array([0.1, 0.4, 0.4, 0.9]), np.array([0, 1, 0, 1])) == 0.875


@pytest.mark.parametrize("use_fp64", [False, True])
def test_checkpoint_preserves_attention_precision(tmp_path: Path, use_fp64: bool) -> None:
    """An old checkpoint stays float32, a new one keeps the explicit choice of precision."""
    cfg = GATrLiteConfig(join_product="wedge", join_attention_fp64=use_fp64)
    model = GATrLite(cfg)
    stored = asdict(cfg)
    if not use_fp64:
        del stored["join_attention_fp64"]
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "model_cfg": stored}, path)
    loaded = load_model(path, torch.device("cpu"))
    assert loaded.join_block.attn.scores_fp64 is use_fp64
    assert loaded.num_parameters() == 196609
