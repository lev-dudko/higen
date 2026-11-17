"""Mathematical contract of the outer product in JoinProduct."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from paper1_demo.gatr_lite.gatr import algebra as A
from paper1_demo.gatr_lite.gatr.layers import JoinProduct
from paper1_demo.gatr_lite.gatr.model import GATrLite, GATrLiteConfig
from paper1_demo.gatr_lite.training.eval import load_model
from paper1_demo.gatr_lite.training.train import _build_parser


def _identity_join() -> JoinProduct:
    """Leaves only the bilinear product under test in the layer."""
    layer = JoinProduct(1, 1, mid_channels=1).double()
    with torch.no_grad():
        for projection in (layer.proj_x, layer.proj_y):
            for weight in projection.weights:
                weight.fill_(1.0)
        layer.combine.fill_(1.0)
    return layer


def test_disjoint_bivectors_produce_pseudoscalar() -> None:
    """e01 ∧ e23 = I; the commutator wrongly gives zero."""
    basis = torch.eye(16, dtype=torch.float64)
    x = basis[5].reshape(1, 1, 1, 16)
    y = basis[10].reshape(1, 1, 1, 16)
    expected = basis[15].reshape(1, 1, 1, 16)
    torch.testing.assert_close(_identity_join()(x, y), expected)


def test_join_product_on_all_basis_pairs() -> None:
    """Checks the permutation sign and the zero for a repeated index."""
    basis = torch.eye(16, dtype=torch.float64)
    x = basis.repeat_interleave(16, dim=0).reshape(256, 1, 1, 16)
    y = basis.repeat(16, 1).reshape(256, 1, 1, 16)
    expected = torch.zeros_like(x)
    for i, left in enumerate(A.BASIS):
        for j, right in enumerate(A.BASIS):
            if set(left).intersection(right):
                continue
            inversions = sum(a > b for a in left for b in right)
            target = A.BASIS.index(tuple(sorted(left + right)))
            expected[16 * i + j, 0, 0, target] = (-1) ** inversions
    torch.testing.assert_close(_identity_join()(x, y), expected)


def test_join_product_mixed_grades_and_gradients() -> None:
    """(2 + e01) ∧ (3 + e23) = 6 + 3e01 + 2e23 + I."""
    x = torch.zeros(1, 1, 1, 16, dtype=torch.float64)
    y = torch.zeros_like(x)
    x[..., 0], x[..., 5] = 2.0, 1.0
    y[..., 0], y[..., 10] = 3.0, 1.0
    x.requires_grad_()
    y.requires_grad_()
    layer = _identity_join()
    expected = torch.zeros_like(x)
    expected[..., 0] = 6.0
    expected[..., 5] = 3.0
    expected[..., 10] = 2.0
    expected[..., 15] = 1.0
    result = layer(x, y)
    torch.testing.assert_close(result, expected)
    result.square().sum().backward()
    for value in (x, y, *layer.parameters()):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
    assert x.grad[..., 5].item() == 20.0
    assert y.grad[..., 10].item() == 10.0


@pytest.mark.parametrize("metadata", ["model_cfg", "args"])
@pytest.mark.parametrize("product", ["commutator", "wedge"])
def test_checkpoint_preserves_join_operation(
    tmp_path: Path, metadata: str, product: str,
) -> None:
    """An old checkpoint keeps the commutator, a new one the selected operation."""
    torch.manual_seed(42)
    cfg = GATrLiteConfig(join_product=product)
    model = GATrLite(cfg).eval()
    assert model.num_parameters() == 196609
    stored_cfg = asdict(cfg)
    if product == "commutator":
        del stored_cfg["join_product"]
    path = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), metadata: stored_cfg}, path)
    restored = load_model(path, torch.device("cpu"))
    assert restored.join_block.join.product == product
    p4 = torch.randn(2, 6, 4)
    p4[..., 0] = p4[..., 1:].norm(dim=-1) + 1.0
    pdg = torch.tensor([[13, -14, 2, -1, 5, -5]]).repeat(2, 1)
    with torch.no_grad():
        torch.testing.assert_close(restored(p4, pdg), model(p4, pdg), rtol=0, atol=0)


def test_invalid_product_is_rejected() -> None:
    """An unknown operation name must not silently change the experiment."""
    with pytest.raises(ValueError, match="product"):
        JoinProduct(1, 1, product="unknown")


def test_training_defaults_to_wedge() -> None:
    """A new run selects the wedge even if the old config format means commutator."""
    args = _build_parser().parse_args([
        "--signal", "signal.h5", "--background", "background.h5", "--out", "run",
    ])
    assert args.join_product == "wedge"
    assert args.join_attention_fp64 is True
    assert GATrLiteConfig().join_product == "commutator"
    assert GATrLiteConfig().join_attention_fp64 is False
