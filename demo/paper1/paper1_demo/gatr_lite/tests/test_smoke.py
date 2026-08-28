"""Forward / backward smoke test for GATrLite (default config)."""

from __future__ import annotations

import torch

from paper1_demo.gatr_lite.gatr.model import GATrLite, GATrLiteConfig


def test_forward_shapes_and_param_budget():
    model = GATrLite()
    n_params = model.num_parameters()
    # Target budget after Groups A + B: ~190k-210k parameters (was 151k for
    # legacy v1; pairing extensions add ~500, JoinBlock adds ~45k).
    assert 180_000 <= n_params <= 220_000, f"Param count {n_params} outside budget"

    p4 = torch.randn(4, 6, 4)
    pdg = torch.tensor([[13, -14, 2, -1, 5, -5]] * 4)
    out = model(p4, pdg)
    assert out.shape == (4, 1)
    assert torch.isfinite(out).all()


def test_backward_runs():
    model = GATrLite()
    p4 = torch.randn(4, 6, 4, requires_grad=False)
    pdg = torch.tensor([[13, -14, 2, -1, 5, -5]] * 4)
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    logit = model(p4, pdg)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
    loss.backward()
    # Confirm at least one parameter received a gradient.
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "no parameter received a gradient"


def test_handles_unknown_pdg():
    """Unknown PDG ids should fall back to default flavor 0 without crashing."""
    model = GATrLite()
    p4 = torch.randn(2, 6, 4)
    pdg = torch.tensor([[999, 13, -14, 2, -1, 5], [13, -14, 2, -1, 5, -5]])
    out = model(p4, pdg)
    assert out.shape == (2, 1)
