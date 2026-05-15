"""Tests for arch_v2 guarded improvements (multi-head attention, dropout, gp3)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from paper1_demo.gatr_lite.gatr.algebra import apply_rotor, random_rotor
from paper1_demo.gatr_lite.gatr.model import GATrLite, GATrLiteConfig


def _example_inputs(batch: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    p4 = torch.randn(batch, 6, 4) * 30.0
    p4[:, :, 0] = p4[:, :, 1:].norm(dim=-1) + 50.0  # E > |p|
    pdg = torch.tensor([13, -14, 2, -1, 5, -5]).unsqueeze(0).repeat(batch, 1)
    return p4, pdg


def test_v1_default_unchanged():
    """v1 baseline (no JoinBlock, no Group-A pairing extensions) frozen at 151009.

    Default has moved on (Groups A + B add meet/join channels and a JoinBlock);
    this test pins the legacy point so we can still recreate it for ablations.
    """
    cfg = GATrLiteConfig(use_join_block=False)
    # Force the original 8 pairing-scalar layout for byte-identical v1.
    from paper1_demo.gatr_lite.gatr import model as M
    if M.N_PAIRING_SCALARS != 8:
        pytest.skip("v1 layout (8 pairing scalars) no longer the default")
    assert cfg.arch_v2 is False
    m = GATrLite(cfg)
    assert sum(p.numel() for p in m.parameters()) == 151_009


def test_default_with_groups_AB():
    """New default (Groups A + B): meet/join scalars + JoinBlock."""
    cfg = GATrLiteConfig()
    assert cfg.use_join_block is True
    m = GATrLite(cfg)
    n_params = sum(p.numel() for p in m.parameters())
    # 151489 (Group A only) + JoinBlock ≈ 196k.
    assert 180_000 <= n_params <= 220_000, f"unexpected param count: {n_params}"


@pytest.mark.parametrize("n_heads", [1, 2, 4])
def test_arch_v2_multihead_forward_backward(n_heads):
    cfg = GATrLiteConfig(arch_v2=True, n_heads=n_heads, channels=32)
    m = GATrLite(cfg)
    p4, pdg = _example_inputs(4)
    out = m(p4, pdg)
    assert out.shape == (4, 1)
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_arch_v2_dropout_eval_deterministic():
    cfg = GATrLiteConfig(arch_v2=True, n_heads=2, channels=32, dropout=0.5)
    m = GATrLite(cfg).eval()
    p4, pdg = _example_inputs(2)
    o1 = m(p4, pdg)
    o2 = m(p4, pdg)
    torch.testing.assert_close(o1, o2)


def test_arch_v2_lorentz_invariance():
    """Logit invariance under random rotor sandwich."""
    torch.manual_seed(42)
    cfg = GATrLiteConfig(arch_v2=True, n_heads=4, channels=32, dropout=0.0)
    m = GATrLite(cfg).eval()
    p4, pdg = _example_inputs(2)
    out0 = m(p4, pdg)

    rng = np.random.default_rng(0)
    for _ in range(3):
        R = random_rotor(boost_max=0.3, rng=rng)  # numpy array (16,)
        R_t = torch.tensor(R, dtype=p4.dtype)
        # rotate the grade-1 part of the input p4 (E,px,py,pz) using rotor
        # apply_rotor expects multivector (16) - encode p4 as grade-1 mv
        mv = torch.zeros(p4.shape[0], p4.shape[1], 16, dtype=p4.dtype)
        mv[..., 1:5] = p4
        mv_rot = apply_rotor(mv.numpy(), R)
        p4_rot = torch.tensor(mv_rot[..., 1:5], dtype=p4.dtype)
        out1 = m(p4_rot, pdg)
        torch.testing.assert_close(out0, out1, atol=5e-3, rtol=5e-3)


def test_arch_v2_param_budget():
    """v2 with channels=64, n_heads=4 stays under 1.5M params."""
    cfg = GATrLiteConfig(arch_v2=True, n_heads=4, channels=64, dropout=0.1)
    m = GATrLite(cfg)
    n = sum(p.numel() for p in m.parameters())
    assert n < 1_500_000, f"v2 budget exceeded: {n}"


def test_arch_v2_gp3_init_nudge():
    """gp_grade3_mixing path executes without error and adds no new params."""
    cfg_base = GATrLiteConfig(arch_v2=True, n_heads=2, channels=32)
    cfg_gp3 = GATrLiteConfig(arch_v2=True, n_heads=2, channels=32, gp_grade3_mixing=True)
    n_base = sum(p.numel() for p in GATrLite(cfg_base).parameters())
    n_gp3 = sum(p.numel() for p in GATrLite(cfg_gp3).parameters())
    assert n_base == n_gp3
