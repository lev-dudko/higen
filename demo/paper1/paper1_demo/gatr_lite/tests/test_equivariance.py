"""Equivariance unit tests for GATr-lite under Spin^+(1,3).

For each test we sample 5 random rotors with rapidity up to 0.5 and rotation
angles in [-pi, pi]; the worst error across all samples must satisfy the
declared tolerance (1e-5 for layer-level checks, looser only where noted).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.paper1_demo.gatr_lite.gatr import algebra as A
from src.paper1_demo.gatr_lite.gatr.layers import (
    BlockConfig,
    EquivariantLinear,
    GATrBlock,
    GeometricProduct,
    JoinBlock,
    JoinProduct,
    ScalarAttention,
)
from src.paper1_demo.gatr_lite.gatr.model import GATrLite, GATrLiteConfig


SEED = 1234
N_TRIALS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _torch_rotor(rotor_np: np.ndarray, dtype=torch.float64) -> torch.Tensor:
    return torch.from_numpy(rotor_np.astype(np.float64)).to(dtype)


def _apply_rotor_to_tokens(x: torch.Tensor, rotor: torch.Tensor) -> torch.Tensor:
    """Apply rotor sandwich to every token / channel of x of shape (..., 16)."""
    return A.apply_rotor(x, rotor)


def _grade1_lift(p4_np: np.ndarray) -> np.ndarray:
    """Embed an array of 4-vectors of shape (..., 4) into Cl(1,3) grade-1.

    Returns array of shape (..., 16) with components placed at e_0..e_3.
    """
    out = np.zeros(p4_np.shape[:-1] + (16,), dtype=np.float64)
    for mu in range(4):
        out[..., A.GRADE1_INDICES[mu]] = p4_np[..., mu]
    return out


def _grade1_extract(x_np: np.ndarray) -> np.ndarray:
    return np.stack([x_np[..., A.GRADE1_INDICES[mu]] for mu in range(4)], axis=-1)


# ---------------------------------------------------------------------------
# Cayley table sanity
# ---------------------------------------------------------------------------

def test_cayley_table_consistency():
    # e_0^2 = +1, e_i^2 = -1
    e0 = A.GRADE_INDICES[1][0]
    for i, idx in enumerate(A.GRADE_INDICES[1]):
        target = int(A.CAYLEY_TARGETS[idx, idx])
        sign = int(A.CAYLEY_SIGNS[idx, idx])
        assert target == 0  # scalar
        expected = 1 if i == 0 else -1
        assert sign == expected, f"e_{i}^2 sign mismatch: {sign} vs {expected}"

    # Anticommutation e_i e_j = - e_j e_i for i != j
    for i in A.GRADE_INDICES[1]:
        for j in A.GRADE_INDICES[1]:
            if i == j:
                continue
            assert A.CAYLEY_TARGETS[i, j] == A.CAYLEY_TARGETS[j, i]
            assert A.CAYLEY_SIGNS[i, j] == -A.CAYLEY_SIGNS[j, i]

    # Pseudoscalar squared in (+,-,-,-) signature equals -1.
    I = 15
    target_I2 = int(A.CAYLEY_TARGETS[I, I])
    sign_I2 = int(A.CAYLEY_SIGNS[I, I])
    assert target_I2 == 0
    assert sign_I2 == -1

    # Reverse signs follow (-1)^{k(k-1)/2}
    expected_rev = [(-1) ** (k * (k - 1) // 2) for k in A.BASIS_GRADES]
    assert list(A.REVERSE_SIGNS) == expected_rev


def test_rotor_sandwich_identity():
    R = np.zeros(16, dtype=np.float64)
    R[0] = 1.0
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(16)
    out = A.apply_rotor(x, R)
    np.testing.assert_allclose(out, x, atol=1e-12)


def test_apply_rotor_grade1_is_lorentz_transform():
    rng = np.random.default_rng(SEED)
    for trial in range(N_TRIALS):
        R = A.random_rotor(boost_max=0.5, rng=rng)
        L = A.lorentz_matrix_from_rotor(R)
        # Lambda^T g Lambda = g
        g = np.diag([1.0, -1.0, -1.0, -1.0])
        np.testing.assert_allclose(L.T @ g @ L, g, atol=1e-6)
        # Apply to a random 4-vector and compare both routes
        p4 = rng.standard_normal(4)
        x = _grade1_lift(p4)
        xp = A.apply_rotor(x, R)
        np.testing.assert_allclose(_grade1_extract(xp), L @ p4, atol=1e-6)


def test_geometric_product_equivariance():
    rng = np.random.default_rng(SEED)
    for trial in range(N_TRIALS):
        R = A.random_rotor(boost_max=0.5, rng=rng)
        x = rng.standard_normal(16)
        y = rng.standard_normal(16)
        lhs = A.geometric_product(A.apply_rotor(x, R), A.apply_rotor(y, R))
        rhs = A.apply_rotor(A.geometric_product(x, y), R)
        np.testing.assert_allclose(lhs, rhs, atol=1e-5)


def test_inner_product_invariance():
    rng = np.random.default_rng(SEED)
    for trial in range(N_TRIALS):
        R = A.random_rotor(boost_max=0.5, rng=rng)
        x = rng.standard_normal(16)
        y = rng.standard_normal(16)
        ip = A.inner_product(x, y)
        ip_R = A.inner_product(A.apply_rotor(x, R), A.apply_rotor(y, R))
        np.testing.assert_allclose(ip_R, ip, atol=1e-5)


# ---------------------------------------------------------------------------
# Layer-level equivariance (torch)
# ---------------------------------------------------------------------------

def _random_input(rng, B=2, T=6, C=4, dtype=torch.float64):
    arr = rng.standard_normal((B, T, C, 16))
    return torch.from_numpy(arr.astype(np.float64)).to(dtype)


def _equivariance_max_err(layer, x: torch.Tensor, R_np: np.ndarray) -> float:
    """Return max abs diff between layer(R x R~) and R layer(x) R~."""
    R = _torch_rotor(R_np, dtype=x.dtype)
    with torch.no_grad():
        out_direct = A.apply_rotor(layer(x), R)
        Rx = A.apply_rotor(x, R)
        out_transformed = layer(Rx)
        return float((out_direct - out_transformed).abs().max().item())


def test_equivariant_linear_equivariance():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    layer = EquivariantLinear(4, 6).double()
    layer.eval()
    x = _random_input(rng, C=4)
    for _ in range(N_TRIALS):
        R_np = A.random_rotor(boost_max=0.5, rng=rng)
        err = _equivariance_max_err(layer, x, R_np)
        assert err < 1e-5, f"EquivariantLinear equivariance err={err}"


def test_geometric_product_layer_equivariance():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    layer = GeometricProduct(4, 4, mid_channels=4).double()
    layer.eval()
    x = _random_input(rng, C=4)
    for _ in range(N_TRIALS):
        R_np = A.random_rotor(boost_max=0.5, rng=rng)
        err = _equivariance_max_err(lambda t: layer(t, t), x, R_np)
        assert err < 1e-5, f"GeometricProduct layer equivariance err={err}"


def test_scalar_attention_equivariance():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    layer = ScalarAttention(4, attn_channels=4).double()
    layer.eval()
    x = _random_input(rng, C=4)
    for _ in range(N_TRIALS):
        R_np = A.random_rotor(boost_max=0.5, rng=rng)
        err = _equivariance_max_err(layer, x, R_np)
        assert err < 1e-5, f"ScalarAttention equivariance err={err}"


def test_gatr_block_equivariance():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    layer = GATrBlock(BlockConfig(channels=4, gp_mid_channels=4, attn_channels=4)).double()
    layer.eval()
    x = _random_input(rng, C=4)
    for _ in range(N_TRIALS):
        R_np = A.random_rotor(boost_max=0.5, rng=rng)
        err = _equivariance_max_err(layer, x, R_np)
        # Slightly looser tolerance: stacked operations accumulate roundoff.
        assert err < 1e-4, f"GATrBlock equivariance err={err}"


def test_join_product_equivariance():
    """JoinProduct = anti-symmetrised gp must be equivariant under sandwich."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    layer = JoinProduct(4, 4, mid_channels=4).double()
    layer.eval()
    x = _random_input(rng, C=4)
    for _ in range(N_TRIALS):
        R_np = A.random_rotor(boost_max=0.5, rng=rng)
        err = _equivariance_max_err(lambda t: layer(t, t), x, R_np)
        assert err < 1e-5, f"JoinProduct equivariance err={err}"


def test_join_block_equivariance():
    """Full JoinBlock (Group B): wedge + attention + gated act, all equivariant."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    layer = JoinBlock(BlockConfig(channels=4, gp_mid_channels=4, attn_channels=4)).double()
    layer.eval()
    x = _random_input(rng, C=4)
    for _ in range(N_TRIALS):
        R_np = A.random_rotor(boost_max=0.5, rng=rng)
        err = _equivariance_max_err(layer, x, R_np)
        # Same tolerance as GATrBlock (stacked operations).
        assert err < 1e-4, f"JoinBlock equivariance err={err}"


def test_gatr_lite_logit_invariance():
    """The full classifier produces a logit that is Lorentz-invariant.

    This test exercises the legacy 6-token architecture (use_pairing_cache=False);
    the pairing-aware variant is covered by test_pairing.py.
    """
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    cfg = GATrLiteConfig(
        channels=8, gp_mid_channels=4, attn_channels=4, head_hidden=16,
        use_pairing_cache=False,
    )
    model = GATrLite(cfg).double()
    model.eval()
    B, T = 3, 6
    p4 = torch.from_numpy(rng.standard_normal((B, T, 4))).double()
    pdg = torch.tensor([[13, -14, 2, -1, 5, -5]] * B, dtype=torch.long)
    base_logit = model(p4, pdg)
    for _ in range(N_TRIALS):
        R_np = A.random_rotor(boost_max=0.5, rng=rng)
        L = A.lorentz_matrix_from_rotor(R_np)
        L_t = torch.from_numpy(L).double()
        # Active transform of every 4-momentum
        p4_t = torch.einsum("mn,btn->btm", L_t, p4)
        new_logit = model(p4_t, pdg)
        err = (new_logit - base_logit).abs().max().item()
        assert err < 1e-4, f"GATrLite logit not invariant: err={err}"
