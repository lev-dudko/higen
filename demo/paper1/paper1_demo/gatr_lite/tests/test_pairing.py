"""Unit tests for the pairing-aware feature cache (gatr.pairing) and its
integration with the GATr-lite model.

The pairing cache adds two extra tokens (one per b-pairing) to the standard
six-parton input. Its scalar features are Lorentz-invariants; its multivector
features (T^(a) and *T^(a)) are Lorentz-covariant. The full model with
``use_pairing_cache=True`` must therefore still produce a Lorentz-invariant
logit, and with ``use_pairing_cache=False`` must reproduce the original
6-token / 8-scalar architecture exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from paper1_demo.gatr_lite.gatr import algebra as A
from paper1_demo.gatr_lite.gatr import pairing as P
from paper1_demo.gatr_lite.gatr.model import (
    GATrLite,
    GATrLiteConfig,
    PDG_PAIRING_1,
    PDG_PAIRING_2,
)


SEED = 4242
N_TRIALS = 5
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_random_event(rng: np.random.Generator, B: int = 3) -> torch.Tensor:
    """Generate a batch of plausible 6-parton events.

    The 4-momenta are not on physical mass shells (we only care about
    Lorentz-covariance properties), but they have positive energy so that
    transformations under random rotors are well-behaved.
    """
    p3 = rng.standard_normal((B, 6, 3)) * 30.0   # GeV scale
    m = rng.uniform(0.0, 5.0, size=(B, 6))
    E = np.sqrt(np.sum(p3 ** 2, axis=-1) + m ** 2)
    p4 = np.concatenate([E[..., None], p3], axis=-1)
    return torch.from_numpy(p4).to(DTYPE)


def _canonical_pdg(B: int) -> torch.Tensor:
    return torch.tensor([[13, -14, 2, -1, 5, -5]] * B, dtype=torch.long)


def _apply_lorentz_to_p4(p4: torch.Tensor, R_np: np.ndarray) -> torch.Tensor:
    L = A.lorentz_matrix_from_rotor(R_np)
    L_t = torch.from_numpy(L).to(p4.dtype)
    return torch.einsum("mn,btn->btm", L_t, p4)


# ---------------------------------------------------------------------------
# Scalar invariance under Lorentz transforms
# ---------------------------------------------------------------------------


def test_pairing_invariants_under_lorentz():
    """All scalar pairing features are Lorentz-invariants."""
    rng = np.random.default_rng(SEED)
    p4 = _make_random_event(rng, B=3)
    base = P.compute_pairing_features(p4)

    scalar_keys = [
        "s_had", "s_lep",
        "sigma_p_had", "sigma_m_had", "sigma_p_lep", "sigma_m_lep",
        "mass_W_had", "mass_W_lep",
        "coplanarity", "pseudoscalar_4",  # Group A scalars
    ]
    for _ in range(N_TRIALS):
        R_np = A.random_rotor(boost_max=0.5, rng=rng)
        p4_t = _apply_lorentz_to_p4(p4, R_np)
        new = P.compute_pairing_features(p4_t)
        for k in scalar_keys:
            diff = (new[k] - base[k]).abs().max().item()
            scale = max(base[k].abs().max().item(), 1.0)
            rel = diff / scale
            assert rel < 1e-6, (
                f"{k} not invariant under Lorentz; rel_err={rel} "
                f"(abs_err={diff}, scale={scale})"
            )


# ---------------------------------------------------------------------------
# Trivector / Hodge-dual grade structure
# ---------------------------------------------------------------------------


def test_T_pair_is_grade3():
    """T^(a) only has nonzero components in grade-3 basis indices."""
    rng = np.random.default_rng(SEED)
    p4 = _make_random_event(rng, B=2)
    pf = P.compute_pairing_features(p4)
    T = pf["T_pair"]  # (B, 2, 16)
    g3_idx = set(A.GRADE_INDICES[3])
    other_idx = [i for i in range(16) if i not in g3_idx]
    other = T[..., other_idx]
    assert other.abs().max().item() < 1e-12, (
        f"T_pair has non-grade-3 components: max={other.abs().max().item()}"
    )
    # And the grade-3 part is non-trivial.
    g3 = T[..., list(g3_idx)]
    assert g3.abs().max().item() > 0.0


def test_T_dual_is_grade1():
    """*T^(a) only has nonzero components in grade-1 basis indices."""
    rng = np.random.default_rng(SEED)
    p4 = _make_random_event(rng, B=2)
    pf = P.compute_pairing_features(p4)
    Td = pf["T_dual_pair"]
    g1_idx = set(A.GRADE_INDICES[1])
    other_idx = [i for i in range(16) if i not in g1_idx]
    other = Td[..., other_idx]
    assert other.abs().max().item() < 1e-12, (
        f"T_dual has non-grade-1 components: max={other.abs().max().item()}"
    )
    assert Td[..., list(g1_idx)].abs().max().item() > 0.0


def test_T_pair_covariance_under_lorentz():
    """T^(a) transforms covariantly: T(L p) = R T(p) R~ for L = Lorentz(R)."""
    rng = np.random.default_rng(SEED)
    p4 = _make_random_event(rng, B=2)
    base_T = P.compute_pairing_features(p4)["T_pair"]

    for _ in range(N_TRIALS):
        R_np = A.random_rotor(boost_max=0.5, rng=rng)
        R = torch.from_numpy(R_np).to(DTYPE)
        p4_t = _apply_lorentz_to_p4(p4, R_np)
        new_T = P.compute_pairing_features(p4_t)["T_pair"]
        rotated_base = A.apply_rotor(base_T, R)
        diff = (new_T - rotated_base).abs().max().item()
        scale = max(rotated_base.abs().max().item(), 1.0)
        rel = diff / scale
        assert rel < 1e-5, (
            f"T_pair not covariant: rel_err={rel} (abs_err={diff}, scale={scale})"
        )


def test_T_lep_covariance_and_grade():
    """Group A: T_lep^(a) is grade-3 and Lorentz-covariant."""
    rng = np.random.default_rng(SEED)
    p4 = _make_random_event(rng, B=2)
    pf = P.compute_pairing_features(p4)
    T_lep = pf["T_lep_pair"]
    g3_idx = set(A.GRADE_INDICES[3])
    other = T_lep[..., [i for i in range(16) if i not in g3_idx]]
    assert other.abs().max().item() < 1e-12

    for _ in range(N_TRIALS):
        R_np = A.random_rotor(boost_max=0.5, rng=rng)
        R = torch.from_numpy(R_np).to(DTYPE)
        p4_t = _apply_lorentz_to_p4(p4, R_np)
        new_T_lep = P.compute_pairing_features(p4_t)["T_lep_pair"]
        rotated = A.apply_rotor(T_lep, R)
        rel = (new_T_lep - rotated).abs().max().item() / max(rotated.abs().max().item(), 1.0)
        assert rel < 1e-5, f"T_lep_pair not covariant: rel_err={rel}"


def test_meet_pair_covariance_and_grade():
    """Group A: meet^(a) is grade-2 and Lorentz-covariant."""
    rng = np.random.default_rng(SEED)
    p4 = _make_random_event(rng, B=2)
    pf = P.compute_pairing_features(p4)
    M = pf["meet_pair"]
    g2_idx = set(A.GRADE_INDICES[2])
    other = M[..., [i for i in range(16) if i not in g2_idx]]
    assert other.abs().max().item() < 1e-10, (
        f"meet_pair has non-grade-2 components: {other.abs().max().item()}"
    )

    for _ in range(N_TRIALS):
        R_np = A.random_rotor(boost_max=0.5, rng=rng)
        R = torch.from_numpy(R_np).to(DTYPE)
        p4_t = _apply_lorentz_to_p4(p4, R_np)
        new_M = P.compute_pairing_features(p4_t)["meet_pair"]
        rotated = A.apply_rotor(M, R)
        rel = (new_M - rotated).abs().max().item() / max(rotated.abs().max().item(), 1.0)
        assert rel < 1e-5, f"meet_pair not covariant: rel_err={rel}"


# ---------------------------------------------------------------------------
# Model-level checks
# ---------------------------------------------------------------------------


def _small_cfg(use_pairing: bool) -> GATrLiteConfig:
    return GATrLiteConfig(
        n_blocks=2,
        channels=8,
        gp_mid_channels=4,
        attn_channels=4,
        head_hidden=16,
        use_pairing_cache=use_pairing,
    )


def test_pairing_token_lorentz_equivariance():
    """Full model with pairing cache is still Lorentz-invariant in the logit."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = GATrLite(_small_cfg(use_pairing=True)).double()
    model.eval()

    p4 = _make_random_event(rng, B=3)
    pdg = _canonical_pdg(3)
    base_logit = model(p4, pdg)

    for _ in range(N_TRIALS):
        R_np = A.random_rotor(boost_max=0.5, rng=rng)
        p4_t = _apply_lorentz_to_p4(p4, R_np)
        new_logit = model(p4_t, pdg)
        diff = (new_logit - base_logit).abs().max().item()
        scale = max(base_logit.abs().max().item(), 1.0)
        rel = diff / scale
        # Logits can be very large (pairing scalars carry GeV^2 / Gamma_t factors)
        # so we check relative invariance under the rotor sandwich.
        assert rel < 1e-5, (
            f"GATrLite (pairing) logit not invariant: rel_err={rel} "
            f"(abs_err={diff}, scale={scale})"
        )


def test_consistency_old_vs_new():
    """``use_pairing_cache=False`` reproduces the original 6-token model.

    With the legacy flag, the model must use the 6-flavor / 8-scalar layout,
    have no pairing-derived inputs, and remain Lorentz-invariant in the
    logit -- i.e. equivalent to the architecture used in the prior tests.
    """
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = GATrLite(_small_cfg(use_pairing=False)).double()
    model.eval()

    # Layout sanity: 6 flavor slots, 8 scalar channels, c_in = 9.
    assert model._n_flavors == 6
    assert model._n_scalars == 8
    assert model.c_in == 9

    p4 = _make_random_event(rng, B=3)
    pdg = _canonical_pdg(3)
    # Embedding has only 6 tokens, not 8.
    emb = model.embed(p4, pdg)
    assert emb.shape == (3, 6, 9, 16)

    base_logit = model(p4, pdg)
    for _ in range(N_TRIALS):
        R_np = A.random_rotor(boost_max=0.5, rng=rng)
        p4_t = _apply_lorentz_to_p4(p4, R_np)
        new_logit = model(p4_t, pdg)
        err = (new_logit - base_logit).abs().max().item()
        assert err < 1e-4, f"Legacy (no-pairing) logit not invariant: err={err}"


def test_pairing_for_known_event():
    """If the hadronic top is on shell, s_had is zero for the correct pairing.

    Construct a synthetic event where p_u + p_dbar + p_b sums to a 4-vector
    of squared mass exactly m_t^2. Then for pairing 1 (hadronic_b = b)
    s_had should vanish; for pairing 2 (hadronic_b = b_bar) it should not.
    """
    rng = np.random.default_rng(SEED)
    # Build u, d_bar, b such that their sum has mass = m_t along z-axis.
    # Pick three-momenta that add up to zero in 3-space and energies that
    # add up to m_t (top at rest).
    #   u   : (E_u, +pz, 0, 0)
    #   d   : (E_d, -pz, 0, 0)
    #   b   : (E_b, 0,  0, 0)
    pz = 30.0
    m_u, m_d, m_b = 0.0, 0.0, 4.7
    E_u = math.sqrt(pz ** 2 + m_u ** 2)
    E_d = math.sqrt(pz ** 2 + m_d ** 2)
    # Top at rest: total E = m_t, so E_b = m_t - E_u - E_d.
    E_b = P.M_T - E_u - E_d
    # Need E_b >= m_b for a real b 3-momentum (we put b at rest).
    assert E_b >= m_b

    p_u = [E_u, pz, 0.0, 0.0]
    p_d = [E_d, -pz, 0.0, 0.0]
    p_b = [E_b, 0.0, 0.0, 0.0]
    # Leptonic side: arbitrary; pick lepton + neutrino + b_bar that don't
    # match the top mass. Stays well-defined; we only test s_had.
    p_mu = [50.0, 10.0, 5.0, 2.0]
    p_nu = [40.0, -7.0, 3.0, -1.0]
    p_bb = [60.0, 0.0, 0.0, 50.0]

    p4 = torch.tensor([[p_mu, p_nu, p_u, p_d, p_b, p_bb]], dtype=DTYPE)
    pf = P.compute_pairing_features(p4, normalize=False)
    s_had = pf["s_had"][0]                # (2,)
    # Pairing 1: hadronic_b = b -> on shell, s_had ~ 0.
    assert abs(float(s_had[0])) < 1e-3, f"s_had^(1) should be ~0, got {s_had[0]}"
    # Pairing 2: hadronic_b = b_bar -> off shell, s_had != 0.
    assert abs(float(s_had[1])) > 10.0, f"s_had^(2) should be far from 0, got {s_had[1]}"


def test_pdg_sentinels_route_to_pairing_slots():
    """Sentinel PDG codes +100/-100 map to the two extra flavor slots."""
    from paper1_demo.gatr_lite.gatr.model import PDG_TO_INDEX
    assert PDG_TO_INDEX[PDG_PAIRING_1] == 6
    assert PDG_TO_INDEX[PDG_PAIRING_2] == 7


def test_normalised_features_are_bounded():
    """All normalised pairing features stay in O(1) range on realistic events.

    Generates 1000 random events with parton-scale 4-momenta (E ~ 100-500 GeV,
    |p3| ~ E * unit) and checks that every normalised feature category sits
    within physically motivated bounds. Used as a guardrail against future
    regressions of the pairing-cache normalisation.
    """
    rng = np.random.default_rng(SEED)
    B = 1000
    # Energies uniform in [100, 500] GeV; 3-momentum magnitude up to E.
    E = rng.uniform(100.0, 500.0, size=(B, 6))
    direc = rng.standard_normal((B, 6, 3))
    direc = direc / np.linalg.norm(direc, axis=-1, keepdims=True)
    pmag = E * rng.uniform(0.0, 1.0, size=(B, 6))
    p3 = direc * pmag[..., None]
    p4 = np.concatenate([E[..., None], p3], axis=-1)
    p4 = torch.from_numpy(p4).to(DTYPE)

    pf = P.compute_pairing_features(p4, normalize=True)

    def _bound(name: str, max_abs: float) -> None:
        v = pf[name]
        m = v.abs().max().item()
        assert m < max_abs, (
            f"{name}: max|.|={m:.3e} exceeds bound {max_abs}; "
            f"min={v.min().item():.3e} mean={v.float().mean().item():.3e} "
            f"std={v.float().std().item():.3e}"
        )

    # tanh-bounded by construction; allow slack just to catch sign issues.
    _bound("s_had", 1.5)
    _bound("s_lep", 1.5)
    # sigma / m_t^2: realistic events run up to ~30-40 (very off-shell tops).
    _bound("sigma_p_had", 100.0)
    _bound("sigma_m_had", 100.0)
    _bound("sigma_p_lep", 100.0)
    _bound("sigma_m_lep", 100.0)
    # mass_W / M_W: at most a few (W virtual mass at parton scale).
    _bound("mass_W_had", 20.0)
    _bound("mass_W_lep", 20.0)
    # T components / E_NORM^3: each component is of order O(1), some up to ~50
    # for very forward events.
    _bound("T_pair", 100.0)
    _bound("T_dual_pair", 100.0)


def test_wedge3_antisymmetry():
    """wedge3 is fully antisymmetric in its three arguments."""
    rng = np.random.default_rng(SEED)
    a = torch.from_numpy(rng.standard_normal(16)).to(DTYPE)
    b = torch.from_numpy(rng.standard_normal(16)).to(DTYPE)
    c = torch.from_numpy(rng.standard_normal(16)).to(DTYPE)
    # Project to grade-1 to keep inputs in the expected space.
    g1 = list(A.GRADE_INDICES[1])
    mask = torch.zeros(16, dtype=DTYPE)
    mask[g1] = 1.0
    a, b, c = a * mask, b * mask, c * mask

    abc = P.wedge3(a, b, c)
    bac = P.wedge3(b, a, c)
    acb = P.wedge3(a, c, b)
    err1 = (abc + bac).abs().max().item()
    err2 = (abc + acb).abs().max().item()
    assert err1 < 1e-10, f"wedge3 not antisymmetric in (a,b): {err1}"
    assert err2 < 1e-10, f"wedge3 not antisymmetric in (b,c): {err2}"
    # And vanishes when two args coincide.
    self_zero = P.wedge3(a, a, c).abs().max().item()
    assert self_zero < 1e-10, f"wedge3(a,a,c) not zero: {self_zero}"
