"""Pairing-aware feature cache for GATr-lite (top-pair classification).

Final state ``pp -> 6 partons`` in canonical order::

    [mu-, nu_mu_bar, u, d_bar, b, b_bar]
       0       1     2    3    4   5

Topology (semileptonic ttbar / DR1):

    Hadronic side : W+ -> u d_bar,  t  -> W+ b
    Leptonic side : W- -> mu- nu_,  t_ -> W- b_bar

The network does not know a priori which b goes to which side. We therefore
expose, for each of the two candidate b-pairings ``a in {1, 2}``::

    pairing 1 : hadronic_b = b,    leptonic_b = b_bar
    pairing 2 : hadronic_b = b_bar,leptonic_b = b

a set of Lorentz-covariant derived features:

* W_had = p_u + p_dbar,           W_lep = p_mu + p_nu
* s_had^(a) = ((W_had + b_had,a)^2 - m_t^2) / Gamma_t   (Breit-Wigner shifted)
* s_lep^(a) = ((W_lep + b_lep,a)^2 - m_t^2) / Gamma_t
* sigma+_Wb^(had) = 1/2 [(W_had + b)^2 + (W_had + bbar)^2]   (pairing-symmetric)
* sigma-_Wb^(had) = 1/2 [(W_had + b)^2 - (W_had + bbar)^2]   (pairing-asymmetric)
* analogous sigma+/-^(lep)
* m_W_had = sqrt(W_had^2),  m_W_lep = sqrt(W_lep^2)  (clipped at 0)

Plus two Lorentz multivectors per pairing:

* T^(a) = p_u  ^  p_dbar  ^  p_{b_had,a}    -- grade-3 trivector
* *T^(a) = T^(a) * e_0123                   -- grade-1 Hodge dual

These multivectors are used by the model to feed extra "pairing tokens"
into the equivariant backbone without breaking equivariance: scalar features
are Lorentz-invariant, T^(a) transforms as a trivector under the rotor
sandwich, *T^(a) transforms as a vector.

Constants
---------
m_t     = 173.0 GeV       (top mass, PDG-ish)
Gamma_t = 1.4   GeV       (top width)

All p4 are contravariant 4-momenta in the (+, -, -, -) metric, i.e. mass^2
is computed as ``E^2 - px^2 - py^2 - pz^2``.
"""

from __future__ import annotations

from typing import Dict

import torch

from . import algebra as A


# Top mass and width pinned to the values used in the paper / generator card.
M_T: float = 173.0       # GeV (PDG t-quark mass)
GAMMA_T: float = 1.4     # GeV (top width)
M_W: float = 80.4        # GeV (PDG W mass)
E_NORM: float = 100.0    # GeV (typical parton energy at 14 TeV partonic CM)

# Numerical-stability normalisation strategy
# ------------------------------------------
# The previous version used plain linear rescaling (SCALE_S_HAD=100,
# SCALE_T=1e5, ...) which left features in O(10^2) and was insufficient:
# off-shell events drove (W+b)^2-m_t^2 into the 10^6 GeV^2 range, scalar
# inputs reached ~1e4, geometric products squared that to ~1e8, and the
# loss exploded (smoke v1: NaN, v2: ~1e10).
#
# We now use:
#   * tanh-clipped Breit-Wigner scaling for s_had/s_lep
#         s_norm = tanh((m^2 - m_t^2) / (Gamma_t * m_t))
#     Bounded to (-1, 1) by construction; the natural argument is
#     dimensionless after dividing by (Gamma_t * m_t) ~ 240 GeV^2 for the
#     resonant scale.
#   * sigma_+/-_Wb / m_t^2  (linear, dimensionless, ~O(1)-O(10))
#   * mass_W / M_W          (linear, ~O(1))
#   * T_pair / E_NORM^3     (linear per-component, ~O(1))
#   * T_dual_pair / E_NORM^3
#
# All transformations are scalar/element-wise rescalings of multivectors
# and therefore commute with the rotor sandwich -- equivariance preserved.
SCALE_S_DENOM: float = GAMMA_T * M_T   # ~242 GeV^2, BW resonance scale
SCALE_SIGMA: float = M_T * M_T         # 29929 GeV^2, top mass squared
SCALE_MASS_W: float = M_W              # 80.4 GeV
# T_pair lives in GeV^3 with components reaching ~(2*E_NORM)^3 in the worst
# events at 14 TeV (high-pT partons). We divide by (2*E_NORM)^3 = 8e6 to put
# the typical magnitude in O(0.1) and the q99 well under 1.
SCALE_T: float = (2.0 * E_NORM) ** 3   # 8e6 GeV^3
# Group A extensions (meet, coplanarity, pseudoscalar): wedge-derived
# multivectors live in higher-energy units; we normalise to keep typical
# components at O(0.1)-O(1).
SCALE_MEET: float = (2.0 * E_NORM) ** 6     # ~ 6.4e13 GeV^6 (grade-2 meet of two 3-blades)
SCALE_PS4: float = (2.0 * E_NORM) ** 4      # ~ 1.6e9 GeV^4 (4-blade pseudoscalar)
SCALE_COPLAN: float = (2.0 * E_NORM) ** 12  # ~ 4.1e27 GeV^12 (<reverse(meet) meet>_0)

# Canonical indices in the (B, 6, ...) parton tensor.
IDX_MU: int = 0          # mu-
IDX_NU: int = 1          # nu_mu_bar
IDX_U: int = 2           # u
IDX_DBAR: int = 3        # d_bar
IDX_B: int = 4           # b
IDX_BBAR: int = 5        # b_bar

# Pseudoscalar index in the 16-element multivector basis (e_0123).
_PSEUDOSCALAR_IDX: int = 15


# ---------------------------------------------------------------------------
# Multivector helpers (torch backend)
# ---------------------------------------------------------------------------


def _signed_log1p(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log compression: sign(x) * log1p(|x|).

    Linear near zero, logarithmic in the tails. Used as a soft-clip on
    scalar pairing features (sigma_+/-_Wb) to tame heavy-tail outliers
    without saturating like tanh; safely differentiable for backprop.
    """
    return torch.sign(x) * torch.log1p(x.abs())


def _grade1_lift(p4: torch.Tensor) -> torch.Tensor:
    """Lift a 4-momentum (..., 4) into a grade-1 multivector (..., 16)."""
    out = torch.zeros(*p4.shape[:-1], 16, dtype=p4.dtype, device=p4.device)
    for mu in range(4):
        out[..., A.GRADE1_INDICES[mu]] = p4[..., mu]
    return out


def _project_grade(x: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all components of x not in grade k. Last axis must be size 16."""
    out = torch.zeros_like(x)
    for i in A.GRADE_INDICES[k]:
        out[..., i] = x[..., i]
    return out


def wedge3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Antisymmetric wedge product of three grade-1 multivectors.

    Inputs ``a``, ``b``, ``c`` are multivectors of shape (..., 16) that are
    expected to live in grade 1 (their other graded components are ignored
    via the final grade-3 projection). The output is a pure grade-3
    multivector of shape (..., 16) with all non-grade-3 components zero.

    Implementation: ``a wedge b wedge c = grade3( gp(gp(a, b), c) )``. For
    pure grade-1 inputs this is equivalent to the antisymmetrised tensor
    product, but using the Cayley table keeps everything equivariant under
    the rotor sandwich (since gp is equivariant and grade projection commutes
    with rotors).
    """
    ab = A.geometric_product(a, b)
    abc = A.geometric_product(ab, c)
    return _project_grade(abc, 3)


def hodge_dual(x: torch.Tensor) -> torch.Tensor:
    """Hodge dual via right-multiplication by the pseudoscalar ``e_0123``.

    Maps grade-k components to grade-(4-k). Concretely, a pure trivector
    ``T = sum_alpha T^alpha e_alpha`` (alpha in {012, 013, 023, 123}) is
    sent to a pure vector ``*T``. Sign factors come from the Cayley table
    in the (+,-,-,-) signature.
    """
    pseudo = torch.zeros(16, dtype=x.dtype, device=x.device)
    pseudo[_PSEUDOSCALAR_IDX] = 1.0
    return A.geometric_product(x, pseudo)


# ---------------------------------------------------------------------------
# Lorentz invariants
# ---------------------------------------------------------------------------


def _dot4(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Lorentz dot product (+,-,-,-): p.q = E_p E_q - p.q (3-vector dot)."""
    return p[..., 0] * q[..., 0] - (p[..., 1] * q[..., 1]
                                     + p[..., 2] * q[..., 2]
                                     + p[..., 3] * q[..., 3])


def _mass_sq(p: torch.Tensor) -> torch.Tensor:
    """p^2 = E^2 - |vec p|^2 in (+,-,-,-)."""
    return _dot4(p, p)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_pairing_features(
    p4: torch.Tensor,
    pdg: torch.Tensor | None = None,
    normalize: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute the pairing-aware feature cache for a batch of events.

    Parameters
    ----------
    p4 : torch.Tensor, shape (B, 6, 4)
        Final-state 4-momenta in canonical order
        ``[mu-, nu_mu_bar, u, d_bar, b, b_bar]`` with metric (+,-,-,-).
    pdg : torch.Tensor, shape (B, 6), optional
        Currently unused (canonical order is assumed); kept for symmetry
        with the model entry point.

    Returns
    -------
    dict with keys::

        s_had          : (B, 2)  Breit-Wigner shifted hadronic-top virtuality
        s_lep          : (B, 2)  Breit-Wigner shifted leptonic-top virtuality
        sigma_p_had    : (B,)    1/2 [(W_had + b)^2 + (W_had + bbar)^2]
        sigma_m_had    : (B,)    1/2 [(W_had + b)^2 - (W_had + bbar)^2]
        sigma_p_lep    : (B,)    1/2 [(W_lep + b)^2 + (W_lep + bbar)^2]
        sigma_m_lep    : (B,)    1/2 [(W_lep + b)^2 - (W_lep + bbar)^2]
        mass_W_had     : (B,)    sqrt(max(W_had^2, 0))
        mass_W_lep     : (B,)    sqrt(max(W_lep^2, 0))
        T_pair         : (B, 2, 16)  trivector p_u ^ p_dbar ^ p_{b_had,a}
        T_dual_pair    : (B, 2, 16)  Hodge dual of T_pair (grade-1)

    Notes
    -----
    All scalar quantities are Lorentz-invariants (built from p.q and p^2).
    ``T_pair`` is a Lorentz-covariant trivector; ``T_dual_pair`` is a
    Lorentz-covariant vector. Feeding them into an equivariant model as
    extra tokens preserves the rotor-sandwich equivariance.
    """
    if p4.dim() != 3 or p4.shape[-1] != 4 or p4.shape[1] != 6:
        raise ValueError(
            f"p4 must have shape (B, 6, 4); got {tuple(p4.shape)}"
        )

    p_mu = p4[:, IDX_MU]            # (B, 4)
    p_nu = p4[:, IDX_NU]
    p_u = p4[:, IDX_U]
    p_db = p4[:, IDX_DBAR]
    p_b = p4[:, IDX_B]
    p_bb = p4[:, IDX_BBAR]

    # W candidates.
    W_had = p_u + p_db                # u + d_bar
    W_lep = p_mu + p_nu               # mu- + nu_bar

    # (W + b)^2 and (W + bbar)^2 for each side.
    Wb_had_b = _mass_sq(W_had + p_b)        # (B,)
    Wb_had_bb = _mass_sq(W_had + p_bb)
    Wb_lep_b = _mass_sq(W_lep + p_b)
    Wb_lep_bb = _mass_sq(W_lep + p_bb)

    # Breit-Wigner-shifted virtuality. Raw form is (m^2 - m_t^2) / Gamma_t in
    # GeV; that gets ~1e4 for off-shell tops and explodes downstream linear
    # layers. The normalised form below applies tanh on the resonance-scaled
    # argument, bounding the feature to (-1, 1) regardless of how off-shell
    # the event is; for on-shell tops it is ~ (m^2-m_t^2)/(Gamma_t*m_t), i.e.
    # essentially linear in the BW argument near the pole.
    m_t2 = M_T * M_T
    inv_gamma = 1.0 / GAMMA_T

    # Pairing 1: hadronic_b = b,     leptonic_b = b_bar
    # Pairing 2: hadronic_b = b_bar, leptonic_b = b
    s_had_1 = (Wb_had_b - m_t2) * inv_gamma
    s_had_2 = (Wb_had_bb - m_t2) * inv_gamma
    s_lep_1 = (Wb_lep_bb - m_t2) * inv_gamma
    s_lep_2 = (Wb_lep_b - m_t2) * inv_gamma

    s_had = torch.stack([s_had_1, s_had_2], dim=-1)   # (B, 2)
    s_lep = torch.stack([s_lep_1, s_lep_2], dim=-1)   # (B, 2)

    # Pairing-symmetric / antisymmetric combinations (per side).
    sigma_p_had = 0.5 * (Wb_had_b + Wb_had_bb)
    sigma_m_had = 0.5 * (Wb_had_b - Wb_had_bb)
    sigma_p_lep = 0.5 * (Wb_lep_b + Wb_lep_bb)
    sigma_m_lep = 0.5 * (Wb_lep_b - Wb_lep_bb)

    # W invariant masses, clipped at zero before sqrt to keep things real-valued.
    mW2_had = _mass_sq(W_had).clamp_min(0.0)
    mW2_lep = _mass_sq(W_lep).clamp_min(0.0)
    mass_W_had = torch.sqrt(mW2_had)
    mass_W_lep = torch.sqrt(mW2_lep)

    # Trivectors: T^(a) = p_u ^ p_dbar ^ p_{b_had,a}.
    p_u_mv = _grade1_lift(p_u)
    p_db_mv = _grade1_lift(p_db)
    p_b_mv = _grade1_lift(p_b)
    p_bb_mv = _grade1_lift(p_bb)

    T1 = wedge3(p_u_mv, p_db_mv, p_b_mv)   # pairing 1: hadronic_b = b
    T2 = wedge3(p_u_mv, p_db_mv, p_bb_mv)  # pairing 2: hadronic_b = b_bar
    T_pair = torch.stack([T1, T2], dim=1)  # (B, 2, 16)

    # Hodge duals (grade-1 multivectors).
    T_dual_pair = hodge_dual(T_pair)        # (B, 2, 16) — only grade-1 nonzero.

    # Group A extensions (meet/join for pairing resolution).
    p_mu_mv = _grade1_lift(p_mu)
    p_nu_mv = _grade1_lift(p_nu)
    # Leptonic decay 3-blade for each pairing: T_lep^(a) = p_mu ∧ p_nu ∧ p_{b_lep,a}
    # where b_lep,1 = b_bar and b_lep,2 = b (other side of the b-jet swap).
    T1_lep = wedge3(p_mu_mv, p_nu_mv, p_bb_mv)  # pairing 1: leptonic_b = b_bar
    T2_lep = wedge3(p_mu_mv, p_nu_mv, p_b_mv)   # pairing 2: leptonic_b = b
    T_lep_pair = torch.stack([T1_lep, T2_lep], dim=1)             # (B, 2, 16) grade-3
    T_lep_dual_pair = A.hodge_dual(T_lep_pair)                     # (B, 2, 16) grade-1

    # Common Lorentz subspace between hadronic and leptonic decay planes:
    # meet(T_had^(a), T_lep^(a)) is a grade-2 multivector ("2-blade") that is
    # nonzero whenever the two decay planes share a Lorentz subspace; its
    # invariant norm is a coplanarity indicator.
    M1 = A.meet(T1, T1_lep, target_grade=2)
    M2 = A.meet(T2, T2_lep, target_grade=2)
    meet_pair = torch.stack([M1, M2], dim=1)                       # (B, 2, 16) grade-2

    # Coplanarity scalar: <reverse(M) M>_0 (Lorentz invariant).
    coplanarity = A.inner_product(meet_pair, meet_pair)            # (B, 2)

    # Sign of the 4-blade pseudoscalar built from the 4 jet partons of the
    # event (u, d_bar, b, b_bar). This is *the same* per-event grade-4 value
    # for both pairings up to a relative sign from b<->b_bar, hence we
    # compute one signed scalar per pairing matching the pairing's parton
    # ordering.
    # ps^(a) = <p_u ∧ p_dbar ∧ p_{b_had,a} ∧ p_{b_lep,a}>_4 = T^(a) ∧ p_{b_lep,a}
    ps1 = A.wedge_to_grade(T1, p_bb_mv, target_grade=4)[..., A.PSEUDOSCALAR_IDX]
    ps2 = A.wedge_to_grade(T2, p_b_mv,  target_grade=4)[..., A.PSEUDOSCALAR_IDX]
    pseudoscalar_4 = torch.stack([ps1, ps2], dim=-1)               # (B, 2)

    if normalize:
        # Per-category dimensionless rescaling. All transforms below are
        # element-wise scalar rescalings (or per-event scalar tanh), so they
        # commute with the rotor sandwich and equivariance is preserved.
        #
        # s_had/s_lep: tanh-clip BW-shifted virtuality, bounded in (-1, 1).
        # The argument is (m^2-m_t^2)/(Gamma_t*m_t) which is dimensionless;
        # equivalently we divide the existing s_* (which has Gamma_t
        # already in the denominator) by m_t and pass through tanh.
        s_had = torch.tanh(s_had / M_T)
        s_lep = torch.tanh(s_lep / M_T)
        # sigma_+/-_Wb : raw GeV^2 -> dimensionless via /m_t^2, then squash
        # heavy-tail outliers with signed_log1p (scalar element-wise, so
        # equivariance is trivially preserved -- these are grade-0 channels).
        # signed_log1p(x) = sign(x) * log1p(|x|) keeps small-x linear and
        # compresses the q99 ~ 10 down to ~ 2.4 while clipping the worst
        # outliers (max ~ 200) to log(200) ~ 5.3.
        sigma_p_had = _signed_log1p(sigma_p_had / SCALE_SIGMA)
        sigma_m_had = _signed_log1p(sigma_m_had / SCALE_SIGMA)
        sigma_p_lep = _signed_log1p(sigma_p_lep / SCALE_SIGMA)
        sigma_m_lep = _signed_log1p(sigma_m_lep / SCALE_SIGMA)
        # m_W / M_W -> dimensionless O(1) (already well-behaved, no squash).
        mass_W_had = mass_W_had / SCALE_MASS_W
        mass_W_lep = mass_W_lep / SCALE_MASS_W
        # T_pair (grade 3) and T_dual_pair (grade 1) are GeV^3 multivectors;
        # divide by E_NORM^3 to land at O(1).
        T_pair = T_pair / SCALE_T
        T_dual_pair = T_dual_pair / SCALE_T
        # Leptonic 3-blade and dual: same units as T_pair.
        T_lep_pair = T_lep_pair / SCALE_T
        T_lep_dual_pair = T_lep_dual_pair / SCALE_T
        # meet (grade-2) is GeV^6, scale aggressively to keep magnitude tame.
        meet_pair = meet_pair / SCALE_MEET
        # coplanarity ~ GeV^12; signed_log1p compresses heavy tails.
        coplanarity = _signed_log1p(coplanarity / SCALE_COPLAN)
        # pseudoscalar_4 ~ GeV^4; signed_log1p preserves CP-odd sign.
        pseudoscalar_4 = _signed_log1p(pseudoscalar_4 / SCALE_PS4)

    return {
        "s_had": s_had,
        "s_lep": s_lep,
        "sigma_p_had": sigma_p_had,
        "sigma_m_had": sigma_m_had,
        "sigma_p_lep": sigma_p_lep,
        "sigma_m_lep": sigma_m_lep,
        "mass_W_had": mass_W_had,
        "mass_W_lep": mass_W_lep,
        "T_pair": T_pair,
        "T_dual_pair": T_dual_pair,
        "T_lep_pair": T_lep_pair,
        "T_lep_dual_pair": T_lep_dual_pair,
        "meet_pair": meet_pair,
        "coplanarity": coplanarity,
        "pseudoscalar_4": pseudoscalar_4,
    }
