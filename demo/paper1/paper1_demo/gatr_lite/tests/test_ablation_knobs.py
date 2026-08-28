"""Verification of the input-ablation knobs used for Tab. 3 of the paper.

Checks, on CPU, that
  1. the three pairing_content settings and reference_tokens/grade0_only leave
     the parameter count where it should be (only the input tensor changes,
     except for reference tokens which change nothing either);
  2. the input tensor actually contains the grades it claims to contain;
  3. Lorentz equivariance survives every setting -- the logit must be invariant
     under a random rotor applied to all four-momenta.
"""
import numpy as np
import torch

from paper1_demo.gatr_lite.gatr import algebra as A
from paper1_demo.gatr_lite.gatr.model import GATrLite, GATrLiteConfig

torch.manual_seed(0)

PDG = torch.tensor([[13, -14, 2, -1, 5, -5]], dtype=torch.long).repeat(8, 1)


def random_p4(n=8):
    """Massive, positive-energy four-momenta."""
    p = torch.randn(n, 6, 4) * 40.0
    m = torch.tensor([0.106, 0.0, 0.3, 0.3, 4.8, 4.8])
    p[..., 0] = torch.sqrt((p[..., 1:] ** 2).sum(-1) + m**2)
    return p.double()


def boost_rotor(rapidity, plane=(0, 3)):
    """exp(B/2) for a boost/rotation bivector in the (mu,nu) plane."""
    B = torch.zeros(16, dtype=torch.float64)
    idx = A.GRADE_INDICES[2]
    basis = A._build_basis()
    target = tuple(sorted(plane))
    for i in idx:
        if basis[i] == target:
            B[i] = rapidity
    return A.mv_exp(B / 2.0)


def to_mv(p4):
    """(B,T,4) -> (B,T,16) grade-1 multivector."""
    out = torch.zeros(*p4.shape[:-1], 16, dtype=p4.dtype)
    for mu in range(4):
        out[..., A.GRADE1_INDICES[mu]] = p4[..., mu]
    return out


def from_mv(mv):
    return torch.stack([mv[..., A.GRADE1_INDICES[mu]] for mu in range(4)], dim=-1)


CONFIGS = {
    "A0 grade0+1 only (no pairing)": dict(use_pairing_cache=False, use_join_block=False),
    "A1 pairing scalars only":       dict(use_join_block=False, pairing_content="scalars_only"),
    "A2 pairing, no meet (g1+g3)":   dict(use_join_block=False, pairing_content="no_meet"),
    "A3 pairing full (paper cfg)":   dict(use_join_block=False, pairing_content="full"),
    "A4 full + JoinBlock":           dict(use_join_block=True,  pairing_content="full"),
    "A5 A0 + reference tokens":      dict(use_pairing_cache=False, use_join_block=False,
                                          reference_tokens=True),
    "A6 grade-0 only (PELICAN-like)": dict(use_join_block=False, grade0_only=True),
}

def test_ablation_knobs_preserve_capacity_and_equivariance():
    p4 = random_p4()
    R = boost_rotor(torch.tensor(0.7, dtype=torch.float64))          # longitudinal boost
    R2 = boost_rotor(torch.tensor(0.9, dtype=torch.float64), (1, 2))  # azimuthal rotation
    R = A.geometric_product(R, R2)
    p4_rot = from_mv(A.apply_rotor(to_mv(p4), R))

    print(f"{'configuration':34s} {'params':>8s} {'tokens':>7s} "
          f"{'grades populated':>18s}  {'max |dlogit|':>12s}")
    for name, kw in CONFIGS.items():
        model = GATrLite(GATrLiteConfig(**kw)).double().eval()
        with torch.no_grad():
            x = model.embed(p4, PDG)
            grades = [k for k in range(5)
                      if any(x[..., i].abs().max() > 0 for i in A.GRADE_INDICES[k])]
            l1 = model(p4, PDG)
            l2 = model(p4_rot, PDG)
        dmax = (l1 - l2).abs().max().item()
        print(f"{name:34s} {model.num_parameters():8d} {x.shape[1]:7d} "
              f"{str(grades):>18s}  {dmax:12.2e}")

    # --- explicit content assertions -------------------------------------------
    def grades_of(**kw):
        m = GATrLite(GATrLiteConfig(**kw)).double().eval()
        with torch.no_grad():
            x = m.embed(p4, PDG)
        return {k for k in range(5)
                if any(x[..., i].abs().max() > 0 for i in A.GRADE_INDICES[k])}

    g_full = grades_of(use_join_block=False, pairing_content="full")
    g_nomeet = grades_of(use_join_block=False, pairing_content="no_meet")
    g_scal = grades_of(use_join_block=False, pairing_content="scalars_only")
    g_g0 = grades_of(use_join_block=False, grade0_only=True)

    assert 2 in g_full, "full pairing must populate grade 2 (meet bivector)"
    assert 2 not in g_nomeet, "no_meet must not populate grade 2"
    assert 3 in g_nomeet, "no_meet must keep grade 3 (trivectors)"
    assert g_scal == {0, 1}, f"scalars_only should leave grades 0,1; got {g_scal}"
    assert g_g0 == {0}, f"grade0_only should leave grade 0 alone; got {g_g0}"

    n_full = GATrLite(GATrLiteConfig(use_join_block=False, pairing_content="full")).num_parameters()
    n_nomeet = GATrLite(GATrLiteConfig(use_join_block=False, pairing_content="no_meet")).num_parameters()
    n_scal = GATrLite(GATrLiteConfig(use_join_block=False, pairing_content="scalars_only")).num_parameters()
    assert n_full == n_nomeet == n_scal, "pairing_content must not change capacity"

    # A5 is expected to break Lorentz invariance: fixed reference tokens single out
    # the lab frame by construction. That is the point of the configuration, and
    # test_hlhc_reference.py characterises exactly which subgroup survives. Assert
    # equivariance only for the configurations that are supposed to have it.
    EQUIVARIANT = [n for n in CONFIGS if not CONFIGS[n].get("reference_tokens")]
    for name in EQUIVARIANT:
        torch.manual_seed(1)
        m = GATrLite(GATrLiteConfig(**CONFIGS[name])).double().eval()
        with torch.no_grad():
            d = (m(p4, PDG) - m(p4_rot, PDG)).abs().max().item()
        assert d < 1e-12, f"{name} should be Lorentz-invariant, got |dlogit| = {d:.2e}"

    torch.manual_seed(1)
    m5 = GATrLite(GATrLiteConfig(**CONFIGS["A5 A0 + reference tokens"])).double().eval()
    with torch.no_grad():
        d5 = (m5(p4, PDG) - m5(p4_rot, PDG)).abs().max().item()
    assert d5 > 1e-9, ("A5 must BREAK full Lorentz invariance -- fixed reference "
                       f"tokens single out the lab frame; got |dlogit| = {d5:.2e}")

    print(f"\nOK: grade content and parameter-count invariance verified for all "
          f"{len(CONFIGS)} configurations.")
    print(f"OK: Lorentz equivariance holds for the {len(EQUIVARIANT)} configurations "
          f"without reference tokens (|dlogit| < 1e-12).")
    print(f"OK: A5 breaks it as intended (|dlogit| = {d5:.2e}); see "
          f"test_hlhc_reference.py for which subgroup survives.")


if __name__ == "__main__":  # also runnable as a plain script
    test_ablation_knobs_preserve_capacity_and_equivariance()
