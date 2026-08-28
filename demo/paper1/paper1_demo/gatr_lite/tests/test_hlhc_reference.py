"""What symmetry do the reference tokens actually leave? (Report #1, point 11)

Fixed reference tokens deliberately break full Spin+(1,3) invariance -- that is
their purpose. The question this script answers is *what is left*, because the
answer determines what we may claim in the paper.

Three transformations are applied to the physical four-momenta while the
reference tokens are held fixed in the lab frame:

  * a generic boost (mixes all four directions),
  * a longitudinal boost along z    -- an element of H_LHC,
  * an azimuthal rotation about z   -- an element of H_LHC.

For each we report the change in the logit. Expectation, to be confirmed:

  gamma_0 + gamma_3 tokens : NOT invariant under either H_LHC generator, because
      <T_i, gamma_0> = E_i and <T_i, gamma_3> = p_z,i both change under a
      longitudinal boost. The tokens supply the *ingredients* of p_T, y, phi;
      invariance would have to be learned, not enforced.

  gamma_03 bivector token : invariant under BOTH H_LHC generators, because the
      (t,z) plane is preserved by longitudinal boosts and untouched by azimuthal
      rotations. This is the token that enforces H_LHC-invariance by construction.
"""
import torch

from paper1_demo.gatr_lite.gatr import algebra as A
from paper1_demo.gatr_lite.gatr.model import GATrLite, GATrLiteConfig

torch.manual_seed(0)
PDG = torch.tensor([[13, -14, 2, -1, 5, -5]], dtype=torch.long).repeat(8, 1)
BASIS = A._build_basis()


def random_p4(n=8):
    p = torch.randn(n, 6, 4) * 40.0
    m = torch.tensor([0.106, 0.0, 0.3, 0.3, 4.8, 4.8])
    p[..., 0] = torch.sqrt((p[..., 1:] ** 2).sum(-1) + m**2)
    return p.double()


def bivector(plane, value):
    B = torch.zeros(16, dtype=torch.float64)
    target = tuple(sorted(plane))
    for i in A.GRADE_INDICES[2]:
        if BASIS[i] == target:
            B[i] = value
    return B


def rotor(plane, value):
    return A.mv_exp(bivector(plane, value) / 2.0)


def to_mv(p4):
    out = torch.zeros(*p4.shape[:-1], 16, dtype=p4.dtype)
    for mu in range(4):
        out[..., A.GRADE1_INDICES[mu]] = p4[..., mu]
    return out


def from_mv(mv):
    return torch.stack([mv[..., A.GRADE1_INDICES[mu]] for mu in range(4)], dim=-1)


class RefBivectorModel(GATrLite):
    """Variant whose single reference token carries the gamma_03 bivector."""

    def _append_reference_tokens(self, out):
        B, T, C, _ = out.shape
        ref = torch.zeros(B, 1, C, 16, dtype=out.dtype, device=out.device)
        ref[:, 0, 0, :] = bivector((0, 3), 1.0).to(out.dtype).to(out.device)
        return torch.cat([out, ref], dim=1)


def test_reference_bivector_realises_the_collider_symmetry():
    p4 = random_p4()
    TRANSFORMS = {
        "generic boost (x-t, y=0.6)": rotor((0, 1), torch.tensor(0.6, dtype=torch.float64)),
        "longitudinal boost (z, y=0.8)": rotor((0, 3), torch.tensor(0.8, dtype=torch.float64)),
        "azimuthal rotation (phi=1.1)": rotor((1, 2), torch.tensor(1.1, dtype=torch.float64)),
    }

    MODELS = {
        "no reference tokens": (GATrLite, dict(use_pairing_cache=False, use_join_block=False)),
        "gamma_0 + gamma_3 tokens": (GATrLite, dict(use_pairing_cache=False, use_join_block=False,
                                                    reference_tokens=True)),
        "gamma_03 bivector token": (RefBivectorModel, dict(use_pairing_cache=False, use_join_block=False,
                                                           reference_tokens=True)),
    }

    print(f"{'model':28s} " + " ".join(f"{k:>30s}" for k in TRANSFORMS))
    for name, (cls, kw) in MODELS.items():
        torch.manual_seed(1)
        m = cls(GATrLiteConfig(**kw)).double().eval()
        row = []
        with torch.no_grad():
            base = m(p4, PDG)
            for R in TRANSFORMS.values():
                pr = from_mv(A.apply_rotor(to_mv(p4), R))
                row.append((m(pr, PDG) - base).abs().max().item())
        print(f"{name:28s} " + " ".join(f"{v:30.2e}" for v in row))

    print("\nInterpretation: a value at the 1e-17 level is exact invariance (float64 noise);")
    print("anything at 1e-2 or above is a genuine symmetry breaking.")


if __name__ == "__main__":  # also runnable as a plain script
    test_reference_bivector_realises_the_collider_symmetry()
