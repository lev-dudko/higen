"""Verify --reference-mode as it is actually wired through the config.

test_hlhc_reference.py proved the physics using a hand-written subclass. This
checks the shipped code path: GATrLiteConfig(reference_mode=...) must reproduce
the same behaviour, or the ladder rungs A5v and A5b would train something other
than what the paper claims they train.

Expected, from the (t,z)-plane argument:
  vectors   gamma_0 + gamma_3 -- 2 extra tokens, breaks the longitudinal boost
  bivector  gamma_03          -- 1 extra token,  invariant under both H_LHC
                                 generators to float64
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


def rotor(plane, value):
    B = torch.zeros(16, dtype=torch.float64)
    tgt = tuple(sorted(plane))
    for i in A.GRADE_INDICES[2]:
        if BASIS[i] == tgt:
            B[i] = value
    return A.mv_exp(B / 2.0)


def to_mv(p4):
    out = torch.zeros(*p4.shape[:-1], 16, dtype=p4.dtype)
    for mu in range(4):
        out[..., A.GRADE1_INDICES[mu]] = p4[..., mu]
    return out


def from_mv(mv):
    return torch.stack([mv[..., A.GRADE1_INDICES[mu]] for mu in range(4)], dim=-1)


def test_reference_mode_is_wired_as_documented():
    p4 = random_p4()
    TRANSFORMS = {
        "generic boost (x-t)":     rotor((0, 1), torch.tensor(0.6, dtype=torch.float64)),
        "longitudinal boost (z)":  rotor((0, 3), torch.tensor(0.8, dtype=torch.float64)),
        "azimuthal rotation":      rotor((1, 2), torch.tensor(1.1, dtype=torch.float64)),
    }

    print(f"{'reference_mode':16s} {'tokens':>7s} {'params':>8s} " +
          " ".join(f"{k:>24s}" for k in TRANSFORMS))
    res = {}
    for mode in ("vectors", "bivector"):
        torch.manual_seed(1)
        cfg = GATrLiteConfig(use_pairing_cache=False, use_join_block=False,
                             reference_tokens=True, reference_mode=mode)
        m = GATrLite(cfg).double().eval()
        with torch.no_grad():
            ntok = m.embed(p4, PDG).shape[1]
            base = m(p4, PDG)
            row = []
            for R in TRANSFORMS.values():
                pr = from_mv(A.apply_rotor(to_mv(p4), R))
                row.append((m(pr, PDG) - base).abs().max().item())
        res[mode] = row
        print(f"{mode:16s} {ntok:7d} {m.num_parameters():8d} " +
              " ".join(f"{v:24.2e}" for v in row))

    FLOOR = 1e-12
    assert res["vectors"][1] > FLOOR, \
        f"vectors mode should break the longitudinal boost, got {res['vectors'][1]:.2e}"
    assert res["bivector"][1] < FLOOR, \
        f"bivector mode must be invariant under the longitudinal boost, got {res['bivector'][1]:.2e}"
    assert res["bivector"][2] < FLOOR, \
        f"bivector mode must be invariant under azimuthal rotation, got {res['bivector'][2]:.2e}"
    assert res["bivector"][0] > FLOOR, \
        "bivector mode should still break a generic boost (it fixes the lab (t,z) plane)"

    print("\nOK: --reference-mode is wired correctly.")
    print("    vectors  breaks H_LHC (the ingredients of p_T are supplied, not the invariance)")
    print("    bivector realises H_LHC by construction, and breaks only boosts outside it")


if __name__ == "__main__":  # also runnable as a plain script
    test_reference_mode_is_wired_as_documented()
