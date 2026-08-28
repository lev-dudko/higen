#!/usr/bin/env python3
"""Where does the non-zero pseudoscalar sign asymmetry come from?

measure_grade_physics.py found frac(eps > 0) departing from 1/2 by up to 12
sigma for some four-particle combinations. At tree level with real amplitudes a
naive-T-odd observable like eps(p_i,p_j,p_k,p_l) must average to zero, so a
12-sigma asymmetry is either (a) not what it looks like, or (b) an artefact.

Three candidate explanations, each tested here:

 1. LABEL ORDERING. eps is antisymmetric, so eps(a,b,c,d) = -eps(b,a,c,d). If the
    canonical parton order correlates with anything physical -- and it does, the
    slots are (mu, nubar, u, dbar, b, bbar), i.e. fixed by PDG identity -- then a
    non-zero mean sign is a statement about a genuine kinematic correlation, not
    about CP. Test: recompute with the two same-W partons swapped; the asymmetry
    must flip sign exactly.

 2. A GENUINE T-ODD CORRELATION WITHOUT CPV. Naive-T-odd observables can be
    non-zero from absorptive parts, but not at tree level. However eps built from
    four momenta of DISTINGUISHABLE partons is not T-odd at all once the labels
    are fixed by particle identity: it is the signed 4-volume, and its sign is
    correlated with the spin-correlation structure of the decay. Test: check
    whether the asymmetry survives when the two W decay products are ordered by
    energy instead of by PDG -- an energy ordering is a P-even, T-even
    prescription, so a real T-odd effect would survive and a labelling effect
    would not.

 3. NUMERICAL. The determinant of four nearly-degenerate momenta is a small
    difference of large numbers. Test: compare float32 (as stored) against
    float64 and against a mean-subtracted, scale-normalised evaluation.

Usage:
    python test_cp_asymmetry.py --data-dir DIR
"""
from __future__ import annotations

import argparse
import logging

import h5py
import numpy as np

LOG = logging.getLogger("cp-asym")
MU, NU, U, DBAR, B, BBAR = range(6)


def eps(p4, idx):
    return np.linalg.det(p4[:, list(idx), :])


def frac_pos(x):
    f = float((x > 0).mean())
    e = float(np.sqrt(f * (1 - f) / len(x)))
    return f, e, (f - 0.5) / e


def report(tag, x):
    f, e, pull = frac_pos(x)
    LOG.info("%-46s frac(+) = %.6f +- %.6f  (%+6.2f sigma)", tag, f, e, pull)
    return pull


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--file", default="dr1.h5")
    p.add_argument("--n", type=int, default=400_000)
    args = p.parse_args(argv)

    with h5py.File(f"{args.data_dir}/{args.file}", "r") as hf:
        n = min(args.n, hf["p4"].shape[0])
        p32 = np.asarray(hf["p4"][:n])                    # float32 as stored
    p64 = p32.astype(np.float64)
    LOG.info("%s: %d events, stored dtype %s", args.file, n, p32.dtype)

    LOG.info("\n--- 1. label ordering: swapping the two W+ partons must flip the sign ---")
    base = eps(p64, (MU, NU, U, DBAR))
    swap = eps(p64, (MU, NU, DBAR, U))
    report("eps(mu, nubar, u, dbar)", base)
    report("eps(mu, nubar, dbar, u)  [u,dbar swapped]", swap)
    LOG.info("exact antisymmetry: max|eps + eps_swapped| = %.3e", np.abs(base + swap).max())

    LOG.info("\n--- 2. energy ordering instead of PDG identity ---")
    # order the two W+ partons by energy, a P- and T-even prescription
    e_u, e_d = p64[:, U, 0], p64[:, DBAR, 0]
    hi = np.where(e_u >= e_d, U, DBAR)
    lo = np.where(e_u >= e_d, DBAR, U)
    idx = np.arange(len(p64))
    p_hi, p_lo = p64[idx, hi, :], p64[idx, lo, :]
    stacked = np.stack([p64[:, MU, :], p64[:, NU, :], p_hi, p_lo], axis=1)
    report("eps(mu, nubar, E-ordered W+ partons)", np.linalg.det(stacked))
    LOG.info("   (a PDG-labelling effect vanishes here; a T-odd effect survives)")

    LOG.info("\n--- 3. numerical precision ---")
    report("float64 (reference)", base)
    report("float32 determinant", eps(p32.astype(np.float32), (MU, NU, U, DBAR)).astype(np.float64))
    scale = np.abs(p64).max(axis=(1, 2), keepdims=True)
    report("scale-normalised float64", eps(p64 / scale, (MU, NU, U, DBAR)))
    LOG.info("median |eps| = %.4e, median |eps| / (max|p|^4) = %.4e",
             float(np.median(np.abs(base))),
             float(np.median(np.abs(base) / scale[:, 0, 0] ** 4)))

    LOG.info("\n--- 4. is the asymmetry in the sign, or in the magnitude? ---")
    pos, neg = base[base > 0], base[base < 0]
    LOG.info("n(+) = %d, n(-) = %d", len(pos), len(neg))
    LOG.info("median |eps| for eps>0: %.4e   for eps<0: %.4e",
             float(np.median(pos)), float(np.median(-neg)))
    LOG.info("mean eps / std eps = %.5f  (a genuine CP asymmetry shifts the MEAN)",
             float(base.mean() / base.std()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
