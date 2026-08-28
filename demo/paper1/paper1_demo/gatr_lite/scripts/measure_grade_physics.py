#!/usr/bin/env python3
"""Single-observable discrimination power of each grade in the tWb final state.

Why this exists
---------------
The ablation ladder says the grade-2 and grade-3 pairing content does not help.
That is a fact about the network, and it invites the question of whether the
higher grades are useless in general. They are not: in THIS final state their
invariant content is either identical between the two classes or empty by
construction, and that is a statement about the physics of tWb which we can
measure directly, without any network.

Final state (canonical order):  [mu-, nubar_mu, u, dbar, b, bbar]
                                   0      1     2    3   4    5

  W- = (0,1)      leptonic W, on shell in BOTH classes
  W+ = (2,3)      hadronic W, on shell in BOTH classes
  t  = (2,3,4)    hadronic top, resonant in BOTH classes
  tbar = (0,1,5)  leptonic top, resonant ONLY in the ttbar class

So the class difference lives in one 3-particle invariant mass, and the paper's
own grade dictionary says where each object sits:

  grade 0   invariant masses and Gram entries
  grade 1   the four-momenta themselves
  grade 2   two-particle planes -> their invariant norm is m(pair)^2
  grade 3   three-particle volumes -> norm is m(triple)^2
  grade 4   the pseudoscalar, eps(p_i,p_j,p_k,p_l): the CP-odd slot

Predictions, each of which this script tests as a single-observable AUC:

  * m(W-) and m(W+): no discrimination. Both classes put the W on shell, so the
    grade-2 pair planes of the decay products carry the same invariant content
    in signal and background. Their AUC should sit at 0.5.
  * m(hadronic top): no discrimination either, resonant in both classes.
  * m(leptonic top): this IS the discriminant, a grade-3 norm -- but equally a
    grade-0 invariant reachable from the Gram matrix, which is what the collapse
    lemma says and why exposing it as a grade-3 token adds nothing.
  * the grade-4 pseudoscalar: this process has no CP violation, so its signed
    distribution must be symmetric about zero in both classes and its AUC must
    sit at 0.5. The CP-odd slot of the algebra is not empty because the
    representation is deficient; it is empty because the physics puts nothing
    in it.

Usage:
    python measure_grade_physics.py --data-dir DIR --out grade_physics.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import h5py
import numpy as np

LOG = logging.getLogger("grade-physics")

MU, NU, U, DBAR, B, BBAR = range(6)
NAMES = ["mu-", "nubar", "u", "dbar", "b", "bbar"]

# Minkowski metric with p = (E, px, py, pz)
ETA = np.array([1.0, -1.0, -1.0, -1.0])


def minkowski_mass(p: np.ndarray) -> np.ndarray:
    """Invariant mass of a summed four-momentum, (N,4) -> (N,)."""
    m2 = (p * p * ETA).sum(axis=-1)
    return np.sqrt(np.clip(m2, 0.0, None))


def mass_of(p4: np.ndarray, idx: tuple[int, ...]) -> np.ndarray:
    return minkowski_mass(p4[:, list(idx), :].sum(axis=1))


def pseudoscalar(p4: np.ndarray, idx: tuple[int, int, int, int]) -> np.ndarray:
    """eps_{mu nu rho sigma} p1^mu p2^nu p3^rho p4^sigma = det[p1;p2;p3;p4].

    This is the grade-4 coefficient of the wedge of four grade-1 vectors, i.e.
    the pseudoscalar slot of Cl(1,3): a Lorentz scalar under Spin+(1,3) that is
    odd under parity, hence the CP-odd one-bit observable of the paper's
    dictionary. Sign conventions do not matter here -- only the symmetry of the
    distribution about zero and the AUC do.
    """
    m = p4[:, list(idx), :]           # (N,4,4)
    return np.linalg.det(m)


def bivector_norm2(p4: np.ndarray, i: int, j: int) -> np.ndarray:
    """Squared norm of the grade-2 wedge p_i ^ p_j.

    <p_i ^ p_j, p_i ^ p_j> = (p_i.p_j)^2 - (p_i.p_i)(p_j.p_j), which for the
    massless partons here reduces to (p_i.p_j)^2 = (m_ij^2 / 2)^2. Reported to
    make explicit that the grade-2 invariant content of a decay pair IS the pair
    mass and nothing else.
    """
    pi, pj = p4[:, i, :], p4[:, j, :]
    dot = lambda a, b: (a * b * ETA).sum(axis=-1)
    return dot(pi, pj) ** 2 - dot(pi, pi) * dot(pj, pj)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC, ties handled by average rank. No sklearn dependency."""
    order = np.argsort(scores, kind="mergesort")
    s, y = scores[order], labels[order]
    # average ranks over tied groups
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load(path: Path, n_max: int, rng: np.random.Generator) -> np.ndarray:
    with h5py.File(path, "r") as hf:
        n = hf["p4"].shape[0]
        take = min(n_max, n)
        idx = np.sort(rng.choice(n, size=take, replace=False)) if take < n else slice(None)
        p4 = np.asarray(hf["p4"][idx], dtype=np.float64)
    LOG.info("%s: %d of %d events", path.name, len(p4), n)
    return p4


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--signal", default="dr1.h5", help="label 1: single-resonant tWb")
    p.add_argument("--background", default="tT.h5", help="label 0: double-resonant ttbar")
    p.add_argument("--n-per-class", type=int, default=400_000)
    p.add_argument("--out", default="grade_physics.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    d = Path(args.data_dir)
    sig = load(d / args.signal, args.n_per_class, rng)
    bkg = load(d / args.background, args.n_per_class, rng)
    p4 = np.concatenate([sig, bkg], axis=0)
    y = np.concatenate([np.ones(len(sig), dtype=np.int8), np.zeros(len(bkg), dtype=np.int8)])

    rows = []
    M_T, G_T = 172.5, 1.42   # top pole mass and width used by the generator
    M_W, G_W = 80.379, 2.085

    def record(grade, obj, parts, scores, note):
        a = auc(scores, y)
        # symmetric observables: |AUC - 0.5| is the discrimination, direction is
        # meaningless, so report the folded value too
        rows.append(dict(grade=grade, object=obj, particles=parts,
                         auc=round(a, 6), separation=round(abs(a - 0.5), 6),
                         median_signal=float(np.median(scores[y == 1])),
                         median_background=float(np.median(scores[y == 0])),
                         note=note))
        LOG.info("grade %s  %-26s AUC %.6f  |AUC-0.5| %.6f", grade, obj, a, abs(a - 0.5))

    # ---- grade 2: two-particle planes (decay pairs) -------------------------
    for (i, j), label in (((MU, NU), "W- plane (mu, nubar)"),
                          ((U, DBAR), "W+ plane (u, dbar)")):
        record(2, f"m({NAMES[i]},{NAMES[j]})", f"{i},{j}", mass_of(p4, (i, j)),
               "W on shell in both classes -> no invariant difference")
        record(2, f"|p_{NAMES[i]} ^ p_{NAMES[j]}|^2", f"{i},{j}",
               bivector_norm2(p4, i, j),
               "grade-2 norm of the same plane: equals (m^2/2)^2, same information")

    # a plane that is NOT a decay pair, for contrast
    record(2, "m(b,bbar)", f"{B},{BBAR}", mass_of(p4, (B, BBAR)),
           "not a resonance pair; carries production-level information")

    # ---- grade 3: three-particle volumes (top candidates) ------------------
    record(3, "m(u,dbar,b)  hadronic t", f"{U},{DBAR},{B}", mass_of(p4, (U, DBAR, B)),
           "resonant in BOTH classes -> no discrimination")
    record(3, "m(mu,nubar,bbar)  leptonic tbar", f"{MU},{NU},{BBAR}",
           mass_of(p4, (MU, NU, BBAR)),
           "THE discriminant: resonant only in the ttbar class")
    # the wrong pairing, which is what the pairing tokens were meant to help with
    record(3, "m(mu,nubar,b)  wrong pairing", f"{MU},{NU},{B}",
           mass_of(p4, (MU, NU, B)),
           "combinatorial alternative; the assignment ambiguity of the task")

    # ---- the SHAPE of the discriminant, not the invariant itself ------------
    # A resonance sits in the middle of the continuum it competes with, so the
    # invariant mass is NOT monotonic in "resonant vs not" and its AUC lands at
    # 0.5 however strong the physics is. The discriminating function of the same
    # grade-3 invariant is the distance to the pole. This distinction is the
    # whole point: the grade-k object supplies the argument, the network has to
    # supply the non-monotonic function of it.
    for idx, label in (((MU, NU, BBAR), "leptonic tbar"), ((U, DBAR, B), "hadronic t")):
        m = mass_of(p4, idx)
        record(3, f"-|m - m_t|  {label}", ",".join(str(k) for k in idx), -np.abs(m - M_T),
               "same grade-3 invariant, now as distance to the top pole")
        # relativistic Breit-Wigner weight: the likelihood-shaped version
        bw = 1.0 / ((m**2 - M_T**2)**2 + (M_T * G_T)**2)
        record(3, f"Breit-Wigner(m_t)  {label}", ",".join(str(k) for k in idx), bw,
               "Breit-Wigner weight on the same invariant")

    # both tops simultaneously: the actual physical distinction between classes
    m_lep = mass_of(p4, (MU, NU, BBAR))
    m_had = mass_of(p4, (U, DBAR, B))
    record("3+3", "-|m_lep - m_t| - |m_had - m_t|", "0,1,5 & 2,3,4",
           -np.abs(m_lep - M_T) - np.abs(m_had - M_T),
           "both top candidates near the pole = double-resonant class")
    record("3+3", "-|m_lep - m_t| (had. already on shell)", "0,1,5",
           -np.abs(m_lep - M_T),
           "the class difference is one top: this is the physical discriminant")

    # and the grade-2 analogue, for completeness: W masses as pole distances
    for (i, j), label in (((MU, NU), "W-"), ((U, DBAR), "W+")):
        record(2, f"-|m - m_W|  {label}", f"{i},{j}",
               -np.abs(mass_of(p4, (i, j)) - M_W),
               "grade-2 pole distance: W on shell in BOTH classes, so still ~0.5")

    # ---- grade 4: the pseudoscalar, i.e. the CP-odd slot -------------------
    for idx, label in (((MU, NU, U, DBAR), "eps(mu,nubar,u,dbar)"),
                       ((MU, B, U, BBAR), "eps(mu,b,u,bbar)"),
                       ((MU, NU, B, BBAR), "eps(mu,nubar,b,bbar)")):
        eps = pseudoscalar(p4, idx)
        record(4, label, ",".join(str(k) for k in idx), eps,
               "CP-odd: no CPV in this process -> expected AUC 0.5")
        record(4, f"|{label}|", ",".join(str(k) for k in idx), np.abs(eps),
               "magnitude only: CP-even, measures the 4-volume scale")

    # ---- how symmetric is the CP-odd slot? --------------------------------
    LOG.info("--- CP-odd symmetry check (no CPV expected) ---")
    sym_rows = []
    for idx, label in (((MU, NU, U, DBAR), "eps(mu,nubar,u,dbar)"),
                       ((MU, B, U, BBAR), "eps(mu,b,u,bbar)")):
        eps = pseudoscalar(p4, idx)
        for cls, name in ((1, "signal tW"), (0, "background ttbar")):
            e = eps[y == cls]
            frac_pos = float((e > 0).mean())
            # binomial error on the positive fraction
            err = float(np.sqrt(frac_pos * (1 - frac_pos) / len(e)))
            sym_rows.append(dict(observable=label, cls=name, n=len(e),
                                 frac_positive=round(frac_pos, 6),
                                 err=round(err, 6),
                                 pull_from_half=round((frac_pos - 0.5) / err, 2)))
            LOG.info("%-24s %-18s frac(eps>0) = %.6f +- %.6f  (%.2f sigma from 1/2)",
                     label, name, frac_pos, err, (frac_pos - 0.5) / err)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    sym_path = str(Path(args.out).with_name(Path(args.out).stem + "_cpsym.csv"))
    with open(sym_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sym_rows[0].keys()))
        w.writeheader(); w.writerows(sym_rows)
    LOG.info("wrote %s and %s", args.out, sym_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
