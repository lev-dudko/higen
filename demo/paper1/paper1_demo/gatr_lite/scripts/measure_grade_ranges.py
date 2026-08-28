"""Measure the numerical dynamic range of each grade on real events.

The sharpest practical obstacle to a geometric-algebra foundation model is a
numerical one: an object of mass dimension k built from momenta scales like
(GeV)^k, so the input tensor spans many orders of magnitude before any network
sees it. Dimensional analysis predicts the powers (grade 1 ~ GeV, grade 2 ~
GeV^2, ..., the coplanarity invariant ~ GeV^12); this script measures them on
the training sample, which is what Sec. 8.1 of the paper quotes.

For each grade and each pairing-token feature we report the median and the
1st/99th percentile of |value| over a random sample of events, plus the implied
span in orders of magnitude. Output is a CSV for the manuscript table.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from paper1_demo.gatr_lite.gatr import algebra as A
from paper1_demo.gatr_lite.gatr.model import GATrLite, GATrLiteConfig
from paper1_demo.gatr_lite.data.dataset import InMemoryPartonDataset, MultiH5Dataset


def summarise(x: np.ndarray) -> dict:
    """Median and 1/99 percentiles of |x| over non-zero entries."""
    a = np.abs(x[np.isfinite(x)])
    a = a[a > 0]
    if a.size == 0:
        return dict(median=0.0, p01=0.0, p99=0.0, decades=0.0, n_nonzero=0)
    p01, med, p99 = np.percentile(a, [1, 50, 99])
    return dict(median=float(med), p01=float(p01), p99=float(p99),
                decades=float(np.log10(p99 / p01)) if p01 > 0 else float("inf"),
                n_nonzero=int(a.size))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-events", type=int, default=20000)
    p.add_argument("--out", default="grade_ranges.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--raw", action="store_true",
                   help="Measure the pairing features BEFORE the per-grade "
                        "normalisation (normalize=False), i.e. the span the "
                        "network would face without the normalisation machinery.")
    args = p.parse_args(argv)

    ds = MultiH5Dataset([
        InMemoryPartonDataset(f"{args.data_dir}/dr1.h5"),
        InMemoryPartonDataset(f"{args.data_dir}/tT.h5"),
    ])
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.n_events, len(ds)), replace=False)

    items = [ds[int(i)] for i in idx]
    p4 = torch.stack([it["p4"] for it in items]).double()
    pdg = torch.stack([it["pdg"] for it in items])
    print(f"sampled {len(idx)} events from {len(ds)}")

    # Build the input tensor exactly as the trained network sees it.
    model = GATrLite(GATrLiteConfig(use_join_block=False,
                                    pairing_content="full")).double().eval()
    with torch.no_grad():
        x = model.embed(p4, pdg)          # (B, T, C_in, 16)
    xn = x.numpy()
    print("input tensor:", xn.shape)

    rows = []

    # --- per grade, over the covariant channels only (channel 0 and 1) --------
    n_cov = model._n_cov_channels
    for k in range(5):
        vals = xn[:, :, :n_cov, :][..., A.GRADE_INDICES[k]]
        s = summarise(vals.ravel())
        rows.append(dict(quantity=f"grade {k} (covariant channels)",
                         expected_scaling=f"GeV^{k}", **s))

    # --- per grade, restricted to the two pairing tokens ---------------------
    for k in range(5):
        vals = xn[:, 6:8, :n_cov, :][..., A.GRADE_INDICES[k]]
        s = summarise(vals.ravel())
        rows.append(dict(quantity=f"grade {k} (pairing tokens only)",
                         expected_scaling=f"GeV^{k}", **s))

    # --- the grade-0 pairing invariants, individually ------------------------
    # These live on the scalar channels of the two pairing tokens. Their names
    # come from the pairing-feature cache; we report them by channel index so
    # the table is honest about what is measured.
    scal = xn[:, 6:8, n_cov:, 0]          # (B, 2, n_scalars)
    for c in range(scal.shape[-1]):
        s = summarise(scal[:, :, c].ravel())
        if s["n_nonzero"] == 0:
            continue
        rows.append(dict(quantity=f"pairing scalar channel {c}",
                         expected_scaling="(invariant)", **s))

    # --- raw (un-normalised) pairing multivectors ---------------------------
    # The tensor above is what the network sees, i.e. AFTER the per-grade
    # rescaling in pairing.py. The physically meaningful statement for the
    # manuscript is what the span would be WITHOUT it, since that is the
    # problem a foundation model has to solve.
    if args.raw:
        from paper1_demo.gatr_lite.gatr.pairing import compute_pairing_features
        with torch.no_grad():
            pf_raw = compute_pairing_features(p4, pdg, normalize=False)
        for name, t in sorted(pf_raw.items()):
            arr = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
            if arr.ndim >= 3 and arr.shape[-1] == 16:
                # multivector: report per populated grade
                for k in range(5):
                    vals = arr[..., A.GRADE_INDICES[k]]
                    s = summarise(vals.ravel())
                    if s["n_nonzero"] == 0:
                        continue
                    rows.append(dict(quantity=f"RAW {name} grade {k}",
                                     expected_scaling="unnormalised", **s))
            else:
                s = summarise(arr.ravel())
                if s["n_nonzero"]:
                    rows.append(dict(quantity=f"RAW {name}",
                                     expected_scaling="unnormalised", **s))

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'quantity':40s} {'median':>12s} {'p01':>12s} {'p99':>12s} {'decades':>8s}")
    for r in rows:
        print(f"{r['quantity']:40s} {r['median']:12.4g} {r['p01']:12.4g} "
              f"{r['p99']:12.4g} {r['decades']:8.1f}")

    cov = [r for r in rows if r["quantity"].startswith("grade")
           and "covariant" in r["quantity"] and r["n_nonzero"] > 0]
    if len(cov) > 1:
        lo = min(r["p01"] for r in cov)
        hi = max(r["p99"] for r in cov)
        print(f"\nFull span across populated grades: {lo:.4g} to {hi:.4g} "
              f"= {np.log10(hi/lo):.1f} orders of magnitude in one input tensor.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
