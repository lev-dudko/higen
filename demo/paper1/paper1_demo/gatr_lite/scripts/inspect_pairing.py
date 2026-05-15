"""Diagnostic dump of pairing-feature ranges on a real H5 sample.

Loads the canonical-order 4-momenta stored in the smoke H5 datasets (DR1 +
tT) and prints, for the first ``--n`` events combined, the min/max/mean/std
of every pairing-cache feature both in raw and normalised form. Used to
self-check the normalisation strategy in ``gatr.pairing``.

Usage::

    python -m paper1_demo.gatr_lite.scripts.inspect_pairing \
        --signal /path/to/dr1.h5 --background /path/to/tT.h5 --n 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch

from paper1_demo.gatr_lite.gatr.pairing import compute_pairing_features


# Canonical order keys expected in the H5 file. Matches the dataset layout
# produced by paper1_demo.gatr_lite.data.preprocess.
_DATASET_CANDIDATES = ("p4", "momenta", "x", "p4_canonical")


def _load_p4(path: Path, n: int) -> torch.Tensor:
    with h5py.File(path, "r") as h:
        for k in _DATASET_CANDIDATES:
            if k in h:
                arr = h[k][:n]
                break
        else:
            raise KeyError(
                f"None of {_DATASET_CANDIDATES} found in {path}; keys={list(h.keys())}"
            )
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (6, 4):
        raise ValueError(f"Expected (N, 6, 4) shape; got {arr.shape} in {path}")
    return torch.from_numpy(arr)


def _summarise(name: str, t: torch.Tensor) -> str:
    flat = t.float().flatten()
    shape_str = str(tuple(t.shape))
    return (
        f"  {name:18s}  shape={shape_str:<14s}  "
        f"min={flat.min().item():+.4e}  max={flat.max().item():+.4e}  "
        f"mean={flat.mean().item():+.4e}  std={flat.std().item():.4e}"
    )


def _dump(p4: torch.Tensor, normalize: bool, *, label: str) -> None:
    pf = compute_pairing_features(p4, normalize=normalize)
    print(f"\n[{label}] (normalize={normalize})")
    for k in (
        "s_had", "s_lep",
        "sigma_p_had", "sigma_m_had", "sigma_p_lep", "sigma_m_lep",
        "mass_W_had", "mass_W_lep",
        "T_pair", "T_dual_pair",
    ):
        print(_summarise(k, pf[k]))


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--signal", type=Path, required=True)
    p.add_argument("--background", type=Path, required=True)
    p.add_argument("--n", type=int, default=1000,
                   help="Number of events per file to inspect (default: 1000)")
    args = p.parse_args(argv)

    sig = _load_p4(args.signal, args.n)
    bkg = _load_p4(args.background, args.n)
    p4 = torch.cat([sig, bkg], dim=0)
    print(f"Loaded {sig.shape[0]} signal + {bkg.shape[0]} background events"
          f" = {p4.shape[0]} total. p4 shape={tuple(p4.shape)}")

    _dump(p4, normalize=False, label="RAW")
    _dump(p4, normalize=True, label="NORMALISED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
