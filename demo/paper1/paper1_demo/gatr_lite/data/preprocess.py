"""Preprocess LHE files into chunked HDF5 datasets.

CLI usage
---------
    python -m paper1_demo.gatr_lite.data.preprocess \\
        --input  /path/to/sample.lhe \\
        --output /path/to/output.h5  \\
        --label  1                   \\
        --max-events 100000          \\
        --pt-cut 10

Output HDF5 layout
------------------
    p4       : (N, 6, 4)  float32   -- (E, px, py, pz), canonical parton order
    pdg      : (N, 6)     int32     -- PDG codes in canonical order
    label    : (N,)       int8      -- 0=tT background, 1=DR1 signal

    attrs:
        xsec_pb, xerr_pb, n_total_in_file, n_passed_cut,
        pt_cut_gev, source_lhe, sqrt_s_gev, generator

Canonical parton order: [mu-, nu_mu_bar, u, d_bar, b, b_bar]
PDG codes:              [+13, -14, +2, -1, +5, -5]

Selection cuts
--------------
* Exactly one outgoing parton of each PDG in {+13, -14, +2, -1, +5, -5}.
* pt of the 4 jets (u, d_bar, b, b_bar), sorted descending, 4th jet > pt_cut_gev.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore

from paper1_demo.gatr_lite.data.lhe_parser import iter_events, parse_init

logger = logging.getLogger(__name__)

# Canonical order of final-state PDG codes
CANONICAL_PDG: List[int] = [+13, -14, +2, -1, +5, -5]
# Indices of jet partons within canonical order (u, d_bar, b, b_bar)
JET_INDICES: List[int] = [2, 3, 4, 5]   # positions of +2, -1, +5, -5


def _pt(p4: Tuple[float, float, float, float]) -> float:
    """Transverse momentum from (E, px, py, pz)."""
    _, px, py, _ = p4
    return math.sqrt(px * px + py * py)


def _extract_canonical_partons(
    partons: List[Tuple[int, int, Tuple[float, float, float, float]]]
) -> Optional[List[Tuple[float, float, float, float]]]:
    """Return list of p4 in canonical PDG order, or None if selection fails.

    Requires exactly one outgoing (ISTUP=+1) parton for each canonical PDG.
    """
    outgoing: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for pdg, istup, p4 in partons:
        if istup == 1 and pdg in CANONICAL_PDG:
            outgoing.setdefault(pdg, []).append(p4)

    for pdg in CANONICAL_PDG:
        found = outgoing.get(pdg, [])
        if len(found) != 1:
            logger.warning(
                "Expected exactly 1 parton with PDG=%d, found %d; dropping event",
                pdg, len(found),
            )
            return None

    return [outgoing[pdg][0] for pdg in CANONICAL_PDG]


def _passes_pt_cut(canonical_p4: List[Tuple[float, float, float, float]], pt_cut: float) -> bool:
    """Check that the 4th leading-pt jet (among u, d_bar, b, b_bar) > pt_cut."""
    jet_pts = sorted(
        [_pt(canonical_p4[i]) for i in JET_INDICES],
        reverse=True,
    )
    return jet_pts[3] > pt_cut   # 4th entry (0-indexed) = minimum of the 4


def preprocess(
    input_lhe: str | Path,
    output_h5: str | Path,
    label: int,
    max_events: Optional[int] = None,
    pt_cut_gev: float = 10.0,
    chunk_size: int = 10_000,
) -> None:
    """Convert an LHE file to HDF5.

    Parameters
    ----------
    input_lhe   : path to .lhe file
    output_h5   : path for output .h5 file
    label       : 0 = tT background, 1 = DR1 signal
    max_events  : stop after this many *input* events (None = all)
    pt_cut_gev  : minimum pT for 4th jet (GeV)
    chunk_size  : HDF5 chunk size along event axis
    """
    if h5py is None:
        raise ImportError("h5py is required for preprocess; install it via dnf or pip")

    input_lhe = Path(input_lhe)
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Parsing init block from %s", input_lhe)
    init = parse_init(input_lhe)
    xsec_pb  = init["xsec_pb"]
    xerr_pb  = init["xerr_pb"]

    logger.info("  xsec_pb=%.6g  xerr_pb=%.6g  ebeam=%.1f GeV", xsec_pb, xerr_pb, init["ebeam_gev"])
    logger.info("Processing events (pt_cut=%.1f GeV, max_events=%s) ...", pt_cut_gev, max_events)

    p4_buf:  List[np.ndarray] = []   # each (6, 4)
    pdg_buf: List[np.ndarray] = []   # each (6,)

    n_total    = 0
    n_passed   = 0
    n_dropped_canonical = 0
    n_dropped_pt        = 0

    with h5py.File(output_h5, "w") as hf:
        # Create resizable datasets
        ds_p4 = hf.create_dataset(
            "p4",
            shape=(0, 6, 4),
            maxshape=(None, 6, 4),
            dtype=np.float32,
            chunks=(chunk_size, 6, 4),
            compression="gzip",
            compression_opts=4,
        )
        ds_pdg = hf.create_dataset(
            "pdg",
            shape=(0, 6),
            maxshape=(None, 6),
            dtype=np.int32,
            chunks=(chunk_size, 6),
            compression="gzip",
            compression_opts=4,
        )
        ds_label = hf.create_dataset(
            "label",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int8,
            chunks=(chunk_size,),
            compression="gzip",
            compression_opts=4,
        )

        def _flush() -> None:
            nonlocal n_passed
            if not p4_buf:
                return
            n_new = len(p4_buf)
            n_cur = ds_p4.shape[0]
            ds_p4.resize(n_cur + n_new, axis=0)
            ds_pdg.resize(n_cur + n_new, axis=0)
            ds_label.resize(n_cur + n_new, axis=0)

            ds_p4[n_cur:] = np.array(p4_buf, dtype=np.float32)
            ds_pdg[n_cur:] = np.full((n_new, 6), CANONICAL_PDG, dtype=np.int32)
            ds_label[n_cur:] = np.full(n_new, label, dtype=np.int8)

            p4_buf.clear()
            pdg_buf.clear()
            n_passed += n_new
            logger.debug("Flushed %d events (total passed so far: %d)", n_new, n_passed)

        for event in iter_events(input_lhe):
            if max_events is not None and n_total >= max_events:
                break
            n_total += 1

            canon = _extract_canonical_partons(event["partons"])
            if canon is None:
                n_dropped_canonical += 1
                continue

            if not _passes_pt_cut(canon, pt_cut_gev):
                n_dropped_pt += 1
                continue

            p4_arr = np.array(canon, dtype=np.float32)  # (6, 4)
            p4_buf.append(p4_arr)
            pdg_buf.append(np.array(CANONICAL_PDG, dtype=np.int32))

            if len(p4_buf) >= chunk_size:
                _flush()

        _flush()  # write remaining

        # Attributes
        hf.attrs["xsec_pb"]           = xsec_pb
        hf.attrs["xerr_pb"]           = xerr_pb
        hf.attrs["n_total_in_file"]    = n_total
        hf.attrs["n_passed_cut"]       = n_passed
        hf.attrs["n_dropped_canonical"] = n_dropped_canonical
        hf.attrs["n_dropped_pt"]       = n_dropped_pt
        hf.attrs["pt_cut_gev"]         = pt_cut_gev
        hf.attrs["source_lhe"]         = str(input_lhe)
        hf.attrs["sqrt_s_gev"]         = 14000.0
        hf.attrs["generator"]          = "CompHEP 4.5.2rc12"
        hf.attrs["label"]              = label
        hf.attrs["canonical_pdg"]      = CANONICAL_PDG

    logger.info(
        "Done. Total=%d  passed=%d  dropped_canonical=%d  dropped_pt=%d",
        n_total, n_passed, n_dropped_canonical, n_dropped_pt,
    )
    logger.info("Output: %s", output_h5)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert LHE file to HDF5 parton dataset.",
    )
    p.add_argument("--input",      required=True,  help="Path to input .lhe file")
    p.add_argument("--output",     required=True,  help="Path for output .h5 file")
    p.add_argument("--label",      required=True,  type=int, choices=[0, 1],
                   help="Event label: 0=tT (background), 1=DR1 (signal)")
    p.add_argument("--max-events", default=None,   type=int,
                   help="Maximum number of input events to process "
                        "(default: all; pass 0 for unlimited explicitly)")
    p.add_argument("--pt-cut",     default=10.0,   type=float,
                   help="Minimum pT of 4th jet in GeV (default: 10.0)")
    p.add_argument("--chunk-size", default=10_000, type=int,
                   help="HDF5 chunk size along event axis (default: 10000)")
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # 0 (or negative) means "no limit" -- treat as None
    max_events = args.max_events
    if max_events is not None and max_events <= 0:
        max_events = None
    preprocess(
        input_lhe   = args.input,
        output_h5   = args.output,
        label       = args.label,
        max_events  = max_events,
        pt_cut_gev  = args.pt_cut,
        chunk_size  = args.chunk_size,
    )
