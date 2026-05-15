"""Streaming LHE parser for LHEF 1.0 files produced by CompHEP 4.5.2rc12.

Does NOT depend on pylhe. Uses only stdlib + numpy.

Public API
----------
parse_init(path) -> dict
    Extract cross-section and beam info from the <init> block.

iter_events(path) -> Iterator[dict]
    Yield events one by one without loading the full file into memory.
    Each event dict has the form::

        {
            "partons": List[Tuple[int, int, Tuple[float,float,float,float]]],
                # (pdg, istup, (E, px, py, pz))
            "event_weight": float,   # XWGTUP
            "idprup":       int,     # process id
            "scalup":       float,   # scale (GeV)
            "aqedup":       float,
            "aqcdup":       float,
        }

Parton p4 layout: (E, px, py, pz)  -- energy-first convention.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# init block
# ---------------------------------------------------------------------------

def parse_init(path: str | Path) -> Dict:
    """Parse the <init> block of an LHE file.

    Returns
    -------
    dict with keys:
        idbmup1, idbmup2 : int        -- beam PDG codes
        ebeam_gev         : float      -- beam energy (both beams assumed equal)
        nprup             : int        -- number of sub-processes
        xsec_pb           : float      -- sum of XSECUP over all processes (pb)
        xerr_pb           : float      -- sum of XERRUP (pb)
    """
    path = Path(path)
    result: Dict = {}

    with path.open("r", errors="replace") as fh:
        in_init = False
        header_parsed = False
        processes: List[Tuple[float, float]] = []  # (xsec, xerr)

        for raw in fh:
            line = raw.strip()

            if "<init>" in line:
                in_init = True
                continue

            if "</init>" in line:
                break

            if not in_init:
                continue

            # Skip XML-style comment / sub-tag lines inside <init>
            if line.startswith("<") or line.startswith("!") or line == "":
                continue

            if not header_parsed:
                # First data line: IDBMUP1 IDBMUP2 EBMUP1 EBMUP2 PDFGUP1 PDFGUP2
                #                  PDFSUP1 PDFSUP2 IDWTUP NPRUP
                parts = line.split()
                if len(parts) < 10:
                    logger.warning("init header line has unexpected number of tokens: %r", line)
                    continue
                result["idbmup1"] = int(parts[0])
                result["idbmup2"] = int(parts[1])
                result["ebeam_gev"] = float(parts[2])   # EBMUP1 (assume EBMUP2 == EBMUP1)
                result["nprup"] = int(parts[9])
                header_parsed = True
            else:
                # Subsequent lines: XSECUP XERRUP XMAXUP LPRUP
                parts = line.split()
                if len(parts) < 4:
                    logger.warning("process line has unexpected tokens: %r", line)
                    continue
                processes.append((float(parts[0]), float(parts[1])))

    if not result:
        raise ValueError(f"No <init> block found in {path}")

    result["xsec_pb"] = sum(x for x, _ in processes)
    result["xerr_pb"] = sum(xe for _, xe in processes)
    return result


# ---------------------------------------------------------------------------
# event iterator
# ---------------------------------------------------------------------------

def iter_events(path: str | Path) -> Iterator[Dict]:
    """Yield events from an LHE file one at a time (streaming).

    Each yielded dict::

        {
            "partons":       List[Tuple[int, int, Tuple[float,float,float,float]]],
            "event_weight":  float,
            "idprup":        int,
            "scalup":        float,
            "aqedup":        float,
            "aqcdup":        float,
        }

    parton tuple: (pdg_id, istup, (E, px, py, pz))

    Malformed events are skipped with a WARNING log.
    """
    path = Path(path)

    with path.open("r", errors="replace") as fh:
        in_event = False
        event_lines: List[str] = []

        for raw in fh:
            line = raw.strip()

            if "<event>" in line:
                in_event = True
                event_lines = []
                continue

            if "</event>" in line:
                in_event = False
                parsed = _parse_event_block(event_lines)
                if parsed is not None:
                    yield parsed
                event_lines = []
                continue

            if in_event:
                event_lines.append(line)


def _parse_event_block(lines: List[str]) -> Dict | None:
    """Parse the lines between <event> and </event> tags.

    Returns None on any parse error (with a WARNING logged).
    """
    # Filter out empty lines and XML tags
    data_lines = [
        l for l in lines
        if l and not l.startswith("<") and not l.startswith("#") and not l.startswith("!")
    ]

    if not data_lines:
        logger.warning("Empty event block, skipping")
        return None

    # --- Header line ---
    # NUP IDPRUP XWGTUP SCALUP AQEDUP AQCDUP
    header = data_lines[0].split()
    if len(header) < 6:
        logger.warning("Event header has < 6 tokens: %r", data_lines[0])
        return None

    try:
        nup      = int(header[0])
        idprup   = int(header[1])
        xwgtup   = float(header[2])
        scalup   = float(header[3])
        aqedup   = float(header[4])
        aqcdup   = float(header[5])
    except (ValueError, IndexError) as exc:
        logger.warning("Could not parse event header: %s", exc)
        return None

    parton_lines = data_lines[1:]

    if len(parton_lines) != nup:
        logger.warning(
            "NUP=%d but got %d parton lines; skipping event", nup, len(parton_lines)
        )
        return None

    partons: List[Tuple[int, int, Tuple[float, float, float, float]]] = []
    for pline in parton_lines:
        p = _parse_parton_line(pline)
        if p is None:
            logger.warning("Could not parse parton line: %r; skipping event", pline)
            return None
        partons.append(p)

    return {
        "partons":      partons,
        "event_weight": xwgtup,
        "idprup":       idprup,
        "scalup":       scalup,
        "aqedup":       aqedup,
        "aqcdup":       aqcdup,
    }


def _parse_parton_line(line: str) -> Tuple[int, int, Tuple[float, float, float, float]] | None:
    """Parse a single parton line.

    Format (LHEF 1.0):
        PDG ISTUP MOTH1 MOTH2 ICOL1 ICOL2 PX PY PZ E M VTIMUP SPINUP

    Returns (pdg, istup, (E, px, py, pz)) or None on error.
    Note: LHE stores momenta as (px, py, pz, E), we reorder to (E, px, py, pz).
    """
    parts = line.split()
    if len(parts) < 10:
        return None
    try:
        pdg   = int(parts[0])
        istup = int(parts[1])
        px    = float(parts[6])
        py    = float(parts[7])
        pz    = float(parts[8])
        e     = float(parts[9])
    except (ValueError, IndexError):
        return None

    return (pdg, istup, (e, px, py, pz))
