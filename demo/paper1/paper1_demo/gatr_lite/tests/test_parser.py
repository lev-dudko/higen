"""pytest tests for lhe_parser.py.

Runs on the first 100 events of the DR1 SM sample. The path to the sample
is read from the ``HIGEN_DR1_LHE`` environment variable; tests are skipped
if it is unset or points to a missing file. See ``demo/paper1/README.md``
for how to download the sample from www-hep.

All tests are fast (<2 s) because they only read a small prefix of the file.
"""

from __future__ import annotations

import math
import os
import itertools
from pathlib import Path

import pytest

DR1_LHE = os.environ.get("HIGEN_DR1_LHE", "")

if not DR1_LHE or not Path(DR1_LHE).is_file():
    pytest.skip(
        "Set HIGEN_DR1_LHE to the DR1 LHE sample to run parser tests "
        "(see demo/paper1/README.md for download instructions).",
        allow_module_level=True,
    )

FINAL_STATE_PDG = {+13, -14, +2, -1, +5, -5}
EXPECTED_EBEAM  = 7000.0   # GeV per beam  (total sqrt_s = 14 TeV)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_n_events(path: str, n: int):
    """Return list of first n events from path."""
    from paper1_demo.gatr_lite.data.lhe_parser import iter_events
    return list(itertools.islice(iter_events(path), n))


# ---------------------------------------------------------------------------
# parse_init tests
# ---------------------------------------------------------------------------

class TestParseInit:
    def setup_method(self):
        from paper1_demo.gatr_lite.data.lhe_parser import parse_init
        self.init = parse_init(DR1_LHE)

    def test_xsec_positive(self):
        assert self.init["xsec_pb"] > 0.0, "xsec_pb must be positive"

    def test_xerr_nonnegative(self):
        assert self.init["xerr_pb"] >= 0.0

    def test_ebeam(self):
        assert math.isclose(self.init["ebeam_gev"], EXPECTED_EBEAM, rel_tol=1e-6), (
            f"Expected ebeam_gev={EXPECTED_EBEAM}, got {self.init['ebeam_gev']}"
        )

    def test_idbmup_protons(self):
        # Both beams are protons (PDG 2212)
        assert self.init["idbmup1"] == 2212
        assert self.init["idbmup2"] == 2212

    def test_nprup_positive(self):
        assert self.init["nprup"] >= 1


# ---------------------------------------------------------------------------
# iter_events tests
# ---------------------------------------------------------------------------

class TestIterEvents:
    """Tests on first 100 events."""

    @pytest.fixture(scope="class")
    def events(self):
        return _first_n_events(DR1_LHE, 100)

    def test_got_100_events(self, events):
        assert len(events) == 100

    def test_event_keys(self, events):
        required = {"partons", "event_weight", "idprup", "scalup", "aqedup", "aqcdup"}
        for i, ev in enumerate(events):
            missing = required - ev.keys()
            assert not missing, f"Event {i} missing keys: {missing}"

    def test_event_weight_finite(self, events):
        for i, ev in enumerate(events):
            assert math.isfinite(ev["event_weight"]), f"Event {i} non-finite weight"

    def test_parton_count_8(self, events):
        """Each event should have 8 partons (2 incoming gluons + 6 final state)."""
        for i, ev in enumerate(events[:10]):
            n = len(ev["partons"])
            assert n == 8, f"Event {i}: expected 8 partons, got {n}"

    def test_final_state_pdg_completeness(self, events):
        """First 10 events must have exactly one outgoing parton per canonical PDG."""
        for i, ev in enumerate(events[:10]):
            outgoing_pdg = [
                pdg for pdg, istup, _ in ev["partons"] if istup == 1
            ]
            assert set(outgoing_pdg) == FINAL_STATE_PDG, (
                f"Event {i}: final state PDGs {set(outgoing_pdg)} "
                f"!= expected {FINAL_STATE_PDG}"
            )
            # Exactly one of each
            for pdg in FINAL_STATE_PDG:
                count = outgoing_pdg.count(pdg)
                assert count == 1, (
                    f"Event {i}: PDG={pdg} appears {count} times (expected 1)"
                )

    def test_incoming_partons_istup(self, events):
        """Incoming partons must have ISTUP=-1, outgoing ISTUP=+1."""
        for i, ev in enumerate(events[:10]):
            for pdg, istup, _ in ev["partons"]:
                assert istup in (-1, 1), f"Event {i}: unexpected ISTUP={istup}"

    def test_p4_energy_first(self, events):
        """p4 tuple should be (E, px, py, pz); E > 0 for all partons."""
        for i, ev in enumerate(events[:10]):
            for pdg, istup, p4 in ev["partons"]:
                e, px, py, pz = p4
                assert e > 0.0, (
                    f"Event {i}, PDG={pdg}: non-positive energy E={e}"
                )

    def test_four_momentum_conservation(self, events):
        """Sum of 4-momenta: incoming == outgoing, tolerance 1e-3 GeV."""
        tol = 1e-3
        for i, ev in enumerate(events[:100]):
            p_in  = [0.0, 0.0, 0.0, 0.0]  # E, px, py, pz
            p_out = [0.0, 0.0, 0.0, 0.0]
            for pdg, istup, p4 in ev["partons"]:
                e, px, py, pz = p4
                if istup == -1:
                    for k, v in enumerate([e, px, py, pz]):
                        p_in[k] += v
                elif istup == 1:
                    for k, v in enumerate([e, px, py, pz]):
                        p_out[k] += v

            for comp, (vi, vo) in enumerate(zip(p_in, p_out)):
                label = ["E", "px", "py", "pz"][comp]
                assert abs(vi - vo) < tol, (
                    f"Event {i}: 4-momentum conservation violated for {label}: "
                    f"in={vi:.6f}  out={vo:.6f}  diff={abs(vi-vo):.2e}"
                )

    def test_incoming_are_gluons(self, events):
        """Incoming partons (ISTUP=-1) should be gluons (PDG=21)."""
        for i, ev in enumerate(events[:10]):
            incoming_pdg = [pdg for pdg, istup, _ in ev["partons"] if istup == -1]
            assert all(pdg == 21 for pdg in incoming_pdg), (
                f"Event {i}: non-gluon incoming partons: {incoming_pdg}"
            )
