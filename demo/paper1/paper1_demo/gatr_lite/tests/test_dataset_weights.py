"""Cross-section weights of the parton datasets.

The attribute ``xsec_pb`` of an HDF5 sample is the cross section of the LHE
file before the event selection.  A distribution of the stored (selected)
events has to be normalised to xsec_pb * n_passed_cut / n_total_in_file.
"""
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from paper1_demo.gatr_lite.data.dataset import InMemoryPartonDataset, PartonH5Dataset


def _write(path, n_stored, xsec_pb, n_total=None, n_passed=None):
    with h5py.File(path, "w") as f:
        f.create_dataset("p4", data=np.zeros((n_stored, 6, 4), dtype=np.float32))
        f.create_dataset("pdg", data=np.zeros((n_stored, 6), dtype=np.int32))
        f.create_dataset("label", data=np.zeros(n_stored, dtype=np.int8))
        f.attrs["xsec_pb"] = xsec_pb
        f.attrs["label"] = 0
        if n_total is not None:
            f.attrs["n_total_in_file"] = n_total
            f.attrs["n_passed_cut"] = n_passed


@pytest.mark.parametrize("cls", [PartonH5Dataset, InMemoryPartonDataset])
def test_weights_sum_to_selected_cross_section(tmp_path, cls):
    path = tmp_path / "s.h5"
    _write(path, n_stored=25, xsec_pb=4.0, n_total=100, n_passed=25)
    ds = cls(str(path))
    assert ds.xsec_pb == pytest.approx(4.0)
    assert ds.xsec_selected_pb == pytest.approx(1.0)          # 4 pb * 25 / 100
    assert ds.event_weight * len(ds) == pytest.approx(1.0)


@pytest.mark.parametrize("cls", [PartonH5Dataset, InMemoryPartonDataset])
def test_falls_back_to_lhe_cross_section_without_counts(tmp_path, cls):
    path = tmp_path / "s.h5"
    _write(path, n_stored=10, xsec_pb=3.0)
    ds = cls(str(path))
    assert ds.xsec_selected_pb == pytest.approx(3.0)
    assert ds.event_weight * len(ds) == pytest.approx(3.0)
