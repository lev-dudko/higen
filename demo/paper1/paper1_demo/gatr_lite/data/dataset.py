"""PyTorch Dataset classes for preprocessed parton HDF5 files.

Classes
-------
PartonH5Dataset
    Wraps a single HDF5 file. Lazy-opens on first access.
    Returns dict with keys: p4, pdg, label, xsec_pb, sample_id.

MultiH5Dataset
    Concatenates multiple HDF5 files, mapping global_idx -> (file_idx, local_idx).
    Exposes per-sample xsec_pb for plotting weights.

Utilities
---------
make_train_val_split(dataset, val_frac, seed) -> (train_subset, val_subset)
    Random permutation split using torch.utils.data.Subset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore

logger = logging.getLogger(__name__)


class PartonH5Dataset(Dataset):
    """Single-file parton dataset backed by an HDF5 file.

    The HDF5 file is opened lazily on first ``__getitem__`` call to support
    use with ``torch.utils.data.DataLoader`` multiprocessing workers.

    Parameters
    ----------
    h5_path : str or Path
        Path to the .h5 file produced by ``preprocess.py``.
    transform : callable, optional
        Optional transform applied to the p4 tensor before returning.

    Dataset item keys
    -----------------
    p4       : FloatTensor (6, 4)  -- (E, px, py, pz) for 6 canonical partons
    pdg      : LongTensor  (6,)   -- PDG codes
    label    : LongTensor  ()     -- 0=tT, 1=DR1
    xsec_pb  : float              -- cross-section from file attrs (pb)
    sample_id: int                -- local index within this file
    """

    def __init__(self, h5_path: Union[str, Path], transform=None) -> None:
        if h5py is None:
            raise ImportError("h5py is required for PartonH5Dataset")
        self.h5_path  = Path(h5_path)
        self.transform = transform
        self._h5: Optional[h5py.File] = None   # opened lazily
        self._len: Optional[int] = None

        # Read length and attrs eagerly (file stays closed after init)
        with h5py.File(self.h5_path, "r") as f:
            self._len     = f["p4"].shape[0]
            self._xsec_pb = float(f.attrs.get("xsec_pb", 0.0))
            self._xerr_pb = float(f.attrs.get("xerr_pb", 0.0))
            self._label   = int(f.attrs.get("label", -1))
            self._n_total = int(f.attrs.get("n_total_in_file", 0))
            self._generator = str(f.attrs.get("generator", ""))

        logger.debug(
            "PartonH5Dataset: %s  n=%d  xsec=%.4g pb  label=%d",
            self.h5_path.name, self._len, self._xsec_pb, self._label,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def xsec_pb(self) -> float:
        return self._xsec_pb

    @property
    def xerr_pb(self) -> float:
        return self._xerr_pb

    @property
    def n_events(self) -> int:
        return self._len  # type: ignore[return-value]

    @property
    def event_weight(self) -> float:
        """xsec_pb / n_events: weight per event for cross-section normalised plots."""
        if self._len == 0:
            return 0.0
        return self._xsec_pb / self._len

    # ------------------------------------------------------------------
    # Lazy file handle
    # ------------------------------------------------------------------

    def _open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def close(self) -> None:
        """Explicitly close the HDF5 file handle."""
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._len  # type: ignore[return-value]

    def __getitem__(self, idx: int) -> Dict:
        self._open()
        assert self._h5 is not None

        p4    = torch.from_numpy(self._h5["p4"][idx].astype(np.float32))    # (6, 4)
        pdg   = torch.from_numpy(self._h5["pdg"][idx].astype(np.int64))     # (6,)
        label = torch.tensor(int(self._h5["label"][idx]), dtype=torch.long)

        if self.transform is not None:
            p4 = self.transform(p4)

        return {
            "p4":       p4,
            "pdg":      pdg,
            "label":    label,
            "xsec_pb":  self._xsec_pb,
            "sample_id": idx,
        }

    # Ensure picklability for DataLoader workers
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None   # drop file handle before pickling
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5 = None


# ---------------------------------------------------------------------------
# InMemoryPartonDataset (random-access friendly)
# ---------------------------------------------------------------------------

class InMemoryPartonDataset(Dataset):
    """Like PartonH5Dataset but eagerly reads tensors into RAM.

    Per-item HDF5 chunked+gzip access is unusably slow under random batching:
    each chunk is decompressed many times across an epoch. Reading the full
    arrays once and indexing in numpy avoids this entirely.

    A subset of indices may be passed via ``indices`` to load only a slice
    (useful for cap-per-class or train/val splits) without ballooning RAM.
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        indices: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.transform = transform
        with h5py.File(self.h5_path, "r") as f:
            n_total = int(f["p4"].shape[0])
            if indices is None:
                self._p4    = f["p4"][:].astype(np.float32)
                self._pdg   = f["pdg"][:].astype(np.int64)
                self._label = f["label"][:].astype(np.int64)
            else:
                idx = np.asarray(indices, dtype=np.int64)
                if not np.all(np.diff(idx) >= 0):
                    idx = np.sort(idx)
                # h5py fancy indexing requires sorted unique increasing
                self._p4    = f["p4"][idx].astype(np.float32)
                self._pdg   = f["pdg"][idx].astype(np.int64)
                self._label = f["label"][idx].astype(np.int64)
            self._xsec_pb = float(f.attrs.get("xsec_pb", 0.0))
            self._n_total_in_file = n_total

        self._len = len(self._p4)
        nbytes = self._p4.nbytes + self._pdg.nbytes + self._label.nbytes
        logger.info(
            "InMemoryPartonDataset: %d events from %s  (%.1f MB)",
            self._len, self.h5_path.name, nbytes / 1e6,
        )

    def __len__(self) -> int:
        return self._len

    @property
    def xsec_pb(self) -> float:
        return self._xsec_pb

    @property
    def event_weight(self) -> float:
        if self._len == 0:
            return 0.0
        return self._xsec_pb / self._len

    @property
    def labels(self) -> np.ndarray:
        return self._label

    def __getitem__(self, idx: int) -> Dict:
        p4 = torch.from_numpy(self._p4[idx])
        if self.transform is not None:
            p4 = self.transform(p4)
        return {
            "p4":       p4,
            "pdg":      torch.from_numpy(self._pdg[idx]),
            "label":    torch.tensor(int(self._label[idx]), dtype=torch.long),
            "xsec_pb":  self._xsec_pb,
            "sample_id": idx,
        }


# ---------------------------------------------------------------------------
# MultiH5Dataset
# ---------------------------------------------------------------------------

class MultiH5Dataset(Dataset):
    """Concatenation of multiple PartonH5Dataset files.

    Parameters
    ----------
    datasets : sequence of PartonH5Dataset (or paths to .h5 files)

    Global index maps linearly into files in order.  Each item additionally
    contains ``file_idx`` (which file) and ``sample_id`` (local index within
    that file) alongside the standard p4 / pdg / label / xsec_pb keys.
    """

    def __init__(self, datasets: Sequence[Union[PartonH5Dataset, str, Path]]) -> None:
        self._datasets: List[PartonH5Dataset] = []
        for ds in datasets:
            if isinstance(ds, (str, Path)):
                ds = PartonH5Dataset(ds)
            self._datasets.append(ds)

        # Build cumulative length table for O(log N) lookup
        self._cum_lengths: List[int] = []
        total = 0
        for ds in self._datasets:
            total += len(ds)
            self._cum_lengths.append(total)

        self._total = total
        logger.info(
            "MultiH5Dataset: %d files, %d total events",
            len(self._datasets), self._total,
        )

    def __len__(self) -> int:
        return self._total

    def _resolve(self, global_idx: int) -> Tuple[int, int]:
        """Return (file_idx, local_idx) for a global index."""
        # Binary search over cumulative lengths
        lo, hi = 0, len(self._cum_lengths) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if global_idx < self._cum_lengths[mid]:
                hi = mid
            else:
                lo = mid + 1
        file_idx = lo
        local_idx = global_idx - (self._cum_lengths[file_idx - 1] if file_idx > 0 else 0)
        return file_idx, local_idx

    def __getitem__(self, global_idx: int) -> Dict:
        file_idx, local_idx = self._resolve(global_idx)
        item = self._datasets[file_idx][local_idx]
        item["file_idx"]  = file_idx
        item["sample_id"] = local_idx
        return item

    @property
    def datasets(self) -> List[PartonH5Dataset]:
        return self._datasets

    def xsec_pbs(self) -> List[float]:
        """Cross-sections (pb) for each constituent file."""
        return [ds.xsec_pb for ds in self._datasets]

    def event_weights(self) -> List[float]:
        """Per-event weights (xsec_pb / n_events) for each constituent file."""
        return [ds.event_weight for ds in self._datasets]


# ---------------------------------------------------------------------------
# Train / val split utility
# ---------------------------------------------------------------------------

def make_train_val_split(
    dataset: Dataset,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """Split a dataset into train and val subsets via random permutation.

    Parameters
    ----------
    dataset  : any torch Dataset (typically MultiH5Dataset or PartonH5Dataset)
    val_frac : fraction of events to use for validation
    seed     : RNG seed for reproducibility

    Returns
    -------
    (train_subset, val_subset) : pair of torch.utils.data.Subset
    """
    n = len(dataset)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val   = max(1, int(round(n * val_frac)))
    n_train = n - n_val

    val_indices   = perm[:n_val].tolist()
    train_indices = perm[n_val:].tolist()

    logger.info(
        "Train/val split: total=%d  train=%d  val=%d  seed=%d",
        n, n_train, n_val, seed,
    )
    return Subset(dataset, train_indices), Subset(dataset, val_indices)
