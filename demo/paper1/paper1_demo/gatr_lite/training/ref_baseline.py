"""REF baseline (Boos:2023kpp) loader: Keras h5 -> pure PyTorch.

The reference network NN_RV_GG is a 4-layer fully-connected MLP
(75 -> 500 -> 500 -> 500 -> 1) with ReLU activations and a sigmoid head.
It was trained in Keras 3.3.3 / TensorFlow on parton-level DR1 vs tT
samples with the cut pt_jet4 > 10 GeV.

This module loads the dense weights directly out of the Keras h5
checkpoint via h5py (no TensorFlow dependency) and runs inference in
PyTorch.

Public API
----------
``RefMLP``           -- nn.Module mirroring the Keras Sequential model
``load_keras_h5``    -- read the four (kernel, bias) pairs into a RefMLP
``predict_root``     -- run a RefMLP over an exam ROOT ntuple (uproot)

Notes
-----
* Keras `Dense.kernel` has shape ``(in, out)``; ``torch.nn.Linear.weight``
  has shape ``(out, in)`` -- a transpose is needed when copying.
* The exam ROOT files store the 75 input features as ``var1..var75``
  *already standardised* (the Keras training pipeline did not contain a
  separate normaliser layer; whatever preprocessing was applied is baked
  into the var* branches).
* The Dropout layers are noise-only at inference time and are absent in
  the PyTorch sequential -- they map to identity in eval mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


# Keras kernel paths inside the h5 (model_weights/<layer>/sequential/<layer>/...)
_DENSE_LAYERS = ("dense", "dense_1", "dense_2", "dense_3")
_INPUT_DIM = 75
_HIDDEN_DIM = 500


class RefMLP(nn.Module):
    """4-layer MLP matching the NN_RV_GG reference architecture.

    Forward returns a *logit* (pre-sigmoid). Apply ``torch.sigmoid`` outside
    if a probability is needed; this matches our training pipeline's
    convention. The Keras model itself ends in a sigmoid Dense, so its
    kernel/bias correspond to the logit linear stage.
    """

    def __init__(self, input_dim: int = _INPUT_DIM, hidden_dim: int = _HIDDEN_DIM) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.fc4(x).squeeze(-1)  # logit, shape (B,)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def _read_kernel_bias(h5: h5py.File, layer: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract (kernel, bias) for a Keras Dense layer from the h5 file.

    Layout (Keras 3.3.3, TF backend):
        model_weights/<layer>/sequential/<layer>/{kernel, bias}
    """
    base = f"model_weights/{layer}/sequential/{layer}"
    kernel = np.asarray(h5[f"{base}/kernel"], dtype=np.float32)
    bias = np.asarray(h5[f"{base}/bias"], dtype=np.float32)
    return kernel, bias


def load_keras_h5(path: str | Path) -> RefMLP:
    """Build a RefMLP and copy weights from a Keras h5 checkpoint.

    Verifies that the four dense layers have the expected
    (75->500, 500->500, 500->500, 500->1) shape.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"REF Keras h5 not found: {path}")

    expected_shapes = [
        (_INPUT_DIM, _HIDDEN_DIM),
        (_HIDDEN_DIM, _HIDDEN_DIM),
        (_HIDDEN_DIM, _HIDDEN_DIM),
        (_HIDDEN_DIM, 1),
    ]
    model = RefMLP()
    fc_modules = [model.fc1, model.fc2, model.fc3, model.fc4]
    with h5py.File(path, "r") as h5:
        backend = h5.attrs.get("backend", b"")
        keras_ver = h5.attrs.get("keras_version", b"")
        logger.info("REF h5: backend=%r keras_version=%r", backend, keras_ver)
        for layer, fc, expect in zip(_DENSE_LAYERS, fc_modules, expected_shapes):
            kernel, bias = _read_kernel_bias(h5, layer)
            if kernel.shape != expect:
                raise ValueError(
                    f"Layer {layer}: kernel shape {kernel.shape} != expected {expect}"
                )
            if bias.shape != (expect[1],):
                raise ValueError(
                    f"Layer {layer}: bias shape {bias.shape} != expected ({expect[1]},)"
                )
            # Keras kernel (in, out) -> torch (out, in): transpose
            with torch.no_grad():
                fc.weight.copy_(torch.from_numpy(kernel.T.copy()))
                fc.bias.copy_(torch.from_numpy(bias.copy()))

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("RefMLP loaded: %d parameters", n_params)
    return model


def _open_vars_tree(root_path: str | Path):
    """Open the ``Vars`` tree from a REF exam ROOT ntuple via uproot."""
    import uproot  # local import: optional dependency
    f = uproot.open(str(root_path))
    if "Vars" not in [k.split(";")[0] for k in f.keys()]:
        raise KeyError(f"{root_path} has no 'Vars' tree (keys={f.keys()})")
    return f["Vars"]


@torch.no_grad()
def predict_root(
    model: RefMLP,
    root_path: str | Path,
    *,
    max_events: Optional[int] = None,
    batch_size: int = 4096,
    device: str | torch.device = "cuda",
) -> Dict[str, np.ndarray]:
    """Run a RefMLP over var1..var75 of an exam ROOT ntuple.

    Returns a dict with keys::
        discriminator : (N,) sigmoid scores in [0, 1]
        target        : (N,) int (0=tT bkg, 1=DR1 sig)
        weight        : (N,) float
        sampleNumber  : (N,) int

    `target`, `weight`, `sampleNumber` are loaded only if present.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        dev = torch.device("cpu")
    model = model.to(dev).eval()

    tree = _open_vars_tree(root_path)
    var_names = [f"var{i}" for i in range(1, _INPUT_DIM + 1)]
    extras = [b for b in ("target", "weight", "sampleNumber") if b in tree.keys()]
    n_total = tree.num_entries
    if max_events is not None:
        n_total = min(n_total, int(max_events))

    feats_arr = tree.arrays(var_names, entry_stop=n_total, library="np")
    feats = np.stack([feats_arr[v].astype(np.float32) for v in var_names], axis=1)
    extras_arr = tree.arrays(extras, entry_stop=n_total, library="np") if extras else {}

    # finite-mask: REF training drops NaN rows; we should too for fairness
    finite = np.all(np.isfinite(feats), axis=1)
    if not finite.all():
        n_drop = int((~finite).sum())
        logger.warning("Dropping %d non-finite rows from %s", n_drop, root_path)
        feats = feats[finite]
        for k in extras_arr:
            extras_arr[k] = extras_arr[k][finite]

    n = feats.shape[0]
    discriminator = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        x = torch.from_numpy(feats[start:end]).to(dev, non_blocking=True)
        prob = model.predict_proba(x).cpu().numpy().astype(np.float32)
        discriminator[start:end] = prob

    out: Dict[str, np.ndarray] = {"discriminator": discriminator}
    if "target" in extras_arr:
        out["target"] = extras_arr["target"].astype(np.int64)
    if "weight" in extras_arr:
        out["weight"] = extras_arr["weight"].astype(np.float64)
    if "sampleNumber" in extras_arr:
        out["sampleNumber"] = extras_arr["sampleNumber"].astype(np.int64)
    return out


def auc_score(scores: np.ndarray, labels: np.ndarray,
              weights: Optional[np.ndarray] = None) -> float:
    """Weighted ROC AUC via sorted threshold sweep (no sklearn dep)."""
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels[order].astype(np.float64)
    w = (weights[order] if weights is not None else np.ones_like(y)).astype(np.float64)
    pos_w = (w * (y == 1)).sum()
    neg_w = (w * (y == 0)).sum()
    if pos_w == 0 or neg_w == 0:
        return float("nan")
    tps = np.cumsum(w * (y == 1)) / pos_w
    fps = np.cumsum(w * (y == 0)) / neg_w
    distinct = np.r_[np.diff(s) != 0, True]
    tpr = np.r_[0.0, tps[distinct]]
    fpr = np.r_[0.0, fps[distinct]]
    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(tpr, fpr))
