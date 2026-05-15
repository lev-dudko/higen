"""CLI: REF baseline inference on raw LHE ROOT samples (DR1 / tT / full tT_tWb).

Reads the 75 inputVarNames + means + sigmas from the REF training config
``tT_DR_Parton_PtJ4x10C.py`` (kept alongside the Keras h5 in pvolkov's
NN_RV_GG/tT_DR directory), recomputes the standardised var1..var75 from
the raw LHE branches via uproot expressions (which understands ``log(...)``
on the fly), applies the same pt_Jet4>10 cut as the training pipeline,
runs the RefMLP, and writes a 3-sample ``ref_predictions_lhe.npz``.

Why this is needed
------------------
Pvolkov has a ``tT_tWb_Parton_PtJ10_examFile_*.root`` with var1..var75
pre-baked, but those vars were standardised by means/sigmas of the
**DR1 vs full** training set, not by the DR1-vs-tT statistics that the
target Keras model was trained on. To get a fair REF discriminator on
the full sample we must apply *the same* preprocessing as the model
saw, i.e. the means/sigmas hardcoded in
``tT_DR_Parton_PtJ4x10C.py``.

Output npz schema (mirrors GATr-lite predictions.npz)
-----------------------------------------------------
    signal_scores        : (N_DR1,)
    signal_weights       : (N_DR1,)  -- xsec_pb / N events (we don't have
                                        xsec here; falls back to 1/N)
    signal_xsec_pb       : (1,)
    signal_n             : (1,)
    background_*         : same for tT
    extra_*              : same for full tT_tWb (label "tT_tWb (full)")
    auc                  : (1,) DR1 vs tT only

Usage
-----
    python -m paper1_demo.gatr_lite.scripts.ref_inference_lhe \\
        --keras-h5  demo/paper1/weights/ref_baseline_boos2023.h5 \\
        --ref-config <path>/tT_DR_Parton_PtJ4x10C.py \\
        --dr1  <path>/GG_DR1_NeeuDbB_RV_mu.root \\
        --tt   <path>/GG_tT_NeeuDbB_RV_mu.root \\
        --full <path>/GG_tT_tWb_NeeuDbB_RV_mu.root \\
        --out  ./runs/ref/predictions_lhe.npz \\
        --device cuda

The REF training config (``tT_DR_Parton_PtJ4x10C.py``) and the per-sample
ROOT ntuples are not redistributed in this repository.  Request access
from the authors of [Boos:2023kpp] (arXiv:2306.08793).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from paper1_demo.gatr_lite.training.ref_baseline import (
    RefMLP,
    auc_score,
    load_keras_h5,
)

logger = logging.getLogger(__name__)


def _load_ref_config(path: str | Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Import ``tT_DR_Parton_PtJ4x10C.py`` as a module and grab the lists."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location("ref_cfg", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load REF config: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = list(mod.inputVarNames)
    means = np.asarray(mod.means, dtype=np.float64)
    sigmas = np.asarray(mod.sigmas, dtype=np.float64)
    if not (len(names) == len(means) == len(sigmas) == 75):
        raise ValueError(
            f"REF config has inconsistent sizes: "
            f"vars={len(names)} means={len(means)} sigmas={len(sigmas)}"
        )
    return names, means, sigmas


def _load_lhe_features(
    root_path: str | Path,
    var_names: List[str],
    means: np.ndarray,
    sigmas: np.ndarray,
    cut_pt_jet4: float = 10.0,
    max_events: Optional[int] = None,
) -> np.ndarray:
    """Read the 75 RAW LHE expressions, apply pt_Jet4>cut, then standardise.

    Returns standardised feature matrix of shape (N, 75) float32.
    """
    import uproot
    f = uproot.open(str(root_path))
    keys = [k.split(";")[0] for k in f.keys()]
    if "LHE" not in keys:
        raise KeyError(f"{root_path}: no LHE tree (got {f.keys()})")
    tree = f["LHE"]

    needed = list(var_names) + ["pt_Jet4"]
    arrs = tree.arrays(needed, library="np", entry_stop=max_events)
    cols = [arrs[v].astype(np.float64) for v in var_names]
    X = np.stack(cols, axis=1)
    pt_j4 = arrs["pt_Jet4"].astype(np.float64)

    finite = np.all(np.isfinite(X), axis=1) & np.isfinite(pt_j4)
    cut = finite & (pt_j4 > cut_pt_jet4)
    n_drop = int((~cut).sum())
    if n_drop:
        logger.info("%s: dropping %d/%d events (pt_Jet4<=%g or non-finite)",
                    Path(root_path).name, n_drop, len(cut), cut_pt_jet4)
    X = X[cut]
    X_std = (X - means[None, :]) / sigmas[None, :]
    return X_std.astype(np.float32)


@torch.no_grad()
def _score(model: RefMLP, X: np.ndarray, *, batch_size: int, device: torch.device) -> np.ndarray:
    n = X.shape[0]
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        x = torch.from_numpy(X[s:e]).to(device, non_blocking=True)
        out[s:e] = model.predict_proba(x).cpu().numpy().astype(np.float32)
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="REF inference on raw LHE samples (DR1/tT/full).")
    p.add_argument("--keras-h5", required=True)
    p.add_argument("--ref-config", required=True,
                   help="Path to tT_DR_Parton_PtJ4x10C.py (provides inputVarNames+means+sigmas)")
    p.add_argument("--dr1", required=True, help="Raw LHE root for DR1 (signal)")
    p.add_argument("--tt", required=True, help="Raw LHE root for tT (background)")
    p.add_argument("--full", default=None, help="Raw LHE root for tT_tWb (full); optional")
    p.add_argument("--out", required=True)
    p.add_argument("--max-events", type=int, default=None,
                   help="Cap per-sample (debug)")
    p.add_argument("--cut-pt-jet4", type=float, default=10.0)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if device.type == "cpu" and args.device == "cuda":
        logger.warning("CUDA unavailable; falling back to CPU")

    var_names, means, sigmas = _load_ref_config(args.ref_config)
    logger.info("REF config: %d vars, sample means=%s ...", len(var_names),
                means[:3].tolist())

    model = load_keras_h5(args.keras_h5).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("RefMLP parameters: %d", n_params)

    samples: Dict[str, str] = {"signal": args.dr1, "background": args.tt}
    if args.full:
        samples["extra"] = args.full

    payload: Dict[str, np.ndarray] = {}
    score_cache: Dict[str, np.ndarray] = {}
    for tag, path in samples.items():
        logger.info("=== %s: %s", tag, path)
        X = _load_lhe_features(path, var_names, means, sigmas,
                               cut_pt_jet4=args.cut_pt_jet4, max_events=args.max_events)
        scores = _score(model, X, batch_size=args.batch_size, device=device)
        n = len(scores)
        weights = np.full(n, 1.0 / max(n, 1), dtype=np.float64)  # equal-area normalisation
        payload[f"{tag}_scores"] = scores.astype(np.float32)
        payload[f"{tag}_weights"] = weights
        payload[f"{tag}_xsec_pb"] = np.array([1.0], dtype=np.float64)  # unknown here
        payload[f"{tag}_n"] = np.array([n], dtype=np.int64)
        score_cache[tag] = scores
        logger.info("  scored %d events; mean=%.4f", n, float(scores.mean()) if n else 0.0)

    # AUC: signal=DR1 (label 1) vs background=tT (label 0)
    if "signal" in score_cache and "background" in score_cache:
        s_sig = score_cache["signal"]
        s_bkg = score_cache["background"]
        scores = np.concatenate([s_sig, s_bkg])
        labels = np.concatenate([np.ones(len(s_sig), dtype=np.int64),
                                 np.zeros(len(s_bkg), dtype=np.int64)])
        auc = auc_score(scores, labels)
        payload["auc"] = np.array([auc], dtype=np.float64)
        payload["n_params"] = np.array([n_params], dtype=np.int64)
        logger.info("REF AUC (DR1 vs tT, raw LHE) = %.4f", auc)
        print(f"REF AUC = {auc:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
