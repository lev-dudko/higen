"""CLI: run the REF baseline (Boos:2023kpp) on its exam ROOT ntuple.

Loads a Keras h5 checkpoint into a pure-PyTorch RefMLP, scores every
event of the exam ntuple, computes the AUC and writes a predictions
.npz that mirrors the GATr-lite ``predictions.npz`` schema (so the
overlay plotting script can consume both side-by-side).

Usage
-----
    python -m paper1_demo.gatr_lite.scripts.ref_inference \\
        --keras-h5 demo/paper1/weights/ref_baseline_boos2023.h5 \\
        --exam-root <path>/tT_DR_Parton_PtJ4x10_examFile_..._VZS.root \\
        --out ./runs/ref/predictions.npz \\
        --device cuda

The exam ROOT file is not redistributed in this repository.  Request
access from the authors of [Boos:2023kpp] (arXiv:2306.08793).

Output npz keys
---------------
    discriminator : (N,) float32   sigmoid score
    target        : (N,) int64     1 = DR1, 0 = tT
    weight        : (N,) float64   per-event weight (~1.0 in this ntuple)
    sampleNumber  : (N,) int64
    auc           : (1,) float64   weighted ROC AUC
    n             : (1,) int64
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from paper1_demo.gatr_lite.training.ref_baseline import (
    auc_score,
    load_keras_h5,
    predict_root,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run REF baseline (Keras h5) on its exam ROOT ntuple.")
    p.add_argument("--keras-h5", required=True, help="REF Keras h5 checkpoint")
    p.add_argument("--exam-root", required=True, help="REF exam ROOT ntuple (Vars tree)")
    p.add_argument("--out", required=True, help="Output predictions.npz path")
    p.add_argument("--max-events", type=int, default=None,
                   help="Optional cap on number of events (for debugging)")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        device = "cpu"

    model = load_keras_h5(args.keras_h5)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("RefMLP: %d parameters", n_params)

    pred = predict_root(
        model, args.exam_root,
        max_events=args.max_events,
        batch_size=args.batch_size,
        device=device,
    )

    if "target" not in pred:
        raise RuntimeError(f"{args.exam_root} has no 'target' branch; cannot compute AUC")

    auc = auc_score(pred["discriminator"], pred["target"], pred.get("weight"))
    logger.info("REF AUC = %.6f over %d events", auc, len(pred["discriminator"]))

    payload = {
        "discriminator": pred["discriminator"].astype(np.float32),
        "target": pred["target"].astype(np.int64),
        "weight": pred.get("weight", np.ones(len(pred["discriminator"]),
                                              dtype=np.float64)).astype(np.float64),
        "sampleNumber": pred.get("sampleNumber", np.zeros(len(pred["discriminator"]),
                                                          dtype=np.int64)).astype(np.int64),
        "auc": np.array([auc], dtype=np.float64),
        "n": np.array([len(pred["discriminator"])], dtype=np.int64),
        "n_params": np.array([n_params], dtype=np.int64),
    }
    np.savez(out_path, **payload)
    logger.info("Wrote %s (AUC=%.4f)", out_path, auc)
    print(f"REF AUC = {auc:.4f}  (N = {len(pred['discriminator'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
