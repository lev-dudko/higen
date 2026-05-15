"""Evaluation utilities for GATr-lite.

Provides
--------
load_model(checkpoint_path, device)
    Reconstruct GATrLite from a `last.pt`/`best.pt` produced by training/train.py.

predict_h5(model, h5_path, device, batch_size)
    Run the model on every event of a preprocessed HDF5 file and return
    (scores, labels, xsec_pb, weights) where weights = xsec_pb / N_events.

CLI
---
    python -m paper1_demo.gatr_lite.training.eval \
        --checkpoint /path/to/run_001/best.pt \
        --signal     /path/to/dr1.h5 \
        --background /path/to/tT.h5 \
        --extra      /path/to/full.h5 \
        --out-dir    /path/to/run_001/eval

(produces predictions.npz with per-sample scores, labels, weights)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from paper1_demo.gatr_lite.data.dataset import InMemoryPartonDataset, PartonH5Dataset
from paper1_demo.gatr_lite.gatr.model import GATrLite, GATrLiteConfig

logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
    cfg_patch: dict | None = None,
    cfg_override: GATrLiteConfig | None = None,
) -> GATrLite:
    """Re-instantiate GATrLite and load weights from checkpoint.

    Config resolution order (highest priority first):
      1. ``cfg_override`` (full replacement, for tests / programmatic use)
      2. ckpt[``model_cfg``] (modern train.py saves this)
      3. ckpt[``args``] mapped through GATrLiteConfig (legacy fallback)
      4. GATrLiteConfig defaults

    ``cfg_patch`` is then applied as a *patch* on top of the chosen base,
    so CLI overrides like ``--cfg-no-join-block`` change only the named
    field and leave the rest of the ckpt-recorded config intact.
    """
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    # strip torch.compile prefix if present
    state = { (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
              for k, v in state.items() }

    if cfg_override is not None:
        cfg = cfg_override
    elif isinstance(ckpt, dict) and "model_cfg" in ckpt and isinstance(ckpt["model_cfg"], dict):
        cfg = GATrLiteConfig(**ckpt["model_cfg"])
    elif isinstance(ckpt, dict) and "args" in ckpt and isinstance(ckpt["args"], dict):
        cfg = _cfg_from_args(ckpt["args"])
    else:
        cfg = GATrLiteConfig()

    if cfg_patch:
        cfg_fields = {f.name for f in fields(GATrLiteConfig)}
        bad = set(cfg_patch) - cfg_fields
        if bad:
            raise ValueError(f"Unknown GATrLiteConfig fields in cfg_patch: {sorted(bad)}")
        cfg = GATrLiteConfig(**{**asdict(cfg), **cfg_patch})

    # If channels recorded in ckpt cfg disagree with state shapes, infer from
    # state -- helps with older checkpoints that saved a stale default cfg.
    inferred_channels = _infer_channels_from_state(state)
    if inferred_channels is not None and inferred_channels != cfg.channels:
        logger.warning(
            "ckpt cfg channels=%d but state implies channels=%d; using state.",
            cfg.channels, inferred_channels,
        )
        cfg = GATrLiteConfig(**{**asdict(cfg), "channels": inferred_channels})

    model = GATrLite(cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info("Loaded model: %s", asdict(cfg))
    return model


def _infer_channels_from_state(state: dict) -> int | None:
    """Infer the ``channels`` field from a state_dict by inspecting the input
    projection weight shape. Returns None if the expected key is missing.
    """
    for key in ("input_proj.weights.0", "input_proj.linears.0.weight",
                "input_proj.weight", "input_proj.bias_scalar"):
        if key in state:
            w = state[key]
            return int(w.shape[0])
    return None


def _cfg_from_args(args_dict: dict) -> GATrLiteConfig:
    """Reconstruct GATrLiteConfig from a stored train-time ``args`` dict.

    Used for legacy ckpts that pre-date the explicit ``model_cfg`` field in
    the checkpoint. We map only the keys that exist on GATrLiteConfig so
    unknown CLI fields (lr, batch_size, seed, ...) are silently dropped.
    """
    cfg_fields = {f.name for f in fields(GATrLiteConfig)}
    overrides = {k: v for k, v in args_dict.items() if k in cfg_fields}
    return GATrLiteConfig(**overrides)


@torch.no_grad()
def predict_h5(
    model: GATrLite,
    h5_path: str | Path,
    device: torch.device,
    batch_size: int = 1024,
    num_workers: int = 2,
) -> Dict[str, np.ndarray | float]:
    """Run inference on every event of an HDF5 file.

    Returns a dict with keys: scores, labels, xsec_pb, n_events, weights.
    """
    # In-memory load: random/sequential access through gzip-chunked HDF5 is
    # unusably slow at scale (per-item __getitem__ stalls), so we eagerly
    # materialise the arrays once and iterate from RAM.
    ds = InMemoryPartonDataset(h5_path)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    scores: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for batch in loader:
        p4 = batch["p4"].to(device, non_blocking=True)
        pdg = batch["pdg"].to(device, non_blocking=True)
        logits = model(p4, pdg).squeeze(-1)
        scores.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels.append(batch["label"].numpy().astype(np.int64))
    scores_arr = np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)
    labels_arr = np.concatenate(labels) if labels else np.zeros(0, dtype=np.int64)
    n = len(ds)
    xsec = ds.xsec_pb
    weight = (xsec / n) if n > 0 else 0.0
    weights = np.full(n, weight, dtype=np.float64)
    return {
        "scores":   scores_arr,
        "labels":   labels_arr,
        "xsec_pb":  float(xsec),
        "n_events": int(n),
        "weights":  weights,
        "h5_path":  str(h5_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate GATr-lite on parton HDF5 files.")
    p.add_argument("--checkpoint", required=True, help="Path to best.pt / last.pt")
    p.add_argument("--signal",     required=True, help="Signal HDF5 (DR1)")
    p.add_argument("--background", required=True, help="Background HDF5 (tT)")
    p.add_argument("--extra",      default=None,  help="Extra HDF5 (e.g. full tT_tWb)")
    p.add_argument("--out-dir",    required=True, help="Directory to write predictions.npz")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    # Manual config override for legacy ckpts that saved a stale default cfg.
    p.add_argument("--cfg-channels",   type=int, default=None)
    p.add_argument("--cfg-n-blocks",   type=int, default=None)
    p.add_argument("--cfg-arch-v2",    action="store_true")
    p.add_argument("--cfg-n-heads",    type=int, default=None)
    p.add_argument("--cfg-dropout",    type=float, default=None)
    p.add_argument("--cfg-input-dropout", type=float, default=None)
    p.add_argument("--cfg-gp-grade3-mixing", action="store_true")
    # Tri-state ablation flags (None = take from ckpt; True/False = override).
    p.add_argument("--cfg-no-join-block",   dest="cfg_use_join_block",
                   action="store_const", const=False, default=None)
    p.add_argument("--cfg-use-join-block",  dest="cfg_use_join_block",
                   action="store_const", const=True)
    p.add_argument("--cfg-no-pairing-cache",  dest="cfg_use_pairing_cache",
                   action="store_const", const=False, default=None)
    p.add_argument("--cfg-use-pairing-cache", dest="cfg_use_pairing_cache",
                   action="store_const", const=True)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info("Device: %s", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cli_patch = {k: v for k, v in [
        ("channels", args.cfg_channels),
        ("n_blocks", args.cfg_n_blocks),
        ("arch_v2",  True if args.cfg_arch_v2 else None),
        ("n_heads",  args.cfg_n_heads),
        ("dropout",  args.cfg_dropout),
        ("input_dropout", args.cfg_input_dropout),
        ("gp_grade3_mixing", True if args.cfg_gp_grade3_mixing else None),
        ("use_join_block",    args.cfg_use_join_block),
        ("use_pairing_cache", args.cfg_use_pairing_cache),
    ] if v is not None}
    model = load_model(args.checkpoint, device, cfg_patch=cli_patch or None)
    logger.info("Loaded model from %s", args.checkpoint)

    samples = {"signal": args.signal, "background": args.background}
    if args.extra:
        samples["extra"] = args.extra

    payload: Dict[str, np.ndarray] = {}
    for name, path in samples.items():
        logger.info("Predicting %s: %s", name, path)
        out = predict_h5(model, path, device, args.batch_size, args.num_workers)
        payload[f"{name}_scores"]  = out["scores"]
        payload[f"{name}_labels"]  = out["labels"]
        payload[f"{name}_weights"] = out["weights"]
        payload[f"{name}_xsec_pb"] = np.array([out["xsec_pb"]], dtype=np.float64)
        payload[f"{name}_n"]       = np.array([out["n_events"]], dtype=np.int64)

    np.savez(out_dir / "predictions.npz", **payload)
    logger.info("Wrote %s", out_dir / "predictions.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
