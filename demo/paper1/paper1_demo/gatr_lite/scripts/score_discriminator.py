#!/usr/bin/env python3
"""Score trained GATr-lite weights on the three parton-level samples, for the
discriminator-distribution figure.

Two things this does that the earlier aggregation did not:

  * final-epoch weights (last.pt), matching the protocol of Tab. 3, instead of
    the best-validation checkpoint that was selected BY the reported metric;
  * for the two training samples it scores only the events held back from that
    run's training, recovered from the seed. The full matrix-element sample was
    never trained on, so all of it is used.

Writes one npz per run with the three score arrays and their event counts.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from paper1_demo.gatr_lite.data.dataset import InMemoryPartonDataset, MultiH5Dataset
from paper1_demo.gatr_lite.scripts.aggregate_ablation import val_indices
from paper1_demo.gatr_lite.training.eval import load_model

logger = logging.getLogger("score_disc")


@torch.no_grad()
def score(model, ds, idx, device, batch_size):
    out = []
    for start in range(0, len(idx), batch_size):
        sel = idx[start:start + batch_size]
        items = [ds[int(i)] for i in sel]
        p4 = torch.stack([it["p4"] for it in items]).to(device)
        pdg = torch.stack([it["pdg"] for it in items]).to(device)
        out.append(torch.sigmoid(model(p4, pdg).squeeze(-1)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="run directory holding last.pt and config.json")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--all-events", action="store_true",
                   help="Score every event in each sample instead of only the events held "
                        "back from this run's training. Legitimate for a distribution SHAPE "
                        "(the signal file is small, so the held-back subset alone gives a "
                        "visibly ragged histogram), but it means the signal and background "
                        "curves include events the model was trained on: never quote a "
                        "figure of merit from these arrays. The held-back fraction is "
                        "recorded in the output so the caption can state it.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run = Path(args.run)
    cfg = json.load(open(run / "config.json"))
    seed = int(cfg["args"]["seed"])
    val_frac = float(cfg["args"]["val_frac"])
    device = torch.device(args.device)

    model = load_model(run / "last.pt", device).eval()

    data = Path(args.data_dir)
    res = {}

    # Must mirror training/train.py exactly: signal file first, background
    # second, concatenated by MultiH5Dataset, because that global index order is
    # what the split permutes. Asserted below against the counts in config.json.
    n_sig = len(InMemoryPartonDataset(str(data / "dr1.h5")))
    ds = MultiH5Dataset([
        InMemoryPartonDataset(str(data / "dr1.h5")),
        InMemoryPartonDataset(str(data / "tT.h5")),
    ])
    val_idx = val_indices(len(ds), val_frac, seed)
    n_pos = int((val_idx < n_sig).sum())
    n_neg = int(len(val_idx) - n_pos)
    exp_pos, exp_neg = int(cfg["n_val_pos"]), int(cfg["n_val_neg"])
    if (n_pos, n_neg) != (exp_pos, exp_neg):
        raise SystemExit(
            f"{run.name}: recovered split does not match the training run "
            f"(got pos={n_pos} neg={n_neg}, config.json records pos={exp_pos} "
            f"neg={exp_neg}). Refusing to plot a distribution from the wrong events."
        )

    # The two training samples. By default only the events held back from this
    # run; with --all-events the whole sample, for a smooth distribution shape.
    for name, is_sig in (("signal", True), ("background", False)):
        held = val_idx[(val_idx < n_sig) if is_sig else (val_idx >= n_sig)]
        if args.all_events:
            sel = np.arange(0, n_sig) if is_sig else np.arange(n_sig, len(ds))
        else:
            sel = held
        s = score(model, ds, sel, device, args.batch_size)
        res[f"{name}_scores"] = s
        res[f"{name}_n"] = np.array([len(s)])
        res[f"{name}_n_heldback"] = np.array([len(held)])
        logger.info("%s: %d events scored (%d of them held back from training), mean score %.4f",
                    name, len(s), len(held), s.mean())

    # The full matrix-element sample was never trained on, so all of it is used.
    ds_full = InMemoryPartonDataset(str(data / "full.h5"))
    s = score(model, ds_full, np.arange(len(ds_full)), device, args.batch_size)
    res["extra_scores"] = s
    res["extra_n"] = np.array([len(s)])
    logger.info("full ME: %d events (none seen in training), mean score %.4f", len(s), s.mean())

    res["config"] = np.array([run.name])
    res["scored"] = np.array(["all_events" if args.all_events else "held_back_only"])
    np.savez_compressed(args.out, **res)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
