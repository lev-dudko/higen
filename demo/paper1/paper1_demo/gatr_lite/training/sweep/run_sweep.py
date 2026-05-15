"""Sweep orchestrator for GATr-lite hyper-parameter search.

Phases
------
1. Smoke: every config is trained for ``--phase1-epochs`` epochs on
   ``--phase1-cap`` events per class. Cheap discrimination of bad configs.
2. Full: top-K configs (by Phase 1 best val AUC) are retrained for
   ``--phase2-epochs`` on ``--phase2-cap`` events per class.
3. Stretch: if the global best val AUC is still below ``--target``, the
   single best config is rerun for ``--phase3-epochs`` on
   ``--phase3-cap`` events per class.

A row is appended to ``summary.csv`` after each run. The script runs
all training jobs sequentially (single GPU is enough).

Defaults are tuned for DR1 vs tT, parton-level, 14 TeV on RTX 5060 Ti
(~53 s / epoch at the baseline config and 100k events per class).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from paper1_demo.gatr_lite.gatr.model import GATrLite, GATrLiteConfig
from paper1_demo.gatr_lite.data.dataset import (
    InMemoryPartonDataset, MultiH5Dataset, make_train_val_split,
)
from paper1_demo.gatr_lite.training.train import (
    _balanced_sampler, _collect_labels, _epoch_train, _epoch_eval,
)

from .configs import SWEEP_CONFIGS, DEFAULT_TRAIN, merged_train

logger = logging.getLogger("sweep")


# ---------------------------------------------------------------------------
# Single-run training
# ---------------------------------------------------------------------------

def _build_loaders(
    signal: str,
    background: str,
    train_args: Dict[str, Any],
    cap_per_class: int,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Build train + val DataLoaders for one sweep run.

    ``cap_per_class`` overrides the value coming from ``train_args``.
    """
    seed = int(train_args["seed"])
    val_frac = float(train_args["val_frac"])
    batch_size = int(train_args["batch_size"])
    num_workers = int(train_args["num_workers"])
    device_str = str(train_args.get("device", "cuda"))
    pin = device_str.startswith("cuda")

    rng = np.random.default_rng(seed)

    def _pick(h5_path: str, max_n: int) -> InMemoryPartonDataset:
        with h5py.File(h5_path, "r") as f:
            n_total = int(f["p4"].shape[0])
        if max_n and 0 < max_n < n_total:
            idx = np.sort(rng.choice(n_total, max_n, replace=False))
        else:
            idx = None
        return InMemoryPartonDataset(h5_path, indices=idx)

    ds_signal = _pick(signal, cap_per_class)
    ds_bkg = _pick(background, cap_per_class)
    ds_full = MultiH5Dataset([ds_signal, ds_bkg])

    train_sub, val_sub = make_train_val_split(ds_full, val_frac, seed)
    train_labels = _collect_labels(train_sub)
    val_labels = _collect_labels(val_sub)
    n_train_pos = int((train_labels == 1).sum())
    n_train_neg = int((train_labels == 0).sum())
    n_val_pos = int((val_labels == 1).sum())
    n_val_neg = int((val_labels == 0).sum())

    steps = int(train_args.get("steps_per_epoch") or 0) or len(train_sub)
    sampler = _balanced_sampler(train_labels, num_samples=steps)

    train_loader = DataLoader(
        train_sub, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_sub, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, val_loader, {
        "n_train_pos": n_train_pos, "n_train_neg": n_train_neg,
        "n_val_pos": n_val_pos, "n_val_neg": n_val_neg,
        "n_train": len(train_sub), "n_val": len(val_sub),
    }


def run_one(
    *,
    name: str,
    model_overrides: Dict[str, Any],
    train_overrides: Dict[str, Any],
    signal: str,
    background: str,
    out_dir: Path,
    epochs: int,
    cap_per_class: int,
) -> Dict[str, Any]:
    """Train one configuration and return a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    train_args = dict(DEFAULT_TRAIN)
    train_args.update(train_overrides)
    train_args["epochs"] = int(epochs)
    train_args["max_train_per_class"] = int(cap_per_class)

    seed = int(train_args["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_str = str(train_args.get("device", "cuda"))
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    train_loader, val_loader, ds_stats = _build_loaders(
        signal, background, train_args, cap_per_class,
    )

    model_cfg = GATrLiteConfig(**model_overrides)
    model = GATrLite(model_cfg).to(device)
    n_params = model.num_parameters()

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_args["lr"]),
        weight_decay=float(train_args["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(int(train_args["epochs"]), 1),
    )
    loss_fn = nn.BCEWithLogitsLoss()

    payload_cfg = {
        "name": name,
        "model_overrides": model_overrides,
        "train_overrides": train_overrides,
        "model_cfg": asdict(model_cfg),
        "train_args": train_args,
        "n_params": n_params,
        "device": str(device),
        "torch_version": torch.__version__,
        **ds_stats,
    }
    with (out_dir / "config.json").open("w") as f:
        json.dump(payload_cfg, f, indent=2)

    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="") as csv_f:
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["epoch", "train_loss", "val_loss", "val_auc",
                        "val_brier", "lr", "epoch_seconds"])

        best_auc = -math.inf
        best_epoch = -1
        no_improve = 0
        patience = int(train_args["early_stop_patience"])

        print(
            f"[{name}] start: params={n_params}  "
            f"train={ds_stats['n_train']}  val={ds_stats['n_val']}  "
            f"epochs={train_args['epochs']}  device={device}",
            flush=True,
        )

        last_state: Dict[str, Any] | None = None

        t_run = time.time()
        for epoch in range(1, int(train_args["epochs"]) + 1):
            t0 = time.time()
            train_loss = _epoch_train(model, train_loader, optim, loss_fn, device)
            val_loss, val_auc, val_brier = _epoch_eval(
                model, val_loader, loss_fn, device,
            )
            scheduler.step()
            dt = time.time() - t0
            lr_now = optim.param_groups[0]["lr"]

            print(
                f"[{name}] epoch {epoch:03d}/{train_args['epochs']}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_auc={val_auc:.4f}  val_brier={val_brier:.4f}  "
                f"lr={lr_now:.2e}  dt={dt:.1f}s",
                flush=True,
            )
            csv_w.writerow([epoch, train_loss, val_loss, val_auc,
                            val_brier, lr_now, dt])
            csv_f.flush()

            last_state = {
                "epoch": epoch,
                "model_state": (
                    model._orig_mod.state_dict()
                    if hasattr(model, "_orig_mod") else model.state_dict()
                ),
                "val_auc": val_auc,
                "val_loss": val_loss,
                "config": payload_cfg,
            }
            torch.save(last_state, out_dir / "last.pt")

            if not math.isnan(val_auc) and val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                no_improve = 0
                torch.save(last_state, out_dir / "best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[{name}] early stop at epoch {epoch}", flush=True)
                    break

    elapsed = time.time() - t_run
    summary = {
        "name": name,
        "best_val_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "n_params": int(n_params),
        "epochs_run": epoch,
        "elapsed_seconds": elapsed,
        "out_dir": str(out_dir),
        **{f"model.{k}": v for k, v in model_overrides.items()},
        **{f"train.{k}": v for k, v in train_overrides.items()},
        "cap_per_class": int(cap_per_class),
        "epochs_target": int(train_args["epochs"]),
    }
    print(
        f"[{name}] done. best_val_auc={best_auc:.4f} (epoch {best_epoch}); "
        f"elapsed={elapsed/60:.1f} min",
        flush=True,
    )

    # Free GPU memory before next run.
    del model, optim, scheduler, loss_fn, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def _append_summary(summary_path: Path, row: Dict[str, Any]) -> None:
    """Append a row to the global summary.csv (creates header if absent)."""
    new_file = not summary_path.exists()
    # Stable column order: identifiers first, results in the middle, params last.
    fixed = ["phase", "name", "best_val_auc", "best_epoch", "epochs_run",
             "epochs_target", "cap_per_class", "n_params", "elapsed_seconds",
             "out_dir"]
    extras = sorted(k for k in row.keys() if k not in fixed)
    cols = fixed + extras
    with summary_path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(cols)
        w.writerow([row.get(c, "") for c in cols])


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GATr-lite sweep orchestrator")
    p.add_argument("--signal",
                   default="./data/dr1.h5")
    p.add_argument("--background",
                   default="./data/tT.h5")
    p.add_argument("--out",
                   default="./runs/sweep_001")
    p.add_argument("--target", type=float, default=0.96,
                   help="Stop Phase 3 if Phase 2 best already exceeds this AUC")
    p.add_argument("--phase1-epochs", type=int, default=10)
    p.add_argument("--phase1-cap", type=int, default=30_000)
    p.add_argument("--phase2-epochs", type=int, default=50)
    p.add_argument("--phase2-cap", type=int, default=100_000)
    p.add_argument("--phase2-topk", type=int, default=3)
    p.add_argument("--phase3-epochs", type=int, default=50)
    p.add_argument("--phase3-cap", type=int, default=200_000)
    p.add_argument("--skip-phase1", action="store_true")
    p.add_argument("--skip-phase2", action="store_true")
    p.add_argument("--skip-phase3", action="store_true")
    return p


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.csv"

    print(f"[sweep] root={out_root}  configs={len(SWEEP_CONFIGS)}", flush=True)
    print(f"[sweep] data signal={args.signal}  background={args.background}",
          flush=True)
    print(f"[sweep] target val AUC = {args.target:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Phase 1 - smoke
    # ------------------------------------------------------------------
    phase1_summaries: List[Dict[str, Any]] = []
    if not args.skip_phase1:
        print(f"[sweep] PHASE 1: smoke ({args.phase1_epochs} epochs, "
              f"cap={args.phase1_cap})", flush=True)
        for cfg in SWEEP_CONFIGS:
            name = cfg["name"]
            run_dir = out_root / "phase1" / name
            try:
                summary = run_one(
                    name=name,
                    model_overrides=cfg.get("model", {}),
                    train_overrides=cfg.get("train", {}),
                    signal=args.signal,
                    background=args.background,
                    out_dir=run_dir,
                    epochs=args.phase1_epochs,
                    cap_per_class=args.phase1_cap,
                )
            except Exception as exc:  # pragma: no cover - runtime safeguard
                logger.exception("[%s] crashed: %s", name, exc)
                summary = {
                    "name": name,
                    "best_val_auc": float("nan"),
                    "best_epoch": -1,
                    "n_params": 0,
                    "epochs_run": 0,
                    "elapsed_seconds": 0.0,
                    "out_dir": str(run_dir),
                    "error": str(exc),
                }
            row = {"phase": "1", **summary}
            _append_summary(summary_path, row)
            phase1_summaries.append(summary)

        print("[sweep] phase1 ranking:", flush=True)
        ranked = sorted(
            phase1_summaries,
            key=lambda s: (s.get("best_val_auc", float("nan")) if not math.isnan(s.get("best_val_auc", float("nan"))) else -1),
            reverse=True,
        )
        for s in ranked:
            print(f"  {s['name']:<12s}  AUC={s['best_val_auc']:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Phase 2 - full retrain of top-K
    # ------------------------------------------------------------------
    phase2_summaries: List[Dict[str, Any]] = []
    if not args.skip_phase2:
        if phase1_summaries:
            ranked = sorted(
                phase1_summaries,
                key=lambda s: (
                    -1.0 if math.isnan(s.get("best_val_auc", float("nan")))
                    else s["best_val_auc"]
                ),
                reverse=True,
            )
            top_names = [s["name"] for s in ranked[:args.phase2_topk]]
        else:
            top_names = [c["name"] for c in SWEEP_CONFIGS[:args.phase2_topk]]

        print(f"[sweep] PHASE 2: full {args.phase2_epochs} epochs "
              f"(cap={args.phase2_cap}); top configs = {top_names}",
              flush=True)
        cfg_by_name = {c["name"]: c for c in SWEEP_CONFIGS}
        for name in top_names:
            cfg = cfg_by_name[name]
            run_dir = out_root / "phase2" / name
            train_overrides = dict(cfg.get("train", {}))
            # Honor per-config overrides except cap_per_class which Phase 2 owns.
            train_overrides.pop("max_train_per_class", None)
            try:
                summary = run_one(
                    name=name,
                    model_overrides=cfg.get("model", {}),
                    train_overrides=train_overrides,
                    signal=args.signal,
                    background=args.background,
                    out_dir=run_dir,
                    epochs=args.phase2_epochs,
                    cap_per_class=args.phase2_cap,
                )
            except Exception as exc:
                logger.exception("[%s] phase2 crashed: %s", name, exc)
                summary = {
                    "name": name,
                    "best_val_auc": float("nan"),
                    "best_epoch": -1,
                    "n_params": 0,
                    "epochs_run": 0,
                    "elapsed_seconds": 0.0,
                    "out_dir": str(run_dir),
                    "error": str(exc),
                }
            row = {"phase": "2", **summary}
            _append_summary(summary_path, row)
            phase2_summaries.append(summary)

        print("[sweep] phase2 ranking:", flush=True)
        for s in sorted(phase2_summaries,
                        key=lambda s: s.get("best_val_auc", -1), reverse=True):
            print(f"  {s['name']:<12s}  AUC={s['best_val_auc']:.4f}",
                  flush=True)

    # ------------------------------------------------------------------
    # Phase 3 - stretch run if we still missed the target
    # ------------------------------------------------------------------
    if not args.skip_phase3:
        if phase2_summaries:
            best = max(phase2_summaries,
                       key=lambda s: (-1.0 if math.isnan(s.get("best_val_auc", float("nan")))
                                      else s["best_val_auc"]))
            best_name = best["name"]
            best_auc = best["best_val_auc"]
        else:
            best_name = SWEEP_CONFIGS[0]["name"]
            best_auc = float("nan")

        if not math.isnan(best_auc) and best_auc >= args.target:
            print(f"[sweep] target met (best_val_auc={best_auc:.4f} >= "
                  f"{args.target:.4f}); skipping Phase 3", flush=True)
        else:
            cfg_by_name = {c["name"]: c for c in SWEEP_CONFIGS}
            cfg = cfg_by_name[best_name]
            run_dir = out_root / "phase3" / best_name
            train_overrides = dict(cfg.get("train", {}))
            train_overrides.pop("max_train_per_class", None)
            print(f"[sweep] PHASE 3: stretch run on {best_name} "
                  f"({args.phase3_epochs} epochs, cap={args.phase3_cap})",
                  flush=True)
            try:
                summary = run_one(
                    name=best_name,
                    model_overrides=cfg.get("model", {}),
                    train_overrides=train_overrides,
                    signal=args.signal,
                    background=args.background,
                    out_dir=run_dir,
                    epochs=args.phase3_epochs,
                    cap_per_class=args.phase3_cap,
                )
            except Exception as exc:
                logger.exception("[%s] phase3 crashed: %s", best_name, exc)
                summary = {
                    "name": best_name,
                    "best_val_auc": float("nan"),
                    "best_epoch": -1,
                    "n_params": 0,
                    "epochs_run": 0,
                    "elapsed_seconds": 0.0,
                    "out_dir": str(run_dir),
                    "error": str(exc),
                }
            row = {"phase": "3", **summary}
            _append_summary(summary_path, row)
            print(f"[sweep] phase3 done: best_val_auc={summary['best_val_auc']:.4f}",
                  flush=True)

    print(f"[sweep] all phases done. summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
