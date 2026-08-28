"""Training loop for GATr-lite binary classifier (DR1 vs tT).

CLI
---
    python -m paper1_demo.gatr_lite.training.train \
        --signal     /path/to/dr1.h5 \
        --background /path/to/tT.h5 \
        --out        /path/to/runs/run_001 \
        --batch-size 512 --epochs 50 --lr 1e-3 --weight-decay 1e-4 \
        --val-frac 0.1 --seed 42 --device cuda \
        --early-stop-patience 15 --max-train-per-class 0

Outputs (under --out)
---------------------
    best.pt       -- best-AUC checkpoint
    last.pt       -- last-epoch checkpoint
    metrics.csv   -- per-epoch metrics
    config.json   -- CLI args + commit hash + dataset sizes
    tb/           -- TensorBoard logs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    _TB_AVAILABLE = True
except Exception:  # pragma: no cover -- tensorboard optional
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False

from paper1_demo.gatr_lite.data.dataset import (
    PartonH5Dataset, make_train_val_split,
)
from paper1_demo.gatr_lite.gatr.model import GATrLite, GATrLiteConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U based AUC (no sklearn dependency)."""
    if labels.min() == labels.max():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # average ranks across ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            mean_rank = ranks[order[i:j]].mean()
            ranks[order[i:j]] = mean_rank
        i = j
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    sum_pos_ranks = ranks[labels == 1].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


# ---------------------------------------------------------------------------
# Sampler / loader helpers
# ---------------------------------------------------------------------------

def _balanced_sampler(labels: np.ndarray, num_samples: int) -> WeightedRandomSampler:
    """WeightedRandomSampler giving equal probability to each class on average."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Need both classes; got n_pos={n_pos} n_neg={n_neg}")
    w_pos = 0.5 / n_pos
    w_neg = 0.5 / n_neg
    weights = np.where(labels == 1, w_pos, w_neg).astype(np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=num_samples,
        replacement=True,
    )


def _collect_labels(subset: Subset) -> np.ndarray:
    """Read labels for a Subset of a (Multi)PartonH5Dataset without iterating items."""
    base = subset.dataset
    indices = np.asarray(subset.indices, dtype=np.int64)
    # Unwrap nested Subset (e.g. cap-per-class Subset wrapped by train/val split Subset)
    while isinstance(base, Subset):
        parent_idx = np.asarray(base.indices, dtype=np.int64)
        indices = parent_idx[indices]
        base = base.dataset
    # InMemoryPartonDataset / MultiH5Dataset of InMemoryPartonDatasets: labels in RAM
    if hasattr(base, "_datasets") and all(
        hasattr(ds, "labels") for ds in base._datasets
    ):
        parts = [ds.labels for ds in base._datasets]
        return np.concatenate(parts)[indices]
    if hasattr(base, "labels"):
        return np.asarray(base.labels)[indices]
    # Direct numpy-backed access via the underlying H5 file(s) -- avoids per-item I/O.
    # MultiH5Dataset / PartonH5Dataset both end at PartonH5Dataset on h5py.
    if hasattr(base, "_datasets"):  # MultiH5Dataset
        # build label vector in global order then index
        labels_all = np.empty(len(base), dtype=np.int64)
        offset = 0
        for ds in base._datasets:
            n = len(ds)
            ds._open()
            assert ds._h5 is not None
            labels_all[offset:offset + n] = ds._h5["label"][:].astype(np.int64)
            offset += n
        return labels_all[indices]
    # plain PartonH5Dataset
    base._open()
    assert base._h5 is not None
    return base._h5["label"][:].astype(np.int64)[indices]


# ---------------------------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------------------------

def _move_batch(batch: Dict, device: torch.device) -> Dict:
    return {
        "p4":   batch["p4"].to(device,   non_blocking=True),
        "pdg":  batch["pdg"].to(device,  non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True).float(),
    }


def _epoch_train(model, loader, optim, loss_fn, device, grad_clip: float = 1.0) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        b = _move_batch(batch, device)
        logits = model(b["p4"], b["pdg"]).squeeze(-1)
        loss = loss_fn(logits, b["label"])
        optim.zero_grad(set_to_none=True)
        loss.backward()
        # Clip gradient norm. Without this a rare high-energy outlier in tT
        # (E ~ TeV) can produce huge geometric-product activations that
        # explode loss to >100 and trigger non-recovery weight collapse.
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optim.step()
        total += float(loss.detach()) * b["label"].numel()
        n += b["label"].numel()
    return total / max(n, 1)


@torch.no_grad()
def _epoch_eval(model, loader, loss_fn, device) -> Tuple[float, float, float]:
    model.eval()
    total = 0.0
    n = 0
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for batch in loader:
        b = _move_batch(batch, device)
        logits = model(b["p4"], b["pdg"]).squeeze(-1)
        loss = loss_fn(logits, b["label"])
        total += float(loss) * b["label"].numel()
        n += b["label"].numel()
        all_scores.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(b["label"].detach().cpu().numpy().astype(np.int64))
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    labels = np.concatenate(all_labels) if all_labels else np.zeros(0, dtype=np.int64)
    auc = _roc_auc(scores, labels) if len(scores) else float("nan")
    brier = _brier(scores, labels) if len(scores) else float("nan")
    return total / max(n, 1), auc, brier


# ---------------------------------------------------------------------------
# Subsetting per-class cap
# ---------------------------------------------------------------------------

def _cap_per_class(dataset, max_per_class: int, seed: int) -> Subset:
    """Return Subset with at most max_per_class items of each class."""
    base = dataset
    # gather labels
    n = len(base)
    if hasattr(base, "_datasets"):
        labels = np.empty(n, dtype=np.int64)
        offset = 0
        for ds in base._datasets:
            ds._open()
            assert ds._h5 is not None
            sl = ds._h5["label"][:].astype(np.int64)
            labels[offset:offset + len(sl)] = sl
            offset += len(sl)
    else:
        base._open()
        assert base._h5 is not None
        labels = base._h5["label"][:].astype(np.int64)

    rng = np.random.default_rng(seed)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    if max_per_class > 0:
        if len(pos) > max_per_class:
            pos = rng.choice(pos, size=max_per_class, replace=False)
        if len(neg) > max_per_class:
            neg = rng.choice(neg, size=max_per_class, replace=False)
    indices = np.concatenate([pos, neg])
    rng.shuffle(indices)
    return Subset(base, indices.tolist())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train GATr-lite binary classifier.")
    p.add_argument("--signal",     required=True, help="Signal HDF5 (label=1)")
    p.add_argument("--background", required=True, help="Background HDF5 (label=0)")
    p.add_argument("--out",        required=True, help="Output directory")
    p.add_argument("--batch-size", type=int,   default=4096,
                   help="Per-step batch size. With balanced sampler, default "
                        "4096 gives ~130 steps/epoch on full data — enough for "
                        "AdamW convergence at lr scaled linearly with batch.")
    p.add_argument("--epochs",     type=int,   default=100,
                   help="Max epochs; early stop usually triggers earlier.")
    p.add_argument("--lr",         type=float, default=8e-3,
                   help="Peak learning rate. Linear-scaling rule: lr=8e-3 "
                        "matches the legacy bs=512 lr=1e-3 point under "
                        "bs=4096 (×8 batch ⇒ ×8 lr). With --warmup-epochs > 0, "
                        "lr ramps from 0 to this peak over the first epochs.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac",   type=float, default=0.1)
    p.add_argument("--train-frac", type=float, default=1.0,
                   help="Keep only this fraction of the TRAINING subset, for "
                        "the low-statistics scan. The validation subset is left "
                        "untouched, so a reduced-statistics run is scored on "
                        "exactly the same events as the full-statistics run of "
                        "the same seed and the paired comparison stays valid. "
                        "The kept training events are the first n_keep of the "
                        "permutation that already defines the split, so the "
                        "1/10 subset is a subset of the 1/3 subset — the scan "
                        "is nested rather than independent draws.")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="Linear-warmup epochs from lr=0 to peak --lr; needed "
                        "for big-batch stable starts.")
    p.add_argument("--early-stop-patience", type=int, default=15,
                   help="Epochs without composite-metric improvement before stop.")
    p.add_argument("--early-stop-min-delta", type=float, default=5e-4,
                   help="Improvement threshold for early stop "
                        "(composite metric). Smaller deltas count as 'no progress'.")
    p.add_argument("--early-stop-warmup", type=int, default=25,
                   help="Minimum epochs before early stop can trigger.")
    p.add_argument("--early-stop-metric", default="auc-brier",
                   choices=["auc", "auc-brier", "auc-loss"],
                   help="Stopping metric. 'auc' uses val_auc only. "
                        "'auc-brier' = val_auc - 0.5*val_brier (rewards "
                        "calibration). 'auc-loss' = val_auc - 0.1*max(0, "
                        "val_loss - val_loss_min) (penalises overfit). "
                        "best.pt is always saved by pure val_auc.")
    p.add_argument("--no-balanced-sampler", action="store_true",
                   help="Disable balanced sampling. Uses natural class "
                        "distribution with pos_weight in BCE loss instead. "
                        "Steps/epoch = N_train / batch_size.")
    p.add_argument("--max-train-per-class",  type=int, default=0,
                   help="Cap train+val per class (0=no cap)")
    p.add_argument("--num-workers",          type=int, default=2)
    p.add_argument("--steps-per-epoch",      type=int, default=0,
                   help="Override sampler num_samples per epoch (0=auto)")
    p.add_argument("--compile",              action="store_true",
                   help="Use torch.compile (default off; some sm_120 builds break)")
    p.add_argument("--log-level",            default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    # Architecture (v2 -- guarded by arch_v2 flag)
    p.add_argument("--arch-v2",              action="store_true",
                   help="Enable arch_v2 (multi-head attn, dropout, gp3 nudge)")
    p.add_argument("--n-heads",              type=int,   default=1,
                   help="arch_v2: scalar attention heads (channels %% n_heads == 0)")
    p.add_argument("--channels",             type=int,   default=32)
    p.add_argument("--n-blocks",             type=int,   default=3)
    p.add_argument("--gp-mid-channels",      type=int,   default=16)
    p.add_argument("--attn-channels",        type=int,   default=16)
    p.add_argument("--head-hidden",          type=int,   default=96)
    p.add_argument("--dropout",              type=float, default=0.0)
    p.add_argument("--input-dropout",        type=float, default=0.0)
    p.add_argument("--gp-grade3-mixing",     action="store_true")
    p.add_argument("--grad-clip",            type=float, default=1.0,
                   help="Clip per-step gradient norm (0 disables)")
    p.add_argument("--use-join-block",       dest="use_join_block",
                   action="store_true", default=True,
                   help="Append a JoinBlock after the standard stack (Group B).")
    p.add_argument("--no-join-block",        dest="use_join_block",
                   action="store_false",
                   help="Disable the JoinBlock (ablation).")
    p.add_argument("--use-pairing-cache",    dest="use_pairing_cache",
                   action="store_true", default=True,
                   help="Append the two event-level pairing tokens (default).")
    p.add_argument("--no-pairing-cache",     dest="use_pairing_cache",
                   action="store_false",
                   help="Disable the two event-level pairing tokens "
                        "(ablation: per-particle backbone only).")
    p.add_argument("--join-warmup-epochs",   type=int, default=0,
                   help="Linearly ramp JoinBlock contribution α from 0 to 1 over "
                        "the first N epochs (model.set_join_alpha). Lets the "
                        "standard stack settle before adding antisymmetric-wedge "
                        "contributions; mitigates early-epoch instability.")
    # --- input-representation ablations (SciPost Report #1, points 10, 11) ---
    p.add_argument("--pairing-content",      default="full",
                   choices=["full", "no_meet", "scalars_only"],
                   help="Covariant content of the two pairing tokens. 'full' "
                        "(default) = grade 1 + 2 + 3 as described in the paper; "
                        "'no_meet' drops the grade-2 meet bivector; "
                        "'scalars_only' drops all covariant content and keeps "
                        "the grade-0 pairing invariants. Parameter count is "
                        "unchanged in all three cases.")
    p.add_argument("--reference-tokens",     dest="reference_tokens",
                   action="store_true", default=False,
                   help="Append fixed gamma_0 (time) and gamma_3 (beam-axis) "
                        "reference tokens, making the H_LHC-invariants "
                        "accessible as scalar products without any change to "
                        "the architecture.")
    p.add_argument("--reference-mode",       default="vectors",
                   choices=["vectors", "bivector"],
                   help="Which reference object the tokens carry. 'vectors' = "
                        "gamma_0 + gamma_3 (supplies the ingredients of p_T, y, "
                        "phi but is NOT H_LHC-invariant); 'bivector' = gamma_03 "
                        "(H_LHC-equivariant by construction).")
    p.add_argument("--grade0-only",          dest="grade0_only",
                   action="store_true", default=False,
                   help="Zero every grade>=1 input channel, leaving Lorentz "
                        "scalars only (PELICAN-like limit of the input).")
    return p


def _git_commit() -> str:
    try:
        repo = Path(__file__).resolve().parents[4]
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tb").mkdir(exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))

    # Datasets: load per-class subsets directly into RAM. h5py per-item reads
    # against gzip-chunked storage are unusably slow under random batching, so
    # we materialise the (capped) tensors once at startup.
    from paper1_demo.gatr_lite.data.dataset import (
        InMemoryPartonDataset, MultiH5Dataset,
    )
    rng = np.random.default_rng(args.seed)

    def _pick(h5_path: str, max_n: int) -> InMemoryPartonDataset:
        with h5py.File(h5_path, "r") as f:
            n_total = int(f["p4"].shape[0])
        if max_n and 0 < max_n < n_total:
            idx = np.sort(rng.choice(n_total, max_n, replace=False))
        else:
            idx = None
        return InMemoryPartonDataset(h5_path, indices=idx)

    ds_signal = _pick(args.signal,     args.max_train_per_class)
    ds_bkg    = _pick(args.background, args.max_train_per_class)
    ds_full   = MultiH5Dataset([ds_signal, ds_bkg])

    train_sub, val_sub = make_train_val_split(ds_full, args.val_frac, args.seed)

    # Low-statistics scan: thin the TRAINING subset only. val_sub is untouched,
    # so every fraction of a given seed is scored on identical events and the
    # paired comparison against the full-statistics run remains meaningful.
    # Subsetting keeps the leading n_keep entries of the split permutation,
    # which makes the fractions nested (1/10 ⊂ 1/3 ⊂ full).
    if not 0.0 < args.train_frac <= 1.0:
        raise SystemExit(f"--train-frac must be in (0, 1], got {args.train_frac}")
    if args.train_frac < 1.0:
        n_before = len(train_sub)
        n_keep = max(1, int(round(n_before * args.train_frac)))
        train_sub = Subset(train_sub, list(range(n_keep)))
        logger.info("low-statistics scan: train %d -> %d events (frac %.4g); "
                    "val untouched at %d", n_before, n_keep, args.train_frac,
                    len(val_sub))

    train_labels = _collect_labels(train_sub)
    val_labels   = _collect_labels(val_sub)
    n_train_pos = int((train_labels == 1).sum())
    n_train_neg = int((train_labels == 0).sum())
    n_val_pos   = int((val_labels == 1).sum())
    n_val_neg   = int((val_labels == 0).sum())
    logger.info("Train: pos=%d neg=%d  Val: pos=%d neg=%d",
                n_train_pos, n_train_neg, n_val_pos, n_val_neg)

    # Sampler: balanced (default) or natural-distribution (ablation).
    # Balanced: WeightedRandomSampler with steps = 2 × min(class_size); BCE
    # loss without pos_weight. One full minority-class pass per epoch.
    # No-balanced: shuffled DataLoader on the full train_sub; BCE pos_weight
    # set to N_neg/N_pos to compensate class imbalance via the loss instead.
    pos_weight: torch.Tensor | None = None
    if args.no_balanced_sampler:
        steps = len(train_sub)
        sampler = None
        pos_weight = torch.tensor(
            [n_train_neg / max(n_train_pos, 1)],
            dtype=torch.float32, device=device,
        )
        logger.info("No-balanced sampler: shuffled %d items/epoch, "
                    "pos_weight=%.3f", steps, pos_weight.item())
    else:
        if args.steps_per_epoch > 0:
            steps = args.steps_per_epoch
        elif args.max_train_per_class > 0:
            steps = len(train_sub)            # back-compat path
        else:
            minor = min(n_train_pos, n_train_neg)
            steps = 2 * minor
            logger.info("Auto steps_per_epoch=%d (2 × min class size)", steps)
        sampler = _balanced_sampler(train_labels, num_samples=steps)

    train_loader = DataLoader(
        train_sub, batch_size=args.batch_size, sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_sub, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # Model
    model_cfg = GATrLiteConfig(
        n_blocks=args.n_blocks,
        channels=args.channels,
        gp_mid_channels=args.gp_mid_channels,
        attn_channels=args.attn_channels,
        head_hidden=args.head_hidden,
        arch_v2=args.arch_v2,
        n_heads=args.n_heads,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        gp_grade3_mixing=args.gp_grade3_mixing,
        use_join_block=args.use_join_block,
        use_pairing_cache=args.use_pairing_cache,
        pairing_content=args.pairing_content,
        reference_tokens=args.reference_tokens,
        reference_mode=args.reference_mode,
        grade0_only=args.grade0_only,
    )
    logger.info("Model config: %s", asdict(model_cfg))
    model = GATrLite(model_cfg).to(device)
    n_params = model.num_parameters()
    logger.info("GATrLite parameters: %d", n_params)
    if args.compile:
        try:
            model = torch.compile(model)
            logger.info("torch.compile enabled")
        except Exception as exc:
            logger.warning("torch.compile failed (%s); continuing eager", exc)

    # Optimizer / scheduler / loss
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    # Linear warmup → cosine anneal. Big-batch training (bs ≥ 2048) benefits
    # from a few epochs of warmup so the cosine doesn't blow up near step 0.
    warmup = max(int(args.warmup_epochs), 0)
    if warmup > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=1.0 / max(warmup, 1), end_factor=1.0,
            total_iters=warmup,
        )
        cos_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(args.epochs - warmup, 1),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optim, schedulers=[warmup_sched, cos_sched], milestones=[warmup],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(args.epochs, 1),
        )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Save config
    config_payload = {
        "args": vars(args),
        "commit": _git_commit(),
        "n_params": n_params,
        "n_train_pos": n_train_pos,
        "n_train_neg": n_train_neg,
        "n_val_pos":   n_val_pos,
        "n_val_neg":   n_val_neg,
        "model_cfg":   asdict(model_cfg),
        "device":      str(device),
        "torch_version": torch.__version__,
    }
    with (out_dir / "config.json").open("w") as f:
        json.dump(config_payload, f, indent=2)

    # Logging sinks
    if _TB_AVAILABLE:
        tb = SummaryWriter(log_dir=str(out_dir / "tb"))
    else:
        tb = None
        logger.warning("tensorboard not available; skipping TB logging")
    csv_path = out_dir / "metrics.csv"
    csv_f = csv_path.open("w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["epoch", "train_loss", "val_loss", "val_auc", "val_brier",
                    "lr", "epoch_seconds"])
    csv_f.flush()

    best_auc = -math.inf
    best_composite = -math.inf
    val_loss_min = math.inf
    epochs_no_improve = 0
    last_state: Dict | None = None

    def _composite(auc: float, brier: float, loss: float) -> float:
        nonlocal val_loss_min
        if args.early_stop_metric == "auc":
            return auc
        if args.early_stop_metric == "auc-brier":
            return auc - 0.5 * brier
        # auc-loss
        val_loss_min = min(val_loss_min, loss)
        overfit = max(0.0, loss - val_loss_min)
        return auc - 0.1 * overfit

    print(f"Train start: params={n_params}  train={len(train_sub)}  val={len(val_sub)}  device={device}", flush=True)

    for epoch in range(1, args.epochs + 1):
        # JoinBlock warmup: ramp α from 0 to 1 over the first N epochs.
        if args.join_warmup_epochs > 0 and hasattr(model, "set_join_alpha"):
            alpha = min(1.0, (epoch - 1) / max(args.join_warmup_epochs, 1))
            model.set_join_alpha(alpha)
        t0 = time.time()
        train_loss = _epoch_train(model, train_loader, optim, loss_fn, device,
                                  grad_clip=args.grad_clip)
        val_loss, val_auc, val_brier = _epoch_eval(model, val_loader, loss_fn, device)
        scheduler.step()
        dt = time.time() - t0
        lr_now = optim.param_groups[0]["lr"]

        print(f"epoch {epoch:03d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_auc={val_auc:.4f}  val_brier={val_brier:.4f}  "
              f"lr={lr_now:.2e}  dt={dt:.1f}s", flush=True)

        if tb is not None:
            tb.add_scalar("loss/train", train_loss, epoch)
            tb.add_scalar("loss/val",   val_loss,   epoch)
            tb.add_scalar("metric/val_auc",   val_auc,   epoch)
            tb.add_scalar("metric/val_brier", val_brier, epoch)
            tb.add_scalar("opt/lr", lr_now, epoch)
        csv_w.writerow([epoch, train_loss, val_loss, val_auc, val_brier, lr_now, dt])
        csv_f.flush()

        last_state = {
            "epoch": epoch,
            "model_state": (model._orig_mod.state_dict()
                             if hasattr(model, "_orig_mod") else model.state_dict()),
            "optim_state": optim.state_dict(),
            "val_auc":     val_auc,
            "val_loss":    val_loss,
            "args":        vars(args),
            # Persist actual model_cfg in the checkpoint so eval can
            # reconstruct the exact architecture without CLI overrides.
            # Without this, ablations like --no-join-block / --no-pairing-cache
            # silently produce ckpts that eval can't load against defaults.
            "model_cfg":   asdict(model_cfg),
        }
        torch.save(last_state, out_dir / "last.pt")

        # best.pt is always selected by pure val_auc (apples-to-apples with
        # the paper). Early stop is driven by the (possibly composite) metric.
        if not math.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(last_state, out_dir / "best.pt")

        composite = _composite(val_auc, val_brier, val_loss)
        improved = (
            not math.isnan(composite)
            and composite > best_composite + args.early_stop_min_delta
        )
        if improved:
            best_composite = composite
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            ready_to_stop = epoch >= args.early_stop_warmup
            if ready_to_stop and epochs_no_improve >= args.early_stop_patience:
                print(f"Early stop: no improvement on '{args.early_stop_metric}' "
                      f"(min_delta={args.early_stop_min_delta}) for "
                      f"{epochs_no_improve} epochs after warmup={args.early_stop_warmup}",
                      flush=True)
                break

    csv_f.close()
    if tb is not None:
        tb.close()
    print(f"Done. best_val_auc={best_auc:.4f}  out={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
