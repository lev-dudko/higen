"""Hyper-parameter sweep configurations.

Each config overrides a subset of training and model hyper-parameters.
Fields not listed fall back to the values defined in
``GATrLiteConfig`` and to the ``DEFAULT_TRAIN`` block below.

Conventions
-----------
* ``model``: keyword overrides for ``GATrLiteConfig``.
* ``train``: keyword overrides for ``train.main`` CLI args.
* ``name``: short identifier used for the run subdirectory.
"""
from __future__ import annotations

from typing import Any, Dict, List


DEFAULT_TRAIN: Dict[str, Any] = {
    "batch_size": 512,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "val_frac": 0.1,
    "seed": 42,
    "device": "cuda",
    "early_stop_patience": 15,
    "max_train_per_class": 100_000,
    "num_workers": 2,
    "steps_per_epoch": 0,
    "compile": False,
    "log_level": "INFO",
}


# Phase 1: smoke probes (fast, ~10 epochs * 30k+30k events).
# Phase 2: top-3 by val AUC continue with their full configs at 100k+100k * 50 epochs.
# Phase 3: if best < 0.96 -> rerun the best config at 200k+200k.
SWEEP_CONFIGS: List[Dict[str, Any]] = [
    # Reproduces full_001 (best AUC = 0.9519). Reference for the sweep.
    {
        "name": "baseline",
        "model": {"n_blocks": 3, "channels": 32},
        "train": {"lr": 1e-3, "batch_size": 512, "weight_decay": 1e-4},
    },
    # Wider channels: more capacity, same depth.
    {
        "name": "larger",
        "model": {"n_blocks": 3, "channels": 64},
        "train": {"lr": 1e-3, "batch_size": 512, "weight_decay": 1e-4},
    },
    # Deeper: more equivariant blocks.
    {
        "name": "deeper",
        "model": {"n_blocks": 5, "channels": 32},
        "train": {"lr": 1e-3, "batch_size": 512, "weight_decay": 1e-4},
    },
    # Wide + deeper.
    {
        "name": "wide_deep",
        "model": {"n_blocks": 4, "channels": 64},
        "train": {"lr": 7e-4, "batch_size": 512, "weight_decay": 1e-4},
    },
    # Lower learning rate (smoother optimization).
    {
        "name": "lr_low",
        "model": {"n_blocks": 3, "channels": 32},
        "train": {"lr": 3e-4, "batch_size": 512, "weight_decay": 1e-4},
    },
    # Higher learning rate (fast warm-up via cosine).
    {
        "name": "lr_high",
        "model": {"n_blocks": 3, "channels": 32},
        "train": {"lr": 3e-3, "batch_size": 512, "weight_decay": 1e-4},
    },
    # Larger mini-batch (fits on 5060 Ti at 32 channels).
    {
        "name": "batch1024",
        "model": {"n_blocks": 3, "channels": 32},
        "train": {"lr": 1.5e-3, "batch_size": 1024, "weight_decay": 1e-4},
    },
    # Stronger weight decay (regularization).
    {
        "name": "wd_high",
        "model": {"n_blocks": 3, "channels": 32},
        "train": {"lr": 1e-3, "batch_size": 512, "weight_decay": 1e-3},
    },
    # Lighter weight decay (under-regularized vs full_001).
    {
        "name": "wd_low",
        "model": {"n_blocks": 3, "channels": 32},
        "train": {"lr": 1e-3, "batch_size": 512, "weight_decay": 1e-5},
    },
    # More data (used directly in Phase 2, not Phase 1).
    {
        "name": "more_data",
        "model": {"n_blocks": 3, "channels": 32},
        "train": {
            "lr": 1e-3,
            "batch_size": 512,
            "weight_decay": 1e-4,
            "max_train_per_class": 200_000,
        },
    },
]


def merged_train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply config overrides on top of DEFAULT_TRAIN."""
    out = dict(DEFAULT_TRAIN)
    out.update(cfg.get("train", {}))
    return out
