"""Tests for the REF baseline (Boos:2023kpp) loader.

A small Keras h5 with the REF weights is shipped under
``demo/paper1/weights/ref_baseline_boos2023.h5``; tests that need it pick
it up automatically.  The exam ROOT file is large and is not redistributed:
set ``HIGEN_REF_EXAM_ROOT`` to its path to enable the end-to-end AUC test.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from paper1_demo.gatr_lite.training.ref_baseline import (
    RefMLP,
    auc_score,
    load_keras_h5,
)

_REF_H5 = Path(os.environ.get(
    "HIGEN_REF_H5",
    str(Path(__file__).resolve().parents[3] / "weights" / "ref_baseline_boos2023.h5"),
))
_REF_EXAM = Path(os.environ.get("HIGEN_REF_EXAM_ROOT", "/__not_present__"))


def test_refmlp_param_count() -> None:
    """4-layer MLP 75->500->500->500->1 has exactly 539501 trainable params."""
    m = RefMLP()
    n = sum(p.numel() for p in m.parameters())
    expected = (75 * 500 + 500) + (500 * 500 + 500) + (500 * 500 + 500) + (500 * 1 + 1)
    assert n == expected == 539_501, f"got {n}, expected {expected}"


def test_refmlp_forward_shapes() -> None:
    m = RefMLP().eval()
    x = torch.zeros(7, 75)
    y = m(x)
    assert y.shape == (7,)
    p = m.predict_proba(x)
    assert p.shape == (7,)
    assert (p >= 0).all() and (p <= 1).all()


def test_auc_score_perfect() -> None:
    s = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([1, 1, 0, 0])
    assert auc_score(s, y) == pytest.approx(1.0, abs=1e-12)


def test_auc_score_random() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    s = rng.random(n)
    y = rng.integers(0, 2, n)
    auc = auc_score(s, y)
    assert 0.45 < auc < 0.55


@pytest.mark.skipif(not _REF_H5.exists(), reason="REF Keras h5 not on this host")
def test_load_keras_h5() -> None:
    """Smoke-load the actual checkpoint and verify no NaN weights."""
    m = load_keras_h5(_REF_H5)
    for name, p in m.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite weights in {name}"
    # forward should produce finite logits
    x = torch.zeros(4, 75)
    out = m(x)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not (_REF_H5.exists() and _REF_EXAM.exists()),
                    reason="REF exam ROOT or h5 not on this host")
def test_ref_exam_auc_above_random() -> None:
    """End-to-end AUC sanity: REF on its own exam set must beat random by a wide margin."""
    pytest.importorskip("uproot")
    from paper1_demo.gatr_lite.training.ref_baseline import predict_root
    m = load_keras_h5(_REF_H5)
    # Score the whole exam set: it is sorted (DR1 first, tT after), so a
    # leading slice would only contain the positive class and yield NaN AUC.
    out = predict_root(m, _REF_EXAM, max_events=None, batch_size=4096, device="cpu")
    auc = auc_score(out["discriminator"], out["target"], out.get("weight"))
    assert auc > 0.85, f"REF AUC too low: {auc}"
