"""Make ROC and discriminator plots from a `predictions.npz` produced by
`paper1_demo.gatr_lite.training.eval`, with optional REF (Boos:2023kpp)
baseline overlay.

CLI
---
    python -m paper1_demo.gatr_lite.scripts.eval_plots \\
        --predictions          /path/to/run/eval/predictions.npz \\
        --ref-predictions      /path/to/ref/predictions.npz       (optional, exam ROOT)
        --ref-full-predictions /path/to/ref/predictions_full.npz  (optional, full SM)
        --ref-lhe              /path/to/ref/predictions_lhe.npz   (optional, raw LHE 3-sample)
        --out-dir              /path/to/run/eval

Outputs
-------
    roc.{pdf,png}              GATr-lite (and REF if available)
    discriminator.{pdf,png}    GATr-lite, signal/background/extra histograms
    discriminator_ref.{pdf,png}  REF discriminator histograms (1, 2, or 3 curves)
    summary.csv                per-sample stats

Three overlay sources are supported for REF:
  * ``--ref-predictions`` -- output of ``ref_inference.py`` (REF exam ROOT
    with var1..var75 + target/sampleNumber). Carries DR1+tT only.
  * ``--ref-full-predictions`` -- output of ``ref_inference.py`` run on the
    full SM tT_tWb exam ntuple.  All events are plotted as a 3rd curve
    (tT_tWb full SM) in discriminator_ref.  Also combined with
    ``--ref-predictions`` to render all three curves from a single exam set.
  * ``--ref-lhe`` -- output of ``ref_inference_lhe.py`` (raw LHE root
    files with the REF config's means/sigmas applied). Carries DR1, tT,
    and optionally tT_tWb (full).

Priority: ref_lhe > ref_predictions+ref_full (exam split) > ref_predictions alone.
For the ROC overlay ``--ref-predictions`` is always used (exam AUC is the
apples-to-apples baseline).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _set_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        try:
            plt.style.use("seaborn-whitegrid")
        except Exception:
            plt.style.use("default")


def _roc_curve(scores: np.ndarray, labels: np.ndarray,
               weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Binary ROC. Returns (fpr, tpr, auc) computed from a sorted threshold sweep."""
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels[order].astype(np.float64)
    w = (weights[order] if weights is not None else np.ones_like(y)).astype(np.float64)
    pos_w = (w * (y == 1)).sum()
    neg_w = (w * (y == 0)).sum()
    if pos_w == 0 or neg_w == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    tps = np.cumsum(w * (y == 1))
    fps = np.cumsum(w * (y == 0))
    tpr = tps / pos_w
    fpr = fps / neg_w
    distinct = np.r_[np.diff(s) != 0, True]
    fpr = np.r_[0.0, fpr[distinct]]
    tpr = np.r_[0.0, tpr[distinct]]
    trap = getattr(np, "trapezoid", np.trapz)
    auc = float(trap(tpr, fpr))
    return fpr, tpr, auc


def make_roc(payload: Dict[str, np.ndarray],
             out_dir: Path,
             ref_exam: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
    sig = payload["signal_scores"]
    bkg = payload["background_scores"]
    scores = np.concatenate([sig, bkg])
    labels = np.concatenate([np.ones(len(sig), dtype=np.int64),
                             np.zeros(len(bkg), dtype=np.int64)])
    fpr, tpr, auc = _roc_curve(scores, labels)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(fpr, tpr, lw=2.0, color="tab:red",
            label=f"GA-NN (GATr-lite)  AUC = {auc:.4f}")

    aucs = {"ga": auc}
    if ref_exam is not None:
        r_scores = ref_exam["discriminator"].astype(np.float64)
        r_labels = ref_exam["target"].astype(np.int64)
        r_w = ref_exam.get("weight", None)
        r_fpr, r_tpr, r_auc = _roc_curve(r_scores, r_labels, r_w)
        ax.plot(r_fpr, r_tpr, lw=2.0, ls="-", color="tab:blue",
                label=f"REF [Boos:2023kpp]  AUC = {r_auc:.4f}")
        aucs["ref"] = r_auc

    ax.plot([0, 1], [0, 1], lw=1, ls="--", color="gray", label="random")
    ax.set_xlabel(r"False positive rate ($t\bar t$)")
    ax.set_ylabel(r"True positive rate ($tW$)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.005)
    ax.set_title(r"ROC: $tW$ vs $t\bar t$ — parton-level, $p_T(j_4) > 10$ GeV")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "roc.pdf")
    fig.savefig(out_dir / "roc.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote roc.{pdf,png}; %s",
                ", ".join(f"{k}={v:.4f}" for k, v in aucs.items()))
    return aucs


# Physics labels used on every discriminator plot. Curves in the article
# are identified by the underlying physical process.
_CURVE_LABELS = {
    "signal":     r"$tW$ (single-resonant)",
    "background": r"$t\bar t$ (double-resonant)",
    "extra":      r"$tWb$ full (incl. interference)",
}
_CURVE_COLORS = {
    "signal":     "tab:red",
    "background": "tab:blue",
    "extra":      "tab:green",
}


def make_discriminator(payload: Dict[str, np.ndarray], out_dir: Path,
                       n_bins: int = 50) -> None:
    samples = [("signal", _CURVE_LABELS["signal"], _CURVE_COLORS["signal"]),
               ("background", _CURVE_LABELS["background"], _CURVE_COLORS["background"])]
    if "extra_scores" in payload:
        samples.append(("extra", _CURVE_LABELS["extra"], _CURVE_COLORS["extra"]))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for key, label, color in samples:
        scores = payload[f"{key}_scores"]
        weights = payload[f"{key}_weights"]
        ax.hist(scores, bins=bin_edges, weights=weights, histtype="step",
                lw=2, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("GATr-lite discriminator")
    ax.set_ylabel(r"$d\sigma/dD$  (pb / bin)")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(r"Discriminator distribution (cross-section weighted, $1\,\mathrm{pb}^{-1}$)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "discriminator.pdf")
    fig.savefig(out_dir / "discriminator.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote discriminator.{pdf,png}")


def make_discriminator_ref(
    out_dir: Path,
    *,
    ref_lhe: Optional[Dict[str, np.ndarray]] = None,
    ref_exam: Optional[Dict[str, np.ndarray]] = None,
    ref_full: Optional[Dict[str, np.ndarray]] = None,
    n_bins: int = 50,
    payload: Optional[Dict[str, np.ndarray]] = None,
) -> bool:
    """REF discriminator histogram.

    Priority for the plot source:
      1. ``ref_lhe``  -- 3 curves from raw LHE with the SAME standardisation
         the REF Keras model was trained on. Cross-section weighted (pb/bin)
         when ``payload`` carries per-sample xsec, otherwise area-normalised.
      2. ``ref_exam`` + ``ref_full``  -- 3 curves from exam ntuples,
         area-normalised. Note: the tT_tWb exam ntuple is standardised under a
         different (DR1-vs-full) training set than the REF model, which can
         distort the full-SM shape. Prefer path 1 whenever raw LHE is available.
      3. ``ref_exam`` alone  -- 2 curves, area-normalised.
    Returns True if a plot was produced.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    title_warn = ""

    # Cross-sections from the GATr-lite payload, if available — used to put
    # the LHE-derived REF histogram on the same footing as the GATr plot.
    xsec_pb: Dict[str, float] = {}
    if payload is not None:
        for key in ("signal", "background", "extra"):
            arr = payload.get(f"{key}_xsec_pb")
            if arr is not None and len(arr) and float(arr[0]) > 0.0:
                xsec_pb[key] = float(arr[0])

    use_xsec = bool(xsec_pb)

    if ref_lhe is not None:
        samples = [("signal", _CURVE_LABELS["signal"], _CURVE_COLORS["signal"]),
                   ("background", _CURVE_LABELS["background"], _CURVE_COLORS["background"])]
        if "extra_scores" in ref_lhe:
            samples.append(("extra", _CURVE_LABELS["extra"], _CURVE_COLORS["extra"]))
        else:
            title_warn = "  [no full sample]"
        for key, label, color in samples:
            scores = ref_lhe[f"{key}_scores"]
            n = len(scores)
            if use_xsec and key in xsec_pb and n > 0:
                weights = np.full(n, xsec_pb[key] / n, dtype=np.float64)
            else:
                weights = ref_lhe[f"{key}_weights"]
            ax.hist(scores, bins=bin_edges, weights=weights, histtype="step",
                    lw=2, color=color, label=label)
        source = ("raw LHE, REF training standardisation, "
                  + ("xsec-weighted" if use_xsec else "area-normalised"))
        ylabel = (r"$d\sigma/dD$  (pb / bin)" if use_xsec
                  else "normalised entries / bin")
    elif ref_exam is not None:
        sig_mask = ref_exam["target"] == 1
        bkg_mask = ref_exam["target"] == 0
        scores_sig = ref_exam["discriminator"][sig_mask]
        scores_bkg = ref_exam["discriminator"][bkg_mask]
        w_sig = (ref_exam["weight"][sig_mask] if "weight" in ref_exam
                 else np.ones(sig_mask.sum()))
        w_bkg = (ref_exam["weight"][bkg_mask] if "weight" in ref_exam
                 else np.ones(bkg_mask.sum()))
        if len(scores_sig):
            w_sig = w_sig / w_sig.sum()
        if len(scores_bkg):
            w_bkg = w_bkg / w_bkg.sum()
        ax.hist(scores_sig, bins=bin_edges, weights=w_sig, histtype="step",
                lw=2, color=_CURVE_COLORS["signal"], label=_CURVE_LABELS["signal"])
        ax.hist(scores_bkg, bins=bin_edges, weights=w_bkg, histtype="step",
                lw=2, color=_CURVE_COLORS["background"], label=_CURVE_LABELS["background"])
        if ref_full is not None:
            if "target" in ref_full:
                sm_mask = ref_full["target"] == 0
                scores_full = ref_full["discriminator"][sm_mask]
                w_full = (ref_full["weight"][sm_mask] if "weight" in ref_full
                          else np.ones(sm_mask.sum(), dtype=np.float64))
            else:
                scores_full = ref_full["discriminator"]
                w_full = (ref_full["weight"] if "weight" in ref_full
                          else np.ones(len(scores_full), dtype=np.float64))
            if len(scores_full):
                w_full = w_full / w_full.sum()
            ax.hist(scores_full, bins=bin_edges, weights=w_full, histtype="step",
                    lw=2, color=_CURVE_COLORS["extra"], label=_CURVE_LABELS["extra"])
            source = "REF exam ntuples (mismatched full-SM standardisation)"
        else:
            title_warn = "  [REF exam set: no full sample]"
            source = "REF exam ntuple"
        ylabel = "normalised entries / bin"
    elif ref_full is not None:
        if "target" in ref_full:
            sm_mask = ref_full["target"] == 0
            scores_full = ref_full["discriminator"][sm_mask]
            w_full = (ref_full["weight"][sm_mask] if "weight" in ref_full
                      else np.ones(sm_mask.sum(), dtype=np.float64))
        else:
            scores_full = ref_full["discriminator"]
            w_full = (ref_full["weight"] if "weight" in ref_full
                      else np.ones(len(scores_full), dtype=np.float64))
        if len(scores_full):
            w_full = w_full / w_full.sum()
        ax.hist(scores_full, bins=bin_edges, weights=w_full, histtype="step",
                lw=2, color=_CURVE_COLORS["extra"], label=_CURVE_LABELS["extra"])
        source = "REF on tT_tWb full SM only"
        ylabel = "normalised entries / bin"
    else:
        plt.close(fig)
        return False

    ax.set_yscale("log")
    ax.set_xlabel(r"REF discriminator [Boos:2023kpp]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, 1.0)
    if "$d\\sigma" in ylabel:
        ax.set_title(r"REF discriminator distribution "
                     r"(cross-section weighted, $1\,\mathrm{pb}^{-1}$)"
                     + title_warn)
    else:
        ax.set_title(f"REF discriminator — {source}{title_warn}")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "discriminator_ref.pdf")
    fig.savefig(out_dir / "discriminator_ref.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote discriminator_ref.{pdf,png} (%s)", source)
    return True


def write_summary(payload: Dict[str, np.ndarray], out_dir: Path,
                  aucs: Optional[Dict[str, float]] = None) -> None:
    keys = [k.replace("_scores", "") for k in payload if k.endswith("_scores")]
    rows = []
    for k in keys:
        s = payload[f"{k}_scores"]
        n = int(payload[f"{k}_n"][0])
        xs = float(payload[f"{k}_xsec_pb"][0])
        if len(s):
            rows.append({
                "sample": k, "n": n, "xsec_pb": xs,
                "min": float(s.min()), "max": float(s.max()),
                "mean": float(s.mean()), "median": float(np.median(s)),
            })
        else:
            rows.append({"sample": k, "n": 0, "xsec_pb": xs,
                         "min": float("nan"), "max": float("nan"),
                         "mean": float("nan"), "median": float("nan")})
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample", "n", "xsec_pb",
                                          "min", "max", "mean", "median"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    logger.info("Wrote %s", csv_path)
    if aucs:
        with (out_dir / "auc.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "auc"])
            w.writeheader()
            for k, v in aucs.items():
                w.writerow({"model": k, "auc": f"{v:.6f}"})
        logger.info("Wrote auc.csv (%s)",
                    ", ".join(f"{k}={v:.4f}" for k, v in aucs.items()))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot ROC and discriminator histogram, "
                                            "with optional REF overlay.")
    p.add_argument("--predictions", required=True, help="GATr-lite predictions.npz")
    p.add_argument("--ref-predictions", default=None,
                   help="REF predictions.npz from ref_inference.py (exam ROOT, DR1+tT)")
    p.add_argument("--ref-full-predictions", default=None,
                   help="REF predictions.npz from ref_inference.py run on full SM tT_tWb "
                        "exam ntuple (adds 3rd curve in discriminator_ref plot)")
    p.add_argument("--ref-lhe", default=None,
                   help="REF predictions.npz from ref_inference_lhe.py (raw LHE, 3 samples)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-bins", type=int, default=50)
    return p


def _load_npz(path: str | None) -> Optional[Dict[str, np.ndarray]]:
    if path is None:
        return None
    return dict(np.load(path, allow_pickle=False))


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_npz(args.predictions)
    ref_exam = _load_npz(args.ref_predictions)
    ref_full = _load_npz(args.ref_full_predictions)
    ref_lhe = _load_npz(args.ref_lhe)

    _set_style()
    aucs = make_roc(payload, out_dir, ref_exam=ref_exam)
    make_discriminator(payload, out_dir, n_bins=args.n_bins)
    make_discriminator_ref(out_dir, ref_lhe=ref_lhe, ref_exam=ref_exam, ref_full=ref_full,
                           n_bins=args.n_bins, payload=payload)
    write_summary(payload, out_dir, aucs=aucs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
