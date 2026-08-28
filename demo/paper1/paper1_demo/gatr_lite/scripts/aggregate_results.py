"""Aggregate multi-seed GA runs + ``nopair`` ablation + REF baseline into the
paper-ready ROC band, discriminator histograms, and AUC summary.

Inputs (all relative to ``--runs-dir``)::

    full_v1_alldata/eval/predictions.npz                # GA seed 42 (baseline)
    full_v1_alldata_seed43/eval/predictions.npz         # GA seed 43
    full_v1_alldata_seed44/eval/predictions.npz         # GA seed 44
    full_v1_alldata_seed45/eval/predictions.npz         # GA seed 45
    full_v1_alldata_seed46/eval/predictions.npz         # GA seed 46
    full_v1_alldata_nopair/eval/predictions.npz         # GA, --no-pairing-cache
    ref/predictions.npz                                 # REF exam scores

Outputs (in ``--out-dir``)::

    auc_summary.csv          rows: GA-mean ± std, GA-nopair, REF
    roc_band.{pdf,png}       ROC: GA band, REF, nopair
    discriminator.{pdf,png}  3-curve discriminator on GA baseline seed
    summary.json             machine-readable digest

Usage::

    python -u -m paper1_demo.gatr_lite.scripts.aggregate_results \\
        --runs-dir ./runs \\
        --out-dir  ./runs/aggregate
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("paper1.aggregate")


# ------------------------------------------------------------------------------
# ROC primitive (re-implementing eval_plots._roc_curve so this script is
# self-contained and re-usable from the aggregator runs).
# ------------------------------------------------------------------------------

def _roc_curve(
    scores: np.ndarray, labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
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
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # trapz removed in NumPy 2.0
    return fpr, tpr, float(trap(tpr, fpr))


def _hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    if n_pos <= 0 or n_neg <= 0 or not math.isfinite(auc):
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    num = auc * (1.0 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)
    return math.sqrt(max(num, 0.0) / (n_pos * n_neg))


# ------------------------------------------------------------------------------
# Run loaders
# ------------------------------------------------------------------------------

@dataclass
class RunScores:
    name: str
    label: str
    fpr: np.ndarray
    tpr: np.ndarray
    auc: float
    auc_se_hm: float
    n_pos: int
    n_neg: int


def load_ga(run_dir: Path, label: str) -> RunScores:
    pred = np.load(run_dir / "eval" / "predictions.npz")
    sig = pred["signal_scores"]
    bkg = pred["background_scores"]
    scores = np.concatenate([sig, bkg])
    labels = np.concatenate([np.ones(len(sig), dtype=np.int64),
                             np.zeros(len(bkg), dtype=np.int64)])
    fpr, tpr, auc = _roc_curve(scores, labels)
    se = _hanley_mcneil_se(auc, len(sig), len(bkg))
    logger.info("loaded GA  %-30s  AUC=%.6f ± %.6f (HM)  n_pos=%d n_neg=%d",
                run_dir.name, auc, se, len(sig), len(bkg))
    return RunScores(
        name=run_dir.name, label=label,
        fpr=fpr, tpr=tpr, auc=auc, auc_se_hm=se,
        n_pos=int(len(sig)), n_neg=int(len(bkg)),
    )


def load_ref(ref_dir: Path) -> RunScores:
    pred = np.load(ref_dir / "predictions.npz")
    if "signal_scores" in pred.files:
        sig = pred["signal_scores"]
        bkg = pred["background_scores"]
        scores = np.concatenate([sig, bkg])
        labels = np.concatenate([np.ones(len(sig), dtype=np.int64),
                                 np.zeros(len(bkg), dtype=np.int64)])
        weights = None
    else:
        scores = pred["discriminator"].astype(np.float64)
        labels = pred["target"].astype(np.int64)
        weights = pred.get("weight", None)
    fpr, tpr, auc = _roc_curve(scores, labels, weights)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    se = _hanley_mcneil_se(auc, n_pos, n_neg)
    logger.info("loaded REF %-30s  AUC=%.6f ± %.6f (HM)  n_pos=%d n_neg=%d",
                ref_dir.name, auc, se, n_pos, n_neg)
    return RunScores(
        name=ref_dir.name, label="REF [Boos:2023kpp]",
        fpr=fpr, tpr=tpr, auc=auc, auc_se_hm=se,
        n_pos=n_pos, n_neg=n_neg,
    )


# ------------------------------------------------------------------------------
# Aggregation: interpolate each ROC onto a fixed FPR grid, compute mean/std.
# ------------------------------------------------------------------------------

def _interp_tpr(runs: List[RunScores], fpr_grid: np.ndarray) -> np.ndarray:
    rows = []
    for r in runs:
        rows.append(np.interp(fpr_grid, r.fpr, r.tpr))
    return np.array(rows)  # (n_runs, n_grid)


def _band(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return arr.mean(axis=0), arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])


# ------------------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------------------

def plot_roc(
    ga_runs: List[RunScores],
    nopair: Optional[RunScores],
    ref: Optional[RunScores],
    out_dir: Path,
) -> Dict[str, float]:
    fpr_grid = np.linspace(0.0, 1.0, 4001)
    tpr_ga = _interp_tpr(ga_runs, fpr_grid)
    tpr_mean, tpr_std = _band(tpr_ga)
    aucs_ga = np.array([r.auc for r in ga_runs])
    auc_mean = float(aucs_ga.mean())
    auc_std = float(aucs_ga.std(ddof=1)) if len(aucs_ga) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.fill_between(fpr_grid, tpr_mean - tpr_std, tpr_mean + tpr_std,
                    color="tab:red", alpha=0.20, lw=0)
    ax.plot(fpr_grid, tpr_mean, lw=2.0, color="tab:red",
            label=fr"GA (GATr-lite, {len(ga_runs)} seeds): "
                  fr"AUC = {auc_mean:.4f} $\pm$ {auc_std:.4f}")

    if nopair is not None:
        ax.plot(nopair.fpr, nopair.tpr, lw=1.5, color="tab:orange",
                ls="--",
                label=fr"GA $-$ pairing tokens (1 seed): AUC = {nopair.auc:.4f}")

    if ref is not None:
        ax.plot(ref.fpr, ref.tpr, lw=2.0, color="tab:blue",
                label=fr"REF [Boos:2023kpp] (1 seed): AUC = {ref.auc:.4f}")

    ax.plot([0, 1], [0, 1], lw=1, ls=":", color="gray", label="random")
    ax.set_xlabel(r"False positive rate ($t\bar t$)")
    ax.set_ylabel(r"True positive rate ($tW$)")
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.005)
    ax.set_title(r"ROC: $tW$ vs $t\bar t$ — parton-level, $p_T(j_4) > 10$ GeV")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_band.pdf")
    fig.savefig(out_dir / "roc_band.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote roc_band.{pdf,png}  GA AUC mean=%.4f std=%.4f", auc_mean, auc_std)
    return {"auc_mean": auc_mean, "auc_std": auc_std,
            "n_seeds": len(ga_runs)}


def plot_discriminator(baseline_run: Path, out_dir: Path) -> None:
    """Discriminator histogram from a single seed (Fig. 6a-equivalent).

    Aggregation across seeds doesn't combine naturally with cross-section
    weighted histograms (each seed sees the same events), so we render this
    from the baseline seed and let the ROC band carry the multi-seed
    statement.
    """
    pred = np.load(baseline_run / "eval" / "predictions.npz")
    samples = [("signal", r"$tW$ (single-resonant)", "tab:red"),
               ("background", r"$t\bar t$ (double-resonant)", "tab:blue"),
               ("extra", r"$tWb$ full (incl. interference)", "tab:green")]
    bin_edges = np.linspace(0.0, 1.0, 51)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for key, lab, color in samples:
        if f"{key}_scores" not in pred.files:
            continue
        ax.hist(pred[f"{key}_scores"], bins=bin_edges, weights=pred[f"{key}_weights"],
                histtype="step", lw=2, color=color, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("GATr-lite discriminator (baseline seed)")
    ax.set_ylabel(r"$d\sigma/dD$  (pb / bin)")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(r"Discriminator distribution (cross-section weighted, $1\,\mathrm{pb}^{-1}$)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "discriminator.pdf")
    fig.savefig(out_dir / "discriminator.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

def write_summary(
    ga_runs: List[RunScores],
    nopair: Optional[RunScores],
    ref: Optional[RunScores],
    out_dir: Path,
    band_stats: Dict[str, float],
) -> None:
    aucs_ga = np.array([r.auc for r in ga_runs])
    rows = ["network,seeds,parameters,auc,auc_err,err_source,n_pos,n_neg"]
    rows.append(
        f"GA-NN baseline,{len(ga_runs)},1.51e5,"
        f"{aucs_ga.mean():.6f},{aucs_ga.std(ddof=1) if len(aucs_ga) > 1 else 0.0:.6f},"
        f"training-std,{ga_runs[0].n_pos},{ga_runs[0].n_neg}"
    )
    for r in ga_runs:
        rows.append(
            f"GA-NN {r.name},1,1.51e5,{r.auc:.6f},{r.auc_se_hm:.6f},"
            f"hanley-mcneil,{r.n_pos},{r.n_neg}"
        )
    if nopair is not None:
        rows.append(
            f"GA-NN -pairing,1,1.40e5,{nopair.auc:.6f},{nopair.auc_se_hm:.6f},"
            f"hanley-mcneil,{nopair.n_pos},{nopair.n_neg}"
        )
    if ref is not None:
        rows.append(
            f"REF [Boos:2023kpp],1,5.4e5,{ref.auc:.6f},{ref.auc_se_hm:.6f},"
            f"hanley-mcneil,{ref.n_pos},{ref.n_neg}"
        )
    (out_dir / "auc_summary.csv").write_text("\n".join(rows) + "\n")

    digest = {
        "ga_baseline": {
            "n_seeds": len(ga_runs),
            "auc_mean": float(aucs_ga.mean()),
            "auc_std":  float(aucs_ga.std(ddof=1)) if len(aucs_ga) > 1 else 0.0,
            "seeds": [r.name for r in ga_runs],
            "aucs":  [r.auc for r in ga_runs],
        },
        "ga_nopair": (None if nopair is None
                      else {"auc": nopair.auc, "auc_se_hm": nopair.auc_se_hm}),
        "ref":       (None if ref is None
                      else {"auc": ref.auc, "auc_se_hm": ref.auc_se_hm}),
        "band_stats": band_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(digest, indent=2))
    logger.info("Wrote auc_summary.csv and summary.json")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", required=True, help="Runs root directory")
    p.add_argument("--out-dir",  required=True, help="Where to write outputs")
    p.add_argument("--ga-baseline",
                   default="full_v1_alldata",
                   help="Run dir for the baseline (default seed 42)")
    p.add_argument("--ga-seeds", nargs="+", type=int,
                   default=[42, 43, 44, 45, 46],
                   help="Seed numbers to include in the multi-seed band")
    p.add_argument("--seed-name-fmt",
                   default="full_v1_alldata_seed{seed}",
                   help="Run dir template for non-baseline seeds")
    p.add_argument("--nopair-run", default="full_v1_alldata_nopair",
                   help="Run dir for the no-pairing-cache ablation (None disables)")
    p.add_argument("--ref-dir",   default="ref")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ga_runs: List[RunScores] = []
    for seed in args.ga_seeds:
        run_dir = (runs_dir / args.ga_baseline
                   if seed == 42 and args.ga_baseline == "full_v1_alldata"
                   else runs_dir / args.seed_name_fmt.format(seed=seed))
        if not (run_dir / "eval" / "predictions.npz").exists():
            logger.warning("Skipping seed %d: %s missing", seed, run_dir)
            continue
        ga_runs.append(load_ga(run_dir, label=f"GA seed {seed}"))

    if not ga_runs:
        logger.error("No GA seeds found under %s", runs_dir)
        return 1

    nopair: Optional[RunScores] = None
    if args.nopair_run:
        nopair_dir = runs_dir / args.nopair_run
        if (nopair_dir / "eval" / "predictions.npz").exists():
            nopair = load_ga(nopair_dir, label="GA (no pairing)")
        else:
            logger.warning("Skipping nopair: %s missing", nopair_dir)

    ref: Optional[RunScores] = None
    ref_dir = runs_dir / args.ref_dir
    if (ref_dir / "predictions.npz").exists():
        ref = load_ref(ref_dir)
    else:
        logger.warning("Skipping REF: %s missing", ref_dir)

    band_stats = plot_roc(ga_runs, nopair, ref, out_dir)
    plot_discriminator(runs_dir / args.ga_baseline, out_dir)
    write_summary(ga_runs, nopair, ref, out_dir, band_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
