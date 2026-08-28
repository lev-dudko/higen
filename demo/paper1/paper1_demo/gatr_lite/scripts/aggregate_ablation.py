"""Aggregate the input-representation ablation ladder on a genuine held-out set.

Evaluation protocol
-------------------
The ablation compares configurations whose differences are of the same order as
the scatter between seeds, so the procedure has to be tighter than a single AUC
per run:

* **Held-out by construction.** ``make_train_val_split`` is deterministic given
  the seed, so the validation indices are recoverable exactly. We evaluate
  ``last.pt`` -- the final epoch -- rather than ``best.pt``. ``best.pt`` is
  selected by ``val_auc``, so scoring it on the validation set would report the
  maximum of 50 draws and carry an optimistic bias; ``last.pt`` never saw the
  validation set in any capacity. Training runs the full 50 epochs
  (``--early-stop-patience 99``), so the last epoch is a converged model.

* **Paired comparison.** The split depends only on the seed, so A0 seed *s* and
  A3 seed *s* share the same validation events. Configurations are therefore
  compared per seed, and the paired difference has a much smaller variance than
  the difference of the across-seed means.

* **Weighted DeLong.** The AUC is a two-sample U-statistic, so its sampling
  error is not the error of a per-event mean. The DeLong decomposition
  generalised to event weights is in
  ``paper1_demo.gatr_lite.training.auc_stats``.

Usage
-----
    python aggregate_ablation.py --runs-dir <dir> --data-dir <dir> \\
        --out ablation_summary.csv
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from paper1_demo.gatr_lite.training.auc_stats import weighted_auc_var
from paper1_demo.gatr_lite.data.dataset import (
    InMemoryPartonDataset, MultiH5Dataset,
)
from paper1_demo.gatr_lite.training.eval import load_model

logger = logging.getLogger("aggregate_ablation")


def val_indices(n_total: int, val_frac: float, seed: int) -> np.ndarray:
    """Recover the validation indices of a run from its seed.

    Mirrors data/dataset.py::make_train_val_split exactly.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_val = max(1, int(round(n_total * val_frac)))
    return perm[:n_val]


@torch.no_grad()
def score_split(model, dataset, indices, device, batch_size=4096):
    """Return (scores, labels) for the given subset indices.

    Dataset items are dicts with keys p4 / pdg / label (see data/dataset.py).
    """
    scores, labels = [], []
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        items = [dataset[int(i)] for i in idx]
        p4 = torch.stack([it["p4"] for it in items]).to(device)
        pdg = torch.stack([it["pdg"] for it in items]).to(device)
        y = torch.stack([it["label"] for it in items])
        scores.append(model(p4, pdg).squeeze(-1).float().cpu().numpy())
        labels.append(y.numpy().astype(np.int64).ravel())
    return np.concatenate(scores), np.concatenate(labels)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", required=True,
                   help="Directory holding run_<TAG>_s<SEED>/ subdirectories")
    p.add_argument("--data-dir", required=True, help="Directory with dr1.h5 and tT.h5")
    p.add_argument("--out", default="ablation_summary.csv")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--save-scores", action="store_true",
                   help="Write heldout_scores.npz (per-event score and label) "
                        "into each run directory, so ROC and discriminator "
                        "figures can be redrawn from the same numbers the "
                        "table quotes, without a second scoring pass.")
    p.add_argument("--only", default="",
                   help="Comma-separated configuration tags to score "
                        "(e.g. A0,A3,A5b); default is every run found.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Must mirror training/train.py exactly: signal file first, background
    # second, concatenated by MultiH5Dataset. The global index order is what
    # make_train_val_split permutes, so any deviation here would recover the
    # wrong validation events. All runs use max_train_per_class=0 (no
    # subsampling), so the plain full-file construction is the right one --
    # asserted below against the counts recorded in each config.json.
    ds = MultiH5Dataset([
        InMemoryPartonDataset(str(Path(args.data_dir) / "dr1.h5")),
        InMemoryPartonDataset(str(Path(args.data_dir) / "tT.h5")),
    ])
    logger.info("dataset: %d events", len(ds))

    rows, per_seed = [], {}
    for run in sorted(Path(args.runs_dir).glob("run_*_s*")):
        ckpt = run / "last.pt"
        cfg_path = run / "config.json"
        if not ckpt.exists() or not cfg_path.exists():
            logger.warning("skipping %s (no last.pt / config.json)", run.name)
            continue

        cfg = json.load(open(cfg_path))
        seed = int(cfg["args"]["seed"])
        val_frac = float(cfg["args"]["val_frac"])
        tag = run.name.split("_")[1]

        only = [t.strip() for t in args.only.split(",") if t.strip()]
        if only and tag not in only:
            continue

        if int(cfg["args"].get("max_train_per_class", 0)) != 0:
            logger.warning("skipping %s: max_train_per_class=%s means the dataset "
                           "was subsampled with a seed-dependent draw, so the split "
                           "cannot be recovered from the seed alone",
                           run.name, cfg["args"]["max_train_per_class"])
            continue

        idx = val_indices(len(ds), val_frac, seed)
        try:
            model = load_model(ckpt, device).eval()
        except RuntimeError as exc:
            # A checkpoint trained by an older revision cannot be loaded by the
            # current model: this is exactly how the code-version drift in the
            # published five-seed band shows up (pre-a5673a1 runs carry 18
            # scalar channels, current code has 20). Report it and move on
            # rather than silently reporting a number from a different model.
            logger.error("STALE CHECKPOINT %s: %s", run.name,
                         str(exc).splitlines()[0])
            rows.append(dict(
                config=tag, seed=seed, n_params=int(cfg["n_params"]),
                n_val=len(idx), n_val_pos=-1,
                auc_heldout=float("nan"), auc_se_delong=float("nan"),
                checkpoint="INCOMPATIBLE — trained by an earlier code revision",
            ))
            continue
        scores, labels = score_split(model, ds, idx, device, args.batch_size)

        # Hard check that we recovered the *training* run's own validation set:
        # the class counts must match what train.py recorded in config.json.
        n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
        exp_pos, exp_neg = int(cfg["n_val_pos"]), int(cfg["n_val_neg"])
        if (n_pos, n_neg) != (exp_pos, exp_neg):
            raise SystemExit(
                f"{run.name}: recovered split does not match the training run "
                f"(got pos={n_pos} neg={n_neg}, config.json records "
                f"pos={exp_pos} neg={exp_neg}). Refusing to report an AUC on a "
                f"split that is not the one held out during training."
            )

        # Persist the per-event scores next to the run. Without these the ROC
        # and discriminator figures can only be redrawn by re-running the whole
        # scoring pass, which is how the submitted version ended up displaying
        # AUC values from a superseded aggregation: the numbers were baked into
        # the image and no text search could find them.
        if args.save_scores:
            np.savez_compressed(run / "heldout_scores.npz",
                                scores=scores.astype(np.float32),
                                labels=labels.astype(np.int8),
                                config=np.array([tag]), seed=np.array([seed]))

        # Unweighted here (w=1): the ablation compares configurations on the
        # same events, so cross-section weighting would only add variance
        # without changing the comparison. weighted_auc_var returns a dict.
        w = np.ones_like(scores, dtype=np.float64)
        st = weighted_auc_var(labels, scores, w)
        auc, se = float(st["auc"]), float(st["sigma"])

        rows.append(dict(
            config=tag, seed=seed, n_params=int(cfg["n_params"]),
            n_val=len(idx), n_val_pos=int((labels == 1).sum()),
            auc_heldout=auc, auc_se_delong=se,
            checkpoint="last.pt (never used for model selection)",
        ))
        per_seed.setdefault(seed, {})[tag] = auc
        logger.info("%-4s seed %d : AUC = %.6f +- %.6f (weighted DeLong, n_val=%d)",
                    tag, seed, auc, se, len(idx))

    if not rows:
        logger.error("no runs found under %s", args.runs_dir)
        return 1

    import csv
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    logger.info("wrote %s (%d rows)", args.out, len(rows))

    # --- paired differences against the paper configuration ------------------
    # Every configuration is compared to A3 (the manuscript's) on the SAME
    # validation events, seed by seed. The paired difference removes the
    # seed-to-seed spread, which is the dominant variance here: across seeds
    # the AUC scatter is ~6e-4, while the effects under test are of that order.
    REF = "A3"
    LABELS = {
        "A0":  "no pairing tokens (grades 0,1 only)",
        "A1":  "pairing tokens, grade-0 invariants only",
        "A2":  "pairing tokens, grades 1+3, no grade-2 meet",
        "A3":  "full pairing tokens (paper configuration)",
        "A4":  "full + JoinBlock",
        "A5v": "reference tokens gamma_0, gamma_3",
        "A5b": "reference token gamma_03 (H_LHC by construction)",
        "A6":  "grade-0 only (PELICAN-like limit)",
    }
    tags = sorted({t for d in per_seed.values() for t in d} - {REF},
                  key=lambda t: list(LABELS).index(t) if t in LABELS else 99)

    print(f"\nPaired differences vs {REF} (same validation events within each seed):")
    print(f"{'config':5s} {'n':>2s} {'mean d(AUC)':>12s} {'sem':>9s} {'t':>6s}   description")
    summary = []
    for tag in tags:
        d = [(s, v[tag] - v[REF]) for s, v in sorted(per_seed.items())
             if tag in v and REF in v]
        if not d:
            continue
        diffs = np.array([x for _, x in d])
        mean = float(diffs.mean())
        if len(diffs) > 1:
            sd = float(diffs.std(ddof=1))
            sem = sd / np.sqrt(len(diffs))
            t = mean / sem if sem > 0 else float("nan")
        else:
            sd = sem = t = float("nan")
        print(f"{tag:5s} {len(diffs):2d} {mean:+12.6f} {sem:9.6f} {t:6.2f}   {LABELS.get(tag,'')}")
        summary.append(dict(config=tag, n_paired=len(diffs), mean_delta_auc=mean,
                            sd=sd, sem=sem, t_stat=t, reference=REF,
                            description=LABELS.get(tag, "")))

    if summary:
        pth = str(Path(args.out).with_name(Path(args.out).stem + "_paired.csv"))
        with open(pth, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            wr.writeheader(); wr.writerows(summary)
        print(f"\nwrote {pth}")
        print("\nA negative mean means the paper configuration scores HIGHER than that")
        print("rung, i.e. the input content it adds helps. A value within ~2 sem of")
        print("zero means the higher-grade content is representationally natural but")
        print("empirically neutral on this benchmark — which is a publishable result,")
        print("and the one the existing single-seed evidence points to.")
    else:
        print(f"  (nothing to compare yet: need {REF} and another rung on a shared seed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
