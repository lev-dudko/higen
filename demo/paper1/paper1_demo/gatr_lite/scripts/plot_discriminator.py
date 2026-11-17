#!/usr/bin/env python3
"""Fig. 7 of paper1: discriminator distributions of the three parton-level samples.

Each curve is d(sigma)/dD in pb, normalised to the cross section of its sample
AFTER the event selection pT(j4) > 10 GeV:

    sigma_sel = xsec_pb * n_passed_cut / n_total_in_file

The attribute xsec_pb of the HDF5 files is the cross section of the LHE file
before the selection.  Dividing it by the number of selected events (as
PartonH5Dataset.event_weight did before 2026-11-17) overstates the tW curve by 1/0.245 = 4.08,
the ttbar curve by 1.087 and the full-matrix-element curve by 1.241, because
the selection efficiencies of the three samples differ.

Input: the npz files written by scripts/score_discriminator.py
(keys signal_scores, background_scores, extra_scores) and the three HDF5 files.

The thin grey line is the incoherent sum tW + ttbar.  Its difference from the
full-matrix-element curve is the interference of the two classes.

Steps to reproduce Fig. 7 (run A3 = pairing grades 1,2,3; run A5b = reference
bivector; seed 42; final-epoch weights):

    python -m paper1_demo.gatr_lite.scripts.score_discriminator --all-events \
        --run RUNS/run_A3_s42  --data-dir DATA --out A3_s42_all.npz
    python -m paper1_demo.gatr_lite.scripts.score_discriminator --all-events \
        --run RUNS/run_A5b_s42 --data-dir DATA --out A5b_s42_all.npz
    python plot_discriminator.py --pairing A3_s42_all.npz --bivector A5b_s42_all.npz \
        --data-dir DATA --out gatr_discriminator_v3 [--no-sum]

DATA holds dr1.h5, tT.h5 and full.h5 written by data/preprocess.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SAMPLES = (("signal", "dr1.h5"), ("background", "tT.h5"), ("extra", "full.h5"))
STYLE = {
    "signal": dict(color="#b2182b", lw=1.6, ls="-", label=r"$tW$ (single resonant)"),
    "background": dict(color="#2166ac", lw=1.6, ls="-", label=r"$t\bar t$ (double resonant)"),
    "extra": dict(color="#333333", lw=1.3, ls=(0, (1, 1.6)), label="full matrix element"),
    "sum": dict(color="#888888", lw=0.9, ls="-", label=r"$tW+t\bar t$ without interference"),
}


def selected_xsec(data_dir: Path) -> dict:
    """Cross section of each sample after the pT(j4) selection, in pb."""
    out = {}
    for key, name in SAMPLES:
        with h5py.File(data_dir / name, "r") as f:
            a = f.attrs
            eff = int(a["n_passed_cut"]) / int(a["n_total_in_file"])
            out[key] = dict(xsec_lhe=float(a["xsec_pb"]), efficiency=eff,
                            xsec_selected=float(a["xsec_pb"]) * eff)
    return out


def dsigma_dD(scores: np.ndarray, xsec_selected: float, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(scores, bins=edges)
    return counts / len(scores) * xsec_selected / np.diff(edges)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairing", required=True)
    p.add_argument("--bivector", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True, help="output stem; .pdf, .png and .json are written")
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--no-sum", action="store_true",
                   help="do not draw the incoherent sum tW + ttbar")
    args = p.parse_args(argv)

    xs = selected_xsec(Path(args.data_dir))
    edges = np.linspace(0.0, 1.0, args.bins + 1)
    panels = (("pairing grades 1,2,3", np.load(args.pairing, allow_pickle=True)),
              (r"reference bivector $\gamma_{03}$", np.load(args.bivector, allow_pickle=True)))

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "xtick.direction": "out", "ytick.direction": "out"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True, gridspec_kw=dict(wspace=0.07))
    summary = {"xsec": xs, "bins": args.bins, "panels": {}}
    for ax, (title, z) in zip(axes, panels):
        h = {k: dsigma_dD(z[f"{k}_scores"], xs[k]["xsec_selected"], edges) for k, _ in SAMPLES}
        if not args.no_sum:
            ax.stairs(h["signal"] + h["background"], edges, **STYLE["sum"])
        for k in ("background", "signal", "extra"):
            ax.stairs(h[k], edges, **STYLE[k])
        ax.set_yscale("log")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.06, 900.0)
        ax.set_xlabel(r"network output $D$")
        ax.set_title(title, loc="left")
        width = edges[1] - edges[0]
        summary["panels"][title] = {
            "n_scored": {k: int(len(z[f"{k}_scores"])) for k, _ in SAMPLES},
            "first_bin": {k: float(h[k][0]) for k in h}, "last_bin": {k: float(h[k][-1]) for k in h},
            "integral_pb": {k: float(h[k].sum() * width) for k in h},
            "interference_pb": float((h["extra"] - h["signal"] - h["background"]).sum() * width),
            "tW_fraction_D_gt_0.95": float((z["signal_scores"] > 0.95).mean()),
            "tt_fraction_D_lt_0.05": float((z["background_scores"] < 0.05).mean()),
        }
    axes[0].set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}D$  [pb]")
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    axes[0].set_xticks(ticks, ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    axes[1].set_xticks(ticks, ["", "0.2", "0.4", "0.6", "0.8", "1.0"])   # "0.0" would touch the left panel's "1.0"
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(STYLE[k]["label"]) for k in ("signal", "background", "extra") + (() if args.no_sum else ("sum",))]
    axes[0].legend([handles[i] for i in order], [labels[i] for i in order], frameon=False,
                   loc="upper center", bbox_to_anchor=(0.56, 1.0), handlelength=2.2)
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.15, top=0.92)
    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png", dpi=300)
    json.dump(summary, open(f"{args.out}.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
