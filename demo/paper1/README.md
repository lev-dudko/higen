# Paper 1 demo — GATr-lite for $tWb$ vs $t\bar t$ at the parton level

This directory contains a self-contained reference implementation of
the geometric-algebra (GA) network used in the first paper of the
`higen` programme, together with the high-level-feature MLP baseline
of [Boos:2023kpp] (arXiv:2306.08793) and everything required to
reproduce Fig. 5 / Fig. 6 / Tab. 3 of the paper from the raw LHE
inputs.

The task is a binary classification of single-resonant $gg\to tWb$
events (DR1, label 1) against double-resonant $gg\to t\bar t$ events
(label 0) at $\sqrt{s}=14~\mathrm{TeV}$, with the full gauge-invariant
SM tWb sample shown alongside as an unlabeled cross-check.

The GA network has $\approx 1.5\times 10^{5}$ parameters and takes
only the six final-state parton 4-momenta plus their PDG ids. It
matches and slightly exceeds the 75-feature MLP baseline, which has
$\approx 5.4\times 10^{5}$ parameters and consumes hand-crafted
high-level inputs.

| Network | Inputs | Parameters | AUC (eval) |
|---|---|---:|---:|
| **GATr-lite** (5 seeds, this code) | 6 × 4-momenta + PDG | 151 009 | $0.9653 \pm 0.0006$ |
| REF MLP [Boos:2023kpp] | 75 high-level features | 539 501 | $0.9594$ |

The GA-network checkpoint of one representative seed is shipped under
[`checkpoints/`](checkpoints/) and the REF Keras weights under
[`weights/`](weights/), so the eval/plot pipeline can be run without
re-training.

## Repository layout

```
demo/paper1/
├── README.md                 # this file
├── pyproject.toml            # `pip install -e .` works from this directory
├── requirements.txt
├── checkpoints/              # GATr-lite best.pt + config.json (seed 42)
├── weights/                  # REF baseline Keras h5 (Boos:2023kpp)
└── paper1_demo/
    └── gatr_lite/
        ├── data/             # LHE parser + preprocess to HDF5 + Dataset
        ├── gatr/             # Cl(1,3) algebra, equivariant layers, GATr-lite
        ├── training/         # train.py, eval.py, ref_baseline.py, sweep/
        ├── scripts/          # preprocess_all.sh, smoke_test.sh, plots, aggregate
        └── tests/            # 51 pytest tests
```

## Physics setup

| Property | Value |
|---|---|
| Process | $gg\to t\bar b W^-$ (DR1 / $t\bar t$ DR2 / full SM) |
| Centre-of-mass energy | $\sqrt{s} = 14$ TeV (7 TeV per beam) |
| Generator | CompHEP 4.5.2rc12 |
| PDF | CTEQ 6L1 |
| Final state (6 partons) | $\mu^- \nu_\mu u \bar d b \bar b$ |
| Format | LHEF 1.0 |
| Selection cut | $p_T(\mathrm{jet}_4) > 10$ GeV |
| Events per file | $\sim 10^6$ |

The leptonic decay is taken from the anti-top side ($\mu^- + \bar\nu_\mu$
from $W^-$); the hadronic side decays to $u + \bar d$ from $W^+$.

The three LHE files are:

| Sample | Diagrams | $\sigma$ (pb) | Events after cut | Role |
|---|---|---:|---:|---|
| **DR1** | single-resonant $tWb$ only | 4.85 | 267 983 | signal (label 1) |
| **tT** | double-resonant $t\bar t$ only | 18.55 | 10 115 042 | background (label 0) |
| **full** | full gauge-invariant SM | 22.08 | 805 639 | cross-check |

## Downloading the LHE samples

The three LHE samples are large (~13 GB uncompressed). They are
distributed compressed with **zstd** from the SINP HEP web server.
Replace the target directory if you prefer to keep the data elsewhere
and override `LHE_BASE` when calling the scripts below.

```bash
mkdir -p samples && cd samples

BASE_URL=https://www-hep.sinp.msu.ru/~dudko/higen/paper1
for f in \
  GG_DR1_NeeuDbB_SM_mu.lhe.zst \
  GG_tT_NeeuDbB_SM_mu.lhe.zst \
  GG_tT_tWb_NeeuDbB_SM_mu.lhe.zst ; do
    curl -O $BASE_URL/$f
    zstd -d $f
done
cd ..
```

The decompressed files are named exactly as the defaults in the
preprocessing scripts expect (`samples/GG_*_NeeuDbB_SM_mu.lhe`).

## Installation

A modern Python 3.10+ is required. The only mandatory dependencies
are PyTorch, NumPy, h5py and matplotlib. Optional extras:

* `uproot` — read the REF exam ROOT ntuple in
  `scripts/ref_inference{,_lhe}.py` (not redistributed here).
* `tensorboard` — per-step training metrics.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# or, equivalently, install the package itself:
pip install -e .
```

Tests:

```bash
# Most tests run without the LHE inputs; pass the DR1 path to enable parser tests:
HIGEN_DR1_LHE=samples/GG_DR1_NeeuDbB_SM_mu.lhe pytest paper1_demo/gatr_lite/tests
```

## 5-minute smoke test

Once the DR1 and tT samples are in `samples/`:

```bash
bash paper1_demo/gatr_lite/scripts/smoke_test.sh
```

It preprocesses 10 000 events per class into `data/smoke/{dr1,tT}.h5`,
trains the GA network for 5 epochs, and writes the run under
`runs/smoke_<timestamp>/`. Expected final val AUC is in the 0.55–0.65
range on a fresh seed.

If no GPU is available, run

```bash
DEVICE=cpu bash paper1_demo/gatr_lite/scripts/smoke_test.sh
```

(slower, but functional).

## Full pipeline

### 1. Preprocess the three LHE files into HDF5

```bash
bash paper1_demo/gatr_lite/scripts/preprocess_all.sh
# writes data/{dr1,tT,full}.h5
```

This applies the canonical 6-parton selection and the $p_T(\mathrm{jet}_4)
> 10$ GeV cut, stores the contravariant 4-momenta and per-parton PDG
ids in HDF5 together with the per-file cross-section attribute.

### 2. Train one full run (50 epochs, ~45 min on a modern GPU)

```bash
python -m paper1_demo.gatr_lite.training.train \
    --signal     data/dr1.h5 \
    --background data/tT.h5 \
    --out        runs/full_seed42 \
    --batch-size 512 --epochs 50 --lr 1e-3 --weight-decay 1e-4 \
    --val-frac 0.1 --seed 42 --device cuda \
    --early-stop-patience 15 --grad-clip 1.0
```

For the paper we trained five seeds (42–46) with the same
hyper-parameters; the multi-seed AUC is $0.9653 \pm 0.0006$.

### 3. Evaluate one run on all three samples

```bash
python -m paper1_demo.gatr_lite.training.eval \
    --checkpoint runs/full_seed42/best.pt \
    --signal     data/dr1.h5 \
    --background data/tT.h5 \
    --extra      data/full.h5 \
    --out-dir    runs/full_seed42/eval \
    --device cuda
```

Writes `predictions.npz` with per-event scores, labels and weights
together with ROC / discriminator PDFs.

### 4. Evaluate REF baseline (Boos:2023kpp)

The REF Keras weights are shipped in
`weights/ref_baseline_boos2023.h5`. To run them on the LHE samples
you also need the REF training config and per-sample ROOT ntuples,
which are not redistributed in this repository; request them from
the authors of [Boos:2023kpp] and follow the docstring of
`paper1_demo/gatr_lite/scripts/ref_inference_lhe.py`.

### 5. Aggregate multi-seed runs and reproduce the paper figures

```bash
python -m paper1_demo.gatr_lite.scripts.aggregate_results \
    --runs-dir ./runs \
    --out-dir  ./runs/aggregate
# writes auc_summary.csv, roc_band.{pdf,png}, discriminator.{pdf,png}, summary.json
```

## Architecture notes

The GA network operates on six 4-momentum tokens plus two
*pairing-aware tokens* whose grade-0/-1/-3 components encode pre-computed
Cayley–Menger-equivalent features ($s_{\rm had}^{(a)}$, $s_{\rm lep}^{(a)}$,
$\sigma_\pm^{Wb}$, $m_W^{(a)}$, $T^{(a)} = p_{j_1}\wedge p_{j_2}\wedge p_b$,
$\star T^{(a)}$). All quantities live in $\mathrm{Cl}(1,3)$ with the
$(+,-,-,-)$ metric.

Each of the three GATr blocks consists of (i) a per-grade
`EquivariantLinear`, (ii) a `GeometricProduct` with pair contraction and
channel mixing, and (iii) a `ScalarAttention` whose logits are Lorentz
invariants. The readout takes the grade-0 part of the final tokens
followed by a permutation-invariant mean pool and a small dense head.
Equivariance under the full $\mathrm{Spin}^+(1,3)$ holds by
construction and is verified by `tests/test_equivariance.py` to a
tolerance of $10^{-5}$.

## License

MIT — see [`../../LICENSE`](../../LICENSE).

## References

* [Boos:2023kpp] E. Boos *et al.*, "Singling out single-top:
  a deep-learning approach", arXiv:2306.08793.
* [Brehmer:2023] J. Brehmer *et al.*, "Geometric Algebra Transformer",
  NeurIPS 2023, arXiv:2305.18415.

## How to cite

A citation entry for the accompanying paper will be added here when
the preprint is released.
