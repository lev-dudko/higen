# Paper 1 demo — GATr-lite for $tWb$ vs $t\bar t$ at the parton level

This directory contains a self-contained reference implementation of
the geometric-algebra (GA) network used in the first paper of the
`higen` programme, together with the high-level-feature MLP baseline
of [Boos:2023kpp] (arXiv:2306.08793) and everything required to
reproduce Tab. 3 and Figs. 5-7 of the paper from the raw LHE inputs.

The task is a binary classification of single-resonant $gg\to tWb$
events (DR1, label 1) against double-resonant $gg\to t\bar t$ events
(label 0) at $\sqrt{s}=14~\mathrm{TeV}$, with the full gauge-invariant
SM tWb sample shown alongside as an unlabeled cross-check.

The GA network has $\approx 1.5\times 10^{5}$ parameters and takes
only the six final-state parton 4-momenta plus their PDG ids. It
exceeds the 75-feature MLP baseline, which has $\approx 5.4\times
10^{5}$ parameters and consumes hand-crafted high-level inputs, at
roughly a quarter of its parameter count.

| Network | Inputs | Parameters | AUC (eval) |
|---|---|---:|---:|
| **GATr-lite**, four-momenta (5 seeds) | 6 × 4-momenta + PDG | 149 409 | $0.9669 \pm 0.0007$ |
| **GATr-lite**, + reference bivector (5 seeds) | the same, plus one fixed $\gamma_{03}$ token | 149 409 | $0.9744 \pm 0.0007$ |
| REF MLP [Boos:2023kpp] | 75 high-level features | 539 501 | $0.9586$ |

Quoted uncertainties are the spread over the five seeds. The reference
bivector is a fixed, weightless token encoding the beam plane, so it
leaves the parameter count unchanged.

Two GA-network checkpoints are shipped under
[`checkpoints/`](checkpoints/) — `full_v1_alldata_seed42` (the pairing
configuration of the first release) and `refbivector_seed42` (the
reference-bivector configuration the revised paper reports, 149 409
parameters) — together with the REF Keras weights under
[`weights/`](weights/), so the eval and plotting steps can be run
without re-training.

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
        ├── training/         # train.py, eval.py, ref_baseline.py, auc_stats.py, sweep/
        ├── scripts/          # preprocess, smoke test, plots, aggregation, measurements
        └── tests/            # pytest suite incl. equivariance and ablation knobs
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

### 6. Reproduce the input ablation (Tab. 3, Fig. 5)

Eight input representations, five seeds each. Every configuration is the
same network on the same events; the flags change only what the input
tensor carries. Configurations within a group have identical parameter
counts, which is what makes the comparison a statement about the input
rather than about capacity.

| Input representation | Params | AUC | Flags added to the command in step 2 |
|---|---:|---|---|
| grade 0 only | 151 489 | $0.9627 \pm 0.0003$ | `--grade0-only` |
| + pairing invariants | 151 489 | $0.9650 \pm 0.0003$ | `--pairing-content scalars_only` |
| + pairing grades 1,2,3 | 151 489 | $0.9649 \pm 0.0008$ | *(default)* |
| + grade-4 join | 196 609 | $0.9652 \pm 0.0009$ | `--use-join-block` |
| + pairing 1,3 (no meet) | 151 489 | $0.9657 \pm 0.0009$ | `--pairing-content no_meet` |
| grade 0⊕1 four-momenta | 149 409 | $0.9669 \pm 0.0007$ | `--no-pairing-cache` |
| + reference tokens $\gamma_0,\gamma_3$ | 149 409 | $0.9670 \pm 0.0004$ | `--no-pairing-cache --reference-tokens --reference-mode vectors` |
| **+ reference bivector $\gamma_{03}$** | 149 409 | $0.9744 \pm 0.0007$ | `--no-pairing-cache --reference-tokens --reference-mode bivector` |

```bash
# one configuration, one seed
python -m paper1_demo.gatr_lite.training.train \
    --signal ./data/dr1.h5 --background ./data/tT.h5 \
    --out ./runs/A5b_s42 --epochs 50 --batch-size 512 --seed 42 \
    --no-pairing-cache --reference-tokens --reference-mode bivector

# collect every run under ./runs into the paper's table
python -m paper1_demo.gatr_lite.scripts.aggregate_ablation \
    --runs-dir ./runs --data-dir ./data --out ablation_summary.csv
```

`aggregate_ablation.py` scores each run on the same held-back events,
pairs configurations by seed, and reports the paired differences with
DeLong uncertainties (`training/auc_stats.py`). Paired differences are
what the paper quotes: the effects under test are of the same order as
the seed-to-seed scatter, so differences of across-seed means would be
too noisy to read.

### 7. Supporting measurements

```bash
# per-event discriminator distributions (Fig. 7)
python -m paper1_demo.gatr_lite.scripts.score_discriminator \
    --run ./runs/A5b_s42 --data-dir ./data --out ./runs/discr_A5b.npz --all-events

# what each grade carries on this final state (Sec. 7.1)
python -m paper1_demo.gatr_lite.scripts.measure_grade_physics --data-dir ./data

# numerical dynamic range per grade (Sec. 8.1)
python -m paper1_demo.gatr_lite.scripts.measure_grade_ranges --data-dir ./data

# is the pseudoscalar sign asymmetry physical, or an artefact of parton
# labelling? (Sec. 8.1)
python -m paper1_demo.gatr_lite.scripts.cp_asymmetry_check --data-dir ./data
```

## Architecture notes

The network has two input branches, and the ablation above walks
between them. In the **per-particle** branch each of the six partons is
one token carrying its 4-momentum at grade 1 and its type one-hot at
grade 0; every higher grade is left empty and is formed inside the
network by attention across tokens and by the geometric product within
a token. In the **pairing** branch two further event-level tokens
arrive with grades 1-3 already filled by the resonance-topology
candidates. Appendix D of the paper lays out both token formats cell by
cell.

The pairing tokens carry, at grades 0/1/3, pre-computed
Cayley–Menger-equivalent features ($s_{\rm had}^{(a)}$, $s_{\rm lep}^{(a)}$,
$\sigma_\pm^{Wb}$, $m_W^{(a)}$, $T^{(a)} = p_{j_1}\wedge p_{j_2}\wedge p_b$,
$\star T^{(a)}$). All quantities live in $\mathrm{Cl}(1,3)$ with the
$(+,-,-,-)$ metric.

Each of the three GATr blocks consists of (i) a per-grade
`EquivariantLinear`, (ii) a `GeometricProduct` with pair contraction and
channel mixing, and (iii) a `ScalarAttention` whose logits are Lorentz
invariants. The readout takes the grade-0 part of the final tokens
followed by a permutation-invariant mean pool and a small dense head.
Equivariance under the full $\mathrm{Spin}^+(1,3)$ follows from the
per-grade structure of the layers and is verified numerically by
`tests/test_equivariance.py` to a tolerance of $10^{-5}$.
`tests/test_ablation_knobs.py` checks that the ablation flags change
what the input carries without changing the parameter count or breaking
equivariance, and `tests/test_hlhc_reference.py` checks that the fixed
$\gamma_{03}$ token is invariant under both generators of the residual
collider symmetry while $\gamma_0,\gamma_3$ separately are not.

## License

MIT — see [`../../LICENSE`](../../LICENSE).

## References

* [Boos:2023kpp] E. Boos *et al.*, "Singling out single-top:
  a deep-learning approach", arXiv:2306.08793.
* [Brehmer:2023] J. Brehmer *et al.*, "Geometric Algebra Transformer",
  NeurIPS 2023, arXiv:2305.18415.

## How to cite

```bibtex
@article{higen-paper1,
  author  = {Abasov, E. and Dudko, L. V. and Grigoryev, F. and
             Volkov, P. and Zaborenko, A.},
  title   = {Geometric algebra as the input language of collider
             foundation models},
  journal = {arXiv preprint},
  eprint  = {2605.15910},
  archivePrefix = {arXiv},
  year    = {2026},
}
```
