# higen

A foundation-model programme for collider events built on the
real spacetime Clifford algebra $\mathrm{Cl}(1,3)$ and (later) the
conformal extension $\mathrm{Cl}(2,4)$. Companion repository for the
publications listed below.

## Contents

| Path | What is in it |
|------|---------------|
| `demo/paper1/` | Self-contained reference implementation of the GATr-lite demo used in our first paper: a Lorentz-equivariant classifier that separates single-resonant $tWb$ from double-resonant $t\bar t$ production at the LHE parton level (14 TeV, CompHEP 4.5.2rc12). Includes pretrained weights, a 75-feature MLP baseline, and the eight-configuration input ablation of the revised paper. |
| `doc/paper1/` | Final PDF of the first paper. *(Coming soon.)* |

## Why this exists

The published demo separates $gg\to t\bar bW^-$ topologies under
exactly the same parton-level setup that has been used in the SINP
top-physics group for two decades, so that the geometric-algebra
network can be compared one-to-one with the high-level-feature
baseline of [Boos:2023kpp] (arXiv:2306.08793). The companion code
makes that comparison reproducible end-to-end from raw LHE inputs.

It also makes the paper's central measurement reproducible: an ablation
over eight input representations that separates what supplying algebraic
content explicitly can and cannot buy. Content the network can
reconstruct from the measured momenta adds nothing when supplied at the
input layer; a fixed reference blade encoding the beam plane, which no
product of those momenta generates, is worth a substantial gain at no
extra parameters. The flags that select each configuration are listed in
[`demo/paper1/README.md`](demo/paper1/README.md).

## Quick start

```bash
git clone https://github.com/lev-dudko/higen.git
cd higen/demo/paper1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 5-minute smoke test on 10k events per class, see demo/paper1/README.md
bash paper1_demo/gatr_lite/scripts/smoke_test.sh
```

See [`demo/paper1/README.md`](demo/paper1/README.md) for the full
pipeline and the URLs to download the three LHE samples used in the
paper.

## License

MIT — see [LICENSE](LICENSE).
