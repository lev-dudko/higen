"""Clifford algebra Cl(1,3) primitives for GATr-lite.

Conventions
-----------
Metric: signature (+, -, -, -), i.e. e_0^2 = +1, e_i^2 = -1 for i = 1, 2, 3.

Basis (16 elements), ordered by grade then lex within grade::

    grade 0: 1
    grade 1: e_0, e_1, e_2, e_3
    grade 2: e_01, e_02, e_03, e_12, e_13, e_23
    grade 3: e_012, e_013, e_023, e_123
    grade 4: e_0123  (pseudoscalar)

A rotor R lies in Spin^+(1,3) and is built as exp(B/2) for a bivector B.
The equivariant action on a multivector X is the sandwich

    X' = R * X * reverse(R).

Both numpy and torch backends are provided; the geometric product, reverse,
inner product, and rotor sandwich operate on tensors of shape (..., 16).

Boost convention: a boost of rapidity y along axis i uses bivector
B = -y/2 * e_{0i}, which yields E' = cosh(y) E + sinh(y) p_i etc. (active
transformation of the 4-vector via R x R~).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is required at runtime
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Static algebra metadata
# ---------------------------------------------------------------------------

METRIC: Tuple[int, ...] = (1, -1, -1, -1)


def _build_basis() -> Tuple[Tuple[int, ...], ...]:
    """Return the 16 basis blades as sorted index tuples in canonical order."""
    blades = []
    for k in range(5):
        for combo in combinations(range(4), k):
            blades.append(tuple(combo))
    return tuple(blades)


BASIS: Tuple[Tuple[int, ...], ...] = _build_basis()
BASIS_GRADES: Tuple[int, ...] = tuple(len(b) for b in BASIS)
_BASIS_TO_INDEX = {b: i for i, b in enumerate(BASIS)}
N_BASIS = 16


def _blade_product(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, Tuple[int, ...]]:
    """Multiply two basis blades; return (sign, sorted index tuple)."""
    seq = list(a) + list(b)
    sign = 1
    changed = True
    while changed:
        changed = False
        for i in range(len(seq) - 1):
            if seq[i] > seq[i + 1]:
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
                sign = -sign
                changed = True
            elif seq[i] == seq[i + 1]:
                idx = seq[i]
                sign *= METRIC[idx]
                del seq[i:i + 2]
                changed = True
                break
    return sign, tuple(seq)


def _build_cayley() -> Tuple[np.ndarray, np.ndarray]:
    signs = np.zeros((N_BASIS, N_BASIS), dtype=np.int8)
    targets = np.zeros((N_BASIS, N_BASIS), dtype=np.int8)
    for i, a in enumerate(BASIS):
        for j, b in enumerate(BASIS):
            s, r = _blade_product(a, b)
            signs[i, j] = s
            targets[i, j] = _BASIS_TO_INDEX[r]
    return signs, targets


CAYLEY_SIGNS, CAYLEY_TARGETS = _build_cayley()


def _build_reverse_signs() -> np.ndarray:
    """Per-basis sign for the Clifford reverse: (-1)^(k(k-1)/2)."""
    out = np.zeros(N_BASIS, dtype=np.int8)
    for i, k in enumerate(BASIS_GRADES):
        out[i] = (-1) ** (k * (k - 1) // 2)
    return out


REVERSE_SIGNS: np.ndarray = _build_reverse_signs()


def _build_dense_cayley() -> np.ndarray:
    """Return a dense (16, 16, 16) tensor M[i, j, k] giving e_i e_j coefficient at k.

    Allows the geometric product to be expressed as a single einsum, which is
    fast on both numpy and torch (and yields proper autograd).
    """
    M = np.zeros((N_BASIS, N_BASIS, N_BASIS), dtype=np.float32)
    for i in range(N_BASIS):
        for j in range(N_BASIS):
            M[i, j, int(CAYLEY_TARGETS[i, j])] = float(CAYLEY_SIGNS[i, j])
    return M


CAYLEY_DENSE: np.ndarray = _build_dense_cayley()


# ---------------------------------------------------------------------------
# Backend dispatch helpers
# ---------------------------------------------------------------------------

def _is_torch(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _cayley_dense_like(x):
    """Return CAYLEY_DENSE on the same backend/device/dtype as x."""
    if _is_torch(x):
        return torch.as_tensor(CAYLEY_DENSE, dtype=x.dtype, device=x.device)
    return CAYLEY_DENSE.astype(x.dtype, copy=False)


def _reverse_signs_like(x):
    if _is_torch(x):
        return torch.as_tensor(REVERSE_SIGNS, dtype=x.dtype, device=x.device)
    return REVERSE_SIGNS.astype(x.dtype, copy=False)


# ---------------------------------------------------------------------------
# Core operations (numpy / torch dual)
# ---------------------------------------------------------------------------

def geometric_product(x, y):
    """Geometric product on the last axis (size 16).

    Both inputs must broadcast to a common shape with last dim 16.
    """
    M = _cayley_dense_like(x)
    if _is_torch(x) or _is_torch(y):
        if not _is_torch(x):
            x = torch.as_tensor(x, dtype=y.dtype, device=y.device)
        if not _is_torch(y):
            y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        # Broadcast x and y to a common shape, then einsum.
        # einsum: "...i, ...j, ijk -> ...k"
        return torch.einsum("...i,...j,ijk->...k", x, y, M)
    return np.einsum("...i,...j,ijk->...k", x, y, M)


def clifford_reverse(x):
    """Apply the Clifford reverse operator on the last axis."""
    s = _reverse_signs_like(x)
    return x * s


def inner_product(x, y):
    """Lorentz-invariant scalar product <reverse(x) y>_0.

    Returns a tensor with the trailing 16-axis removed.
    """
    z = geometric_product(clifford_reverse(x), y)
    return z[..., 0]


def apply_rotor(x, rotor):
    """Sandwich product R x R~. Both args must have last dim 16; broadcast."""
    rt = clifford_reverse(rotor)
    return geometric_product(geometric_product(rotor, x), rt)


def mv_exp(bivector, n_terms: int = 14):
    """Exponential of a multivector via truncated Taylor series.

    For a pure bivector this converges quickly; n_terms=14 gives <1e-7 error
    for ||B|| < ~2.5 (boost rapidity 1, rotation pi/2 well covered).
    """
    if _is_torch(bivector):
        out = torch.zeros_like(bivector)
        out[..., 0] = 1.0
        term = torch.zeros_like(bivector)
        term[..., 0] = 1.0
        for n in range(1, n_terms + 1):
            term = geometric_product(term, bivector) / float(n)
            out = out + term
        return out
    out = np.zeros_like(bivector, dtype=np.result_type(bivector, np.float64))
    out[..., 0] = 1.0
    term = np.zeros_like(out)
    term[..., 0] = 1.0
    for n in range(1, n_terms + 1):
        term = geometric_product(term, bivector) / float(n)
        out = out + term
    return out


# ---------------------------------------------------------------------------
# Pseudoscalar, Hodge dual, wedge, meet
# ---------------------------------------------------------------------------

PSEUDOSCALAR_IDX: int = 15  # index of e_0123 in the canonical basis
N_VECTOR_DIM: int = 4       # dimension of the underlying vector space (Cl(1,3))


def pseudoscalar_like(x):
    """Return a pseudoscalar e_0123 multivector on the same backend as x."""
    if _is_torch(x):
        out = torch.zeros(N_BASIS, dtype=x.dtype, device=x.device)
        out[PSEUDOSCALAR_IDX] = 1.0
        return out
    out = np.zeros(N_BASIS, dtype=x.dtype)
    out[PSEUDOSCALAR_IDX] = 1.0
    return out


def hodge_dual(x):
    """Hodge dual via right-multiplication by the pseudoscalar e_0123.

    Maps grade-k components to grade-(4-k); sign factors come from the
    Cayley table in the (+,-,-,-) signature.
    """
    return geometric_product(x, pseudoscalar_like(x))


def project_grade(x, k: int):
    """Zero out all components of x not in grade k."""
    if _is_torch(x):
        out = torch.zeros_like(x)
        for i in GRADE_INDICES[k]:
            out[..., i] = x[..., i]
        return out
    out = np.zeros_like(x)
    for i in GRADE_INDICES[k]:
        out[..., i] = x[..., i]
    return out


def wedge_to_grade(a, b, target_grade: int):
    """Wedge product as the grade-target_grade part of gp(a, b).

    For pure-grade inputs this is the antisymmetric outer product. Used in
    practice for known grade combinations: grade-1 ∧ grade-1 → grade-2,
    grade-2 ∧ grade-1 → grade-3, grade-1 ∧ grade-1 ∧ grade-1 ∧ grade-1 →
    grade-4, etc.
    """
    return project_grade(geometric_product(a, b), target_grade)


def meet(a, b, target_grade: int):
    """Meet operation: meet(A, B) = star( star(A) ∧ star(B) ).

    For two 3-blades in Cl(1,3) the meet is a 2-blade describing the
    common Lorentz subspace; for two 2-blades it is a 1-blade. The
    target_grade selects the projection of the result (since the wedge of
    duals may pick up off-grade components from the chosen Cayley
    convention).
    """
    a_dual = hodge_dual(a)
    b_dual = hodge_dual(b)
    inner_target = N_VECTOR_DIM - target_grade  # 4 - target_grade
    inner = wedge_to_grade(a_dual, b_dual, inner_target)
    return project_grade(hodge_dual(inner), target_grade)


# ---------------------------------------------------------------------------
# Random rotor generation (numpy by default; for tests)
# ---------------------------------------------------------------------------

# Bivector basis indices (in canonical order)
IDX_E01 = _BASIS_TO_INDEX[(0, 1)]
IDX_E02 = _BASIS_TO_INDEX[(0, 2)]
IDX_E03 = _BASIS_TO_INDEX[(0, 3)]
IDX_E12 = _BASIS_TO_INDEX[(1, 2)]
IDX_E13 = _BASIS_TO_INDEX[(1, 3)]
IDX_E23 = _BASIS_TO_INDEX[(2, 3)]

GRADE1_INDICES = tuple(i for i, g in enumerate(BASIS_GRADES) if g == 1)
GRADE_INDICES = {k: tuple(i for i, g in enumerate(BASIS_GRADES) if g == k) for k in range(5)}


def random_rotor(boost_max: float = 0.5,
                 rotation_max: float = np.pi,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample a Spin^+(1,3) rotor as exp(B/2) with B random bivector.

    Boost rapidities are sampled uniformly in [-boost_max, boost_max] for the
    three e_{0i} components; rotation angles uniformly in [-rotation_max,
    rotation_max] for the three e_{ij} components.
    """
    if rng is None:
        rng = np.random.default_rng()
    B = np.zeros(N_BASIS, dtype=np.float64)
    boosts = rng.uniform(-boost_max, boost_max, size=3)
    rots = rng.uniform(-rotation_max, rotation_max, size=3)
    # Convention: boost rotor R = exp(-y/2 e_{0i}); see module docstring.
    B[IDX_E01] = -0.5 * boosts[0]
    B[IDX_E02] = -0.5 * boosts[1]
    B[IDX_E03] = -0.5 * boosts[2]
    # Rotation rotor R = exp(-theta/2 e_{ij}): standard 3D right-handed convention.
    B[IDX_E12] = -0.5 * rots[0]
    B[IDX_E13] = -0.5 * rots[1]
    B[IDX_E23] = -0.5 * rots[2]
    return mv_exp(B)


# ---------------------------------------------------------------------------
# Convenience helpers for tests / models
# ---------------------------------------------------------------------------

def lorentz_matrix_from_rotor(rotor: np.ndarray) -> np.ndarray:
    """Extract the 4x4 Lorentz matrix Lambda^mu_nu acting on grade-1 components.

    For each basis 4-vector e_nu, compute R e_nu R~ and read out grade-1
    coefficients. Result satisfies (R x R~)^mu = Lambda^mu_nu x^nu for any
    grade-1 x.
    """
    L = np.zeros((4, 4), dtype=np.float64)
    for nu in range(4):
        ev = np.zeros(N_BASIS)
        ev[GRADE1_INDICES[nu]] = 1.0
        out = apply_rotor(ev, rotor)
        for mu in range(4):
            L[mu, nu] = out[GRADE1_INDICES[mu]]
    return L


@dataclass(frozen=True)
class AlgebraInfo:
    """Lightweight container exposing static algebra info to layers."""
    n_basis: int = N_BASIS
    grade_indices: Tuple[Tuple[int, ...], ...] = tuple(GRADE_INDICES[k] for k in range(5))


ALGEBRA_INFO = AlgebraInfo()
