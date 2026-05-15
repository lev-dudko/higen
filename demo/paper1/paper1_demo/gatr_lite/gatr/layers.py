"""Equivariant layers for GATr-lite (Cl(1,3) backbone).

All layers act on tensors of shape (B, T, C, 16) where the last axis indexes
the multivector basis. Equivariance under the sandwich action of a rotor
R \\in Spin^+(1,3) (i.e. X -> R X reverse(R)) is preserved by every layer.

Design choices that make the layers equivariant:

* ``EquivariantLinear`` only mixes channels within the same grade -- five
  weight matrices, one per grade. Mixing across grades would break
  equivariance because rotors act differently on each grade.

* ``GeometricProduct`` is bilinear in two equivariant inputs and the
  geometric product itself is equivariant: gp(RxR~, RyR~) = R gp(x,y) R~.
  Channel mixing is implemented as scalar weights per channel pair, so the
  overall map remains equivariant.

* ``ScalarAttention`` derives attention scores from the Lorentz-invariant
  inner product <reverse(q) k>_0; values are full multivectors and are
  combined via scalar attention weights.

* ``GatedActivation`` produces a non-negative scalar gate from the scalar
  channel of the multivector and rescales the *whole* multivector by it. A
  scalar multiplier commutes with the rotor sandwich, hence equivariance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from . import algebra as A


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------

def _grade_masks(device, dtype) -> Tuple[torch.Tensor, ...]:
    """Return tuple of 5 masks of shape (16,) selecting each grade's basis.

    The masks are dtype-cast tensors where ``mask[i] = 1`` iff basis index
    ``i`` belongs to that grade.
    """
    masks = []
    for k in range(5):
        m = torch.zeros(16, dtype=dtype, device=device)
        for i in A.GRADE_INDICES[k]:
            m[i] = 1.0
        masks.append(m)
    return tuple(masks)


# ---------------------------------------------------------------------------
# EquivariantLinear: per-grade channel mixing
# ---------------------------------------------------------------------------

class EquivariantLinear(nn.Module):
    """Linear mixing of channels independently within each Cl(1,3) grade.

    Input shape:  (B, T, in_channels, 16)
    Output shape: (B, T, out_channels, 16)

    Five (out, in) weight matrices -- one per grade -- act on the channel
    axis. The 16 basis components belonging to a given grade share the same
    matrix, so the layer commutes with any rotor sandwich.
    """

    def __init__(self, in_channels: int, out_channels: int, bias_scalar: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = nn.ParameterList()
        for k in range(5):
            w = torch.empty(out_channels, in_channels)
            # Match nn.Linear's default init (Kaiming uniform with a=sqrt(5))
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            self.weights.append(nn.Parameter(w))
        # Bias only on the scalar (grade-0) channel -- bias on any other grade
        # would break equivariance because constants are not rotor-invariant.
        if bias_scalar:
            self.bias_scalar = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_scalar", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C_in, 16)
        out = torch.zeros(*x.shape[:-2], self.out_channels, 16, dtype=x.dtype, device=x.device)
        for k in range(5):
            idx = A.GRADE_INDICES[k]
            if not idx:
                continue
            sub = x[..., :, list(idx)]  # (B, T, C_in, n_k)
            # mix channels: einsum over C_in
            mixed = torch.einsum("...ci,oc->...oi", sub, self.weights[k])
            out[..., :, list(idx)] = mixed
        if self.bias_scalar is not None:
            out[..., :, 0] = out[..., :, 0] + self.bias_scalar
        return out


# ---------------------------------------------------------------------------
# GeometricProduct layer
# ---------------------------------------------------------------------------

class GeometricProduct(nn.Module):
    """Equivariant bilinear layer based on the Cl(1,3) geometric product.

    Two inputs ``x`` and ``y`` of shape (B, T, C, 16) are combined channel-
    wise: ``out_c = sum_{a,b} W[c, a, b] gp(x_a, y_b)``. To keep the
    parameter count modest, the layer first projects each input down to
    ``mid_channels`` via an internal ``EquivariantLinear``, then performs the
    pairwise product, then projects up.

    The resulting map is equivariant because ``gp(R x R~, R y R~) = R gp(x,
    y) R~`` and channel mixing only uses scalar weights.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels or max(4, min(in_channels, 8))
        self.proj_x = EquivariantLinear(in_channels, self.mid_channels, bias_scalar=False)
        self.proj_y = EquivariantLinear(in_channels, self.mid_channels, bias_scalar=False)
        # Combine pairwise products via a learnable (out, mid, mid) tensor.
        w = torch.empty(out_channels, self.mid_channels, self.mid_channels)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        # Scale down: pair-wise sums grow with mid_channels.
        with torch.no_grad():
            w.mul_(1.0 / math.sqrt(self.mid_channels))
        self.combine = nn.Parameter(w)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if y is None:
            y = x
        xp = self.proj_x(x)  # (B, T, M, 16)
        yp = self.proj_y(y)  # (B, T, M, 16)
        # gp_pair[a, b] = gp(xp[a], yp[b]) of shape (B, T, M, M, 16)
        # einsum directly: "...ai,...bj,ijk->...abk"
        cay = torch.as_tensor(A.CAYLEY_DENSE, dtype=x.dtype, device=x.device)
        gp_pair = torch.einsum("...ai,...bj,ijk->...abk", xp, yp, cay)
        # combine into out_channels: out[c] = sum_{a,b} W[c,a,b] * gp_pair[a,b]
        out = torch.einsum("...abk,oab->...ok", gp_pair, self.combine)
        return out


# ---------------------------------------------------------------------------
# ScalarAttention layer
# ---------------------------------------------------------------------------

class ScalarAttention(nn.Module):
    """Scalar-weighted, multivector-valued attention.

    Q, K, V each come from a separate ``EquivariantLinear``. Pairwise
    attention scores between tokens are derived from the Lorentz-invariant
    inner product ``<reverse(q) k>_0`` summed over channels. Softmax along
    the key/token axis turns these scores into scalar weights, which we use
    to mix multivector values. Since the weights are scalars and the values
    transform as multivectors, the attention output is itself equivariant.
    """

    def __init__(self, channels: int, attn_channels: int | None = None):
        super().__init__()
        self.channels = channels
        self.attn_channels = attn_channels or channels
        self.q_proj = EquivariantLinear(channels, self.attn_channels, bias_scalar=False)
        self.k_proj = EquivariantLinear(channels, self.attn_channels, bias_scalar=False)
        self.v_proj = EquivariantLinear(channels, channels, bias_scalar=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, 16)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Inner product per channel between tokens: <reverse(q_t) k_s>_0
        # Sum over channel axis -> (B, T_q, T_k) scalar score.
        q_rev = A.clifford_reverse(q)  # (B, T, C, 16)
        # gp on last axis, reduced to scalar (component 0) and summed over channels
        # We want: scores[b, i, j] = sum_c <reverse(q[b,i,c]) k[b,j,c]>_0
        # = sum_c sum_{a,b'} CAYLEY_DENSE[a,b',0] q_rev[b,i,c,a] k[b,j,c,b']
        cay = torch.as_tensor(A.CAYLEY_DENSE, dtype=x.dtype, device=x.device)
        cay0 = cay[..., 0]  # (16, 16)
        # Explicit matmul + bmm form (rather than a single 3-tensor einsum):
        # equivalent to the einsum but slightly faster, and works around an
        # observed cuBLAS strided-batched gemm bug on CUDA 12.8 / sm_120 (RTX
        # 5060 Ti). When run with the system CUDA 12.9 cuBLAS LD_PRELOAD'd,
        # both forms produce identical numerics.
        # (B, T, C, 16) @ (16, 16) -> (B, T, C, 16)
        q_rev_cay = torch.matmul(q_rev.contiguous(), cay0)
        B_, T_, C_, _ = q_rev_cay.shape
        q2d = q_rev_cay.reshape(B_, T_, C_ * 16)
        k2d = k.contiguous().reshape(B_, T_, C_ * 16)
        # scores[b, i, j] = sum_{c, q} q_rev_cay[b, i, c, q] * k[b, j, c, q]
        scores = torch.bmm(q2d, k2d.transpose(1, 2))
        scores = scores / math.sqrt(self.attn_channels * 16)
        attn = torch.softmax(scores, dim=-1)  # over keys
        # Mix values: out[b, i, c, k] = sum_j attn[b, i, j] v[b, j, c, k]
        # bmm form, mirrors the score computation above.
        Bv, Tv, Cv, Kv = v.shape
        v2d = v.contiguous().reshape(Bv, Tv, Cv * Kv)  # (B, T, C*16)
        out2d = torch.bmm(attn, v2d)                    # (B, T, C*16)
        out = out2d.reshape(Bv, Tv, Cv, Kv)
        return out


# ---------------------------------------------------------------------------
# MultiHeadScalarAttention layer (arch_v2)
# ---------------------------------------------------------------------------

class MultiHeadScalarAttention(nn.Module):
    """Multi-head version of ScalarAttention.

    Splits the ``channels`` axis into ``n_heads`` disjoint groups of size
    ``channels // n_heads``. For each head we run a separate scalar-attention
    head with its own Q/K/V ``EquivariantLinear`` projections, then
    concatenate the per-head outputs along the channel axis. ``attn_channels``
    is the per-head Q/K width (so the total Q/K parameter budget grows ~
    linearly with ``n_heads``).

    Equivariance argument is the same as the single-head case: scores are
    Lorentz-invariant per head, the softmax mixes value multivectors with
    scalar weights, and the channel-wise concat preserves grades. With
    ``n_heads=1`` the layer is functionally identical to ``ScalarAttention``.
    """

    def __init__(self, channels: int, n_heads: int = 1, attn_channels: int | None = None):
        super().__init__()
        if channels % n_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by n_heads ({n_heads})"
            )
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        # Per-head Q/K width; defaults to head_dim if not specified.
        self.attn_channels = attn_channels or self.head_dim

        self.q_projs = nn.ModuleList([
            EquivariantLinear(self.head_dim, self.attn_channels, bias_scalar=False)
            for _ in range(n_heads)
        ])
        self.k_projs = nn.ModuleList([
            EquivariantLinear(self.head_dim, self.attn_channels, bias_scalar=False)
            for _ in range(n_heads)
        ])
        self.v_projs = nn.ModuleList([
            EquivariantLinear(self.head_dim, self.head_dim, bias_scalar=False)
            for _ in range(n_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, 16); split channels into n_heads groups of head_dim.
        cay = torch.as_tensor(A.CAYLEY_DENSE, dtype=x.dtype, device=x.device)
        cay0 = cay[..., 0]  # (16, 16)
        scale = 1.0 / math.sqrt(self.attn_channels * 16)

        per_head_outs = []
        for h in range(self.n_heads):
            xh = x[..., h * self.head_dim:(h + 1) * self.head_dim, :]  # (B, T, hd, 16)
            q = self.q_projs[h](xh)
            k = self.k_projs[h](xh)
            v = self.v_projs[h](xh)

            q_rev = A.clifford_reverse(q)
            q_rev_cay = torch.matmul(q_rev.contiguous(), cay0)
            B_, T_, C_, _ = q_rev_cay.shape
            q2d = q_rev_cay.reshape(B_, T_, C_ * 16)
            k2d = k.contiguous().reshape(B_, T_, C_ * 16)
            scores = torch.bmm(q2d, k2d.transpose(1, 2)) * scale
            attn = torch.softmax(scores, dim=-1)

            Bv, Tv, Cv, Kv = v.shape
            v2d = v.contiguous().reshape(Bv, Tv, Cv * Kv)
            out2d = torch.bmm(attn, v2d)
            out_h = out2d.reshape(Bv, Tv, Cv, Kv)
            per_head_outs.append(out_h)

        return torch.cat(per_head_outs, dim=-2)  # concat along channel axis


# ---------------------------------------------------------------------------
# GatedActivation layer
# ---------------------------------------------------------------------------

class GatedActivation(nn.Module):
    """Per-channel scalar gate driven by the scalar (grade-0) component.

    For each channel we compute ``g = sigmoid(W * scalar_part(x) + b)`` and
    multiply the *whole* 16-dim multivector by ``g``. Because the scaling is
    by a scalar, the layer is equivariant.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.gate = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, 16)
        scalar = x[..., 0]  # (B, T, C)
        g = torch.sigmoid(self.gate(scalar))  # (B, T, C)
        return x * g.unsqueeze(-1)


# ---------------------------------------------------------------------------
# JoinProduct: anti-symmetrised geometric product (explicit wedge layer)
# ---------------------------------------------------------------------------

class JoinProduct(nn.Module):
    """Anti-symmetrised geometric product.

    For pure-grade A (grade k) and B (grade l), gp(A,B) = A∧B + lower-grade
    contractions; the anti-symmetric part isolates the wedge component
    (grade k+l) up to inner-product corrections. Concretely::

        out_c = sum_{a,b} W[c,a,b] * (gp(x_a, y_b) - gp(y_b, x_a)) / 2

    This biases the layer to build "join" objects (higher-grade products
    like 3-blades and pseudoscalars) that the symmetric GeometricProduct
    underplays, since gp(x, x) is symmetric and its wedge part vanishes.

    Equivariance is preserved: gp itself is equivariant, and the
    antisymmetric combination of two equivariant maps is equivariant.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 mid_channels: int | None = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels or max(4, min(in_channels, 8))
        self.proj_x = EquivariantLinear(in_channels, self.mid_channels, bias_scalar=False)
        self.proj_y = EquivariantLinear(in_channels, self.mid_channels, bias_scalar=False)
        w = torch.empty(out_channels, self.mid_channels, self.mid_channels)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        with torch.no_grad():
            w.mul_(1.0 / math.sqrt(self.mid_channels))
        self.combine = nn.Parameter(w)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if y is None:
            y = x
        xp = self.proj_x(x)  # (B, T, M, 16)
        yp = self.proj_y(y)  # (B, T, M, 16)
        cay = torch.as_tensor(A.CAYLEY_DENSE, dtype=x.dtype, device=x.device)
        gp_xy = torch.einsum("...ai,...bj,ijk->...abk", xp, yp, cay)
        gp_yx = torch.einsum("...bj,...ai,ijk->...abk", yp, xp, cay)
        wedge_pair = 0.5 * (gp_xy - gp_yx)
        out = torch.einsum("...abk,oab->...ok", wedge_pair, self.combine)
        return out


# ---------------------------------------------------------------------------
# JoinBlock: applies JoinProduct + token-mixing attention + gated nonlinearity
# ---------------------------------------------------------------------------

class JoinBlock(nn.Module):
    """A single residual block built around the antisymmetric wedge.

    Used after the stack of standard GATrBlocks: gives the network a direct
    way to construct grade-(k+l) objects (e.g. 4-blades from 3-blades and
    1-blades) which encode CP-odd / coplanarity invariants needed for
    pairing resolution. Without this, the network has to recover them
    indirectly through deep gp-residual interactions.
    """

    def __init__(self, cfg: BlockConfig | None = None):
        super().__init__()
        cfg = cfg or BlockConfig()
        self.cfg = cfg
        C = cfg.channels
        self.join_pre = EquivariantLinear(C, C, bias_scalar=False)
        self.join = JoinProduct(C, C, mid_channels=cfg.gp_mid_channels)
        self.join_post = EquivariantLinear(C, C)

        self.attn_pre = EquivariantLinear(C, C, bias_scalar=False)
        if cfg.n_heads > 1:
            self.attn = MultiHeadScalarAttention(
                C, n_heads=cfg.n_heads, attn_channels=cfg.attn_channels,
            )
        else:
            self.attn = ScalarAttention(C, attn_channels=cfg.attn_channels)

        self.gate_pre = EquivariantLinear(C, C, bias_scalar=False)
        self.gate = GatedActivation(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Join residual (antisymmetrised wedge).
        h = self.join_pre(x)
        h = self.join(h, h)
        h = self.join_post(h)
        x = x + h
        # Attention residual (token mixing on enriched features).
        h = self.attn_pre(x)
        h = self.attn(h)
        x = x + h
        # Gated nonlinearity.
        h = self.gate_pre(x)
        x = self.gate(h)
        return x


# ---------------------------------------------------------------------------
# GATrBlock
# ---------------------------------------------------------------------------

@dataclass
class BlockConfig:
    channels: int = 16
    gp_mid_channels: int = 8
    attn_channels: int = 8
    n_heads: int = 1            # arch_v2: multi-head scalar attention
    scalar_dropout: float = 0.0  # arch_v2: dropout on grade-0 channels post-block


class GATrBlock(nn.Module):
    """A residual block: GP-mix, scalar attention, gated nonlinearity.

    Each sub-step is wrapped in pre-projections so the block keeps a fixed
    channel width. All sub-layers are equivariant; sums of equivariant
    tensors stay equivariant, hence so does the residual.

    With ``cfg.n_heads > 1`` the attention sub-layer is replaced by
    ``MultiHeadScalarAttention``. With ``cfg.scalar_dropout > 0`` a per-channel
    dropout is applied to the grade-0 (scalar) components after the block;
    masking only the scalar part keeps the rotor-sandwich equivariance intact
    (scalars are rotor invariants, so multiplying them by a constant from a
    Bernoulli mask commutes with the sandwich).
    """

    def __init__(self, cfg: BlockConfig | None = None):
        super().__init__()
        cfg = cfg or BlockConfig()
        self.cfg = cfg
        C = cfg.channels
        self.gp_pre = EquivariantLinear(C, C, bias_scalar=False)
        self.gp = GeometricProduct(C, C, mid_channels=cfg.gp_mid_channels)
        self.gp_post = EquivariantLinear(C, C)

        self.attn_pre = EquivariantLinear(C, C, bias_scalar=False)
        if cfg.n_heads > 1:
            self.attn = MultiHeadScalarAttention(
                C, n_heads=cfg.n_heads, attn_channels=cfg.attn_channels,
            )
        else:
            self.attn = ScalarAttention(C, attn_channels=cfg.attn_channels)

        self.gate_pre = EquivariantLinear(C, C, bias_scalar=False)
        self.gate = GatedActivation(C)

        # Dropout on the scalar (grade-0) channels only -- preserves
        # equivariance because vector/bivector/trivector parts are untouched.
        self.scalar_dropout_p = float(cfg.scalar_dropout)
        if self.scalar_dropout_p > 0.0:
            self.scalar_dropout = nn.Dropout(self.scalar_dropout_p)
        else:
            self.scalar_dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GP residual
        h = self.gp_pre(x)
        h = self.gp(h, h)
        h = self.gp_post(h)
        x = x + h
        # Attention residual
        h = self.attn_pre(x)
        h = self.attn(h)
        x = x + h
        # Gated nonlinearity (no residual; gate is multiplicative)
        h = self.gate_pre(x)
        x = self.gate(h)
        # Scalar-only dropout (arch_v2): mask grade-0 channels per-channel.
        if self.scalar_dropout is not None and self.training:
            scalar = x[..., 0]                           # (B, T, C)
            scalar = self.scalar_dropout(scalar)
            x = torch.cat(
                [scalar.unsqueeze(-1), x[..., 1:]], dim=-1,
            )
        return x
