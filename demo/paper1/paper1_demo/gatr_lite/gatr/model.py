"""GATrLite: equivariant transformer for gg -> 6-parton classification.

Inputs
------
* ``p4``  shape ``(B, 6, 4)`` -- contravariant 4-momenta (E, px, py, pz)
* ``pdg`` shape ``(B, 6)``    -- PDG ids of the final-state partons

Output
------
* logit of shape ``(B, 1)`` -- a single binary discriminator value per event.

Equivariance
------------
The 4-momentum embedding places ``(E, px, py, pz)`` directly into the four
grade-1 basis components ``(e_0, e_1, e_2, e_3)``. Under any active Lorentz
transform L applied as ``p -> L p``, the embedded multivector transforms by
the corresponding rotor sandwich. Scalar features (flavor one-hot, electric
charge, b-tag, pairing-cache invariants) are placed in the grade-0 channels
and are Lorentz-scalars by construction. The pairing-cache trivector
``T^(a)`` is placed in the grade-3 components and the Hodge dual ``*T^(a)``
in grade-1 components -- both transform covariantly. The body of the
network (``GATrBlock`` x N) is equivariant, and the readout takes only the
scalar part of the final tokens, so the logit is Lorentz-invariant.

Pairing-aware feature cache
---------------------------
When ``cfg.use_pairing_cache=True`` (default), two extra tokens are added to
the standard 6 partons. They carry the pairing-cache features computed in
``gatr.pairing``:

* token 7 (pairing 1, hadronic_b = b)
* token 8 (pairing 2, hadronic_b = b_bar)

For these tokens we use sentinel PDG codes +100 / -100 and dedicated flavor
slots in the scalar one-hot, and the eight pairing scalars
``(s_had^(a), s_lep^(a), sigma_p_had, sigma_m_had, sigma_p_lep, sigma_m_lep,
mass_W_had, mass_W_lep)`` go into a separate block of grade-0 channels that
are zero for regular partons.

When ``cfg.use_pairing_cache=False`` the model reverts to the original
6-token / 8-scalar architecture (one-hot + charge + b-tag) for backward
compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn

from . import algebra as A
from .layers import BlockConfig, EquivariantLinear, GATrBlock, JoinBlock
from .pairing import (
    IDX_MU, IDX_NU, IDX_U, IDX_DBAR, IDX_B, IDX_BBAR,
    compute_pairing_features,
)


# ---------------------------------------------------------------------------
# PDG -> flavor index lookup
# ---------------------------------------------------------------------------

# Six final-state species expected for gg -> mu^- nu_mu_bar u dbar b bbar
# (and the conjugate t-channel; in our binary task the partner-flavor map is
# what matters, not which one is "signal"). Sentinel codes +-100 mark the
# two pairing tokens.
PDG_TO_INDEX: Dict[int, int] = {
    13: 0,   # mu-
    -14: 1,  # nu_mu_bar
    2: 2,    # u
    -1: 3,   # dbar
    5: 4,    # b
    -5: 5,   # bbar
    100: 6,  # pairing-1 token
    -100: 7, # pairing-2 token
    # accept conjugate species too in case loaders flip charge
    -13: 0,
    14: 1,
    -2: 2,
    1: 3,
}

# Slot name preserved for callers that still import it.
PDG_TO_FLAVOR_INDEX: Dict[int, int] = PDG_TO_INDEX

# Electric charges per *flavor index* in PDG_TO_INDEX (lepton convention used
# for the network features only -- it's a scalar feature). Pairing tokens
# carry no electric charge.
FLAVOR_CHARGE: Tuple[float, ...] = (
    -1.0, 0.0, 2.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0,
)
FLAVOR_BTAG: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)

# Number of flavor slots (incl. two pairing-token slots). The first six are
# the physical partons; slots 6 and 7 are pairing-1 and pairing-2 tokens.
N_FLAVORS = 8

# 8 one-hot + charge + b-tag = 10 base scalar features.
N_BASE_SCALARS = N_FLAVORS + 2

# Pairing scalars filled only on the two pairing tokens (zero on physical
# partons): (s_had, s_lep, sigma_p_had, sigma_m_had, sigma_p_lep,
# sigma_m_lep, mass_W_had, mass_W_lep, coplanarity, pseudoscalar_4).
# The last two are the Group-A meet/join derived invariants.
N_PAIRING_SCALARS = 10

N_SCALAR_FEATURES = N_BASE_SCALARS + N_PAIRING_SCALARS  # 20

# Number of covariant (multivector) channels per pairing token. Channel 0
# carries (*T_had, meet, T_had) — grade 1 + 2 + 3, non-overlapping. Channel
# 1 carries (*T_lep, T_lep) — grade 1 + 3.
N_PAIRING_COV_CHANNELS = 2

# Sentinel PDG codes for the two pairing tokens (chosen to lie outside the
# Standard Model PDG numbering scheme used by the LHE files).
PDG_PAIRING_1 = 100
PDG_PAIRING_2 = -100


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GATrLiteConfig:
    """Static config for the GATr-lite classifier.

    Defaults are sized to land at ~150-200k parameters (3 blocks, 32 channels).
    """
    n_blocks: int = 3
    channels: int = 32
    gp_mid_channels: int = 16
    attn_channels: int = 16
    head_hidden: int = 96
    n_particles: int = 6
    # When True, two extra "pairing tokens" are appended carrying the
    # pairing-aware feature cache. When False, the model falls back to the
    # original 6-token / 8-scalar layout used before the pairing-cache work.
    use_pairing_cache: bool = True

    # ---------------------------------------------------------------------
    # arch_v2 -- guarded architectural improvements (off by default).
    # ---------------------------------------------------------------------
    # Master switch: when False, the model is byte-identical to v1 and the
    # other arch_v2 fields are ignored. When True the additional knobs below
    # become active.
    arch_v2: bool = False
    # Multi-head scalar attention (head_dim = channels / n_heads). Requires
    # channels % n_heads == 0. With n_heads=1 the layer reduces to v1
    # ScalarAttention.
    n_heads: int = 1
    # Reserved hook for future grade-3 mixing experiments inside
    # GeometricProduct. Currently a no-op besides initialising the GP combine
    # tensor with a scalar-channel boost so grade-3 components survive (see
    # GATrLite.__init__).
    gp_grade3_mixing: bool = False
    # Dropout in the head MLP between Linear/GELU layers and per-channel
    # scalar dropout in each block (grade-0 only -> preserves equivariance).
    dropout: float = 0.0
    # Random masking of the 8 pairing-token scalar input channels at training
    # (only when use_pairing_cache=True). Acts as a feature-level
    # regulariser; equivariance is preserved because pairing scalars are
    # Lorentz invariants.
    input_dropout: float = 0.0
    # Group B: append one JoinBlock (antisymmetric-wedge residual) after the
    # standard stack. Lets the network build grade-(k+l) objects directly
    # — needed for CP-odd / 4-blade pseudoscalar invariants without relying
    # on deep gp-residual interactions through symmetric GeometricProduct.
    use_join_block: bool = True


# ---------------------------------------------------------------------------
# Scalar feature embeddings
# ---------------------------------------------------------------------------


class _PDGEmbedding(nn.Module):
    """Map PDG ids to per-particle scalar feature vectors.

    The lookup is fixed (no learnable parameters): one-hot flavor, electric
    charge, b-tag flag. Pairing-cache scalars are *not* filled here -- they
    are added by the model's ``embed`` method, which has access to the
    pairing-feature dict.
    """

    def __init__(self, n_flavors: int, n_scalar_features: int,
                 charges: Tuple[float, ...], btags: Tuple[float, ...]) -> None:
        super().__init__()
        feats = torch.zeros(n_flavors, n_scalar_features)
        for f in range(n_flavors):
            feats[f, f] = 1.0
            feats[f, n_flavors] = charges[f]
            feats[f, n_flavors + 1] = btags[f]
        self.register_buffer("flavor_features", feats)

    def forward(self, pdg: torch.Tensor) -> torch.Tensor:
        """pdg: (B, T) long. Return: (B, T, n_scalar_features)."""
        flat = pdg.detach().cpu().tolist()
        idx = torch.tensor(
            [[PDG_TO_INDEX.get(int(p), 0) for p in row] for row in flat],
            dtype=torch.long, device=pdg.device,
        )
        return self.flavor_features[idx]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class GATrLite(nn.Module):
    """Equivariant binary classifier for gg -> 6 partons (+ 2 pairing tokens)."""

    def __init__(self, cfg: GATrLiteConfig | None = None):
        super().__init__()
        self.cfg = cfg or GATrLiteConfig()

        if self.cfg.use_pairing_cache:
            # Full layout: 8 flavor slots, 20 scalar channels, 6 + 2 tokens.
            self._n_flavors = N_FLAVORS
            self._n_scalars = N_SCALAR_FEATURES
            charges = FLAVOR_CHARGE
            btags = FLAVOR_BTAG
            self._n_cov_channels = N_PAIRING_COV_CHANNELS  # 2
        else:
            # Legacy layout: 6 flavor slots, 8 scalar channels, 6 tokens.
            self._n_flavors = 6
            self._n_scalars = 6 + 2  # 6 one-hot + charge + b-tag
            charges = FLAVOR_CHARGE[:6]
            btags = FLAVOR_BTAG[:6]
            self._n_cov_channels = 1

        self.pdg_embed = _PDGEmbedding(
            self._n_flavors, self._n_scalars, charges, btags,
        )

        # Channel layout: N_cov_channels covariant channels (4-momentum on
        # physical tokens; meet/join trivectors on pairing tokens) plus
        # N_SCALAR_FEATURES grade-0 channels.
        self.c_in = self._n_cov_channels + self._n_scalars
        self.input_proj = EquivariantLinear(self.c_in, self.cfg.channels)

        # arch_v2 plumbing: multi-head attention and scalar dropout per block.
        if self.cfg.arch_v2:
            n_heads = self.cfg.n_heads
            scalar_dropout = self.cfg.dropout
        else:
            n_heads = 1
            scalar_dropout = 0.0

        block_cfg = BlockConfig(
            channels=self.cfg.channels,
            gp_mid_channels=self.cfg.gp_mid_channels,
            attn_channels=self.cfg.attn_channels,
            n_heads=n_heads,
            scalar_dropout=scalar_dropout,
        )
        self.blocks = nn.ModuleList([GATrBlock(block_cfg) for _ in range(self.cfg.n_blocks)])
        # Optional JoinBlock after the stack. Same BlockConfig as the
        # standard blocks so it inherits multi-head / dropout settings.
        self.join_block = JoinBlock(block_cfg) if self.cfg.use_join_block else None
        # JoinBlock-warmup factor (0..1). When < 1, the JoinBlock output is
        # blended with the pre-JoinBlock state x:  x ← x + α·(JoinBlock(x) − x).
        # α=0 disables JoinBlock entirely; α=1 applies it fully. Ramped from
        # the train loop over the first few epochs to let the standard stack
        # settle before adding antisymmetric-wedge contributions.
        self._join_alpha: float = 1.0

        # Head MLP. arch_v2 inserts dropout between Linear/GELU layers.
        if self.cfg.arch_v2 and self.cfg.dropout > 0.0:
            self.head = nn.Sequential(
                nn.Linear(self.cfg.channels, self.cfg.head_hidden),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.head_hidden, self.cfg.head_hidden),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.head_hidden, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(self.cfg.channels, self.cfg.head_hidden),
                nn.GELU(),
                nn.Linear(self.cfg.head_hidden, self.cfg.head_hidden),
                nn.GELU(),
                nn.Linear(self.cfg.head_hidden, 1),
            )

        # arch_v2: optional pairing-scalar input dropout module. Acts on the
        # 8 pairing-feature scalar channels of the two pairing tokens at
        # training time only (eval is deterministic).
        if (
            self.cfg.arch_v2
            and self.cfg.input_dropout > 0.0
            and self.cfg.use_pairing_cache
        ):
            self.input_dropout = nn.Dropout(self.cfg.input_dropout)
        else:
            self.input_dropout = None

        # arch_v2 + gp_grade3_mixing: bias-init GP combine kernels so grade-3
        # components from GP(grade-2 x grade-1) are not suppressed at init.
        # Adds a small identity-flavoured nudge to each block's GeometricProduct
        # combine tensor (named `combine` inside layers.GeometricProduct).
        if self.cfg.arch_v2 and self.cfg.gp_grade3_mixing:
            with torch.no_grad():
                for blk in self.blocks:
                    gp = getattr(blk, "gp", None) or getattr(blk, "geometric_product", None)
                    if gp is None:
                        continue
                    for pname, p in gp.named_parameters(recurse=False):
                        if p.dim() >= 2 and p.shape[-1] == p.shape[-2]:
                            d = p.shape[-1]
                            p.add_(0.05 * torch.eye(d, device=p.device).expand_as(p))

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_legacy(self, p4: torch.Tensor, pdg: torch.Tensor) -> torch.Tensor:
        """Original 6-token / 8-scalar embedding (pairing cache disabled)."""
        B, T, _ = p4.shape
        scalar_feats = self.pdg_embed(pdg)  # (B, T, n_scalars)
        out = torch.zeros(B, T, self.c_in, 16, dtype=p4.dtype, device=p4.device)
        # Channel 0: grade-1 4-momentum at indices (e_0, e_1, e_2, e_3).
        for mu in range(4):
            out[:, :, 0, A.GRADE1_INDICES[mu]] = p4[:, :, mu]
        # Channels 1..n_scalars: scalar (grade-0).
        c_cov = self._n_cov_channels  # = 1 in legacy
        out[:, :, c_cov:, 0] = scalar_feats
        return out

    def _embed_with_pairing(self, p4: torch.Tensor, pdg: torch.Tensor) -> torch.Tensor:
        """Embedding with two extra pairing-aware tokens appended.

        Builds a (B, 8, c_in, 16) multivector tensor:

        * tokens 0..5  : the six physical partons in canonical order
        * tokens 6, 7  : pairing-1 and pairing-2 tokens with grade-3
                         trivector ``T^(a)``, grade-1 vector ``*T^(a)``
                         (Hodge dual), and the eight pairing scalars in
                         dedicated grade-0 channels.
        """
        B, T_in, _ = p4.shape
        if T_in != 6:
            raise ValueError(
                f"Pairing cache requires 6 partons in canonical order; got T={T_in}"
            )

        device = p4.device
        dtype = p4.dtype

        # Per-token PDG vector for all 8 tokens (6 physical + 2 sentinels).
        pdg_full = torch.empty(B, 8, dtype=pdg.dtype, device=device)
        pdg_full[:, :6] = pdg
        pdg_full[:, 6] = PDG_PAIRING_1
        pdg_full[:, 7] = PDG_PAIRING_2

        scalar_feats = self.pdg_embed(pdg_full)  # (B, 8, n_scalars)
        pf = compute_pairing_features(p4, pdg)
        # Layout of the 10 pairing scalar channels (per-pairing token):
        #   [s_had^(a), s_lep^(a),
        #    sigma_p_had, sigma_m_had, sigma_p_lep, sigma_m_lep,
        #    mass_W_had, mass_W_lep,
        #    coplanarity^(a), pseudoscalar_4^(a)]
        # The middle 6 are shared between both pairing tokens (per-event
        # invariants); the others depend on the pairing index.
        sigma_block = torch.stack([
            pf["sigma_p_had"], pf["sigma_m_had"],
            pf["sigma_p_lep"], pf["sigma_m_lep"],
            pf["mass_W_had"], pf["mass_W_lep"],
        ], dim=-1)                                       # (B, 6)

        for a in range(2):
            row = torch.stack([
                pf["s_had"][:, a], pf["s_lep"][:, a],
            ], dim=-1)                                   # (B, 2)
            tail = torch.stack([
                pf["coplanarity"][:, a], pf["pseudoscalar_4"][:, a],
            ], dim=-1)                                   # (B, 2)
            row = torch.cat([row, sigma_block, tail], dim=-1)  # (B, 10)
            scalar_feats[:, 6 + a, N_BASE_SCALARS:] = row

        # Build the (B, 8, c_in, 16) tensor.
        out = torch.zeros(B, 8, self.c_in, 16, dtype=dtype, device=device)

        # Channel 0 covariant components:
        #   - tokens 0..5: physical 4-momentum p^mu (grade 1 only)
        #   - tokens 6, 7: *T_had^(a) (grade 1) + meet^(a) (grade 2) + T_had^(a) (grade 3)
        for mu in range(4):
            out[:, :6, 0, A.GRADE1_INDICES[mu]] = p4[:, :, mu]
        # *T_had on pairing tokens — grade-1 components.
        out[:, 6:8, 0, :] = out[:, 6:8, 0, :] + pf["T_dual_pair"]
        # meet on pairing tokens — grade-2 components.
        for i in A.GRADE_INDICES[2]:
            out[:, 6, 0, i] = out[:, 6, 0, i] + pf["meet_pair"][:, 0, i]
            out[:, 7, 0, i] = out[:, 7, 0, i] + pf["meet_pair"][:, 1, i]
        # T_had on pairing tokens — grade-3 components.
        for i in A.GRADE_INDICES[3]:
            out[:, 6, 0, i] = out[:, 6, 0, i] + pf["T_pair"][:, 0, i]
            out[:, 7, 0, i] = out[:, 7, 0, i] + pf["T_pair"][:, 1, i]

        # Channel 1 covariant components (only on pairing tokens; zero on
        # physical partons): *T_lep (grade 1) + T_lep (grade 3).
        if self._n_cov_channels >= 2:
            out[:, 6:8, 1, :] = out[:, 6:8, 1, :] + pf["T_lep_dual_pair"]
            for i in A.GRADE_INDICES[3]:
                out[:, 6, 1, i] = out[:, 6, 1, i] + pf["T_lep_pair"][:, 0, i]
                out[:, 7, 1, i] = out[:, 7, 1, i] + pf["T_lep_pair"][:, 1, i]

        # Channels c_cov..(c_cov + n_scalars - 1): grade-0 scalar features.
        c_cov = self._n_cov_channels
        out[:, :, c_cov:, 0] = scalar_feats

        return out

    def embed(self, p4: torch.Tensor, pdg: torch.Tensor) -> torch.Tensor:
        """Build the (B, T, C_in, 16) multivector input from p4 + pdg.

        T is 6 if ``use_pairing_cache=False`` and 8 otherwise.
        """
        if self.cfg.use_pairing_cache:
            return self._embed_with_pairing(p4, pdg)
        return self._embed_legacy(p4, pdg)

    def forward(self, p4: torch.Tensor, pdg: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, 1)."""
        x = self.embed(p4, pdg)              # (B, T, C_in, 16)
        x = self.input_proj(x)               # (B, T, C, 16)
        for block in self.blocks:
            x = block(x)
        if self.join_block is not None:
            jb_out = self.join_block(x)
            if self._join_alpha < 1.0:
                x = x + self._join_alpha * (jb_out - x)
            else:
                x = jb_out
        # Readout: scalar part, mean-pool tokens (permutation invariant), MLP
        scalar = x[..., 0]                   # (B, T, C)
        pooled = scalar.mean(dim=1)          # (B, C)
        logit = self.head(pooled)            # (B, 1)
        return logit

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_join_alpha(self, alpha: float) -> None:
        """Set the JoinBlock blend factor for warmup. 0 disables, 1 enables fully."""
        self._join_alpha = float(max(0.0, min(1.0, alpha)))


def count_parameters_breakdown(model: GATrLite) -> List[Tuple[str, int]]:
    """Return [(submodule_name, n_params), ...] for diagnostics."""
    out: List[Tuple[str, int]] = []
    for name, module in model.named_modules():
        if name == "" or any("." in name[len(name):] for _ in [0]):  # noqa
            pass
        n = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        if n:
            out.append((name or "<root>", n))
    return out
