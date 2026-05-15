"""GATr-lite: a lightweight Geometric-Algebra Transformer in Cl(1,3)."""

from .algebra import (
    BASIS,
    BASIS_GRADES,
    CAYLEY_DENSE,
    CAYLEY_SIGNS,
    CAYLEY_TARGETS,
    GRADE_INDICES,
    METRIC,
    REVERSE_SIGNS,
    apply_rotor,
    clifford_reverse,
    geometric_product,
    inner_product,
    lorentz_matrix_from_rotor,
    mv_exp,
    random_rotor,
)

__all__ = [
    "BASIS",
    "BASIS_GRADES",
    "CAYLEY_DENSE",
    "CAYLEY_SIGNS",
    "CAYLEY_TARGETS",
    "GRADE_INDICES",
    "METRIC",
    "REVERSE_SIGNS",
    "geometric_product",
    "clifford_reverse",
    "inner_product",
    "apply_rotor",
    "lorentz_matrix_from_rotor",
    "mv_exp",
    "random_rotor",
]


def _load_torch_components():
    """Lazy import of torch-dependent classes."""
    from .layers import (  # noqa: F401
        EquivariantLinear,
        GatedActivation,
        GATrBlock,
        GeometricProduct,
        ScalarAttention,
    )
    from .model import GATrLite, GATrLiteConfig  # noqa: F401
    return {
        "EquivariantLinear": EquivariantLinear,
        "GeometricProduct": GeometricProduct,
        "ScalarAttention": ScalarAttention,
        "GatedActivation": GatedActivation,
        "GATrBlock": GATrBlock,
        "GATrLite": GATrLite,
        "GATrLiteConfig": GATrLiteConfig,
    }


def __getattr__(name):
    if name in {
        "EquivariantLinear",
        "GeometricProduct",
        "ScalarAttention",
        "GatedActivation",
        "GATrBlock",
        "GATrLite",
        "GATrLiteConfig",
    }:
        comps = _load_torch_components()
        return comps[name]
    raise AttributeError(name)
