"""约束检查模块。"""

from .spacing import (
    SPACING_TOLERANCE,
    SpacingConstraint,
    SpacingFeasibilityError,
    SpacingViolation,
    check_directional_spacing,
    check_min_spacing,
    compute_min_spacing_from_diameters,
    compute_pairwise_distances,
    enforce_min_spacing,
)

__all__ = [
    "SpacingConstraint",
    "SpacingViolation",
    "SpacingFeasibilityError",
    "check_min_spacing",
    "check_directional_spacing",
    "compute_min_spacing_from_diameters",
    "compute_pairwise_distances",
    "enforce_min_spacing",
    "SPACING_TOLERANCE",
]
