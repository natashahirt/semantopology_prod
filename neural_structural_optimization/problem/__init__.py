"""Boundary conditions and named structural problems."""

from . import problems
from .problems import (
    PROBLEMS_BY_CATEGORY,
    PROBLEMS_BY_NAME,
    Problem,
    StructuralParams,
    apply_discretization_params,
    resolve_analysis_filter_width,
    resolve_discretization_value,
)

__all__ = [
    'problems',
    'Problem',
    'StructuralParams',
    'apply_discretization_params',
    'resolve_analysis_filter_width',
    'resolve_discretization_value',
    'PROBLEMS_BY_NAME',
    'PROBLEMS_BY_CATEGORY',
]
