"""
Post-processing module - Image manipulation operations.

This module provides:
- Operations registry: Modular post-processing functions
- Compositor: Alpha channel manipulation and compositing
- Filters: Sigmoid contrast, blur, power adjustments

Available operations:
- comp_on_white: Composite image over white background
- comp_on_black: Composite image over black background
- gamma_alpha: Apply inverse gamma to alpha then composite
- alpha_contrast: Sigmoid contrast + blur + power on alpha
- multiply_alpha: Multiply RGB by alpha (premultiply)
- divide_alpha: Divide RGB by alpha (unpremultiply)

To add new operations, edit vdg/postprocess/operations.py and use
the @register_operation decorator.
"""

from vdg.postprocess.operations import (
    register_operation,
    get_operations,
    get_operation,
    apply_operation,
    # Built-in operations
    comp_on_white,
    comp_on_black,
    refine_alpha,
    divide_alpha,
    unpremult_on_white,
)

__all__ = [
    "register_operation",
    "get_operations",
    "get_operation",
    "apply_operation",
    "comp_on_white",
    "comp_on_black",
    "refine_alpha",
    "divide_alpha",
    "unpremult_on_white",
]
