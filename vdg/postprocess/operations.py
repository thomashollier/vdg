"""
Post-processing operations for VDG.

This module provides a registry of image processing operations that can be
applied to an image + alpha pair and produce a single output image.

To add a new operation:
1. Create a function with signature: func(image: np.ndarray, alpha: np.ndarray, **params) -> np.ndarray
2. Register it with the @register_operation decorator

All operations receive:
- image: RGB image as numpy array (uint8 or uint16, HxWx3)
- alpha: Alpha/mask image as numpy array (uint8 or uint16, HxW or HxWx1)
- **params: Additional parameters from the node

All operations must return:
- A single RGB image as numpy array (same dtype as input)
"""

import numpy as np
from typing import Callable, Dict, Any

# Registry of available operations
_OPERATIONS: Dict[str, Dict[str, Any]] = {}


def register_operation(name: str, description: str = ""):
    """Decorator to register a post-processing operation."""
    def decorator(func: Callable):
        _OPERATIONS[name] = {
            'func': func,
            'description': description,
        }
        return func
    return decorator


def get_operations() -> list:
    """Return list of available operation names."""
    return list(_OPERATIONS.keys())


def get_operation(name: str) -> Callable:
    """Get an operation function by name."""
    if name not in _OPERATIONS:
        raise ValueError(f"Unknown operation: {name}. Available: {get_operations()}")
    return _OPERATIONS[name]['func']


def apply_operation(name: str, image: np.ndarray, alpha: np.ndarray, **params) -> np.ndarray:
    """Apply a named operation to image and alpha."""
    func = get_operation(name)
    return func(image, alpha, **params)


# =============================================================================
# Built-in Operations
# =============================================================================

@register_operation("comp_on_white", "Composite image over white background using alpha")
def comp_on_white(image: np.ndarray, alpha: np.ndarray, **params) -> np.ndarray:
    """
    Composite premultiplied RGB over white background.

    Takes the red channel from alpha as the compositing alpha.
    """
    # Normalize alpha to 0-1 range
    if alpha.dtype == np.uint16:
        alpha_norm = alpha.astype(np.float32) / 65535.0
    else:
        alpha_norm = alpha.astype(np.float32) / 255.0

    # Get single channel alpha (red channel if multi-channel)
    if alpha_norm.ndim == 3:
        alpha_1ch = alpha_norm[:, :, 0]
    else:
        alpha_1ch = alpha_norm

    # Normalize image to 0-1
    if image.dtype == np.uint16:
        img_norm = image.astype(np.float32) / 65535.0
        max_val = 65535
        out_dtype = np.uint16
    else:
        img_norm = image.astype(np.float32) / 255.0
        max_val = 255
        out_dtype = np.uint8

    # Create RGBA and composite over white
    alpha_3ch = np.dstack([alpha_1ch, alpha_1ch, alpha_1ch])
    white = np.ones_like(img_norm)

    # Over operation: foreground + background * (1 - alpha)
    result = img_norm + white * (1 - alpha_3ch)

    # Clamp and convert back
    result = np.clip(result * max_val, 0, max_val).astype(out_dtype)
    return result


@register_operation("comp_on_black", "Composite image over black background using alpha")
def comp_on_black(image: np.ndarray, alpha: np.ndarray, **params) -> np.ndarray:
    """
    Composite premultiplied RGB over black background.

    Essentially just clamps the image values.
    """
    if image.dtype == np.uint16:
        max_val = 65535
    else:
        max_val = 255

    return np.clip(image, 0, max_val)


@register_operation("refine_alpha", "Gamma + contrast + blur + power on alpha, composite on white")
def refine_alpha(image: np.ndarray, alpha: np.ndarray,
                 gamma: float = 2.2, contrast: float = 60.0, threshold: float = 0.0015,
                 blur_size: float = 5.0, power: float = 8.0, **params) -> np.ndarray:
    """
    Refine alpha with gamma, sigmoid contrast, blur, and power curve, then composite.

    Processing order:
    1. Inverse gamma (1/gamma) - brightens alpha
    2. Sigmoid contrast - sharpens edges around threshold
    3. Gaussian blur - smooths result
    4. Power curve - adjusts falloff

    Set gamma=1 to skip gamma. Set contrast=0 to skip contrast. Set blur_size=0 to skip blur.
    Set power=1 to skip power curve.
    """
    from scipy.ndimage import gaussian_filter

    # Normalize alpha
    if alpha.dtype == np.uint16:
        alpha_norm = alpha.astype(np.float32) / 65535.0
    else:
        alpha_norm = alpha.astype(np.float32) / 255.0

    # Get single channel
    if alpha_norm.ndim == 3:
        alpha_1ch = alpha_norm[:, :, 0].copy()
    else:
        alpha_1ch = alpha_norm.copy()

    # 1. Apply inverse gamma (skip if gamma == 1)
    if gamma != 1.0:
        alpha_1ch = np.power(np.clip(alpha_1ch, 0, 1), 1.0 / gamma)
        alpha_1ch = np.clip(alpha_1ch, 0, 1)

    # 2. Sigmoid contrast (skip if contrast == 0)
    if contrast > 0:
        exponent = -contrast * (alpha_1ch - threshold)
        exponent = np.clip(exponent, -88, 88)  # Prevent overflow
        alpha_1ch = 1.0 / (1.0 + np.exp(exponent))
        alpha_1ch = np.clip(alpha_1ch, 0, 1)

    # 3. Gaussian blur (skip if blur_size == 0)
    if blur_size > 0:
        alpha_1ch = gaussian_filter(alpha_1ch, sigma=blur_size / 2.0)
        alpha_1ch = np.clip(alpha_1ch, 0, 1)

    # 4. Power curve (skip if power == 1)
    if power != 1.0:
        alpha_1ch = np.power(alpha_1ch, power)
        alpha_1ch = np.clip(alpha_1ch, 0, 1)

    # Normalize image
    if image.dtype == np.uint16:
        img_norm = image.astype(np.float32) / 65535.0
        max_val = 65535
        out_dtype = np.uint16
    else:
        img_norm = image.astype(np.float32) / 255.0
        max_val = 255
        out_dtype = np.uint8

    # Unpremultiply with original alpha
    original_alpha = alpha_norm[:, :, 0] if alpha_norm.ndim == 3 else alpha_norm
    original_alpha = np.clip(original_alpha, 0.001, 1.0)
    rgb_straight = img_norm / np.dstack([original_alpha] * 3)
    rgb_straight = np.clip(rgb_straight, 0, 1)

    # Premultiply with processed alpha
    alpha_3ch = np.dstack([alpha_1ch, alpha_1ch, alpha_1ch])
    rgb_premult = rgb_straight * alpha_3ch

    # Composite over white
    white = np.ones_like(rgb_premult)
    result = rgb_premult + white * (1 - alpha_3ch)

    result = np.clip(result * max_val, 0, max_val).astype(out_dtype)
    return result


@register_operation("divide_alpha", "Divide RGB by alpha (unpremultiply, on black)")
def divide_alpha(image: np.ndarray, alpha: np.ndarray, **params) -> np.ndarray:
    """
    Divide RGB by alpha to unpremultiply the image.
    Result is on black background.
    """
    # Normalize alpha
    if alpha.dtype == np.uint16:
        alpha_norm = alpha.astype(np.float32) / 65535.0
    else:
        alpha_norm = alpha.astype(np.float32) / 255.0

    # Get single channel
    if alpha_norm.ndim == 3:
        alpha_1ch = alpha_norm[:, :, 0]
    else:
        alpha_1ch = alpha_norm

    # Clamp to avoid division by zero
    alpha_1ch = np.clip(alpha_1ch, 0.001, 1.0)

    # Normalize image
    if image.dtype == np.uint16:
        img_norm = image.astype(np.float32) / 65535.0
        max_val = 65535
        out_dtype = np.uint16
    else:
        img_norm = image.astype(np.float32) / 255.0
        max_val = 255
        out_dtype = np.uint8

    alpha_3ch = np.dstack([alpha_1ch, alpha_1ch, alpha_1ch])
    result = img_norm / alpha_3ch

    result = np.clip(result * max_val, 0, max_val).astype(out_dtype)
    return result


@register_operation("unpremult_on_white", "Unpremultiply then add inverse alpha for white background")
def unpremult_on_white(image: np.ndarray, alpha: np.ndarray, **params) -> np.ndarray:
    """
    Divide RGB by alpha, divide alpha by itself, then: RGB + (1 - divided_alpha)

    This gives unpremultiplied colors where there's content, white where there isn't.
    """
    # Normalize alpha
    if alpha.dtype == np.uint16:
        alpha_norm = alpha.astype(np.float32) / 65535.0
    else:
        alpha_norm = alpha.astype(np.float32) / 255.0

    # Get single channel
    if alpha_norm.ndim == 3:
        alpha_1ch = alpha_norm[:, :, 0]
    else:
        alpha_1ch = alpha_norm

    # Normalize image
    if image.dtype == np.uint16:
        img_norm = image.astype(np.float32) / 65535.0
        max_val = 65535
        out_dtype = np.uint16
    else:
        img_norm = image.astype(np.float32) / 255.0
        max_val = 255
        out_dtype = np.uint8

    # Divide RGB by alpha (with safe division)
    alpha_safe = np.clip(alpha_1ch, 0.001, 1.0)
    alpha_3ch_safe = np.dstack([alpha_safe, alpha_safe, alpha_safe])
    rgb_divided = img_norm / alpha_3ch_safe

    # Divide alpha by itself: 1 where alpha > 0, 0 where alpha = 0
    # Use same threshold as RGB division for consistency
    divided_alpha = np.where(alpha_1ch > 0.001, 1.0, 0.0)

    # RGB + (1 - divided_alpha)
    divided_alpha_3ch = np.dstack([divided_alpha, divided_alpha, divided_alpha])
    result = rgb_divided + (1.0 - divided_alpha_3ch)

    result = np.clip(result * max_val, 0, max_val).astype(out_dtype)
    return result
