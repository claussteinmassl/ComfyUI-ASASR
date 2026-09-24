"""Color fix wrappers operating on ``[H, W, C]`` float tensors in 0..1.

Wraps the vendored StableSR color-fix routines (``asasr.color_fix``) which
work on ``[B, C, H, W]`` tensors.
"""
import torch

from .color_fix import adaptive_instance_normalization, wavelet_reconstruction

COLOR_FIX_MODES = ("adain", "wavelet", "none")


def apply_color_fix(result: torch.Tensor, reference: torch.Tensor, mode: str) -> torch.Tensor:
    """Transfers the reference image's color statistics onto the result.

    Args:
        result: Super-resolved image, ``[H, W, C]`` float in 0..1.
        reference: Color reference (upscaled input), same shape as ``result``.
        mode: One of ``"adain"``, ``"wavelet"``, ``"none"``.

    Returns:
        Color-corrected image, ``[H, W, C]`` float clamped to 0..1.
    """
    if mode == "none":
        return result
    if mode not in COLOR_FIX_MODES:
        raise ValueError(f"Unknown color fix mode: {mode!r}")
    target = result.permute(2, 0, 1).unsqueeze(0).float()
    source = reference.permute(2, 0, 1).unsqueeze(0).float()
    if mode == "adain":
        fixed = adaptive_instance_normalization(target, source)
    else:
        fixed = wavelet_reconstruction(target, source)
    return fixed.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)
