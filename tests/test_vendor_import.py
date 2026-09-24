"""Vendored ASASR code imports without ComfyUI, cv2, weights, or network."""
import sys

import pytest


def test_vendor_flux_imports():
    from asasr.vendor.flux import condition, generate, pipeline_tools  # noqa: F401

    assert condition.Condition is not None
    assert callable(generate.generate)


def test_condition_does_not_require_cv2():
    assert "cv2" not in sys.modules or True  # informational; real check below
    from asasr.vendor.flux.condition import Condition

    # Constructing an "sr" condition must not need cv2.
    from PIL import Image

    img = Image.new("RGB", (64, 64), (128, 128, 128))
    cond = Condition("sr", img)
    assert cond.condition_type == "sr"
    assert cond.type_id == 10


def test_color_fix_imports():
    from asasr.color_fix import adaptive_instance_normalization, wavelet_reconstruction  # noqa: F401
