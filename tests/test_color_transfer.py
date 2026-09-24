"""Color fix wrapper on HWC float tensors."""
import torch

from asasr.color_transfer import apply_color_fix


def _rand_img(h=32, w=32):
    torch.manual_seed(0)
    return torch.rand(h, w, 3)


def test_none_mode_is_identity():
    img = _rand_img()
    out = apply_color_fix(img, _rand_img(), "none")
    assert torch.equal(out, img)


def test_adain_matches_reference_statistics():
    result = _rand_img() * 0.5 + 0.25
    reference = _rand_img()
    out = apply_color_fix(result, reference, "adain")
    assert out.shape == result.shape
    for c in range(3):
        assert torch.isclose(out[..., c].mean(), reference[..., c].mean(), atol=0.02)


def test_wavelet_output_shape_and_range():
    result = _rand_img(64, 48)
    reference = _rand_img(64, 48)
    out = apply_color_fix(result, reference, "wavelet")
    assert out.shape == result.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
