"""Tile split/merge roundtrip properties."""
import numpy as np
import pytest

from asasr.tiling import SCALE, merge_tiles, pad_to_grid, tile_boxes


def _nearest_upscale(img: np.ndarray) -> np.ndarray:
    return img.repeat(SCALE, axis=0).repeat(SCALE, axis=1)


@pytest.mark.parametrize("h,w", [(128, 128), (200, 300), (50, 40), (128, 200)])
@pytest.mark.parametrize("overlap", [0, 16])
def test_roundtrip_exact(h, w, overlap):
    rng = np.random.default_rng(0)
    img = rng.random((h, w, 3)).astype(np.float32)
    tile = 128

    padded, (oh, ow) = pad_to_grid(img, tile, overlap)
    boxes = tile_boxes(padded.shape[0], padded.shape[1], tile, overlap)

    hr_tiles = [_nearest_upscale(padded[y0:y1, x0:x1]) for (y0, x0, y1, x1) in boxes]
    merged = merge_tiles(hr_tiles, boxes, padded.shape[0], padded.shape[1], overlap)
    merged = merged[: oh * SCALE, : ow * SCALE]

    expected = _nearest_upscale(img)
    np.testing.assert_allclose(merged, expected, atol=1e-5)


def test_boxes_cover_and_are_exact_tiles():
    tile, overlap = 128, 16
    padded, _ = pad_to_grid(np.zeros((300, 500, 3), np.float32), tile, overlap)
    boxes = tile_boxes(padded.shape[0], padded.shape[1], tile, overlap)
    cover = np.zeros(padded.shape[:2], bool)
    for y0, x0, y1, x1 in boxes:
        assert y1 - y0 == tile and x1 - x0 == tile
        cover[y0:y1, x0:x1] = True
    assert cover.all()


def test_small_image_pads_to_single_tile():
    padded, (oh, ow) = pad_to_grid(np.zeros((50, 40, 3), np.float32), 128, 16)
    assert padded.shape[:2] == (128, 128)
    assert (oh, ow) == (50, 40)
