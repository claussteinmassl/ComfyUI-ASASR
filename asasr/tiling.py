"""Tile-based processing helpers for 4x super-resolution.

Images are split on the low-resolution side into overlapping ``tile x tile``
squares, upscaled independently, and blended back together at ``SCALE`` times
the resolution using feathered weight masks normalized over all contributors.
"""
import math

import numpy as np

SCALE = 4


def pad_to_grid(img: np.ndarray, tile: int, overlap: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Pads an ``[H, W, C]`` image (edge mode) so tiles cover it exactly.

    Args:
        img: Input image array.
        tile: Tile edge length in pixels.
        overlap: Overlap between neighboring tiles in pixels.

    Returns:
        The padded image and the original ``(height, width)``.
    """
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile size")

    def target(n: int) -> int:
        if n <= tile:
            return tile
        return tile + math.ceil((n - tile) / stride) * stride

    h, w = img.shape[:2]
    pad_h, pad_w = target(h) - h, target(w) - w
    padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return padded, (h, w)


def tile_boxes(h: int, w: int, tile: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Returns ``(y0, x0, y1, x1)`` boxes tiling a padded ``h x w`` image."""
    stride = tile - overlap
    ys = list(range(0, h - tile + 1, stride))
    xs = list(range(0, w - tile + 1, stride))
    return [(y, x, y + tile, x + tile) for y in ys for x in xs]


def _feather_mask(tile_hr: int, overlap_hr: int) -> np.ndarray:
    """Builds a 2D weight mask ramping from ~0 to 1 over the overlap band."""
    ramp = np.ones(tile_hr, dtype=np.float32)
    if overlap_hr > 0:
        edge = np.linspace(0.0, 1.0, overlap_hr + 2, dtype=np.float32)[1:-1]
        ramp[:overlap_hr] = edge
        ramp[-overlap_hr:] = edge[::-1]
    return np.outer(ramp, ramp)


def merge_tiles(
    tiles: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
    h: int,
    w: int,
    overlap: int,
) -> np.ndarray:
    """Blends upscaled tiles into a full image with feathered weights.

    Args:
        tiles: Upscaled tiles, each ``[tile*SCALE, tile*SCALE, C]``.
        boxes: LR-side boxes matching ``tiles``.
        h: Padded LR height.
        w: Padded LR width.
        overlap: LR-side overlap used when tiling.

    Returns:
        Merged ``[h*SCALE, w*SCALE, C]`` float32 image.
    """
    channels = tiles[0].shape[2]
    canvas = np.zeros((h * SCALE, w * SCALE, channels), dtype=np.float32)
    weights = np.zeros((h * SCALE, w * SCALE, 1), dtype=np.float32)
    mask = _feather_mask(tiles[0].shape[0], overlap * SCALE)[..., None]
    for tile_img, (y0, x0, y1, x1) in zip(tiles, boxes):
        sy, sx = y0 * SCALE, x0 * SCALE
        ey, ex = y1 * SCALE, x1 * SCALE
        canvas[sy:ey, sx:ex] += tile_img.astype(np.float32) * mask
        weights[sy:ey, sx:ex] += mask
    return canvas / np.maximum(weights, 1e-8)
