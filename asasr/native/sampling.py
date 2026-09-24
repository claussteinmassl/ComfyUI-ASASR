"""Upstream-exact sampling primitives for the native ASASR path.

The upstream pipeline uses FlowMatchEulerDiscreteScheduler with sigmas
``linspace(1, 1/steps)`` dynamically shifted by ``mu`` computed for the
512x512 sequence length (1024 tokens). FLUX.1-dev embeds guidance, so the
loop runs without CFG.
"""
import math

import torch

BASE_SHIFT = 0.5
MAX_SHIFT = 1.15
BASE_SEQ = 256
MAX_SEQ = 4096


def make_sigmas(steps, image_seq_len=1024):
    """Computes the shifted flow-match sigma schedule.

    Mirrors diffusers ``FlowMatchEulerDiscreteScheduler.set_timesteps`` with
    dynamic shifting: a linear sigma schedule from 1 to ``1/steps`` is
    exponentially shifted by ``mu``, computed from ``image_seq_len`` via the
    same linear interpolation as ``calculate_shift`` in the FLUX pipeline.

    Args:
        steps: Number of denoising steps.
        image_seq_len: Packed token sequence length used to compute ``mu``.

    Returns:
        A ``[steps + 1]`` float32 tensor with a terminal 0 appended.
    """
    sigmas = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64)
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ - BASE_SEQ)
    mu = m * image_seq_len + (BASE_SHIFT - m * BASE_SEQ)
    sigmas = math.exp(mu) / (math.exp(mu) + (1.0 / sigmas - 1.0))
    return torch.cat([sigmas.float(), torch.zeros(1)])


def pack_latent(x):
    """Packs spatial latents into flattened patch tokens.

    Numerically identical to diffusers ``FluxPipeline._pack_latents``: each
    2x2 spatial block is flattened into a single token along the channel
    axis.

    Args:
        x: Latent tensor of shape ``[B, C, H, W]``, with ``H`` and ``W`` even.

    Returns:
        Token tensor of shape ``[B, (H/2)*(W/2), C*4]``.
    """
    b, c, h, w = x.shape
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)


def unpack_latent(tokens, h, w):
    """Inverse of :func:`pack_latent`.

    Numerically identical to diffusers ``FluxPipeline._unpack_latents``.

    Args:
        tokens: Token tensor of shape ``[B, (H/2)*(W/2), C*4]``.
        h: Target spatial height (even).
        w: Target spatial width (even).

    Returns:
        Latent tensor of shape ``[B, C, h, w]``.
    """
    b, _, packed = tokens.shape
    c = packed // 4
    x = tokens.view(b, h // 2, w // 2, c, 2, 2)
    return x.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


def latent_ids(h_len, w_len, batch=1, device="cpu"):
    """Builds position ids for a packed token grid.

    Matches the ``grid_ids`` convention used by the test fixtures (and
    diffusers ``FluxPipeline._prepare_latent_image_ids``): axis 0 is always
    0, axis 1 holds the row index, axis 2 holds the column index, in
    row-major token order.

    Args:
        h_len: Number of rows in the token grid.
        w_len: Number of columns in the token grid.
        batch: Batch size to repeat the grid over.
        device: Device to allocate the ids on.

    Returns:
        Position id tensor of shape ``[batch, h_len * w_len, 3]``.
    """
    ids = torch.zeros(h_len, w_len, 3, device=device)
    ids[:, :, 1] = torch.arange(h_len, device=device).unsqueeze(1)
    ids[:, :, 2] = torch.arange(w_len, device=device).unsqueeze(0)
    return ids.reshape(1, h_len * w_len, 3).repeat(batch, 1, 1)


def euler_denoise(model_fn, x, sigmas, step_callback=None):
    """Flow-match Euler loop: ``x += (sigma_next - sigma) * model_fn(x, sigma)``.

    Args:
        model_fn: Callable ``(x, sigma_scalar_tensor) -> velocity``.
        x: Initial noise, any shape.
        sigmas: ``[steps + 1]`` schedule from :func:`make_sigmas`.
        step_callback: Optional ``fn(step_index, total_steps)`` (interrupt hook).

    Returns:
        The denoised sample.
    """
    total = sigmas.shape[0] - 1
    for i in range(total):
        if step_callback is not None:
            step_callback(i, total)
        v = model_fn(x, sigmas[i])
        x = x + (sigmas[i + 1] - sigmas[i]) * v
    return x
