"""Tiny twin FLUX models (diffusers + comfy) with identical weights.

The diffusers model is the ground truth (the vendored ASASR forward runs on
it); the comfy model receives the same weights via the inverted
``flux_to_diffusers`` key map. All tensors fp32 on CPU for determinism.
"""
import torch

import comfy.utils
from comfy.ldm.flux.model import Flux as ComfyFlux
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

TINY = {
    "num_layers": 2,
    "num_single_layers": 2,
    "attention_head_dim": 16,
    "num_attention_heads": 4,
    "in_channels": 16,
    "joint_attention_dim": 24,
    "pooled_projection_dim": 32,
    "axes_dims_rope": (4, 6, 6),
    "guidance_embeds": True,
    "patch_size": 1,
}

COMFY_PARAMS = {
    "in_channels": 16,
    "out_channels": 16,
    "vec_in_dim": 32,
    "context_in_dim": 24,
    "hidden_size": 64,
    "mlp_ratio": 4.0,
    "num_heads": 4,
    "depth": 2,
    "depth_single_blocks": 2,
    "axes_dim": [4, 6, 6],
    "theta": 10000,
    "patch_size": 1,
    "qkv_bias": True,
    "guidance_embed": True,
    "txt_ids_dims": [],
}

UNET_CONFIG = {"depth": 2, "depth_single_blocks": 2, "hidden_size": 64}


def _swap_scale_shift(w):
    """Swaps the two halves of a fused (shift, scale) modulation tensor.

    Args:
        w: Tensor whose leading dimension holds ``[shift, scale]``.

    Returns:
        Tensor with the halves reordered to ``[scale, shift]``.
    """
    shift, scale = w.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)


def convert_diffusers_to_comfy(diffusers_sd, unet_config):
    """Builds a comfy Flux state dict from a diffusers Flux state dict.

    Args:
        diffusers_sd: State dict of a ``FluxTransformer2DModel``.
        unet_config: Minimal comfy model config (``depth``,
            ``depth_single_blocks``, ``hidden_size``) used to build the key map.

    Returns:
        dict: State dict keyed for ``comfy.ldm.flux.model.Flux``.
    """
    key_map = comfy.utils.flux_to_diffusers(unet_config, output_prefix="")
    comfy_sd = {}
    fused = {}
    for dkey, target in key_map.items():
        if dkey not in diffusers_sd:
            continue
        w = diffusers_sd[dkey]
        if isinstance(target, tuple):
            if len(target) == 3 and target[1] is None:
                comfy_sd[target[0]] = target[2](w)  # e.g. swap_scale_shift
            else:
                comfy_key, (_dim, offset, _size) = target[0], target[1]
                fused.setdefault(comfy_key, {})[offset] = w
        else:
            comfy_sd[target] = w
    for comfy_key, parts in fused.items():
        pieces = [parts[off] for off in sorted(parts)]
        comfy_sd[comfy_key] = torch.cat(pieces, dim=0)
    return comfy_sd


def build_tiny_pair(seed=0):
    """Builds a diffusers/comfy Flux pair that holds identical weights.

    Args:
        seed: Seed for the diffusers weight initialisation.

    Returns:
        tuple: ``(diffusers_model, comfy_model, cfg)`` where ``cfg`` is a copy
        of the comfy unet config used for the key map.
    """
    torch.manual_seed(seed)
    dm = FluxTransformer2DModel(**TINY).float().eval()
    cm = ComfyFlux(dtype=torch.float32, device="cpu",
                   operations=torch.nn, **COMFY_PARAMS).eval()
    comfy_sd = convert_diffusers_to_comfy(dm.state_dict(), UNET_CONFIG)
    missing, unexpected = cm.load_state_dict(comfy_sd, strict=False)
    assert not unexpected, f"unexpected comfy keys: {unexpected}"
    real_missing = [m for m in missing if not m.endswith("num_batches_tracked")]
    assert not real_missing, f"missing comfy keys: {real_missing}"
    return dm, cm, dict(UNET_CONFIG)


def make_inputs(seed=0, batch=1, txt_len=8, img_hw=(4, 4), cond_hw=(4, 4)):
    """Builds matched random inputs for both models (packed-token convention).

    Args:
        seed: Seed for the input tensors.
        batch: Batch size.
        txt_len: Number of text tokens.
        img_hw: ``(height, width)`` of the packed image token grid.
        cond_hw: ``(height, width)`` of the packed condition token grid.

    Returns:
        dict: Tensors shared by both call conventions.
    """
    g = torch.Generator().manual_seed(seed)

    def grid_ids(h, w):
        ids = torch.zeros(h, w, 3)
        ids[:, :, 1] = torch.arange(h).unsqueeze(1)
        ids[:, :, 2] = torch.arange(w).unsqueeze(0)
        return ids.reshape(1, h * w, 3).repeat(batch, 1, 1)

    h, w = img_hw
    ch, cw = cond_hw
    return {
        "img": torch.randn(batch, h * w, 16, generator=g),
        "img_ids": grid_ids(h, w),
        "cond": torch.randn(batch, ch * cw, 16, generator=g),
        "cond_ids": grid_ids(ch, cw),
        "txt": torch.randn(batch, txt_len, 24, generator=g),
        "txt_ids": torch.zeros(batch, txt_len, 3),
        "y": torch.randn(batch, 32, generator=g),
        "guidance": torch.full((batch,), 3.5),
    }
