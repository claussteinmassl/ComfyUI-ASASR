"""ASASR LoRA delta store for the native ComfyUI forward.

The ASASR LoRAs must not be merged into model weights: upstream disables
them on the main (txt+img) branch and enables them only on the condition
branch. This module loads the checkpoints (diffusers naming,
``transformer.<path>.lora_A/.lora_B.weight``, no alpha keys) and maps them
onto comfy module paths using ``comfy.utils.flux_to_diffusers``, keeping
fused-target output slices (qkv / linear1) so deltas can be added to the
correct rows of the fused base output.
"""
from dataclasses import dataclass, field

import torch

import comfy.utils


@dataclass
class LoraDelta:
    """One LoRA pair targeting rows [out_offset, out_offset+out_size) of a module output."""

    down: torch.Tensor  # [rank, in_features]
    up: torch.Tensor    # [out_size, rank]
    adapter: str        # "sr" or "dpo"
    out_offset: int
    out_size: int


@dataclass
class LoraDeltaStore:
    """Deltas grouped by comfy module path (relative to the Flux model root)."""

    deltas: dict = field(default_factory=dict)

    def add(self, path, delta):
        self.deltas.setdefault(path, []).append(delta)

    def apply(self, path, x, out, scales):
        """Adds all deltas for ``path`` into ``out`` in place.

        Args:
            path: Comfy module path, e.g. ``"double_blocks.0.img_attn.qkv"``.
            x: Module input activations ``[..., in_features]``.
            out: Base module output to modify in place.
            scales: ``{"sr": float, "dpo": float}`` adapter scales.
        """
        for d in self.deltas.get(path, ()):
            scale = scales.get(d.adapter, 0.0)
            if scale == 0.0:
                continue
            delta = (x.to(d.down.dtype) @ d.down.T) @ d.up.T
            out[..., d.out_offset:d.out_offset + d.out_size] += (
                delta.to(out.dtype) * scale
            )

    def to(self, device, dtype=None):
        """Moves all delta tensors; returns self."""
        for entries in self.deltas.values():
            for d in entries:
                d.down = d.down.to(device=device, dtype=dtype or d.down.dtype)
                d.up = d.up.to(device=device, dtype=dtype or d.up.dtype)
        return self


def _diffusers_to_comfy_map(unet_config):
    """Returns {diffusers module path -> (comfy module path, offset, size|None)}."""
    key_map = comfy.utils.flux_to_diffusers(unet_config, output_prefix="")
    out = {}
    for dkey, target in key_map.items():
        if not dkey.endswith(".weight"):
            continue
        module = dkey[: -len(".weight")]
        if isinstance(target, tuple):
            comfy_key, slice_info = target[0], target[1]
            if slice_info is None:
                # Function-transformed full-tensor mapping (e.g. norm_out
                # swap_scale_shift). No ASASR key targets these; skip so an
                # accidental hit fails loudly via the unknown-key path.
                continue
            _, offset, size = slice_info
            out[module] = (comfy_key[: -len(".weight")], offset, size)
        else:
            out[module] = (target[: -len(".weight")], 0, None)
    return out


def build_delta_store(state_dict, adapter, unet_config, store=None):
    """Builds/extends a delta store from one LoRA state dict.

    Args:
        state_dict: Mapping of checkpoint keys to tensors.
        adapter: Adapter name recorded on each delta ("sr" or "dpo").
        unet_config: Comfy unet config dict (needs depth, depth_single_blocks,
            hidden_size).
        store: Existing store to extend, or None for a new one.

    Returns:
        The populated LoraDeltaStore.

    Raises:
        ValueError: A key does not match the known ASASR naming or cannot be
            mapped onto the comfy Flux module tree.
    """
    mapping = _diffusers_to_comfy_map(unet_config)
    store = store or LoraDeltaStore()
    pairs = {}
    for key in state_dict:
        if not key.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"Unexpected non-LoRA key in ASASR checkpoint: {key}")
        prefix = key.rsplit(".lora_", 1)[0]
        pairs.setdefault(prefix, key)
    for prefix in pairs:
        module = prefix[len("transformer."):] if prefix.startswith("transformer.") else prefix
        if module not in mapping:
            raise ValueError(f"Cannot map LoRA module onto comfy Flux tree: {module}")
        down = state_dict[f"{prefix}.lora_A.weight"]
        up = state_dict[f"{prefix}.lora_B.weight"]
        comfy_path, offset, size = mapping[module]
        store.add(
            comfy_path,
            LoraDelta(
                down=down.float(), up=up.float(), adapter=adapter,
                out_offset=offset, out_size=size if size is not None else up.shape[0],
            ),
        )
    return store


def load_delta_store(sr_path, dpo_path, unet_config):
    """Loads both ASASR LoRA files into one delta store."""
    sr_sd = comfy.utils.load_torch_file(str(sr_path), safe_load=True)
    dpo_sd = comfy.utils.load_torch_file(str(dpo_path), safe_load=True)
    store = build_delta_store(sr_sd, "sr", unet_config)
    return build_delta_store(dpo_sd, "dpo", unet_config, store=store)
