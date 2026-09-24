"""LoRA delta store: key mapping, slicing, application."""
import torch

import pytest

from asasr.native.lora import LoraDeltaStore, build_delta_store
from tests.fixtures.asasr_lora_patterns import all_module_paths

UNET_CONFIG = {"depth": 19, "depth_single_blocks": 38, "hidden_size": 8}
H = UNET_CONFIG["hidden_size"]


def _sd(paths, rank=2, in_features=8, out_features=8):
    sd = {}
    for p in paths:
        sd[f"transformer.{p}.lora_A.weight"] = torch.randn(rank, in_features)
        sd[f"transformer.{p}.lora_B.weight"] = torch.randn(out_features, rank)
    return sd


def test_fused_qkv_mapping_offsets():
    sd = _sd(["transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_k",
              "transformer_blocks.0.attn.to_v"])
    store = build_delta_store(sd, "sr", UNET_CONFIG)
    deltas = store.deltas["double_blocks.0.img_attn.qkv"]
    assert sorted(d.out_offset for d in deltas) == [0, H, 2 * H]
    assert all(d.out_size == H for d in deltas)


def test_single_block_linear1_mapping():
    sd = _sd(["single_transformer_blocks.3.attn.to_q"]) | _sd(
        ["single_transformer_blocks.3.proj_mlp"], out_features=4 * H)
    store = build_delta_store(sd, "sr", UNET_CONFIG)
    deltas = store.deltas["single_blocks.3.linear1"]
    offsets = {d.out_offset: d.out_size for d in deltas}
    assert offsets[0] == H          # to_q
    assert offsets[3 * H] == 4 * H  # proj_mlp


def test_plain_module_mappings():
    sd = _sd(["x_embedder", "transformer_blocks.1.norm1.linear",
              "single_transformer_blocks.2.norm.linear",
              "transformer_blocks.1.attn.to_out.0", "transformer_blocks.1.ff.net.2",
              "single_transformer_blocks.2.proj_out"])
    store = build_delta_store(sd, "sr", UNET_CONFIG)
    for key in ["img_in", "double_blocks.1.img_mod.lin", "single_blocks.2.modulation.lin",
                "double_blocks.1.img_attn.proj", "double_blocks.1.img_mlp.2",
                "single_blocks.2.linear2"]:
        assert key in store.deltas, key


def test_apply_matches_manual_math():
    sd = _sd(["transformer_blocks.0.attn.to_q"])
    store = build_delta_store(sd, "sr", UNET_CONFIG)
    x = torch.randn(2, 5, 8)
    out = torch.zeros(2, 5, 3 * H)
    store.apply("double_blocks.0.img_attn.qkv", x, out, {"sr": 0.5, "dpo": 1.0})
    down = sd["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"]
    up = sd["transformer.transformer_blocks.0.attn.to_q.lora_B.weight"]
    expected = (x @ down.T @ up.T) * 0.5
    assert torch.allclose(out[..., :H], expected, atol=1e-6)
    assert out[..., H:].abs().max() == 0


def test_apply_unknown_path_is_noop():
    store = LoraDeltaStore(deltas={})
    out = torch.zeros(1, 2, 4)
    store.apply("nonexistent", torch.randn(1, 2, 4), out, {"sr": 1.0, "dpo": 1.0})
    assert out.abs().max() == 0


def test_unknown_key_raises():
    sd = {"transformer.bogus_module.lora_A.weight": torch.randn(2, 8),
          "transformer.bogus_module.lora_B.weight": torch.randn(8, 2)}
    with pytest.raises(ValueError, match="bogus_module"):
        build_delta_store(sd, "sr", UNET_CONFIG)


def test_store_to_moves_tensors():
    sd = _sd(["transformer_blocks.0.attn.to_q"])
    store = build_delta_store(sd, "sr", UNET_CONFIG)
    store.to("cpu", torch.float16)
    for deltas in store.deltas.values():
        for d in deltas:
            assert d.down.dtype == torch.float16 and d.up.dtype == torch.float16
            assert d.down.device.type == "cpu"
    # dtype=None preserves existing dtype
    store.to("cpu")
    for deltas in store.deltas.values():
        for d in deltas:
            assert d.down.dtype == torch.float16


def test_full_real_inventory_maps():
    mlp = {"proj_mlp": 4 * H, "ff.net.2": H}
    sd = {}
    for p in all_module_paths():
        out_f = 4 * H if p.endswith("proj_mlp") else H
        sd[f"transformer.{p}.lora_A.weight"] = torch.randn(2, 8)
        sd[f"transformer.{p}.lora_B.weight"] = torch.randn(out_f, 2)
    store = build_delta_store(sd, "sr", UNET_CONFIG)
    assert len(store.deltas) > 0
    del mlp
