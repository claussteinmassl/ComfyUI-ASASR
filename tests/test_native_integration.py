"""Opt-in end-to-end native test with a real FLUX checkpoint.

Enable with:
    ASASR_RUN_NATIVE_INTEGRATION=1 \
    ASASR_NATIVE_UNET=/path/flux1-dev-fp8.safetensors \
    ASASR_NATIVE_CLIP_L=/path/clip_l.safetensors \
    ASASR_NATIVE_T5=/path/t5xxl_fp8.safetensors \
    ASASR_NATIVE_VAE=/path/ae.safetensors \
    .venv/bin/pytest tests/test_native_integration.py -v
"""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.environ.get("ASASR_RUN_NATIVE_INTEGRATION"),
    reason="native integration disabled; set ASASR_RUN_NATIVE_INTEGRATION=1 "
           "and ASASR_NATIVE_UNET/_CLIP_L/_T5/_VAE paths",
)


def test_native_end_to_end_single_tile(tmp_path):
    import comfy.sd
    import comfy.utils

    from asasr.native.nodes_native import ASASRUpscale

    model = comfy.sd.load_diffusion_model(os.environ["ASASR_NATIVE_UNET"])
    clip = comfy.sd.load_clip(
        ckpt_paths=[os.environ["ASASR_NATIVE_CLIP_L"], os.environ["ASASR_NATIVE_T5"]],
        clip_type=comfy.sd.CLIPType.FLUX)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(os.environ["ASASR_NATIVE_VAE"]))

    image = torch.rand(1, 128, 128, 3)
    (out,) = ASASRUpscale().upscale(
        model=model, clip=clip, vae=vae, image=image, steps=4, guidance=3.5,
        seed=0, sr_lora_scale=1.0, dpo_lora_scale=1.0, condition_scale=1.0,
        tile_size=128, tile_overlap=16, color_fix="adain",
        auto_download=True)
    assert out.shape == (1, 512, 512, 3)
    assert torch.isfinite(out).all()
