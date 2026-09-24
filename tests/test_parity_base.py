"""Base parity: comfy Flux forward_orig == vendored diffusers forward (no LoRA/cond)."""
import pytest
import torch

from asasr.vendor.flux.transformer import tranformer_forward
from tests.fixtures.tiny_flux import build_tiny_pair, make_inputs


@pytest.mark.parametrize("t", [0.9, 0.1])
def test_base_forward_parity(t):
    dm, cm, _ = build_tiny_pair()
    x = make_inputs()
    timestep = torch.full((1,), t)

    with torch.no_grad():
        ref = tranformer_forward(
            dm, condition_latents=None, condition_ids=None, condition_type_ids=None,
            hidden_states=x["img"], encoder_hidden_states=x["txt"],
            pooled_projections=x["y"], timestep=timestep,
            img_ids=x["img_ids"][0], txt_ids=x["txt_ids"][0],
            guidance=x["guidance"], return_dict=False,
        )[0]
        out = cm.forward_orig(
            img=x["img"], img_ids=x["img_ids"], txt=x["txt"], txt_ids=x["txt_ids"],
            timesteps=timestep, y=x["y"], guidance=x["guidance"],
        )
    assert ref.shape == out.shape
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
