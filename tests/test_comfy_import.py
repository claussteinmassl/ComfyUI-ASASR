"""ComfyUI (pinned in .dev/) is importable in the test environment."""


def test_comfy_flux_modules_import():
    import comfy.latent_formats
    import comfy.utils
    from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock, apply_mod  # noqa: F401
    from comfy.ldm.flux.model import Flux  # noqa: F401

    key_map = comfy.utils.flux_to_diffusers(
        {"depth": 1, "depth_single_blocks": 1, "hidden_size": 8}
    )
    assert "transformer_blocks.0.attn.to_q.weight" in key_map
    fmt = comfy.latent_formats.Flux()
    assert fmt.scale_factor is not None
