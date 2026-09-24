"""Native node orchestration without real models."""
import sys
import types
from unittest import mock

import pytest
import torch


def _import_native_nodes():
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_folder_paths = lambda name: ["/tmp/models/loras"]
    sys.modules.setdefault("folder_paths", folder_paths)
    import importlib

    from asasr.native import nodes_native

    return importlib.reload(nodes_native)


class _FakeVAE:
    def encode(self, pixels):
        b, h, w, _ = pixels.shape
        return torch.zeros(b, 16, h // 8, w // 8)

    def decode(self, latent):
        b, _, h, w = latent.shape
        return torch.full((b, h * 8, w * 8, 3), 0.5)


class _FakeCLIP:
    def tokenize(self, text):
        assert text == ""
        return {"t5xxl": [[0] * 4], "l": [[0] * 4]}

    def encode_from_tokens(self, tokens, return_pooled=False):
        return torch.zeros(1, 8, 24), torch.zeros(1, 32)


class _FakeLatentFormat:
    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent


def _fake_diffusion_model(*, double_blocks=True, guidance_embed=True):
    """Builds a stub diffusion_model passing (or failing) the FLUX guard."""
    kwargs = {"params": types.SimpleNamespace(guidance_embed=guidance_embed)}
    if double_blocks:
        kwargs["double_blocks"] = [object(), object()]
    return types.SimpleNamespace(**kwargs)


def _fake_model_patcher(diffusion_model=None):
    inner = types.SimpleNamespace(
        latent_format=_FakeLatentFormat(),
        diffusion_model=diffusion_model or _fake_diffusion_model(),
        manual_cast_dtype=None,
        get_dtype=lambda: torch.float32,
        model_config=types.SimpleNamespace(unet_config={
            "depth": 2, "depth_single_blocks": 2, "hidden_size": 64}),
    )
    return types.SimpleNamespace(model=inner, load_device="cpu")


def test_input_types_and_registration():
    nn = _import_native_nodes()
    req = nn.ASASRUpscale.INPUT_TYPES()["required"]
    for key in ("model", "clip", "vae", "image", "steps", "guidance", "seed",
                "sr_lora_scale", "dpo_lora_scale", "condition_scale",
                "tile_size", "tile_overlap", "color_fix", "auto_download"):
        assert key in req, key
    assert req["model"][0] == "MODEL"
    assert req["steps"][1]["default"] == 28
    assert "ASASRUpscale" in nn.NATIVE_NODE_CLASS_MAPPINGS


def test_upscale_orchestration_with_stubs():
    nn = _import_native_nodes()
    calls = {"tiles": 0, "seeds": []}

    def fake_denoise_tile(**kwargs):
        calls["tiles"] += 1
        calls["seeds"].append(kwargs["seed"])
        return torch.full((512, 512, 3), 0.5)

    with mock.patch.object(nn, "_load_store_cached", return_value=mock.MagicMock()), \
         mock.patch.object(nn, "_denoise_tile", side_effect=fake_denoise_tile), \
         mock.patch.object(nn, "_load_model", return_value=None):
        node = nn.ASASRUpscale()
        (out,) = node.upscale(
            model=_fake_model_patcher(), clip=_FakeCLIP(), vae=_FakeVAE(),
            image=torch.rand(1, 100, 130, 3), steps=4, guidance=3.5, seed=7,
            sr_lora_scale=1.0, dpo_lora_scale=1.0, condition_scale=1.0,
            tile_size=128, tile_overlap=16, color_fix="none", auto_download=False,
        )
    assert out.shape == (1, 400, 520, 3)
    assert calls["tiles"] == 2
    assert calls["seeds"] == [7, 8]


def test_upscale_runs_tiles_in_inference_mode():
    """The node must not rely on ComfyUI's executor to disable autograd.

    Outside the executor (e.g. scripts, tests) an autograd-enabled forward
    keeps every cast weight alive across all steps and runs out of memory.
    """
    nn = _import_native_nodes()
    modes = []

    def fake_denoise_tile(**kwargs):
        modes.append(torch.is_inference_mode_enabled())
        return torch.full((512, 512, 3), 0.5)

    assert not torch.is_inference_mode_enabled()
    with mock.patch.object(nn, "_load_store_cached", return_value=mock.MagicMock()), \
         mock.patch.object(nn, "_denoise_tile", side_effect=fake_denoise_tile), \
         mock.patch.object(nn, "_load_model", return_value=None):
        nn.ASASRUpscale().upscale(
            model=_fake_model_patcher(), clip=_FakeCLIP(), vae=_FakeVAE(),
            image=torch.rand(1, 100, 130, 3), steps=4, guidance=3.5, seed=7,
            sr_lora_scale=1.0, dpo_lora_scale=1.0, condition_scale=1.0,
            tile_size=128, tile_overlap=16, color_fix="none", auto_download=False,
        )
    assert modes and all(modes)


def test_upscale_rejects_non_flux_model():
    nn = _import_native_nodes()
    non_flux = _fake_model_patcher(
        diffusion_model=_fake_diffusion_model(double_blocks=False))
    node = nn.ASASRUpscale()
    with pytest.raises(ValueError, match="FLUX.1-dev"):
        node.upscale(
            model=non_flux, clip=_FakeCLIP(), vae=_FakeVAE(),
            image=torch.rand(1, 100, 130, 3), steps=4, guidance=3.5, seed=7,
            sr_lora_scale=1.0, dpo_lora_scale=1.0, condition_scale=1.0,
            tile_size=128, tile_overlap=16, color_fix="none", auto_download=False,
        )


def test_upscale_rejects_schnell_model():
    nn = _import_native_nodes()
    schnell = _fake_model_patcher(
        diffusion_model=_fake_diffusion_model(guidance_embed=False))
    node = nn.ASASRUpscale()
    with pytest.raises(ValueError, match="schnell"):
        node.upscale(
            model=schnell, clip=_FakeCLIP(), vae=_FakeVAE(),
            image=torch.rand(1, 100, 130, 3), steps=4, guidance=3.5, seed=7,
            sr_lora_scale=1.0, dpo_lora_scale=1.0, condition_scale=1.0,
            tile_size=128, tile_overlap=16, color_fix="none", auto_download=False,
        )


def test_root_registration_includes_native():
    """The package root __init__.py exports the native node mappings.

    ``import __init__`` (as a bare top-level module) fails with "attempted
    relative import with no known parent package": __init__.py uses relative
    imports (``.asasr.native...``) that only resolve when Python
    knows it is a package's __init__, which requires __package__ to be set.
    ComfyUI itself always loads this file that way (as
    ``custom_nodes.ComfyUI-ASASR``). This test recreates that exact context
    via importlib with ``submodule_search_locations`` pointing at the repo
    root, then asserts on the real merge output rather than using a plain
    top-level import, which is impractical under pytest.
    """
    import importlib.util
    import pathlib

    sys.modules.setdefault("folder_paths", types.ModuleType("folder_paths"))
    sys.modules["folder_paths"].get_folder_paths = lambda name: ["/tmp"]

    root = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "asasr_root_pkg_for_test", root / "__init__.py",
        submodule_search_locations=[str(root)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pkg
    spec.loader.exec_module(pkg)

    assert "ASASRUpscale" in pkg.NODE_CLASS_MAPPINGS
    assert pkg.NODE_DISPLAY_NAME_MAPPINGS["ASASRUpscale"] == "ASASR Upscale (4x)"

    # nodes_native also exports the mappings on its own.
    nn = _import_native_nodes()
    assert "ASASRUpscale" in nn.NATIVE_NODE_CLASS_MAPPINGS
