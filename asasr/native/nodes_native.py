"""Native ComfyUI node: ASASR 4x upscale on comfy FLUX models (fp8/GGUF-ready)."""
import numpy as np
import torch
from PIL import Image

from asasr import weights
from asasr.color_transfer import COLOR_FIX_MODES, apply_color_fix
from asasr.tiling import SCALE, merge_tiles, pad_to_grid, tile_boxes
from asasr.native.forward import asasr_forward
from asasr.native.lora import load_delta_store
from asasr.native.sampling import (
    euler_denoise, latent_ids, make_sigmas, pack_latent, unpack_latent)

CONDITION_RESOLUTION = 512
_STORE_CACHE: dict = {}


def _interrupt_check():
    """Raises ComfyUI's interrupt exception when the user cancels the run."""
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    mm.throw_exception_if_processing_interrupted()


def _load_model(model):
    """Loads the model patcher onto its compute device via comfy's manager.

    Also a safe no-op re-check: calling this again when ``model`` is already
    loaded on its compute device does nothing. This is used both for the
    initial load and to reload the FLUX model after other comfy calls
    (CLIP encode, per-tile VAE encode/decode) that internally invoke
    ``load_models_gpu`` for their own patchers and may evict this one.
    """
    import comfy.model_management as mm

    mm.load_models_gpu([model])


def _make_progress_bar(total: int):
    """Builds a comfy progress bar, or None if comfy.utils is unavailable."""
    try:
        from comfy.utils import ProgressBar
    except ImportError:
        return None
    return ProgressBar(total)


def _load_store_cached(auto_download, unet_config, device, dtype):
    """Resolves the ASASR LoRA weights and loads (or reuses) the delta store.

    The store is cached by resolved weight paths, device, and dtype -- not by
    adapter scale, since scales are applied per-forward-call rather than
    baked into the stored deltas. At most one store is kept in memory.

    Args:
        auto_download: Whether to download missing LoRA weights from HF.
        unet_config: Comfy unet config dict used to map LoRA keys onto the
            comfy Flux module tree.
        device: Target device for the delta store tensors.
        dtype: Target dtype for the delta store tensors.

    Returns:
        A ``LoraDeltaStore`` moved to ``device``/``dtype``.
    """
    lora_paths = weights.ensure_lora_weights(auto_download=auto_download)
    key = (str(lora_paths.sr), str(lora_paths.dpo), str(device), str(dtype),
           unet_config.get("depth"), unet_config.get("depth_single_blocks"),
           unet_config.get("hidden_size"))
    store = _STORE_CACHE.get(key)
    if store is None:
        _STORE_CACHE.clear()  # at most one delta store in memory
        store = load_delta_store(lora_paths.sr, lora_paths.dpo, unet_config)
        store.to(device, dtype)
        _STORE_CACHE[key] = store
    return store


def _tile_to_pixels(tile):
    """LR tile [h, w, 3] float -> bicubic 512x512 tensor [1, 512, 512, 3]."""
    arr = (np.clip(tile, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    img = Image.fromarray(arr).resize(
        (CONDITION_RESOLUTION, CONDITION_RESOLUTION), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)


def _denoise_tile(*, model, latent_format, vae, tile, txt, txt_pooled, steps,
                  guidance, seed, store, sr_scale, dpo_scale, condition_scale,
                  device, dtype):
    """Runs the full native pipeline for one LR tile; returns [512, 512, 3] float.

    Args:
        model: ComfyUI ``ModelPatcher`` wrapping the FLUX model. Comfy's
            ``vae.encode``/``vae.decode`` internally call ``load_models_gpu``
            for their own patchers and may evict this model from the compute
            device as a side effect, so it is reloaded here (a no-op when
            still resident) before running the diffusion_model forward calls.
        latent_format: The model's ``comfy.latent_formats`` instance.
        vae: Comfy VAE object with ``encode``/``decode``.
        tile: LR tile array ``[h, w, 3]`` float in 0..1.
        txt: Empty-prompt text embeddings ``[1, L_txt, D_ctx]``.
        txt_pooled: Empty-prompt pooled vector ``[1, vec_in_dim]``.
        steps: Number of denoising steps.
        guidance: Guidance scale.
        seed: Per-tile random seed.
        store: LoraDeltaStore with the ASASR deltas.
        sr_scale: SR adapter scale.
        dpo_scale: DPO adapter scale.
        condition_scale: OminiControl condition attention scale.
        device: Compute device.
        dtype: Compute dtype.

    Returns:
        Decoded tile pixels ``[512, 512, 3]`` float clamped to 0..1, on CPU.
    """
    pixels = _tile_to_pixels(tile)
    cond_latent = latent_format.process_in(vae.encode(pixels)).to(device, dtype)
    _load_model(model)  # vae.encode may have evicted the model; reload before denoising
    flux = model.model.diffusion_model
    cond_tokens = pack_latent(cond_latent)
    h_len, w_len = cond_latent.shape[2] // 2, cond_latent.shape[3] // 2
    ids = latent_ids(h_len, w_len, batch=1, device=device)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(cond_latent.shape, generator=generator).to(device, dtype)
    x = pack_latent(noise)

    txt = txt.to(device, dtype)
    y = txt_pooled.to(device, dtype)
    txt_ids = torch.zeros(1, txt.shape[1], 3, device=device)
    g = torch.full((1,), guidance, device=device, dtype=dtype)
    sigmas = make_sigmas(steps, image_seq_len=h_len * w_len).to(device)

    def model_fn(xt, sigma):
        _interrupt_check()
        t = sigma.reshape(1).to(device=device, dtype=dtype)
        return asasr_forward(
            flux, img=xt, img_ids=ids, txt=txt, txt_ids=txt_ids,
            cond=cond_tokens, cond_ids=ids, timesteps=t, y=y, guidance=g,
            store=store, sr_scale=sr_scale, dpo_scale=dpo_scale,
            condition_scale=condition_scale)

    tokens = euler_denoise(model_fn, x, sigmas,
                           step_callback=lambda i, n: _interrupt_check())
    latent = unpack_latent(tokens, cond_latent.shape[2], cond_latent.shape[3])
    decoded = vae.decode(latent_format.process_out(latent.float()))
    return decoded[0].clamp(0.0, 1.0).cpu()


class ASASRUpscale:
    """Tiled ASASR 4x super-resolution on a native ComfyUI FLUX model."""

    CATEGORY = "ASASR"
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sr_lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "dpo_lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "condition_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "tile_size": ("INT", {"default": 128, "min": 64, "max": 512, "step": 16}),
                "tile_overlap": ("INT", {"default": 16, "min": 0, "max": 63, "step": 1}),
                "color_fix": (list(COLOR_FIX_MODES), {"default": "adain"}),
                "auto_download": ("BOOLEAN", {"default": True}),
            }
        }

    @torch.inference_mode()
    def upscale(self, model, clip, vae, image, steps, guidance, seed,
                sr_lora_scale, dpo_lora_scale, condition_scale,
                tile_size, tile_overlap, color_fix, auto_download):
        """Runs tiled ASASR 4x super-resolution on a native FLUX ModelPatcher.

        Args:
            model: ComfyUI ``ModelPatcher`` wrapping a FLUX.1-dev model.
            clip: ComfyUI CLIP object (used only to encode the empty prompt).
            vae: ComfyUI VAE object.
            image: ``[B, H, W, C]`` float tensor in 0..1.
            steps: Diffusion steps per tile.
            guidance: Guidance scale.
            seed: Base random seed (per-tile seeds derive from this).
            sr_lora_scale: SR adapter scale.
            dpo_lora_scale: DPO adapter scale.
            condition_scale: OminiControl condition attention scale.
            tile_size: LR-side tile edge length.
            tile_overlap: LR-side overlap between tiles.
            color_fix: ``adain``, ``wavelet`` or ``none`` (applied per image).
            auto_download: Download missing ASASR LoRA weights from HF.

        Returns:
            A one-tuple ``([B, H*4, W*4, C] float tensor in 0..1,)``.

        Runs under ``torch.inference_mode`` itself rather than relying on
        ComfyUI's executor: with autograd enabled, every per-layer weight
        cast stays referenced across all steps and exhausts VRAM.
        """
        flux = model.model.diffusion_model
        if not hasattr(flux, "double_blocks"):
            raise ValueError(
                f"ASASR requires a FLUX.1-dev model; got {type(flux).__name__}")
        if not flux.params.guidance_embed:
            raise ValueError(
                "ASASR requires a guidance-distilled FLUX model "
                "(FLUX.1-dev); FLUX.1-schnell is not supported")

        # Overlap beyond half the tile breaks the feather-mask normalization.
        tile_overlap = min(tile_overlap, tile_size // 2)

        inner = model.model
        device = model.load_device
        dtype = inner.manual_cast_dtype or inner.get_dtype()
        store = _load_store_cached(
            auto_download, inner.model_config.unet_config, device, torch.float32)

        # CLIP.encode_from_tokens loads its own patcher via load_models_gpu,
        # which may evict the FLUX model -- load FLUX only after this call.
        txt, pooled = clip.encode_from_tokens(clip.tokenize(""), return_pooled=True)
        _load_model(model)

        # LR tile grid is identical across the batch (all images share the
        # same H, W), so the first image's tiling gives the total tile count.
        first_lr = image[0].detach().cpu().float().numpy()
        first_padded, _ = pad_to_grid(first_lr, tile_size, tile_overlap)
        tiles_per_image = len(tile_boxes(
            first_padded.shape[0], first_padded.shape[1], tile_size, tile_overlap))
        progress = _make_progress_bar(tiles_per_image * image.shape[0])

        results = []
        for index in range(image.shape[0]):
            _interrupt_check()
            lr = image[index].detach().cpu().float().numpy()
            padded, (orig_h, orig_w) = pad_to_grid(lr, tile_size, tile_overlap)
            boxes = tile_boxes(padded.shape[0], padded.shape[1], tile_size, tile_overlap)
            hr_edge = tile_size * SCALE
            hr_tiles = []
            for tindex, (y0, x0, y1, x1) in enumerate(boxes):
                out = _denoise_tile(
                    model=model, latent_format=inner.latent_format,
                    vae=vae, tile=padded[y0:y1, x0:x1], txt=txt, txt_pooled=pooled,
                    steps=steps, guidance=guidance,
                    seed=(seed + index * len(boxes) + tindex) % (2 ** 63),
                    store=store, sr_scale=sr_lora_scale, dpo_scale=dpo_lora_scale,
                    condition_scale=condition_scale, device=device, dtype=dtype)
                tile_img = out.numpy()
                if tile_img.shape[0] != hr_edge:
                    tile_img = np.asarray(Image.fromarray(
                        (np.clip(tile_img, 0, 1) * 255).round().astype(np.uint8)
                    ).resize((hr_edge, hr_edge), Image.BICUBIC), dtype=np.float32) / 255.0
                hr_tiles.append(tile_img.astype(np.float32))
                if progress is not None:
                    progress.update(1)
            merged = merge_tiles(hr_tiles, boxes, padded.shape[0], padded.shape[1], tile_overlap)
            merged = merged[: orig_h * SCALE, : orig_w * SCALE]
            out_t = torch.from_numpy(np.ascontiguousarray(merged))
            if color_fix != "none":
                ref = Image.fromarray((np.clip(lr, 0, 1) * 255).round().astype(np.uint8))
                ref = ref.resize((orig_w * SCALE, orig_h * SCALE), Image.BICUBIC)
                ref_t = torch.from_numpy(np.asarray(ref, dtype=np.float32) / 255.0)
                out_t = apply_color_fix(out_t, ref_t, color_fix)
            results.append(out_t.clamp(0.0, 1.0))
        return (torch.stack(results, dim=0),)


NATIVE_NODE_CLASS_MAPPINGS = {"ASASRUpscale": ASASRUpscale}
NATIVE_NODE_DISPLAY_NAME_MAPPINGS = {"ASASRUpscale": "ASASR Upscale (4x)"}
