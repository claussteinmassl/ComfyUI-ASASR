"""Native ASASR three-branch forward on comfy.ldm.flux modules.

Branches: txt and img (main, base weights only) and cond (condition tokens,
base weights + LoRA deltas). Condition modulation vectors are computed at
timestep 0 with the same guidance/pooled inputs, matching the upstream
OminiControl-style implementation. Only module calls are used -- never raw
weights -- so quantized checkpoints work unchanged.
"""
import math

import torch

from comfy.ldm.flux.layers import ModulationOut, apply_mod, timestep_embedding
from comfy.ldm.flux.math import attention


def _mod_with_delta(mod_module, path, vec, store, scales, chunks):
    """Runs a comfy Modulation linear with the ASASR LoRA delta applied.

    Args:
        mod_module: ``comfy.ldm.flux.layers.Modulation`` instance.
        path: Comfy module path of the modulation linear.
        vec: Modulation input vector ``[B, D]`` or ``[B, 1, D]``.
        store: LoraDeltaStore holding the ASASR deltas.
        scales: ``{"sr": float, "dpo": float}`` adapter scales.
        chunks: 6 for double blocks, 3 for single blocks.

    Returns:
        A ``ModulationOut`` pair for ``chunks == 6``, else a single
        ``ModulationOut``.
    """
    x = torch.nn.functional.silu(vec)
    if x.ndim == 2:
        x = x[:, None, :]
    out = mod_module.lin(x)
    store.apply(path, x, out, scales)
    parts = out.chunk(chunks, dim=-1)
    if chunks == 6:
        return ModulationOut(*parts[:3]), ModulationOut(*parts[3:])
    return ModulationOut(*parts[:3])


def _split_qkv(qkv, num_heads):
    """Splits a fused qkv projection into per-head q, k, v tensors."""
    return qkv.view(qkv.shape[0], qkv.shape[1], 3, num_heads, -1).permute(2, 0, 3, 1, 4)


def _attn_bias(main_len, cond_len, condition_scale, device, dtype):
    """Builds the additive log(condition_scale) attention bias.

    The bias is applied between the main tokens (txt+img) and the condition
    tokens in both directions, matching the upstream ``c_factor`` behaviour.

    Memory: this is a dense ``[1, 1, L, L]`` tensor over the *joint* sequence
    (txt + img + cond). At production tile sizes (512px tile -> ~1024 img and
    ~1024 cond tokens, L ~= 2560) that is tens of MB, and comfy's attention
    backends make it worse: ``attention_sub_quad`` expands it to
    ``[B * heads, L, L]`` on every call, and the flash backend cannot consume an
    additive mask at all and falls back to sdpa with a per-call warning.
    ``condition_scale == 1.0`` therefore short-circuits to None and stays the
    allocation-free fast path.

    Args:
        main_len: Number of txt+img tokens.
        cond_len: Number of condition tokens.
        condition_scale: Multiplicative attention factor; 1.0 disables the bias.
        device: Target device.
        dtype: Target dtype.

    Returns:
        A ``[1, 1, L, L]`` additive bias, or None when ``condition_scale`` is 1.0.
    """
    if condition_scale == 1.0:
        return None
    total = main_len + cond_len
    bias = torch.zeros(total, total, device=device, dtype=dtype)
    # Mirror upstream ``torch.log(c_factor)``: a scale of 0 fully masks the
    # cross terms via -inf instead of raising like ``math.log(0)`` would.
    log_c = math.log(condition_scale) if condition_scale > 0 else float("-inf")
    bias[-cond_len:, :-cond_len] = log_c
    bias[:-cond_len, -cond_len:] = log_c
    return bias.unsqueeze(0).unsqueeze(0)


def asasr_forward(model, *, img, img_ids, txt, txt_ids, cond, cond_ids,
                  timesteps, y, guidance, store, sr_scale, dpo_scale,
                  condition_scale=1.0):
    """Runs the ASASR conditional forward on a comfy Flux model.

    Args:
        model: ``comfy.ldm.flux.model.Flux`` instance.
        img: Packed noisy latent tokens ``[B, L_img, C_in]``.
        img_ids: Image position ids ``[B, L_img, 3]``.
        txt: Text embeddings ``[B, L_txt, D_ctx]``.
        txt_ids: Text position ids ``[B, L_txt, 3]``.
        cond: Packed condition latent tokens ``[B, L_cond, C_in]``.
        cond_ids: Condition position ids ``[B, L_cond, 3]``.
        timesteps: ``[B]`` sigma values in [0, 1].
        y: Pooled text vector ``[B, vec_in_dim]``.
        guidance: ``[B]`` guidance values (embedded, FLUX.1-dev).
        store: LoraDeltaStore with the ASASR deltas.
        sr_scale: SR adapter scale.
        dpo_scale: DPO adapter scale.
        condition_scale: Attention bias factor between main and cond tokens.
            The default 1.0 is the fast path: no bias tensor is built at all.
            Any other value allocates a dense ``[1, 1, L, L]`` float bias over
            the joint txt+img+cond sequence, which is costly at production tile
            sizes (a 512px tile is ~1024 img + ~1024 cond tokens, so L ~= 2560
            and the bias is tens of MB). Comfy's ``attention_sub_quad`` backend
            additionally materializes a ``[B * heads, L, L]`` copy per attention
            call, and the flash backend cannot take an additive mask and falls
            back to sdpa with a warning on every call.

    Returns:
        Output tokens ``[B, L_img, C_out]`` (condition tokens dropped).
    """
    scales = {"sr": sr_scale, "dpo": dpo_scale}
    num_heads = model.num_heads
    dtype = img.dtype

    img_t = model.img_in(img)
    cond_t = model.img_in(cond)
    store.apply("img_in", cond, cond_t, scales)

    vec_t = model.time_in(timestep_embedding(timesteps, 256).to(dtype))
    vec_0 = model.time_in(
        timestep_embedding(torch.zeros_like(timesteps), 256).to(dtype))
    # Comfy convention: a missing guidance skips the embedding entirely, while
    # the vendored path substitutes zeros. ASASR always passes a guidance scale,
    # so the two never diverge in practice.
    if model.params.guidance_embed and guidance is not None:
        g = model.guidance_in(timestep_embedding(guidance, 256).to(dtype))
        vec_t = vec_t + g
        vec_0 = vec_0 + g
    vy = model.vector_in(y[:, :model.params.vec_in_dim])
    vec_t = vec_t + vy
    vec_0 = vec_0 + vy

    txt_t = model.txt_in(txt)

    pe_main = model.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))
    pe_cond = model.pe_embedder(cond_ids)
    pe_all = torch.cat((pe_main, pe_cond), dim=2)

    txt_len = txt_t.shape[1]
    main_len = txt_len + img_t.shape[1]
    cond_len = cond_t.shape[1]
    bias = _attn_bias(main_len, cond_len, condition_scale, img_t.device, dtype)

    for i, block in enumerate(model.double_blocks):
        path = f"double_blocks.{i}"
        # The condition MLP path below indexes img_mlp as a 3-element GELU
        # Sequential (Linear, GELU, Linear); other variants are unsupported.
        if len(block.img_mlp) != 3:
            raise NotImplementedError(
                "img_mlp variants other than the 3-element GELU Sequential "
                "are not supported")
        img_mod1, img_mod2 = block.img_mod(vec_t)
        txt_mod1, txt_mod2 = block.txt_mod(vec_t)
        cond_mod1, cond_mod2 = _mod_with_delta(
            block.img_mod, f"{path}.img_mod.lin", vec_0, store, scales, chunks=6)

        img_n = apply_mod(block.img_norm1(img_t), 1 + img_mod1.scale, img_mod1.shift)
        img_qkv = block.img_attn.qkv(img_n)
        cond_n = apply_mod(block.img_norm1(cond_t), 1 + cond_mod1.scale, cond_mod1.shift)
        cond_qkv = block.img_attn.qkv(cond_n)
        store.apply(f"{path}.img_attn.qkv", cond_n, cond_qkv, scales)
        txt_n = apply_mod(block.txt_norm1(txt_t), 1 + txt_mod1.scale, txt_mod1.shift)
        txt_qkv = block.txt_attn.qkv(txt_n)

        img_q, img_k, img_v = _split_qkv(img_qkv, num_heads)
        img_q, img_k = block.img_attn.norm(img_q, img_k, img_v)
        cond_q, cond_k, cond_v = _split_qkv(cond_qkv, num_heads)
        cond_q, cond_k = block.img_attn.norm(cond_q, cond_k, cond_v)
        txt_q, txt_k, txt_v = _split_qkv(txt_qkv, num_heads)
        txt_q, txt_k = block.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q, cond_q), dim=2)
        k = torch.cat((txt_k, img_k, cond_k), dim=2)
        v = torch.cat((txt_v, img_v, cond_v), dim=2)
        attn = attention(q, k, v, pe_all, mask=bias)
        txt_a = attn[:, :txt_len]
        img_a = attn[:, txt_len:main_len]
        cond_a = attn[:, main_len:]

        img_t = img_t + apply_mod(block.img_attn.proj(img_a), img_mod1.gate)
        cond_proj = block.img_attn.proj(cond_a)
        store.apply(f"{path}.img_attn.proj", cond_a, cond_proj, scales)
        cond_t = cond_t + apply_mod(cond_proj, cond_mod1.gate)
        txt_t = txt_t + apply_mod(block.txt_attn.proj(txt_a), txt_mod1.gate)

        img_m = apply_mod(block.img_norm2(img_t), 1 + img_mod2.scale, img_mod2.shift)
        img_t = img_t + apply_mod(block.img_mlp(img_m), img_mod2.gate)
        cond_m = apply_mod(block.img_norm2(cond_t), 1 + cond_mod2.scale, cond_mod2.shift)
        cond_h = block.img_mlp[1](block.img_mlp[0](cond_m))
        cond_ff = block.img_mlp[2](cond_h)
        store.apply(f"{path}.img_mlp.2", cond_h, cond_ff, scales)
        cond_t = cond_t + apply_mod(cond_ff, cond_mod2.gate)
        txt_m = apply_mod(block.txt_norm2(txt_t), 1 + txt_mod2.scale, txt_mod2.shift)
        txt_t = txt_t + apply_mod(block.txt_mlp(txt_m), txt_mod2.gate)
        if txt_t.dtype == torch.float16:  # comfy DoubleStreamBlock tail
            txt_t = torch.nan_to_num(txt_t, nan=0.0, posinf=65504, neginf=-65504)

    x = torch.cat((txt_t, img_t), dim=1)
    for i, block in enumerate(model.single_blocks):
        path = f"single_blocks.{i}"
        # FLUX.1 uses the plain (non-gated) mlp activation; the yak_mlp / SiLU
        # gated variants would need the halved-split handling comfy applies.
        if getattr(block, "yak_mlp", False):
            raise NotImplementedError("yak_mlp variants are not supported")
        mod, _ = block.modulation(vec_t)
        cond_mod = _mod_with_delta(
            block.modulation, f"{path}.modulation.lin", vec_0, store, scales, chunks=3)

        x_n = apply_mod(block.pre_norm(x), 1 + mod.scale, mod.shift)
        x_l1 = block.linear1(x_n)
        c_n = apply_mod(block.pre_norm(cond_t), 1 + cond_mod.scale, cond_mod.shift)
        c_l1 = block.linear1(c_n)
        store.apply(f"{path}.linear1", c_n, c_l1, scales)

        # Older comfy installs may not define mlp_hidden_dim_first (added
        # alongside yak_mlp support); fall back to the plain mlp_hidden_dim.
        mlp_hidden_dim = getattr(block, "mlp_hidden_dim_first", block.mlp_hidden_dim)
        split = [3 * block.hidden_size, mlp_hidden_dim]
        x_qkv, x_mlp = torch.split(x_l1, split, dim=-1)
        c_qkv, c_mlp = torch.split(c_l1, split, dim=-1)

        x_q, x_k, x_v = _split_qkv(x_qkv, num_heads)
        x_q, x_k = block.norm(x_q, x_k, x_v)
        c_q, c_k, c_v = _split_qkv(c_qkv, num_heads)
        c_q, c_k = block.norm(c_q, c_k, c_v)

        q = torch.cat((x_q, c_q), dim=2)
        k = torch.cat((x_k, c_k), dim=2)
        v = torch.cat((x_v, c_v), dim=2)
        attn = attention(q, k, v, pe_all, mask=bias)
        x_a = attn[:, :main_len]
        c_a = attn[:, main_len:]

        x_cat = torch.cat((x_a, block.mlp_act(x_mlp)), dim=2)
        x = x + apply_mod(block.linear2(x_cat), mod.gate)
        if x.dtype == torch.float16:  # comfy SingleStreamBlock tail
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        c_cat = torch.cat((c_a, block.mlp_act(c_mlp)), dim=2)
        c_out = block.linear2(c_cat)
        store.apply(f"{path}.linear2", c_cat, c_out, scales)
        cond_t = cond_t + apply_mod(c_out, cond_mod.gate)

    return model.final_layer(x[:, txt_len:], vec_t)
