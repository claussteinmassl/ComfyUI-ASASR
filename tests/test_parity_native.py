"""Full parity: native comfy forward vs vendored diffusers forward with LoRA + condition."""
import pytest
import torch

from asasr.native.forward import asasr_forward
from asasr.native.lora import build_delta_store
from asasr.vendor.flux.transformer import tranformer_forward
from tests.fixtures.asasr_lora_patterns import all_module_paths
from tests.fixtures.tiny_flux import build_tiny_pair, make_inputs

H = 64
RANK = 4


def _random_lora_sd(seed, n_double=2, n_single=2):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for p in all_module_paths(n_double=n_double, n_single=n_single):
        if p.endswith("proj_mlp"):
            out_f = 4 * H
        elif p.endswith("norm1.linear"):
            out_f = 6 * H
        elif p.endswith("norm.linear"):
            out_f = 3 * H
        else:
            out_f = H
        in_f = {"x_embedder": 16}.get(p, H)
        if p.endswith("ff.net.2"):
            in_f = 4 * H
        if p.endswith("proj_out") and "single" in p:
            in_f = 5 * H
        sd[f"transformer.{p}.lora_A.weight"] = torch.randn(RANK, in_f, generator=g) * 0.1
        sd[f"transformer.{p}.lora_B.weight"] = torch.randn(out_f, RANK, generator=g) * 0.1
    return sd


def _load_peft_adapters(dm, sr_sd, dpo_sd, sr_scale, dpo_scale):
    dm.load_lora_adapter(dict(sr_sd), adapter_name="sr")
    dm.load_lora_adapter(dict(dpo_sd), adapter_name="dpo")
    dm.set_adapters(["sr", "dpo"], weights=[sr_scale, dpo_scale])
    assert set(dm.peft_config) == {"sr", "dpo"}


def _vendored(dm, x, timestep):
    return tranformer_forward(
        dm, condition_latents=x["cond"], condition_ids=x["cond_ids"][0],
        condition_type_ids=torch.ones(x["cond"].shape[1], 1) * 10,
        hidden_states=x["img"], encoder_hidden_states=x["txt"],
        pooled_projections=x["y"], timestep=timestep,
        img_ids=x["img_ids"][0], txt_ids=x["txt_ids"][0],
        guidance=x["guidance"], return_dict=False,
    )[0]


@pytest.mark.parametrize("t", [0.9, 0.5, 0.1])
@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.7, 0.3)])
def test_parity_with_lora_and_condition(t, scales):
    dm, cm, cfg = build_tiny_pair()
    sr_sd = _random_lora_sd(1)
    dpo_sd = _random_lora_sd(2)
    _load_peft_adapters(dm, sr_sd, dpo_sd, *scales)

    store = build_delta_store(dict(sr_sd), "sr", cfg)
    store = build_delta_store(dict(dpo_sd), "dpo", cfg, store=store)

    x = make_inputs()
    timestep = torch.full((1,), t)

    with torch.no_grad():
        ref = _vendored(dm, x, timestep)
        out = asasr_forward(
            cm, img=x["img"], img_ids=x["img_ids"], txt=x["txt"],
            txt_ids=x["txt_ids"], cond=x["cond"], cond_ids=x["cond_ids"],
            timesteps=timestep, y=x["y"], guidance=x["guidance"],
            store=store, sr_scale=scales[0], dpo_scale=scales[1],
        )
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "input_kwargs",
    [{"batch": 2}, {"cond_hw": (2, 3)}],
    ids=["batch2", "asymmetric_cond"],
)
def test_parity_input_shapes(input_kwargs):
    """Parity must hold for batched inputs and a cond grid unlike the img grid."""
    dm, cm, cfg = build_tiny_pair()
    sr_sd = _random_lora_sd(9)
    dpo_sd = _random_lora_sd(10)
    _load_peft_adapters(dm, sr_sd, dpo_sd, 0.7, 0.3)
    store = build_delta_store(dict(sr_sd), "sr", cfg)
    store = build_delta_store(dict(dpo_sd), "dpo", cfg, store=store)

    x = make_inputs(**input_kwargs)
    timestep = torch.full((x["img"].shape[0],), 0.5)
    with torch.no_grad():
        ref = _vendored(dm, x, timestep)
        out = asasr_forward(
            cm, img=x["img"], img_ids=x["img_ids"], txt=x["txt"],
            txt_ids=x["txt_ids"], cond=x["cond"], cond_ids=x["cond_ids"],
            timesteps=timestep, y=x["y"], guidance=x["guidance"],
            store=store, sr_scale=0.7, dpo_scale=0.3,
        )
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_parity_condition_scale_bias():
    dm, cm, cfg = build_tiny_pair()
    sr_sd = _random_lora_sd(3)
    dpo_sd = _random_lora_sd(4)
    _load_peft_adapters(dm, sr_sd, dpo_sd, 1.0, 1.0)
    store = build_delta_store(dict(sr_sd), "sr", cfg)
    store = build_delta_store(dict(dpo_sd), "dpo", cfg, store=store)

    biased = 0
    for name, module in dm.named_modules():
        if name.endswith(".attn"):
            module.c_factor = torch.ones(1, 1) * 2.0
            biased += 1
    assert biased == 4  # 2 double + 2 single blocks

    x = make_inputs()
    timestep = torch.full((1,), 0.5)
    with torch.no_grad():
        ref = _vendored(dm, x, timestep)
        out = asasr_forward(
            cm, img=x["img"], img_ids=x["img_ids"], txt=x["txt"],
            txt_ids=x["txt_ids"], cond=x["cond"], cond_ids=x["cond_ids"],
            timesteps=timestep, y=x["y"], guidance=x["guidance"],
            store=store, sr_scale=1.0, dpo_scale=1.0, condition_scale=2.0,
        )
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_condition_scale_one_allocates_no_mask(monkeypatch):
    """condition_scale == 1.0 must take the shortcut: no attention bias tensor."""
    import asasr.native.forward as fwd

    dm, cm, cfg = build_tiny_pair()
    store = build_delta_store(dict(_random_lora_sd(5)), "sr", cfg)
    x = make_inputs()

    seen = []
    real_attention = fwd.attention

    def spy(q, k, v, pe, mask=None, **kwargs):
        seen.append(mask)
        return real_attention(q, k, v, pe, mask=mask, **kwargs)

    monkeypatch.setattr(fwd, "attention", spy)
    with torch.no_grad():
        asasr_forward(
            cm, img=x["img"], img_ids=x["img_ids"], txt=x["txt"],
            txt_ids=x["txt_ids"], cond=x["cond"], cond_ids=x["cond_ids"],
            timesteps=torch.full((1,), 0.5), y=x["y"], guidance=x["guidance"],
            store=store, sr_scale=1.0, dpo_scale=0.0,
        )
    assert seen and all(m is None for m in seen)
    assert fwd._attn_bias(8, 4, 1.0, torch.device("cpu"), torch.float32) is None


def test_condition_scale_changes_output():
    """The c_factor bias must actually reach attention."""
    dm, cm, cfg = build_tiny_pair()
    store = build_delta_store(dict(_random_lora_sd(6)), "sr", cfg)
    x = make_inputs()
    kwargs = dict(
        img=x["img"], img_ids=x["img_ids"], txt=x["txt"], txt_ids=x["txt_ids"],
        cond=x["cond"], cond_ids=x["cond_ids"], timesteps=torch.full((1,), 0.5),
        y=x["y"], guidance=x["guidance"], store=store, sr_scale=1.0, dpo_scale=0.0,
    )
    with torch.no_grad():
        plain = asasr_forward(cm, **kwargs)
        scaled = asasr_forward(cm, condition_scale=2.0, **kwargs)
    assert not torch.allclose(plain, scaled, atol=1e-5)


def test_adapters_change_reference_output():
    """Guards against a silently no-op PEFT load making parity vacuous."""
    dm, _cm, _cfg = build_tiny_pair()
    x = make_inputs()
    timestep = torch.full((1,), 0.5)
    with torch.no_grad():
        before = _vendored(dm, x, timestep)
        _load_peft_adapters(dm, _random_lora_sd(7), _random_lora_sd(8), 1.0, 1.0)
        after = _vendored(dm, x, timestep)
    assert not torch.allclose(before, after, atol=1e-5)
