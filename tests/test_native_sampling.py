"""Sigma schedule, packing, and Euler-loop parity with diffusers."""
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux import calculate_shift

from asasr.native.sampling import euler_denoise, latent_ids, make_sigmas, pack_latent, unpack_latent

SCHED_CFG = dict(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True,
                 base_shift=0.5, max_shift=1.15,
                 base_image_seq_len=256, max_image_seq_len=4096)


def _reference_sigmas(steps, seq_len=1024):
    sched = FlowMatchEulerDiscreteScheduler(**SCHED_CFG)
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    mu = calculate_shift(seq_len, SCHED_CFG["base_image_seq_len"],
                         SCHED_CFG["max_image_seq_len"], SCHED_CFG["base_shift"],
                         SCHED_CFG["max_shift"])
    sched.set_timesteps(sigmas=sigmas, mu=mu)
    return sched


def test_sigma_schedule_matches_diffusers():
    for steps in (4, 28):
        sched = _reference_sigmas(steps)
        ours = make_sigmas(steps)
        torch.testing.assert_close(
            ours, sched.sigmas.float(), atol=1e-6, rtol=0)


def test_pack_unpack_roundtrip():
    x = torch.randn(2, 16, 8, 8)
    tokens = pack_latent(x)
    assert tokens.shape == (2, 16, 64)
    torch.testing.assert_close(unpack_latent(tokens, 8, 8), x)


def test_latent_ids_grid():
    ids = latent_ids(4, 3, batch=2)
    assert ids.shape == (2, 12, 3)
    assert ids[0, 0].tolist() == [0.0, 0.0, 0.0]
    assert ids[0, 1].tolist() == [0.0, 0.0, 1.0]
    assert ids[0, 3].tolist() == [0.0, 1.0, 0.0]


def test_euler_matches_scheduler_step():
    steps = 4
    sched = _reference_sigmas(steps)
    torch.manual_seed(0)
    x0 = torch.randn(1, 16, 4, 4)
    outputs = [torch.randn(1, 16, 4, 4) for _ in range(steps)]

    x_ref = x0.clone()
    for i, t in enumerate(sched.timesteps):
        x_ref = sched.step(outputs[i], t, x_ref, return_dict=False)[0]

    calls = iter(outputs)
    x_ours = euler_denoise(lambda x, sigma: next(calls), x0.clone(), make_sigmas(steps))
    torch.testing.assert_close(x_ours, x_ref, atol=1e-6, rtol=0)
