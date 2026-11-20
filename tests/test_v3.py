"""v3: noise-conditioned LayerNorms, ensemble scores and loss, and the adapter on stochastic and
diagnostic (v2-shaped) models."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vit_interp import circuits as C
from vit_interp.ensemble import EnsembleLoss, RealSHT, _legendre, crps, energy_score
from vit_interp.model import ViT, ViTConfig

TINY = dict(in_chans=5, out_chans=3, img_size=(16, 32), patch=4, dim=32, depth=2, heads=2)


def _stochastic(noise_type="gaussian", live=True):
    torch.manual_seed(0)
    m = ViT(ViTConfig(**TINY, noise_dim=4, noise_type=noise_type)).double()
    torch.nn.init.normal_(m.head.weight, std=0.1)
    if live:  # zero-initialized conditioning would hide bugs
        for mod in m.modules():
            if hasattr(mod, "W_scale"):
                torch.nn.init.normal_(mod.W_scale.weight, std=0.2)
                torch.nn.init.normal_(mod.W_bias.weight, std=0.2)
    return m


def test_a_fresh_stochastic_model_is_the_plain_model():
    plain = ViT(ViTConfig(**TINY)).double()
    torch.nn.init.normal_(plain.head.weight, std=0.1)
    noisy = ViT(ViTConfig(**TINY, noise_dim=4)).double()
    missing, unexpected = noisy.load_state_dict(plain.state_dict(), strict=False)
    assert all("W_scale" in k or "W_bias" in k for k in missing) and not unexpected
    x = torch.randn(2, 5, 16, 32, dtype=torch.float64)
    torch.testing.assert_close(noisy(x), plain(x))  # zero-initialized conditioning


@pytest.mark.parametrize("noise_type", ["gaussian", "isotropic"])
def test_noise_changes_the_output_and_a_seed_fixes_it(noise_type):
    m = _stochastic(noise_type)
    x = torch.randn(2, 5, 16, 32, dtype=torch.float64)
    a = m(x, generator=torch.Generator().manual_seed(1))
    b = m(x, generator=torch.Generator().manual_seed(1))
    c = m(x, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)
    z = m.sample_noise(2, "cpu", torch.Generator().manual_seed(3)).double()
    torch.testing.assert_close(m(x, noise=z), m(x, noise=z))


def test_hooks_see_the_noise_and_the_residual_stream_still_decomposes():
    m = _stochastic()
    x = torch.randn(2, 5, 16, 32, dtype=torch.float64)
    z = m.sample_noise(2, "cpu", torch.Generator().manual_seed(0)).double()
    _, c = m.run_with_cache(x, noise=z)
    torch.testing.assert_close(c["hook_noise"], z)
    assert c["blocks.1.ln1.hook_cond_scale"].shape == (2, m.cfg.n_tokens, m.cfg.dim)
    r = c["hook_embed"] + c["hook_pos_embed"]
    for i in range(m.cfg.depth):
        torch.testing.assert_close(r, c[f"blocks.{i}.hook_resid_pre"])
        r = r + c[f"blocks.{i}.attn.hook_result"].sum(2) + m.blocks[i].attn.b_O + c[f"blocks.{i}.hook_mlp_out"]
    torch.testing.assert_close(r, c[f"blocks.{m.cfg.depth - 1}.hook_resid_post"])


def test_edge_patching_with_shared_noise_is_exact_at_both_ends():
    m = _stochastic()
    xc = torch.randn(2, 5, 16, 32, dtype=torch.float64)
    xk = torch.randn(2, 5, 16, 32, dtype=torch.float64)
    z = m.sample_noise(2, "cpu", torch.Generator().manual_seed(0)).double()
    corr = C.sender_outputs(m, xk, noise=z)
    torch.testing.assert_close(C.edge_forward(m, xc, corr, C.full_mask(m, 1.0), noise=z), m(xc, noise=z))
    torch.testing.assert_close(C.edge_forward(m, xc, corr, C.full_mask(m, 0.0), noise=z), m(xk, noise=z))
    with pytest.raises(ValueError, match="noise"):
        C.sender_outputs(m, xk)


def test_isotropic_noise_has_unit_variance_and_is_smooth():
    m = ViT(ViTConfig(**{**TINY, "img_size": (180, 360)}, noise_dim=8, noise_type="isotropic"))
    z = m.sample_noise(8, "cpu", torch.Generator().manual_seed(0)).reshape(8, 45, 90, 8)
    assert abs(float(z.var()) - 1) < 0.05
    assert float((z[:, :, 1:] * z[:, :, :-1]).mean() / z.var()) > 0.3  # neighbours correlate


# -- scores --------------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1.0, 0.95])
def test_crps_is_the_fair_estimator(alpha):
    rng = np.random.default_rng(0)
    gen, y = rng.normal(size=(3, 4, 5)), rng.normal(size=(3, 1, 5))
    n = gen.shape[1]
    eps = (1 - alpha) / 2
    brute = np.abs(gen - y).mean(1) - (1 - eps) * sum(
        np.abs(gen[:, i] - gen[:, j]) for i, j in itertools.permutations(range(n), 2)
    ) / (2 * n * (n - 1))
    np.testing.assert_allclose(crps(torch.tensor(gen), torch.tensor(y), alpha).numpy(), brute)
    # one member: the absolute error
    np.testing.assert_allclose(crps(torch.tensor(gen[:, :1]), torch.tensor(y)).numpy(), np.abs(gen[:, :1] - y)[:, 0])


def test_energy_score_on_toy_inputs():
    g = torch.tensor([[1 + 0j, 3 + 0j]]).reshape(1, 2)
    y = torch.tensor([[2 + 0j]]).reshape(1, 1)
    assert float(energy_score(g, y)[0]) == pytest.approx(0.0)  # E|X-y| = 1, E|X-X'| = 2
    g = torch.tensor([[1j, -1j, 1 + 0j]]).reshape(1, 3)
    y = torch.zeros(1, 1, dtype=torch.complex64)
    pairs = (2 + 2 ** 0.5 + 2 ** 0.5) / 3
    assert float(energy_score(g, y)[0]) == pytest.approx(1 - 0.5 * pairs, rel=1e-5)


def _gauss_lat(n=180):
    return np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(n)[0]))


def test_the_sht_maps_a_harmonic_to_one_coefficient():
    lat = _gauss_lat(60)
    sht = RealSHT(lat, 120)
    P = _legendre(np.sin(np.deg2rad(lat)), 60, 61)
    lon = np.deg2rad(np.arange(120) * 3.0)
    for l, m in [(0, 0), (7, 4), (30, 11)]:
        a = sht(torch.tensor(P[m, l][:, None] * np.cos(m * lon)[None])).abs()
        top = a.flatten().topk(2).values
        assert a[l, m] == top[0] and float(top[1]) < 1e-5
        assert float(top[0]) == pytest.approx(1.0 if m == 0 else 0.5, rel=1e-5)


def test_the_ensemble_loss_is_zero_for_a_perfect_collapsed_ensemble_and_positive_otherwise():
    lat = _gauss_lat(30)
    loss = EnsembleLoss(lat, 60, 0.9, 0.1)
    y = torch.randn(2, 1, 3, 30, 60)
    total, parts = loss(y.expand(2, 2, 3, 30, 60), y)
    assert float(total) == pytest.approx(0.0, abs=1e-6)
    total, parts = loss(torch.randn(2, 2, 3, 30, 60), y)
    assert float(total) > 0 and set(parts) == {"crps", "energy"}


# -- the adapter: stochastic and v2-shaped (diagnostic) models -------------------------------------

SMOKE2 = Path("/lcrc/globalscratch/ac.ngmahfouz/vit-interp/smoke-v2")
needs_data = pytest.mark.skipif(not (SMOKE2 / "1991.npy").exists(), reason="smoke-v2 data not here")


def _ckpt(tmp_path, noise_dim, n_diag):
    from vit_interp.data import N_ALL, N_PROG, Stats
    from vit_interp.emulator import Emulator

    torch.manual_seed(0)
    meta = json.loads((SMOKE2 / "meta.json").read_text())
    cfg = ViTConfig(in_chans=N_ALL, out_chans=N_PROG + n_diag, patch=4, dim=32, depth=2, heads=2,
                    noise_dim=noise_dim, head_conv=3 if n_diag else 0)
    em = Emulator(cfg, Stats(str(SMOKE2)), meta["lat"])
    torch.nn.init.normal_(em.net.head.weight, std=0.05)
    for mod in em.net.modules():
        if hasattr(mod, "W_scale"):
            torch.nn.init.normal_(mod.W_scale.weight, std=0.2)
            torch.nn.init.normal_(mod.W_bias.weight, std=0.2)
    path = tmp_path / f"v3_{noise_dim}_{n_diag}.pt"
    em.save(path, epoch=0, step=0)
    return str(path)


@needs_data
@pytest.mark.parametrize("noise_dim,n_diag", [(0, 10), (4, 10)])
def test_the_adapter_on_stochastic_and_diagnostic_models_keeps_patching_exact(tmp_path, noise_dim, n_diag):
    from exege.circuits import ComponentPatch, EdgePatch, open_patchable

    ck = _ckpt(tmp_path, noise_dim, n_diag)
    s = open_patchable(ck, "vit-interp", data=str(SMOKE2), year=1991, device="cpu", dtype="float64")
    base, src = s.states([0, 1]), s.states([5, 9])
    g = s.graph()

    def same(a, b):
        for k in a:
            np.testing.assert_allclose(a[k], b[k], rtol=1e-9, atol=1e-9)

    same(s.step(base), s.step(base))  # deterministic given the adapter's seed
    same(s.step(base, EdgePatch(src, g.mask(1.0))), s.step(base))
    same(s.step(base, EdgePatch(src, g.mask(0.0))), s.step(src))
    same(s.step(base, ComponentPatch(src, ())), s.step(base))
    same(s.step(base, ComponentPatch(src, g.senders)), s.step(src))
    if noise_dim:
        other = open_patchable(ck, "vit-interp", data=str(SMOKE2), year=1991, device="cpu",
                               dtype="float64", seed=7)
        assert not np.allclose(other.step(base)["U_3"], s.step(base)["U_3"])


@needs_data
def test_a_stochastic_intervenable_draws_its_noise_from_the_seed(tmp_path):
    from exege.latents import open_intervenable

    ck = _ckpt(tmp_path, 4, 10)
    tok = open_intervenable(ck, "vit-interp-tokens", data=str(SMOKE2), year=1991, device="cpu")
    st = tok.initial_state(0)
    a = tok.run(st, 2, noise_seed=3).fields["U_3"]
    b = tok.run(st, 2, noise_seed=3).fields["U_3"]
    c = tok.run(st, 2, noise_seed=4).fields["U_3"]
    d = tok.run(st, 2).fields["U_3"]
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    np.testing.assert_array_equal(d, tok.run(st, 2).fields["U_3"])  # no seed: no noise
