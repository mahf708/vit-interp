import numpy as np
import torch

from vit_interp.filters import SeamFilter


def test_seam_filter_keeps_large_scales_and_means_kills_patch_period():
    lat = np.linspace(-89.5, 89.5, 180)
    f = SeamFilter(lat, 360)
    lon = torch.arange(360.0)
    la = torch.tensor(lat, dtype=torch.float32)[:, None]
    big = torch.cos(torch.deg2rad(lon * 3))[None, :] * torch.cos(torch.deg2rad(la)) + 0.0 * la
    seam = torch.cos(2 * torch.pi * lon / 4)[None, :].expand(180, 360)  # the 4° patch period
    out_big = f(big[None, None])[0, 0]
    out_seam = f(seam[None, None].contiguous())[0, 0]
    mid = slice(30, 150)
    assert torch.allclose(out_big[mid], big[mid], atol=2e-2)
    assert out_seam[mid].abs().max() < 1e-3
    const = torch.full((1, 1, 180, 360), 3.0)
    assert torch.allclose(f(const), const, atol=1e-5)
    s = torch.zeros(1, 1, 180, 360)
    assert torch.allclose(f.apply(s, s + const), const, atol=1e-5)


def test_patch_jitter_identity_net_roundtrips_and_cycles_all_offsets():
    from vit_interp.filters import PatchJitter

    x = torch.randn(1, 3, 180, 360)
    j = PatchJitter(4, "cycle16")
    seen = set()
    for _ in range(16):
        oy, ox = j.next_shift()
        seen.add((oy, ox))
        y = j._unshift(j._shift(x, oy, ox), oy, ox)
        assert torch.equal(y[..., : 180 - 3, :][..., oy:, :], x[..., : 180 - 3, :][..., oy:, :])
    assert len(seen) == 16
    j2 = PatchJitter(4, "lon")
    out = j2(lambda z: z, x)
    assert torch.equal(out, x)


def test_dealiased_embedding_identity_filter_equals_patchify_and_is_smooth_under_shift():
    from vit_interp.model import ViT, ViTConfig

    torch.manual_seed(0)
    plain = ViT(ViTConfig(in_chans=5, out_chans=3, img_size=(40, 80), dim=32, depth=1, heads=2))
    x = torch.randn(1, 5, 40, 80)
    e_plain = plain.patch_embed(plain.patchify(x))
    ident = ViT(ViTConfig(in_chans=5, out_chans=3, img_size=(40, 80), dim=32, depth=1, heads=2, embed_filter=(1.0,)))
    ident.load_state_dict(plain.state_dict())
    assert torch.allclose(ident.embed_dealiased(x), e_plain, atol=1e-5)
    da = ViT(ViTConfig(in_chans=5, out_chans=3, img_size=(40, 80), dim=32, depth=1, heads=2,
                       embed_filter=(1.0, 4.0, 6.0, 4.0, 1.0)))
    da.load_state_dict(plain.state_dict())
    e = da.embed_dealiased(x)
    assert e.shape == e_plain.shape
    # a 1-px longitude shift changes the anti-aliased tokens less than the plain ones
    xs = torch.roll(x, 1, dims=-1)
    d_plain = (plain.patch_embed(plain.patchify(xs)) - e_plain).norm()
    d_da = (da.embed_dealiased(xs) - e).norm()
    assert d_da < d_plain
