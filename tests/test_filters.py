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
