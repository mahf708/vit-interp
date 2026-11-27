import pytest
import torch

from vit_interp.model import ViTConfig, build_net


@pytest.mark.parametrize("attn3d,pos3d,filt", [("split", "abs", ()), ("factorized", "rope", (1, 4, 6, 4, 1)),
                                                ("alternate", "abs", ()), ("split", "rope", ())])
def test_vit3d_shapes_init_and_grad(attn3d, pos3d, filt):
    torch.manual_seed(0)
    cfg = ViTConfig(in_chans=39, out_chans=44, img_size=(16, 32), dim=32, depth=2, heads=4, heads_v=2,
                    arch="vit3d", attn3d=attn3d, pos3d=pos3d, embed_filter=filt, head_conv=3, noise_dim=4)
    net = build_net(cfg)
    x = torch.randn(2, 39, 16, 32)
    y = net(x)
    assert y.shape == (2, 44, 16, 32)
    assert torch.count_nonzero(y) == 0  # zero heads: starts at persistence
    for h in (net.head_atm, net.head_surf):
        torch.nn.init.normal_(h.weight, std=0.1)
    cache = {}
    y = net(x, cache=cache, names={"blocks.1.hook_resid_post", "hook_embed"})
    assert cache["blocks.1.hook_resid_post"].shape == (2, 9, 32, 32)
    y.square().mean().backward()
    assert all(p.grad is not None for n, p in net.named_parameters() if "W_scale" not in n and "W_bias" not in n)


def test_vit3d_routes_levels_to_the_right_channels():
    """A perturbation of T_3 in the input reaches level-3 tokens only, at the embedding."""
    from vit_interp.data import ALL_NAMES

    cfg = ViTConfig(in_chans=39, out_chans=34, img_size=(16, 32), dim=16, depth=1, heads=2, heads_v=1, arch="vit3d")
    net = build_net(cfg)
    x = torch.zeros(1, 39, 16, 32)
    c0 = {}; net(x, cache=c0, names={"hook_embed"})
    x[:, ALL_NAMES.index("T_3")] = 1.0
    c1 = {}; net(x, cache=c1, names={"hook_embed"})
    changed = (c1["hook_embed"] - c0["hook_embed"]).abs().sum((0, 2, 3)) > 0
    assert changed.tolist() == [lv == 3 for lv in range(9)]


def test_vertical_heads_mix_only_within_columns():
    cfg = ViTConfig(in_chans=39, out_chans=34, img_size=(16, 32), dim=16, depth=1, heads=2, heads_v=2,
                    arch="vit3d", attn3d="split")
    net = build_net(cfg)
    blk = net.blocks[0].attn
    x = torch.randn(1, 9, 32, 16, requires_grad=True)
    from vit_interp.model import _Ctx
    out = blk(x, _Ctx(None, None, None), "a")
    out[0, 0, 5].sum().backward()  # token at level 0, patch 5
    g = x.grad[0].abs().sum(-1)  # (9, 32)
    assert g[:, 5].sum() > 0 and g[:, [i for i in range(32) if i != 5]].sum() == 0


@pytest.mark.parametrize("attn3d,pos3d", [("split", "rope"), ("alternate", "abs")])
def test_vit3d_edge_patching_is_exact(attn3d, pos3d):
    from vit_interp import circuits as C

    torch.manual_seed(0)
    cfg = ViTConfig(in_chans=39, out_chans=44, img_size=(16, 32), dim=32, depth=2, heads=4, heads_v=2,
                    arch="vit3d", attn3d=attn3d, pos3d=pos3d, head_conv=3, noise_dim=4)
    net = build_net(cfg).eval()
    for m in net.modules():  # make every part matter (zero-init heads and noise maps otherwise)
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.1)
    xa, xb = torch.randn(2, 39, 16, 32), torch.randn(2, 39, 16, 32)
    z = net.sample_noise(2, "cpu")
    with torch.no_grad():
        ya, yb = net(xa, noise=z), net(xb, noise=z)
        corrupt = C.sender_outputs(net, xb, noise=z)
        clean = C.edge_forward(net, xa, corrupt, C.full_mask(net, 1.0), noise=z)
        dirty = C.edge_forward(net, xa, corrupt, C.full_mask(net, 0.0), noise=z)
    assert torch.allclose(clean, ya, atol=1e-5)
    assert torch.allclose(dirty, yb, atol=1e-5)
    assert len(C.graph_of(net).edges()) > 0


def test_factorized_refuses_edge_circuits():
    from vit_interp import circuits as C

    cfg = ViTConfig(in_chans=39, out_chans=34, img_size=(16, 32), dim=16, depth=1, heads=2, arch="vit3d",
                    attn3d="factorized")
    net = build_net(cfg)
    with pytest.raises(NotImplementedError):
        C.sender_outputs(net, torch.randn(1, 39, 16, 32))
