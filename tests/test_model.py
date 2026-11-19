import torch

from vit_interp.model import ViT, ViTConfig

CFG = ViTConfig(in_chans=5, out_chans=3, img_size=(16, 32), patch=4, dim=32, depth=3, heads=4)


def _model():
    torch.manual_seed(0)
    m = ViT(CFG).double()
    torch.nn.init.normal_(m.head.weight, std=0.1)  # zero-init head would hide bugs
    return m


def test_shapes_and_hook_paths_agree():
    m = _model()
    x = torch.randn(2, 5, 16, 32, dtype=torch.float64)
    y = m(x)
    assert y.shape == (2, 3, 16, 32)
    # explicit attention (pattern hooked) and per-head result path give the same output
    y2, cache = m.run_with_cache(x)
    torch.testing.assert_close(y, y2)
    p = cache["blocks.1.attn.hook_pattern"]
    assert p.shape == (2, 4, CFG.n_tokens, CFG.n_tokens)
    torch.testing.assert_close(p.sum(-1), torch.ones_like(p.sum(-1)))
    res = cache["blocks.1.attn.hook_result"]
    torch.testing.assert_close(res.sum(2) + m.blocks[1].attn.b_O, cache["blocks.1.hook_attn_out"])


def test_residual_stream_decomposes():
    m = _model()
    x = torch.randn(2, 5, 16, 32, dtype=torch.float64)
    _, c = m.run_with_cache(x)
    r = c["hook_embed"] + c["hook_pos_embed"]
    for i in range(CFG.depth):
        torch.testing.assert_close(r, c[f"blocks.{i}.hook_resid_pre"])
        r = r + c[f"blocks.{i}.attn.hook_result"].sum(2) + m.blocks[i].attn.b_O + c[f"blocks.{i}.hook_mlp_out"]
    torch.testing.assert_close(r, c[f"blocks.{CFG.depth - 1}.hook_resid_post"])


def test_hook_edits_propagate():
    m = _model()
    x = torch.randn(1, 5, 16, 32, dtype=torch.float64)
    y = m(x)
    y0 = m(x, hooks={"blocks.0.attn.hook_z": lambda z: z * 0})
    assert not torch.allclose(y, y0)


def test_patchify_roundtrip():
    m = _model()
    x = torch.randn(2, 3, 16, 32, dtype=torch.float64)
    torch.testing.assert_close(m.unpatchify(m.patchify(x), 3), x)
    # token 0 is the first (lat, lon) patch
    torch.testing.assert_close(m.patchify(x)[0, 0].reshape(3, 4, 4), x[0, :, :4, :4])
