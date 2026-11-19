import torch

from vit_interp import circuits as C
from vit_interp.model import ViT, ViTConfig

CFG = ViTConfig(in_chans=5, out_chans=3, img_size=(16, 32), patch=4, dim=32, depth=3, heads=4)


def _setup():
    torch.manual_seed(0)
    m = ViT(CFG).double()
    torch.nn.init.normal_(m.head.weight, std=0.1)
    xc = torch.randn(2, 5, 16, 32, dtype=torch.float64)
    xk = torch.randn(2, 5, 16, 32, dtype=torch.float64)
    return m, xc, xk, C.sender_outputs(m, xk)


def test_graph_counts():
    g = C.Graph(depth=3, heads=4)
    assert len(g.senders) == 1 + 3 * 5 and len(g.receivers) == 7
    # a0 sees only embed; m0 sees embed + 4 heads; out sees everything
    assert g.upstream("a0") == [0] and len(g.upstream("m0")) == 5 and len(g.upstream("out")) == 16
    assert g.sender_index(1, "m") == g.senders.index("m1")
    assert g.sender_index(2, "a", 3) == g.senders.index("a2.h3")


def test_all_clean_and_all_corrupt_are_exact():
    m, xc, xk, corr = _setup()
    torch.testing.assert_close(C.edge_forward(m, xc, corr, C.full_mask(m, 1.0)), m(xc))
    torch.testing.assert_close(C.edge_forward(m, xc, corr, C.full_mask(m, 0.0)), m(xk))


def test_single_edge_matches_hook_patch():
    # corrupting embed -> a0 alone == patching block 0's resid_pre with the corrupt embedding
    m, xc, xk, corr = _setup()
    g = C.graph_of(m)
    mask = C.full_mask(m)
    mask[0, g.receiver_index("a0")] = 0
    y = C.edge_forward(m, xc, corr, mask)
    _, ck = m.run_with_cache(xk, names={"blocks.0.ln1.hook_normalized"})
    y2 = m(xc, hooks={"blocks.0.ln1.hook_normalized": lambda t: ck["blocks.0.ln1.hook_normalized"]})
    torch.testing.assert_close(y, y2)


def test_eap_and_prune_run():
    m, xc, xk, corr = _setup()
    metric = C.box_metric(1, slice(0, 8), slice(0, 8))
    s = C.eap_scores(m, xc, corr, metric)
    assert s.shape == C.full_mask(m).shape and s.abs().sum() > 0
    mask, log = C.prune(m, xc, corr, metric, tau=1e-3)
    assert len(log) > 0
