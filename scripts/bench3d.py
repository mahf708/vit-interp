"""Forward+backward time and memory of 2D vs 3D variants at the training resolution (one GPU)."""
import time

import torch

from vit_interp.model import ViTConfig, build_net

dev = torch.device("cuda")
import sys
B = int(sys.argv[1]) if len(sys.argv) > 1 else 8
CK = len(sys.argv) > 2
base = dict(in_chans=39, out_chans=44, img_size=(180, 360), head_conv=3, noise_dim=32)
cands = {
    "vit d384 L8 h6 (v2-v7)": dict(dim=384, depth=8, heads=6),
    "3d split d256 L8 h4 hv2": dict(arch="vit3d", attn3d="split", dim=256, depth=8, heads=4, heads_v=2),
    "3d split d384 L8 h6 hv2": dict(arch="vit3d", attn3d="split", dim=384, depth=8, heads=6, heads_v=2),
    "3d factorized d256 L8 h4": dict(arch="vit3d", attn3d="factorized", dim=256, depth=8, heads=4),
    "3d alternate d256 L8 h4": dict(arch="vit3d", attn3d="alternate", dim=256, depth=8, heads=4),
    "3d split d256 L8 h4 rope": dict(arch="vit3d", attn3d="split", dim=256, depth=8, heads=4, heads_v=2, pos3d="rope"),
    "3d split d256 L6 h4 hv2": dict(arch="vit3d", attn3d="split", dim=256, depth=6, heads=4, heads_v=2),
}
for name, kw in cands.items():
    net = build_net(ViTConfig(**base, **kw)).to(dev)
    net.grad_checkpoint = CK
    net.train()
    for h in [m for m in net.modules() if isinstance(m, torch.nn.Linear)]:
        pass
    npar = sum(p.numel() for p in net.parameters()) / 1e6
    x = torch.randn(B, 39, 180, 360, device=dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-4)
    torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(6):
            if i == 2:
                torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = net(x)
            y.float().square().mean().backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 4
        print(f"{name:28s} {npar:5.1f} M params  {dt:6.3f} s/step (batch {B}{" ckpt" if CK else ""})  peak {torch.cuda.max_memory_allocated() / 2**30:5.1f} GB", flush=True)
    except torch.OutOfMemoryError:
        print(f"{name:28s} {npar:5.1f} M params  OOM at batch {B}", flush=True)
    del net, opt, x
    torch.cuda.empty_cache()
