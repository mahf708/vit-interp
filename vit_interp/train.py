"""Train the ViT emulator. Single process or torchrun (DDP).

    torchrun --nproc-per-node 8 -m vit_interp.train --out RUN_DIR [options]

Resumes from RUN_DIR/latest.pt if present, and stops cleanly before --walltime-min.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .data import ALL_NAMES, DATA_ROOT, N_ALL, N_PROG, PROG_NAMES, Stats, Trajectories
from .emulator import Emulator
from .model import ViTConfig

REPORT = ["PS", "TS", "T_7", "T_3", "U_3", "V_3", "U_6", "specific_total_water_7"]


def years(s: str) -> list[int]:
    lo, hi = (int(v) for v in s.split("-"))
    return list(range(lo, hi + 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=DATA_ROOT)
    p.add_argument("--train-years", default="2000-2009")
    p.add_argument("--val-years", default="2010-2010")
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--batch", type=int, default=4, help="per GPU")
    p.add_argument("--n-steps", type=int, default=1, help="autoregressive steps in the loss")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--val-stride", type=int, default=7, help="subsample validation windows")
    p.add_argument("--rollout-steps", type=int, default=20)
    p.add_argument("--walltime-min", type=float, default=1e9)
    p.add_argument("--init", default=None, help="weights to start from (fine-tuning)")
    p.add_argument("--compile", action="store_true")
    a = p.parse_args()
    t_start = time.time()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    dev = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(1234 + rank)
    out = Path(a.out)
    log = (lambda *m: print(*m, flush=True)) if rank == 0 else (lambda *m: None)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    meta = json.loads((Path(a.data) / "meta.json").read_text())
    stats = Stats(a.data)
    cfg = ViTConfig(in_chans=N_ALL, out_chans=N_PROG, img_size=(180, 360), patch=a.patch,
                    dim=a.dim, depth=a.depth, heads=a.heads, mlp_ratio=a.mlp_ratio)
    em = Emulator(cfg, stats, meta["lat"]).to(dev)
    log(f"params: {sum(p.numel() for p in em.net.parameters()) / 1e6:.2f} M, tokens {cfg.n_tokens}")

    train = Trajectories(years(a.train_years), a.n_steps, a.data)
    val = Trajectories(years(a.val_years), 1, a.data, stride=a.val_stride)
    roll = Trajectories(years(a.val_years), a.rollout_steps, a.data, stride=4 * 15)  # a 5-day rollout every 15 days
    sampler = DistributedSampler(train, world, rank, shuffle=True, drop_last=True) if ddp else None
    dl = DataLoader(train, a.batch, sampler=sampler, shuffle=sampler is None, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
    steps_per_epoch = len(dl)
    total = a.epochs * steps_per_epoch

    opt = torch.optim.AdamW(em.net.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95),
                            fused=dev.type == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    epoch, step, best = 0, 0, float("inf")
    if (out / "latest.pt").exists():
        ck = torch.load(out / "latest.pt", map_location=dev, weights_only=False)
        em.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        epoch, step, best = ck["epoch"], ck["step"], ck["best"]
        log(f"resumed at epoch {epoch} step {step}")
    elif a.init:
        em.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)["state_dict"])
        log(f"initialized from {a.init}")

    net = DDP(em.net, device_ids=[local]) if ddp else em.net
    fwd = torch.compile(net) if a.compile else net

    def predict(x):  # physical (B, C_all, H, W) -> (next prog physical, normalized tendency)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            d = fwd(em.normalize(x))
        d = d.float()
        return x[:, :N_PROG] + d * em.res_std, d

    def loss_fn(traj):
        state, loss = traj[:, 0, :N_PROG], 0.0
        for t in range(a.n_steps):
            x = torch.cat([state, traj[:, t, N_PROG:]], 1)
            state, d = predict(x)
            target = (traj[:, t + 1, :N_PROG] - x[:, :N_PROG]) / em.res_std
            loss = loss + ((d - target) ** 2 * em.area_w).mean()
        return loss / a.n_steps

    def save(name):
        if rank == 0:
            em.save(out / name, opt=opt.state_dict(), sched=sched.state_dict(), epoch=epoch,
                    step=step, best=best, meta={"names": ALL_NAMES, "lat": meta["lat"], "lon": meta["lon"]})

    @torch.no_grad()
    def validate():
        em.eval()
        se = torch.zeros(N_PROG, device=dev, dtype=torch.float64)
        lsum, n = 0.0, 0
        idx = range(rank, len(val), world)
        for i in range(0, len(idx), a.batch):
            traj = torch.stack([torch.from_numpy(val[j]) for j in list(idx)[i:i + a.batch]]).to(dev)
            pred, d = predict(traj[:, 0])
            target = em.residual_target(traj[:, 0], traj[:, 1])
            lsum += float(((d - target) ** 2 * em.area_w).mean()) * len(traj)
            se += (((pred - traj[:, 1, :N_PROG]) ** 2) * em.area_w).mean((2, 3)).sum(0).double()
            n += len(traj)
        # rollouts (unbatched across ranks for simplicity: rank 0 only)
        rse = torch.zeros(a.rollout_steps, N_PROG, device=dev, dtype=torch.float64)
        nr = 0
        if rank == 0:
            for j in range(len(roll)):
                traj = torch.from_numpy(roll[j])[None].to(dev)
                state = traj[:, 0, :N_PROG]
                for t in range(a.rollout_steps):
                    state, _ = predict(torch.cat([state, traj[:, t, N_PROG:]], 1))
                    rse[t] += ((state - traj[:, t + 1, :N_PROG]) ** 2 * em.area_w).mean((0, 2, 3)).double()
                nr += 1
        tot = torch.tensor([lsum, n], device=dev, dtype=torch.float64)
        if ddp:
            dist.all_reduce(tot)
            dist.all_reduce(se)
        em.train()
        rmse1 = (se / tot[1]).sqrt().cpu().numpy()
        rmser = (rse / max(nr, 1)).sqrt().cpu().numpy()
        return float(tot[0] / tot[1]), rmse1, rmser

    log(f"{steps_per_epoch} steps/epoch, {total} total")
    em.train()
    t0, seen = time.time(), 0
    stop = False
    while epoch < a.epochs and not stop:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for traj in dl:
            traj = traj.to(dev, non_blocking=True)
            loss = loss_fn(traj)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(em.net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            seen += 1
            if step % 50 == 0:
                dt = (time.time() - t0) / seen
                log(f"ep {epoch} step {step} loss {loss.item():.4f} gnorm {gn.item():.3f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {dt:.3f}s/step")
            if (time.time() - t_start) / 60 > a.walltime_min:
                stop = True
                break
        if stop:
            save("latest.pt")
            log("walltime budget reached; saved latest.pt")
            break
        epoch += 1
        vloss, rmse1, rmser = validate()
        if vloss < best:
            best = vloss
            save("best.pt")
        save("latest.pt")
        if rank == 0:
            rep = {n: float(rmse1[PROG_NAMES.index(n)]) for n in REPORT}
            r5 = {n: float(rmser[-1, PROG_NAMES.index(n)]) for n in REPORT}
            log(f"== epoch {epoch} val loss {vloss:.4f} (best {best:.4f})")
            log("   1-step RMSE: " + " ".join(f"{k}={v:.3g}" for k, v in rep.items()))
            log(f"   {a.rollout_steps}-step RMSE: " + " ".join(f"{k}={v:.3g}" for k, v in r5.items()))
            with open(out / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"epoch": epoch, "step": step, "val_loss": vloss, "rmse_1step": rep,
                                    f"rmse_{a.rollout_steps}step": r5}) + "\n")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
