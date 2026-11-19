"""Train the ViT emulator. Single process or torchrun (DDP).

    torchrun --nproc-per-node 8 -m vit_interp.train --out RUN_DIR [options]

Resumes from RUN_DIR/latest.pt if present, and stops cleanly before --walltime-min.

v2 options (all off by default, so v1 runs are unchanged): a dataset with diagnostics
(meta["diag_names"]) makes the model predict them too; --corrector puts ACE-style
conservation in the loop (the loss is taken on corrected outputs, as in ACE); --input-noise
perturbs the prognostic inputs (targets are relative to the perturbed state, so the model
learns to correct small errors); --ema keeps an exponential moving average of the weights,
which is what validation uses and what checkpoints carry as ``state_dict`` (the raw weights
are kept as ``raw_net`` for resuming); --head-conv adds a smoothing convolution to the head.

v3 options (off by default): --noise-dim makes the model stochastic (noise-conditioned
LayerNorms, as ACE2S); --loss ensemble with --n-ensemble n >= 2 trains on ACE's EnsembleLoss
(fair CRPS + spectral energy score; --crps-weight, --energy-weight, --crps-alpha) instead of
MSE: each sample is run as n members with independent noise, each rolled out on its own.
Validation then reports the ensemble mean's RMSE, the CRPS, and the spread/skill ratio.
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
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch", type=int, default=2, help="batches per worker (shared memory!)")
    p.add_argument("--val-stride", type=int, default=7, help="subsample validation windows")
    p.add_argument("--rollout-steps", type=int, default=20)
    p.add_argument("--walltime-min", type=float, default=1e9)
    p.add_argument("--init", default=None, help="weights to start from (fine-tuning)")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--head-conv", type=int, default=0, help="k x k output convolution (0: none)")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0: off)")
    p.add_argument("--input-noise", type=float, default=0.0,
                   help="std of noise on prognostic inputs, in units of each field's std")
    p.add_argument("--corrector", default="",
                   help="comma list of dry_air,positive,moisture,energy (ACE-style, in the loop)")
    p.add_argument("--unaccounted-heating", type=float, default=0.0, help="W/m2, for energy")
    p.add_argument("--noise-dim", type=int, default=0, help="noise channels per token (0: deterministic)")
    p.add_argument("--noise-type", default="gaussian", choices=["gaussian", "isotropic"])
    p.add_argument("--loss", default="mse", choices=["mse", "ensemble"])
    p.add_argument("--n-ensemble", type=int, default=1, help="members per sample (ensemble loss)")
    p.add_argument("--crps-weight", type=float, default=1.0)
    p.add_argument("--energy-weight", type=float, default=0.0,
                   help="ACE adds 0.1 x a spectral energy score; off by default (CRPS alone)")
    p.add_argument("--crps-alpha", type=float, default=1.0, help="1: fair CRPS; <1: almost fair")
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
    diag_names = meta.get("diag_names", [])
    cfg = ViTConfig(in_chans=N_ALL, out_chans=N_PROG + len(diag_names), img_size=(180, 360),
                    patch=a.patch, dim=a.dim, depth=a.depth, heads=a.heads, mlp_ratio=a.mlp_ratio,
                    head_conv=a.head_conv, noise_dim=a.noise_dim, noise_type=a.noise_type)
    em = Emulator(cfg, stats, meta["lat"]).to(dev)
    flags = {f for f in a.corrector.split(",") if f}
    if flags:
        from .corrector import Corrector, CorrectorConfig

        ccfg = CorrectorConfig(**{k: k in flags for k in ("dry_air", "positive", "moisture", "energy")},
                               unaccounted_heating=a.unaccounted_heating)
        em.corrector = Corrector(ccfg, meta["lat"], meta["ak"], meta["bk"], diag_names=diag_names).to(dev)
        log(f"corrector in the loop: {sorted(flags)}")
    if diag_names:
        log(f"diagnostics predicted: {', '.join(diag_names)}")
    ens_loss = None
    if a.loss == "ensemble":
        if a.n_ensemble < 2 or not a.noise_dim:
            raise SystemExit("--loss ensemble needs --n-ensemble >= 2 and --noise-dim > 0")
        from .ensemble import EnsembleLoss

        ens_loss = EnsembleLoss(meta["lat"], len(meta["lon"]), a.crps_weight, a.energy_weight,
                                a.crps_alpha).to(dev)
        log(f"ensemble loss: {a.n_ensemble} members, crps {a.crps_weight}, energy {a.energy_weight}")
    n_ens = a.n_ensemble if ens_loss is not None else 1
    log(f"params: {sum(p.numel() for p in em.net.parameters()) / 1e6:.2f} M, tokens {cfg.n_tokens}")

    train = Trajectories(years(a.train_years), a.n_steps, a.data)
    val = Trajectories(years(a.val_years), 1, a.data, stride=a.val_stride)
    roll = Trajectories(years(a.val_years), a.rollout_steps, a.data, stride=4 * 15)  # a 5-day rollout every 15 days
    sampler = DistributedSampler(train, world, rank, shuffle=True, drop_last=True) if ddp else None
    dl = DataLoader(train, a.batch, sampler=sampler, shuffle=sampler is None, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=a.prefetch)
    steps_per_epoch = len(dl)
    total = a.epochs * steps_per_epoch

    opt = torch.optim.AdamW(em.net.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95),
                            fused=dev.type == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    epoch, step, best = 0, 0, float("inf")
    ema = None
    if a.ema:
        ema = {k: v.detach().clone().float() for k, v in em.net.state_dict().items()}
    if (out / "latest.pt").exists():
        ck = torch.load(out / "latest.pt", map_location=dev, weights_only=False)
        missing, unexpected = em.load_state_dict(ck["state_dict"], strict=False)
        missing = [k for k in missing if not k.startswith("corrector.")]  # never saved: built from meta
        if missing or unexpected:
            raise RuntimeError(f"resume mismatch: missing {missing}, unexpected {unexpected}")
        if ema is not None:
            ema = {k[len("net."):]: v.clone().float() for k, v in ck["state_dict"].items()
                   if k.startswith("net.")}
        if "raw_net" in ck:
            em.net.load_state_dict(ck["raw_net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        epoch, step, best = ck["epoch"], ck["step"], ck["best"]
        log(f"resumed at epoch {epoch} step {step}")
    elif a.init:
        sd = torch.load(a.init, map_location=dev, weights_only=False)["state_dict"]
        # A deterministic (v2) checkpoint has no noise-conditioning weights; they are
        # zero-initialized, so a stochastic model started from it is exactly that model.
        # Anything else missing or unexpected is still an error.
        missing, unexpected = em.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("corrector.")]  # never saved: built from meta
        noise_only = all(".W_scale." in k or ".W_bias." in k or "noise" in k for k in missing)
        if unexpected or not noise_only:
            raise RuntimeError(f"--init mismatch: missing {missing}, unexpected {unexpected}")
        if missing:
            log(f"--init: {len(missing)} noise-conditioning tensors start at zero (from a deterministic model)")
        if ema is not None:
            ema = {k: v.detach().clone().float() for k, v in em.net.state_dict().items()}
        log(f"initialized from {a.init}")

    net = DDP(em.net, device_ids=[local], broadcast_buffers=False) if ddp else em.net  # rank-0-only rollouts
    fwd = torch.compile(net) if a.compile else net

    def predict(x, ref=None, forcing_next=None):
        """physical (B, 39, H, W) -> (next prog physical, normalized output, ref dry mass).
        The output is corrected (if a corrector is attached) and re-normalized."""
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            d = fwd(em.normalize(x))
        nxt, diag = em.physical(x, d)
        if em.corrector is not None:
            nxt, diag, ref = em.corrector(x[:, :N_ALL], nxt, diag, ref, forcing_next)
            d = em.normalized(x, nxt, diag)
        return nxt, d.float(), ref

    def target_of(x, nxt_true):
        t = (nxt_true[:, :N_PROG] - x[:, :N_PROG]) / em.res_std
        return torch.cat([t, em.diag_target(nxt_true)], 1) if em.n_diag else t

    def members(traj):  # (B, ...) -> (B * n, ...), each sample's members contiguous
        return traj.repeat_interleave(n_ens, 0) if n_ens > 1 else traj

    def ens_gen(nxt, d, truth_now):
        """members' outputs, scored against a common reference (the true state at t):
        (B, n, C, H, W) normalized tendency w.r.t. truth (+ diagnostics)."""
        g = (nxt - members(truth_now[:, :N_PROG])) / em.res_std
        if em.n_diag:
            g = torch.cat([g, d[:, N_PROG:]], 1)
        return g.reshape(-1, n_ens, *g.shape[1:])

    def loss_fn(traj):
        if ens_loss is not None:
            tr = members(traj)
            state, loss, ref = tr[:, 0, :N_PROG], 0.0, None
            for t in range(a.n_steps):
                if a.input_noise > 0:
                    state = state + a.input_noise * em.std[:, :N_PROG] * torch.randn_like(state)
                x = torch.cat([state, tr[:, t, N_PROG:N_ALL]], 1)
                state, d, ref = predict(x, ref, tr[:, t + 1, N_PROG:N_ALL])
                gen = ens_gen(state, d, traj[:, t])
                tgt = target_of(traj[:, t], traj[:, t + 1])[:, None]
                loss = loss + ens_loss(gen, tgt)[0]
            return loss / a.n_steps
        state, loss, ref = traj[:, 0, :N_PROG], 0.0, None
        for t in range(a.n_steps):
            if a.input_noise > 0:  # perturb what the model reads; targets follow the perturbed state
                state = state + a.input_noise * em.std[:, :N_PROG] * torch.randn_like(state)
            x = torch.cat([state, traj[:, t, N_PROG:N_ALL]], 1)
            state, d, ref = predict(x, ref, traj[:, t + 1, N_PROG:N_ALL])
            loss = loss + ((d - target_of(x, traj[:, t + 1])) ** 2 * em.area_w).mean()
        return loss / a.n_steps

    def em_state(use_ema: bool):
        sd = {k: v for k, v in em.state_dict().items() if not k.startswith("corrector.")}
        if use_ema and ema is not None:
            sd.update({f"net.{k}": v for k, v in ema.items()})
        return sd

    class _ema_weights:  # validate with the EMA weights, then put the raw ones back
        def __enter__(self):
            if ema is not None:
                self.raw = {k: v.detach().clone() for k, v in em.net.state_dict().items()}
                em.net.load_state_dict(ema)

        def __exit__(self, *exc):
            if ema is not None:
                em.net.load_state_dict(self.raw)

    def save(name):
        if rank == 0:
            import dataclasses as _dc

            torch.save({"cfg": _dc.asdict(cfg), "state_dict": em_state(True),
                        "raw_net": em.net.state_dict() if ema is not None else None,
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
                        "step": step, "best": best, "args": vars(a),
                        "meta": {"names": meta["names"], "diag_names": diag_names,
                                 "lat": meta["lat"], "lon": meta["lon"]}}, out / name)

    ens_stats: dict = {}

    @torch.no_grad()
    def validate():
        em.eval()
        se = torch.zeros(N_PROG, device=dev, dtype=torch.float64)
        spread2 = torch.zeros((), device=dev, dtype=torch.float64)  # ensemble variance (normalized)
        err2 = torch.zeros((), device=dev, dtype=torch.float64)  # ensemble-mean error^2 (normalized)
        crps_sum = torch.zeros((), device=dev, dtype=torch.float64)
        lsum, n = 0.0, 0
        idx = range(rank, len(val), world)
        for i in range(0, len(idx), a.batch):
            traj = torch.stack([torch.from_numpy(val[j]) for j in list(idx)[i:i + a.batch]]).to(dev)
            x0 = traj[:, 0, :N_ALL]
            if ens_loss is not None:
                tr = members(traj)
                nxt, d, _ = predict(tr[:, 0, :N_ALL], None, tr[:, 1, N_PROG:N_ALL])
                gen = ens_gen(nxt, d, traj[:, 0])
                tgt = target_of(x0, traj[:, 1])[:, None]
                l, parts = ens_loss(gen, tgt)
                lsum += float(l) * len(traj)
                crps_sum += float(parts.get("crps", 0.0)) * len(traj)
                mean = gen.mean(1)
                spread2 += (gen.var(1) * em.area_w).mean().double() * len(traj)
                err2 += (((mean - tgt[:, 0]) ** 2) * em.area_w).mean().double() * len(traj)
                pred = nxt.reshape(-1, n_ens, *nxt.shape[1:]).mean(1)
            else:
                pred, d, _ = predict(x0, None, traj[:, 1, N_PROG:N_ALL])
                target = target_of(x0, traj[:, 1])
                lsum += float(((d - target) ** 2 * em.area_w).mean()) * len(traj)
            se += (((pred - traj[:, 1, :N_PROG]) ** 2) * em.area_w).mean((2, 3)).sum(0).double()
            n += len(traj)
        # rollouts (unbatched across ranks for simplicity: rank 0 only)
        rse = torch.zeros(a.rollout_steps, N_PROG, device=dev, dtype=torch.float64)
        nr = 0
        if rank == 0:
            for j in range(len(roll)):
                traj = members(torch.from_numpy(roll[j])[None].to(dev))  # n members of one start
                state, ref = traj[:, 0, :N_PROG], None
                for t in range(a.rollout_steps):
                    state, _, ref = predict(torch.cat([state, traj[:, t, N_PROG:N_ALL]], 1), ref,
                                            traj[:, t + 1, N_PROG:N_ALL])
                    ens_mean = state.mean(0, keepdim=True)  # the ensemble mean (one member: itself)
                    rse[t] += ((ens_mean - traj[:1, t + 1, :N_PROG]) ** 2 * em.area_w).mean((0, 2, 3)).double()
                nr += 1
        tot = torch.tensor([lsum, n], device=dev, dtype=torch.float64)
        ens = torch.stack([spread2, err2, crps_sum])
        if ddp:
            dist.all_reduce(tot)
            dist.all_reduce(se)
            dist.all_reduce(ens)
        ens_stats.clear()
        if ens_loss is not None:
            # unbiased-ish spread/skill: sqrt((n+1)/n * var) vs RMSE of the mean (Fortin et al. 2014)
            ratio = float(((n_ens + 1) / n_ens * ens[0] / ens[1]).sqrt())
            ens_stats.update(crps=float(ens[2] / tot[1]), spread_skill=ratio)
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
            if ema is not None:
                with torch.no_grad():
                    for k, v in em.net.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(a.ema).add_(v.float(), alpha=1 - a.ema)
                        else:
                            ema[k].copy_(v)
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
            if not (out / "best.pt").exists():  # a stage cut short before its first validation
                save("best.pt")
                log("no validated best.pt yet: saved these (unvalidated) weights as best.pt too")
            break
        epoch += 1
        with _ema_weights():
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
            if ens_stats:
                log(f"   ensemble ({n_ens}): 1-step CRPS {ens_stats['crps']:.4f}, "
                    f"spread/skill {ens_stats['spread_skill']:.3f}  (RMSEs above are of the mean)")
            with open(out / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"epoch": epoch, "step": step, "val_loss": vloss, "rmse_1step": rep,
                                    f"rmse_{a.rollout_steps}step": r5, **ens_stats}) + "\n")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
