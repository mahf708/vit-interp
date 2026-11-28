"""Paired climate-profile response on 2010 states, with common noise and jitter (drafted by a GPT-6.2 reviewer, reviewed here).

Plants an energy-neutral temperature-profile perturbation (default: the shape of the decade drift) at +a and -a,
steps both and a control with identical noise and jitter, and reports the projection of the difference onto the
mode vs lead (gain 1: no restoring; 0: restored; >1: amplified), intact and with components ablated.

    python scripts/profile_response.py --n-starts 8 --days 30 --ablate L7H3,L6MLP --out response.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from vit_interp.corrector import attach
from vit_interp.data import ALL_NAMES, N_ALL, N_PROG, PROG_NAMES, Stats, resolve_checkpoint
from vit_interp.emulator import Emulator
from vit_interp.evaluate import Stepper, _prescribe, _to, _weights
from vit_interp.filters import PatchJitter, SeamFilter

T_NAMES = [f"T_{k}" for k in range(8)]
T_IDX = [PROG_NAMES.index(n) for n in T_NAMES]
PHIS = ALL_NAMES.index("PHIS")
ROOT = Path("/lcrc/globalscratch/ac.ngmahfouz/vit-interp")


def drift_mode(path):
    with np.load(path) as z:
        idx = [list(z["names"]).index(n) for n in T_NAMES]
        d = z["g_e"][:, idx] - z["g_t"][:, idx]
    if len(d) < 10 * 365 - 10:
        raise ValueError("need a completed decade run")
    # First/last 365 daily samples (also works with old skipped-boundary runs).
    return d[-365:].mean(0) - d[:365].mean(0)


def neutral_mode(em, x, raw):
    """Remove uniform dT to hold the corrector's global energy fixed.

    At fixed PS and q, total_energy_path is linear in T. Normalize separately
    at each start so --amp is the maximum initial absolute dT in K.
    """
    c = em.corrector
    state, phis = x[:, :N_PROG].double(), x[:, PHIS].double()
    m = torch.as_tensor(raw, device=x.device, dtype=torch.float64)
    trial = state.clone()
    trial[:, T_IDX] += m[None, :, None, None]
    de = c.gmean(c.total_energy_path(trial, phis) - c.total_energy_path(state, phis))
    m = m - de[0] / c.gmean(c._energy_factor(state))[0]
    scale = m.abs().max()
    if not torch.isfinite(scale) or scale < 1e-9:
        raise ValueError("mode has no energy-neutral contrast")
    return m / scale


def ablation(em, name, keep):
    """Same per-level spatial mean ablation as long_run.py; one component."""
    match = re.fullmatch(r"L(\d+)(MLP|H(\d+))", name)
    if not match or int(match[1]) >= em.cfg.depth:
        raise ValueError(f"invalid component {name}")
    layer = int(match[1])
    head = None if match[2] == "MLP" else int(match[3])
    if head is not None and head >= em.cfg.heads:
        raise ValueError(f"invalid head {name}")
    key = f"blocks.{layer}." + ("hook_mlp_out" if head is None else "attn.hook_result")

    def hook(t):
        t = t.clone()
        v = t if head is None else t[..., head, :]
        m = v.mean(-2, keepdim=True)
        y = m + keep * (v - m)
        if head is None:
            return y
        t[..., head, :] = y
        return t

    return {key: hook}


@torch.no_grad()
def rollout(em, arr, start, days, mode, amp, seed, smooth, jitter, hooks, w):
    dev = next(em.parameters()).device
    # Identical random stream and fresh corrector/jitter state for each arm.
    torch.manual_seed(seed)
    step = Stepper(em, dev, smooth,
                   PatchJitter(em.cfg.patch, jitter, seed=seed) if jitter else None, hooks)
    state = _to(dev, arr[start][None, :N_PROG])
    state[:, T_IDX] += (amp * mode).to(state.dtype)[None, :, None, None]
    series = np.full((4 * days + 1, len(T_IDX)), np.nan)

    def record(lead):
        series[lead] = (state[0, T_IDX].double() * w).sum((-2, -1)).cpu().numpy()

    record(0)
    for t in range(start, start + 4 * days):
        forcing = _to(dev, arr[t][None, N_PROG:N_ALL])
        truth = _to(dev, arr[t + 1][None, :N_ALL])
        state = _prescribe(step(state, forcing, truth[:, N_PROG:N_ALL]), truth)
        if not torch.isfinite(state).all():
            break
        record(t - start + 1)
    return series


def json_array(x):
    """Strict JSON: uncompleted/non-finite leads are null, not NaN."""
    x = np.asarray(x)
    return np.where(np.isfinite(x), x, None).tolist()


def summary(y):
    y = np.asarray(y)
    n = np.isfinite(y).sum(0)
    mean = np.divide(np.nansum(y, axis=0), n, out=np.full(y.shape[1:], np.nan), where=n > 0)
    return {"mean": json_array(mean), "n": n.tolist(), "per_start": json_array(y)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="v8")
    p.add_argument("--data", default=str(ROOT / "data-v2"))
    p.add_argument("--year", type=int, default=2010)
    p.add_argument("--n-starts", type=int, default=8)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--amp", type=float, default=1.0)
    p.add_argument("--mode", help="eight comma-separated dT values, top to bottom")
    p.add_argument("--drift", default=str(ROOT / "drift/long-f3d-rope-jit-f.npz"))
    p.add_argument("--filter", type=float, default=60)
    p.add_argument("--shapiro", type=int, default=1)
    p.add_argument("--jitter", default="cycle16", choices=["", "cycle16", "cycle16d", "lon", "random"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ablate", default="", help="comma-separated components, each tested separately")
    p.add_argument("--ablate-scale", type=float, default=0.0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.days < 1 or a.n_starts < 1 or not np.isfinite(a.amp) or a.amp <= 0 or a.seed < 0:
        p.error("positive days, n-starts, amp and nonnegative seed required")
    if not 0 <= a.ablate_scale <= 1:
        p.error("ablate-scale must be in [0, 1]")
    raw = np.array([float(v) for v in a.mode.split(",")]) if a.mode else drift_mode(a.drift)
    if raw.shape != (8,) or not np.isfinite(raw).all() or np.ptp(raw) < 1e-9:
        p.error("mode must contain eight finite, nonconstant values")
    dev = torch.device("cuda")
    meta = json.loads((Path(a.data) / "meta.json").read_text())
    lat = np.asarray(meta["lat"])
    em, ck = Emulator.load(resolve_checkpoint(a.ckpt), Stats(a.data), lat, map_location=dev)
    em = em.to(dev).eval()
    attach(em, ck, meta)
    if em.corrector is None:
        p.error("need a corrector for energy-neutral perturbations")
    smooth = SeamFilter(lat, len(meta["lon"]), a.filter, a.shapiro).to(dev) if (a.filter or a.shapiro) else None
    arr = np.load(Path(a.data) / f"{a.year}.npy", mmap_mode="r")
    times = json.loads((Path(a.data) / f"{a.year}.times.json").read_text())
    last = len(arr) - 1 - 4 * a.days
    if last < 0 or a.n_starts > last + 1:
        p.error("too many starts or too long a rollout for one year")
    starts = np.linspace(0, last, a.n_starts).round().astype(int).tolist()
    w = _weights(lat)
    w = torch.tensor(w / (w.sum() * arr.shape[-1]), device=dev, dtype=torch.float64)[:, None]
    modes = [neutral_mode(em, _to(dev, arr[s][None, :N_ALL]), raw) for s in starts]
    variants = {"intact": None}
    for name in filter(None, a.ablate.split(",")):
        variants[name] = ablation(em, name, a.ablate_scale)
    result = {"args": vars(a), "levels": T_NAMES, "raw_mode": raw.tolist(),
              "modes": [m.cpu().tolist() for m in modes], "lead_days": (np.arange(4 * a.days + 1) / 4).tolist(),
              "starts": [times[s] for s in starts], "variants": {}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    intact = None
    for name, hooks in variants.items():
        plus, minus = [], []
        for i, (s, mode) in enumerate(zip(starts, modes)):
            args = (a.seed + i, smooth, a.jitter, hooks, w)
            ctrl, pos, neg = [rollout(em, arr, s, a.days, mode, amp, *args)
                              for amp in (0.0, a.amp, -a.amp)]
            m = mode.cpu().numpy()
            project = lambda d: d @ m / (m @ m)  # noqa: E731
            plus.append(project(pos - ctrl))
            minus.append(project(neg - ctrl))
            print(name, times[s], flush=True)
        plus, minus = np.array(plus), np.array(minus)
        gain = (plus - minus) / (2 * a.amp)
        entry = {"plus_K": summary(plus), "minus_K": summary(minus),
                 "gain": summary(gain), "even_K": summary((plus + minus) / 2)}
        if name == "intact":
            intact = gain
        else:
            entry["gain_minus_intact"] = summary(gain - intact)
        result["variants"][name] = entry
        out.write_text(json.dumps(result, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
