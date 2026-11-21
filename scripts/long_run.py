"""Multi-year free run: is the emulator stable for a decade? Carries the state across year files
(true forcings, ocean TS prescribed, the checkpoint's correctors, optional seam filter) and
saves daily global means (emulator and E3SM), the daily max |T_3 error|, the 4° patch-period
power ratio of T_3, and monthly-mean emulator − E3SM maps of a few fields.

    python scripts/long_run.py --ckpt v4 --years 2000-2009 --filter 60 --shapiro 1 --out long-v4f.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vit_interp.corrector import attach
from vit_interp.data import ALL_NAMES, N_ALL, N_PROG, PROG_NAMES, Stats, resolve_checkpoint
from vit_interp.emulator import Emulator
from vit_interp.evaluate import Stepper, _prescribe, _to, _weights
from vit_interp.filters import SeamFilter

LANDFRAC, TS = ALL_NAMES.index("LANDFRAC"), ALL_NAMES.index("TS")
MAPS = ["TS", "T_3", "T_7", "specific_total_water_7", "U_3", "PS"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="/lcrc/globalscratch/ac.ngmahfouz/vit-interp/data-v2")
    p.add_argument("--years", default="2000-2009")
    p.add_argument("--filter", type=float, default=0.0)
    p.add_argument("--shapiro", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prescribe", choices=["ocean", "nonland"], default="ocean",
                   help="where TS is prescribed: open ocean (OCNFRAC > 0.5, as evaluate) or every non-land cell (sea ice too)")
    p.add_argument("--clamp-ts", default="", help="e.g. 180,340: clip emulated TS to this range (K)")
    p.add_argument("--damp", type=float, default=0.0,
                   help="increment damping for every channel: state + (1 - damp) * increment")
    p.add_argument("--damp-top", type=float, default=None, help="damping for the top --damp-top-levels levels")
    p.add_argument("--damp-top-levels", type=int, default=2)
    p.add_argument("--damp-small", type=float, default=0.0,
                   help="scale-selective damping: remove this fraction of the increment's small-scale part "
                        "(what a kc=30, 2-pass smoother takes out); --damp-small-top for the top levels")
    p.add_argument("--damp-small-top", type=float, default=None)
    p.add_argument("--jitter", default="", help="patch jitter mode: cycle16, lon, random")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    meta = json.loads((Path(a.data) / "meta.json").read_text())
    lat = np.asarray(meta["lat"])
    em, ck = Emulator.load(resolve_checkpoint(a.ckpt), Stats(a.data), lat, map_location=dev)
    em = em.to(dev).eval()
    attach(em, ck, meta)
    smooth = SeamFilter(lat, len(meta["lon"]), a.filter, a.shapiro).to(dev) if (a.filter or a.shapiro) else None
    from vit_interp.filters import PatchJitter
    step = Stepper(em, dev, smooth, PatchJitter(em.cfg.patch, a.jitter) if a.jitter else None)
    scale = None
    if a.damp or a.damp_top:
        sc = torch.full((N_PROG,), 1.0 - a.damp)
        if a.damp_top is not None:
            for i, nme in enumerate(PROG_NAMES):
                if "_" in nme and nme.rsplit("_", 1)[1].isdigit() and int(nme.rsplit("_", 1)[1]) < a.damp_top_levels:
                    sc[i] = 1.0 - a.damp_top
        scale = sc.to(dev)[None, :, None, None]
        print("increment scale:", {n: round(float(v), 3) for n, v in zip(PROG_NAMES, sc) if v != 1.0}, flush=True)
    small = small_frac = None
    if a.damp_small or a.damp_small_top:
        small = SeamFilter(lat, len(meta["lon"]), 30.0, 2).to(dev)
        fr = torch.full((N_PROG,), a.damp_small)
        if a.damp_small_top is not None:
            for i, nme in enumerate(PROG_NAMES):
                if "_" in nme and nme.rsplit("_", 1)[1].isdigit() and int(nme.rsplit("_", 1)[1]) < a.damp_top_levels:
                    fr[i] = a.damp_small_top
        small_frac = fr.to(dev)[None, :, None, None]
        print("small-scale damping:", {n: round(float(v), 2) for n, v in zip(PROG_NAMES, fr) if v}, flush=True)
    # diagnostics: day-to-day variability per channel, and Nino-3.4 box means (SST response)
    dd_e = torch.zeros(N_PROG, device=dev, dtype=torch.float64); dd_t = torch.zeros_like(dd_e); ndd = 0
    prev_e = prev_t = None
    la_, lo_ = torch.tensor(lat), torch.tensor(np.asarray(meta["lon"]))
    box = ((la_.abs() <= 5)[:, None] & ((lo_ >= 190) & (lo_ <= 240))[None, :]).to(dev)
    bw = (box * torch.tensor(_weights(lat), device=dev)[:, None]).double(); bw = bw / bw.sum()
    NINO = [PROG_NAMES.index(n) for n in ("T_7", "U_7", "specific_total_water_7", "T_3", "U_3")]
    nino_e, nino_t = [], []
    ts_lo, ts_hi = (float(x) for x in a.clamp_ts.replace(":", ",").split(",")) if a.clamp_ts else (None, None)
    y0, y1 = (int(x) for x in a.years.split("-"))
    w = _weights(lat)
    wt = torch.tensor(w / (w.sum() * len(meta["lon"])), device=dev, dtype=torch.float64)[:, None]
    cw = torch.tensor(w / w.sum(), device=dev, dtype=torch.float64)[:, None]
    gmean = lambda x: (x.double() * wt).sum((-2, -1))  # noqa: E731
    mi, i3 = [PROG_NAMES.index(n) for n in MAPS], PROG_NAMES.index("T_3")
    g_e, g_t, maxerr, k90 = [], [], [], []
    mon_maps, acc, nacc = [], None, 0
    state, t0, k, blow = None, time.time(), 0, None
    for y in range(y0, y1 + 1):
        arr = np.load(Path(a.data) / f"{y}.npy", mmap_mode="r")
        if state is None:
            state = _to(dev, arr[0][None, :N_PROG])
        n = arr.shape[0] - 1
        for t in range(n):
            forcing = _to(dev, arr[t][None, N_PROG:N_ALL])
            truth = _to(dev, arr[t + 1][None, :N_ALL])
            prev_state = state
            state = step(state, forcing, truth[:, N_PROG:N_ALL])
            if scale is not None:
                state = prev_state + scale * (state - prev_state)
            if small is not None:
                inc = state - prev_state
                low = small(inc)
                state = prev_state + low + (1 - small_frac) * (inc - low)
            if a.prescribe == "nonland":
                sea = truth[:, LANDFRAC] < 0.5
                state[:, TS] = torch.where(sea, truth[:, TS], state[:, TS])
            else:
                state = _prescribe(state, truth)
            if ts_lo is not None:
                state[:, TS] = state[:, TS].clamp(ts_lo, ts_hi)
            k += 1
            if not torch.isfinite(state).all():
                blow = {"step": k, "year": y}
                break
            if k % 4:
                continue
            err = (state[0] - truth[0, :N_PROG]).double()
            se, st = state[0].double(), truth[0, :N_PROG].double()
            if prev_e is not None:
                dd_e += gmean((se - prev_e) ** 2); dd_t += gmean((st - prev_t) ** 2); ndd += 1
            prev_e, prev_t = se, st
            nino_e.append((se[NINO] * bw).sum((-2, -1)).cpu().numpy()); nino_t.append((st[NINO] * bw).sum((-2, -1)).cpu().numpy())
            g_e.append(gmean(state[0]).cpu().numpy()); g_t.append(gmean(truth[0, :N_PROG]).cpu().numpy())
            maxerr.append(float(err[i3].abs().max()))
            fe = (torch.fft.rfft(state[0, i3].double(), dim=-1).abs() ** 2 * cw).sum(0)
            ft = (torch.fft.rfft(truth[0, i3].double(), dim=-1).abs() ** 2 * cw).sum(0)
            k90.append(float(fe[88:93].sum() / ft[88:93].sum()))
            acc = err[mi] if acc is None else acc + err[mi]
            nacc += 1
            if nacc == 30:
                mon_maps.append((acc / nacc).float().cpu().numpy().astype(np.float16)); acc, nacc = None, 0
        if blow:
            break
        # the next year's file starts where this one ends: keep the emulated state
        ge = g_e[-1] - g_t[-1]
        print(f"{y} done ({k} steps, {(time.time() - t0) / k:.3f} s/step): global T_7 {ge[PROG_NAMES.index('T_7')]:+.2f} K, "
              f"T_3 {ge[i3]:+.2f} K, PS {ge[PROG_NAMES.index('PS')]:+.0f} Pa, max|dT_3| {maxerr[-1]:.0f} K, "
              f"k90 x{k90[-1]:.0f}", flush=True)
    np.savez_compressed(a.out, g_e=np.array(g_e), g_t=np.array(g_t), maxerr=np.array(maxerr), k90=np.array(k90),
                        mon_maps=np.array(mon_maps), map_names=np.array(MAPS), names=np.array(PROG_NAMES),
                        lat=lat, lon=np.asarray(meta["lon"]),
                        dd_e=(dd_e / max(ndd, 1)).sqrt().cpu().numpy(), dd_t=(dd_t / max(ndd, 1)).sqrt().cpu().numpy(),
                        nino_e=np.array(nino_e), nino_t=np.array(nino_t),
                        nino_names=np.array(["T_7", "U_7", "specific_total_water_7", "T_3", "U_3"]),
                        damp_small=a.damp_small, damp=a.damp, damp_top=-1 if a.damp_top is None else a.damp_top, years=a.years, blow=json.dumps(blow),
                        filter=a.filter, shapiro=a.shapiro, ckpt=str(a.ckpt))
    print(f"wrote {a.out}; blow-up: {blow}", flush=True)


if __name__ == "__main__":
    main()
