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
    step = Stepper(em, dev, smooth)
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
            state = step(state, forcing, truth[:, N_PROG:N_ALL])
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
                        lat=lat, lon=np.asarray(meta["lon"]), years=a.years, blow=json.dumps(blow),
                        filter=a.filter, shapiro=a.shapiro, ckpt=str(a.ckpt))
    print(f"wrote {a.out}; blow-up: {blow}", flush=True)


if __name__ == "__main__":
    main()
