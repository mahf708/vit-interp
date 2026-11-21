"""Where and how does a free run go wrong? Reruns the evaluation's free run (same stepper,
correctors, prescribed ocean TS) and saves, once a day:

- the global-mean error of every channel;
- error maps of T_0, T_3, T_7, total water at levels 3 and 7, U_3 (float16);
- the zonal power spectrum of the emulated and true U_3, T_3 and water-3 fields (averaged over
  latitude), to catch energy piling up at small scales and at the 4° patch period (k = 90).

    python scripts/drift_probe.py --ckpt v4 --days 200 --out probe-v4.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vit_interp.data import N_ALL, N_PROG, PROG_NAMES, Stats, resolve_checkpoint
from vit_interp.emulator import Emulator
from vit_interp.evaluate import Stepper, _arr, _prescribe, _to, _weights

MAPS = ["T_0", "T_3", "T_7", "specific_total_water_3", "specific_total_water_7", "U_3"]
SPEC = ["U_3", "T_3", "specific_total_water_3"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="/lcrc/globalscratch/ac.ngmahfouz/vit-interp/data")
    p.add_argument("--year", type=int, default=2010)
    p.add_argument("--days", type=int, default=200)
    p.add_argument("--start", type=int, default=0, help="start index (6-hourly) in the year")
    p.add_argument("--correct", default=None, help="corrector flags; default: the checkpoint's")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--filter", type=float, default=0.0,
                   help="damp each step's increment above this zonal wavenumber (polar-scaled by cos lat); 0 = off")
    p.add_argument("--shapiro", type=int, default=0, help="1-2-1 passes in latitude on the increment")
    p.add_argument("--jitter", default="", help="patch jitter mode: cycle16, lon, random")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((Path(a.data) / "meta.json").read_text())
    lat = np.array(meta["lat"])
    em, ck = Emulator.load(resolve_checkpoint(a.ckpt), Stats(a.data), lat, map_location=dev)
    flags = a.correct if a.correct is not None else (ck.get("args") or {}).get("corrector", "")
    flags = {f for f in (flags or "").replace("+", ",").split(",") if f and f != "none"}
    if flags:  # exactly as vit_interp.evaluate
        from vit_interp.corrector import Corrector, CorrectorConfig, diag_names_of
        from vit_interp.data import vertical_coordinate

        heat = float((ck.get("args") or {}).get("unaccounted_heating", 0.0))
        ccfg = CorrectorConfig(**{k: k in flags for k in ("dry_air", "positive", "moisture", "energy")},
                               unaccounted_heating=heat)
        ak, bk = vertical_coordinate(meta)
        em.corrector = Corrector(ccfg, lat, ak, bk, diag_names=diag_names_of(ck)).to(dev)
    print(f"{a.ckpt}: corrector {sorted(flags)}, noise_dim {em.cfg.noise_dim}", flush=True)
    em = em.to(dev).eval()
    from vit_interp.filters import PatchJitter
    step = Stepper(em, dev, None, PatchJitter(em.cfg.patch, a.jitter) if a.jitter else None)
    filt = None
    if a.filter:  # a GCM-style polar/hyperdiffusion filter on the increment, outside the network
        k = torch.arange(len(meta["lon"]) // 2 + 1, device=dev, dtype=torch.float64)
        kc = torch.clamp(a.filter * torch.cos(torch.deg2rad(torch.tensor(lat, device=dev))), min=6.0)
        filt = torch.exp(-((k[None, :] / kc[:, None]) ** 8)).float()  # (H, K)

    def smooth(inc):
        if filt is not None:
            inc = torch.fft.irfft(torch.fft.rfft(inc, dim=-1) * filt, n=inc.shape[-1], dim=-1)
        for _ in range(a.shapiro):
            pad = torch.cat([inc[..., :1, :], inc, inc[..., -1:, :]], -2)
            inc = 0.25 * pad[..., :-2, :] + 0.5 * pad[..., 1:-1, :] + 0.25 * pad[..., 2:, :]
        return inc

    arr = _arr(a.data, a.year)
    w = _weights(lat)
    wt = torch.tensor(w / (w.sum() * arr.shape[3]), device=dev, dtype=torch.float64)[:, None]
    mi = [PROG_NAMES.index(n) for n in MAPS]
    si = [PROG_NAMES.index(n) for n in SPEC]
    n = min(a.days * 4, arr.shape[0] - 1 - a.start)
    g_err, maps, spec_e, spec_t = [], [], [], []
    state = _to(dev, arr[a.start][None, :N_PROG])
    t0 = time.time()
    for t in range(n):
        i = a.start + t
        forcing = _to(dev, arr[i][None, N_PROG:N_ALL])
        truth = _to(dev, arr[i + 1][None])
        nxt = step(state, forcing, truth[:, N_PROG:N_ALL])
        if a.filter or a.shapiro:
            nxt = state + smooth(nxt - state)
        state = _prescribe(nxt, truth)
        if not torch.isfinite(state).all():
            print(f"non-finite at step {t + 1}", flush=True)
            break
        if (t + 1) % 4:
            continue
        err = (state[0] - truth[0, :N_PROG]).double()
        g_err.append((err * wt).sum((-2, -1)).cpu().numpy())
        maps.append(err[mi].float().cpu().numpy().astype(np.float16))
        fe = torch.fft.rfft(state[0, si].double(), dim=-1).abs().pow(2)
        ft = torch.fft.rfft(truth[0, si].double(), dim=-1).abs().pow(2)
        cw = torch.tensor(w / w.sum(), device=dev, dtype=torch.float64)[None, :, None]
        spec_e.append((fe * cw).sum(1).cpu().numpy())
        spec_t.append((ft * cw).sum(1).cpu().numpy())
        d = (t + 1) // 4
        if d % 10 == 0:
            ge = g_err[-1]
            print(f"day {d}: T_3 {ge[PROG_NAMES.index('T_3')]:+.2f} K, q_3 "
                  f"{1e3 * ge[PROG_NAMES.index('specific_total_water_3')]:+.3f} g/kg, PS "
                  f"{ge[PROG_NAMES.index('PS')]:+.1f} Pa  max|dT_3| {float(err[PROG_NAMES.index('T_3')].abs().max()):.1f} K "
                  f"({(time.time() - t0) / (t + 1):.3f} s/step)", flush=True)
    np.savez_compressed(a.out, g_err=np.array(g_err), maps=np.array(maps), map_names=np.array(MAPS),
                        spec_e=np.array(spec_e), spec_t=np.array(spec_t), spec_names=np.array(SPEC),
                        names=np.array(PROG_NAMES), lat=lat, lon=np.array(meta["lon"]),
                        start=a.start, ckpt=str(a.ckpt), flags=",".join(sorted(flags)))
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
