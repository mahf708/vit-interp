"""The network's raw (pre-corrector) global-mean tendency bias per level: the mean, over many states, of
the area-weighted global mean of [raw 6-h tendency - true 6-h tendency] for T and water at each level.
Uncorrected steps, noise zero. This is the bias the energy corrector has been hiding (see the v8 drift).

    python scripts/raw_bias.py --ckpts v8,/path/v9-ctrl/best.pt --years 2010,2011 --out raw_bias.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vit_interp import playground as pg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", required=True)
    p.add_argument("--names", default="")
    p.add_argument("--years", default="2010,2011,2012")
    p.add_argument("--stride", type=int, default=30, help="every stride-th 6-h state (30: every 7.5 days)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    lat, _ = pg.lat_lon()
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 360))
    w = torch.tensor(w / w.sum(), dtype=torch.float32)
    T = [pg.ch(f"T_{k}") for k in range(8)]
    Q = [pg.ch(f"specific_total_water_{k}") for k in range(8)]
    names = a.names.split(",") if a.names else a.ckpts.split(",")
    res = {"years": a.years, "stride": a.stride, "models": {}}
    for nm, ck in zip(names, a.ckpts.split(",")):
        em, _ = pg.load(ck)
        dev = next(em.parameters()).device
        ww = w.to(dev)
        bT, bQ, rT, n = torch.zeros(8, device=dev), torch.zeros(8, device=dev), torch.zeros(8, device=dev), 0
        for y in (int(v) for v in a.years.split(",")):
            for t in range(0, 1459, a.stride):
                x = pg.state(t, year=y).float()
                tr = pg.truth(t, 1, year=y).float()[0]
                _, d = pg.step(em, x)
                true = tr[1, :pg.N_PROG] - tr[0, :pg.N_PROG]
                e = d[0] - true
                bT += (e[T] * ww).sum((-2, -1)); bQ += (e[Q] * ww).sum((-2, -1))
                rT += ((e[T] ** 2) * ww).sum((-2, -1)); n += 1
        bT, bQ, rT = (bT / n).cpu().numpy(), (bQ / n).cpu().numpy(), (rT / n).sqrt().cpu().numpy()
        res["models"][nm] = {"T_bias_K_per_6h": bT.tolist(), "q_bias_per_6h": bQ.tolist(), "T_rmse_K": rT.tolist(), "n": n}
        print(nm, "raw T bias K/6h by level:", np.round(bT, 4), "| T rmse", np.round(rT, 3), flush=True)
        del em
        torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
