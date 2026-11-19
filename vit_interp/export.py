"""Write the ViT's residual stream as an exege latent archive (one node per patch token).

    python -m vit_interp.export CKPT OUT --year 2010 --start 0 --n 120 [--every 1]
    exege latents info OUT
    exege nn sae OUT --out OUT/bases/sae-l6.npz ...

Layer index i = blocks.{i}.hook_resid_post (index `depth` is not written; the embedding is
index -> "embed" as an extra first layer when --embed is given). Reference fields are the
physical fields averaged onto the same patch grid, at the input time of each forward call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from exege.adapters.latent_archive import write_archive
from exege.latents.grid import Grid

from .data import ALL_NAMES, DATA_ROOT, Stats, Trajectories
from .emulator import Emulator
from .model import token_latlon

REF = ["PS", "TS", "T_7", "T_3", "U_3", "V_3", "U_6", "V_6", "specific_total_water_7", "SOLIN"]


def coarsen(x: np.ndarray, p: int) -> np.ndarray:
    """(T, H, W) -> (T, H/p * W/p) patch means, row-major like the tokens."""
    T, H, W = x.shape
    return x.reshape(T, H // p, p, W // p, p).mean((2, 4)).reshape(T, -1)


@torch.no_grad()
def export(ckpt, out, year=2010, start=0, n=120, every=1, layers=None, embed=False,
           data=DATA_ROOT, device=None, batch=8, overwrite=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((Path(data) / "meta.json").read_text())
    em, ck = Emulator.load(ckpt, Stats(data), meta["lat"], map_location=device)
    em = em.to(device).eval()
    cfg = em.cfg
    layers = list(range(cfg.depth)) if layers is None else list(layers)
    names = {f"blocks.{i}.hook_resid_post" for i in layers} | ({"hook_embed", "hook_pos_embed"} if embed else set())

    src = Trajectories([year], 0, data)
    times_all = src.times(year)
    idx = list(range(start, min(start + n * every, len(src)), every))
    acc = {k: [] for k in names}
    states = []
    for b in range(0, len(idx), batch):
        x = torch.stack([torch.from_numpy(src[i][0]) for i in idx[b:b + batch]]).to(device)
        cache: dict = {}
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device != "cpu"):
            em.step(x, cache=cache, names=names)
        for k in names:
            acc[k].append(cache[k].float().cpu().numpy().astype(np.float16))
        states.append(x.cpu().numpy())
    X = np.concatenate(states)
    lat, lon, shape = token_latlon(meta["lat"], meta["lon"], cfg.patch)
    grid = Grid(lat=lat.astype(np.float64), lon=lon.astype(np.float64), shape=shape)
    out_layers, net_layers = [], []
    if embed:
        e = np.concatenate(acc["hook_embed"]) + np.concatenate(acc["hook_pos_embed"])
        out_layers.append(("embed + pos", e))
        net_layers.append(0)
    for i in layers:
        out_layers.append((f"blocks.{i}.resid_post", np.concatenate(acc[f"blocks.{i}.hook_resid_post"])))
        net_layers.append(i + 1)
    fields = {n_: coarsen(X[:, ALL_NAMES.index(n_)], cfg.patch).astype(np.float32) for n_ in REF}
    times = [times_all[i] for i in idx]
    return write_archive(
        out, grid=grid, times=times, layers=out_layers, network_layers=net_layers,
        fields=fields, field_times=times, model="vit-interp E3SMv3-1deg-AMIP",
        component="atmosphere", checkpoint=str(Path(ckpt).resolve()), calendar="noleap",
        timestep_seconds=21600,
        experiment={"patch": cfg.patch, "dim": cfg.dim, "depth": cfg.depth, "heads": cfg.heads,
                    "epoch": ck.get("epoch"), "step": ck.get("step")},
        overwrite=overwrite,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("out")
    p.add_argument("--year", type=int, default=2010)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--layers", default=None, help="comma list, default all")
    p.add_argument("--embed", action="store_true")
    p.add_argument("--data", default=DATA_ROOT)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    lay = [int(v) for v in a.layers.split(",")] if a.layers else None
    print(export(a.ckpt, a.out, a.year, a.start, a.n, a.every, lay, a.embed, a.data, overwrite=a.overwrite))
