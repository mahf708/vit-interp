"""E3SMv3 1-degree AMIP data: monthly netCDF -> per-year float32 memmaps, stats, and a Dataset.

Layout under DATA_ROOT:
    {year}.npy          (T, C, 180, 360) float32, channels in ALL_NAMES order
    {year}.times.json   ISO time labels (noleap calendar) for axis 0
    stats.npz           area-weighted mean/std (full field) and residual std, per channel
    meta.json           names, lat, lon, years
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# The sibling amip151_1951_2015.zarr is incomplete (unwritten chunks read as NaN); use the netCDF.
NC_DIR = "/lcrc/group/e3sm/ac.ngmahfouz/fme/amip_151/traindata"
DATA_ROOT = "/lcrc/globalscratch/ac.ngmahfouz/vit-interp/data"

LEVELS = range(8)  # 0 = model top ... 7 = near surface
PROG_NAMES = (
    ["PS", "TS"]
    + [f"T_{k}" for k in LEVELS]
    + [f"U_{k}" for k in LEVELS]
    + [f"V_{k}" for k in LEVELS]
    + [f"specific_total_water_{k}" for k in LEVELS]
)
FORCING_NAMES = ["SOLIN", "PHIS", "LANDFRAC", "OCNFRAC", "ICEFRAC"]
ALL_NAMES = PROG_NAMES + FORCING_NAMES
N_PROG = len(PROG_NAMES)
N_ALL = len(ALL_NAMES)


def _read_year(year: int, arr_path: Path, max_steps: int | None = None):
    import cftime
    import netCDF4

    files = [Path(NC_DIR) / f"{year}-{m:02d}.nc" for m in range(1, 13)]
    sizes, times = [], []
    for f in files:
        with netCDF4.Dataset(f) as d:
            t = d["time"]
            sizes.append(t.shape[0])
            times += list(cftime.num2date(t[:], t.units, t.calendar))
            lat = np.asarray(d["lat"][:])
    T = sum(sizes) if max_steps is None else min(sum(sizes), max_steps)
    arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.float32, shape=(T, N_ALL, 180, 360))
    s = 0
    for f, n in zip(files, sizes):
        n = min(n, T - s)
        if n <= 0:
            break
        with netCDF4.Dataset(f) as d:
            d.set_auto_mask(False)
            for c, name in enumerate(ALL_NAMES):
                v = d[name]
                arr[s : s + n, c] = v[:n] if v.dimensions[0] == "time" else v[:][None]
        s += n
    return arr, times[:T], lat


def _prep_year(year: int, out: str, block: int = 248, max_steps: int | None = None) -> str:
    path = Path(out) / f"{year}.npy"
    tmp = Path(out) / f"{year}.partial.npy"
    arr, times, lat = _read_year(year, tmp, max_steps)
    T = arr.shape[0]
    if not np.isfinite(arr[:: max(1, T // 50)]).all():
        raise ValueError(f"{year}: non-finite values")
    arr.flush()

    # area-weighted partial sums for stats (float64)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 360))
    w = w / w.sum()
    s1 = np.zeros(N_ALL)
    s2 = np.zeros(N_ALL)
    r1 = np.zeros(N_ALL)
    r2 = np.zeros(N_ALL)
    for s in range(0, T, block):
        x = np.asarray(arr[s : min(s + block + 1, T)], dtype=np.float64)
        xs = x[: min(block, T - s)]
        s1 += np.einsum("tchw,hw->c", xs, w)
        s2 += np.einsum("tchw,hw->c", xs**2, w)
        d = np.diff(x, axis=0)
        r1 += np.einsum("tchw,hw->c", d, w)
        r2 += np.einsum("tchw,hw->c", d**2, w)
    np.savez(Path(out) / f"{year}.sums.npz", s1=s1, s2=s2, r1=r1, r2=r2, n=T, nr=T - 1)
    times = [str(t.isoformat()) for t in times]
    (Path(out) / f"{year}.times.json").write_text(json.dumps(times))
    del arr
    os.replace(tmp, path)
    return f"{year}: {T} steps"


def prep(years: list[int], stats_years: list[int], out: str = DATA_ROOT, workers: int = 12) -> None:
    Path(out).mkdir(parents=True, exist_ok=True)
    todo = [y for y in years if not (Path(out) / f"{y}.npy").exists()]
    with ProcessPoolExecutor(workers) as ex:
        for msg in ex.map(_prep_year, todo, [out] * len(todo)):
            print(msg, flush=True)
    sums = [np.load(Path(out) / f"{y}.sums.npz") for y in stats_years]
    n = sum(float(s["n"]) for s in sums)
    nr = sum(float(s["nr"]) for s in sums)
    mean = sum(s["s1"] for s in sums) / n
    std = np.sqrt(np.maximum(sum(s["s2"] for s in sums) / n - mean**2, 0))
    rmean = sum(s["r1"] for s in sums) / nr
    rstd = np.sqrt(np.maximum(sum(s["r2"] for s in sums) / nr - rmean**2, 0))
    std = np.where(std < 1e-8, 1.0, std)  # e.g. constant fields
    rstd = np.where(rstd < 1e-12, 1.0, rstd)
    np.savez(Path(out) / "stats.npz", mean=mean, std=std, res_std=rstd, names=np.array(ALL_NAMES))
    import netCDF4

    with netCDF4.Dataset(Path(NC_DIR) / f"{stats_years[0]}-01.nc") as d:
        lat, lon = np.asarray(d["lat"][:]), np.asarray(d["lon"][:])
    meta = dict(
        names=ALL_NAMES,
        prog_names=PROG_NAMES,
        forcing_names=FORCING_NAMES,
        lat=lat.tolist(),
        lon=lon.tolist(),
        years=sorted(int(p.stem) for p in Path(out).glob("[0-9]*.npy")),
        stats_years=stats_years,
        calendar="noleap",
        timestep_seconds=21600,
    )
    (Path(out) / "meta.json").write_text(json.dumps(meta))
    for name, m, s, r in zip(ALL_NAMES, mean, std, rstd):
        print(f"{name:26s} mean {m: .4e} std {s:.4e} res_std {r:.4e}")


class Stats:
    def __init__(self, root: str = DATA_ROOT):
        st = np.load(Path(root) / "stats.npz")
        self.mean = st["mean"].astype(np.float32)
        self.std = st["std"].astype(np.float32)
        self.res_std = st["res_std"][:N_PROG].astype(np.float32)


class Trajectories:
    """Windows of n_steps+1 consecutive 6-hourly states, never crossing a year file."""

    def __init__(self, years, n_steps: int = 1, root: str = DATA_ROOT, stride: int = 1):
        self.root, self.years, self.n_steps = root, list(years), n_steps
        self._arrs: dict[int, np.ndarray] = {}
        self.index = []
        for y in self.years:
            T = len(json.loads((Path(root) / f"{y}.times.json").read_text()))
            self.index += [(y, t) for t in range(0, T - n_steps, stride)]

    def arr(self, y: int) -> np.ndarray:
        if y not in self._arrs:  # opened lazily so each DataLoader worker has its own maps
            self._arrs[y] = np.load(Path(self.root) / f"{y}.npy", mmap_mode="r")
        return self._arrs[y]

    def times(self, y: int) -> list[str]:
        return json.loads((Path(self.root) / f"{y}.times.json").read_text())

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> np.ndarray:
        y, t = self.index[i]
        return np.array(self.arr(y)[t : t + self.n_steps + 1])  # a writable copy


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--years", default="2000-2010")
    p.add_argument("--stats-years", default="2000-2009")
    p.add_argument("--out", default=DATA_ROOT)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()

    def rng(s):
        lo, hi = (int(v) for v in s.split("-"))
        return list(range(lo, hi + 1))

    prep(rng(a.years), rng(a.stats_years), a.out, a.workers)
