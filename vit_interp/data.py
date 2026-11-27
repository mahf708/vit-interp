"""E3SMv3 1-degree AMIP data: monthly netCDF -> per-year float32 memmaps, stats, and a Dataset.

Layout under DATA_ROOT:
    {year}.npy          (T, C, 180, 360) float32, channels in meta["names"] order: the
                        model inputs (ALL_NAMES: prognostic, then forcing), then, in a
                        "diag" dataset, the diagnostics (DIAG_NAMES) the model may predict
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
_LCRC_DATA = "/lcrc/globalscratch/ac.ngmahfouz/vit-interp/data"
HF_DATA_REPO = "mahf708/vit-interp-e3sm-2010"  # the 2010 inputs, for use off LCRC
HF_MODEL_REPO = "mahf708/vit-interp-e3sm"  # released checkpoints: v1/v1.pt, v2/v2.pt


def _default_data_root() -> str:
    """$VIT_INTERP_DATA, else the LCRC copy when it is there, else a local cache that
    ``ensure_data`` fills from Hugging Face."""
    if os.environ.get("VIT_INTERP_DATA"):
        return os.environ["VIT_INTERP_DATA"]
    if (Path(_LCRC_DATA) / "meta.json").exists():
        return _LCRC_DATA
    cache = Path(os.environ.get("VIT_INTERP_CACHE", Path.home() / ".cache" / "vit-interp"))
    return str(cache / "data-2010")


DATA_ROOT = _default_data_root()


def ensure_data(root: str = DATA_ROOT, year: int = 2010) -> str:
    """Make sure ``root`` holds the metadata, stats and ``year``; off LCRC, download the 2010
    inputs (about 15 GB) from Hugging Face the first time."""
    r = Path(root)
    if (r / "meta.json").exists() and (r / f"{year}.npy").exists():
        return str(r)
    if year != 2010:
        raise FileNotFoundError(f"{r} has no {year}.npy, and only 2010 is published")
    from huggingface_hub import snapshot_download

    print(f"downloading the 2010 inputs from huggingface.co/datasets/{HF_DATA_REPO} into {r} (~15 GB, once)")
    snapshot_download(HF_DATA_REPO, repo_type="dataset", local_dir=str(r))
    return str(r)


def resolve_checkpoint(spec: str | Path) -> Path:
    """A checkpoint file, or ``v1``..``v4``, ``v7``, ``v8`` (also ``hf:v2``): the released checkpoints, from the
    LCRC release directory when it is there, else downloaded from Hugging Face."""
    s = str(spec)
    if Path(s).is_file():
        return Path(s)
    name = s.removeprefix("hf:")
    if name in ("v1", "v2", "v3", "v4", "v7", "v8"):
        lcrc = Path("/lcrc/globalscratch/ac.ngmahfouz/vit-interp/runs/vit-p4-d384-l8/release") / f"{name}.pt"
        if lcrc.is_file():
            return lcrc
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(HF_MODEL_REPO, f"{name}/{name}.pt"))
    raise FileNotFoundError(f"no checkpoint {s!r} (a path, or v1 / v2 / v3 / v4 / v7 / v8)")

LEVELS = range(8)  # 0 = model top ... 7 = near surface
PROG_NAMES = (
    ["PS", "TS"]
    + [f"T_{k}" for k in LEVELS]
    + [f"U_{k}" for k in LEVELS]
    + [f"V_{k}" for k in LEVELS]
    + [f"specific_total_water_{k}" for k in LEVELS]
)
FORCING_NAMES = ["SOLIN", "PHIS", "LANDFRAC", "OCNFRAC", "ICEFRAC"]
ALL_NAMES = PROG_NAMES + FORCING_NAMES  # what the network reads
# Diagnostics ACE also predicts (fluxes over the 6 h ending at the output time). Never inputs.
DIAG_NAMES = [
    "surface_precipitation_rate",
    "LHFLX",
    "SHFLX",
    "surface_upward_longwave_flux",
    "surface_upward_shortwave_flux",
    "FLDS",
    "FSDS",
    "FLUT",
    "top_of_atmos_upward_shortwave_flux",
    "tendency_of_total_water_path_due_to_advection",
]
_FIRST_STEP_UNDEFINED = {"tendency_of_total_water_path_due_to_advection"}
N_PROG = len(PROG_NAMES)
N_ALL = len(ALL_NAMES)
N_DIAG = len(DIAG_NAMES)
DATA_ROOT_V2 = "/lcrc/globalscratch/ac.ngmahfouz/vit-interp/data-v2"


def _read_year(year: int, arr_path: Path, max_steps: int | None = None, names=ALL_NAMES):
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
    arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.float32, shape=(T, len(names), 180, 360))
    s = 0
    for f, n in zip(files, sizes):
        n = min(n, T - s)
        if n <= 0:
            break
        with netCDF4.Dataset(f) as d:
            d.set_auto_mask(False)
            for c, name in enumerate(names):
                v = d[name]
                block = v[:n] if v.dimensions[0] == "time" else v[:][None]
                if name in _FIRST_STEP_UNDEFINED and n > 1 and not np.isfinite(block[0]).all():
                    # a time-difference residual: undefined at each monthly file's first step
                    # (its previous step is in the previous file). Persist the next step's.
                    block = np.array(block)
                    block[0] = block[1]
                arr[s : s + n, c] = block
        arr.flush()  # write each month out: dirty mapped pages count against the job's memory
        s += n
    return arr, times[:T], lat


def _prep_year(year: int, out: str, block: int = 32, max_steps: int | None = None,
               names=ALL_NAMES) -> str:
    path = Path(out) / f"{year}.npy"
    tmp = Path(out) / f"{year}.partial.npy"
    arr, times, lat = _read_year(year, tmp, max_steps, names)
    C = len(names)
    T = arr.shape[0]
    for s0 in range(0, T, 124):
        if not np.isfinite(arr[s0 : s0 + 2]).all():
            raise ValueError(f"{year}: non-finite values near step {s0}")
    arr.flush()

    # area-weighted partial sums for stats (float64)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 360))
    w = w / w.sum()
    s1, s2, r1, r2 = (np.zeros(C) for _ in range(4))
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


def prep(years: list[int], stats_years: list[int], out: str = DATA_ROOT, workers: int = 12,
         diag: bool = False) -> None:
    names = ALL_NAMES + (DIAG_NAMES if diag else [])
    Path(out).mkdir(parents=True, exist_ok=True)
    todo = [y for y in years if not (Path(out) / f"{y}.npy").exists()]
    with ProcessPoolExecutor(workers) as ex:
        jobs = [ex.submit(_prep_year, y, out, 32, None, names) for y in todo]  # 32 steps: ~3 GB/worker in the stats pass
        for job in jobs:
            msg = job.result()
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
    np.savez(Path(out) / "stats.npz", mean=mean, std=std, res_std=rstd, names=np.array(names))
    import netCDF4

    with netCDF4.Dataset(Path(NC_DIR) / f"{stats_years[0]}-01.nc") as d:
        lat, lon = np.asarray(d["lat"][:]), np.asarray(d["lon"][:])
        ak = [float(d[f"ak_{k}"][...]) for k in range(9)]  # hybrid interfaces, top to surface:
        bk = [float(d[f"bk_{k}"][...]) for k in range(9)]  # p_k = ak_k + bk_k * PS
    meta = dict(
        names=names,
        diag_names=DIAG_NAMES if diag else [],
        ak=ak,
        bk=bk,
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
    for name, m, s, r in zip(names, mean, std, rstd):
        print(f"{name:26s} mean {m: .4e} std {s:.4e} res_std {r:.4e}")


def vertical_coordinate(meta: dict) -> tuple[list[float], list[float]]:
    """(ak, bk) hybrid interface coefficients, top to surface; from meta, or (older datasets
    without them) from the first netCDF file."""
    if "ak" in meta and "bk" in meta:
        return list(meta["ak"]), list(meta["bk"])
    import netCDF4

    with netCDF4.Dataset(sorted(Path(NC_DIR).glob("*.nc"))[0]) as d:
        return ([float(d[f"ak_{k}"][...]) for k in range(9)],
                [float(d[f"bk_{k}"][...]) for k in range(9)])


class Stats:
    def __init__(self, root: str = DATA_ROOT):
        st = np.load(Path(root) / "stats.npz")
        # inputs: the first N_ALL channels; diagnostics (if any) after them
        self.mean = st["mean"][:N_ALL].astype(np.float32)
        self.std = st["std"][:N_ALL].astype(np.float32)
        self.res_std = st["res_std"][:N_PROG].astype(np.float32)
        self.diag_mean = st["mean"][N_ALL:].astype(np.float32)
        self.diag_std = st["std"][N_ALL:].astype(np.float32)


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
    p.add_argument("--diag", action="store_true", help="also store DIAG_NAMES (v2 datasets)")
    a = p.parse_args()

    def rng(s):
        lo, hi = (int(v) for v in s.split("-"))
        return list(range(lo, hi + 1))

    prep(rng(a.years), rng(a.stats_years), a.out, a.workers, a.diag)
