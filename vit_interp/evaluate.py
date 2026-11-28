"""Does the planet look like Earth/E3SM? A basic skill evaluation of the ViT emulator.

    python -m vit_interp.evaluate CKPT OUTDIR [--year 2010] [--days 365] [--n-starts 8]
                                              [--lead-days 10] [--data DATA_ROOT]

1. A free-running rollout from <year>-01-01T00 with the true forcings (SOLIN, surface
   fractions) and prescribed ocean TS, streamed step by step from the memmap so a year fits
   on one GPU. Kept: time-mean maps, monthly zonal means, global-mean series and a few
   snapshots, for the emulator and for E3SM over the same steps. A non-finite state or a
   global-mean T_7 more than 30 K off stops the run and is recorded as a blow-up.
2. Short-range skill: --n-starts rollouts of --lead-days spread through the year,
   area-weighted RMSE and ACC against persistence and climatology. The climatology is the
   monthly mean of the training years (``meta.json`` ``stats_years``, 00Z and 12Z only),
   cached in DATA_ROOT/clim_monthly.npz; anomalies for ACC are taken against the
   climatology of the target month.

Writes PNG figures, metrics.json and summary.md into OUTDIR.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import ALL_NAMES, DATA_ROOT, N_ALL, N_PROG, PROG_NAMES, Stats
from .emulator import OCNFRAC, TS, Emulator

MAP_VARS = ["PS", "TS", "T_7", "T_3", "U_3", "V_3", "U_7", "specific_total_water_7"]
SKILL_VARS = ["PS", "T_7", "U_3", "V_3", "specific_total_water_7"]
SERIES_VARS = ["T_7", "PS", "specific_total_water_7", "U_3"]
SNAP_DAYS = [5, 30, 180]
UNITS = {"PS": "Pa", "TS": "K", "specific_total_water_7": "kg/kg"}


def _unit(v):
    return UNITS.get(v, "K" if v.startswith("T") else "m/s" if v[0] in "UV" else "")


def _weights(lat):
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    return w / w.mean()


def _months(times):
    return np.array([int(t[5:7]) for t in times])


def _times(data, year):
    return json.loads((Path(data) / f"{year}.times.json").read_text())


def _arr(data, year):
    return np.load(Path(data) / f"{year}.npy", mmap_mode="r")


# ---- climatology ------------------------------------------------------------------

def monthly_climatology(data, years, channels, stride=2):
    """(12, len(channels), H, W) monthly means over `years`, cached by channel list."""
    path = Path(data) / "clim_monthly.npz"
    if path.exists():
        z = np.load(path)
        names = list(z["names"])
        if all(c in names for c in channels) and list(z["years"]) == list(years):
            return z["clim"][:, [names.index(c) for c in channels]]
    idx = [ALL_NAMES.index(c) for c in channels]
    s = None
    n = np.zeros(12)
    for y in years:
        a, mon = _arr(data, y), _months(_times(data, y))
        for t in range(0, a.shape[0], stride):
            x = np.asarray(a[t, idx], dtype=np.float64)
            if s is None:
                s = np.zeros((12,) + x.shape)
            s[mon[t] - 1] += x
            n[mon[t] - 1] += 1
    clim = (s / np.maximum(n, 1)[:, None, None, None]).astype(np.float32)
    for m in range(12):  # months never seen (short test data) fall back to the overall mean
        if n[m] == 0:
            clim[m] = (s.sum(0) / n.sum()).astype(np.float32)
    try:
        np.savez(path, clim=clim, names=np.array(channels), years=np.array(years))
    except OSError:
        pass
    return clim


# ---- the emulator, one step at a time ------------------------------------------------

class Stepper:
    """One step of the emulator; with a corrector attached to ``em``, the corrected step, with
    the dry-air mass pinned to the value at the last ``reset`` (the start of a run)."""

    def __init__(self, em, device, smooth=None, jitter=None, hooks=None):
        self.em, self.device, self.smooth, self.jitter, self.hooks = em, device, smooth, jitter, hooks
        self.ref = None

    def reset(self):
        self.ref = None

    @torch.no_grad()
    def __call__(self, state, forcing, forcing_next=None):
        x = torch.cat([state, forcing], 1)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            xn = self.em.normalize(x)
            net = (lambda z: self.em.net(z, hooks=self.hooks)) if self.hooks else self.em.net
            d = self.jitter(net, xn) if self.jitter is not None else net(xn)
        nxt, diag = self.em.physical(x, d)
        if self.em.corrector is not None:
            nxt, diag, self.ref = self.em.corrector(x, nxt, diag, self.ref, forcing_next)
        if self.smooth is not None:
            nxt = self.smooth.apply(state, nxt)
        self.diag = diag  # the step's (corrected) diagnostics, physical units, or None
        return nxt


def _to(dev, a):
    return torch.from_numpy(np.array(a, dtype=np.float32)).to(dev)  # a writable copy of the memmap slice


def _prescribe(state, truth_next):
    ocean = truth_next[:, OCNFRAC] > 0.5
    state[:, TS] = torch.where(ocean, truth_next[:, TS], state[:, TS])
    return state


@torch.no_grad()
def free_run(step, a, times, n_steps, w, dev, log=print):
    """Stream a rollout of n_steps from a[0]; return accumulators (numpy)."""
    H, W = a.shape[2:]
    wt = torch.tensor(w / (w.sum() * W), device=dev, dtype=torch.float64)[:, None]  # sums to 1 on the grid
    gmean = lambda x: (x.double() * wt).sum((-2, -1))  # noqa: E731
    sum_e = torch.zeros(N_PROG, H, W, device=dev, dtype=torch.float64)
    sum_t = torch.zeros_like(sum_e)
    mon = _months(times)
    zm_e = torch.zeros(12, N_PROG, H, device=dev, dtype=torch.float64)
    zm_t = torch.zeros_like(zm_e)
    nm = np.zeros(12)
    ser_e = np.full((n_steps, N_PROG), np.nan)
    ser_t = np.full((n_steps, N_PROG), np.nan)
    snaps = {}
    snap_steps = {d * 4: d for d in SNAP_DAYS}
    state = _to(dev, a[0][None, :N_PROG])
    t7 = PROG_NAMES.index("T_7")
    blow = None
    done = 0
    t0 = time.time()
    for t in range(n_steps):
        forcing = _to(dev, a[t][None, N_PROG:N_ALL])
        truth = _to(dev, a[t + 1][None])
        state = _prescribe(step(state, forcing, truth[:, N_PROG:N_ALL]), truth)
        ge, gt = gmean(state[0]), gmean(truth[0, :N_PROG])
        if not torch.isfinite(state).all() or abs(float(ge[t7] - gt[t7])) > 30:
            blow = {"step": t + 1, "day": (t + 1) / 4, "time": times[t + 1],
                    "reason": "non-finite" if not torch.isfinite(state).all() else "global-mean T_7 off by >30 K"}
            log(f"blow-up at step {t + 1}: {blow['reason']}")
            break
        sum_e += state[0].double()
        sum_t += truth[0, :N_PROG].double()
        m = mon[t + 1] - 1
        zm_e[m] += state[0].double().mean(-1)
        zm_t[m] += truth[0, :N_PROG].double().mean(-1)
        nm[m] += 1
        ser_e[t], ser_t[t] = ge.cpu().numpy(), gt.cpu().numpy()
        if (t + 1) in snap_steps:
            snaps[snap_steps[t + 1]] = (state[0].cpu().numpy(), truth[0, :N_PROG].cpu().numpy())
        done = t + 1
        if (t + 1) % 400 == 0:
            log(f"  step {t + 1}/{n_steps}  T_7 global {float(ge[t7]):.2f} vs {float(gt[t7]):.2f} K  "
                f"({(time.time() - t0) / (t + 1):.3f} s/step)")
    k = max(done, 1)
    nmk = np.maximum(nm, 1)[:, None, None]
    return {
        "steps": done, "blowup": blow,
        "mean_e": (sum_e / k).cpu().numpy(), "mean_t": (sum_t / k).cpu().numpy(),
        "zm_e": zm_e.cpu().numpy() / nmk, "zm_t": zm_t.cpu().numpy() / nmk, "months_seen": nm,
        "ser_e": ser_e[:done], "ser_t": ser_t[:done], "snaps": snaps,
    }


@torch.no_grad()
def skill(step, a, times, starts, lead, w, clim, dev, members: int = 1):
    """RMSE/ACC vs lead (1..lead steps) for SKILL_VARS; model, persistence, climatology.
    With members > 1 (a stochastic model), RMSE/ACC are of the ensemble mean, and the CRPS
    (area-weighted, fair) and the ensemble spread (sqrt of the mean member variance) are added."""
    vi = [PROG_NAMES.index(v) for v in SKILL_VARS]
    mon = _months(times)
    wn = (w / w.sum())[:, None] / a.shape[3]
    out = {k: np.zeros((len(starts), lead, len(vi))) for k in
           ("rmse", "rmse_pers", "rmse_clim", "acc", "acc_pers", "crps", "spread")}
    n_ok = np.zeros(lead)

    def rmse(x, y):
        return np.sqrt(((x - y) ** 2 * wn).sum((-2, -1)))

    def acc(x, y, c):
        xa, ya = x - c, y - c
        xa = xa - (xa * wn).sum((-2, -1), keepdims=True)
        ya = ya - (ya * wn).sum((-2, -1), keepdims=True)
        return (xa * ya * wn).sum((-2, -1)) / np.sqrt((xa**2 * wn).sum((-2, -1)) * (ya**2 * wn).sum((-2, -1)))

    for i, s in enumerate(starts):
        if hasattr(step, "reset"):
            step.reset()
        state = _to(dev, a[s][None, :N_PROG]).expand(members, -1, -1, -1).contiguous()
        x0 = np.asarray(a[s, vi], dtype=np.float64)
        for l in range(lead):
            truth = _to(dev, a[s + l + 1][None])
            forcing = _to(dev, a[s + l][None, N_PROG:N_ALL]).expand(members, -1, -1, -1)
            state = _prescribe(step(state, forcing, truth[:, N_PROG:N_ALL].expand(members, -1, -1, -1)),
                               truth.expand(members, -1, -1, -1))
            ens = state[:, vi].double().cpu().numpy()  # (members, vars, H, W)
            pred = ens.mean(0)
            y = truth[0, vi].double().cpu().numpy()
            c = clim[mon[s + l + 1] - 1].astype(np.float64)
            if not np.isfinite(pred).all():
                out["rmse"][i, l:] = np.nan
                out["acc"][i, l:] = np.nan
                break
            out["rmse"][i, l] = rmse(pred, y)
            out["rmse_pers"][i, l] = rmse(x0, y)
            out["rmse_clim"][i, l] = rmse(c, y)
            out["acc"][i, l] = acc(pred, y, c)
            out["acc_pers"][i, l] = acc(x0, y, c)
            if members > 1:
                n = members
                pair = sum(np.abs(ens[p] - ens[q]) for p in range(n) for q in range(p + 1, n)) / (n * (n - 1) / 2)
                out["crps"][i, l] = ((np.abs(ens - y).mean(0) - 0.5 * pair) * wn).sum((-2, -1))
                out["spread"][i, l] = np.sqrt((ens.var(0, ddof=1) * wn).sum((-2, -1)))
            n_ok[l] += 1
    return {k: np.nanmean(v, 0) for k, v in out.items()}


# ---- figures -----------------------------------------------------------------------

def _map_axes(fig, n, m):
    try:
        import cartopy.crs as ccrs

        proj = ccrs.PlateCarree(central_longitude=180)
        axes = [fig.add_subplot(n, m, i + 1, projection=proj) for i in range(n * m)]
        return axes, ccrs.PlateCarree()
    except Exception:
        return [fig.add_subplot(n, m, i + 1) for i in range(n * m)], None


def _show(ax, f, lat, lon, tr, **kw):
    if tr is not None:  # pcolormesh, not imshow: cartopy's image warping needs scipy
        im = ax.pcolormesh(lon, lat, f, transform=tr, shading="auto", rasterized=True, **kw)
        try:
            ax.coastlines(lw=0.4)
        except Exception:  # no Natural Earth data offline: the map is still drawn
            pass
    else:
        im = ax.pcolormesh(lon, lat, f, shading="auto", rasterized=True, **kw)
    return im


def _triplet_rows(fig_path, rows, lat, lon, title):
    """rows: [(label, truth, emu)] -> columns E3SM | ViT | ViT - E3SM."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 2.6 * len(rows)))
    axes, tr = _map_axes(fig, len(rows), 3)
    for r, (label, t, e) in enumerate(rows):
        lo, hi = np.nanpercentile(t, [1, 99])
        d = e - t
        dm = np.nanpercentile(np.abs(d), 99) or 1.0
        for c, (f, kw, name) in enumerate([
            (t, dict(vmin=lo, vmax=hi, cmap="viridis"), "E3SM"),
            (e, dict(vmin=lo, vmax=hi, cmap="viridis"), "ViT"),
            (d, dict(vmin=-dm, vmax=dm, cmap="RdBu_r"), "ViT − E3SM"),
        ]):
            ax = axes[3 * r + c]
            im = _show(ax, f, lat, lon, tr, **kw)
            ax.set_title(f"{label} — {name}", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)


def figures(out, res, sk, lat, lon, lead_hours, year):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = PROG_NAMES.index
    days = res["steps"] / 4
    _triplet_rows(out / "maps_time_mean.png",
                  [(f"{v} [{_unit(v)}]", res["mean_t"][P(v)], res["mean_e"][P(v)]) for v in MAP_VARS],
                  lat, lon, f"Time mean over {days:.0f} days of {year}, free-running ViT vs E3SM")

    # zonal-mean cross sections (time mean over months seen, weighted by count)
    nm = res["months_seen"]
    wmon = nm / max(nm.sum(), 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for r, var in enumerate(["U", "T"]):
        ch = [P(f"{var}_{k}") for k in range(8)]
        t = (res["zm_t"][:, ch] * wmon[:, None, None]).sum(0)
        e = (res["zm_e"][:, ch] * wmon[:, None, None]).sum(0)
        lo, hi = np.percentile(t, [1, 99])
        d = e - t
        dm = np.abs(d).max() or 1.0
        for c, (f, kw, name) in enumerate([(t, dict(vmin=lo, vmax=hi, cmap="viridis"), "E3SM"),
                                           (e, dict(vmin=lo, vmax=hi, cmap="viridis"), "ViT"),
                                           (d, dict(vmin=-dm, vmax=dm, cmap="RdBu_r"), "ViT − E3SM")]):
            ax = axes[r, c]
            im = ax.imshow(f, aspect="auto", extent=[lat[0], lat[-1], 7.5, -0.5], interpolation="nearest", **kw)
            ax.set_title(f"zonal-mean {var} [{'m/s' if var == 'U' else 'K'}] — {name}", fontsize=9)
            ax.set_xlabel("latitude")
            ax.set_ylabel("model level (0 = top)")
            fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "zonal_mean_U_T.png", dpi=110)
    plt.close(fig)

    # seasonal cycle: zonal-mean T_7 by month
    seen = nm > 0
    if seen.sum() > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        t, e = res["zm_t"][seen, P("T_7")], res["zm_e"][seen, P("T_7")]
        months = np.arange(1, 13)[seen]
        for ax, f, name, kw in [(axes[0], t, "E3SM", {}), (axes[1], e, "ViT", {}),
                                (axes[2], e - t, "ViT − E3SM", {"cmap": "RdBu_r"})]:
            if name.startswith("ViT −"):
                dm = np.abs(f).max() or 1.0
                kw = dict(kw, vmin=-dm, vmax=dm)
            else:
                kw = dict(vmin=t.min(), vmax=t.max())
            im = ax.pcolormesh(months, lat, f.T, shading="nearest", **kw)
            ax.set_title(f"zonal-mean T_7 by month — {name}", fontsize=9)
            ax.set_xlabel("month")
            ax.set_ylabel("latitude")
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out / "seasonal_zonal_T7.png", dpi=110)
        plt.close(fig)

    # global-mean series
    tday = (np.arange(res["steps"]) + 1) / 4
    fig, axes = plt.subplots(1, len(SERIES_VARS), figsize=(4.5 * len(SERIES_VARS), 3.2))
    for ax, v in zip(axes, SERIES_VARS):
        ax.plot(tday, res["ser_t"][:, P(v)], label="E3SM", lw=1)
        ax.plot(tday, res["ser_e"][:, P(v)], label="ViT", lw=1)
        ax.set_title(f"global mean {v} [{_unit(v)}]", fontsize=9)
        ax.set_xlabel("day")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "global_mean_series.png", dpi=110)
    plt.close(fig)

    # snapshots
    rows = []
    for d, (e, t) in sorted(res["snaps"].items()):
        for v in ("T_7", "V_3"):
            rows.append((f"day {d} {v} [{_unit(v)}]", t[P(v)], e[P(v)]))
    if rows:
        _triplet_rows(out / "snapshots.png", rows, lat, lon,
                      "Free-running snapshots (weather is not expected to match after ~2 weeks)")

    # skill
    if sk is not None:
        fig, axes = plt.subplots(2, len(SKILL_VARS), figsize=(4.2 * len(SKILL_VARS), 6.5))
        for j, v in enumerate(SKILL_VARS):
            ax = axes[0, j]
            ax.plot(lead_hours / 24, sk["rmse"][:, j], label="ViT")
            ax.plot(lead_hours / 24, sk["rmse_pers"][:, j], "--", label="persistence")
            ax.plot(lead_hours / 24, sk["rmse_clim"][:, j], ":", label="climatology")
            ax.set_title(f"RMSE {v} [{_unit(v)}]", fontsize=9)
            ax = axes[1, j]
            ax.plot(lead_hours / 24, sk["acc"][:, j], label="ViT")
            ax.plot(lead_hours / 24, sk["acc_pers"][:, j], "--", label="persistence")
            ax.axhline(0.6, color="gray", lw=0.5)
            ax.set_ylim(-0.1, 1.0)
            ax.set_title(f"ACC {v}", fontsize=9)
            ax.set_xlabel("lead (days)")
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "skill_vs_lead.png", dpi=110)
        plt.close(fig)


# ---- driver ------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("out")
    p.add_argument("--year", type=int, default=2010)
    p.add_argument("--days", type=float, default=365)
    p.add_argument("--data", default=DATA_ROOT)
    p.add_argument("--filter", type=float, default=0.0,
                   help="seam filter: damp each step's increment above this zonal wavenumber (60 recommended)")
    p.add_argument("--shapiro", type=int, default=0, help="seam filter: 1-2-1 latitude passes (1 recommended)")
    p.add_argument("--jitter", default="", help="patch jitter mode (cycle16, lon, random); empty = off")
    p.add_argument("--n-starts", type=int, default=8)
    p.add_argument("--lead-days", type=float, default=10)
    p.add_argument("--members", type=int, default=1, help="ensemble members for the skill (stochastic models)")
    p.add_argument("--seed", type=int, default=None, help="seed the model's noise (stochastic models)")
    p.add_argument("--clim-years", default=None, help="e.g. 2000-2009; default meta stats_years")
    p.add_argument("--correct", default=None,
                   help="corrector flags (dry_air,positive,moisture,energy); default: what the "
                   "checkpoint was trained with; 'none' for none")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda *m: print(*m, flush=True)  # noqa: E731
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    meta = json.loads((Path(a.data) / "meta.json").read_text())
    lat, lon = np.asarray(meta["lat"]), np.asarray(meta["lon"])
    em, ck = Emulator.load(a.ckpt, Stats(a.data), lat, map_location=dev)
    em = em.to(dev).eval()
    flags = a.correct if a.correct is not None else (ck.get("args") or {}).get("corrector", "")
    flags = {f for f in (flags or "").split(",") if f and f != "none"}
    if flags:
        from .corrector import Corrector, CorrectorConfig, diag_names_of

        heat = float((ck.get("args") or {}).get("unaccounted_heating", 0.0))
        ccfg = CorrectorConfig(**{k: k in flags for k in ("dry_air", "positive", "moisture", "energy")},
                               unaccounted_heating=heat)
        from .data import vertical_coordinate

        ak, bk = vertical_coordinate(meta)
        em.corrector = Corrector(ccfg, lat, ak, bk,
                                 diag_names=diag_names_of(ck)).to(dev)
        log(f"corrector: {sorted(flags)}")
    if a.seed is not None:
        torch.manual_seed(a.seed)  # the stochastic model's noise draws
    smooth = None
    if a.filter or a.shapiro:
        from .filters import SeamFilter

        smooth = SeamFilter(lat, len(lon), a.filter, a.shapiro).to(dev)
        log(f"seam filter: zonal kc {a.filter}, {a.shapiro} latitude pass(es)")
    jit = None
    if a.jitter:
        from .filters import PatchJitter

        jit = PatchJitter(em.cfg.patch, a.jitter)
        log(f"patch jitter: {a.jitter}")
    step = Stepper(em, dev, smooth, jit)
    w = _weights(lat)
    arr, times = _arr(a.data, a.year), _times(a.data, a.year)
    n_steps = min(int(round(a.days * 4)), arr.shape[0] - 1)
    log(f"checkpoint epoch {ck.get('epoch')} step {ck.get('step')}; free run {n_steps} steps "
        f"({n_steps / 4:.1f} days) from {times[0]} on {dev}")

    t0 = time.time()
    step.reset()
    res = free_run(step, arr, times, n_steps, w, dev, log)
    log(f"free run: {res['steps']} steps in {time.time() - t0:.0f} s; blow-up: {res['blowup']}")

    # short-range skill
    lead = int(round(a.lead_days * 4))
    last = arr.shape[0] - 1 - lead
    sk = None
    if last >= 0 and a.n_starts > 0:
        starts = np.unique(np.linspace(0, last, a.n_starts).round().astype(int)).tolist()
        if a.clim_years:
            lo, hi = (int(v) for v in a.clim_years.split("-"))
            cy = list(range(lo, hi + 1))
        else:
            cy = meta.get("stats_years") or [y for y in meta["years"] if y != a.year]
        cy = [y for y in cy if (Path(a.data) / f"{y}.npy").exists()]
        t1 = time.time()
        clim = monthly_climatology(a.data, cy, SKILL_VARS)
        log(f"climatology from {cy[0]}-{cy[-1]} in {time.time() - t1:.0f} s")
        sk = skill(step, arr, times, starts, lead, w, clim, dev, members=a.members)
        sk["starts"] = [times[s] for s in starts]
    lead_hours = (np.arange(lead) + 1) * 6.0

    # metrics
    P = PROG_NAMES.index
    wn = (w / w.sum())[:, None] / len(lon)
    land = arr[0, ALL_NAMES.index("LANDFRAC")] > 0.5
    metrics = {"checkpoint": str(Path(a.ckpt).resolve()), "epoch": ck.get("epoch"), "step": ck.get("step"),
               "year": a.year, "steps_run": res["steps"], "days_run": res["steps"] / 4,
               "blowup": res["blowup"], "time_mean": {}, "global_mean_drift_last30d": {}}
    for v in MAP_VARS:
        e, t = res["mean_e"][P(v)], res["mean_t"][P(v)]
        m = {"rmse": float(np.sqrt(((e - t) ** 2 * wn).sum())), "bias": float(((e - t) * wn).sum()),
             "pattern_corr": float(np.corrcoef((e * np.sqrt(wn)).ravel(), (t * np.sqrt(wn)).ravel())[0, 1])}
        if v == "TS":
            wl = wn * land
            m["rmse_land"] = float(np.sqrt(((e - t) ** 2 * wl).sum() / wl.sum()))
            m["bias_land"] = float(((e - t) * wl).sum() / wl.sum())
        metrics["time_mean"][v] = m
    k = min(120, res["steps"])
    if k:
        for v in SERIES_VARS:
            metrics["global_mean_drift_last30d"][v] = float(
                np.mean(res["ser_e"][-k:, P(v)] - res["ser_t"][-k:, P(v)]))
    if sk is not None:
        pick = {f"{h // 24:g}d": int(h // 6) - 1 for h in (24, 72, 120, 240) if int(h // 6) - 1 < lead}
        pick.setdefault(f"{lead_hours[-1] / 24:g}d", lead - 1)
        metrics["skill"] = {
            v: {lab: {"rmse": float(sk["rmse"][i, j]), "rmse_persistence": float(sk["rmse_pers"][i, j]),
                      "rmse_climatology": float(sk["rmse_clim"][i, j]), "acc": float(sk["acc"][i, j]),
                      "acc_persistence": float(sk["acc_pers"][i, j]),
                      **({"crps": float(sk["crps"][i, j]), "spread": float(sk["spread"][i, j]),
                          "spread_skill": float(sk["spread"][i, j] / sk["rmse"][i, j])}
                         if a.members > 1 else {})} for lab, i in pick.items()}
            for j, v in enumerate(SKILL_VARS)}
        if a.members > 1:
            metrics["members"] = a.members
        metrics["skill_starts"] = sk["starts"]
        metrics["climatology"] = "monthly means of the training years (00Z and 12Z), anomalies vs target month"
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    np.savez_compressed(out / "fields.npz", mean_e=res["mean_e"], mean_t=res["mean_t"], zm_e=res["zm_e"],
                        zm_t=res["zm_t"], ser_e=res["ser_e"], ser_t=res["ser_t"], names=np.array(PROG_NAMES),
                        **({k: v for k, v in sk.items() if k != "starts"} if sk else {}))
    figures(out, res, sk, lat, lon, lead_hours, a.year)

    # summary
    L = [f"# ViT emulator evaluation, {a.year}", "",
         f"Checkpoint `{a.ckpt}` (epoch {ck.get('epoch')}, step {ck.get('step')}).", "",
         f"**Free run:** {res['steps']} steps ({res['steps'] / 4:.1f} days) from {times[0]}, true forcings, "
         f"ocean TS prescribed. " + ("Stable for the whole period." if not res["blowup"] else
                                     f"**Blew up** at day {res['blowup']['day']:.1f} ({res['blowup']['reason']})."),
         "", "| variable | time-mean RMSE | bias | pattern corr |", "|---|---|---|---|"]
    for v, m in metrics["time_mean"].items():
        extra = f" (land: RMSE {m['rmse_land']:.3g}, bias {m['bias_land']:.3g})" if "rmse_land" in m else ""
        L.append(f"| {v} [{_unit(v)}] | {m['rmse']:.3g}{extra} | {m['bias']:.3g} | {m['pattern_corr']:.3f} |")
    if metrics["global_mean_drift_last30d"]:
        L += ["", "Global-mean ViT − E3SM over the last 30 days: " + ", ".join(
            f"{v} {d:+.3g} {_unit(v)}" for v, d in metrics["global_mean_drift_last30d"].items())]
    if sk is not None:
        L += ["", f"**Short-range skill** ({len(sk['starts'])} starts, climatology = {metrics['climatology']}):", "",
              "| variable | lead | RMSE ViT | persistence | climatology | ACC ViT | ACC persistence |",
              "|---|---|---|---|---|---|---|"]
        for v, d in metrics["skill"].items():
            for lab, m in d.items():
                L.append(f"| {v} | {lab} | {m['rmse']:.3g} | {m['rmse_persistence']:.3g} | "
                         f"{m['rmse_climatology']:.3g} | {m['acc']:.3f} | {m['acc_persistence']:.3f} |")
        if a.members > 1:
            L += ["", f"**Ensemble** ({a.members} members; RMSE/ACC above are of the ensemble mean):", "",
                  "| variable | lead | CRPS | spread | spread/skill |", "|---|---|---|---|---|"]
            for v, d in metrics["skill"].items():
                for lab, m in d.items():
                    L.append(f"| {v} | {lab} | {m['crps']:.3g} | {m['spread']:.3g} | {m['spread_skill']:.2f} |")
    L += ["", "Figures: " + ", ".join(sorted(p.name for p in out.glob("*.png")))]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    log((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
