"""Fingerprints of emergent dynamics: plant a counterfactual on real 2010 states, step the
emulator with and without it (noise held at zero, raw network, no correctors unless asked),
and compare the response with a theoretical prediction.

    python scripts/emergent.py lamb --ckpt v8 --out runs/f3d-rope/lab/emergent

Probes:
  lamb   a surface-pressure bump (Gaussian, 200 Pa, 600 km) over open ocean. A real atmosphere
         carries it away as a Lamb wave at ~310 m/s (Hunga Tonga, 2022): a ring ~6700 km out after
         one 6-h step. We read the azimuthally averaged PS response vs great-circle distance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vit_interp import playground as pg

R_EARTH = 6.371e6


def _gc_dist(lat, lon, lat0, lon0):
    """Great-circle distance (m) from (lat0, lon0) to every grid point."""
    la, lo = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
    a0, o0 = np.deg2rad(lat0), np.deg2rad(lon0)
    c = np.sin(la) * np.sin(a0) + np.cos(la) * np.cos(a0) * np.cos(lo - o0)
    return R_EARTH * np.arccos(np.clip(c, -1, 1))


def _radial(field, dist, area, edges):
    """Area-weighted mean of field in distance rings."""
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (dist >= a) & (dist < b)
        out.append(float((field * area)[m].sum() / area[m].sum()) if m.any() else float("nan"))
    return np.array(out)


@torch.no_grad()
def _chain(em, x, steps, correct):
    """Prognostic states after 1..steps raw steps (forcings held at the start's: a probe, not a forecast)."""
    out, cur = [], x.clone()
    forcing = x[:, pg.N_PROG:]
    ref = None
    for _ in range(steps):
        if correct and em.corrector is not None:
            nxt, _, ref = em.step_corrected(cur, ref, noise=pg._noise(em, cur.shape[0], cur.device))
        else:
            nxt, _ = pg.step(em, cur)
        out.append(nxt)
        cur = torch.cat([nxt, forcing], 1)
    return out


def lamb(em, out: Path, args) -> dict:
    lat, lon = pg.lat_lon()
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(lon)[None, :]
    centers = [(0.0, 200.0), (-20.0, 240.0), (30.0, 170.0), (-40.0, 300.0)]  # open ocean
    starts = list(range(0, 1440, 180))  # 8 states of 2010
    edges = np.arange(0, 20.0e6 + 1, 400e3)
    mid = 0.5 * (edges[1:] + edges[:-1])
    ps = pg.ch("PS")
    prof = {k: [] for k in range(1, args.steps + 1)}
    maps = {}
    for t in starts:
        x = pg.state(t).float()
        for (la0, lo0) in centers:
            d = _gc_dist(lat, lon, la0, lo0)
            bump = args.amp * np.exp(-0.5 * (d / args.width) ** 2)
            xp = x.clone()
            xp[0, ps] += torch.tensor(bump, dtype=xp.dtype, device=xp.device)
            a, b = _chain(em, x, args.steps, args.correct), _chain(em, xp, args.steps, args.correct)
            for k in range(1, args.steps + 1):
                # response of the state relative to the planted bump's own persistence
                r = (b[k - 1][0, ps] - a[k - 1][0, ps]).cpu().numpy()
                prof[k].append(_radial(r, d, area, edges))
                if t == starts[0] and (la0, lo0) == centers[0]:
                    maps[k] = r.astype(np.float32)
    res = {"probe": "lamb", "ckpt": args.ckpt, "amp_Pa": args.amp, "width_m": args.width, "centers": centers,
           "starts": starts, "dist_km": (mid / 1e3).tolist(), "correct": args.correct, "steps": {}}
    for k, rows in prof.items():
        m = np.nanmean(np.array(rows), 0)
        # the ring: the largest positive response beyond 2 bump widths
        far = mid > 2 * args.width
        i = int(np.nanargmax(np.where(far, m, -np.inf)))
        res["steps"][k] = {"profile_Pa": m.tolist(), "ring_km": float(mid[i] / 1e3),
                           "ring_Pa": float(m[i]), "center_Pa": float(m[0]),
                           "implied_speed_m_s": float(mid[i] / (k * 21600.0)),
                           "spread_Pa": np.nanstd(np.array(rows), 0).tolist()}
        print(f"lamb step {k}: centre {m[0]:+.1f} Pa, ring at {mid[i] / 1e3:.0f} km ({m[i]:+.2f} Pa) "
              f"-> {mid[i] / (k * 21600.0):.0f} m/s", flush=True)
    np.savez_compressed(out / f"lamb-{args.ckpt}{'-corr' if args.correct else ''}-maps.npz",
                        **{f"step{k}": v for k, v in maps.items()}, lat=lat, lon=lon)
    return res


def adjust(em, out: Path, args) -> dict:
    """Geostrophic adjustment: the same PS bump at latitudes -60..60 (lon 190-200, open ocean). Theory:
    the part that survives adjustment is balanced, so the wind around a high turns anticyclonic,
    clockwise in the north (counter-clockwise tangential wind < 0) and counter-clockwise in the south
    (> 0), growing with |f|; on the equator nothing is retained and nothing rotates. We read the
    tangential wind response (ring 300-1500 km) at levels 7, 5, 3 and the PS kept at the centre."""
    lat, lon = pg.lat_lon()
    lats = [-60, -45, -30, -15, -5, 0, 5, 15, 30, 45, 60]
    starts = list(range(0, 1440, 180))
    ps = pg.ch("PS")
    rows = []
    for la0 in lats:
        lo0 = 190.0 if abs(la0) >= 60 else 200.0
        d = _gc_dist(lat, lon, la0, lo0)
        dy = np.deg2rad(lat - la0)[:, None] * R_EARTH * np.ones_like(lon)[None, :]
        dlon = ((lon - lo0 + 180) % 360 - 180)[None, :]
        dx = np.deg2rad(dlon) * R_EARTH * np.cos(np.deg2rad(lat))[:, None]
        rr = np.hypot(dx, dy) + 1.0
        ring = (d > 300e3) & (d < 1500e3)
        w = (np.cos(np.deg2rad(lat))[:, None] * np.ones_like(lon)[None, :]) * ring
        cen = d < 300e3
        wc = np.cos(np.deg2rad(lat))[:, None] * cen
        bump = args.amp * np.exp(-0.5 * (d / args.width) ** 2)
        acc = {k: {"vt": {lv: [] for lv in (7, 5, 3)}, "ps_kept": []} for k in range(1, args.steps + 1)}
        for t in starts:
            x = pg.state(t).float()
            xp = x.clone()
            xp[0, ps] += torch.tensor(bump, dtype=xp.dtype, device=xp.device)
            a, b = _chain(em, x, args.steps, args.correct), _chain(em, xp, args.steps, args.correct)
            for k in range(1, args.steps + 1):
                dlt = (b[k - 1] - a[k - 1])[0].cpu().numpy()
                acc[k]["ps_kept"].append(float((dlt[ps] * wc).sum() / wc.sum() /
                                               ((bump * wc).sum() / wc.sum())))
                for lv in (7, 5, 3):
                    u, v = dlt[pg.ch(f"U_{lv}")], dlt[pg.ch(f"V_{lv}")]
                    vt = (-dy * u + dx * v) / rr  # counter-clockwise tangential wind
                    acc[k]["vt"][lv].append(float((vt * w).sum() / w.sum()))
        row = {"lat": la0, "lon": lo0, "f": 2 * 7.292e-5 * float(np.sin(np.deg2rad(la0)))}
        for k, v in acc.items():
            row[f"step{k}"] = {"ps_kept": float(np.mean(v["ps_kept"])),
                               "vt_ccw": {lv: float(np.mean(x)) for lv, x in v["vt"].items()},
                               "vt_ccw_sd": {lv: float(np.std(x)) for lv, x in v["vt"].items()}}
        rows.append(row)
        s1 = row["step1"]
        print(f"adjust lat {la0:+3d}: PS kept {s1['ps_kept']:.2f}, ccw wind L7 {s1['vt_ccw'][7]:+.2f} "
              f"L5 {s1['vt_ccw'][5]:+.2f} L3 {s1['vt_ccw'][3]:+.2f} m/s", flush=True)
    return {"probe": "adjust", "ckpt": args.ckpt, "amp_Pa": args.amp, "width_m": args.width,
            "correct": args.correct, "rows": rows}


OMEGA, RD, DT = 7.292e-5, 287.04, 21600.0


def _ablation_hooks(em, spec: str):
    """Mean-ablate components at every step, e.g. "v8-vertical" (all heads 4-5) or "L7H3,L6MLP"."""
    if not spec:
        return None
    names = ([f"L{i}H{h}" for i in range(em.cfg.depth) for h in (4, 5)] if spec == "vertical" else
             [f"L{i}H{h}" for i in range(em.cfg.depth) for h in range(4)] if spec == "horizontal" else spec.split(","))
    by: dict = {}
    for n in names:
        if n.endswith("MLP"):
            by.setdefault(f"blocks.{int(n[1:-3])}.hook_mlp_out", None)
        else:
            i, h = n[1:].split("H")
            by.setdefault(f"blocks.{int(i)}.attn.hook_result", []).append(int(h))

    def make(hs):
        def f(t):
            t = t.clone()
            sl = (Ellipsis, hs, slice(None)) if hs is not None else (Ellipsis,)
            tok = -3 if hs is not None else -2
            t[sl] = t[sl].mean(tok, keepdim=True).expand_as(t[sl])
            return t
        return f
    return {k: make(v) for k, v in by.items()}


def _clamp_hooks(em, x, spec: str):
    """Hooks that replace chosen heads' outputs with their values on the UNPERTURBED state x, so a paired
    run removes only those heads' response to the perturbation (activation patching, not mean ablation)."""
    if not spec:
        return None
    heads = (4, 5) if spec == "vertical" else (0, 1, 2, 3) if spec == "horizontal" else None
    if heads is None:
        raise ValueError("clamp: vertical | horizontal")
    names = {f"blocks.{i}.attn.hook_result" for i in range(em.cfg.depth)}
    cache: dict = {}
    pg.step(em, x, cache=cache, names=names)

    def make(ref):
        def f(t):
            t = t.clone()
            t[..., list(heads), :] = ref[..., list(heads), :].to(t.dtype)
            return t
        return f
    return {n: make(cache[n]) for n in names}


@torch.no_grad()
def _pair(em, x, delta: dict, hooks=None, clamp: str = ""):
    """Linear tendency response to a planted perturbation: ([step(x+d)-(x+d)] - [step(x-d)-(x-d)]) / 2."""
    if clamp:
        hooks = {**(hooks or {}), **_clamp_hooks(em, x, clamp)}
    xp, xm = x.clone(), x.clone()
    for name, arr in delta.items():
        t = torch.as_tensor(arr, dtype=x.dtype, device=x.device)
        xp[0, pg.ch(name)] += t
        xm[0, pg.ch(name)] -= t
    a, _ = pg.step(em, xp, hooks=hooks)
    b, _ = pg.step(em, xm, hooks=hooks)
    r = 0.5 * (a - b)[0].cpu().numpy()
    for name, arr in delta.items():
        r[pg.ch(name)] -= arr
    return r


def _rotate(fx, fy, f, t=DT):
    """Exact response of a resting parcel to a constant force (fx, fy) with Coriolis f over t:
    w = u + i v, dw/dt = F - i f w  ->  w(t) = F (1 - exp(-i f t)) / (i f)."""
    F = fx + 1j * fy
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.abs(f) > 1e-9, F * (1 - np.exp(-1j * f * t)) / (1j * f), F * t)
    return w.real, w.imag


def _proj(field, env, s, lon, area):
    """Complex projection on env(lat) * exp(i s lon)."""
    ph = np.exp(-1j * s * np.deg2rad(lon))[None, :]
    w = env[:, None] * area
    return complex(2 * (field * ph * w).sum() / (env[:, None] ** 2 * area).sum())  # amplitude of a cos(s lon) wave


def inertial(em, out: Path, args) -> dict:
    """A zonally uniform U band (all levels) at latitude phi0: over one step Coriolis should turn it into
    dV = -A sin(f dt) and dU = A (cos(f dt) - 1)."""
    lat, lon = pg.lat_lon()
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(lon)[None, :]
    hooks = _ablation_hooks(em, args.ablate)
    starts = list(range(0, 1440, 180))
    rows = []
    for la0 in (-60, -45, -30, -15, 0, 15, 30, 45, 60):
        env = np.exp(-0.5 * ((lat - la0) / 6.0) ** 2)
        g = env[:, None] * np.ones_like(lon)[None, :]
        A = args.wind
        delta = {f"U_{k}": A * g for k in range(8)}
        f = 2 * OMEGA * np.sin(np.deg2rad(lat))[:, None]
        tu, tv = _rotate(np.zeros_like(g), np.zeros_like(g), f)  # no force: rotate the initial wind
        th_u = A * g * (np.cos(f * DT) - 1)
        th_v = -A * g * np.sin(f * DT)
        w = g * area
        th = {"dU": float((th_u * w).sum() / (A * g * w).sum()), "dV": float((th_v * w).sum() / (A * g * w).sum())}
        mod = {lv: {"dU": [], "dV": []} for lv in (2, 3, 4, 5)}
        for t in starts:
            r = _pair(em, pg.state(t).float(), delta, hooks, args.clamp)
            for lv in mod:
                mod[lv]["dU"].append(float((r[pg.ch(f"U_{lv}")] * w).sum() / (A * g * w).sum()))
                mod[lv]["dV"].append(float((r[pg.ch(f"V_{lv}")] * w).sum() / (A * g * w).sum()))
        row = {"lat": la0, "theory": th,
               "model": {lv: {k: float(np.mean(v)) for k, v in d.items()} for lv, d in mod.items()},
               "model_sd": {lv: {k: float(np.std(v)) for k, v in d.items()} for lv, d in mod.items()}}
        rows.append(row)
        m3 = row["model"][3]
        print(f"inertial {la0:+3d}: theory dV {th['dV']:+.2f} dU {th['dU']:+.2f} | model L3 dV {m3['dV']:+.2f} "
              f"dU {m3['dU']:+.2f}", flush=True)
    return {"probe": "inertial", "ckpt": args.ckpt, "ablate": args.ablate, "clamp": args.clamp, "wind": args.wind, "rows": rows}


def hydrostatic(em, out: Path, args) -> dict:
    """Warm one level m (a wave s=4 in a band at 45N, nothing else changed). Geopotential is not a model
    variable: to get the wind response right the network must integrate temperature upward. Theory:
    dPhi_k = R_d sum_{m>k} dT_m dln p_m + R_d dT_k * (half of dln p_k); the response of V_k is the exact
    Coriolis-rotated response to -grad dPhi_k over 6 h. Fills an 8x8 (observed level x warmed level) matrix."""
    lat, lon = pg.lat_lon()
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(lon)[None, :]
    md = pg.meta()
    ak, bk = np.array(md["ak"]), np.array(md["bk"])
    pi = ak + bk * 1.0e5
    dlnp = np.diff(np.log(np.maximum(pi, 1.0)))
    hooks = _ablation_hooks(em, args.ablate)
    s, la0, sig, A = 4, (45.0 if args.lat is None else args.lat), 10.0, args.temp
    env = np.exp(-0.5 * ((lat - la0) / sig) ** 2)
    wave = env[:, None] * np.cos(s * np.deg2rad(lon))[None, :]
    f = 2 * OMEGA * np.sin(np.deg2rad(lat))[:, None]
    a = R_EARTH
    dphi_dlam = np.gradient(wave, np.deg2rad(lon), axis=1)
    dphi_dlat = np.gradient(wave, np.deg2rad(lat), axis=0)
    coslat = np.maximum(np.cos(np.deg2rad(lat)), 1e-3)[:, None]
    starts = list(range(0, 1440, 180))
    th = np.zeros((8, 8), complex); mo = np.zeros((8, 8), complex); mo_u = np.zeros((8, 8), complex)
    th_u = np.zeros((8, 8), complex)
    for m in range(8):
        for k in range(8):
            weight = dlnp[m] if m > k else (0.5 * dlnp[m] if m == k else 0.0)
            c = RD * A * weight
            fx, fy = -c * dphi_dlam / (a * coslat), -c * dphi_dlat / a
            du, dv = _rotate(fx, fy, f)
            th[k, m] = _proj(dv, env, s, lon, area)
            th_u[k, m] = _proj(du, env, s, lon, area)
        acc_v = np.zeros(8, complex); acc_u = np.zeros(8, complex)
        for t in starts:
            r = _pair(em, pg.state(t).float(), {f"T_{m}": A * wave}, hooks)
            for k in range(8):
                acc_v[k] += _proj(r[pg.ch(f"V_{k}")], env, s, lon, area) / len(starts)
                acc_u[k] += _proj(r[pg.ch(f"U_{k}")], env, s, lon, area) / len(starts)
        mo[:, m], mo_u[:, m] = acc_v, acc_u
        print(f"hydrostatic: warmed T_{m}: |dV| per level model "
              + " ".join(f"{abs(v):.2f}" for v in acc_v) + " | theory " + " ".join(f"{abs(v):.2f}" for v in th[:, m]),
              flush=True)
    flat_t, flat_m = np.concatenate([th.real.ravel(), th.imag.ravel()]), np.concatenate([mo.real.ravel(), mo.imag.ravel()])
    corr = float(np.corrcoef(flat_t, flat_m)[0, 1])
    slope = float((flat_t @ flat_m) / (flat_t @ flat_t))
    print(f"hydrostatic: V matrix correlation with theory {corr:.3f}, regression slope {slope:.2f}", flush=True)
    enc = lambda z: {"re": z.real.tolist(), "im": z.imag.tolist()}  # noqa: E731
    return {"probe": "hydrostatic", "ckpt": args.ckpt, "ablate": args.ablate, "lat": la0, "s": s, "temp_K": A,
            "dlnp": dlnp.tolist(), "theory_V": enc(th), "model_V": enc(mo), "theory_U": enc(th_u), "model_U": enc(mo_u),
            "corr_V": corr, "slope_V": slope}


@torch.no_grad()
def gill(em, out: Path, args) -> dict:
    """Matsuno-Gill: a steady mid-tropospheric heating (T_3..T_5, 0.25 K per 6 h at the peak, Gaussian
    12 deg lon x 7 deg lat at 0N 180E) added at every step of paired +/- rollouts (true forcings,
    prescribed ocean TS, noise zero, no correctors). Gill (1980): a Kelvin response to the east
    (low-level easterlies extending ~3x farther than to the west), twin Rossby gyres to the west
    (low-level cyclones at about +-10-15 deg), and the upper flow reversed against the lower."""
    from vit_interp.evaluate import _prescribe

    lat, lon = pg.lat_lon()
    lo0, la0 = 180.0, (0.0 if args.lat is None else args.lat)
    dlon = (lon - lo0 + 180) % 360 - 180
    heat = np.exp(-0.5 * (dlon[None, :] / 12.0) ** 2 - 0.5 * ((lat[:, None] - la0) / 7.0) ** 2)
    prof = {3: 0.7, 4: 1.0, 5: 0.7}
    starts = [0, 120, 240, 360, 720, 840, 960, 1080]  # Jan-Mar and Jul-Sep 2010
    n = 4 * args.days
    hooks = _ablation_hooks(em, args.ablate)
    keep = [pg.ch(c) for c in ("U_7", "V_7", "U_2", "V_2", "PS", "T_4", "specific_total_water_5")]
    names = ["U_7", "V_7", "U_2", "V_2", "PS", "T_4", "specific_total_water_5"]
    acc = np.zeros((len(keep), 180, 360))
    series = np.zeros((n, 360))  # equatorial U_7 response by step
    for t0 in starts:
        tr = pg.truth(t0, n).float()
        st = {}
        for sgn in (1, -1):
            x = tr[0, 0].clone()[None]
            traj = []
            for k in range(n):
                cur = x.clone()
                for lv, w in prof.items():
                    cur[0, pg.ch(f"T_{lv}")] += sgn * args.heat * w * torch.as_tensor(heat, dtype=cur.dtype, device=cur.device)
                nxt, _ = pg.step(em, cur, hooks=hooks)
                nxt = _prescribe(nxt, tr[:, k + 1])
                traj.append(nxt[0, keep].cpu().numpy())
                x = torch.cat([nxt, tr[:, k + 1, pg.N_PROG:]], 1)
            st[sgn] = np.array(traj)
        d = 0.5 * (st[1] - st[-1])  # (n, C, H, W)
        acc += d[n // 2:].mean(0) / len(starts)
        eq = np.abs(lat - la0) <= 5
        series += d[:, 0][:, eq].mean(1) / len(starts)
        print("gill start", t0, flush=True)
    u7 = acc[0][np.abs(lat - la0) <= 5].mean(0)
    east, west = (dlon > 0) & (dlon < 120), (dlon < 0) & (dlon > -120)
    # e-folding distances of the low-level zonal wind response away from the source
    def efold(mask, sign):
        xs, ys = np.abs(dlon[mask]), u7[mask] * sign
        ok = ys > 0
        if ok.sum() < 5:
            return float("nan")
        b = np.polyfit(xs[ok], np.log(ys[ok]), 1)[0]
        return float(-1 / b) if b < 0 else float("inf")
    le, lw = efold(east, -1), efold(west, 1)  # Gill: easterlies to the east, westerlies to the west
    a2, a7 = acc[2][np.abs(lat - la0) <= 10].ravel(), acc[0][np.abs(lat - la0) <= 10].ravel()
    r27 = float(np.corrcoef(a2, a7)[0, 1])
    # low-level relative vorticity west of the source at +-12 deg (Gill: cyclones in both hemispheres)
    v7, u7f = acc[1], acc[0]
    zeta = (np.gradient(v7, np.deg2rad(lon), axis=1) / np.maximum(np.cos(np.deg2rad(lat))[:, None], 1e-3)
            - np.gradient(u7f * np.cos(np.deg2rad(lat))[:, None], np.deg2rad(lat), axis=0)
            / np.maximum(np.cos(np.deg2rad(lat))[:, None], 1e-3)) / R_EARTH
    wbox = (dlon[None, :] > -40) & (dlon[None, :] < -5)
    zn = float(zeta[(lat[:, None] > la0 + 6) & (lat[:, None] < la0 + 20) & wbox].mean())
    zs = float(zeta[(lat[:, None] < la0 - 6) & (lat[:, None] > la0 - 20) & wbox].mean())
    res = {"probe": "gill", "ckpt": args.ckpt, "heat_K_per_step": args.heat, "lat": la0, "days": args.days,
           "u7_east_mean": float(u7[east].mean()), "u7_west_mean": float(u7[west].mean()),
           "efold_east_deg": le, "efold_west_deg": lw, "corr_U2_U7": r27,
           "zeta7_west_north": zn, "zeta7_west_south": zs, "u7_eq": u7.tolist(), "dlon": dlon.tolist()}
    print(f"gill: equatorial U_7 east {res['u7_east_mean']:+.2f} west {res['u7_west_mean']:+.2f} m/s; e-folding east "
          f"{le:.0f} west {lw:.0f} deg (Gill ~3:1); corr(U_2, U_7) {r27:+.2f} (Gill < 0); low-level vorticity west of "
          f"source N {zn:+.2e} S {zs:+.2e} (cyclonic: N > 0, S < 0)", flush=True)
    np.savez_compressed(out / f"gill-{args.ckpt}{'-' + args.tag if args.tag else ''}.npz", maps=acc.astype(np.float32),
                        names=np.array(names), series=series.astype(np.float32), lat=lat, lon=lon)
    return res


def _levels_p():
    md = pg.meta()
    pi = np.array(md["ak"]) + np.array(md["bk"]) * 1.0e5
    bmid = 0.5 * (np.array(md["bk"])[:-1] + np.array(md["bk"])[1:])
    return pi, 0.5 * (pi[:-1] + pi[1:]), bmid


def balance(em, out: Path, args) -> dict:
    """Balanced vs unbalanced at the same scale: a PS bump alone vs the same bump with its geostrophic wind
    (at every level, from the bump's pressure there), widths 600/1200/2400 km, at 30N, 45N, 45S. Learned
    balance predicts the balanced vortex keeps far more of its pressure (and its wind) than the bare bump."""
    lat, lon = pg.lat_lon()
    _, pmid, bmid = _levels_p()
    starts = list(range(0, 1440, 180))
    hooks = _ablation_hooks(em, args.ablate)
    rows = []
    for la0 in (30.0, 45.0, -45.0):
        lo0 = 200.0
        d = _gc_dist(lat, lon, la0, lo0)
        cen = d < 300e3
        wc = np.cos(np.deg2rad(lat))[:, None] * cen
        f = 2 * OMEGA * np.sin(np.deg2rad(lat))[:, None]
        coslat = np.cos(np.deg2rad(lat))[:, None]
        for width in (600e3, 1200e3, 2400e3):
            bump = args.amp * np.exp(-0.5 * (d / width) ** 2)
            dpy = np.gradient(bump, np.deg2rad(lat), axis=0) / R_EARTH
            dpx = np.gradient(bump, np.deg2rad(lon), axis=1) / (R_EARTH * coslat)
            out_row = {"lat": la0, "width_km": width / 1e3}
            for kind in ("ps_only", "balanced"):
                delta = {"PS": bump}
                ug_all = {}
                if kind == "balanced":
                    for k in range(8):
                        x0 = pg.state(starts[0]).float()
                        tk = float(x0[0, pg.ch(f"T_{k}")].mean())
                        rho = pmid[k] / (RD * tk)
                        sc = bmid[k] / rho  # pressure perturbation at level k ~ b_k * PS'
                        ug, vg = -sc * dpy / f, sc * dpx / f
                        delta[f"U_{k}"], delta[f"V_{k}"] = ug, vg
                        ug_all[k] = (ug, vg)
                kept, wkept = [], []
                for t in starts:
                    r = _pair(em, pg.state(t).float(), delta, hooks)
                    kept.append(1 + float((r[pg.ch("PS")] * wc).sum() / (bump * wc).sum()))
                    if ug_all:
                        k = 5
                        ug, vg = ug_all[k]
                        num = ((r[pg.ch(f"U_{k}")] + ug) * ug + (r[pg.ch(f"V_{k}")] + vg) * vg).sum()
                        wkept.append(float(num / (ug * ug + vg * vg).sum()))
                out_row[kind] = {"ps_kept": float(np.mean(kept)), "ps_kept_sd": float(np.std(kept))}
                if wkept:
                    out_row[kind]["wind_kept_L5"] = float(np.mean(wkept))
            rows.append(out_row)
            print(f"balance {la0:+.0f} {width / 1e3:.0f} km: PS kept, bare {out_row['ps_only']['ps_kept']:.2f} vs balanced "
                  f"{out_row['balanced']['ps_kept']:.2f} (wind kept L5 {out_row['balanced']['wind_kept_L5']:.2f})", flush=True)
    return {"probe": "balance", "ckpt": args.ckpt, "ablate": args.ablate, "amp_Pa": args.amp, "rows": rows}


@torch.no_grad()
def _roll_pair(em, t0, delta: dict, n: int, hooks=None, chans=None):
    """Paired rollouts from 2010 state t0 +/- delta (true forcings, prescribed ocean TS, noise zero, raw network):
    the half-difference state anomaly at leads 0..n for channels ``chans``, (n+1, C, H, W)."""
    from vit_interp.evaluate import _prescribe

    tr = pg.truth(t0, n).float()
    idx = [pg.ch(c) for c in chans]
    runs = {}
    for sgn in (1, -1):
        x = tr[0, 0].clone()[None]
        for name, arr in delta.items():
            x[0, pg.ch(name)] += sgn * torch.as_tensor(arr, dtype=x.dtype, device=x.device)
        seq = [x[0, idx].cpu().numpy()]
        for k in range(n):
            nxt, _ = pg.step(em, x, hooks=hooks)
            nxt = _prescribe(nxt, tr[:, k + 1])
            seq.append(nxt[0, idx].cpu().numpy())
            x = torch.cat([nxt, tr[:, k + 1, pg.N_PROG:]], 1)
        runs[sgn] = np.array(seq)
    return 0.5 * (runs[1] - runs[-1])


def kelvin(em, out: Path, args) -> dict:
    """A first-baroclinic equatorial Kelvin wave (s=2, Gaussian 10 deg, U and T in quadrature-free Kelvin
    polarization, V=0, balancing PS), 4-day paired rollouts on 16 states. Theory: eastward intrinsic phase
    speed c (~15-45 m/s), V stays small, T stays in phase with the baroclinic wind. Control: anti-Kelvin
    (T, PS sign flipped) should not propagate east."""
    lat, lon = pg.lat_lon()
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(lon)[None, :]
    pi, pmid, _ = _levels_p()
    hooks = _ablation_hooks(em, args.ablate)
    s, A, c0, ps_, pt = 2, 3.0, 35.0, 1.0e5, 1.4e4
    env = np.exp(-0.5 * (lat / 10.0) ** 2)
    wave = env[:, None] * np.cos(s * np.deg2rad(lon))[None, :]
    starts = list(range(0, 1440, 90))[:16]
    n = 4 * args.days
    res = {"probe": "kelvin", "ckpt": args.ckpt, "ablate": args.ablate, "s": s, "c0": c0, "variants": {}}
    for variant in ("kelvin", "anti"):
        sg = 1.0 if variant == "kelvin" else -1.0
        delta = {}
        for k in range(2, 8):
            z = np.pi * np.log(ps_ / pmid[k]) / np.log(ps_ / pt)
            delta[f"U_{k}"] = -A * np.cos(z) * wave
            delta[f"T_{k}"] = sg * np.pi * c0 * A / (RD * np.log(ps_ / pt)) * np.sin(z) * wave
        delta["PS"] = -sg * 1.15 * c0 * A * wave
        Z, ZT, ZV = np.zeros(n + 1, complex), np.zeros(n + 1, complex), np.zeros(n + 1)
        ubar = 0.0
        for t0 in starts:
            r = _roll_pair(em, t0, delta, n, hooks, ["U_2", "U_7", "T_4", "V_2", "V_7"])
            for k in range(n + 1):
                Z[k] += _proj(r[k, 0] - r[k, 1], env, s, lon, area) / len(starts)
                ZT[k] += _proj(r[k, 2], env, s, lon, area) / len(starts)
                ZV[k] += 0.5 * (abs(_proj(r[k, 3], env, s, lon, area)) + abs(_proj(r[k, 4], env, s, lon, area))) / len(starts)
            x0 = pg.state(t0)
            eq = np.abs(lat) <= 10
            ubar += float((x0[0, pg.ch("U_2")] - x0[0, pg.ch("U_7")]).cpu().numpy()[eq].mean()) / len(starts)
        ph = np.unwrap(np.angle(Z))
        tt = np.arange(n + 1) * DT
        slope = np.polyfit(tt[1:], ph[1:], 1)[0]
        c = float(-slope * R_EARTH / s)
        res["variants"][variant] = {"phase_speed_m_s": c, "amp": np.abs(Z).tolist(), "phase": ph.tolist(),
                                    "amp_kept_end": float(abs(Z[-1]) / abs(Z[0])), "v_over_u": (ZV / np.abs(Z)).tolist(),
                                    "t_u_phase_diff_end": float(np.angle(ZT[-1] / Z[-1]))}
        print(f"kelvin {variant}: phase speed {c:+.1f} m/s (theory ~ +15..45 east), amplitude kept {abs(Z[-1]) / abs(Z[0]):.2f}, "
              f"V/U end {ZV[-1] / abs(Z[-1]):.2f}", flush=True)
    return res


def baroclinic(em, out: Path, args) -> dict:
    """Baroclinic growth: a geostrophic, hydrostatic wave (s=7, 45S, 8 deg envelope) whose phase tilts westward
    with height, is untilted, or tilts eastward. Eady/Charney: only the westward tilt draws on the shear and
    grows (and carries heat poleward); the eastward tilt decays. 4-day paired rollouts on 16 states."""
    lat, lon = pg.lat_lon()
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(lon)[None, :]
    pi, pmid, _ = _levels_p()
    hooks = _ablation_hooks(em, args.ablate)
    la0 = -45.0 if args.lat is None else args.lat
    s, phi0 = 7, 100.0
    env = np.exp(-0.5 * ((lat - la0) / 8.0) ** 2)
    f = 2 * OMEGA * np.sin(np.deg2rad(lat))[:, None]
    f = np.where(np.abs(f) < 2e-5, np.sign(f + 1e-12) * 2e-5, f)
    coslat = np.cos(np.deg2rad(lat))[:, None]
    lnp = np.log(pmid)
    starts = list(range(0, 1440, 90))[:16]
    n = 4 * args.days
    res = {"probe": "baroclinic", "ckpt": args.ckpt, "ablate": args.ablate, "lat": la0, "s": s, "variants": {}}
    for variant, tilt in (("west", np.pi / 2), ("none", 0.0), ("east", -np.pi / 2)):
        psi = [tilt * (7 - max(k, 2)) / 5 for k in range(8)]
        phis = [phi0 * env[:, None] * np.cos(s * np.deg2rad(lon)[None, :] + psi[k]) for k in range(8)]
        delta = {}
        for k in range(8):
            delta[f"U_{k}"] = -np.gradient(phis[k], np.deg2rad(lat), axis=0) / (f * R_EARTH)
            delta[f"V_{k}"] = np.gradient(phis[k], np.deg2rad(lon), axis=1) / (f * R_EARTH * coslat)
        dphi = np.gradient(np.array(phis), lnp, axis=0)
        for k in range(8):
            delta[f"T_{k}"] = -dphi[k] / RD
        delta["PS"] = 1.2 * phis[7]
        E = np.zeros(n + 1); tiltv = np.zeros(n + 1); heat = np.zeros(n + 1)
        chans = [f"U_{k}" for k in range(2, 8)] + [f"V_{k}" for k in range(2, 8)] + ["T_5"]
        for t0 in starts:
            r = _roll_pair(em, t0, delta, n, hooks, chans)
            for k in range(n + 1):
                zu = [_proj(r[k, i], env, s, lon, area) for i in range(6)]
                zv = [_proj(r[k, 6 + i], env, s, lon, area) for i in range(6)]
                E[k] += 0.5 * sum(abs(a) ** 2 + abs(b) ** 2 for a, b in zip(zu, zv)) / len(starts)
                tiltv[k] += float(np.angle(zv[0] / zv[5])) / len(starts)
                zt = _proj(r[k, 12], env, s, lon, area)
                heat[k] += float(np.real(np.conj(zv[3]) * zt)) / len(starts)
        tt = np.arange(n + 1) / 4.0
        sigma = float(np.polyfit(tt[4:], 0.5 * np.log(E[4:]), 1)[0])
        res["variants"][variant] = {"E": E.tolist(), "growth_per_day": sigma, "amp_ratio_end": float(np.sqrt(E[-1] / E[0])),
                                    "tilt_V2_V7": tiltv.tolist(), "heat_flux_V5T5": heat.tolist()}
        print(f"baroclinic {variant}: amplitude x{np.sqrt(E[-1] / E[0]):.2f} after {args.days} d, growth {sigma:+.2f}/day "
              f"(days 1-{args.days}), V2-V7 phase {tiltv[0]:+.2f} -> {tiltv[-1]:+.2f} rad, heat flux end {heat[-1]:+.3g}", flush=True)
    return res


PROBES = {"lamb": lamb, "adjust": adjust, "inertial": inertial, "hydrostatic": hydrostatic, "gill": gill,
          "balance": balance, "kelvin": kelvin, "baroclinic": baroclinic}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("probe", choices=sorted(PROBES))
    p.add_argument("--ckpt", default="v8")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--amp", type=float, default=200.0)
    p.add_argument("--width", type=float, default=600e3)
    p.add_argument("--correct", action="store_true", help="apply the checkpoint's correctors each step")
    p.add_argument("--wind", type=float, default=2.0, help="inertial: planted U band (m/s)")
    p.add_argument("--temp", type=float, default=1.0, help="hydrostatic: planted T wave (K)")
    p.add_argument("--lat", type=float, default=None, help="hydrostatic band centre (45) / gill heating latitude (0)")
    p.add_argument("--heat", type=float, default=0.25, help="gill: heating, K per 6-h step at the peak (T_4)")
    p.add_argument("--days", type=int, default=None, help="gill: 6, kelvin/baroclinic: 4")
    p.add_argument("--ablate", default="", help="vertical | horizontal | L7H3,L6MLP ...")
    p.add_argument("--clamp", default="", help="vertical | horizontal: patch those heads from the unperturbed run")
    p.add_argument("--tag", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.days is None:
        a.days = 6 if a.probe == "gill" else 4
    em, _ = pg.load(a.ckpt)
    res = PROBES[a.probe](em, out, a)
    (out / f"{a.probe}-{a.ckpt}{'-corr' if a.correct else ''}{'-' + a.tag if a.tag else ''}.json").write_text(
        json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
