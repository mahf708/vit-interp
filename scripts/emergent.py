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


@torch.no_grad()
def _pair(em, x, delta: dict, hooks=None):
    """Linear tendency response to a planted perturbation: ([step(x+d)-(x+d)] - [step(x-d)-(x-d)]) / 2."""
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
    return complex((field * ph * w).sum() / (env[:, None] ** 2 * area).sum() / field.shape[1] * 2)


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
            r = _pair(em, pg.state(t).float(), delta, hooks)
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
    return {"probe": "inertial", "ckpt": args.ckpt, "ablate": args.ablate, "wind": args.wind, "rows": rows}


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
    s, la0, sig, A = 4, args.lat, 10.0, args.temp
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


PROBES = {"lamb": lamb, "adjust": adjust, "inertial": inertial, "hydrostatic": hydrostatic}


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
    p.add_argument("--lat", type=float, default=45.0, help="hydrostatic: band centre")
    p.add_argument("--ablate", default="", help="vertical | horizontal | L7H3,L6MLP ...")
    p.add_argument("--tag", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    em, _ = pg.load(a.ckpt)
    res = PROBES[a.probe](em, out, a)
    (out / f"{a.probe}-{a.ckpt}{'-corr' if a.correct else ''}{'-' + a.tag if a.tag else ''}.json").write_text(
        json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
