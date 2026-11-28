"""E3SM-referenced wave tests: run the emulator freely over held-out years (true forcings, prescribed ocean
TS, its correctors, patch jitter + seam filter) saving 6-hourly tropical fields, and compare with E3SM over
the same years:

  * Wheeler-Kiladis wavenumber-frequency spectra of outgoing longwave radiation (FLUT) and low-level wind,
    symmetric and antisymmetric about the equator, divided by a smoothed background: Kelvin, equatorial
    Rossby, mixed Rossby-gravity and MJO signals, in E3SM and in the emulator.
  * a Gill-type lag-0 regression of low- and upper-level wind on equatorial Pacific precipitation.

    python scripts/tropical_waves.py run --ckpt v8 --years 2011-2012 --out DIR
    python scripts/tropical_waves.py analyze --out DIR --ckpt v8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIELDS = ["FLUT", "surface_precipitation_rate", "U_6", "U_2", "V_6"]
BAND = 25.0  # degrees either side of the equator
DATA = "/lcrc/globalscratch/ac.ngmahfouz/vit-interp/data-v2"


def run(a):
    import torch

    from vit_interp.corrector import attach, diag_names_of
    from vit_interp.data import N_ALL, N_PROG, PROG_NAMES, Stats, resolve_checkpoint
    from vit_interp.emulator import Emulator
    from vit_interp.evaluate import Stepper, _prescribe, _to
    from vit_interp.filters import PatchJitter, SeamFilter

    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    meta = json.loads((Path(DATA) / "meta.json").read_text())
    lat = np.asarray(meta["lat"])
    band = np.abs(lat) <= BAND
    em, ck = Emulator.load(resolve_checkpoint(a.ckpt), Stats(DATA), lat, map_location=dev)
    em = em.to(dev).eval()
    attach(em, ck, meta)
    dnames = diag_names_of(ck)
    step = Stepper(em, dev, SeamFilter(lat, len(meta["lon"]), 60.0, 1).to(dev), PatchJitter(em.cfg.patch, "cycle16"))
    names = meta["names"]
    src = {f: ("prog", PROG_NAMES.index(f)) if f in PROG_NAMES else ("diag", dnames.index(f)) for f in FIELDS}
    tidx = [names.index(f) for f in FIELDS]
    y0, y1 = (int(x) for x in a.years.split("-"))
    emu, tru, state, last_forcing = [], [], None, None
    for y in range(y0, y1 + 1):
        arr = np.load(Path(DATA) / f"{y}.npy", mmap_mode="r")
        if state is None:
            state = _to(dev, arr[0][None, :N_PROG])
        elif last_forcing is not None:
            state = _prescribe(step(state, last_forcing, _to(dev, arr[0][None, N_PROG:N_ALL])), _to(dev, arr[0][None, :N_ALL]))
        for t in range(arr.shape[0] - 1):
            forcing = _to(dev, arr[t][None, N_PROG:N_ALL])
            truth = _to(dev, arr[t + 1][None, :N_ALL])
            state = _prescribe(step(state, forcing, truth[:, N_PROG:N_ALL]), truth)
            row = []
            for f in FIELDS:
                kind, i = src[f]
                v = state[0, i] if kind == "prog" else step.diag[0, i]
                row.append(v[torch.as_tensor(band, device=dev)].float().cpu().numpy())
            emu.append(np.array(row, dtype=np.float32))
            tru.append(np.asarray(arr[t + 1][tidx][:, band], dtype=np.float32))
        last_forcing = _to(dev, arr[-1][None, N_PROG:N_ALL])
        print(f"{y} done, {len(emu)} steps", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"tropics-{a.ckpt}.npz", emu=np.array(emu, dtype=np.float16), truth=np.array(tru, dtype=np.float16),
             fields=np.array(FIELDS), lat=lat[band], lon=np.asarray(meta["lon"]), years=a.years)
    print("wrote", out / f"tropics-{a.ckpt}.npz", flush=True)


# -- Wheeler-Kiladis -----------------------------------------------------------------------------

def _wk(x, lat, seg=96 * 4, overlap=60 * 4):
    """x (time, lat, lon) 6-hourly. Returns (k, freq cpd, P_sym, P_asym) averaged over segments, with the
    seasonal cycle (first 3 annual harmonics) and the zonal-time mean removed and a taper applied."""
    T = x.shape[0]
    t = np.arange(T)
    # remove annual cycle harmonics and mean per grid point
    A = [np.ones(T)]
    for h in (1, 2, 3):
        w = 2 * np.pi * h * t / (365 * 4)
        A += [np.cos(w), np.sin(w)]
    A = np.array(A).T
    flat = x.reshape(T, -1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, flat, rcond=None)
    x = (flat - A @ coef).reshape(x.shape)
    sym = 0.5 * (x + x[:, ::-1])
    asym = 0.5 * (x - x[:, ::-1])
    half = lat >= 0
    taper = np.hanning(seg)[:, None, None]
    Ps, Pa, n = 0, 0, 0
    for s0 in range(0, T - seg + 1, seg - overlap):
        for P, comp in ((0, sym), (1, asym)):
            c = comp[s0:s0 + seg][:, half] * taper
            c = c - c.mean(0, keepdims=True)
            F = np.fft.fft2(c, axes=(0, 2))  # time, lon
            p = (np.abs(F) ** 2).sum(1)  # sum over latitudes
            if P == 0:
                Ps = Ps + p
            else:
                Pa = Pa + p
        n += 1
    freq = np.fft.fftfreq(seg, d=0.25)  # cycles per day
    k = np.fft.fftfreq(x.shape[2], d=1.0 / x.shape[2])
    # convention: eastward-propagating waves at positive k for positive frequency
    Ps, Pa = np.fft.fftshift(Ps / n), np.fft.fftshift(Pa / n)
    freq, k = np.fft.fftshift(freq), -np.fft.fftshift(k)
    return k, freq, Ps, Pa


def _smooth121(P, n_freq=10, n_k=10):
    def pass121(P, axis):
        Q = np.moveaxis(P, axis, 0)
        pad = np.concatenate([Q[:1], Q, Q[-1:]], 0)
        return np.moveaxis(0.25 * pad[:-2] + 0.5 * pad[1:-1] + 0.25 * pad[2:], 0, axis)

    for _ in range(n_freq):
        P = pass121(P, 0)
    for _ in range(n_k):
        P = pass121(P, 1)
    return P


def _disp(kind, h, k):
    """Matsuno dispersion (cpd) for wavenumber k (positive = eastward) at equivalent depth h."""
    g, beta, a = 9.81, 2.28e-11, 6.371e6
    c = np.sqrt(g * h)
    kk = 2 * np.pi * k / (2 * np.pi * a)
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind == "kelvin":
            om = c * kk
        elif kind == "er1":
            om = -beta * kk / (kk ** 2 + 3 * beta / c)
        elif kind == "mrg":
            om = kk * c / 2 * (1 - np.sqrt(1 + 4 * beta / (kk ** 2 * c)))
            om = np.where(kk < 0, om, np.nan) if False else om
    return om * 86400 / (2 * np.pi)


def analyze(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(a.out)
    z = np.load(out / f"tropics-{a.ckpt}.npz")
    fields, lat, lon = list(z["fields"]), z["lat"], z["lon"]
    res = {"ckpt": a.ckpt, "years": str(z["years"]), "spectra": {}}
    for f in ("FLUT", "U_6"):
        i = fields.index(f)
        sp = {}
        for src in ("truth", "emu"):
            x = z[src][:, i].astype(np.float32)
            k, fr, Ps, Pa = _wk(x, lat)
            bg = _smooth121(0.5 * (Ps + Pa), 15, 15)
            sp[src] = (k, fr, Ps, Pa, bg)
        # shared background (E3SM's) so the two are on one scale
        kk, ff = np.meshgrid(sp["truth"][0], sp["truth"][1])
        m_k = (kk >= 1) & (kk <= 14) & (ff >= 1 / 20) & (ff <= 1 / 2.5)
        hk = lambda h: _disp("kelvin", h, kk)  # noqa: E731
        kel = m_k & (ff >= hk(8)) & (ff <= hk(90))
        er = (kk <= -1) & (kk >= -10) & (ff >= 1 / 90) & (ff <= 1 / 4) & (ff <= _disp("er1", 90, kk)) & (ff >= _disp("er1", 8, kk))
        mjo = (kk >= 1) & (kk <= 5) & (ff >= 1 / 96) & (ff <= 1 / 30)
        stats = {}
        for src in ("truth", "emu"):
            _, _, Ps, Pa, bg = sp[src]
            stats[src] = {"kelvin_power": float(Ps[kel].sum()), "er_power": float(Ps[er].sum()),
                          "mjo_power": float(Ps[mjo].sum()),
                          "kelvin_ratio": float((Ps / sp["truth"][4])[kel].mean()),
                          "er_ratio": float((Ps / sp["truth"][4])[er].mean()),
                          "total_sym": float(Ps[(np.abs(ff) <= 0.8) & (np.abs(kk) <= 15)].sum())}
            # equivalent depth of the Kelvin peak: fit c from the power-weighted frequency/wavenumber in the band
            w = Ps[kel]
            c = float((w * ff[kel] / np.maximum(kk[kel], 1)).sum() / w.sum()) * 2 * np.pi * 6.371e6 / 86400
            stats[src]["kelvin_speed_m_s"] = c
            stats[src]["kelvin_h_m"] = c * c / 9.81
        res["spectra"][f] = stats
        print(f, json.dumps(stats, indent=0), flush=True)
        # figure: symmetric spectrum / E3SM background, E3SM vs emulator
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, src, ttl in zip(axs, ("truth", "emu"), ("E3SM", a.ckpt)):
            k_, f_, Ps, Pa, bg = sp[src]
            R = Ps / sp["truth"][4]
            sel_k = (k_ >= -15) & (k_ <= 15)
            sel_f = (f_ > 0) & (f_ <= 0.8)
            im = ax.contourf(k_[sel_k], f_[sel_f], R[np.ix_(sel_f, sel_k)], levels=np.linspace(0.6, 2.4, 13),
                             cmap="viridis", extend="both")
            kx = np.linspace(0.5, 15, 50)
            for h in (12, 25, 50):
                ax.plot(kx, _disp("kelvin", h, kx), color="w", lw=0.8)
                ax.plot(-kx, _disp("er1", h, -kx), color="w", lw=0.8, ls="--")
            ax.set_ylim(0, 0.8); ax.set_xlim(-15, 15)
            ax.set_title(f"{ttl}: {f} symmetric power / E3SM background", fontsize=10)
            ax.set_xlabel("zonal wavenumber (> 0 eastward)"); ax.set_ylabel("frequency (cycles per day)")
        fig.colorbar(im, ax=axs.tolist(), shrink=0.85)
        fig.savefig(out / f"wk-{f}-{a.ckpt}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    # Gill-type regression on equatorial Pacific precipitation (160E-180, 5S-5N)
    ip = fields.index("surface_precipitation_rate")
    box = (np.abs(lat) <= 5)[:, None] & ((lon >= 160) & (lon <= 180))[None, :]
    reg = {}
    for src in ("truth", "emu"):
        P = z[src][:, ip].astype(np.float64)
        idx = P[:, box].mean(1)
        idx = (idx - idx.mean()) / idx.std()
        maps = {}
        for f in ("U_6", "U_2", "V_6"):
            X = z[src][:, fields.index(f)].astype(np.float64)
            X = X - X.mean(0)
            maps[f] = (np.tensordot(idx, X, axes=(0, 0)) / len(idx)).astype(np.float32)
        reg[src] = maps
    pc = {f: float(np.corrcoef(reg["truth"][f].ravel(), reg["emu"][f].ravel())[0, 1]) for f in ("U_6", "U_2", "V_6")}
    amp = {f: float(np.abs(reg["emu"][f]).mean() / np.abs(reg["truth"][f]).mean()) for f in ("U_6", "U_2", "V_6")}
    res["gill_regression"] = {"pattern_corr": pc, "amplitude_ratio": amp}
    print("gill regression: pattern correlation emulator vs E3SM", pc, "amplitude ratio", amp, flush=True)
    np.savez_compressed(out / f"gillreg-{a.ckpt}.npz", **{f"{s}_{f}": reg[s][f] for s in reg for f in reg[s]}, lat=lat, lon=lon)
    (out / f"tropics-{a.ckpt}.json").write_text(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["run", "analyze"])
    p.add_argument("--ckpt", default="v8")
    p.add_argument("--years", default="2011-2012")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    (run if a.what == "run" else analyze)(a)


if __name__ == "__main__":
    main()
