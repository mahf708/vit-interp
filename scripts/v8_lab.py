"""Interpretability studies that only the 3D model (v8) makes possible. Each subcommand writes JSON
(and .npz where there are maps) into --out; scripts/build_report.py turns them into the page.

    python scripts/v8_lab.py heads        --out DIR   # full head atlas + what vertical heads read (land/ocean/ice)
    python scripts/v8_lab.py transmission --out DIR   # a wave planted at one level: who carries it to the others?
    python scripts/v8_lab.py lawful       --out DIR   # does the Rossby law live in horizontal or vertical heads?
    python scripts/v8_lab.py surface      --out DIR   # a circuit for surface heating reaching the atmosphere (Tibet)
    python scripts/v8_lab.py errors       --out DIR   # which components write the model's climate drift?
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vit_interp import playground as pg
from vit_interp.data import ALL_NAMES, PROG_NAMES

STARTS = ["2010-01-15T00", "2010-04-15T00", "2010-07-15T00", "2010-10-15T00"]


def _save(out: Path, name: str, obj) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    print(f"wrote {out / name}.json", flush=True)


def _dev(em):
    return next(em.parameters()).device


# -- 1. heads: the full atlas, and where vertical heads look ------------------------------------

def heads(em, out: Path) -> None:
    dev = _dev(em)
    xs = torch.cat([pg.state(t) for t in STARTS[::2]]).to(dev)
    atlas = pg.head_atlas(em, xs)
    _save(out, "atlas", atlas.to_dict(orient="records"))
    # what each vertical head reads, by surface type of the column
    x0 = pg.state(STARTS[0])[0].cpu()
    land = (x0[ALL_NAMES.index("LANDFRAC")] > 0.5).numpy()
    ice = (x0[ALL_NAMES.index("ICEFRAC")] > 0.5).numpy()
    p = 4
    coarse = lambda m: m.reshape(45, p, 90, p).mean((1, 3)).ravel() > 0.5  # noqa: E731
    masks = {"land": coarse(land), "sea ice": coarse(ice), "open ocean": ~coarse(land) & ~coarse(ice)}
    tib = np.zeros(4050, bool)
    for la in (28, 32, 36):
        for lo in (80, 84, 88, 92, 96):
            tib[pg.token_of(la, lo)] = True
    masks["Tibetan Plateau"] = tib
    kinds = pg.head_kinds(em)
    rows = []
    for layer in range(em.cfg.depth):
        vh = [h for h in range(em.cfg.heads) if kinds[(layer, h)] == "v"]
        acc = 0
        for t in STARTS:
            acc = acc + pg.patterns(em, pg.state(t).to(dev), layer, "v")[0].cpu().numpy()  # (Hv, N, 9, 9)
        acc /= len(STARTS)
        for j, h in enumerate(vh):
            for name, m in masks.items():
                a = acc[j][m].mean(0)  # (9, 9): reading level x read level
                rows.append({"layer": layer, "head": h, "columns": name,
                             "surface_weight": float(a[:8, 8].mean()),  # atmosphere levels reading the surface token
                             "mean_read_offset": float((a * (np.arange(9)[None, :] - np.arange(9)[:, None])).sum(1).mean()),
                             "pattern": a.round(4).tolist()})
    _save(out, "vertical_reading", rows)


# -- 2. vertical transmission of a planted wave ------------------------------------------------

def _plant(x, level: int, s: int = 6, phi0: float = 45.0, amp: float = 5.0):
    """A wave of zonal wavenumber s in a 10-degree band at one level (U and V of that level)."""
    lat, lon = pg.lat_lon()
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    G = np.exp(-(((LA - phi0) / 10.0) ** 2))
    v = amp * G * np.sin(s * np.deg2rad(LO))
    u = amp * 0.5 * G * np.cos(s * np.deg2rad(LO)) * (-(LA - phi0) / 10.0)
    x = x.clone()
    x[:, ALL_NAMES.index(f"V_{level}")] += torch.tensor(v, dtype=x.dtype)
    x[:, ALL_NAMES.index(f"U_{level}")] += torch.tensor(u, dtype=x.dtype)
    return x


def _band_rms(t, lat):
    w = np.exp(-(((lat - 45.0) / 10.0) ** 2))[:, None] * np.cos(np.deg2rad(lat))[:, None]
    w = torch.tensor(w / w.sum() / 360, dtype=t.dtype, device=t.device)
    return float((t ** 2 * w).sum().sqrt())


def transmission(em, out: Path) -> None:
    dev = _dev(em)
    lat, _ = pg.lat_lon()
    kinds = pg.head_kinds(em)
    vertical = [f"L{l}H{h}" for (l, h), k in kinds.items() if k == "v"]
    horizontal = [f"L{l}H{h}" for (l, h), k in kinds.items() if k == "h"]
    variants = {"intact": [], "no vertical heads": vertical, "no horizontal heads": horizontal}
    res = {}
    for name, ablate in variants.items():
        M = np.zeros((8, 8))
        for t in STARTS:
            x = pg.state(t).to(dev)
            for src in range(8):
                xp = _plant(x.cpu(), src).to(dev)
                if ablate:
                    d0 = pg.ablate(em, x, names=ablate).delta + pg.step(em, x)[1][0]
                    d1 = pg.ablate(em, xp, names=ablate).delta + pg.step(em, xp)[1][0]
                else:
                    d0, d1 = pg.step(em, x)[1][0], pg.step(em, xp)[1][0]
                for k in range(8):
                    M[src, k] += _band_rms(d1[pg.ch(f"V_{k}")] - d0[pg.ch(f"V_{k}")], lat) / len(STARTS)
        res[name] = M.tolist()
        print(name, "done", flush=True)
    _save(out, "transmission", {"rows": "planted level 0..7", "cols": "responding level 0..7 (V tendency, band RMS, m/s per 6 h)",
                                **res})


# -- 3. LAWFUL with head-type groups -----------------------------------------------------------

def lawful(out: Path) -> None:
    from exege.circuits import open_patchable
    from exege.laws import Rossby, deviations, measure, search
    from exege.laws.law import Envelope

    system = open_patchable("v8", "vit-interp", year=2010)
    g = system.graph()
    vert = [f"a{i}.h{h}" for i in range(8) for h in (4, 5)]
    horiz = [f"a{i}.h{h}" for i in range(8) for h in range(4)]
    mlps = [f"m{i}" for i in range(8)]
    result = {}
    for phi in (45, -45):
        law = Rossby(u=("U_2", "U_3", "U_4"), v=("V_2", "V_3", "V_4"), observe_field="V_3", ps="PS",
                     band=Envelope(phi, 10.0))
        base = system.states([0, 120, 240, 360])
        ctrl = [system.states([60, 180, 300, 420])]
        m = measure(system, base, law, score="correlation")
        entry = {"M0": m.M, "groups": {}}
        for name, comps in (("horizontal heads", horiz), ("vertical heads", vert), ("MLPs", mlps),
                            ("heads (both kinds)", horiz + vert)):
            d = deviations(system, base, law, comps, M0=m.M, controls=ctrl, score="correlation")
            entry["groups"][name] = d.to_dict() if hasattr(d, "to_dict") else {k: getattr(d, k) for k in ("d_suf", "d_nec")}
            print(phi, name, entry["groups"][name], flush=True)
        groups = {f"L{i} {kind}": ([f"a{i}.h{h}" for h in (range(4) if kind == "horizontal" else (4, 5))])
                  for i in range(8) for kind in ("horizontal", "vertical")}
        groups.update({f"L{i} MLP": [f"m{i}"] for i in range(8)})
        try:
            s = search(system, base, law, taus=(0.3, 0.5, 0.7, 0.9), groups=groups, controls=ctrl, score="correlation")
            entry["search"] = [{"tau": c.tau, "units": list(c.units or c.components),
                                "d": c.deviations.to_dict() if hasattr(c.deviations, "to_dict") else None}
                               for c in s.circuits]
        except Exception as exc:  # e.g. M0 <= 0: no circuit to look for
            entry["search_error"] = str(exc)
        result[f"{abs(phi)}{'N' if phi > 0 else 'S'}"] = entry
    _save(out, "lawful_groups", result)


# -- 4. a circuit for surface heating ----------------------------------------------------------

def surface(out: Path) -> None:
    from exege.circuits import attribute, field_mean, open_patchable, prune

    system = open_patchable("v8", "vit-interp", year=2010)
    g, grid = system.graph(), system.grid()
    rows = {}
    for label, t, (la, lo) in (("Tibetan Plateau, June", "2010-06-15T06", (32, 88)),
                               ("Sahara, July", "2010-07-15T12", (23, 10))):
        base = system.states([t])
        f = system.fields(base)
        lat, lon = grid.lat, grid.lon
        d = np.zeros_like(f["TS"])
        near = ((lat - la) ** 2 + (((lon - lo + 180) % 360) - 180) ** 2 * np.cos(np.deg2rad(la)) ** 2) < 8 ** 2
        d[:, near] = 5.0  # +5 K surface skin temperature over the region
        source = system.perturb(base, {"TS": d})
        where = grid.within(la, lo, 1000)
        out_rows = {}
        for field in ("T_7", "T_5", "T_3"):
            metric = field_mean(grid, field, where, label=f"{field} over {label}")
            att = attribute(system, base, source, metric, method="patch")  # exact; EAP gradients exceed 40 GB on 3D tokens
            circ = prune(system, base, source, metric, tau=0.02)
            edges = list(circ.edges())
            out_rows[field] = {"clean": att.clean, "corrupt": att.corrupt, "n_edges": circ.n_edges, "edges": edges,
                               "recovery": circ.recovery,
                               "top_eap": [[g.senders[i] + "->" + g.receivers[j], float(att.scores[i, j])]
                                           for i, j in zip(*np.unravel_index(np.argsort(-np.abs(att.scores), axis=None)[:15], att.scores.shape))]}
            print(label, field, out_rows[field]["n_edges"], "edges", flush=True)
        rows[label] = out_rows
    _save(out, "surface_circuits", rows)


# -- 5. which components write the climate drift? ----------------------------------------------

def errors(em, out: Path) -> None:
    """The decade runs drift the same way in every seed: the model top warms and the mid-troposphere
    cools. Per 6-h step, over 24 states of 2010: the mean global error of each level's temperature
    tendency, and how mean-ablating each component changes that bias (and the step's skill)."""
    dev = _dev(em)
    times = json.loads((Path(pg.DATA_ROOT) / "2010.times.json").read_text())
    idx = list(range(0, 1440, 60))
    arr = np.load(Path(pg.DATA_ROOT) / "2010.npy", mmap_mode="r")
    w = torch.tensor(np.cos(np.deg2rad(pg.lat_lon()[0]))[:, None], dtype=torch.float32, device=dev)
    w = w / (w.sum() * 360)
    T = [pg.ch(f"T_{k}") for k in range(8)]

    def gmean(t):
        return (t * w).sum((-2, -1))

    xs = [torch.from_numpy(np.array(arr[i])).to(dev)[None] for i in idx]
    truth = [torch.from_numpy(np.array(arr[i + 1][:len(PROG_NAMES)])).to(dev)[None] - x[:, :len(PROG_NAMES)] for i, x in zip(idx, xs)]

    def bias_and_rmse(names=()):
        b, r = torch.zeros(8, device=dev), 0.0
        for x, tr in zip(xs, truth):
            _, t = pg.step(em, x)
            if names:
                t = t + pg.ablate(em, x, names=list(names)).delta[None].to(dev)
            e = t - tr
            b += gmean(e[0, T]) / len(xs)
            r += float(gmean(e[0, T] ** 2).mean().sqrt()) / len(xs)
        return b.cpu().numpy(), r

    b0, r0 = bias_and_rmse()
    print("baseline T bias per level (K/6h):", np.round(b0, 4), "rmse", round(r0, 4), flush=True)
    rows = []
    for c in pg.components(em)[1:]:
        b, r = bias_and_rmse([c])
        rows.append({"component": c, "kind": pg.head_kinds(em).get((int(c[1:].split("H")[0]), int(c.split("H")[1])), "")
                     if "H" in c else "mlp", "bias": b.tolist(), "rmse": r,
                     "top_bias_change": float(b[0] - b0[0]), "mid_bias_change": float(b[3] - b0[3])})
        print(c, "dT0", round(rows[-1]["top_bias_change"], 5), "dT3", round(rows[-1]["mid_bias_change"], 5),
              "rmse", round(r, 4), flush=True)
    # direct attribution of the top-level temperature tendency, averaged over the states: who writes it
    att = {}
    for x in xs[::4]:
        _, df = pg.direct_attribution(em, x, "T_0")
        for c, v in zip(df.component, df.contribution):
            att[c] = att.get(c, 0.0) + v / len(xs[::4])
    _save(out, "error_components", {"baseline_bias": b0.tolist(), "baseline_rmse": r0, "components": rows,
                                    "t0_direct_attribution": att, "starts": [times[i] for i in idx]})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("study", choices=["heads", "transmission", "lawful", "surface", "errors"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    t0 = time.time()
    if a.study in ("lawful", "surface"):
        (lawful if a.study == "lawful" else surface)(out)
    else:
        em, _ = pg.load("v8")
        {"heads": heads, "transmission": transmission, "errors": errors}[a.study](em, out)
    print(f"{a.study} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
