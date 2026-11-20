"""Build the interactive architecture explorer from a checkpoint's own internals.

    .venv/bin/python scripts/build_arch.py [--ckpt PATH] [--out DIR] [--time 2010-01-15T00]

Renders, for one 2010 state: the input fields and their 4-degree patches, the learned position
embedding (its principal components as maps), every head's attention from three query
points, the logit lens after every block, and the predicted vs true tendency; computes each
head's reach and direction over the northern midlatitudes; and writes architecture.html, a
self-contained page (inline JSON + vanilla JS) that lays these out along the residual stream.
Runs on CPU in a few minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vit_interp import playground as pg  # noqa: E402
from vit_interp.data import N_PROG  # noqa: E402
from vit_interp.model import _Ctx  # noqa: E402

OUT = Path("/lcrc/group/e3sm/public_html/diagnostic_output/ac.ngmahfouz/vit-interp")
QUERIES = [  # (key, label, lat, lon)
    ("npac", "North Pacific jet (45°N, 180°)", 45.0, 180.0),
    ("trop", "Tropical Pacific (2°N, 210°E)", 2.0, 210.0),
    ("socn", "Southern Ocean (55°S, 30°E)", -55.0, 30.0),
]
SURF = "#fcfcfb"


SCHEMATIC = """<svg viewBox="0 0 1120 560" role="img" aria-label="Architecture of the ViT emulator: 39 input fields are cut into 4-degree patches, linearly embedded with a learned position embedding into 4050 tokens of width 384, passed through 8 pre-LayerNorm transformer blocks (each: LayerNorm, 6-head attention, residual add; LayerNorm, MLP, residual add), decoded by a LayerNorm and linear head and un-patchified into a 6-hour tendency, which is added to the state and corrected." font-family="system-ui,-apple-system,sans-serif" font-size="12" style="width:100%;height:auto">
<defs>
<marker id="sa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker>
<marker id="sb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="16" markerHeight="16" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#2a78d6"/></marker>
</defs>
<!-- ===== top: the pipeline ===== -->
<g fill="none" stroke="currentColor" stroke-width="1.3">
<rect x="28" y="78" width="70" height="46" rx="3"/><rect x="24" y="82" width="70" height="46" rx="3"/><rect x="20" y="86" width="70" height="46" rx="3" fill="#2a78d6" fill-opacity=".06"/>
<rect x="140" y="80" width="78" height="48" rx="3"/>
<path d="M155.6,80 V128 M171.2,80 V128 M186.8,80 V128 M202.4,80 V128 M140,96 H218 M140,112 H218" stroke-width=".8"/>
<rect x="250" y="88" width="12" height="34" rx="2"/><rect x="266" y="88" width="12" height="34" rx="2"/><rect x="282" y="88" width="12" height="34" rx="2"/><rect x="318" y="88" width="12" height="34" rx="2"/>
<rect x="362" y="80" width="66" height="48" rx="6"/>
<circle cx="462" cy="104" r="11"/>
<rect x="430" y="172" width="64" height="34" rx="6" stroke-dasharray="4 3"/>
<rect x="500" y="62" width="160" height="84" rx="10" stroke="#2a78d6" stroke-width="2.4" fill="#2a78d6" fill-opacity=".07"/>
<rect x="690" y="80" width="82" height="48" rx="6"/>
<rect x="796" y="80" width="82" height="48" rx="6"/>
<circle cx="906" cy="104" r="11"/>
<rect x="934" y="80" width="68" height="48" rx="6" stroke-dasharray="5 4"/>
<rect x="1028" y="80" width="78" height="48" rx="6"/>
</g>
<g fill="none" stroke="currentColor" stroke-width="1.2" marker-end="url(#sa)">
<line x1="94" y1="104" x2="138" y2="104"/><line x1="218" y1="104" x2="246" y2="104"/><line x1="332" y1="104" x2="360" y2="104"/>
<line x1="428" y1="104" x2="449" y2="104"/><line x1="462" y1="172" x2="462" y2="117"/>
<line x1="473" y1="104" x2="498" y2="104"/><line x1="660" y1="104" x2="688" y2="104"/><line x1="772" y1="104" x2="794" y2="104"/>
<line x1="878" y1="104" x2="893" y2="104"/><line x1="917" y1="104" x2="932" y2="104"/><line x1="1002" y1="104" x2="1026" y2="104"/>
<path d="M55,78 V36 H906 V91"/>
</g>
<g fill="currentColor" text-anchor="middle">
<text x="55" y="152" font-weight="600">x(t)</text><text x="55" y="167" font-size="11" opacity=".72">39 × 180 × 360</text>
<text x="179" y="146" font-weight="600">4°×4° patches</text><text x="179" y="161" font-size="11" opacity=".72">45 × 90 grid</text>
<text x="300" y="109" font-size="13">…</text>
<text x="290" y="146" font-weight="600">tokens</text><text x="290" y="161" font-size="11" opacity=".72">4050 × 624</text>
<text x="395" y="100" font-weight="600">linear</text><text x="395" y="115" font-size="11" opacity=".75">embed E</text>
<text x="395" y="146" font-size="11" opacity=".72">624 → 384</text>
<text x="462" y="108.5" font-weight="700">+</text>
<text x="462" y="187" font-size="11">position</text><text x="462" y="200" font-size="11" opacity=".75">(learned)</text>
<text x="580" y="92" font-weight="700" fill="#2a78d6">transformer block</text>
<text x="580" y="110" font-weight="700" fill="#2a78d6">× 8</text>
<text x="580" y="130" font-size="11" opacity=".8">residual stream 4050 × 384</text>
<text x="731" y="100" font-weight="600">LN · linear</text><text x="731" y="115" font-size="11" opacity=".75">head</text>
<text x="731" y="146" font-size="11" opacity=".72">384 → 34·16</text>
<text x="837" y="100" font-weight="600" font-size="11.5">un-patchify</text><text x="837" y="115" font-size="11" opacity=".75">(+3×3 conv)</text>
<text x="837" y="146" font-weight="600">Δx: tendency</text><text x="837" y="161" font-size="11" opacity=".72">34 × 180 × 360</text>
<text x="906" y="108.5" font-weight="700">+</text>
<text x="968" y="100" font-weight="600">correct</text><text x="968" y="115" font-size="11" opacity=".75">ACE-style</text>
<text x="1067" y="108" font-weight="600">x(t + 6 h)</text>
<text x="480" y="28" font-size="11" opacity=".8">skip: carry the state forward (outside the network)</text>
</g>
<!-- zoom lines -->
<g stroke="currentColor" stroke-width=".9" stroke-dasharray="3 4" opacity=".55" fill="none">
<line x1="500" y1="146" x2="30" y2="262"/><line x1="660" y1="146" x2="1090" y2="262"/>
</g>
<!-- ===== bottom: one block ===== -->
<rect x="30" y="262" width="1060" height="286" rx="12" fill="none" stroke="currentColor" stroke-width="1" opacity=".35"/>
<text x="48" y="286" font-weight="700" fill="currentColor">one transformer block (pre-LayerNorm), 8 in series</text>
<line x1="60" y1="470" x2="1068" y2="470" stroke="#2a78d6" stroke-width="6" stroke-linecap="round" marker-end="url(#sb)"/>
<g fill="none" stroke="currentColor" stroke-width="1.3">
<rect x="178" y="360" width="58" height="40" rx="6"/>
<rect x="276" y="318" width="250" height="118" rx="8"/>
<rect x="290" y="346" width="30" height="62" rx="4"/><rect x="328" y="346" width="30" height="62" rx="4"/><rect x="366" y="346" width="30" height="62" rx="4"/>
<rect x="404" y="346" width="30" height="62" rx="4"/><rect x="442" y="346" width="30" height="62" rx="4"/><rect x="480" y="346" width="30" height="62" rx="4"/>
<circle cx="580" cy="470" r="12" fill="var(--bg,#fcfcfb)"/>
<rect x="688" y="360" width="58" height="40" rx="6"/>
<rect x="786" y="340" width="178" height="80" rx="8"/>
<circle cx="1004" cy="470" r="12" fill="var(--bg,#fcfcfb)"/>
</g>
<g fill="none" stroke="currentColor" stroke-width="1.2" marker-end="url(#sa)">
<path d="M140,466 V380 H176"/><line x1="236" y1="380" x2="274" y2="380"/>
<path d="M526,377 H580 V456"/>
<path d="M640,466 V380 H686"/><line x1="746" y1="380" x2="784" y2="380"/>
<path d="M964,380 H1004 V456"/>
</g>
<g fill="currentColor" text-anchor="middle">
<text x="207" y="385" font-weight="600">LN</text>
<text x="401" y="336" font-weight="600">self-attention · 6 heads × 64</text>
<text x="305" y="381">h0</text><text x="343" y="381">h1</text><text x="381" y="381">h2</text><text x="419" y="381">h3</text><text x="457" y="381">h4</text><text x="495" y="381">h5</text>
<text x="401" y="426" font-size="10.5" opacity=".78">softmax(q·kᵀ/8)·v over 4050 tokens, then W_O</text>
<text x="580" y="475" font-weight="700">+</text>
<text x="717" y="385" font-weight="600">LN</text>
<text x="875" y="370" font-weight="600">MLP</text><text x="875" y="389" font-size="11" opacity=".78">384 → 1536 → GELU → 384</text>
<text x="875" y="406" font-size="11" opacity=".78">per token</text>
<text x="1004" y="475" font-weight="700">+</text>
</g>
<g fill="currentColor" font-size="10.5" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" opacity=".8">
<text x="64" y="494">hook_resid_pre</text>
<text x="590" y="494">hook_resid_mid</text>
<text x="930" y="494">hook_resid_post</text>
<text x="282" y="452">hook_q, _k, _v · hook_pattern · hook_z</text>
<text x="532" y="352">hook_result</text><text x="532" y="364">(per head)</text>
<text x="968" y="368">hook_mlp_out</text>
<text x="786" y="334">mlp.hook_pre · hook_post</text>
</g>
<g fill="currentColor" font-size="10.5" opacity=".8" text-anchor="middle">
<text x="207" y="416">scale, shift</text><text x="207" y="429">← noise (v3)</text>
<text x="717" y="416">scale, shift</text><text x="717" y="429">← noise (v3)</text>
</g>
<text x="560" y="532" font-size="11.5" text-anchor="middle" fill="currentColor" opacity=".85">every head and the MLP <tspan font-weight="700">add</tspan> to the stream; every LayerNorm reads the running sum. Writers: embed + 48 heads + 8 MLPs; readers: 8 + 8 + the head; 513 edges.</text>
</svg>"""


def coastlines():
    import cartopy.io.shapereader as shp

    segs = []
    for geom in shp.Reader(shp.natural_earth("110m", "physical", "coastline")).geometries():
        for line in getattr(geom, "geoms", [geom]):
            x, y = np.asarray(line.coords).T
            x = np.where(x < 0, x + 360, x)
            cut = np.where(np.abs(np.diff(x)) > 180)[0] + 1  # split at the dateline jump
            for a, b in zip(np.r_[0, cut], np.r_[cut, len(x)], strict=True):
                if b - a > 1:
                    segs.append((x[a:b], y[a:b]))
    return segs


def save_map(path, field, extent=(0, 360, -90, 90), cmap="RdBu_r", vmax=None, symmetric=True,
             coast=None, mark=None, size=(3.6, 1.8), grid=None, box=None, vmin=None):
    fig = plt.figure(figsize=size, dpi=100, facecolor=SURF)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    if symmetric:
        v = vmax if vmax is not None else float(np.nanpercentile(np.abs(field), 99)) or 1.0
        lo, hi = -v, v
    else:
        lo = float(np.nanmin(field)) if vmin is None else vmin
        hi = float(np.nanmax(field)) if vmax is None else vmax
    ax.imshow(field, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi, aspect="auto",
              interpolation="nearest")
    if coast:
        for x, y in coast:
            ax.plot(x, y, color="#3a3a38", lw=0.45)
    if grid is not None:
        for gx in np.arange(extent[0], extent[1] + 1e-6, grid):
            ax.axvline(gx, color="#0b0b0b", lw=0.5, alpha=0.55)
        for gy in np.arange(extent[2], extent[3] + 1e-6, grid):
            ax.axhline(gy, color="#0b0b0b", lw=0.5, alpha=0.55)
    if box is not None:
        x0, x1, y0, y1 = box
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#eb6834", lw=2)
    if mark is not None:
        ax.plot(mark[1], mark[0], marker="o", ms=5, mfc="#eb6834", mec="white", mew=1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--time", default="2010-01-15T00")
    a = ap.parse_args()
    out = Path(a.out)
    adir = out / "arch"
    adir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)

    em, meta = pg.load(a.ckpt, device="cpu")
    ckpt = str(a.ckpt or pg.default_checkpoint())
    cfg = em.cfg
    lat, lon = np.asarray(meta["lat"]), np.asarray(meta["lon"])
    tl, tn = pg.token_latlon()
    coast = coastlines()
    x = pg.state(a.time, device="cpu")
    nxt_true = pg.truth(a.time, steps=1)[0, 1, :N_PROG]

    names = {"hook_embed", "hook_pos_embed"}
    for i in range(cfg.depth):
        names |= {f"blocks.{i}.attn.hook_q", f"blocks.{i}.attn.hook_k", f"blocks.{i}.hook_resid_post",
                  f"blocks.{i}.hook_attn_out", f"blocks.{i}.hook_mlp_out"}
    cache: dict = {}
    _, d = em.step(x, cache=cache, names=names)

    # -- input fields and the patch zoom
    fields = {"T_7": ("near-surface temperature", "RdYlBu_r", False), "U_3": ("jet-level zonal wind", "RdBu_r", True),
              "V_3": ("jet-level meridional wind", "RdBu_r", True), "PS": ("surface pressure", "viridis", False)}
    for n, (_, cmap, sym) in fields.items():
        save_map(adir / f"in_{n}.png", x[0, pg.ch(n)].numpy(), cmap=cmap, symmetric=sym, coast=coast)
    q0 = QUERIES[0]
    zx0, zx1, zy0, zy1 = 140, 220, 25, 65  # 80° x 40° around the North Pacific query
    sub = x[0, pg.ch("U_3")].numpy()[(lat >= zy0) & (lat <= zy1)][:, (lon >= zx0) & (lon <= zx1)]
    qi = pg.token_of(q0[2], q0[3])
    qlat, qlon = tl[qi // 90], tn[qi % 90]
    save_map(adir / "zoom.png", sub, extent=(zx0, zx1, zy0, zy1), cmap="RdBu_r", coast=coast, grid=4,
             box=(qlon - 2, qlon + 2, qlat - 2, qlat + 2), size=(3.6, 1.8))
    save_map(adir / "in_U_3_box.png", x[0, pg.ch("U_3")].numpy(), coast=coast, box=(zx0, zx1, zy0, zy1))

    # -- learned position embedding: principal components as maps
    P = em.net.pos_embed[0].numpy()
    P = P - P.mean(0)
    U, S, _ = np.linalg.svd(P, full_matrices=False)
    var = (S**2 / (S**2).sum())[:4]
    for k in range(3):
        save_map(adir / f"pos_pc{k + 1}.png", (U[:, k] * S[k]).reshape(45, 90), coast=coast, cmap="PuOr_r")

    # -- attention from each query, every head
    scale = 1.0 / np.sqrt(cfg.d_head)
    att_rows = {}
    for key, _, la_, lo_ in QUERIES:
        t = pg.token_of(la_, lo_)
        for i in range(cfg.depth):
            q = cache[f"blocks.{i}.attn.hook_q"][0, :, t]  # (H, dh)
            k = cache[f"blocks.{i}.attn.hook_k"][0]  # (H, N, dh)
            p = torch.softmax(torch.einsum("hd,hnd->hn", q, k) * scale, -1).numpy()
            for h in range(cfg.heads):
                row = p[h].reshape(45, 90)
                att_rows[(key, i, h)] = row
                lg = np.log10(row + 1e-12)
                save_map(adir / f"att_{key}_L{i}H{h}.png", lg, extent=(0, 360, -90, 90), cmap="Blues",
                         symmetric=False, vmin=lg.max() - 3, vmax=lg.max(), coast=coast,
                         mark=(la_, lo_), size=(1.8, 0.9))

    # -- head statistics over the northern midlatitudes (attention-weighted)
    LA, LO = np.meshgrid(tl, tn, indexing="ij")
    la_f, lo_f = LA.ravel(), LO.ravel()
    qset = np.where((la_f >= 30) & (la_f <= 60))[0]
    dx = ((lo_f[None, :] - lo_f[qset, None]) + 180) % 360 - 180
    dy = la_f[None, :] - la_f[qset, None]
    km = 111.2 * np.hypot(dx * np.cos(np.deg2rad(la_f[qset, None])), dy)
    heads = []
    for i in range(cfg.depth):
        q = cache[f"blocks.{i}.attn.hook_q"][0][:, qset]  # (H, Q, dh)
        k = cache[f"blocks.{i}.attn.hook_k"][0]
        p = torch.softmax(torch.einsum("hqd,hnd->hqn", q, k) * scale, -1).numpy()  # (H, Q, N)
        for h in range(cfg.heads):
            ph = p[h]
            ent = float((-(ph * np.log(ph + 1e-12)).sum(-1)).mean())
            heads.append({"layer": i, "head": h, "dx": float((ph * dx).sum(-1).mean()),
                          "dy": float((ph * dy).sum(-1).mean()), "km": float((ph * km).sum(-1).mean()),
                          "entropy": ent, "self": float(ph[np.arange(len(qset)), qset].mean())})

    # -- what each block writes, and the logit lens on V_3
    blank = _Ctx(None, None, None)
    v3 = pg.ch("V_3")
    rs = [cache["hook_embed"] + cache["hook_pos_embed"]] + [cache[f"blocks.{i}.hook_resid_post"]
                                                             for i in range(cfg.depth)]
    final = d[:, v3].float() * em.res_std[0, v3]
    w = np.cos(np.deg2rad(lat))[:, None]
    lens_err, decoded = [], []
    for r in rs:
        dv = em.net.unembed(r, blank)[0, v3].float() * em.res_std[0, v3]
        decoded.append(dv.numpy())
        lens_err.append(float(np.sqrt((((dv - final[0]) ** 2).numpy() * w).mean() / ((final[0] ** 2).numpy() * w).mean())))
    vmax = float(np.percentile(np.abs(final.numpy()), 99))
    for k, dv in enumerate(decoded):
        save_map(adir / f"lens_{k}.png", dv, vmax=vmax, coast=coast, size=(1.8, 0.9))
    true_tend = (nxt_true[pg.ch("V_3")] - x[0, pg.ch("V_3")]).numpy()
    save_map(adir / "out_pred.png", final[0].numpy(), vmax=vmax, coast=coast)
    save_map(adir / "out_true.png", true_tend, vmax=vmax, coast=coast)
    writes = []
    for i in range(cfg.depth):
        a_ = cache[f"blocks.{i}.hook_attn_out"][0].norm(dim=-1).mean().item()
        m_ = cache[f"blocks.{i}.hook_mlp_out"][0].norm(dim=-1).mean().item()
        writes.append({"attn": a_, "mlp": m_})

    focused = [h for h in heads if h["entropy"] < 7.5]  # not a global average
    upstream = sorted(focused, key=lambda h: h["dx"])[:3]
    first_below = next((k for k, e in enumerate(lens_err) if 0 < k and e < 1.0), None)
    data = {
        "upstream": [{"name": f"L{h['layer']}H{h['head']}", "dx": h["dx"], "km": h["km"]} for h in upstream],
        "lens_first_below": first_below,
        "ckpt": ckpt, "time": a.time, "cfg": {"depth": cfg.depth, "heads": cfg.heads, "dim": cfg.dim, "patch": cfg.patch,
                                              "out": cfg.out_chans, "head_conv": getattr(cfg, "head_conv", 0),
                                              "tokens": cfg.n_tokens,
                                              "params": int(sum(p.numel() for p in em.net.parameters()))},
        "queries": [{"key": k, "label": l, "lat": la_, "lon": lo_} for k, l, la_, lo_ in QUERIES],
        "heads": heads, "lens_err": lens_err, "writes": writes, "pos_var": [float(v) for v in var],
        "fields": {n: v[0] for n, v in fields.items()},
    }
    (adir / "arch.json").write_text(json.dumps(data))
    (adir / "schematic.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" ' + SCHEMATIC[5:]
                                        .replace("currentColor", "#1b1b1a").replace("var(--bg,#fcfcfb)", "#fcfcfb"))
    (out / "architecture.html").write_text(page(data))
    for p in [out / "architecture.html", *adir.iterdir()]:
        p.chmod(0o644)
    adir.chmod(0o755)
    print(f"wrote {out / 'architecture.html'} ({len(list(adir.iterdir()))} assets)")


def _upstream_text(data: dict) -> str:
    u = data["upstream"]
    if not u or u[0]["dx"] > -5:
        return "No head in this model looks strongly upstream on average."
    rest = ", ".join(f"{h['name']} ({-h['dx']:.0f}°)" for h in u[1:] if h["dx"] < -5)
    return (f"In this model {u[0]['name']} reads about {-u[0]['dx']:.0f}° upstream (west) of a midlatitude token on average"
            + (f"; {rest} also look west." if rest else "."))


def _lens_text(data: dict) -> str:
    e, k = data["lens_err"], data["lens_first_below"]
    if k is None:
        return "here the error stays above 100% until the end."
    return (f"the decoded error stays above 100% until block {k - 1}, drops to {100 * e[k]:.0f}% after it, "
            f"{100 * e[-2]:.0f}% after the second-to-last block, and 0 at the end.")


def page(data: dict) -> str:
    c = data["cfg"]
    js = json.dumps(data)
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ViT architecture explorer</title>
<style>
:root{--bg:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--line:#e4e3df;--card:#fff;--stream:#2a78d6;--hi:#eb6834;
 --t1:#eb6834;--t2:#1baf7a;--t3:#c98500;--t4:#d55181;--t5:#6250d6}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--line:#383835;--card:#232322;--stream:#3987e5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:22px 16px 80px}a{color:var(--stream)}
h1{font-size:1.8rem;margin:.2em 0}.lede{color:var(--fg2);max-width:860px}.muted{color:var(--fg2);font-size:.88rem}
img{display:block;width:100%;height:auto;border-radius:4px;background:#fcfcfb}
.stage{display:grid;grid-template-columns:150px 1fr 190px;gap:14px;align-items:start;margin:0;position:relative}
.stage+.stage{margin-top:6px}
.label{padding-top:10px}.label b{display:block;font-size:1rem}.label span{color:var(--fg2);font-size:.82rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.spine{position:relative}
.spine:before{content:"";position:absolute;left:-8px;top:-6px;bottom:-6px;width:5px;border-radius:3px;background:var(--stream);opacity:.9}
.flow{text-align:center;color:var(--fg2);font-size:.8rem;margin:2px 0 2px 164px}
.tools{display:flex;flex-direction:column;gap:6px;padding-top:6px}
.tool{display:inline-block;border-radius:12px;padding:2px 9px;font-size:.78rem;color:#fff;text-decoration:none;width:fit-content}
.row4{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.row3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.cap{font-size:.78rem;color:var(--fg2);margin-top:3px}
.block{display:grid;grid-template-columns:repeat(6,1fr) 74px 1.15fr;gap:6px;align-items:center}
.head{position:relative;cursor:zoom-in;border:2px solid transparent;border-radius:6px}
.head:hover{border-color:var(--hi)}
.head .nm{position:absolute;left:3px;top:1px;font-size:.66rem;color:#0b0b0b;background:rgba(252,252,251,.8);border-radius:3px;padding:0 3px}
.arrowtag{position:absolute;right:3px;bottom:2px;font-size:.66rem;color:#0b0b0b;background:rgba(252,252,251,.85);border-radius:3px;padding:0 3px}
.mlp{font-size:.72rem;text-align:center;color:var(--fg2)}
.bar{height:6px;border-radius:3px;background:var(--line);margin:3px 0}.bar i{display:block;height:100%;border-radius:3px;background:var(--fg2)}
.lens{position:relative}.lens .nm{position:absolute;left:3px;top:1px;font-size:.66rem;background:rgba(252,252,251,.8);color:#0b0b0b;border-radius:3px;padding:0 3px}
.hdr{display:grid;grid-template-columns:repeat(6,1fr) 74px 1.15fr;gap:6px;font-size:.74rem;color:var(--fg2);margin-bottom:4px}
.picker{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}
.picker button{font:inherit;font-size:.85rem;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:16px;padding:4px 12px;cursor:pointer}
.picker button.on{border-color:var(--hi);box-shadow:0 0 0 1px var(--hi)}
#tip{position:fixed;pointer-events:none;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:.82rem;max-width:280px;box-shadow:0 4px 18px rgba(0,0,0,.18);display:none;z-index:9}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;align-items:center;justify-content:center;z-index:10;padding:16px}
#modal .box{background:var(--card);border-radius:10px;padding:14px;max-width:900px;width:100%}
.legend{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 18px}
.why{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}.why h3{margin:.1em 0 .3em;font-size:1.02rem}.why p{margin:.3em 0}
h2{border-bottom:1px solid var(--line);padding-bottom:.2em}table{border-collapse:collapse;font-size:.9rem}th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}th{color:var(--fg2)}.tw{overflow-x:auto}code{font-size:.88em;background:var(--line);padding:0 .3em;border-radius:4px}
@media (max-width:860px){.stage{grid-template-columns:1fr}.flow{margin-left:0}.tools{flex-direction:row;flex-wrap:wrap}
 .block,.hdr{grid-template-columns:repeat(3,1fr)}.row4{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main>
<p class="muted"><a href="index.html">← results</a></p>
<h1>Architecture explorer</h1>
<p class="lede">The ViT emulator, drawn with its own internals for one 2010 state: what goes in, what the learned position embedding
encodes, where each of the 48 attention heads looks, what the residual stream would predict after every block, and what comes out.
Colored tags mark where each hackathon tool cuts in.</p>
<figure style="margin:14px 0 6px"><div class="card" style="color:var(--fg)">__SCHEMATIC__</div>
<figcaption class="cap" style="margin-top:6px">The architecture at a glance. Top: one 6-hour step, with tensor shapes. Bottom: inside each of the 8 blocks,
with the hook points the tools use. v2 adds the 3×3 conv and 10 diagnostic outputs (44 channels); v3 adds the noise-conditioned LayerNorms.
Below, the explorer shows each stage with this model's own internals.</figcaption></figure>
<div class="legend" id="legend"></div>
<div class="picker" id="picker"><span class="muted" style="align-self:center">Attention from:</span></div>

<div class="stage"><div class="label"><b>1 · Input</b><span>39 fields on the 1° grid; 4 of them shown</span></div>
<div class="card"><div class="row4" id="inputs"></div>
<div class="row3" style="margin-top:8px"><div><img src="arch/in_U_3_box.png" alt="U_3 with zoom box"><div class="cap">zoom region (orange)</div></div>
<div style="grid-column:span 2"><img src="arch/zoom.png" alt="4-degree patches over the North Pacific"><div class="cap"><b>4°×4° patches</b>: each square is one token
(39 fields × 16 points = 624 numbers); the orange square is the query token of the North Pacific view.</div></div></div></div>
<div class="tools"></div></div>
<div class="flow">↓ flatten each patch · linear map to __DIM__ dims · add a learned position vector</div>

<div class="stage"><div class="label"><b>2 · Tokens</b><span>__TOK__ tokens × __DIM__; position is learned</span></div>
<div class="card spine"><div class="row3" id="pos"></div>
<div class="cap" style="margin-top:6px">Principal components of the <b>learned position embedding</b>, the model's only knowledge of where a token is
(and so of latitude, β, land and sea). Its first three components explain <span id="posvar"></span> of its variance.</div></div>
<div class="tools" data-tools="t4"></div></div>
<div class="flow">↓ the residual stream: every block reads the running sum and adds its heads' and MLP's outputs back</div>

<div class="stage"><div class="label"><b>3 · Blocks</b><span>__DEPTH__ × (LayerNorm → 6 heads, all-to-all; LayerNorm → MLP)</span></div>
<div class="card spine"><div class="hdr" id="hdr"></div><div id="blocks"></div>
<div class="cap" style="margin-top:6px"><b>Heads:</b> where the query token's attention goes (log scale; the dot is the query). Hover for each head's
average reach and direction over the northern midlatitudes; click to enlarge. <b>MLP / attn bars:</b> mean size of what each block writes into the
stream. <b>Lens:</b> the V_3 tendency the model would output if decoding stopped after that block, and its error relative to the final output.</div></div>
<div class="tools" data-tools="t1 t2 t3 t5"></div></div>
<div class="flow">↓ final LayerNorm · linear to __OUT__ channels × 4×4 per token · un-patchify__CONV__</div>

<div class="stage"><div class="label"><b>4 · Output</b><span>the 6-h tendency; the skip and the correctors are outside the network</span></div>
<div class="card"><div class="row3"><div><img src="arch/out_pred.png" alt="predicted tendency"><div class="cap">ViT: 6-h V_3 tendency</div></div>
<div><img src="arch/out_true.png" alt="true tendency"><div class="cap">E3SM: the same</div></div>
<div class="cap" style="align-self:center">next state = state + tendency (the <b>persistence skip</b>), then ACE-style corrections (v2+); a free run feeds it back
as the next input. Circuit and LAWFUL measurements read the tendency, so the skip and correctors never enter them.</div></div></div>
<div class="tools" data-tools="t5"></div></div>

<h2 style="margin-top:2.4em">What each design choice gives us</h2>
<p class="lede">The model is deliberately plain. Each choice below trades some skill or efficiency for something we can take apart.</p>
<div class="why">
<div class="card"><h3>4°×4° patches → 4050 tokens</h3><p>Each token is one place on Earth, so anything per token (attention, latents,
SAE features) is a map. Full attention over 4050 tokens is 16 million pairs per head and layer: affordable here, not at 0.25°. Inside a
patch the model only has a linear map in and out, so sub-patch structure (and part of the Rossby response) lives on the direct
embedding → output path, and patch seams appear in long runs.</p><p class="muted">Try: 01 attention atlas; 06 Rossby probe.</p></div>
<div class="card"><h3>Global attention, nothing else</h3><p>No windows, no convolutions in the body, no spectral layers: the only way information
moves between places is a head looking somewhere. Horizontal transport and teleconnections must be done by heads, so their attention
maps are the transport story. __UPSTREAM__</p><p class="muted">Try: 01 attention atlas; 03 ablation lab.</p></div>
<div class="card"><h3>A learned position embedding</h3><p>The model is never told latitude or longitude; it learns a vector per token. That is its
only source of where it is, and so of the Coriolis parameter, β, land and sea. Its leading components are latitude bands and an
east–west dipole, learned from data. Swap two rows of it and you test what the model does with location.</p><p class="muted">Try: 05 latents; the position-swap exercise in 06.</p></div>
<div class="card"><h3>A pre-LayerNorm residual stream</h3><p>Every head and MLP adds its output to one running sum; every block reads the sum. So
the output decomposes exactly into per-component contributions (direct attribution), any intermediate state can be decoded (the logit
lens), and the network is a graph of 57 writers, 17 readers and 513 edges: the object Vi-CD patches.</p><p class="muted">Try: 02 lens; 04 attribution; 07 circuits.</p></div>
<div class="card"><h3>A linear decoder</h3><p>Final LayerNorm, one linear map per token, un-patchify. The same decoder can be applied after any block,
which is what makes the lens meaningful: __LENS__ v2 adds a 3×3
convolution against seams after the last reader, so the graph is unchanged.</p><p class="muted">Try: 02 logit lens.</p></div>
<div class="card"><h3>Predict the tendency; keep the rest outside</h3><p>The network outputs only the 6-hour change. Carrying the state forward (the
skip) and the ACE-style conservation corrections happen after it. Circuit and LAWFUL measurements therefore read the network's own
contribution, and the correctors never confound them.</p><p class="muted">Try: 07 circuits and laws.</p></div>
<div class="card"><h3>Rollout training</h3><p>After one-step training, the model is trained on its own 4- and 8-step forecasts. That is what took v1's free
run from 58 to 248 days and its 5-day anomaly correlation from about 0.6 to over 0.8. It changes the weights, not the architecture, so every
tool applies unchanged.</p><p class="muted">Compare versions on the results page.</p></div>
<div class="card"><h3>Noise-conditioned norms (v3)</h3><p>Per-token noise modulates every LayerNorm's scale and shift, zero-initialized, so the
model becomes stochastic and produces ensembles (trained with CRPS). Noise enters only through the norms: the graph is the same, but
circuits and law scores now depend on the draw, so compare across seeds.</p><p class="muted">Branch v3-stochastic.</p></div>
</div>
<h2 style="margin-top:2em">Where each tool cuts</h2>
<div class="tw"><table><thead><tr><th>tool</th><th>what it touches</th><th>hook points</th><th>notebook</th></tr></thead><tbody>
<tr><td>Attention atlas</td><td>where each head looks from a query token</td><td><code>blocks.i.attn.hook_pattern</code> (or hook_q, hook_k)</td><td><a href="notebooks/01_attention_atlas.html">01</a></td></tr>
<tr><td>Logit lens</td><td>decode the residual stream after each block</td><td><code>blocks.i.hook_resid_post</code></td><td><a href="notebooks/02_logit_lens.html">02</a></td></tr>
<tr><td>Ablation</td><td>remove one head's or MLP's output</td><td><code>attn.hook_result</code>, <code>hook_mlp_out</code></td><td><a href="notebooks/03_ablation_lab.html">03</a></td></tr>
<tr><td>Direct attribution</td><td>split the output into each writer's share</td><td><code>hook_result</code>, <code>hook_mlp_out</code>, <code>hook_embed</code></td><td><a href="notebooks/04_direct_attribution.html">04</a></td></tr>
<tr><td>Latents and SAEs (exege)</td><td>the residual stream on the token grid</td><td><code>blocks.i.hook_resid_post</code></td><td><a href="notebooks/05_exege_latents.html">05</a></td></tr>
<tr><td>Rossby probe (LAWFUL)</td><td>patch whole writers from a perturbed run</td><td>component patching</td><td><a href="notebooks/06_rossby_probe.html">06</a>, <a href="notebooks/07_exege_circuits_and_laws.html">07</a></td></tr>
<tr><td>Edge circuits (Vi-CD)</td><td>patch single writer → reader edges</td><td>edge patching on the 513-edge graph</td><td><a href="notebooks/07_exege_circuits_and_laws.html">07</a></td></tr>
</tbody></table></div>
<p class="muted" style="margin-top:24px">Checkpoint <code>__CKPT__</code> · state __TIME__ · __PARAMS__ parameters ·
regenerate with <code>scripts/build_arch.py</code>.</p>
<div id="tip"></div><div id="modal"><div class="box"><img id="mimg" alt=""><div id="mcap" class="cap"></div></div></div>
<script>
const D = __JS__;
const TOOLS = {t1:["01 attention atlas","notebooks/01_attention_atlas.html","--t1"],t2:["02 logit lens","notebooks/02_logit_lens.html","--t2"],
 t3:["03–04 ablation · attribution","notebooks/03_ablation_lab.html","--t3"],t4:["05 latents & SAEs (exege)","notebooks/05_exege_latents.html","--t4"],
 t5:["06–07 LAWFUL · Vi-CD circuits","notebooks/07_exege_circuits_and_laws.html","--t5"]};
const tag = k => `<a class="tool" style="background:var(${TOOLS[k][2]})" href="${TOOLS[k][1]}">${TOOLS[k][0]}</a>`;
document.getElementById("legend").innerHTML = Object.keys(TOOLS).map(tag).join("");
document.querySelectorAll(".tools[data-tools]").forEach(el => el.innerHTML = el.dataset.tools.split(" ").map(tag).join(""));
document.getElementById("inputs").innerHTML = Object.entries(D.fields).map(([n,l]) =>
  `<div><img src="arch/in_${n}.png" alt="${n}"><div class="cap"><b>${n}</b> ${l}</div></div>`).join("");
document.getElementById("pos").innerHTML = [1,2,3].map(k =>
  `<div><img src="arch/pos_pc${k}.png" alt="position PC${k}"><div class="cap">PC${k} (${(100*D.pos_var[k-1]).toFixed(0)}%)</div></div>`).join("");
document.getElementById("posvar").textContent = (100*(D.pos_var[0]+D.pos_var[1]+D.pos_var[2])).toFixed(0) + "%";
document.getElementById("hdr").innerHTML = [...Array(D.cfg.heads).keys()].map(h=>`<div>head ${h}</div>`).join("") + "<div>writes</div><div>lens (V_3)</div>";
const H = {}; D.heads.forEach(r => H[`${r.layer}.${r.head}`] = r);
const maxw = Math.max(...D.writes.map(w => Math.max(w.attn, w.mlp)));
const dir = r => { const ew = Math.abs(r.dx) < 1.5 ? "" : (r.dx < 0 ? `${(-r.dx).toFixed(0)}° west` : `${r.dx.toFixed(0)}° east`);
  const ns = Math.abs(r.dy) < 1.5 ? "" : (r.dy < 0 ? `${(-r.dy).toFixed(0)}° south` : `${r.dy.toFixed(0)}° north`);
  return [ew, ns].filter(Boolean).join(", ") || "centered"; };
const kind = r => r.entropy > 7.5 ? "global average" : (r.km < 900 ? "local" : (Math.abs(r.dx) > 8 ? (r.dx < 0 ? "looks upstream (west)" : "looks east") : "regional"));
let Q = D.queries[0].key;
function draw() {
  let html = "";
  for (let i = 0; i < D.cfg.depth; i++) {
    html += `<div class="block" style="margin-bottom:6px">`;
    for (let h = 0; h < D.cfg.heads; h++) {
      const r = H[`${i}.${h}`], arrow = Math.abs(r.dx) > 8 ? (r.dx < 0 ? "← " + (-r.dx).toFixed(0) + "°" : "→ " + r.dx.toFixed(0) + "°") : "";
      html += `<div class="head" data-l="${i}" data-h="${h}"><img src="arch/att_${Q}_L${i}H${h}.png" alt="L${i}H${h}"><span class="nm">L${i}H${h}</span>${arrow ? `<span class="arrowtag">${arrow}</span>` : ""}</div>`;
    }
    const w = D.writes[i];
    html += `<div class="mlp">attn<div class="bar"><i style="width:${100*w.attn/maxw}%"></i></div>MLP<div class="bar"><i style="width:${100*w.mlp/maxw}%"></i></div></div>`;
    html += `<div class="lens"><img src="arch/lens_${i+1}.png" alt="lens after block ${i}"><div class="cap" style="margin-top:1px">after block ${i}: error ${(100*D.lens_err[i+1]).toFixed(0)}%</div></div></div>`;
  }
  document.getElementById("blocks").innerHTML = html;
  document.querySelectorAll(".head").forEach(el => {
    const r = H[`${el.dataset.l}.${el.dataset.h}`];
    const txt = `<b>L${r.layer}H${r.head}</b>: ${kind(r)}<br>average target: ${dir(r)}<br>mean distance ${Math.round(r.km).toLocaleString()} km · entropy ${r.entropy.toFixed(1)} of 8.3<br><span class="muted">over all northern-midlatitude query tokens</span>`;
    el.onmousemove = e => { const t = document.getElementById("tip"); t.innerHTML = txt; t.style.display = "block";
      t.style.left = Math.min(e.clientX + 14, innerWidth - 300) + "px"; t.style.top = (e.clientY + 14) + "px"; };
    el.onmouseleave = () => document.getElementById("tip").style.display = "none";
    el.onclick = () => { document.getElementById("mimg").src = el.querySelector("img").src;
      document.getElementById("mcap").innerHTML = txt + ` · query: ${D.queries.find(q => q.key === Q).label}`;
      document.getElementById("modal").style.display = "flex"; };
  });
}
document.getElementById("modal").onclick = () => document.getElementById("modal").style.display = "none";
const pk = document.getElementById("picker");
D.queries.forEach(q => { const b = document.createElement("button"); b.textContent = q.label; if (q.key === Q) b.className = "on";
  b.onclick = () => { Q = q.key; pk.querySelectorAll("button").forEach(x => x.className = ""); b.className = "on"; draw(); }; pk.appendChild(b); });
draw();
</script></main></body></html>""".replace("__JS__", js).replace("__DIM__", str(c["dim"])).replace("__TOK__", str(c["tokens"])) \
        .replace("__DEPTH__", str(c["depth"])).replace("__OUT__", str(c["out"])) \
        .replace("__CONV__", " · 3×3 conv" if c["head_conv"] else "").replace("__CKPT__", data["ckpt"]) \
        .replace("__SCHEMATIC__", SCHEMATIC).replace("__UPSTREAM__", _upstream_text(data)).replace("__LENS__", _lens_text(data)) \
        .replace("__TIME__", data["time"]).replace("__PARAMS__", f"{c['params'] / 1e6:.1f} M")


if __name__ == "__main__":
    main()
