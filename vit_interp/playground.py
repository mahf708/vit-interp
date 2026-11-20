"""Small helpers for poking at the ViT emulator in a notebook, one to three lines per cell.

    from vit_interp import playground as pg
    em, meta = pg.load()                      # best checkpoint, on the GPU if there is one
    x = pg.state("2010-01-15")                # physical state + forcing, (1, 39, 180, 360)
    nxt, tend = pg.step(em, x)                # next prognostic state, 6-h tendency (physical)
    pg.plot_map(tend[0, pg.ch("V_3")], "V_3 tendency [m/s per 6 h]")

What is here:
    load, state, truth, step, rollout          running the model
    ch, PROG_NAMES, ALL_NAMES, box, box_mean   channels and regions
    plot_map, plot_maps, plot_tokens           maps (cartopy coastlines when available)
    token_of, token_latlon                     the 45 x 90 grid of 4-degree patch tokens
    attention, head_atlas                      where heads look
    ablate, sweep_components                   knock things out and see what breaks
    lens                                       decode every layer's residual stream (logit lens)
    direct_attribution                         which head/MLP writes a quantity, and where

Tendencies are physical units per 6-hour step (``d * res_std``), shaped
(batch, 34, 180, 360) in PROG_NAMES order. Set ``PLAYGROUND_FAST=1`` for small settings
(used to test the notebooks on a CPU).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import ALL_NAMES, DATA_ROOT, N_PROG, PROG_NAMES, Stats
from .emulator import Emulator
from .model import _Ctx

RUN = Path("/lcrc/globalscratch/ac.ngmahfouz/vit-interp/runs/vit-p4-d384-l8")
FAST = os.environ.get("PLAYGROUND_FAST") == "1"
_META: dict | None = None

__all__ = [
    "ALL_NAMES", "FAST", "PROG_NAMES", "ablate", "attention", "box", "box_mean", "ch",
    "components", "direct_attribution", "head_atlas", "lens", "load", "meta", "plot_map",
    "plot_maps", "plot_tokens", "rollout", "state", "step", "sweep_components", "token_latlon",
    "token_of", "truth",
]


# -- loading and running ---------------------------------------------------------------------


def meta(data: str = DATA_ROOT) -> dict:
    global _META
    if _META is None:
        _META = json.loads((Path(data) / "meta.json").read_text())
    return _META


def default_checkpoint() -> Path:
    """The released checkpoint: release/v2.pt when it exists, else release/v1.pt (frozen copies,
    so a running training job can never change what a notebook loads)."""
    for p in (RUN / "release" / "v2.pt", RUN / "release" / "v1.pt", RUN / "stage1" / "best.pt"):
        if p.exists():
            return p
    raise FileNotFoundError(f"no checkpoint under {RUN}")


def load(ckpt: str | Path | None = None, device: str | None = None, data: str = DATA_ROOT):
    """(emulator, meta). The emulator is in eval mode, float32, on ``device`` (cuda if any).
    The checkpoint is ``ckpt``, else ``$PLAYGROUND_CKPT``, else the run's best."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    ckpt = ckpt or os.environ.get("PLAYGROUND_CKPT")
    path = Path(ckpt) if ckpt else default_checkpoint()
    em, ck = Emulator.load(path, Stats(data), meta(data)["lat"], map_location=device)
    em = em.to(device).eval()
    for p in em.parameters():
        p.requires_grad_(False)
    print(f"loaded {path} (epoch {ck.get('epoch')}, step {ck.get('step')}) on {device}")
    return em, meta(data)


def _year(year: int, data: str = DATA_ROOT):
    arr = np.load(Path(data) / f"{year}.npy", mmap_mode="r")
    times = json.loads((Path(data) / f"{year}.times.json").read_text())
    return arr, times


def _index(t: str | int, times: list[str]) -> int:
    if isinstance(t, (int, np.integer)):
        return int(t)
    for i, label in enumerate(times):
        if label.startswith(str(t)):
            return i
    raise KeyError(f"no time starting with {t!r} (have {times[0]} .. {times[-1]})")


def state(t: str | int = 0, year: int = 2010, device=None, n: int = 1) -> torch.Tensor:
    """Physical state + forcing at time ``t`` (index or ISO prefix like "2010-01-15T06"),
    shaped (1, 39, 180, 360); ``n`` > 1 stacks n consecutive 6-hourly states."""
    arr, times = _year(year)
    i = _index(t, times)
    x = torch.from_numpy(np.array(arr[i : i + n]))
    return x.to(device or _device())


def truth(t: str | int = 0, steps: int = 1, year: int = 2010, device=None) -> torch.Tensor:
    """The E3SM trajectory from ``t``: (1, steps + 1, 39, 180, 360)."""
    arr, times = _year(year)
    i = _index(t, times)
    return torch.from_numpy(np.array(arr[i : i + steps + 1]))[None].to(device or _device())


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def step(em, x, hooks=None, cache=None, names=None):
    """(next prognostic state, 6-h tendency), both physical, (B, 34, 180, 360)."""
    nxt, d = em.step(x, hooks=hooks, cache=cache, names=names)
    return nxt, d[:, :N_PROG].float() * em.res_std


@torch.no_grad()
def rollout(em, t: str | int = 0, steps: int = 20, year: int = 2010):
    """(emulated, true) prognostic trajectories, each (steps + 1, 34, 180, 360), from E3SM's
    state at ``t`` with true forcings and prescribed ocean TS."""
    tr = truth(t, steps, year, device=next(em.parameters()).device)
    states, _ = em.rollout(tr, steps)
    return states[0].cpu(), tr[0, :, :N_PROG].cpu()


# -- channels, regions, the token grid ---------------------------------------------------------


def ch(name: str) -> int:
    """Index of a prognostic channel, e.g. ch("V_3")."""
    return PROG_NAMES.index(name)


def lat_lon():
    m = meta()
    return np.asarray(m["lat"]), np.asarray(m["lon"])


def box(lat0: float, lat1: float, lon0: float, lon1: float) -> np.ndarray:
    """Boolean (180, 360) mask; longitudes 0..360, a box may wrap (lon0 > lon1)."""
    lat, lon = lat_lon()
    la = (lat >= lat0) & (lat <= lat1)
    lo = (lon >= lon0) & (lon <= lon1) if lon0 <= lon1 else (lon >= lon0) | (lon <= lon1)
    return la[:, None] & lo[None, :]


def _area(lat=None):
    lat = lat_lon()[0] if lat is None else lat
    return np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 360))


def box_mean(field, mask: np.ndarray) -> float:
    """Area-weighted mean of a (180, 360) array (or tensor) over a mask."""
    f = field.detach().cpu().numpy() if torch.is_tensor(field) else np.asarray(field)
    w = _area() * mask
    return float((f * w).sum() / w.sum())


def token_latlon(patch: int = 4):
    """Centers of the patch tokens: lat (45,), lon (90,)."""
    lat, lon = lat_lon()
    return lat.reshape(-1, patch).mean(1), lon.reshape(-1, patch).mean(1)


def token_of(lat: float, lon: float, patch: int = 4) -> int:
    """Index (row-major on the 45 x 90 grid) of the token containing a point."""
    tl, tn = token_latlon(patch)
    i = int(np.argmin(np.abs(tl - lat)))
    j = int(np.argmin(np.abs(((tn - lon % 360) + 180) % 360 - 180)))
    return i * len(tn) + j


# -- maps ------------------------------------------------------------------------------------


def _axes(n: int, cols: int, size=(5.2, 2.8)):
    import matplotlib.pyplot as plt

    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(size[0] * cols, size[1] * rows))
    try:
        import cartopy.crs as ccrs

        axes = [fig.add_subplot(rows, cols, i + 1, projection=ccrs.PlateCarree(180))
                for i in range(n)]
        return fig, axes, ccrs.PlateCarree()
    except Exception:
        return fig, [fig.add_subplot(rows, cols, i + 1) for i in range(n)], None


def _draw(ax, f, lat, lon, tr, title, cmap, symmetric, vmax=None):
    f = f.detach().cpu().numpy() if torch.is_tensor(f) else np.asarray(f)
    kw = {"cmap": cmap, "shading": "auto", "rasterized": True}
    if symmetric:
        v = vmax if vmax is not None else float(np.nanpercentile(np.abs(f), 99.5)) or 1.0
        kw.update(vmin=-v, vmax=v)
    elif vmax is not None:
        kw.update(vmax=vmax)
    if tr is not None:
        im = ax.pcolormesh(lon, lat, f, transform=tr, **kw)
        try:
            ax.coastlines(lw=0.4)
        except Exception:
            pass
    else:
        im = ax.pcolormesh(lon, lat, f, **kw)
    ax.set_title(title, fontsize=9)
    ax.figure.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    return im


def plot_map(field, title: str = "", cmap: str = "RdBu_r", symmetric: bool = True, vmax=None):
    """One (180, 360) or token-grid (45, 90) map. Diverging and centered by default."""
    return plot_maps({title: field}, cmap=cmap, symmetric=symmetric, cols=1, vmax=vmax)


def plot_maps(fields: dict, cmap: str = "RdBu_r", symmetric: bool = True, cols: int = 2,
              vmax=None, shared: bool = False):
    """Several maps, titled by the dict's keys. ``shared=True`` puts them on one color scale."""
    fig, axes, tr = _axes(len(fields), min(cols, len(fields)))
    if shared and symmetric and vmax is None:
        vals = [np.abs(f.detach().cpu().numpy() if torch.is_tensor(f) else np.asarray(f))
                for f in fields.values()]
        vmax = float(max(np.nanpercentile(v, 99.5) for v in vals)) or 1.0
    for ax, (title, f) in zip(axes, fields.items()):
        shape = tuple(f.shape)
        lat, lon = lat_lon() if shape == (180, 360) else token_latlon()
        _draw(ax, f, lat, lon, tr, title, cmap, symmetric, vmax)
    fig.tight_layout()
    return fig


def plot_tokens(values, title: str = "", cmap: str = "viridis", mark=None, log: bool = False):
    """A (45, 90) or (4050,) token map, e.g. an attention pattern; ``mark=(lat, lon)`` puts a
    star on the query."""
    v = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    v = v.reshape(45, 90)
    if log:
        v = np.log10(np.maximum(v, 1e-6))
    fig = plot_maps({title: v}, cmap=cmap, symmetric=False, cols=1)
    if mark is not None:
        ax = fig.axes[0]
        kw = {}
        try:
            import cartopy.crs as ccrs

            kw["transform"] = ccrs.PlateCarree()
        except Exception:
            pass
        ax.plot(mark[1] % 360, mark[0], marker="*", color="red", ms=12, **kw)
    return fig


# -- attention ---------------------------------------------------------------------------------


@torch.no_grad()
def patterns(em, x, layer: int) -> torch.Tensor:
    """Attention patterns of one layer: (B, heads, 4050, 4050), rows are queries."""
    name = f"blocks.{layer}.attn.hook_pattern"
    cache: dict = {}
    em.step(x, cache=cache, names={name})
    return cache[name].float()


def attention(em, x, layer: int, head: int, query: tuple[float, float]) -> np.ndarray:
    """Where the token at ``query=(lat, lon)`` looks, in head ``head`` of ``layer``: (45, 90)."""
    q = token_of(*query)
    return patterns(em, x[:1], layer)[0, head, q].reshape(45, 90).cpu().numpy()


def _offsets(device):
    tl, tn = token_latlon()
    LA, LO = np.meshgrid(tl, tn, indexing="ij")
    la, lo = LA.ravel(), LO.ravel()
    dx = ((lo[None, :] - lo[:, None]) + 180) % 360 - 180  # key minus query, east positive
    dy = la[None, :] - la[:, None]
    dist = np.hypot(dx * np.cos(np.deg2rad(la))[:, None], dy)  # degrees, roughly
    as_t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    return as_t(dx), as_t(dy), as_t(dist), la


@torch.no_grad()
def head_atlas(em, xs: torch.Tensor, layers: Iterable[int] | None = None,
               bands: dict | None = None):
    """A table of what every head does, averaged over the states in ``xs`` and the query
    tokens in each latitude band:

    dx, dy    attention-weighted mean offset of what is looked at (degrees; dx > 0 = east,
              i.e. a head with dx < 0 looks *upstream* in westerlies)
    dist      attention-weighted mean distance (degrees)
    self      weight on the query's own token
    entropy   of the pattern, in nats (log 4050 = 8.3 is uniform)
    """
    import pandas as pd

    bands = bands or {"all": (-90, 90), "NH mid": (30, 60), "tropics": (-20, 20),
                      "SH mid": (-60, -30)}
    dev = xs.device
    dx, dy, dist, la = _offsets(dev)
    layers = range(em.cfg.depth) if layers is None else layers
    rows = []
    eye = torch.eye(dx.shape[0], device=dev, dtype=torch.bool)
    for layer in layers:
        acc = None
        for b in range(xs.shape[0]):
            p = patterns(em, xs[b : b + 1], layer)[0]  # (H, N, N)
            stats = torch.stack([
                (p * dx).sum(-1), (p * dy).sum(-1), (p * dist).sum(-1),
                p[:, eye], -(p * torch.log(p.clamp_min(1e-12))).sum(-1),
            ])  # (5, H, N)
            acc = stats if acc is None else acc + stats
        acc = (acc / xs.shape[0]).cpu().numpy()
        for h in range(acc.shape[1]):
            for band, (lo, hi) in bands.items():
                q = (la >= lo) & (la <= hi)
                rows.append({"layer": layer, "head": h, "band": band,
                             **{k: float(acc[i, h, q].mean()) for i, k in
                                enumerate(["dx", "dy", "dist", "self", "entropy"])}})
    return pd.DataFrame(rows)


# -- components: ablation and attribution ------------------------------------------------------


def components(em) -> list[str]:
    """Component names: "embed", "L{i}H{h}" heads, "L{i}MLP"."""
    out = ["embed"]
    for i in range(em.cfg.depth):
        out += [f"L{i}H{h}" for h in range(em.cfg.heads)] + [f"L{i}MLP"]
    return out


def _parse(name: str):
    if name.endswith("MLP"):
        return int(name[1:-3]), None
    i, h = name[1:].split("H")
    return int(i), int(h)


@dataclass
class Ablation:
    delta: torch.Tensor  # change in the 6-h tendency, physical (34, 180, 360), batch mean
    base: torch.Tensor  # the unablated tendency, physical (34, 180, 360), batch mean

    def rel_change(self) -> dict:
        """Area-weighted RMS of the change over that of the tendency, per channel."""
        w = torch.tensor(_area(), dtype=torch.float32)
        d = (self.delta.cpu() ** 2 * w).sum((1, 2)).sqrt()
        b = (self.base.cpu() ** 2 * w).sum((1, 2)).sqrt()
        return {n: float(v) for n, v in zip(PROG_NAMES, d / b)}


@torch.no_grad()
def ablate(em, x, heads: Sequence[tuple[int, int]] = (), mlps: Sequence[int] = (),
           mode: str = "mean", names: Sequence[str] = ()) -> Ablation:
    """Knock out heads ((layer, head) pairs) and MLPs (layers), or ``names`` like "L3H2",
    "L5MLP". ``mode="zero"`` writes nothing in their place; ``"mean"`` writes their mean over
    tokens (keeps the average contribution, removes the spatial pattern)."""
    heads, mlps = list(heads), list(mlps)
    for n in names:
        i, h = _parse(n)
        (mlps.append(i) if h is None else heads.append((i, h)))
    by_layer: dict[int, list[int]] = {}
    for i, h in heads:
        by_layer.setdefault(i, []).append(h)
    hooks = {}

    def knock(t, idx=None):
        t = t.clone()
        sl = (slice(None), slice(None), idx) if idx is not None else (Ellipsis,)
        if mode == "zero":
            t[sl] = 0
        elif mode == "mean":
            t[sl] = t[sl].mean(1, keepdim=True).expand_as(t[sl])
        else:
            raise ValueError("mode is 'zero' or 'mean'")
        return t

    for i, hs in by_layer.items():
        hooks[f"blocks.{i}.attn.hook_result"] = lambda t, hs=hs: knock(t, hs)
    for i in mlps:
        hooks[f"blocks.{i}.hook_mlp_out"] = lambda t: knock(t)
    _, base = step(em, x)
    _, new = step(em, x, hooks=hooks)
    return Ablation((new - base).mean(0), base.mean(0))


@torch.no_grad()
def sweep_components(em, x, metric, mode: str = "mean", which: Iterable[str] | None = None):
    """Ablate each component alone and record ``metric(tendency)`` -> a DataFrame sorted by
    effect. ``metric`` takes the physical tendency (B, 34, 180, 360) and returns a float, e.g.
    ``lambda t: pg.box_mean(t[0, pg.ch("V_3")], mask)``."""
    import pandas as pd

    _, base = step(em, x)
    m0 = metric(base)
    rows = []
    for name in which or components(em)[1:]:
        a = ablate(em, x, names=[name], mode=mode)
        rows.append({"component": name, "metric": metric(base + a.delta[None]),
                     "effect": metric(base + a.delta[None]) - m0})
    df = pd.DataFrame(rows)
    df.attrs["baseline"] = m0
    return df.reindex(df.effect.abs().sort_values(ascending=False).index)


@torch.no_grad()
def lens(em, x):
    """Decode the residual stream after every block with the final LayerNorm + head (the
    "logit lens"). Returns (decoded, table): decoded[k] is the physical tendency the model
    would output if it stopped after block k-1 (k = 0: the embedding alone), and the table has,
    per layer and channel group, the RMSE against the real output relative to its RMS."""
    import pandas as pd

    names = {"hook_embed", "hook_pos_embed"} | {
        f"blocks.{i}.hook_resid_post" for i in range(em.cfg.depth)}
    cache: dict = {}
    _, final = step(em, x, cache=cache, names=names)
    rs = [cache["hook_embed"] + cache["hook_pos_embed"]] + [
        cache[f"blocks.{i}.hook_resid_post"] for i in range(em.cfg.depth)]
    blank = _Ctx(None, None, None)
    decoded = [em.net.unembed(r, blank)[:, :N_PROG].float() * em.res_std for r in rs]
    w = torch.tensor(_area(), dtype=torch.float32, device=final.device)
    groups = {"PS/TS": ["PS", "TS"], "T": [f"T_{k}" for k in range(8)],
              "U": [f"U_{k}" for k in range(8)], "V": [f"V_{k}" for k in range(8)],
              "water": [f"specific_total_water_{k}" for k in range(8)]}
    rows = []
    for k, d in enumerate(decoded):
        row = {"after": "embed" if k == 0 else f"block {k - 1}"}
        for g, ns in groups.items():
            idx = [ch(n) for n in ns]
            err = ((d[:, idx] - final[:, idx]) ** 2 * w).mean((0, 2, 3))
            ref = (final[:, idx] ** 2 * w).mean((0, 2, 3))
            row[g] = float((err / ref).sqrt().mean())
        rows.append(row)
    return [d.cpu() for d in decoded], pd.DataFrame(rows)


@torch.no_grad()
def direct_attribution(em, x, channel: str, mask: np.ndarray | None = None):
    """Split the output tendency of ``channel`` into what each component writes directly.

    The final residual stream is a sum: embed + every head's output + attention biases +
    every MLP's output. The final LayerNorm divides by a per-token scale that depends on all
    of them; freezing that scale at its real value makes the split exact and linear (the
    standard "direct logit attribution" approximation: it says what each component writes,
    not what it causes through later layers -- ablation does that).

    Returns (maps, table): maps[name] is that component's (180, 360) contribution to the
    physical tendency (batch mean), and the table ranks components by their contribution
    to the area mean over ``mask`` (whole globe without one). The contributions sum to the
    real output.
    """
    import pandas as pd

    L = em.cfg.depth
    names = {"hook_embed", "hook_pos_embed", f"blocks.{L - 1}.hook_resid_post"}
    for i in range(L):
        names |= {f"blocks.{i}.attn.hook_result", f"blocks.{i}.hook_mlp_out"}
    cache: dict = {}
    _, final = step(em, x, cache=cache, names=names)
    r = cache[f"blocks.{L - 1}.hook_resid_post"]
    ln = em.net.ln_final
    scale = (r.var(-1, keepdim=True, unbiased=False) + ln.eps).sqrt()
    c = ch(channel)
    C = em.cfg.out_chans
    W = em.net.head.weight  # (C * p * p, D)
    k = getattr(em.cfg, "head_conv", 0)

    def decode(t):  # (B, N, C*p*p) head output -> channel c, after the output conv if any
        y = em.net.unpatchify(t, C)
        if k:  # the conv mixes channels, so decode all of them before taking c (it is linear)
            y = torch.nn.functional.pad(y, (k // 2, k // 2, 0, 0), mode="circular")
            y = torch.nn.functional.pad(y, (0, 0, k // 2, k // 2), mode="replicate")
            y = em.net.out_conv(y)
        return y[:, c]

    def write(contrib):  # (B, N, D) component -> (180, 360) physical tendency, batch mean
        z = (contrib - contrib.mean(-1, keepdim=True)) / scale * ln.weight
        return (decode(z @ W.T).float() * em.res_std[0, c]).mean(0)

    maps = {"embed": write(cache["hook_embed"] + cache["hook_pos_embed"])}
    b_O = sum(em.net.blocks[i].attn.b_O for i in range(L))
    for i in range(L):
        res = cache[f"blocks.{i}.attn.hook_result"]
        for h in range(em.cfg.heads):
            maps[f"L{i}H{h}"] = write(res[:, :, h])
        maps[f"L{i}MLP"] = write(cache[f"blocks.{i}.hook_mlp_out"])
    const = (ln.bias @ W.T + em.net.head.bias).expand(1, r.shape[1], -1)  # (1, N, C*p*p)
    const_map = (decode(const).float() * em.res_std[0, c])[0]
    maps["biases"] = write(b_O.expand_as(r)) + const_map
    total = sum(maps.values())
    err = float((total - final[:, c].mean(0)).abs().max())
    mask = np.ones((180, 360), dtype=bool) if mask is None else mask
    rows = [{"component": k, "contribution": box_mean(v, mask)} for k, v in maps.items()]
    df = pd.DataFrame(rows)
    df = df.reindex(df.contribution.abs().sort_values(ascending=False).index)
    df.attrs.update(total=box_mean(final[:, c].mean(0), mask), max_abs_reconstruction_error=err)
    return {k: v.cpu() for k, v in maps.items()}, df
