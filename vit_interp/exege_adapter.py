"""exege adapters for the ViT emulator: the trained model, reachable by name from exege.

    from exege.circuits import open_patchable, prune, field_mean
    system = open_patchable("runs/.../best.pt", "vit-interp", year=2010)      # Patchable
    from exege.latents import open_intervenable
    tokens = open_intervenable("runs/.../best.pt", "vit-interp-tokens")     # Intervenable

Two classes, because exege asks one adapter for one grid:

- ``ViTEmulator`` (``vit-interp``) works on the native 180x360 grid: ``circuits.Patchable``
  and ``circuits.EdgeGradients``. A step writes the next physical state of the 34
  prognostic channels. Senders and receivers are exege's ``residual_graph(depth, heads)``,
  in the same order as ``vit_interp.circuits.Graph``.
- ``ViTTokens`` (``vit-interp-tokens``) works on the 45x90 token grid:
  ``latents.Intervenable``, with layer ``i`` = ``blocks.{i}.hook_resid_post``.

**The persistence skip.** The emulator's next state is ``x + net(x) * res_std``: the
``x +`` sits outside the network. Patching contracts want a source patch of *everything*
to be exactly the source's step, so the skip is carried by the input's direct path to
the output: under an ``EdgePatch`` it follows edge ``embed->out`` (``mask[embed, out]``
of the base's ``x`` and the rest of the source's), and under a ``ComponentPatch`` it
follows the component ``embed``. A patch that leaves ``embed`` alone keeps the base's
``x``, so ``step(base, patch) - step(base)`` is then the change in the network's
tendency alone.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exege.circuits.contract import ComponentPatch, EdgePatch
from exege.circuits.graph import INPUT, OUTPUT, Graph, residual_graph
from exege.core.errors import AdapterError, RequestError
from exege.latents.grid import Grid
from exege.latents.source import LatentInfo, LayerInfo
from exege.latents.steering import Hook, Rollout

from . import circuits as C
from .data import ALL_NAMES, DATA_ROOT, N_ALL, N_PROG, PROG_NAMES, Stats, ensure_data, resolve_checkpoint
from .emulator import OCNFRAC, TS, Emulator
from .model import token_latlon

TOKEN_FIELDS = ("PS", "TS", "T_7", "U_3", "V_3", "specific_total_water_7")
_DTYPES = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.float32}


@dataclass(frozen=True)
class ViTState:
    """A batch of physical input states ``(B, 39, 180, 360)`` on the model's device, with
    the year positions they were read from (None once perturbed). Opaque to exege."""

    x: torch.Tensor
    starts: tuple[int | None, ...]

    @property
    def batch(self) -> int:
        return int(self.x.shape[0])


class _Base:
    def __init__(
        self,
        source: str,
        *,
        data: str = DATA_ROOT,
        year: int = 2010,
        device: str | None = None,
        dtype: str = "float32",
        batch_limit: int | None = None,
    ) -> None:
        try:
            path = resolve_checkpoint(source)  # a path, or v1 / v2 (downloaded off LCRC)
        except FileNotFoundError:
            raise AdapterError(f"no checkpoint at {source} (a path, or v1 / v2)") from None
        if data == DATA_ROOT and year == 2010:
            data = ensure_data(data, year)
        if dtype not in _DTYPES:
            raise RequestError(f"dtype must be one of {', '.join(_DTYPES)}, not {dtype!r}")
        root = Path(data)
        try:
            self.meta = json.loads((root / "meta.json").read_text())
            stats = Stats(str(root))
            self._times = tuple(json.loads((root / f"{year}.times.json").read_text()))
            self._arr = np.load(root / f"{year}.npy", mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise AdapterError(f"cannot read the data for {year} under {data}: {exc}") from exc
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            em, ck = Emulator.load(path, stats, self.meta["lat"], map_location="cpu")
        except Exception as exc:  # a file that is not one of our checkpoints
            raise AdapterError(f"{source} is not a vit-interp checkpoint: {exc}") from exc
        self.dtype = _DTYPES[dtype]
        self.autocast = dtype == "bfloat16" and self.device.type == "cuda"
        self.em = em.to(self.device, self.dtype).eval()
        self.cfg = em.cfg
        self.checkpoint = str(path.resolve())
        self.epoch = ck.get("epoch")
        self.year = year
        self.batch_limit = batch_limit
        self._options = {"data": str(data), "year": year, "dtype": dtype}

    # -- shared --

    def _ctx(self):
        if self.autocast:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _position(self, start: str | int) -> int:
        if isinstance(start, str):
            if start in self._times:
                return self._times.index(start)
            hits = [i for i, label in enumerate(self._times) if label.startswith(start)]
            if len(hits) == 1:  # a unique prefix: "2010-01-15T06" or "2010-01-15T06:00"
                return hits[0]
            if hits:
                raise RequestError(
                    f"{start!r} matches {len(hits)} times ({self._times[hits[0]]} .. "
                    f"{self._times[hits[-1]]}); give the hour, e.g. {self._times[hits[0]][:13]}"
                )
            raise RequestError(
                f"no time {start!r} in {self.year}; times run {self._times[0]} .. "
                f"{self._times[-1]}"
            )
        t = int(start)
        if not 0 <= t < len(self._times):
            raise RequestError(f"start {start} is not a position 0..{len(self._times) - 1}")
        return t

    def _plain(self, x: torch.Tensor, hooks=None) -> torch.Tensor:
        """Next prognostic state, in the state's own precision (Emulator.step casts the
        tendency to float32, which would break float64 exactness)."""
        with self._ctx():
            d = self.em.net(self.em.normalize(x), hooks=hooks)
        return x[:, :N_PROG] + d[:, :N_PROG].to(x.dtype) * self.em.res_std

    def _read(self, t: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self._arr[t])).to(self.device, self.dtype)


class ViTEmulator(_Base):
    """``circuits.Patchable`` and ``circuits.EdgeGradients`` on the native grid."""

    def __init__(self, source: str, **kwargs: Any) -> None:
        super().__init__(source, **kwargs)
        self._graph = residual_graph(self.cfg.depth, self.cfg.heads)
        if list(self._graph.senders) != C.graph_of(self.em.net).senders or list(
            self._graph.receivers
        ) != C.graph_of(self.em.net).receivers:
            raise AdapterError("exege's residual graph and the model's sender order disagree")
        lat, lon = np.asarray(self.meta["lat"]), np.asarray(self.meta["lon"])
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        self._grid = Grid(lat=LA.ravel(), lon=LO.ravel(), shape=(len(lat), len(lon)))
        self._embed = self._graph.sender(INPUT)
        self._out = self._graph.receiver(OUTPUT)

    def graph(self) -> Graph:
        return self._graph

    def grid(self) -> Grid:
        return self._grid

    def field_names(self) -> tuple[str, ...]:
        return tuple(PROG_NAMES)

    def states(self, starts: Sequence[str | int]) -> ViTState:
        starts = list(starts)
        if not starts:
            raise RequestError("no starts")
        if self.batch_limit is not None and len(starts) > self.batch_limit:
            raise RequestError(f"at most {self.batch_limit} states at once")
        pos = [self._position(s) for s in starts]
        return ViTState(torch.stack([self._read(t) for t in pos]), tuple(pos))

    def fields(self, state: ViTState) -> dict[str, np.ndarray]:
        x = self._check_state(state).x.detach().double().cpu().numpy().reshape(state.batch, len(ALL_NAMES), -1)
        return {name: x[:, c].copy() for c, name in enumerate(ALL_NAMES)}

    def perturb(self, state: ViTState, delta: Mapping[str, np.ndarray]) -> ViTState:
        self._check_state(state)
        n = self._grid.n_nodes
        x = state.x.clone()
        H, W = x.shape[-2:]
        for name, d in delta.items():
            if name not in ALL_NAMES:
                raise RequestError(f"no field {name!r}; fields are the 39 of vit_interp.data.ALL_NAMES")
            d = np.asarray(d, dtype=np.float64)
            if d.shape not in ((n,), (state.batch, n)):
                raise RequestError(f"a delta is (n_nodes,) or (batch, n_nodes), got {d.shape}")
            t = torch.as_tensor(d, device=x.device, dtype=x.dtype)
            x[:, ALL_NAMES.index(name)] += t.reshape(-1, H, W) if t.ndim == 2 else t.reshape(H, W)
        return ViTState(x, (None,) * state.batch)

    @torch.no_grad()
    def step(self, state: ViTState, patch=None) -> dict[str, np.ndarray]:
        return self._out_dict(self._step(state, patch))

    def edge_gradients(self, state: ViTState, source: ViTState, weights) -> np.ndarray:
        self._check_pair(state, source)
        w = {}
        for name, v in weights.items():
            if name not in PROG_NAMES:
                raise RequestError(f"a step writes no field {name!r}")
            v = np.asarray(v, dtype=np.float64)
            if v.shape != (self._grid.n_nodes,):
                raise RequestError(f"weights are (n_nodes,), got {v.shape} for {name}")
            w[PROG_NAMES.index(name)] = torch.as_tensor(v, device=self.device, dtype=self.dtype)
        mask = torch.as_tensor(self._graph.mask(), device=self.device, dtype=self.dtype)
        mask.requires_grad_(True)
        with torch.enable_grad():
            nxt = self._edge_step(state, source, mask).flatten(2)  # (B, N_PROG, n)
            metric = sum((nxt[:, c] * v).sum(-1) for c, v in w.items()).mean()
            metric.backward()
        g = mask.grad.detach().double().cpu().numpy()
        return np.where(self._graph.edges, g, 0.0)

    # -- internals --

    def _check_state(self, state) -> ViTState:
        if not isinstance(state, ViTState):
            raise RequestError("not a state of this system; get one from states()")
        return state

    def _check_pair(self, state, source) -> None:
        self._check_state(state)
        if not isinstance(source, ViTState) or source.batch != state.batch:
            raise RequestError("a patch's source must be a state of the same batch size")

    def _step(self, state: ViTState, patch) -> torch.Tensor:
        self._check_state(state)
        if patch is None:
            return self._plain(state.x)
        if isinstance(patch, EdgePatch):
            self._check_pair(state, patch.source)
            m = torch.as_tensor(self._graph.check_mask(patch.mask), device=self.device, dtype=self.dtype)
            return self._edge_step(state, patch.source, m)
        if isinstance(patch, ComponentPatch):
            self._check_pair(state, patch.source)
            comps = self._graph.check_components(patch.components)
            return self._component_step(state, patch.source, comps)
        raise RequestError(f"unknown patch {type(patch).__name__}")

    def _edge_step(self, state: ViTState, source: ViTState, mask: torch.Tensor) -> torch.Tensor:
        em = self.em
        with self._ctx():
            corrupt = C.sender_outputs(em.net, em.normalize(source.x))
            d = C.edge_forward(em.net, em.normalize(state.x), corrupt, mask)
        keep = mask[self._embed, self._out]
        skip = source.x[:, :N_PROG] + keep * (state.x[:, :N_PROG] - source.x[:, :N_PROG])
        return skip + d[:, :N_PROG].to(skip.dtype) * em.res_std

    def _component_step(self, state: ViTState, source: ViTState, comps: tuple[str, ...]) -> torch.Tensor:
        em = self.em
        if not comps:
            return self._plain(state.x)
        names, heads = set(), {}
        for c in comps:
            if c == INPUT:
                names.add("hook_embed")
            elif c.startswith("m"):
                names.add(f"blocks.{int(c[1:])}.hook_mlp_out")
            else:
                layer, h = c[1:].split(".h")
                heads.setdefault(int(layer), []).append(int(h))
                names.add(f"blocks.{int(layer)}.attn.hook_result")
        src: dict = {}
        with self._ctx():
            em.net(em.normalize(source.x), cache=src, names=names)
        hooks = {}
        for k in names:
            if k.endswith("hook_result"):
                hs = heads[int(k.split(".")[1])]

                def swap(t, k=k, hs=hs):
                    t = t.clone()
                    t[:, :, hs] = src[k][:, :, hs]
                    return t

                hooks[k] = swap
            else:
                hooks[k] = lambda t, k=k: src[k]
        with self._ctx():
            d = em.net(em.normalize(state.x), hooks=hooks)
        base_x = source.x if INPUT in comps else state.x
        return base_x[:, :N_PROG] + d[:, :N_PROG].to(base_x.dtype) * em.res_std

    def _out_dict(self, nxt: torch.Tensor) -> dict[str, np.ndarray]:
        a = nxt.detach().double().cpu().numpy().reshape(nxt.shape[0], N_PROG, -1)
        return {name: a[:, c].copy() for c, name in enumerate(PROG_NAMES)}


@dataclass(frozen=True)
class _Start:
    t: int


class ViTTokens(_Base):
    """``latents.Intervenable`` on the 45x90 token grid; layer ``i`` is the residual
    stream after block ``i``. Runs are deterministic: ``noise_seed`` changes nothing. A run
    reads true forcings and prescribes the ocean's TS from the year it was opened on."""

    def __init__(self, source: str, **kwargs: Any) -> None:
        super().__init__(source, **kwargs)
        lat, lon, shape = token_latlon(self.meta["lat"], self.meta["lon"], self.cfg.patch)
        self._grid = Grid(lat=lat.astype(np.float64), lon=lon.astype(np.float64), shape=shape)
        self._info = LatentInfo(
            source=self.checkpoint,
            times=self._times,
            layers=tuple(
                LayerInfo(i, f"blocks.{i}.resid_post", self.cfg.dim) for i in range(self.cfg.depth)
            ),
            n_nodes=self._grid.n_nodes,
            model="vit-interp E3SMv3-1deg-AMIP",
            component="atmosphere",
            checkpoint=self.checkpoint,
            calendar="noleap",
            timestep_seconds=21600,
            experiment={"epoch": self.epoch, "patch": self.cfg.patch, "dim": self.cfg.dim,
                        "depth": self.cfg.depth, "heads": self.cfg.heads},
            options=self._options,
        )

    def info(self) -> LatentInfo:
        return self._info

    def grid(self) -> Grid:
        return self._grid

    def initial_state(self, start: str | int = 0) -> _Start:
        return _Start(self._position(start))

    def _coarsen(self, x: torch.Tensor) -> np.ndarray:
        p = self.cfg.patch
        H, W = x.shape[-2:]
        return x.reshape(H // p, p, W // p, p).mean((1, 3)).reshape(-1).double().cpu().numpy()

    @torch.no_grad()
    def run(
        self,
        initial_state: Any,
        steps: int,
        hooks: Sequence[Hook] = (),
        *,
        noise_seed: int | None = None,
        record: Sequence[tuple[int, int]] = (),
    ) -> Rollout:
        if not isinstance(initial_state, _Start):
            raise RequestError("not a state of this system; get one from initial_state()")
        if steps < 1:
            raise RequestError("steps must be at least 1")
        t0 = initial_state.t
        if t0 + steps >= len(self._times):
            raise RequestError(f"a run of {steps} from {self._times[t0]} leaves {self.year}")
        depth = self.cfg.depth
        edits: dict[tuple[int, int], list] = {}
        for hk in hooks:
            if not 0 <= hk.layer < depth or not 0 <= hk.time < steps:
                raise RequestError(f"no place at layer {hk.layer}, time {hk.time}")
            edits.setdefault((hk.layer, hk.time), []).append(hk.edit)
        for layer, t in record:
            if not 0 <= layer < depth or not 0 <= t < steps:
                raise RequestError(f"cannot record layer {layer} at time {t}")
        record = set(map(tuple, record))
        kept: dict[tuple[int, int], np.ndarray] = {}
        fields = {n: np.empty((steps, self._grid.n_nodes)) for n in TOKEN_FIELDS}
        state = self._read(t0)[None, :N_PROG]
        for t in range(steps):
            forcing = self._read(t0 + t)[None, N_PROG:N_ALL]
            hk_t = {}
            for layer in range(depth):
                if (layer, t) not in edits and (layer, t) not in record:
                    continue

                def hook(h, layer=layer, t=t):
                    a = h[0].float().cpu().numpy()
                    for edit in edits.get((layer, t), ()):
                        a = np.asarray(edit(a.copy()), dtype=np.float32)
                        if a.shape != tuple(h.shape[1:]):
                            raise RequestError(f"an edit returned {a.shape}, not {tuple(h.shape[1:])}")
                    if (layer, t) in record:
                        kept[(layer, t)] = a.astype(np.float32).copy()
                    return torch.as_tensor(a, device=h.device, dtype=h.dtype)[None]

                hk_t[f"blocks.{layer}.hook_resid_post"] = hook
            nxt = self._plain(torch.cat([state, forcing], 1), hooks=hk_t or None)
            truth = self._read(t0 + t + 1)
            ocean = truth[OCNFRAC] > 0.5
            nxt[0, TS] = torch.where(ocean, truth[TS], nxt[0, TS])
            state = nxt.to(self.dtype)
            for n in TOKEN_FIELDS:
                fields[n][t] = self._coarsen(state[0, PROG_NAMES.index(n)])
        return Rollout(fields=fields, latents=kept)
