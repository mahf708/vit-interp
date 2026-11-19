"""Physical-units wrapper around the ViT: normalization, one step, rollouts, checkpoints."""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn as nn

from .data import ALL_NAMES, N_PROG, PROG_NAMES, Stats
from .model import ViT, ViTConfig

TS = PROG_NAMES.index("TS")
OCNFRAC = ALL_NAMES.index("OCNFRAC")


class Emulator(nn.Module):
    def __init__(self, cfg: ViTConfig, stats: Stats, lat: np.ndarray):
        super().__init__()
        self.cfg = cfg
        self.net = ViT(cfg)
        self.register_buffer("mean", torch.tensor(stats.mean)[None, :, None, None])
        self.register_buffer("std", torch.tensor(stats.std)[None, :, None, None])
        self.register_buffer("res_std", torch.tensor(stats.res_std)[None, :, None, None])
        w = torch.cos(torch.deg2rad(torch.tensor(np.asarray(lat), dtype=torch.float32)))
        self.register_buffer("area_w", (w / w.mean())[None, None, :, None])

    def normalize(self, x):
        return (x - self.mean) / self.std

    def residual_target(self, x_now, x_next):
        """Normalized 6-h tendency of the prognostic channels (the training target)."""
        return (x_next[:, :N_PROG] - x_now[:, :N_PROG]) / self.res_std

    def step(self, x, hooks=None, cache=None, names=None):
        """x: physical state+forcing (B, C_all, H, W) -> (next prognostic state, normalized tendency)."""
        d = self.net(self.normalize(x), hooks=hooks, cache=cache, names=names)
        return x[:, :N_PROG] + d.float() * self.res_std, d

    @torch.no_grad()
    def rollout(self, traj, n_steps: int, prescribe_ocean_ts: bool = True, hooks_at=None, record=None):
        """traj: (B, T>=n_steps+1, C_all, H, W) physical truth (forcings are read from it).

        hooks_at: optional fn(step) -> hooks dict.  record: optional set of hook names to
        cache at every step; returns (states (B, n_steps+1, N_PROG, H, W), [cache per step]).
        """
        state = traj[:, 0, :N_PROG]
        out, caches = [state], []
        for t in range(n_steps):
            x = torch.cat([state, traj[:, t, N_PROG:]], 1)
            cache = {} if record is not None else None
            state, _ = self.step(x, hooks=hooks_at(t) if hooks_at else None, cache=cache, names=record)
            if prescribe_ocean_ts:  # AMIP: SST is a boundary condition
                ocean = traj[:, t + 1, OCNFRAC] > 0.5
                state[:, TS] = torch.where(ocean, traj[:, t + 1, TS], state[:, TS])
            out.append(state)
            if cache is not None:
                caches.append({k: v.detach().cpu() for k, v in cache.items()})
        return torch.stack(out, 1), caches

    def save(self, path, **extra):
        torch.save({"cfg": dataclasses.asdict(self.cfg), "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, stats: Stats, lat, map_location="cpu"):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ck["cfg"]
        cfg["img_size"] = tuple(cfg["img_size"])
        em = cls(ViTConfig(**cfg), stats, lat)
        em.load_state_dict(ck["state_dict"])
        return em, ck
