"""Physical-units wrapper around the ViT: normalization, one step, rollouts, checkpoints.

The network reads the 39 input channels (34 prognostic + 5 forcing, normalized) and writes
``N_PROG + n_diag`` channels: the normalized 6-h *tendency* of the prognostic fields, then
(v2 models) ACE's diagnostics as normalized full fields (precipitation and energy fluxes
over the step). ``n_diag`` is 0 for v1 checkpoints.

An optional ``Corrector`` (ACE-style conservation: dry air, positivity, moisture and energy
budgets) acts on the physical output *after* the network, so circuits are unaffected; use
``step_corrected`` / ``rollout(..., correct=True)``. ``step`` itself is always uncorrected.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn as nn

from .data import ALL_NAMES, N_ALL, N_PROG, PROG_NAMES, Stats
from .model import ViT, ViTConfig, build_net

TS = PROG_NAMES.index("TS")
OCNFRAC = ALL_NAMES.index("OCNFRAC")


class Emulator(nn.Module):
    def __init__(self, cfg: ViTConfig, stats: Stats, lat: np.ndarray, corrector: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        self.net = build_net(cfg)
        self.n_diag = cfg.out_chans - N_PROG
        self.register_buffer("mean", torch.tensor(stats.mean[:N_ALL])[None, :, None, None])
        self.register_buffer("std", torch.tensor(stats.std[:N_ALL])[None, :, None, None])
        self.register_buffer("res_std", torch.tensor(stats.res_std)[None, :, None, None])
        if self.n_diag:
            # Placeholders when the stats have no diagnostics (a dataset without them): every
            # normalization buffer is saved in the checkpoint and restored by load_state_dict,
            # so a v2 checkpoint loads against v1 data (whose 39 input channels are the same).
            dm = getattr(stats, "diag_mean", np.zeros(0))
            ds = getattr(stats, "diag_std", np.zeros(0))
            if len(dm) != self.n_diag:
                dm, ds = np.zeros(self.n_diag, np.float32), np.ones(self.n_diag, np.float32)
            self.register_buffer("diag_mean", torch.tensor(dm)[None, :, None, None])
            self.register_buffer("diag_std", torch.tensor(ds)[None, :, None, None])
        w = torch.cos(torch.deg2rad(torch.tensor(np.asarray(lat), dtype=torch.float32)))
        self.register_buffer("area_w", (w / w.mean())[None, None, :, None])
        self.corrector = corrector  # not saved with the weights; attach after loading

    def normalize(self, x):
        return (x[:, :N_ALL] - self.mean) / self.std

    def residual_target(self, x_now, x_next):
        """Normalized 6-h tendency of the prognostic channels (the training target)."""
        return (x_next[:, :N_PROG] - x_now[:, :N_PROG]) / self.res_std

    def diag_target(self, x_next):
        """Normalized diagnostics at the output time (v2 targets)."""
        return (x_next[:, N_ALL:N_ALL + self.n_diag] - self.diag_mean) / self.diag_std

    def physical(self, x, d):
        """Network output d (normalized) -> (next prognostic state, diagnostics or None)."""
        d = d.float()
        nxt = x[:, :N_PROG] + d[:, :N_PROG] * self.res_std
        diag = d[:, N_PROG:] * self.diag_std + self.diag_mean if self.n_diag else None
        return nxt, diag

    def normalized(self, x, nxt, diag=None):
        """The inverse of ``physical``: corrected physical outputs back to network units."""
        d = (nxt - x[:, :N_PROG]) / self.res_std
        if self.n_diag and diag is not None:
            d = torch.cat([d, (diag - self.diag_mean) / self.diag_std], 1)
        return d

    def step(self, x, hooks=None, cache=None, names=None, noise=None, generator=None):
        """x: physical state+forcing (B, >=39, H, W) -> (next prognostic state, normalized output).
        Uncorrected. The normalized output holds the diagnostics after the tendencies (v2).
        Stochastic models (cfg.noise_dim > 0): ``noise`` (B, N, d) or a ``generator``."""
        d = self.net(self.normalize(x), hooks=hooks, cache=cache, names=names, noise=noise,
                     generator=generator)
        return x[:, :N_PROG] + d[:, :N_PROG].float() * self.res_std, d

    def step_corrected(self, x, ref_dry_mass=None, hooks=None, cache=None, names=None, forcing_next=None,
                       noise=None, generator=None):
        """One step through the corrector (if attached): (next prog, diagnostics, ref_dry_mass).
        ``forcing_next`` (B, 5, H, W): forcings at the output time (ACE's energy budget uses them)."""
        d = self.net(self.normalize(x), hooks=hooks, cache=cache, names=names, noise=noise,
                     generator=generator)
        nxt, diag = self.physical(x, d)
        if self.corrector is not None:
            nxt, diag, ref_dry_mass = self.corrector(x[:, :N_ALL], nxt, diag, ref_dry_mass, forcing_next)
        return nxt, diag, ref_dry_mass

    @torch.no_grad()
    def rollout(self, traj, n_steps: int, prescribe_ocean_ts: bool = True, hooks_at=None, record=None,
                correct: bool = False, seed: int | None = None, smooth=None):
        """traj: (B, T>=n_steps+1, C, H, W) physical truth (forcings are read from it).

        hooks_at: optional fn(step) -> hooks dict.  record: optional set of hook names to
        cache at every step; returns (states (B, n_steps+1, N_PROG, H, W), [cache per step]).
        correct: apply the attached corrector at every step (dry air pinned to the start).
        seed: stochastic models draw their noise from a generator with this seed (reproducible).
        smooth: a ``vit_interp.filters.SeamFilter`` applied to every step's increment (after the
        correctors, before ocean TS is prescribed); recommended for runs longer than a few weeks.
        """
        gen = None
        if seed is not None and self.cfg.noise_dim:
            gen = torch.Generator(device=traj.device).manual_seed(int(seed))
        state = traj[:, 0, :N_PROG]
        out, caches, ref = [state], [], None
        for t in range(n_steps):
            x = torch.cat([state, traj[:, t, N_PROG:N_ALL]], 1)
            cache = {} if record is not None else None
            hooks = hooks_at(t) if hooks_at else None
            if correct and self.corrector is not None:
                state, _, ref = self.step_corrected(x, ref, hooks=hooks, cache=cache, names=record,
                                                    forcing_next=traj[:, t + 1, N_PROG:N_ALL],
                                                    generator=gen)
            else:
                state, _ = self.step(x, hooks=hooks, cache=cache, names=record, generator=gen)
            if smooth is not None:
                state = smooth.apply(x[:, :N_PROG], state)
            if prescribe_ocean_ts:  # AMIP: SST is a boundary condition
                ocean = traj[:, t + 1, OCNFRAC] > 0.5
                state[:, TS] = torch.where(ocean, traj[:, t + 1, TS], state[:, TS])
            out.append(state)
            if cache is not None:
                caches.append({k: v.detach().cpu() for k, v in cache.items()})
        return torch.stack(out, 1), caches

    @torch.no_grad()
    def ensemble(self, x, n: int, seed: int | None = None, correct: bool = False, forcing_next=None):
        """n members of one step from each state in x (B, C, H, W): next prognostic states
        (B, n, N_PROG, H, W). A deterministic model gives n identical members."""
        B = x.shape[0]
        xx = x.unsqueeze(1).expand(B, n, *x.shape[1:]).reshape(B * n, *x.shape[1:])
        gen = torch.Generator(device=x.device).manual_seed(int(seed)) if seed is not None else None
        if correct and self.corrector is not None:
            fn = None if forcing_next is None else forcing_next.unsqueeze(1).expand(
                B, n, *forcing_next.shape[1:]).reshape(B * n, *forcing_next.shape[1:])
            nxt, _, _ = self.step_corrected(xx, forcing_next=fn, generator=gen)
        else:
            nxt, _ = self.step(xx, generator=gen)
        return nxt.reshape(B, n, *nxt.shape[1:])

    def save(self, path, **extra):
        sd = {k: v for k, v in self.state_dict().items() if not k.startswith("corrector.")}
        torch.save({"cfg": dataclasses.asdict(self.cfg), "state_dict": sd, **extra}, path)

    @classmethod
    def load(cls, path, stats: Stats, lat, map_location="cpu", corrector=None):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        cfg = dict(ck["cfg"])
        cfg["img_size"] = tuple(cfg["img_size"])
        cfg["embed_filter"] = tuple(cfg.get("embed_filter") or ())
        em = cls(ViTConfig(**cfg), stats, lat)
        em.load_state_dict(ck["state_dict"])
        if corrector is not None:
            em.corrector = corrector.to(next(em.parameters()).device)
        return em, ck
