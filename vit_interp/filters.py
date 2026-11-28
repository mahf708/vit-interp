"""A seam filter for free runs: damps the 4° patch-scale noise the network adds at every step.

Each step's increment (next state − state) is low-passed in longitude (an exp(-(k/kc)^8)
response above zonal wavenumber kc, kc shrinking with cos(latitude) as a GCM polar filter) and
smoothed by a 1-2-1 pass in latitude. The global mean of each field and every scale larger
than ~6° are kept. It is applied outside the network, after the correctors: hooks, circuits
and LAWFUL measurements never see it. Without it, patch-period power in a free run grows
~30× in a month and ~150× in three, until a hot spot (Tibet for v4, the poles for v2)
destabilizes the whole run; with it, v4 holds a realistic climate for 300+ days.
"""

from __future__ import annotations

import numpy as np
import torch


class SeamFilter(torch.nn.Module):
    def __init__(self, lat, n_lon: int, kc: float = 60.0, shapiro: int = 1, min_k: float = 6.0):
        super().__init__()
        k = torch.arange(n_lon // 2 + 1, dtype=torch.float64)
        kcl = torch.clamp(kc * torch.cos(torch.deg2rad(torch.as_tensor(np.asarray(lat), dtype=torch.float64))),
                          min=min_k)
        self.register_buffer("resp", torch.exp(-((k[None, :] / kcl[:, None]) ** 8)).float())  # (H, K)
        self.kc, self.shapiro, self.n_lon = kc, shapiro, n_lon

    def forward(self, inc: torch.Tensor) -> torch.Tensor:
        dt = inc.dtype
        x = inc.float()
        if self.kc:
            x = torch.fft.irfft(torch.fft.rfft(x, dim=-1) * self.resp, n=self.n_lon, dim=-1)
        for _ in range(self.shapiro):
            pad = torch.cat([x[..., :1, :], x, x[..., -1:, :]], -2)
            x = 0.25 * pad[..., :-2, :] + 0.5 * pad[..., 1:-1, :] + 0.25 * pad[..., 2:, :]
        return x.to(dt)

    def apply(self, state: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
        """The filtered next state: state + filter(nxt − state)."""
        return state + self(nxt - state)


class PatchJitter:
    """Per-step sub-patch shift of the patch grid (after STRATA's ``tile_jitter``, NVlabs/STRATA).

    Every step injects patch-period noise at the same phase, so it adds coherently and grows.
    Running each step on the input shifted by ``(oy, ox)`` pixels (``cycle16``: all 16 offsets of a
    4×4 patch in turn; ``cycle16d``: the same, not locked to the time of day) puts each step's seam pattern at a different phase, so it cancels instead.
    Longitude shifts are exact (periodic roll); latitude shifts replicate the polar rows. The
    network itself never changes: ``jitter(net_fn, x)`` shifts ``x``, calls ``net_fn``, and shifts
    the output back."""

    def __init__(self, patch: int = 4, mode: str = "cycle16", seed: int = 0):
        self.p, self.mode, self.t = patch, mode, 0
        self.rng = np.random.default_rng(seed)

    def next_shift(self) -> tuple[int, int]:
        t, p = self.t, self.p
        self.t += 1
        if self.mode == "cycle16":
            return (t // p) % p, t % p
        if self.mode == "cycle16d":
            # cycle16 with the axes swapped: with 6-h steps (4 a day) cycle16's longitude offset t % 4 is
            # locked to the UTC hour, so the diurnal cycle always meets the same seam phase. Here the
            # longitude offset advances once a day and the latitude offset follows the hour.
            return t % p, (t // p) % p
        if self.mode == "lon":
            return 0, t % p
        return int(self.rng.integers(0, p)), int(self.rng.integers(0, p))

    @staticmethod
    def _shift(x, oy: int, ox: int):
        if ox:
            x = torch.roll(x, -ox, dims=-1)
        if oy:
            x = torch.cat([x[..., oy:, :], x[..., -1:, :].expand(*x.shape[:-2], oy, x.shape[-1])], -2)
        return x

    @staticmethod
    def _unshift(y, oy: int, ox: int):
        if oy:
            y = torch.cat([y[..., :1, :].expand(*y.shape[:-2], oy, y.shape[-1]), y[..., :-oy, :]], -2)
        if ox:
            y = torch.roll(y, ox, dims=-1)
        return y

    def __call__(self, net_fn, x):
        oy, ox = self.next_shift()
        return self._unshift(net_fn(self._shift(x, oy, ox)), oy, ox)
