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
