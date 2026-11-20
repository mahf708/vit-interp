"""Ensemble scores and ACE's ensemble loss (CRPS + energy score), in torch, standalone.

Mirrors ACE (fme/core/ensemble.py: get_crps, get_energy_score; fme/core/loss.py: CRPSLoss,
EnergyScoreLoss, EnsembleLoss):

- ``crps``: the fair CRPS estimator, E|X - y| - (1 - eps)/2 E|X - X'| over unique member
  pairs, eps = (1 - alpha)/2 ("almost fair" CRPS for alpha < 1; alpha = 1 is fair).
- ``energy_score``: the same for complex spherical-harmonic coefficients (beta = 1, modulus).
- ``EnsembleLoss``: crps_weight * mean_grid(CRPS) + energy_weight * ES over the SHT modes,
  with ACE's mode weights (2 for m > 0, 1 for m = 0) and scaling 2 sqrt(L M), averaged over
  batch and channels. As in ACE, the CRPS is a plain mean over grid points (not area
  weighted); ``area_weights`` makes it area weighted instead.

``RealSHT`` is a forward real spherical-harmonic transform on a Gaussian (Legendre-Gauss)
grid with orthonormal harmonics (torch-harmonics' "ortho" convention, which ACE uses through
its gridded operations): an rFFT in longitude, then Gauss-Legendre quadrature against
orthonormal associated Legendre functions. lmax = n_lat, mmax = n_lon // 2 + 1.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def crps(gen: torch.Tensor, target: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """gen (B, n_ens, ...), target (B, 1, ...) -> (B, ...)."""
    n = gen.shape[1]
    eps = (1.0 - alpha) / 2.0
    target_term = (gen - target).abs().mean(1)
    if n == 1:
        return target_term
    i, j = torch.triu_indices(n, n, offset=1, device=gen.device)
    internal = -0.5 * (gen[:, i] - gen[:, j]).abs().mean(1)
    return target_term + (1.0 - eps) * internal


def energy_score(gen: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Complex gen (B, n_ens, ...), target (B, 1, ...) -> (B, ...): E|X-y| - 1/2 E|X-X'|."""
    n = gen.shape[1]
    target_term = (gen - target).abs().mean(1)
    if n == 1:
        return target_term
    i, j = torch.triu_indices(n, n, offset=1, device=gen.device)
    return target_term - 0.5 * (gen[:, i] - gen[:, j]).abs().mean(1)


def _legendre(x: np.ndarray, lmax: int, mmax: int) -> np.ndarray:
    """(mmax, lmax, len(x)) orthonormal P_lm(x): 2 pi int P_lm^2 dx = 1 (zero where l < m)."""
    s = np.sqrt(np.clip(1 - x * x, 0, None))
    P = np.zeros((mmax, lmax, len(x)))
    pmm = np.full(len(x), math.sqrt(1 / (4 * math.pi)))
    for m in range(min(mmax, lmax)):
        if m > 0:
            pmm = pmm * math.sqrt((2 * m + 1) / (2 * m)) * s
        P[m, m] = pmm
        if m + 1 < lmax:
            P[m, m + 1] = x * math.sqrt(2 * m + 3) * pmm
        for l in range(m + 2, lmax):
            a = math.sqrt((4 * l * l - 1) / (l * l - m * m))
            b = math.sqrt(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1))
            P[m, l] = a * (x * P[m, l - 1] - b * P[m, l - 2])
    return P


class RealSHT(nn.Module):
    """Forward real SHT on a Gaussian grid: (..., n_lat, n_lon) -> complex (..., lmax, mmax).

    ``lat`` is the grid's latitudes in degrees (any order); quadrature weights are the
    Gauss-Legendre weights of the nearest nodes (checked to match to 1e-3 degrees).
    """

    def __init__(self, lat, n_lon: int):
        super().__init__()
        lat = np.asarray(lat, dtype=np.float64)
        n_lat = len(lat)
        nodes, weights = np.polynomial.legendre.leggauss(n_lat)  # x ascending
        order = np.argsort(np.sin(np.deg2rad(lat)))
        x = np.sin(np.deg2rad(lat))
        if np.abs(np.rad2deg(np.arcsin(nodes)) - lat[order]).max() > 1e-3:
            raise ValueError("RealSHT needs a Gaussian (Legendre-Gauss) latitude grid")
        w = np.empty(n_lat)
        w[order] = weights
        self.lmax, self.mmax, self.n_lon = n_lat, n_lon // 2 + 1, n_lon
        P = _legendre(x, self.lmax, self.mmax) * w  # quadrature folded in
        self.register_buffer("pw", torch.tensor(P, dtype=torch.float32), persistent=False)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        F_m = torch.fft.rfft(f.float(), dim=-1) * (2 * math.pi / self.n_lon)  # (..., lat, m)
        pw = self.pw.to(F_m.real.dtype)
        re = torch.einsum("...tm,mlt->...lm", F_m.real, pw)
        im = torch.einsum("...tm,mlt->...lm", F_m.imag, pw)
        return torch.complex(re, im)


class EnsembleLoss(nn.Module):
    """ACE's EnsembleLoss: crps_weight * CRPS + energy_weight * energy score (spectral).

    forward(gen (B, n_ens, C, H, W), target (B, 1, C, H, W)) -> scalar, and the two parts.
    """

    def __init__(self, lat, n_lon: int, crps_weight: float = 0.9, energy_weight: float = 0.1,
                 alpha: float = 1.0, area_weights: torch.Tensor | None = None):
        super().__init__()
        if crps_weight < 0 or energy_weight < 0 or crps_weight + energy_weight == 0:
            raise ValueError("crps_weight and energy_weight must be >= 0 and not both 0")
        self.crps_weight, self.energy_weight, self.alpha = crps_weight, energy_weight, alpha
        self.sht = RealSHT(lat, n_lon) if energy_weight > 0 else None
        if area_weights is not None:
            self.register_buffer("area_w", area_weights / area_weights.mean(), persistent=False)
        else:
            self.area_w = None
        if self.sht is not None:
            L, M = self.sht.lmax, self.sht.mmax
            mw = torch.full((L, M), 2.0)
            mw[:, 0] = 1.0
            self.register_buffer("mode_w", mw, persistent=False)
            self.es_factor = (L * M) / (2 * math.sqrt(L * M))  # mean over modes * n / scaling

    def forward(self, gen: torch.Tensor, target: torch.Tensor):
        parts = {}
        total = gen.new_zeros(())
        if self.crps_weight > 0:
            c = crps(gen, target, self.alpha)  # (B, C, H, W)
            c = (c * self.area_w).mean((-2, -1)) if self.area_w is not None else c.mean((-2, -1))
            parts["crps"] = c.mean()
            total = total + self.crps_weight * parts["crps"]
        if self.energy_weight > 0:
            es = energy_score(self.sht(gen), self.sht(target)) * self.mode_w  # (B, C, L, M)
            parts["energy"] = (es.mean((-2, -1)) * self.es_factor).mean()
            total = total + self.energy_weight * parts["energy"]
        return total, parts
