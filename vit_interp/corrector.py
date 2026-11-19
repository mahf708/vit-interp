"""ACE-style conservation correctors, standalone and differentiable (torch).

Mirrors ``fme.core.corrector.atmosphere`` (ACE, ai2cm/ace) without importing it. Applied
after the network, to the predicted physical state, in ACE's order:

1. ``positive``   ForcePositive: total water at every level (and the non-negative
                  diagnostics, when present) clamped at 0.
2. ``dry_air``    ConserveDryAir: a globally-constant offset to each column's dry-air
                  pressure, ``ps - g * TWP``, pins its global mean to the run's initial
                  value; PS is then solved from it given the predicted water.
3. ``moisture``   ZeroGlobalMeanMoistureAdvection, then MoistureBudgetCorrection
                  "advection_and_precipitation": precipitation is rescaled so the global
                  budget dTWP/dt = E - P holds, and advection is recomputed as the column
                  residual, so every column's budget dTWP/dt = E - P + A closes exactly.
4. ``energy``     TotalEnergyBudgetCorrection "constant_temperature": one temperature
                  offset for every level and column makes the global-mean ACE2 total
                  energy path change by (net flux into the atmosphere + unaccounted
                  heating) * dt.

Global means are cos-latitude area weighted (as ACE's lat-lon operations). Dry air is
computed in float64, as ACE does off MPS. The energy budget uses the forcing at the
*output* time (ACE passes ``next_step_input_data``): pass it as ``forcing_next``; without it
the input's forcing is used.

Conventions (as ACE): q is specific *total* water standing in for specific humidity;
E = LHFLX / L_v; total energy is c_v T + L_v q + g z (no kinetic energy); z is hydrostatic
from virtual temperature and the surface height PHIS / g (clipped at 0).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import torch

from .data import ALL_NAMES, FORCING_NAMES, N_PROG, PROG_NAMES

# fme/core/constants.py
LATENT_HEAT_OF_VAPORIZATION = 2.5e6  # J/kg
GRAVITY = 9.80665  # m/s^2
CP_DRY = 1004.6  # J/kg/K
RVGAS = 461.5  # J/kg/K
RDGAS = 287.05  # J/kg/K
CV_DRY = CP_DRY - RDGAS
TIMESTEP_SECONDS = 21600.0

LEVELS = 8
PS = PROG_NAMES.index("PS")
T_IDX = [PROG_NAMES.index(f"T_{k}") for k in range(LEVELS)]
Q_IDX = [PROG_NAMES.index(f"specific_total_water_{k}") for k in range(LEVELS)]
SOLIN = ALL_NAMES.index("SOLIN")
PHIS = ALL_NAMES.index("PHIS")
_F_SOLIN, _F_PHIS = FORCING_NAMES.index("SOLIN"), FORCING_NAMES.index("PHIS")

# the diagnostics ACE's E3SM configs force non-negative (fluxes that are magnitudes)
POSITIVE_DIAG = (
    "surface_precipitation_rate",
    "surface_upward_longwave_flux",
    "surface_upward_shortwave_flux",
    "FLDS",
    "FSDS",
    "FLUT",
    "top_of_atmos_upward_shortwave_flux",
)
MOISTURE_NEEDS = ("surface_precipitation_rate", "LHFLX", "tendency_of_total_water_path_due_to_advection")
ENERGY_NEEDS = (
    "LHFLX",
    "SHFLX",
    "surface_upward_longwave_flux",
    "surface_upward_shortwave_flux",
    "FLDS",
    "FSDS",
    "FLUT",
    "top_of_atmos_upward_shortwave_flux",
)


@dataclasses.dataclass
class CorrectorConfig:
    dry_air: bool = True
    positive: bool = True
    moisture: bool = False
    energy: bool = False
    unaccounted_heating: float = 0.0  # W/m^2, ACE constant_unaccounted_heating


class Corrector(torch.nn.Module):
    def __init__(self, cfg: CorrectorConfig, lat, ak, bk, diag_names: Sequence[str] = ()):
        super().__init__()
        self.cfg = cfg
        self.diag_names = list(diag_names)
        w = torch.cos(torch.deg2rad(torch.as_tensor(np.asarray(lat), dtype=torch.float64)))
        self.register_buffer("w", w / w.sum())  # per latitude row, sums to 1
        self.register_buffer("ak", torch.as_tensor(np.asarray(ak), dtype=torch.float64))
        self.register_buffer("bk", torch.as_tensor(np.asarray(bk), dtype=torch.float64))
        if self.ak.numel() != LEVELS + 1 or self.bk.numel() != LEVELS + 1:
            raise ValueError(f"ak and bk need {LEVELS + 1} interfaces")
        for flag, needs in (("moisture", MOISTURE_NEEDS), ("energy", ENERGY_NEEDS)):
            if getattr(cfg, flag):
                missing = [n for n in needs if n not in self.diag_names]
                if missing:
                    raise ValueError(f"the {flag} correction needs diagnostics {missing}")

    # -- operators (ACE's vertical coordinate and gridded operations) --

    def gmean(self, x: torch.Tensor) -> torch.Tensor:
        """Area-weighted global mean over the last two dims -> (B,)."""
        return (x.mean(-1) * self.w.to(x.dtype)).sum(-1)

    def interface_pressure(self, ps: torch.Tensor) -> torch.Tensor:
        """(B, H, W) -> (B, H, W, 9), top to surface."""
        return self.ak.to(ps.dtype) + self.bk.to(ps.dtype) * ps[..., None]

    def vertical_integral(self, integrand: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
        """integrand (B, H, W, 8) -> (1/g) sum x dp."""
        dp = self.interface_pressure(ps).diff(dim=-1)
        return (integrand * dp).sum(-1) / GRAVITY

    def total_water_path(self, prog: torch.Tensor) -> torch.Tensor:
        q = torch.stack([prog[:, i] for i in Q_IDX], -1)
        return self.vertical_integral(q, prog[:, PS])

    def dry_air(self, prog: torch.Tensor) -> torch.Tensor:
        """metrics.surface_pressure_due_to_dry_air: ps - g * TWP."""
        return prog[:, PS] - GRAVITY * self.total_water_path(prog)

    def _levels(self, prog):
        T = torch.stack([prog[:, i] for i in T_IDX], -1)
        q = torch.stack([prog[:, i] for i in Q_IDX], -1)
        return T, q

    def _thickness(self, pi, T, q):
        """atmosphere_data.compute_layer_thickness."""
        tv = T * (1 + (RVGAS / RDGAS - 1.0) * q)
        dlogp = torch.log(torch.clamp(pi, min=1.0)).diff(dim=-1)
        return dlogp * RDGAS * tv / GRAVITY

    def total_energy_path(self, prog: torch.Tensor, phis: torch.Tensor) -> torch.Tensor:
        """atmosphere_data.total_energy_ace2_path: (1/g) int (c_v T + L_v q + g z_mid) dp."""
        T, q = self._levels(prog)
        ps = prog[:, PS]
        dz = self._thickness(self.interface_pressure(ps), T, q)
        cum = torch.cumsum(dz.flip(dims=(-1,)), dim=-1).flip(dims=(-1,))
        hsfc = torch.clamp(phis / GRAVITY, min=0.0)[..., None]
        z_int = torch.cat([cum + hsfc, hsfc], dim=-1)
        z_mid = 0.5 * (z_int[..., :-1] + z_int[..., 1:])
        e = T * CV_DRY + q * LATENT_HEAT_OF_VAPORIZATION + z_mid * GRAVITY
        return self.vertical_integral(e, ps)

    def _energy_factor(self, prog: torch.Tensor) -> torch.Tensor:
        """atmosphere._energy_correction_factor: d(energy path) / d(uniform dT)."""
        T, q = self._levels(prog)
        ps = prog[:, PS]
        q_dlogp = self._thickness(self.interface_pressure(ps), T, q) * GRAVITY / T
        cum = torch.cumsum(q_dlogp.flip(dims=(-1,)), dim=-1).flip(dims=(-1,))
        return self.vertical_integral(CV_DRY - 0.5 * q_dlogp + cum, ps)

    def d(self, diag: torch.Tensor, name: str) -> torch.Tensor:
        return diag[:, self.diag_names.index(name)]

    def net_flux_into_atmosphere(self, diag: torch.Tensor, solin: torch.Tensor) -> torch.Tensor:
        """metrics.net_top_of_atmosphere_energy_flux - net_surface_energy_flux (no frozen)."""
        d = lambda n: self.d(diag, n)  # noqa: E731
        toa = solin - d("top_of_atmos_upward_shortwave_flux") - d("FLUT")
        sfc = (d("FSDS") - d("surface_upward_shortwave_flux") + d("FLDS")
               - d("surface_upward_longwave_flux") - d("LHFLX") - d("SHFLX"))
        return toa - sfc

    def seed_dry_mass(self, x_in: torch.Tensor) -> torch.Tensor:
        """_seed_global_dry_air_mass: (B,) float64 global-mean dry-air pressure of x_in."""
        return self.gmean(self.dry_air(x_in[:, :N_PROG].double()))

    # -- the corrections --

    def forward(self, x_in, prog_next, diag=None, ref_dry_mass=None, forcing_next=None):
        cfg = self.cfg
        if (cfg.moisture or cfg.energy) and diag is None:
            raise ValueError("the moisture and energy corrections need the diagnostics (diag)")
        if diag is not None and diag.shape[1] != len(self.diag_names):
            raise ValueError(f"diag has {diag.shape[1]} channels, expected {len(self.diag_names)}")
        prog = prog_next
        if cfg.positive:
            idx = torch.zeros(N_PROG, dtype=torch.bool, device=prog.device)
            idx[Q_IDX] = True
            prog = torch.where(idx[None, :, None, None], prog.clamp(min=0.0), prog)
            if diag is not None:
                pos = [n in POSITIVE_DIAG for n in self.diag_names]
                if any(pos):
                    m = torch.tensor(pos, device=diag.device)[None, :, None, None]
                    diag = torch.where(m, diag.clamp(min=0.0), diag)
        if cfg.dry_air or ref_dry_mass is not None:
            if ref_dry_mass is None:
                ref_dry_mass = self.seed_dry_mass(x_in)
        if cfg.dry_air:
            p64 = prog.double()
            dry = self.dry_air(p64)
            new_dry = dry - (self.gmean(dry) - ref_dry_mass.double())[:, None, None]
            q = torch.stack([p64[:, i] for i in Q_IDX], -1)
            ak_d, bk_d = self.ak.diff(), self.bk.diff()
            # dry = ps - sum((ak_d + bk_d ps) q)  =>  ps = (dry + sum(ak_d q)) / (1 - sum(bk_d q))
            ps = (new_dry + (ak_d * q).sum(-1)) / (1 - (bk_d * q).sum(-1))
            prog = torch.cat([prog[:, :PS], ps.to(prog.dtype)[:, None], prog[:, PS + 1:]], 1)
        if cfg.moisture:
            i_p = self.diag_names.index("surface_precipitation_rate")
            i_a = self.diag_names.index("tendency_of_total_water_path_due_to_advection")
            adv = diag[:, i_a]
            adv = adv - self.gmean(adv)[:, None, None]  # zero global-mean advection
            twp_tend = (self.total_water_path(prog) - self.total_water_path(x_in[:, :N_PROG])) / TIMESTEP_SECONDS
            evap = self.d(diag, "LHFLX") / LATENT_HEAT_OF_VAPORIZATION
            precip = diag[:, i_p]
            new_p_gm = self.gmean(evap) - self.gmean(twp_tend)
            precip = precip * (new_p_gm / self.gmean(precip))[:, None, None]
            adv = twp_tend - (evap - precip)
            cols = list(diag.unbind(1))
            cols[i_p], cols[i_a] = precip, adv
            diag = torch.stack(cols, 1)
        if cfg.energy:
            # float64: the energy path is ~2.5e9 J/m^2 and the budget term ~1e6 per step
            f = (x_in[:, N_PROG:] if forcing_next is None else forcing_next).double()
            solin, phis = f[:, _F_SOLIN], f[:, _F_PHIS]
            p64, x64 = prog.double(), x_in.double()
            e_gen = self.gmean(self.total_energy_path(p64, phis))
            e_in = self.gmean(self.total_energy_path(x64[:, :N_PROG], x64[:, PHIS]))
            flux = self.gmean(self.net_flux_into_atmosphere(diag.double(), solin))
            desired = e_in + (flux + cfg.unaccounted_heating) * TIMESTEP_SECONDS
            dT = ((desired - e_gen) / self.gmean(self._energy_factor(p64))).to(prog.dtype)
            t_mask = torch.zeros(N_PROG, dtype=prog.dtype, device=prog.device)
            t_mask[T_IDX] = 1.0
            prog = prog + t_mask[None, :, None, None] * dT[:, None, None, None]
        return prog, diag, ref_dry_mass
