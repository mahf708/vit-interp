"""Global-mean budget residuals of the TRUE data, by the correctors' own formulas.

    .venv/bin/python scripts/budget_residuals.py [DATA_ROOT] [--years 1990,1991] [--stride 4]

energy: mean over steps of d(global-mean ACE2 total energy path)/dt - global-mean net flux
        into the atmosphere (W/m^2): ACE's constant_unaccounted_heating is this number.
moisture: global-mean dTWP/dt - (E - P) and the global-mean advection (kg/m^2/s, and in
        mm/day), and the column RMS of dTWP/dt - (E - P + A).
dry air: drift of the global-mean dry-air pressure per step (Pa).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vit_interp.corrector import (
    LATENT_HEAT_OF_VAPORIZATION,
    TIMESTEP_SECONDS,
    Corrector,
    CorrectorConfig,
)
from vit_interp.data import ALL_NAMES, DIAG_NAMES, N_ALL, N_PROG

p = argparse.ArgumentParser()
p.add_argument("root", nargs="?", default="/lcrc/globalscratch/ac.ngmahfouz/vit-interp/data-v2")
p.add_argument("--years", default=None)
p.add_argument("--stride", type=int, default=4, help="every n-th step (default: daily)")
a = p.parse_args()
root = Path(a.root)
meta = json.loads((root / "meta.json").read_text())
years = [int(y) for y in a.years.split(",")] if a.years else sorted(
    int(f.stem) for f in root.glob("[0-9]*.npy") if not f.stem.endswith("partial"))
c = Corrector(CorrectorConfig(), meta["lat"], meta["ak"], meta["bk"], DIAG_NAMES)
d = lambda x, n: x[:, N_ALL + DIAG_NAMES.index(n)]  # noqa: E731
energy, moist, adv_gm, col_rms, dry = [], [], [], [], []
for y in years:
    arr = np.load(root / f"{y}.npy", mmap_mode="r")
    for t in range(0, arr.shape[0] - 2, a.stride):
        ia = N_ALL + DIAG_NAMES.index("tendency_of_total_water_path_due_to_advection")
        if np.array_equal(arr[t + 1, ia], arr[t + 2, ia]):
            continue  # a month's first step, whose advection residual was filled from the next
        x = torch.tensor(np.array(arr[t : t + 2]), dtype=torch.float64)
        x0, x1 = x[:1], x[1:]
        e1 = c.gmean(c.total_energy_path(x1[:, :N_PROG], x1[:, ALL_NAMES.index("PHIS")]))
        e0 = c.gmean(c.total_energy_path(x0[:, :N_PROG], x0[:, ALL_NAMES.index("PHIS")]))
        flux = c.gmean(c.net_flux_into_atmosphere(x1[:, N_ALL:], x1[:, ALL_NAMES.index("SOLIN")]))
        energy.append(float((e1 - e0) / TIMESTEP_SECONDS - flux))
        twp_t = (c.total_water_path(x1[:, :N_PROG]) - c.total_water_path(x0[:, :N_PROG])) / TIMESTEP_SECONDS
        E = d(x1, "LHFLX") / LATENT_HEAT_OF_VAPORIZATION
        P = d(x1, "surface_precipitation_rate")
        A = d(x1, "tendency_of_total_water_path_due_to_advection")
        moist.append(float(c.gmean(twp_t - (E - P))))
        adv_gm.append(float(c.gmean(A)))
        col_rms.append(float(c.gmean((twp_t - (E - P + A)) ** 2).sqrt()))
        dry.append(float(c.gmean(c.dry_air(x1[:, :N_PROG])) - c.gmean(c.dry_air(x0[:, :N_PROG]))))
    print(f"{y}: {len(energy)} steps so far", flush=True)
mmday = 86400.0  # kg/m^2/s -> mm/day
print(f"years {years}, {len(energy)} steps, every {a.stride}")
print(f"energy residual [W/m^2]: mean {np.mean(energy):+.3f}, std {np.std(energy):.3f}  "
      f"-> unaccounted_heating = {np.mean(energy):+.2f}")
print(f"moisture: global dTWP/dt - (E-P) mean {np.mean(moist) * mmday:+.4f} mm/day (std "
      f"{np.std(moist) * mmday:.4f}); global-mean advection {np.mean(adv_gm) * mmday:+.4f} mm/day; "
      f"column RMS of the full budget residual {np.mean(col_rms) * mmday:.4f} mm/day")
print(f"dry air: global-mean change per step {np.mean(dry):+.4f} Pa (std {np.std(dry):.4f})")
