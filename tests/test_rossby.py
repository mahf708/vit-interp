import dataclasses

import numpy as np
import torch

from vit_interp import rossby as R
from vit_interp.data import ALL_NAMES

LAT = np.linspace(-89.5, 89.5, 180)
LON = np.arange(0.5, 360, 1.0)


def test_plant_amplitude_and_structure():
    w = R.Wave(s=6, phi0=45, v_amp=2.0)
    d = R.plant(LAT, LON, w)
    v = d[ALL_NAMES.index("V_3")]
    assert abs(np.abs(v).max() - 2.0) < 0.1
    assert np.abs(d[ALL_NAMES.index("V_0")]).max() == 0  # untouched level
    ps = d[ALL_NAMES.index("PS")]
    # geostrophic: |p'| = rho f |v| / k at the center latitude
    k = w.s / (R.A_EARTH * np.cos(np.deg2rad(45)))
    expect = R.RHO_S * 2 * R.OMEGA * np.sin(np.deg2rad(45)) * 2.0 / k
    assert abs(np.abs(ps).max() / expect - 1) < 0.1


def test_bridge_recovers_a_known_translation():
    w = R.Wave(s=5, phi0=50)
    c_true, growth_true = -7.5, 0.3  # m/s, 1/day
    v0 = torch.tensor(R.plant(LAT, LON, w)[ALL_NAMES.index("V_3")])[None]
    shift = c_true * R.DT / (R.A_EARTH * np.cos(np.deg2rad(w.phi0)))  # radians of longitude
    w1 = dataclasses.replace(w, phase=-w.s * shift)
    v1 = torch.tensor(R.plant(LAT, LON, w1)[ALL_NAMES.index("V_3")])[None] * np.exp(growth_true * R.DT / 86400)
    c, g = R.phase_speed(R._projection(v0, LAT, LON, w), R._projection(v1, LAT, LON, w), w)
    assert abs(c - c_true) < 1e-6 and abs(g - growth_true) < 1e-6


def test_theory_signs():
    long, short = R.theory(10.0, R.Wave(s=2)), R.theory(10.0, R.Wave(s=12))
    assert long["intrinsic"] < short["intrinsic"] < 0  # long waves retrogress faster
    assert R.consistency([1, 2, 3], [1, 2, 3]) == 1.0
