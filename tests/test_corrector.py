"""ACE-style correctors against real E3SM states (the v2 smoke data)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vit_interp.corrector import (
    GRAVITY,
    LATENT_HEAT_OF_VAPORIZATION,
    TIMESTEP_SECONDS,
    Corrector,
    CorrectorConfig,
)
from vit_interp.data import ALL_NAMES, DIAG_NAMES, N_ALL, N_PROG, PROG_NAMES

SMOKE = Path("/lcrc/globalscratch/ac.ngmahfouz/vit-interp/smoke-v2")
pytestmark = pytest.mark.skipif(not (SMOKE / "1990.npy").exists(), reason="no v2 smoke data")


@pytest.fixture(scope="module")
def data():
    meta = json.loads((SMOKE / "meta.json").read_text())
    a = np.load(SMOKE / "1990.npy", mmap_mode="r")
    t = [10, 40]  # two samples; not the first step of the month
    x0 = torch.tensor(np.array(a[t]), dtype=torch.float64)
    x1 = torch.tensor(np.array(a[[i + 1 for i in t]]), dtype=torch.float64)
    return meta, x0, x1


def _make(meta, **kw):
    return Corrector(CorrectorConfig(**kw), meta["lat"], meta["ak"], meta["bk"], DIAG_NAMES).double()


def _split(x0, x1):
    return x0[:, :N_ALL], x1[:, :N_PROG], x1[:, N_ALL:], x1[:, N_PROG:N_ALL]


def test_flags_off_is_identity(data):
    meta, x0, x1 = data
    c = _make(meta, dry_air=False, positive=False)
    x_in, prog, diag, _ = _split(x0, x1)
    p, d, ref = c(x_in, prog, diag)
    assert torch.equal(p, prog) and torch.equal(d, diag) and ref is None


def test_dry_air_is_pinned_to_the_reference(data):
    meta, x0, x1 = data
    c = _make(meta, dry_air=True, positive=True)
    x_in, prog, diag, _ = _split(x0, x1)
    noisy = prog.clone()
    noisy[:, PROG_NAMES.index("PS")] += 300.0  # a global mass error
    p, _, ref = c(x_in, noisy, diag)
    torch.testing.assert_close(c.gmean(c.dry_air(p)), ref, rtol=0, atol=1e-6)
    # the truth itself needs only a tiny change
    p_true, _, _ = c(x_in, prog, diag)
    dps = (p_true - prog)[:, PROG_NAMES.index("PS")].abs().max()
    print("max |PS change| on truth (Pa):", float(dps))
    assert dps < 5.0


def test_moisture_budget_closes_in_every_column(data):
    meta, x0, x1 = data
    c = _make(meta, dry_air=True, moisture=True)
    x_in, prog, diag, _ = _split(x0, x1)
    p, d, _ = c(x_in, prog, diag)
    twp_tend = (c.total_water_path(p) - c.total_water_path(x_in[:, :N_PROG])) / TIMESTEP_SECONDS
    evap = d[:, DIAG_NAMES.index("LHFLX")] / LATENT_HEAT_OF_VAPORIZATION
    precip = d[:, DIAG_NAMES.index("surface_precipitation_rate")]
    adv = d[:, DIAG_NAMES.index("tendency_of_total_water_path_due_to_advection")]
    torch.testing.assert_close(twp_tend, evap - precip + adv, rtol=0, atol=1e-12)
    assert c.gmean(adv).abs().max() < 1e-12
    p0 = diag[:, DIAG_NAMES.index("surface_precipitation_rate")]
    rel = float(c.gmean((precip - p0).abs()).max() / c.gmean(p0).max())
    print("relative precipitation change on truth:", rel)
    assert rel < 0.2


def test_energy_budget_closes_globally(data):
    meta, x0, x1 = data
    c = _make(meta, dry_air=True, moisture=True, energy=True, unaccounted_heating=3.0)
    x_in, prog, diag, f_next = _split(x0, x1)
    p, d, _ = c(x_in, prog, diag, forcing_next=f_next)
    phis = x_in[:, ALL_NAMES.index("PHIS")]
    e1 = c.gmean(c.total_energy_path(p, phis))
    e0 = c.gmean(c.total_energy_path(x_in[:, :N_PROG], phis))
    flux = c.gmean(c.net_flux_into_atmosphere(d, f_next[:, 0]))
    torch.testing.assert_close((e1 - e0) / TIMESTEP_SECONDS, flux + 3.0, rtol=0, atol=1e-3)
    dT = (p - prog)[:, PROG_NAMES.index("T_7")]
    print("uniform dT on truth (K):", dT.mean((1, 2)).tolist())
    assert dT.abs().max() < 0.5


def test_gradients_flow_through_all_four(data):
    meta, x0, x1 = data
    c = _make(meta, dry_air=True, positive=True, moisture=True, energy=True)
    x_in, prog, diag, f_next = _split(x0, x1)
    prog = (prog + 0.01 * torch.randn_like(prog)).requires_grad_(True)
    diag = diag.clone().requires_grad_(True)
    p, d, _ = c(x_in, prog, diag, forcing_next=f_next)
    (p.sum() + d.sum()).backward()
    assert torch.isfinite(prog.grad).all() and prog.grad.abs().sum() > 0
    assert torch.isfinite(diag.grad).all() and diag.grad.abs().sum() > 0


def test_inputs_are_not_modified(data):
    meta, x0, x1 = data
    c = _make(meta, dry_air=True, positive=True, moisture=True, energy=True)
    x_in, prog, diag, f_next = _split(x0, x1)
    before = (x_in.clone(), prog.clone(), diag.clone())
    c(x_in, prog, diag, forcing_next=f_next)
    for a, b in zip(before, (x_in, prog, diag)):
        assert torch.equal(a, b)


def test_positivity_clamps_water(data):
    meta, x0, x1 = data
    c = _make(meta, dry_air=False, positive=True)
    x_in, prog, diag, _ = _split(x0, x1)
    neg = prog.clone()
    neg[:, PROG_NAMES.index("specific_total_water_7")] -= 1.0
    p, _, _ = c(x_in, neg, None)
    assert (p[:, PROG_NAMES.index("specific_total_water_7")] >= 0).all()


def test_missing_diagnostics_raise(data):
    meta, x0, x1 = data
    x_in, prog, _, _ = _split(x0, x1)
    c = _make(meta, moisture=True)
    with pytest.raises(ValueError, match="diag"):
        c(x_in, prog, None)
    with pytest.raises(ValueError, match="needs diagnostics"):
        Corrector(CorrectorConfig(energy=True), meta["lat"], meta["ak"], meta["bk"], DIAG_NAMES[:2])


def test_ref_is_carried_across_steps(data):
    meta, x0, x1 = data
    c = _make(meta, dry_air=True)
    x_in, prog, diag, _ = _split(x0, x1)
    _, _, ref = c(x_in, prog, diag)
    assert ref.dtype == torch.float64 and ref.shape == (2,)
    # a later step pinned to the first step's reference
    p2, _, ref2 = c(x_in, prog + 0.0, diag, ref_dry_mass=ref)
    assert torch.equal(ref2, ref)
    assert abs(float(GRAVITY)) > 0  # constants importable


def test_float32_inputs_close_the_energy_budget(data):
    meta, x0, x1 = data
    c = Corrector(CorrectorConfig(dry_air=True, positive=True, moisture=True, energy=True),
                  meta["lat"], meta["ak"], meta["bk"], DIAG_NAMES)
    x_in, prog, diag, f_next = (t.float() for t in _split(x0, x1))
    p, d, ref = c(x_in, prog, diag, forcing_next=f_next)
    assert p.dtype == torch.float32 and d.dtype == torch.float32 and ref.dtype == torch.float64
    phis = x_in[:, ALL_NAMES.index("PHIS")].double()
    e1 = c.gmean(c.total_energy_path(p.double(), phis))
    e0 = c.gmean(c.total_energy_path(x_in[:, :N_PROG].double(), phis))
    flux = c.gmean(c.net_flux_into_atmosphere(d.double(), f_next[:, 0].double()))
    # float32 rounding of T after the fix: within ~0.5 W/m^2 of the budget
    assert ((e1 - e0) / TIMESTEP_SECONDS - flux).abs().max() < 0.5
