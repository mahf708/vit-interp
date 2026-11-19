"""Does the ViT move Rossby waves the way the dispersion relation says? LAWFUL, instantiated.

LAWFUL (Chen, Parker & Arora 2026) pairs a learned model f with a physics model g, joined
by an input bridge B, an output bridge D, and a family P of perturbations p_alpha; it then
scores consistency M_P(f, g) and finds the components that carry it by sufficiency /
necessity activation patching. Here:

  g        barotropic Rossby waves on a zonal flow U(phi):  c - U = -beta / (k^2 + l^2),
           k = s / (a cos phi0),  beta = 2 Omega cos(phi0) / a  (optionally minus U_yy)
  B        plant a small wave psi = A G(phi) cos(s lambda) on real states: non-divergent
           u' = -(1/a) dpsi/dphi, v' = (1/(a cos phi)) dpsi/dlambda on the chosen levels,
           and (optionally) the geostrophically balancing PS' = rho_s f psi
  P        alpha = zonal wavenumber s (or latitude phi0, or amplitude for a linearity check)
  D        the phase speed of the planted wave in the model's response: project v' on
           G(phi) exp(i s lambda) before and after the step; the phase advance gives c
  M_P      1 - sum_alpha (c_f - c_g)^2 / sum_alpha (c_g - mean c_g)^2, on the intrinsic
           speed c - U (LAWFUL eq. 10, with phase speed as the slope)

Patching is at the component level LAWFUL uses (heads and MLPs): `component_hooks` swaps
chosen heads' / MLPs' outputs for those of another forward pass.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from .data import ALL_NAMES, PROG_NAMES

A_EARTH = 6.371e6
OMEGA = 7.292e-5
DT = 21600.0
RHO_S = 1.2


@dataclasses.dataclass(frozen=True)
class Wave:
    s: int = 6  # zonal wavenumber
    phi0: float = 45.0  # center latitude, degrees (negative for the south)
    sigma: float = 10.0  # e-folding half-width of the envelope, degrees
    v_amp: float = 1.0  # peak meridional wind of the planted wave, m/s
    levels: tuple[int, ...] = (2, 3, 4, 5, 6)
    balance: bool = True  # add PS' = rho_s f psi
    phase: float = 0.0


def _envelope(lat_deg, w: Wave):
    return np.exp(-(((lat_deg - w.phi0) / w.sigma) ** 2))


def plant(lat, lon, w: Wave, n_channels: int = len(ALL_NAMES)) -> np.ndarray:
    """delta (C, H, W) in physical units, on the full channel list (zero off U, V, PS)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    phi = np.deg2rad(lat)[:, None]
    lam = np.deg2rad(lon)[None, :]
    G = _envelope(lat, w)[:, None]
    dG_dphi = G * (-2 * (lat[:, None] - w.phi0) / w.sigma**2) * (180 / np.pi)
    A = w.v_amp * A_EARTH * np.cos(np.deg2rad(w.phi0)) / w.s  # so max |v'| ~ v_amp
    cosl = np.cos(w.s * lam + w.phase)
    sinl = np.sin(w.s * lam + w.phase)
    psi = A * G * cosl
    u = -(A / A_EARTH) * dG_dphi * cosl
    v = -(A * w.s / (A_EARTH * np.cos(phi))) * G * sinl
    d = np.zeros((n_channels, len(lat), len(lon)), dtype=np.float32)
    for k in w.levels:
        d[ALL_NAMES.index(f"U_{k}")] = u
        d[ALL_NAMES.index(f"V_{k}")] = v
    if w.balance:
        d[ALL_NAMES.index("PS")] = RHO_S * 2 * OMEGA * np.sin(phi) * psi
    return d


def _projection(v, lat, lon, w: Wave) -> torch.Tensor:
    """Complex amplitude of v (B, H, W) on G(phi) exp(i s lambda), area weighted -> (B,)."""
    lat_t = torch.as_tensor(lat, dtype=torch.float64, device=v.device)
    lon_t = torch.as_tensor(lon, dtype=torch.float64, device=v.device)
    G = torch.exp(-(((lat_t - w.phi0) / w.sigma) ** 2)) * torch.cos(torch.deg2rad(lat_t))
    e = torch.exp(-1j * (w.s * torch.deg2rad(lon_t) + w.phase))
    return torch.einsum("bhw,h,w->b", v.double().to(torch.complex128), G.to(torch.complex128), e)


def phase_speed(a0: torch.Tensor, a1: torch.Tensor, w: Wave, dt: float = DT) -> tuple[float, float]:
    """Phase speed (m/s, eastward positive) and growth rate (1/day) from complex amplitudes
    before/after (summed over the batch: the response is linear in a small wave)."""
    r = a1.sum() / a0.sum()
    dlam = -float(torch.angle(r)) / w.s  # v ~ sin(s(lam - c t / (a cos phi)))
    c = dlam * A_EARTH * np.cos(np.deg2rad(w.phi0)) / dt
    growth = float(torch.log(torch.abs(r))) / (dt / 86400)
    return c, growth


def theory(U_bar: float, w: Wave, l2: float | None = None, U_yy: float = 0.0) -> dict:
    """Barotropic Rossby phase speed on a uniform U_bar. l2 defaults to the mean squared
    meridional wavenumber of the Gaussian envelope, 2 / (sigma a)^2."""
    phi0 = np.deg2rad(w.phi0)
    k = w.s / (A_EARTH * np.cos(phi0))
    if l2 is None:
        l2 = 2.0 / (np.deg2rad(w.sigma) * A_EARTH) ** 2
    beta = 2 * OMEGA * np.cos(phi0) / A_EARTH - U_yy
    intrinsic = -beta / (k**2 + l2)
    return {"c": U_bar + intrinsic, "intrinsic": intrinsic, "U_bar": U_bar, "k": k, "l2": l2, "beta": beta}


def band_mean_U(X, lat, w: Wave, level: int) -> float:
    """Zonal-mean U at `level`, weighted by the envelope squared, averaged over the batch."""
    lat_t = torch.as_tensor(lat, dtype=torch.float64, device=X.device)
    G2 = torch.exp(-2 * (((lat_t - w.phi0) / w.sigma) ** 2)) * torch.cos(torch.deg2rad(lat_t))
    U = X[:, ALL_NAMES.index(f"U_{level}")].double().mean((0, 2))
    return float((U * G2).sum() / G2.sum())


@torch.no_grad()
def respond(em, X, lat, lon, w: Wave, level: int = 3, hooks_fn=None, symmetric: bool = False):
    """Phase speed and growth of the planted wave at `level` after one model step.

    X: (B, C_all, H, W) physical base states. hooks_fn(x_base, x_pert) -> hooks for the
    perturbed pass (for patching); None = plain model. With symmetric=True the response is
    (f(X+d) - f(X-d)) / 2, which cancels the even part; LAWFUL's one-sided base/source
    convention is the default.
    """
    d = torch.as_tensor(plant(lat, lon, w), device=X.device)[None]
    vi = PROG_NAMES.index(f"V_{level}")
    base, _ = em.step(X)
    hooks = hooks_fn(X, X + d) if hooks_fn else None
    pert, _ = em.step(X + d, hooks=hooks) if hooks_fn is None else em.step(X, hooks=hooks)
    if symmetric and hooks_fn is None:
        minus, _ = em.step(X - d)
        resp = (pert - minus)[:, vi] / 2
    else:
        resp = (pert - base)[:, vi]
    a0 = _projection(d[:, ALL_NAMES.index(f"V_{level}")].expand(X.shape[0], -1, -1), lat, lon, w)
    a1 = _projection(resp, lat, lon, w)
    return phase_speed(a0, a1, w)


def consistency(c_f, c_g) -> float:
    """LAWFUL eq. 10 on phase speeds across the family."""
    c_f, c_g = np.asarray(c_f, float), np.asarray(c_g, float)
    return float(1 - ((c_f - c_g) ** 2).sum() / ((c_g - c_g.mean()) ** 2).sum())


def sweep(em, X, lat, lon, base: Wave = Wave(), ss=range(2, 15), level: int = 3, hooks_fn=None):
    """The family alpha = s. Returns rows and M on the intrinsic speed c - U."""
    U = band_mean_U(X, lat, base, level)
    rows = []
    for s in ss:
        w = dataclasses.replace(base, s=s)
        c, g = respond(em, X, lat, lon, w, level, hooks_fn)
        th = theory(U, w)
        rows.append({"s": s, "c_model": c, "growth_per_day": g, "c_theory": th["c"],
                     "intrinsic_model": c - U, "intrinsic_theory": th["intrinsic"], "U_bar": U})
    M = consistency([r["intrinsic_model"] for r in rows], [r["intrinsic_theory"] for r in rows])
    return rows, M


# ---- component patching (LAWFUL sufficiency / necessity) -----------------------------

def components(depth: int, heads: int) -> list[tuple]:
    out = []
    for i in range(depth):
        out += [("a", i, h) for h in range(heads)] + [("m", i)]
    return out


def _names(comps):
    n = set()
    for c in comps:
        n.add(f"blocks.{c[1]}.attn.hook_result" if c[0] == "a" else f"blocks.{c[1]}.hook_mlp_out")
    return n


def component_hooks(em, comps, source_X):
    """hooks_fn for `respond`: run the base X, with the outputs of `comps` taken from a
    forward pass on source_X (computed once per call; source_X None = use x_pert)."""

    def hooks_fn(x_base, x_pert):
        src = x_pert if source_X is None else source_X
        cache: dict = {}
        em.step(src, cache=cache, names=_names(comps))
        hooks = {}
        by_layer: dict[int, list[int]] = {}
        for c in comps:
            if c[0] == "a":
                by_layer.setdefault(c[1], []).append(c[2])
            else:
                hooks[f"blocks.{c[1]}.hook_mlp_out"] = (lambda t, k=f"blocks.{c[1]}.hook_mlp_out": cache[k])
        for i, hs in by_layer.items():
            k = f"blocks.{i}.attn.hook_result"

            def f(t, k=k, hs=hs):
                t = t.clone()
                t[:, :, hs] = cache[k][:, :, hs]
                return t

            hooks[k] = f
        return hooks

    return hooks_fn


def sufficiency_necessity(em, X, lat, lon, comps, all_comps, base: Wave = Wave(), ss=range(2, 15),
                          level: int = 3):
    """M_suf: perturbation routed through comps only; M_nec: through everything else."""
    _, M0 = sweep(em, X, lat, lon, base, ss, level)
    rest = [c for c in all_comps if c not in comps]
    _, M_suf = sweep(em, X, lat, lon, base, ss, level, hooks_fn=component_hooks(em, comps, None))
    _, M_nec = sweep(em, X, lat, lon, base, ss, level, hooks_fn=component_hooks(em, rest, None))
    return {"M0": M0, "M_suf": M_suf, "M_nec": M_nec, "d_suf": M0 - M_suf, "d_nec": M0 - M_nec}
