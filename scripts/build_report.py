"""Build the public results page: one page per model version, and a front page comparing them.

    .venv/bin/python scripts/build_report.py [--out DIR]

Versions are registered in VERSIONS below: what each changed, where its runs live, which
evaluation is its headline. Everything else is read from the run directories, so rerun the
script whenever a run or an evaluation lands; a version still training shows its latest
validation numbers, and one only planned shows its recipe.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({  # one look for every figure on the site, legible at ~1000 px wide
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 10, "figure.dpi": 110, "savefig.dpi": 130,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.15, "axes.titlepad": 8,
})
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("/lcrc/globalscratch/ac.ngmahfouz/vit-interp")
RUNS = ROOT / "runs"
NB_EXEC = ROOT / "notebooks-executed"
OUT = Path("/lcrc/group/e3sm/public_html/diagnostic_output/ac.ngmahfouz/vit-interp")
DEFAULT_MODEL = "v2"  # what vit_interp.playground.load() returns, and what the notebooks were executed on
URL = "https://web.lcrc.anl.gov/public/e3sm/diagnostic_output/ac.ngmahfouz/vit-interp/"

# reference categorical palette, fixed order; one hue per version, never reused
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5af5", "#0fa3b1", "#8c564b", "#5a5a5a"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LEADS = ["1d", "3d", "5d", "10d"]
ACC_VARS = ["PS", "T_7", "U_3", "V_3"]

# -- the registry: add a version here --------------------------------------------------------
VERSIONS = [
    {
        "name": "v1",
        "title": "vanilla ViT, MSE, rollout-tuned",
        "run": "vit-p4-d384-l8",
        "checkpoint": "vit-p4-d384-l8/release/v1.pt",
        "headline_eval": "eval-ft8-corrected",
        "stages": [  # (label, run subdir with metrics.jsonl)
            ("one-step, 40 epochs", "stage1"),
            ("4-step rollouts (2 epochs)", "stage2"),
            ("4-step rollouts (6 more)", "ft2/ft4"),
            ("8-step rollouts", "ft2/ft8"),
        ],
        "evals": [  # (label, eval dir): the history of how v1 got where it is
            ("one-step model", "eval-stage1-final"),
            ("one-step + dry-air/positivity correction", "eval-stage1-final-corrected"),
            ("+ 4/8-step rollout fine-tuning (release)", "eval-ft8-corrected"),
        ],
        "recipe": [
            "Data: 2000–2009 (10 years), 6-hourly, 34 prognostic + 5 forcing channels; validation 2010.",
            "Model: 4×4 patches (4050 tokens), 8 pre-LN blocks × 6 heads, dim 384, linear un-patchify; 16.2 M parameters.",
            "Loss: area-weighted MSE on normalized 6-h tendencies.",
            "Training: one-step 40 epochs (lr 3e-4, cosine), then 4-step rollouts for 8 epochs (2 + 6, lr 5e-5) and 8-step for 4 (lr 3e-5). <a href=\"../training.html#v1\">Exact curriculum</a>.",
            "Inference: optional ACE-style dry-air and water-positivity correction (outside the network).",
        ],
        "changes": "baseline",
    },
    {
        "name": "v2",
        "title": "+ 3× data, diagnostics, ACE correctors in the loss, smoothing head, EMA, input noise",
        "run": "vit2-p4-d384-l8",
        "checkpoint": "vit-p4-d384-l8/release/v2.pt",
        "headline_eval": "eval-ft8",
        "stages": [("one-step, 36 epochs", "stage1"), ("4-step rollouts", "ft4"), ("8-step rollouts", "ft8"),
                   ("8-step rollouts, extended", "ft8b")],
        "evals": [("rollout-tuned (1 epoch at 8 steps)", "eval-ft8"),
                  ("+ 6 more epochs at 8 steps", "eval-ft8b"),
                  ("release + seam filter (inference)", "eval-ft8-filter")],
        "recipe": [
            "Data: <b>1980–2009 (30 years)</b>, plus ACE's 10 diagnostics (precipitation, latent/sensible heat, surface and TOA radiation, moisture advection) as extra outputs.",
            "Model: as v1 plus a 3×3 output convolution after un-patchify (identity-initialized; after the circuit graph's last receiver).",
            "Loss: MSE on tendencies + diagnostics, taken <b>after ACE-style correctors</b>: dry air, positivity, moisture budget, total energy (0.6 W/m² unaccounted heating).",
            "Training: EMA 0.999 weights; input noise (2% of each field's std); one-step 36 epochs (lr 3e-4), then 4-step (3 epochs, lr 5e-5) and 8-step (1, lr 2e-5) rollouts; the release is the 8-step checkpoint. <a href=\"../training.html#v2\">Exact curriculum</a>.",
        ],
        "changes": "data ×3; diagnostic outputs; correctors in the loss; output conv; EMA; input noise",
    },
    {
        "name": "v3",
        "title": "+ noise-conditioned LayerNorms (stochastic), CRPS loss",
        "run": "vit3-p4-d384-l8",
        "checkpoint": "vit-p4-d384-l8/release/v3.pt",
        "headline_eval": "eval-ft8",
        "stages": [("one-step, 12 epochs", "stage1"), ("4-step rollouts", "ft4"), ("8-step rollouts", "ft8"),
                   ("8-step rollouts, extended", "ft8b")],
        "evals": [("rollout-tuned (1 epoch at 8 steps)", "eval-ft8"),
                  ("+ 6 more epochs at 8 steps (4-member ensemble)", "eval-ft8b"),
                  ("release + seam filter (inference)", "eval-ft8-filter"),
                  ("extended + seam filter (inference)", "eval-ft8b-filter")],
        "recipe": [
            "As v2, plus noise-conditioned LayerNorms (per-token noise embedding modulates every norm's scale and shift; zero-initialized), as in ACE2S.",
            "Loss: fair CRPS over 2 members per sample, replacing MSE (ACE adds 0.1 × a spectral energy score; we leave it out: CRPS is the proper per-point score that calibrates the ensemble, and our spectral transform is not ACE's).",
            "Training: starts from v2's one-step checkpoint (the noise weights start at zero, so it starts as v2): 12 more one-step epochs with CRPS (lr 3e-4), then 4-step (3, lr 5e-5) and 8-step (1, lr 2e-5). <a href=\"../training.html#v3\">Exact curriculum</a>.",
            "Evaluation: ensemble-mean skill, CRPS and spread–skill.",
            "Load: <code>pg.load(\"v3\")</code>, <code>--source v3</code> (on Hugging Face as <code>v3/v3.pt</code>).",
        ],
        "changes": "stochastic (noise-conditioned norms); ensemble CRPS loss instead of MSE",
    },
    {
        "name": "v4",
        "title": "+ long probabilistic rollouts (4–16 steps), gradient on the last step only",
        "run": "vit4-p4-d384-l8",
        "checkpoint": "vit-p4-d384-l8/release/v4.pt",
        "headline_eval": "eval-long",
        "stages": [("long rollouts, 4–16 steps", "long")],
        "evals": [("long rollouts (4-member ensemble)", "eval-long"),
                  ("+ seam filter (inference)", "eval-long-filter")],
        "recipe": [
            "As v3 (stochastic, CRPS over 2 members, ACE-style correctors), starting from v3's one-step checkpoint, so it parallels v3's own short-rollout stages.",
            "Rollouts: each batch a random length between 4 and 16 six-hour steps (1–4 days); every step but the last runs without gradients, and the loss is taken on the last step only (the pushforward trick; ACE's optimize_last_step_only). Memory stays at one step's worth.",
            "Training: 4 epochs (lr 5e-5, global batch 8) from v3's one-step checkpoint (epoch 12). <a href=\"../training.html#v4\">Exact curriculum</a>.",
            "Why: rollout training was the biggest win in v1 and v2 and had not converged; with MSE, long horizons blur, but CRPS keeps the ensemble sharp.",
            "Load: <code>pg.load(\"v4\")</code>, <code>--source v4</code> (on Hugging Face as <code>v4/v4.pt</code>).",
        ],
        "changes": "long random-length rollouts with last-step gradient (stochastic, CRPS)",
    },
    {
        "name": "v5",
        "title": "+ spectral energy score, 4–24-step rollouts",
        "run": "vit5a-p4-d384-l8",
        "checkpoint": "vit5a-p4-d384-l8/long/best.pt",
        "headline_eval": "eval-long",
        "stages": [("4–24-step rollouts, energy score", "long")],
        "evals": [("v5 (4-member ensemble)", "eval-long"), ("+ seam filter (inference)", "eval-long-filter")],
        "recipe": [
            "v4 continued 4 epochs (lr 5e-5) with ACE's spectral energy score (0.1) beside CRPS (0.9), and random rollouts of 4–24 steps with the last-step gradient.",
            "Aimed at the patch-seam instability (see <a href=\"../index.html#stability\">Stability</a>): it slowed the seam growth by about a third and lengthened free runs, at a cost in 10-day skill. Not released.",
        ],
        "changes": "spectral energy score; longer rollouts",
    },
    {
        "name": "v6",
        "title": "+ patch jitter in training",
        "run": "vit6-p4-d384-l8",
        "checkpoint": "vit6-p4-d384-l8/long/best.pt",
        "headline_eval": "eval-long",
        "stages": [("4–24-step rollouts, random patch-grid shifts", "long")],
        "evals": [("v6", "eval-long"), ("+ patch jitter (inference)", "eval-long-jitter"),
                  ("+ jitter + seam filter", "eval-long-jitter-filter")],
        "recipe": [
            "v5 continued 4 epochs (lr 3e-5) with the network's input shifted by a random sub-patch offset at every forward pass (output shifted back before the loss), after NVlabs STRATA's tile jitter.",
            "Made the model ~50× more tolerant of shifted grids, but jitter alone still could not hold the climate. Not released.",
        ],
        "changes": "trained with patch jitter",
    },
    {
        "name": "v7",
        "title": "+ anti-aliased patch embedding: a realistic climate for 7–10 years",
        "run": "vit7-p4-d384-l8",
        "checkpoint": "vit-p4-d384-l8/release/v7.pt",
        "headline_eval": "eval-long",
        "stages": [("4–24-step rollouts, jitter, anti-aliased embedding", "long")],
        "evals": [("v7", "eval-long"), ("+ patch jitter (inference)", "eval-long-jitter"),
                  ("+ jitter + seam filter (recommended for long runs)", "eval-long-jitter-filter")],
        "recipe": [
            "v6 continued 4 epochs (lr 3e-5) with STRATA's anti-aliased patch embedding: the same patch weights applied as a stride-1 convolution, a (1,4,6,4,1) low-pass, then every 4th pixel. No new parameters; <code>hook_embed</code> is unchanged.",
            "Run with patch jitter and the seam filter, it holds a realistic climate for <b>6.6–10 years</b> (3 noise seeds over 2000–2009; one held all 10) with E3SM's variability and its response to SSTs, then settles about 11 K warm (see <a href=\"../index.html#decade\">Decade runs</a>).",
            "Load: <code>pg.load(\"v7\")</code>; <code>pg.rollout</code> turns on jitter and the filter for runs over 10 days.",
        ],
        "changes": "anti-aliased patch embedding",
    },
    {
        "name": "v8",
        "title": "3D (ViViT-style): one token per level, split horizontal/vertical heads",
        "run": "f3d-rope",
        "checkpoint": "f3d-rope/long/best.pt",
        "headline_eval": "eval-long-jitter-filter",
        "stages": [("4–24-step rollouts, v7's recipe, 8 epochs", "long")],
        "evals": [("v8 + jitter + seam filter", "eval-long-jitter-filter")],
        "recipe": [
            "A new architecture (<code>vit_interp/model3d.py</code>): each of the 8 levels (T, U, V, water) and a surface level (PS, TS, forcings) is its own token, 9 × 4050 tokens; in every block 4 heads attend across their level and 2 up and down their column (ViViT's factorized dot-product attention); rotary horizontal positions, periodic in longitude; anti-aliased patch embedding; 14.4 M parameters.",
            "Trained from scratch: 20 one-step epochs (the winner of a 7-way architecture screen, see <a href=\"../index.html#threed\">3D model</a>), then 8 epochs of v7's rollout recipe on Swing's backfill queue.",
            "The best forecaster of any version (5-day ACC 0.93 / 0.90 / 0.88 / 0.85, 10-day 0.63 / 0.59 / 0.56 / 0.45); its 10-year run stays realistic all decade with a slow drift in the last two years. Not yet released: multi-seed decade check, and the interpretability tools learning 3D tokens, first.",
        ],
        "changes": "3D tokens with vertical attention heads; trained from scratch",
    },
]


# -- reading runs --------------------------------------------------------------------------------


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _eval(v: dict, name: str) -> dict | None:
    p = RUNS / v["run"] / name / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def faithful_days(v: dict, name: str) -> float | None:
    """Days the free run's global means stay close to E3SM's: T_7 within 2 K, level-7 water within 10%,
    PS within 100 Pa, U_3 within 3 m/s (one-day running means; the first breach of any)."""
    p = RUNS / v["run"] / name / "fields.npz"
    if not p.exists():
        return None
    d = np.load(p)
    names = [str(x) for x in d["names"]]
    se, st = d["ser_e"], d["ser_t"]
    n = int(np.isfinite(se).all(1).sum())
    if n < 8:
        return None
    k = np.ones(4) / 4
    tests = [("T_7", lambda e, t: np.abs(e - t) > 2.0),
             ("specific_total_water_7", lambda e, t: np.abs(e - t) / np.abs(t) > 0.10),
             ("PS", lambda e, t: np.abs(e - t) > 100.0), ("U_3", lambda e, t: np.abs(e - t) > 3.0)]
    days = []
    for nm, bad in tests:
        c = names.index(nm)
        e, t = np.convolve(se[:n, c], k, "valid"), np.convolve(st[:n, c], k, "valid")
        b = np.flatnonzero(bad(e, t))
        days.append((b[0] + 3) / 4 if len(b) else n / 4)
    return float(min(days))


def _status(v: dict) -> str:
    if "/release/" in v["checkpoint"] and (RUNS / v["checkpoint"]).exists():
        return "released"
    if _eval(v, v["headline_eval"]):
        return "evaluated"
    if any(_jsonl(RUNS / v["run"] / d / "metrics.jsonl") for _, d in v["stages"]):
        return "training"
    return "planned"


def _onestep_end(v: dict) -> dict | None:
    rows = _jsonl(RUNS / v["run"] / v["stages"][0][1] / "metrics.jsonl")
    return rows[-1] if rows else None


# -- figures -------------------------------------------------------------------------------------


def _style(ax, title=None):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left")


def fig_acc(series: list[tuple[str, dict, str, object]], path: Path) -> bool:
    """series: (label, eval metrics, color, linestyle). Persistence from the first. A 2 x 2 grid
    and a legend underneath, wrapped, so no label is ever cut off or drawn over the axes."""
    import textwrap

    if not series:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), facecolor=SURF, sharey=True, sharex=True)
    x = [1, 3, 5, 10]
    for ax, v in zip(axes.flat, ACC_VARS, strict=True):
        for label, m, c, ls in series:
            ax.plot(x, [m["skill"][v][k]["acc"] for k in LEADS], linestyle=ls, marker="o", color=c,
                    lw=2, ms=5, label=textwrap.fill(label, 42))
        ax.plot(x, [series[0][1]["skill"][v][k]["acc_persistence"] for k in LEADS], "--", color=INK2,
                lw=1.2, label="persistence")
        _style(ax, v)
        ax.set_xticks(x)
        ax.axhline(0.6, color=INK2, lw=0.6, ls=":")
        ax.set_ylim(-0.1, 1.02)
    for ax in axes[1]:
        ax.set_xlabel("lead (days)", color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("anomaly correlation (2010)", color=INK2)
    n = len(series) + 1
    ncol = n if n <= 5 and max(len(l) for l, *_ in series) < 14 else (3 if n > 4 else 2)
    fig.tight_layout(rect=(0, 0.04 + 0.035 * ((n + ncol - 1) // ncol), 1, 1))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=ncol,
               frameon=False, labelcolor=INK, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_training(v: dict, path: Path) -> bool:
    """Validation loss of the first stage, then each stage's multi-day U_3 error, three panels a row."""
    import textwrap

    stages = [(label, _jsonl(RUNS / v["run"] / d / "metrics.jsonl")) for label, d in v["stages"]]
    stages = [(l, r) for l, r in stages if r]
    if not stages:
        return False
    panels = [("loss", stages[0])] + [("rmse", st) for st in stages]
    if len(stages) == 1 and len(stages[0][1]) == 1:
        panels = panels[1:]
    ncol = min(3, len(panels))
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.3 * nrow), facecolor=SURF, squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (kind, (label, rows)) in zip(axes.flat, panels):
        ep = [r["step"] / max(1, round(min(r2["step"] / r2["epoch"] for r2 in rows if r2["epoch"] >= 1)
                                    if any(r2["epoch"] >= 1 for r2 in rows) else r["step"])) for r in rows]
        if kind == "loss":
            ax.plot(ep, [r["val_loss"] for r in rows], "-", color=PALETTE[0], lw=2)
            ax.set_ylabel("validation loss", color=INK2)
        else:
            k = next((k for k in rows[0] if k.startswith("rmse_") and k != "rmse_1step"), None)
            if not k:
                continue
            days = int(k.split("_")[1].replace("step", "")) / 4
            ax.plot(ep, [r[k]["U_3"] for r in rows], "-o", color=PALETTE[1], lw=2, ms=3)
            ax.set_ylabel(f"{days:g}-day U_3 error (m/s)", color=INK2)
        _style(ax, textwrap.fill(label, 34))
        ax.set_xlabel("epoch", color=INK2)
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


# -- evaluation figures, redrawn from each evaluation's fields.npz in the site's style ------------

_PRETTY = {"PS": ("surface pressure", "hPa", 0.01), "TS": ("surface temperature", "K", 1),
           "T_7": ("near-surface temperature, T_7", "K", 1), "T_3": ("mid-troposphere temperature, T_3", "K", 1),
           "U_3": ("jet-level zonal wind, U_3", "m/s", 1), "V_3": ("jet-level meridional wind, V_3", "m/s", 1),
           "U_7": ("near-surface zonal wind, U_7", "m/s", 1),
           "specific_total_water_7": ("near-surface water, q_7", "g/kg", 1000)}


def _pretty(v):
    return _PRETTY.get(v, (v, "", 1))


def fig_eval_maps(z, lat, lon, days: float, path: Path) -> None:
    from vit_interp.evaluate import MAP_VARS

    names = list(z["names"])
    try:
        import cartopy.crs as ccrs

        proj, tr = ccrs.PlateCarree(central_longitude=180), ccrs.PlateCarree()
    except Exception:
        proj = tr = None
    fig = plt.figure(figsize=(13, 2.55 * len(MAP_VARS)), facecolor=SURF, layout="constrained")
    for r, v in enumerate(MAP_VARS):
        label, unit, k = _pretty(v)
        t, e = z["mean_t"][names.index(v)] * k, z["mean_e"][names.index(v)] * k
        lo, hi = np.nanpercentile(t, [1, 99]); d = e - t; dm = float(np.nanpercentile(np.abs(d), 99)) or 1.0
        for c, (f, kw, col) in enumerate([(t, dict(vmin=lo, vmax=hi, cmap="viridis"), "E3SM"),
                                          (e, dict(vmin=lo, vmax=hi, cmap="viridis"), "emulator"),
                                          (d, dict(vmin=-dm, vmax=dm, cmap="RdBu_r"), "emulator − E3SM")]):
            ax = fig.add_subplot(len(MAP_VARS), 3, 3 * r + c + 1, projection=proj) if proj else \
                fig.add_subplot(len(MAP_VARS), 3, 3 * r + c + 1)
            im = ax.pcolormesh(lon, lat, f, shading="auto", rasterized=True, **({"transform": tr} if tr else {}), **kw)
            if tr:
                try:
                    ax.coastlines(lw=0.35)
                except Exception:
                    pass
            if r == 0:
                ax.set_title(col, color=INK, fontsize=13, fontweight="bold")
            if c == 0:
                ax.text(-0.04, 0.5, label, transform=ax.transAxes, rotation=90, va="center", ha="right", color=INK, fontsize=11)
            cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, aspect=14)
            cb.ax.tick_params(labelsize=9, colors=INK2)
            cb.set_label(unit, color=INK2, fontsize=9)
    fig.suptitle(f"Time mean over {days:.0f} days of 2010: free-running emulator vs E3SM", color=INK, fontsize=14)
    fig.savefig(path); plt.close(fig)


def fig_eval_zonal(z, lat, path: Path) -> None:
    from vit_interp.data import PROG_NAMES

    P = PROG_NAMES.index
    seen = np.abs(z["zm_t"]).sum((1, 2)) > 0
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=SURF, layout="constrained")
    for r, (var, unit) in enumerate([("U", "m/s"), ("T", "K")]):
        ch = [P(f"{var}_{lv}") for lv in range(8)]
        t, e = z["zm_t"][seen][:, ch].mean(0), z["zm_e"][seen][:, ch].mean(0)
        lo, hi = np.percentile(t, [1, 99]); d = e - t; dm = float(np.abs(d).max()) or 1.0
        for c, (f, kw, col) in enumerate([(t, dict(vmin=lo, vmax=hi, cmap="viridis"), "E3SM"),
                                          (e, dict(vmin=lo, vmax=hi, cmap="viridis"), "emulator"),
                                          (d, dict(vmin=-dm, vmax=dm, cmap="RdBu_r"), "emulator − E3SM")]):
            ax = axes[r, c]
            im = ax.imshow(f, aspect="auto", extent=[lat[0], lat[-1], 7.5, -0.5], interpolation="nearest", **kw)
            ax.set_title(f"zonal-mean {'wind U' if var == 'U' else 'temperature T'}: {col}", color=INK, loc="left")
            ax.set_xlabel("latitude", color=INK2)
            if c == 0:
                ax.set_ylabel("model level (0 = top)", color=INK2)
            ax.tick_params(colors=INK2)
            cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label(unit, color=INK2)
    fig.savefig(path); plt.close(fig)


def fig_eval_series(z, path: Path) -> None:
    from vit_interp.evaluate import SERIES_VARS

    names = list(z["names"])
    n = len(z["ser_e"]); ok = ~np.isnan(z["ser_e"][:, 0])
    tday = (np.arange(n) + 1) / 4
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), facecolor=SURF, sharex=True)
    for ax, v in zip(axes.flat, SERIES_VARS):
        label, unit, k = _pretty(v)
        ax.plot(tday[ok], z["ser_t"][ok, names.index(v)] * k, color=INK2, lw=1.2, label="E3SM")
        ax.plot(tday[ok], z["ser_e"][ok, names.index(v)] * k, color=PALETTE[1], lw=1.2, label="emulator")
        _style(ax, f"global mean {label} ({unit})")
    for ax in axes[1]:
        ax.set_xlabel("day of 2010", color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, labelcolor=INK)
    fig.savefig(path); plt.close(fig)


def eval_figures(ev: Path, d: Path) -> dict[str, str]:
    """Redraw an evaluation's maps, zonal means and series into d; snapshots are copied (not saved
    as data). Returns {figure name: file name} of what exists."""
    figs = {}
    f = ev / "fields.npz"
    meta = json.loads((ROOT / "data-v2" / "meta.json").read_text())
    lat, lon = np.asarray(meta["lat"]), np.asarray(meta["lon"])
    if f.exists():
        z = np.load(f)
        days = float((~np.isnan(z["ser_e"][:, 0])).sum()) / 4
        for name, fn in (("maps_time_mean", lambda p: fig_eval_maps(z, lat, lon, days, p)),
                         ("zonal_mean_U_T", lambda p: fig_eval_zonal(z, lat, p)),
                         ("global_mean_series", lambda p: fig_eval_series(z, p))):
            try:
                fn(d / f"{name}.png"); figs[name] = f"{name}.png"
            except Exception as exc:  # fall back to the evaluation's own figure
                print(f"  {ev.name}/{name}: redraw failed ({exc}); copying")
    for name in ("snapshots", "maps_time_mean", "zonal_mean_U_T", "global_mean_series"):
        if name not in figs and (ev / f"{name}.png").exists():
            shutil.copy(ev / f"{name}.png", d / f"{name}.png"); figs[name] = f"{name}.png"
    return figs


def fig_rossby(explore: Path, path: Path) -> list[dict]:
    phis = [p for p in (30, 45, 60, -45) if (explore / f"rossby_phi{p}.json").exists()]
    if not phis:
        return []
    ncol = 2 if len(phis) > 2 else len(phis)
    nrow = (len(phis) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.6 * nrow), facecolor=SURF, squeeze=False)
    for ax in axes.flat[len(phis):]:
        ax.set_visible(False)
    axes = axes.flat
    out = []
    for ax, phi in zip(axes, phis, strict=True):
        m = json.loads((explore / f"rossby_phi{phi}.json").read_text())["measure"]
        s = [r["alpha"] for r in m["rows"]]
        f = np.array([r["model"] for r in m["rows"]])
        g = np.array([r["law"] for r in m["rows"]])
        ax.plot(s, f, "-o", color=PALETTE[0], lw=2, ms=4, label="ViT")
        ax.plot(s, g, "-s", color=PALETTE[1], lw=2, ms=4, label="barotropic theory")
        lat = f"{abs(phi)}°{'N' if phi > 0 else 'S'}"
        _style(ax, lat)
        ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
        ax.set_xlabel("zonal wavenumber s", color=INK2)
        ax.set_ylabel("ω − kŪ (rad/day)", color=INK2)
        dop = explore / f"rossby_doppler_phi{phi}.json"
        out.append({"lat": lat, "corr": float(np.corrcoef(f, g)[0, 1]), "ratio": float(np.polyfit(g, f, 1)[0]),
                    "M": m["M"], "M_doppler": json.loads(dop.read_text())["measure"]["M"] if dop.exists() else None})
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, labelcolor=INK)
    fig.savefig(path)
    plt.close(fig)
    return out


# -- html pieces ---------------------------------------------------------------------------------


CSS = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--line:#e4e3df;--card:#ffffff;--acc:#2a78d6}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--line:#383835;--card:#232322;--acc:#3987e5}
 img.fig{background:#fcfcfb}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.9rem;margin:.2em 0}h2{margin-top:2.2em;border-bottom:1px solid var(--line);padding-bottom:.2em}
h3{margin-top:1.6em}p.lede{color:var(--fg2);font-size:1.08rem}code{font-size:.9em;background:var(--line);padding:0 .3em;border-radius:4px}
.tw{overflow-x:auto}table{border-collapse:collapse;margin:.6em 0;font-size:.92rem}th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}
th{color:var(--fg2);font-weight:600}img.fig{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;margin:.4em 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.split2{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:16px;align-items:center}
@media (max-width:700px){.split2{grid-template-columns:1fr}}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.kpi{font-size:1.7rem;font-weight:650}.muted{color:var(--fg2);font-size:.9rem}a{color:var(--acc)}nav a{margin-right:14px}
.tag{display:inline-block;font-size:.78rem;padding:1px 8px;border-radius:10px;border:1px solid var(--line);color:var(--fg2)}
.tag.released{border-color:#1baf7a}.tag.training{border-color:#eda100}
"""


def _page(title: str, body: str, up: str = "") -> str:
    nav = f'<nav class="muted"><a href="{up}index.html">← all versions</a></nav>' if up else ""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style>'
            f"</head><body><main>{nav}{body}<p class='muted' style='margin-top:3em'>Updated "
            f"{datetime.now():%Y-%m-%d %H:%M} CST.</p></main></body></html>")


def _table(header, rows) -> str:
    th = "".join(f"<th>{h}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def _img(src, alt) -> str:
    return f'<img class="fig" src="{src}" alt="{html.escape(alt)}" loading="lazy">' if src else ""


def _fmt(x, f="{:.2f}"):
    return "–" if x is None else f.format(x)


def _eval_row(label, m, faithful=None) -> list:
    free = (f"{faithful:.0f} d climate holds* · " if faithful is not None else "") + \
        f"{m['days_run']:.0f} d run" + (" (then drifts)" if m.get("blowup") else " (stable)")
    acc = " / ".join(f"{m['skill'][v]['5d']['acc']:.2f}" for v in ACC_VARS)
    rmse = " / ".join(f"{m['skill'][v]['5d']['rmse']:.3g}" for v in ACC_VARS)
    return [html.escape(label), free, acc, rmse]


EVAL_HEADER = ["model", "free run (2010)", "5-day ACC: PS / T_7 / U_3 / V_3",
               "5-day RMSE: PS (Pa) / T_7 (K) / U_3, V_3 (m/s)"]


def version_page(v: dict, out: Path) -> dict:
    d = out / v["name"]
    d.mkdir(parents=True, exist_ok=True)
    status = _status(v)
    body = [f"<h1>{v['name']}: {html.escape(v['title'])}</h1>",
            f'<p><span class="tag {status}">{status}</span> &nbsp; changes from the previous version: '
            f"{html.escape(v['changes'])}</p>",
            "<h2>Recipe</h2><ul>" + "".join(f"<li>{r}</li>" for r in v["recipe"]) + "</ul>"]
    if fig_training(v, d / "training.png"):
        body.append("<h2>Training</h2>" + _img("training.png", "training curves"))
        end = _onestep_end(v)
        if end:
            r1 = end["rmse_1step"]
            body.append(f"<p class='muted'>End of the one-step stage so far (epoch {end['epoch']}): validation loss "
                        f"{end['val_loss']:.4f}; 1-step RMSE PS {r1['PS']:.1f} Pa, T_7 {r1['T_7']:.3f} K, "
                        f"U_3 {r1['U_3']:.2f} m/s.</p>")
    evals = [(label, _eval(v, e), e) for label, e in v["evals"]]
    evals = [(l, m, e) for l, m, e in evals if m]
    if evals:
        body.append("<h2>Skill on 2010</h2>" + _table(EVAL_HEADER, [_eval_row(l, m, faithful_days(v, e)) for l, m, e in evals]))
        styles = ["-", (0, (4, 3)), "-", "-", "-"]
        if fig_acc([(l, m, PALETTE[i], styles[i]) for i, (l, m, _) in enumerate(evals)], d / "skill_acc.png"):
            body.append(_img("skill_acc.png", "anomaly correlation vs lead"))
        last = evals[-1][2]
        figs = eval_figures(RUNS / v["run"] / last, d)
        if len(evals) > 1:  # before/after snapshots across the version's own history
            first = RUNS / v["run"] / evals[0][2] / "snapshots.png"
            if first.exists():
                shutil.copy(first, d / "snapshots_first.png")
                body.append(f"<h3>Snapshots: {html.escape(evals[0][0])} vs {html.escape(evals[-1][0])}</h3>"
                            f'<div class="grid"><div>{_img("snapshots_first.png", "first")}</div>'
                            f'<div>{_img(figs.get("snapshots"), "last")}</div></div>')
        elif figs.get("snapshots"):
            body.append("<h3>Snapshots</h3>" + _img(figs["snapshots"], "snapshots"))
        for name, title in (("maps_time_mean", "Time-mean maps"), ("zonal_mean_U_T", "Zonal means"),
                            ("global_mean_series", "Global-mean drift")):
            if figs.get(name):
                body.append(f"<h3>{title}</h3>" + _img(figs[name], title))
    elif status == "planned":
        body.append("<p class='muted'>Not trained yet.</p>")
    else:
        body.append("<p class='muted'>Evaluation pending (it runs automatically at the end of training).</p>")
    if (RUNS / v["checkpoint"]).exists():
        body.append(f"<p class='muted'>Checkpoint: <code>{RUNS / v['checkpoint']}</code></p>")
    (d / "index.html").write_text(_page(f"vit-interp {v['name']}", "".join(body), up="../"))
    # the headline is the released model as it comes (no inference-time filter)
    head_name = v["headline_eval"] if _eval(v, v["headline_eval"]) else next((e for _, _, e in reversed(evals)), v["headline_eval"])
    head = _eval(v, head_name)
    return {"v": v, "status": status, "eval": head, "onestep": _onestep_end(v),
            "faithful": faithful_days(v, head_name)}


def versions_table(summ: list[dict]) -> str:
    rows = []
    for s in summ:
        v, m, o = s["v"], s["eval"], s["onestep"]
        r1 = o["rmse_1step"]["U_3"] if o else None
        k5 = next((k for k in (o or {}) if k.startswith("rmse_") and k != "rmse_1step"), None)
        r5 = o[k5]["U_3"] if k5 else None
        rows.append([
            f'<a href="{v["name"]}/index.html"><b>{v["name"]}</b></a>',
            html.escape(v["changes"]),
            f'<span class="tag {s["status"]}">{s["status"]}</span>',
            (f"{_fmt(o['val_loss'], '{:.3f}')} (ep {o['epoch']})" if o else "–"),
            f"{_fmt(r1)} / {_fmt(r5, '{:.1f}')}" + (
                f" ({int(k5.split('_')[1].replace('step', '')) / 4:g}-day)" if k5 and k5 != "rmse_20step" else ""),
            (f"{s['faithful']:.0f} d" if s.get("faithful") is not None else "–"),
            (f"{m['days_run']:.0f} d" + (" (drifts)" if m.get("blowup") else "")) if m else "–",
            " / ".join(f"{m['skill'][x]['5d']['acc']:.2f}" for x in ACC_VARS) if m else "–",
        ])
    return _table(["version", "what changed", "status", "one-step val loss",
                   "first-stage U_3 RMSE: 1-step / 5-day (m/s)", "climate holds*", "free run (2010)",
                   "5-day ACC: PS / T_7 / U_3 / V_3 (final model)"], rows)


def notebooks_html(out: Path) -> list[tuple[str, str]]:
    nbdir = out / "notebooks"
    nbdir.mkdir(exist_ok=True)
    done = []
    for nb in sorted(NB_EXEC.glob("[0-9]*.ipynb")):
        target = nbdir / (nb.stem + ".html")
        if not target.exists() or target.stat().st_mtime < nb.stat().st_mtime:
            subprocess.run([str(Path.home() / "vit-interp/.venv/bin/jupyter"), "nbconvert", "--to", "html",
                            "--output-dir", str(nbdir), str(nb)], check=True, capture_output=True)
        done.append((nb.stem, f"notebooks/{nb.stem}.html"))
    return done


NB_TITLES = {
    "00_quickstart": "one step and a 5-day forecast against E3SM",
    "01_attention_atlas": "where each of the 48 heads looks",
    "02_logit_lens": "decode every block: when does the model know the answer?",
    "03_ablation_lab": "knock out heads and MLPs",
    "04_direct_attribution": "who writes the jet-level wind over a box",
    "05_exege_latents": "latent archives, region analysis, sparse autoencoders (exege)",
    "06_rossby_probe": "plant a Rossby wave: does the model know β?",
    "07_exege_circuits_and_laws": "Vi-CD edge circuits and LAWFUL through exege",
}


DRIFT = ROOT / "drift"


def fig_stability(path: Path) -> bool:
    """Patch-scale power, global-mean T_3 error, and the hot spot, from scripts/drift_probe.py runs."""
    need = ["probe-v4-all4", "probe-v4-f60s1-365", "probe-v4-f60s1-seed1", "probe-v4-f60s1-seed2", "probe-v2-all4"]
    if not all((DRIFT / f"{n}.npz").exists() for n in need):
        return False
    z = {n: np.load(DRIFT / f"{n}.npz") for n in need}
    fig = plt.figure(figsize=(11, 9.0), facecolor=SURF, layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    runs = [("probe-v2-all4", "v2", PALETTE[1], "-"), ("probe-v4-all4", "v4", PALETTE[0], "-"),
            ("probe-v4-f60s1-365", "v4 + seam filter (3 seeds)", PALETTE[2], "-"),
            ("probe-v4-f60s1-seed1", None, PALETTE[2], "-"), ("probe-v4-f60s1-seed2", None, PALETTE[2], "-")]
    for n, lab, c, ls in runs:
        q = z[n]
        sn = list(q["spec_names"]); j = sn.index("T_3")
        r = q["spec_e"][:, j, 88:93].sum(-1) / q["spec_t"][:, j, 88:93].sum(-1)
        d = np.arange(1, len(r) + 1)
        ax[0].semilogy(d, r, color=c, ls=ls, lw=1.6, label=lab)
        g = q["g_err"][:, list(q["names"]).index("T_3")]
        ax[1].plot(np.arange(1, len(g) + 1), g, color=c, ls=ls, lw=1.6, label=lab)
    _style(ax[0], "patch-scale (4°) power in T_3, ÷ E3SM")
    ax[0].set_xlabel("day of free run", color=INK2); ax[0].axhline(1, color="#888", lw=0.8)
    ax[0].legend(frameon=False, labelcolor=INK, loc="upper left")
    _style(ax[1], "global-mean T_3 error (K)")
    ax[1].set_xlabel("day of free run", color=INK2); ax[1].axhline(0, color="#888", lw=0.8); ax[1].set_ylim(-5, 30)
    q = z["probe-v4-all4"]; mn = list(q["map_names"])
    e = q["maps"][124, mn.index("T_3")].astype(np.float32)
    try:
        import cartopy.crs as ccrs

        m = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree(central_longitude=180))
        im = m.pcolormesh(q["lon"], q["lat"], e, cmap="RdBu_r", vmin=-30, vmax=30, shading="auto",
                          transform=ccrs.PlateCarree(), rasterized=True)
        m.contour(q["lon"], q["lat"], np.abs(e) > 20, levels=[0.5], colors="k", linewidths=0.7, transform=ccrs.PlateCarree())
        try:
            m.coastlines(lw=0.4)
        except Exception:
            pass
    except Exception:
        m = fig.add_subplot(gs[1, :])
        im = m.pcolormesh(q["lon"], q["lat"], e, cmap="RdBu_r", vmin=-30, vmax=30, shading="auto")
    m.set_title("v4 after 125 days: T_3 error (K); the runaway starts at 26°N, 85°E (black: |error| > 20 K)",
                color=INK, fontsize=12, loc="left")
    fig.colorbar(im, ax=m, shrink=0.8, pad=0.02, label="K")
    fig.savefig(path); plt.close(fig)
    return True


def stability_section(out: Path) -> str:
    rows = []
    for v, run, raw, filt in [("v2", "vit2-p4-d384-l8", "eval-ft8", "eval-ft8-filter"),
                              ("v3", "vit3-p4-d384-l8", "eval-ft8", "eval-ft8-filter"),
                              ("v3, extended", "vit3-p4-d384-l8", "eval-ft8b", "eval-ft8b-filter"),
                              ("v4", "vit4-p4-d384-l8", "eval-long", "eval-long-filter")]:
        a, b = _eval({"run": run}, raw), _eval({"run": run}, filt)
        if not (a and b):
            continue
        fa, fb = faithful_days({"run": run}, raw), faithful_days({"run": run}, filt)
        acc = lambda m, lead: " / ".join(f"{m['skill'][x][lead]['acc']:.2f}" for x in ACC_VARS)
        run_days = lambda m: f"{m['days_run']:.0f} d" + (" (blew up)" if m.get("blowup") else "")
        rows.append([v, f"{fa:.0f} d → <b>{fb:.0f} d</b>", f"{run_days(a)} → {run_days(b)}",
                     f"{acc(a, '5d')} → {acc(b, '5d')}", f"{acc(a, '10d')} → {acc(b, '10d')}"])
    if not rows:
        return ""
    img = _img("stability.png", "patch-period power growth, global-mean drift and the hot spot") if fig_stability(out / "stability.png") else ""
    return f"""<h2 id="stability">Why free runs go unstable, and the fix</h2>
<p>Every version tracks E3SM's global means for months, then jumps within weeks: mid-troposphere temperature rises by 10–25 K,
with water and surface pressure following. The cause is the <b>4° patch grid</b>. Every step, the network adds a little energy at
the patch period (about 2× E3SM's after one step). Training never shows it what that does over months, and the excess grows
exponentially (×34 in a month and ×145 in three for v4; ×85 and ×1,400 for v2) until it sets off a local runaway where the
seams meet strong forcing: the southern edge of the Tibetan Plateau for v4, the poles for v2. From there it spreads globally.
The conservation correctors cannot stop it (runs with and without the moisture and energy budgets fail on the same day).</p>
{img}
<p><b>The fix at inference:</b> a seam filter on each step's increment, outside the network (so hooks, circuits and LAWFUL are
unaffected). It damps zonal wavenumbers above 60, more strongly toward the poles as GCMs do, and makes one 1-2-1 pass in
latitude, which is the part that matters most (filtering in longitude alone made v4 worse). Every model then runs the full year
without blowing up, and the climate holds two to three times longer for the stochastic models, for 0.01–0.02 of 5-day anomaly correlation:</p>
<div class="tw">{_table(["model", "climate holds*", "free run (2010)", "5-day ACC PS / T_7 / U_3 / V_3", "10-day ACC"], rows)}</div>
<p>Use it with <code>pg.rollout(em, t, steps, smooth=True)</code> (the default for runs longer than 10 days),
<code>em.rollout(..., smooth=SeamFilter(lat, 360))</code> or <code>python -m vit_interp.evaluate ... --filter 60 --shapiro 1</code>.
<b>The fix in the model:</b> the spectral energy score (v5), training with patch jitter (v6) and NVIDIA STRATA's
anti-aliased patch embedding (v7), run with jitter and the seam filter, give a realistic climate for 6.6–10 years
(<a href="#decade">Decade runs</a>); the 3D model v8 is the best forecaster yet (<a href="#threed">3D model</a>).</p>"""


def decade_section(out: Path) -> str:
    """10-year free runs (scripts/long_run.py, 2000-2009) of the main configurations."""
    runs = [("v4 + filter", "long-v4f"), ("v5a + jitter + filter", "long-v5a-jf"),
            ("v5a + jitter + filter + small-scale damping", "long-v5a-jf-s50t90"),
            ("v6 + jitter + filter", "long-v6-jit-f"), ("v7, no help", "long-v7"), ("v7 + jitter", "long-v7-jit"),
            ("<b>v7 + jitter + filter (seed 0)</b>", "long-v7-jit-f"), ("v7 + jitter + filter (seed 1)", "long-v7-jit-f-seed1"),
            ("v7 + jitter + filter (seed 2)", "long-v7-jit-f-seed2"),
            ("<b>3D (split heads, rotary) + jitter + filter</b>", "long-f3d-rope-jit-f")]
    rows, series = [], []
    for label, tag in runs:
        f = DRIFT / f"{tag}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        n = list(z["names"]); d = z["g_e"] - z["g_t"]
        t7, ps, u3, q7 = n.index("T_7"), n.index("PS"), n.index("U_3"), n.index("specific_total_water_7")
        hold = next((i + 1 for i in range(len(d)) if abs(d[i, t7]) > 2 or abs(d[i, ps]) > 100 or abs(d[i, u3]) > 3), len(d))
        last = d[-730:] if len(d) >= 730 else d
        var = ""
        if "dd_e" in z.files:
            r = z["dd_e"] / z["dd_t"]
            var = f"{min(r[n.index(v)] for v in ('T_0', 'T_3', 'T_7', 'U_3')):.2f}–{max(r[n.index(v)] for v in ('T_0', 'T_3', 'T_7', 'U_3')):.2f}"
        rows.append([label, f"{hold / 365:.1f} y", f"{last[:, t7].mean():+.1f} K", f"{1e3 * last[:, q7].mean():+.2f} g/kg", var or "–"])
        if tag in ("long-v4f", "long-v5a-jf", "long-v6-jit-f", "long-v7-jit-f", "long-v7-jit-f-seed1", "long-v7-jit-f-seed2",
                   "long-f3d-rope-jit-f"):
            series.append((label.replace("<b>", "").replace("</b>", ""), np.arange(1, len(d) + 1) / 365, d[:, t7]))
    if not rows:
        return ""
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=SURF)
    for i, (lab, t, y) in enumerate(series):
        ax.plot(t, y, color=PALETTE[i % len(PALETTE)], lw=1.3, label=lab)
    _style(ax, "global-mean near-surface temperature error (K), 10-year free runs")
    ax.set_ylim(-15, 15); ax.axhline(0, color="#888", lw=0.8); ax.set_xlabel("years from 2000-01-01", color=INK2)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.savefig(out / "decade.png"); plt.close(fig)
    return ("<h2 id='decade'>Decade runs</h2><p>Free runs over 2000–2009 (true forcings, prescribed ocean TS, the "
            "checkpoint's correctors), state carried across years. <b>v7 with patch jitter and the seam filter holds a "
            "realistic climate for 6.6 to 10 years</b> across three noise seeds (one held the full decade): yearly global "
            "biases within 0.1 K near the surface and 0.05 g/kg of water while realistic, E3SM's day-to-day variability at "
            "every level, and Niño-3.4 anomalies that follow E3SM's (r = 0.91–0.96 near-surface temperature, 0.90–0.94 "
            "water). The two shorter runs then settle into a steady state about 11 K warm (no blow-up). Caveat: these "
            "years lie in the training range (1980–2009); 2010 is the only held-out year.</p>"
            + _img("decade.png", "10-year free runs: global near-surface temperature error")
            + _table(["configuration", "climate holds*", "T_7 bias, last 2 y", "water bias, last 2 y",
                      "day-to-day variability ÷ E3SM (T, U)"], rows)
            + "<p class='muted'>Patch jitter: the patch grid shifts by a sub-patch offset every step (after NVlabs STRATA). "
              "Seam filter: <code>vit_interp.filters.SeamFilter</code>. Both run outside the network.</p>")


def threed_section(out: Path) -> str:
    """The 3D (ViViT-style) line: the architecture screen and the rollout fine-tune in progress."""
    screen = [("2D baseline (16.7 M)", "s2d-d384"), ("3D split heads, 256 (7.5 M)", "s3d-split-d256"),
              ("3D split heads, 256, rotary", "s3d-split-d256-rope"), ("3D factorized, 256 (10 M)", "s3d-fact-d256"),
              ("3D alternating, 256 (7.8 M)", "s3d-alt-d256"), ("3D split heads, 384 (16.4 M)", "s3d-split-d384"),
              ("<b>3D split heads, 384, rotary (14.4 M)</b>", "s3d-split-d384-rope")]
    rows, curves = [], []
    for label, r in screen:
        m = _jsonl(RUNS / r / "stage1" / "metrics.jsonl")
        if not m:
            continue
        by = {d["epoch"]: d for d in m}
        last = m[-1]
        k = next(x for x in last if x.startswith("rmse_") and x != "rmse_1step")
        cell = lambda e: f"{by[e]['val_loss']:.4f}" if e in by else "–"  # noqa: E731
        rows.append([label, cell(5), cell(10), cell(20), f"{last['rmse_1step']['U_3']:.2f} / {last['rmse_1step']['T_7']:.2f}",
                     f"{last[k]['U_3']:.2f} / {last[k]['T_7']:.2f}"])
        curves.append((label.replace("<b>", "").replace("</b>", ""), [d["epoch"] for d in m], [d["val_loss"] for d in m]))
    if not rows:
        return ""
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURF)
    for i, (lab, e, v) in enumerate(curves):
        ax.plot(e, v, color=PALETTE[i % len(PALETTE)], ls="-" if i < len(PALETTE) else (0, (4, 2)), lw=1.6, label=lab)
    ax.set_ylim(0.05, 0.13)
    _style(ax, "one-step validation loss (2010), same recipe for every architecture")
    ax.set_xlabel("epoch", color=INK2)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.savefig(out / "screen3d.png"); plt.close(fig)
    ft = _jsonl(RUNS / "f3d-rope" / "long" / "metrics.jsonl")
    v7 = _jsonl(RUNS / "vit7-p4-d384-l8" / "long" / "metrics.jsonl")
    ft_html = ""
    if ft and v7:
        f = lambda d: [f"{d['crps']:.4f}", f"{d['spread_skill']:.2f}", f"{d['rmse_1step']['U_3']:.2f} / {d['rmse_1step']['T_7']:.3f}",  # noqa: E731
                       f"{d['rmse_40step']['U_3']:.2f} / {d['rmse_40step']['T_7']:.2f} / {d['rmse_40step']['PS']:.0f}"]
        per = min(d["step"] / d["epoch"] for d in ft if d["epoch"] >= 1) if any(d["epoch"] >= 1 for d in ft) else 1
        seen = {}
        for d in ft:  # one row per tenth of an epoch (a walltime stop can land on an epoch end)
            seen[round(d["step"] / per, 1)] = d
        frows = [[f"3D, epoch {e:.1f}"] + f(d) for e, d in sorted(seen.items())]
        frows.append(["<b>v7 (released)</b>"] + f(v7[-1]))
        ev = _eval({"run": "f3d-rope"}, "eval-long-jitter-filter")
        v7e = _eval({"run": "vit7-p4-d384-l8"}, "eval-long-jitter-filter")
        status = "Its 2010 evaluation and 10-year run follow the last epoch."
        if ev and v7e:
            accs = lambda m, L: " / ".join(f"{m['skill'][x][L]['acc']:.2f}" for x in ACC_VARS)  # noqa: E731
            status = (f"<b>On the 2010 evaluation (8 starts, 4 members, jitter + filter): 5-day anomaly correlation "
                      f"{accs(ev, '5d')} against v7's {accs(v7e, '5d')}, 10-day {accs(ev, '10d')} against "
                      f"{accs(v7e, '10d')} (PS / T_7 / U_3 / V_3), the best of any version</b>; the 2010 free run "
                      f"holds its climate for {faithful_days({'run': 'f3d-rope'}, 'eval-long-jitter-filter'):.0f} days. "
                      "Its 10-year run (one seed) stays realistic all decade, with a slow drift in the last two years "
                      "(see <a href='#decade'>Decade runs</a>).")
        ft_html = ("<h3>The winner through v7's rollout recipe</h3><p>The 3D split-head model with rotary "
                   "positions, fine-tuned for 8 epochs with v7's recipe (noise-conditioned norms, CRPS + spectral energy "
                   "score, 4–24-step rollouts with patch jitter, anti-aliased embedding), on Swing's backfill queue in "
                   f"4-hour links. Validation on 2010, against v7's final numbers. {status}</p>"
                   + _table(["checkpoint", "CRPS", "spread/skill", "1-step error U_3 / T_7", "10-day error U_3 / T_7 / PS"], frows))
    return ("<h2 id='threed'>A 3D model: one token per level (ViViT-style)</h2>"
            "<p>The 2D model folds all 8 levels of a patch into one token, so vertical coupling happens out of attention's "
            "reach. The 3D models (<code>vit_interp/model3d.py</code>) give each level its own token plus a surface level, "
            "and factorize attention as ViViT does: <b>split heads</b> (in every block some heads attend up and down the "
            "column, the rest across the level), <b>factorized</b> (a horizontal then a vertical attention sublayer per "
            "block) or <b>alternating</b> blocks (as NVIDIA STRATA). Every architecture below was trained the same way "
            "(v2's one-step recipe, 20 epochs, global batch 16, anti-aliased embedding), including a 2D baseline.</p>"
            + _img("screen3d.png", "one-step validation loss per epoch, every architecture")
            + _table(["architecture", "val loss, epoch 5", "epoch 10", "epoch 20", "1-step error U_3 / T_7", "5-day error U_3 / T_7"], rows)
            + "<p>At the same size, the 3D split-head model has 27% lower one-step loss than the 2D model; rotary positions "
              "give it the best 5-day errors of any model. One-step models of every kind blow up within months in free "
              "runs: stability comes from the rollout stage.</p>" + ft_html)


def analysis_sections(out: Path, release: dict) -> str:
    """Rossby and circuit results of the released model's exploration run."""
    explore = RUNS / release["run"] / "explore-final"
    parts = [f"<p class='muted'>Analyses below are of <b>{release['name']}</b> "
             f"(<code>{explore}</code>).</p>"]
    ross = fig_rossby(explore, out / "rossby.png")
    if ross:
        corr = ", ".join(f"{r['corr']:+.2f} at {r['lat']}" for r in ross)
        ratio = ", ".join(f"{r['ratio']:.1f}× at {r['lat']}" for r in ross)
        steer = ""
        u3, u5 = explore / "rossby_corr_phi45_U_3.json", explore / "rossby_corr_phi45_U_5.json"
        if u3.exists() and u5.exists():
            c3, c5 = (json.loads(p.read_text())["measure"]["M"] for p in (u3, u5))
            steer = (f" At 45°N the shape correlation {'rises' if c5 > c3 else 'falls'} from {c3:.2f} to {c5:.2f} when "
                     "the flow the law Doppler-shifts by is taken lower (U_5 instead of U_3).")
        rows = [[r["lat"], f"{r['corr']:+.2f}", f"{r['ratio']:.1f}×", f"{r['M']:.0f}",
                 _fmt(r["M_doppler"])] for r in ross]
        parts.append(
            "<h2 id='rossby'>Do Rossby waves emerge? (LAWFUL)</h2>"
            "<p>A small, geostrophically balanced, non-divergent wave of zonal wavenumber s = 2–14 is planted on 8 real "
            "2010 states; the model takes one step; the wave's frequency is read off its response and compared with "
            "barotropic theory, ω − kŪ = −βk/(k²+l²).</p>" + _img("rossby.png", "Rossby dispersion")
            + _table(["latitude", "shape correlation with theory", "model / theory magnitude", "LAWFUL M (eq. 10)",
                      "M incl. Doppler shift"], rows)
            + f"<p>Everywhere the model moves planted waves <b>westward relative to the U_3 flow</b> and faster than "
            f"barotropic theory (fitted magnitude {ratio}), so the misfit score M is strongly negative. The shape across "
            f"wavenumbers agrees to varying degrees: correlation {corr}.{steer}</p>"
            "<p>LAWFUL's circuit search over whole blocks finds no τ-consistent set: much of the response runs through "
            "the <b>direct path</b> (patch embedding → output), which LAWFUL by design does not patch.</p>")
    eap, pr = explore / "eap_v3_npac.json", explore / "prune_v3_npac.json"
    if eap.exists() or pr.exists():
        parts.append("<h2 id='circuits'>Circuits (Vi-CD)</h2>")
    if eap.exists():
        a = json.loads(eap.read_text())
        parts.append("<h3>Edge attribution (EAP): the North Pacific jet-level V tendency</h3>"
                     f"<p>Area-mean 6-h V_3 change within 2000 km of 45°N, 180°; base 2010-01-15, counterfactual "
                     f"2010-01-20 (metric {a['clean']:.3f} vs {a['corrupt']:.3f}).</p>"
                     + _table(["edge", "metric change if this edge alone is patched"],
                              [[f"<code>{html.escape(e)}</code>", f"{s:+.3f}"] for e, s in list(a["scores"].items())[:10]]))
    if pr.exists():
        c = json.loads(pr.read_text())
        by = Counter(e.split("->")[0].split(".")[0] for e in c["edges"])
        parts.append(f"<h3>Greedy pruning</h3><p><b>{c['n_edges']} of {c['of']}</b> edges recover "
                     f"<b>{c['recovery']:.0%}</b> of the gap (τ = {c.get('tau')}). Edges by sender: "
                     + ", ".join(f"<code>{k}</code> {n}" for k, n in by.most_common(10)) + ".</p>")
    return "".join(parts)


def rossby_versions(out: Path) -> str:
    """Rossby-wave behaviour across every version with an exploration run (same probe, same states)."""
    tmp = out / "_rossby_tmp"
    tmp.mkdir(exist_ok=True)
    rows, fig_series = [], []
    for i, v in enumerate(VERSIONS):
        ex = RUNS / v["run"] / "explore-final"
        if v["name"] == "v1" and not (ex / "rossby_phi45.json").exists():
            continue
        r = fig_rossby(ex, tmp / f"{v['name']}.png")
        if not r:
            continue
        by = {x["lat"]: x for x in r}
        cells = [v["name"]]
        for lat in ("30°N", "45°N", "60°N", "45°S"):
            x = by.get(lat)
            cells.append(f"{x['corr']:+.2f} / {x['ratio']:.1f}×" if x else "–")
        srch = ex / "rossby_search_corr_phi45.json"
        found = "–"
        if srch.exists():
            d = json.loads(srch.read_text())
            hits = [c for c in (d.get("search") or {}).get("circuits", []) if isinstance(c, dict)]
            def blocks(c):
                comp = c.get("components") or c.get("groups") or []
                ids = sorted({int(x.split(".")[0][1:]) for x in comp if x[:1] in "am" and x.split(".")[0][1:].isdigit()})
                return "blocks " + ", ".join(map(str, ids)) if ids else f"{len(comp)} components"
            found = "; ".join(f"τ {c.get('tau')}: {blocks(c)}" for c in hits) or "none τ-consistent"
        cells.append(found)
        rows.append(cells)
        m45 = ex / "rossby_phi45.json"
        if m45.exists():
            m = json.loads(m45.read_text())["measure"]
            fig_series.append((v["name"], [q["alpha"] for q in m["rows"]], [q["model"] for q in m["rows"]],
                               [q["law"] for q in m["rows"]], PALETTE[i % len(PALETTE)]))
    shutil.rmtree(tmp, ignore_errors=True)
    if len(rows) < 2:
        return ""
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURF)
    for name, s_, f, g, c in fig_series:
        ax.plot(s_, f, "-o", color=c, lw=1.8, ms=3.5, label=name)
    if fig_series:
        ax.plot(fig_series[0][1], fig_series[0][3], "k--", lw=1.5, label="barotropic theory")
    _style(ax, "45°N: intrinsic frequency of a planted wave, every version")
    ax.set_xlabel("zonal wavenumber s", color=INK2)
    ax.set_ylabel("ω − kŪ (rad/day)", color=INK2)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.savefig(out / "rossby_versions.png"); plt.close(fig)
    return ("<h3 id='rossby-versions'>Rossby waves across versions</h3>"
            "<p>The same probe (8 states of 2010, wavenumbers 2–14, levels 2–4) on every version that has been explored. "
            "Each cell: correlation of the model's dispersion with barotropic theory across wavenumbers / fitted magnitude "
            "ratio (model ÷ theory; 1 is right). Stochastic models are probed with their noise held at one fixed draw.</p>"
            + _img("rossby_versions.png", "Rossby dispersion at 45N across versions")
            + _table(["version", "30°N", "45°N", "60°N", "45°S", "LAWFUL block circuit at 45°N (correlation score)"], rows))


def architecture_section(out: Path) -> str:
    j = out / "arch" / "arch.json"
    up = json.loads(j.read_text()).get("upstream", []) if j.exists() else []
    if up:
        head = up[0]["name"]
        img = f"arch/att_npac_{head}.png"
        cap = (f"Head {head}, attending from the North Pacific (orange dot): on average it reads "
               f"{-up[0]['dx']:.0f}° upstream of a midlatitude token.")
    else:
        img, cap = "arch/att_npac_L0H0.png", "Head L0H0, attending from the North Pacific (orange dot)."
    return f"""<h2 id="architecture">How we dissect the ViT</h2>
<figure style="margin:0 0 12px"><a href="architecture.html"><img class="fig" src="arch/schematic.svg" alt="architecture schematic: patches, embedding, 8 pre-LayerNorm transformer blocks, decoder; and the inside of one block"></a>
<figcaption class="muted">The architecture: one 6-hour step with tensor shapes (top), and the inside of each of the 8 blocks with the hook points
the tools use (bottom).</figcaption></figure>
<div class="card split2">
<div><p style="margin-top:0">A deliberately plain Vision Transformer. The state is cut into <b>4°×4° patches</b> (4050 tokens); a
<b>learned position embedding</b> is its only sense of place; <b>8 pre-LayerNorm blocks</b> of 6 global attention heads and an MLP
write into one <b>residual stream</b>; a linear decoder turns it back into the 6-hour <b>tendency</b>. Carrying the state forward and the
conservation corrections happen outside the network.</p>
<p>The <a href="architecture.html"><b>architecture explorer</b></a> draws this with the model's own internals: the patches, what the position
embedding learned, where each of the 48 heads looks from three points on the globe, what every block would predict (the logit lens),
and what each design choice lets us dissect, with each hackathon tool attached where it cuts in.</p>
<p><a href="architecture.html" class="tag released" style="font-size:.95rem;padding:4px 12px">Open the architecture explorer →</a></p></div>
<div><a href="architecture.html"><img class="fig" src="{img}" alt="{cap}"></a>
<div class="muted">{cap}</div>
<a href="architecture.html"><img class="fig" src="arch/pos_pc1.png" alt="first principal component of the learned position embedding"></a>
<div class="muted">The learned position embedding's leading component: latitude, discovered from data.</div></div>
</div>"""


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summ = [version_page(v, out) for v in VERSIONS]
    evaluated = [(s["v"]["name"], s["eval"], PALETTE[i % len(PALETTE)], "-" if i < len(PALETTE) else (0, (4, 2)))
                 for i, s in enumerate(summ) if s["eval"]]
    has_cmp = fig_acc(evaluated, out / "versions_acc.png")
    released = [s for s in summ if s["status"] == "released"]
    # the default model (what pg.load() and the notebooks use), not the newest release
    current = next((x for x in summ if x["v"]["name"] == DEFAULT_MODEL), released[-1] if released else summ[0])
    # analyses (Rossby, circuits) from the newest version that has an exploration run
    analysed = current["v"] if (RUNS / current["v"]["run"] / "explore-final").exists() else \
        next((x["v"] for x in reversed(summ) if (RUNS / x["v"]["run"] / "explore-final").exists()), current["v"])
    m = current["eval"] or {}
    kpis = ""
    if m:
        sk = m["skill"]
        kpis = (f'<div class="grid"><div class="card"><div class="muted">default model (<code>pg.load()</code>)</div><div class="kpi">'
                f'{current["v"]["name"]}</div><div class="muted">{html.escape(current["v"]["title"])}</div></div>'
                f'<div class="card"><div class="muted">5-day anomaly correlation, U_3</div><div class="kpi">'
                f'{sk["U_3"]["5d"]["acc"]:.2f}</div><div class="muted">persistence {sk["U_3"]["5d"]["acc_persistence"]:.2f}</div></div>'
                f'<div class="card"><div class="muted">free run on 2010</div><div class="kpi">{m["days_run"]:.0f} days</div>'
                f'<div class="muted">{"then drifts" if m.get("blowup") else "stable"}</div></div></div>')
    nbs = notebooks_html(out)
    nb_list = "".join(f'<li><a href="{h}">{s}</a>: {html.escape(NB_TITLES.get(s, ""))}</li>' for s, h in nbs)
    body = f"""<nav class="muted"><a href="#architecture">Architecture</a><a href="training.html">Training</a><a href="#versions">Versions</a><a href="#stability">Stability</a><a href="#decade">Decade runs</a><a href="#threed">3D model</a><a href="#rossby">Rossby waves</a><a href="#circuits">Circuits</a><a href="#notebooks">Notebooks</a></nav>
<h1>Inside a ViT emulator of E3SM</h1>
<p class="lede">A close-to-vanilla Vision Transformer that steps E3SMv3 (1°, AMIP) forward 6 hours, built to be taken apart for the
interpretability hackathon: attention heads, circuits (Vi-CD) and physical laws (LAWFUL), with <a href="https://github.com/mahf708/exege">exege</a>. Checkpoints and latent archives are on Hugging Face:
<a href="https://huggingface.co/mahf708/vit-interp-e3sm">mahf708/vit-interp-e3sm</a> and
<a href="https://huggingface.co/datasets/mahf708/vit-interp-e3sm-latents">mahf708/vit-interp-e3sm-latents</a>
(and the 2010 inputs, <a href="https://huggingface.co/datasets/mahf708/vit-interp-e3sm-2010">mahf708/vit-interp-e3sm-2010</a>);
code and notebooks: <a href="https://github.com/mahf708/vit-interp">github.com/mahf708/vit-interp</a>.
4°×4° patches (4050 tokens), 8 pre-LN blocks × 6 heads, all horizontal mixing in attention; everything evaluated on <b>2010</b>, which was never trained on (it did pick the best epoch of each stage: see <a href=\"training.html#m-select\">how checkpoints were chosen</a>).</p>
<p><a href=\"training.html\" class=\"tag released\" style=\"font-size:.95rem;padding:4px 12px\">How the models were trained: curriculum and methods →</a></p>
{kpis}
{architecture_section(out)}
<h2 id="versions">Versions</h2>
<p>Each version has its own page with its recipe, training curves and evaluation. Every version is scored the same way:
a year-long free run on 2010 with true forcing and prescribed SST, and 1–10-day forecasts from 8 starts against persistence and climatology.</p>
{versions_table(summ)}
<p class="muted">* <b>Climate holds:</b> days the year-long free run's global means stay close to E3SM's (near-surface temperature within 2 K,
level-7 water within 10%, surface pressure within 100 Pa, jet-level wind within 3 m/s; one-day means, first breach of any). The <b>free run</b>
column is the older, laxer criterion (global near-surface temperature off by more than 30 K).</p>
{_img("versions_acc.png", "anomaly correlation vs lead, all evaluated versions") if has_cmp else ""}
{stability_section(out)}
{decade_section(out)}
{threed_section(out)}
{analysis_sections(out, analysed)}
{rossby_versions(out)}
<h2 id="notebooks">Hackathon notebooks (executed on {current["v"]["name"]})</h2><ul>{nb_list}</ul>
<p class="muted">Run them on Swing: <code>HACKATHON.md</code> in the vit-interp repository; the exege layers are on the
<code>mahf708/circuits</code> branch of exege.</p>"""
    from build_training import training_page
    training_page(RUNS, out, _page)
    (out / "index.html").write_text(_page("ViT emulator internals", body))
    for p in out.rglob("*"):
        p.chmod(0o755 if p.is_dir() else 0o644)
    out.chmod(0o755)
    print(f"wrote {out}/index.html  ->  {URL}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    build(Path(ap.parse_args().out))
