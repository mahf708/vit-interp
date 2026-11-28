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
        "checkpoint": "vit-p4-d384-l8/release/v8.pt",
        "headline_eval": "eval-long-jitter-filter",
        "stages": [("4–24-step rollouts, v7's recipe, 8 epochs", "long")],
        "evals": [("v8 + jitter + seam filter", "eval-long-jitter-filter")],
        "recipe": [
            "A new architecture (<code>vit_interp/model3d.py</code>): each of the 8 levels (T, U, V, water) and a surface level (PS, TS, forcings) is its own token, 9 × 4050 tokens; in every block 4 heads attend across their level and 2 up and down their column (ViViT's factorized dot-product attention); rotary horizontal positions, periodic in longitude; anti-aliased patch embedding; 14.4 M parameters.",
            "Trained from scratch: 20 one-step epochs (the winner of a 7-way architecture screen, see <a href=\"../index.html#threed\">3D model</a>), then 8 epochs of v7's rollout recipe on Swing's backfill queue.",
            "The best forecaster of any version (5-day ACC 0.93 / 0.90 / 0.88 / 0.85, 10-day 0.63 / 0.59 / 0.56 / 0.45); with jitter + the seam filter all three 10-year runs stay realistic the whole decade, with a slow shared drift in the last two years.",
            "Load: <code>pg.load(\"v8\")</code>, <code>--source v8</code> (Hugging Face <code>v8/v8.pt</code>). Every tool works on it: circuits and LAWFUL (heads 4–5 of each block are the vertical ones), <code>pg.vertical_attention</code>, the head atlas, ablation, attribution, and the latents adapter at a chosen level (<code>level=</code>). Notebook 08 is a tour of its vertical heads.",
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
    "08_vertical_heads_v8": "the 3D model's vertical heads (executed on v8, quick settings)",
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
            ("<b>v8 (3D) + jitter + filter (seed 0)</b>", "long-f3d-rope-jit-f"),
            ("v8 (3D) + jitter + filter (seed 1)", "long-f3d-rope-jit-f-seed1"),
            ("v8 (3D) + jitter + filter (seed 2)", "long-f3d-rope-jit-f-seed2")]
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
                      "All three 10-year runs (3 noise seeds) stay realistic the whole decade, against one of three for "
                      "v7, with a slow, shared drift in the last two years (model top warming 2.5–2.8 K, mid-troposphere "
                      "cooling about 2 K; see <a href='#decade'>Decade runs</a>).")
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


# -- Inside v8: interpretability of the 3D model ------------------------------------------------

LAB = RUNS / "f3d-rope" / "lab"


def _lawful_groups() -> dict | None:
    """lawful_groups.json if the job finished; else the group deviations it already printed to its log."""
    p = LAB / "lawful_groups.json"
    if p.exists():
        return json.loads(p.read_text())
    import ast
    import re
    out: dict = {}
    for log in sorted((Path(__file__).resolve().parents[1] / "logs").glob("*.OU")):
        txt = log.read_text(errors="ignore")
        if "vertical heads {'components'" not in txt:
            continue
        for line in txt.splitlines():
            m = re.match(r"^(-?45) (horizontal heads|vertical heads|MLPs|heads \(both kinds\)) (\{.*\})\s*$", line)
            if m:
                key = "45N" if m.group(1) == "45" else "45S"
                out.setdefault(key, {"groups": {}})["groups"][m.group(2)] = ast.literal_eval(m.group(3))
    return out or None


def _hold_days(d, n):
    t7, ps, u3 = n.index("T_7"), n.index("PS"), n.index("U_3")
    return next((i + 1 for i in range(len(d)) if abs(d[i, t7]) > 2 or abs(d[i, ps]) > 100 or abs(d[i, u3]) > 3), len(d))


def _profile_response_html() -> str:
    p = LAB / "profile_response-v8.json"
    if not p.exists():
        return ""
    r = json.loads(p.read_text())
    g = {k: [x for x in v["gain"]["mean"]] for k, v in r["variants"].items()}
    at = lambda k, d: g[k][int(d * 4)]  # noqa: E731
    rows = [[k if k == "intact" else f"<code>{k}</code> off", *(f"{at(k, d):.2f}" for d in (1, 5, 10, 20, 30, 60))] for k in g]
    return ("<p><b>Is the drift an instability?</b> No. The drift's vertical pattern (warm top, cool middle; taken from the "
            "seed-0 run, year 10 minus year 1, rescaled to be energy-neutral) is planted at ±1 K on 16 states of 2010 and run "
            "60 days in paired rollouts with identical noise (a probe drafted by the GPT reviewer). The table gives how much of "
            "it remains (1 = none removed):</p>"
            + _table(["run", "1 day", "5 days", "10 days", "20 days", "30 days", "60 days"], rows)
            + "<p>v8 damps the pattern with an e-folding time of about three weeks, and no single head changes that by more "
              "than 0.06. The decade drift is therefore a slowly <i>forced</i> pattern (fed, per the table above, by "
              "small-scale noise at the model top), not a growing mode.</p>")


def inside_v8_section(out: Path) -> str:
    if not LAB.exists():
        return ""
    parts = ["<h2 id='inside-v8'>Inside v8: what the vertical heads do</h2>"
             "<p>v8 gives every level its own token, and only heads 4–5 of each block can look up and down a column "
             "(heads 0–3 look across a level). That makes questions answerable that the 2D models could not pose: which "
             "components move information between levels, which read the surface, and whether a physical law "
             "lives in the horizontal or the vertical machinery. All probes use 2010 states (held out) and v8's released "
             "checkpoint, with its noise held at zero; code in <code>scripts/v8_lab.py</code>.</p>"]

    # 1. transmission
    tp = LAB / "transmission.json"
    if tp.exists():
        t = json.loads(tp.read_text())
        variants = [k for k in ("intact", "no vertical heads", "no horizontal heads") if k in t]
        fig, axs = plt.subplots(1, len(variants), figsize=(3.6 * len(variants) + 1.2, 3.8), facecolor=SURF)
        axs = np.atleast_1d(axs)
        from matplotlib.colors import LogNorm
        norm = LogNorm(1e-3, 1.5)
        for ax, k in zip(axs, variants):
            M = np.array(t[k])
            im = ax.imshow(M, cmap="viridis", norm=norm, origin="upper")
            ax.set_title(k, fontsize=10, color=INK)
            ax.set_xticks(range(8)); ax.set_yticks(range(8))
            ax.set_xlabel("responding level", color=INK2, fontsize=9)
            ax.tick_params(labelsize=8, colors=INK2)
        axs[0].set_ylabel("planted level", color=INK2, fontsize=9)
        cb = fig.colorbar(im, ax=axs.tolist(), shrink=0.85, pad=0.02)
        cb.set_label("V tendency response (m/s per 6 h)", fontsize=8.5, color=INK2); cb.ax.tick_params(labelsize=8)
        fig.savefig(out / "v8_transmission.png"); plt.close(fig)

        def off(M):
            M = np.array(M)
            return float(M.sum() - np.trace(M)) / 56

        o = {k: off(t[k]) for k in variants}
        diag = {k: float(np.trace(np.array(t[k]))) / 8 for k in variants}
        rows = [[k, f"{diag[k]:.3f}", f"{o[k]:.4f}", f"{o[k] / o['intact']:.0%}"] for k in variants]
        parts.append(
            "<h3 id='v8-transmission'>Who carries information between levels?</h3>"
            "<p>A wave is planted in the meridional wind of one level (rows); the model takes one step; we measure how "
            "much each level's V tendency changes (columns, band RMS over 4 states). Then the same with all 16 vertical "
            "heads, or all 32 horizontal heads, mean-ablated.</p>"
            + _img("v8_transmission.png", "level-to-level transmission, intact and with head groups removed")
            + _table(["variant", "same-level response", "other-level response (mean)", "other-level, vs intact"], rows)
            + f"<p><b>Removing the 16 vertical heads cuts the coupling between levels by "
              f"{1 - o['no vertical heads'] / o['intact']:.0%}</b> (the same-level response halves too, as it does without the "
              f"horizontal heads); removing "
              f"the 32 horizontal heads instead leaves {o.get('no horizontal heads', float('nan')) / o['intact']:.0%} of it. "
              "Horizontal heads still matter for vertical coupling because what a vertical head reads at another level "
              "was first shaped by horizontal mixing there. The remaining few percent enter outside attention (not yet "
              "traced, but the likely route is the 3×3 output convolution, which mixes all 44 output channels, levels "
              "included; in absolute terms it is largest from level 6 into level 7). Inside the network, exchange between "
              "levels is attention's job, done by a small, named set of heads.</p>")

    # 2. surface-reading heads
    vp = LAB / "vertical_reading.json"
    if vp.exists():
        v = json.loads(vp.read_text())
        cols = ["Tibetan Plateau", "land", "open ocean", "sea ice"]
        by: dict = {}
        for r in v:
            by.setdefault((r["layer"], r["head"]), {})[r["columns"]] = r
        heads = sorted(by, key=lambda k: -by[k].get("Tibetan Plateau", {}).get("surface_weight", 0))
        rows = []
        for (l, h) in heads[:8]:
            rows.append([f"<code>L{l}H{h}</code>"] + [f"{by[(l, h)][c]['surface_weight']:.2f}" if c in by[(l, h)] else "–" for c in cols]
                        + [f"{by[(l, h)].get('open ocean', {}).get('mean_read_offset', float('nan')):+.1f}"])
        # small multiples: mean level->level attention of every vertical head over open ocean
        pats = [(l, h, np.array(by[(l, h)]["open ocean"]["pattern"])) for (l, h) in sorted(by) if "open ocean" in by[(l, h)]]
        if pats:
            n = len(pats)
            fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(1.3 * ((n + 1) // 2) + 0.5, 3.4), facecolor=SURF, layout="constrained")
            for ax, (l, h, P) in zip(axs.flat, pats):
                ax.imshow(P, cmap="magma", vmin=0, vmax=max(0.5, float(P.max())))
                ax.set_title(f"L{l}H{h}", fontsize=8.5, color=INK, pad=2)
                ax.set_xticks([]); ax.set_yticks([])
            for ax in list(axs.flat)[n:]:
                ax.axis("off")
            fig.supxlabel("key level (0 = model top … 7 = lowest, 8 = surface)", fontsize=8.5, color=INK2)
            fig.supylabel("query level", fontsize=8.5, color=INK2)
            fig.savefig(out / "v8_vertical_patterns.png"); plt.close(fig)
        parts.append(
            "<h3 id='v8-surface-heads'>Heads that read the surface</h3>"
            "<p>For each vertical head: the attention weight that the 8 atmospheric tokens of a column put on the surface "
            "token (PS, TS and the forcings), averaged over columns of each kind, and the mean offset in levels from query "
            "to key over open ocean (+ = reads lower, toward the surface). Uniform attention would put 1/9 = 0.11 on the surface.</p>"
            + _table(["head", "Tibetan Plateau", "land", "open ocean", "sea ice", "mean read offset, open ocean (levels)"], rows)
            + "<p>Head 5 of blocks 0, 1 and 3 is the surface reader: over the Tibetan Plateau, where the surface sits "
              "inside the mid-troposphere, its atmospheric tokens put 0.45–0.60 of their attention on the surface token, "
              "against 0.2–0.4 elsewhere (0.11 would be uniform). Block 0's head 4, by contrast, spreads its attention widely "
              "and on average reads one level below the query.</p>"
            + (_img("v8_vertical_patterns.png", "mean attention pattern of every vertical head, open ocean") if pats else "")
            + "<p class='muted'>Each panel: rows are query levels, columns key levels (top-left = model top, last "
              "row/column = surface), averaged over open-ocean columns of 4 states.</p>")

    # 3. vertical-head atlas
    ap = LAB / "atlas.json"
    if ap.exists():
        a = [r for r in json.loads(ap.read_text()) if r["kind"] == "v"]
        bands = ["tropics", "NH mid", "SH mid"]
        by = {}
        for r in a:
            by.setdefault((r["layer"], r["head"]), {})[r["band"]] = r
        rows = []
        for (l, h) in sorted(by):
            al = by[(l, h)].get("all", {})
            rows.append([f"<code>L{l}H{h}</code>", f"{al.get('self', float('nan')):.2f}", f"{al.get('entropy', float('nan')):.2f}"]
                        + [f"{by[(l, h)][b]['dlev']:+.2f}" if b in by[(l, h)] else "–" for b in bands])
        parts.append(
            "<h3 id='v8-atlas'>Vertical-head atlas</h3>"
            "<p>Per vertical head, over 2010 states: attention on its own level, entropy of the level distribution "
            "(ln 9 = 2.20 is uniform), and the mean signed level offset of what it reads in three latitude bands "
            "(positive = reads levels below, toward the surface).</p>"
            + _table(["head", "self-attention", "entropy", "offset, tropics", "NH mid-latitudes", "SH mid-latitudes"], rows))

    # 4. NP circuit across versions
    crow = []
    for v in VERSIONS:
        p = RUNS / v["run"] / "explore-final" / "prune_v3_npac.json"
        if not p.exists():
            continue
        c = json.loads(p.read_text())
        snd = Counter(e.split("->")[0] for e in c["edges"])
        blk = Counter(e.split("->")[0].split(".")[0] for e in c["edges"])
        is3d = v["name"] == "v8"
        vert = sum(n for s, n in snd.items() if s.endswith((".h4", ".h5"))) if is3d else None
        heads = sum(n for s, n in snd.items() if ".h" in s)
        crow.append([v["name"], f"{c['n_edges']} of {c['of']}", f"{c['recovery']:.2f}",
                     ", ".join(f"<code>{k}</code> {n}" for k, n in blk.most_common(4)),
                     f"{vert} of {heads} head edges" if is3d else "– (no vertical heads)"])
    if crow:
        parts.append(
            "<h3 id='v8-circuit'>The North Pacific circuit, v2 to v8</h3>"
            "<p>The same Vi-CD problem on every version: the 6-h V_3 change within 2000 km of 45°N, 180°, base "
            "2010-01-15 against 2010-01-20, greedy pruning at τ = 0.02. For v8 the metric reads the level-3 tokens.</p>"
            + _table(["version", "edges kept", "recovery", "busiest senders", "from vertical heads"], crow)
            + "<p>The 2D models route this jet-level response through block 4 (17 edges in each of v2–v4) and the embedding. "
              "v8 needs fewer edges, spreads them over blocks 4–6 and the embedding, and a minority run through vertical "
              "heads: the jet-level wind is a horizontal problem, and v8 treats it as one. Recovery above 1 (the 2D models) "
              "means the kept edges overshoot the full model's change; v8's circuit recovers 87%.</p>")

    # 5. LAWFUL by head type
    lg = _lawful_groups()
    if lg:
        rows = []
        for band in ("45N", "45S"):
            e = lg.get(band)
            if not e:
                continue
            G = e["groups"]
            m0 = next(iter(G.values()))["M0"]
            for name in ("horizontal heads", "vertical heads", "MLPs"):
                if name in G:
                    g = G[name]
                    rows.append([band, name, len(g["components"]) if isinstance(g.get("components"), list) else "",
                                 f"{m0:.2f}", f"{g['M_suf']:+.2f}", f"{g['M_nec']:+.2f}"])
        srows = []
        for band in ("45N", "45S"):
            prev: list = []
            for c in (lg.get(band) or {}).get("search", []):
                new = [u for u in c["units"] if u not in prev]
                srows.append([band, c["tau"], len(c["units"]), ", ".join(new)])
                prev = list(c["units"])
        parts.append(
            "<h3 id='v8-lawful'>Where does the Rossby law live? (LAWFUL by head type)</h3>"
            "<p>The barotropic Rossby probe of <a href='#rossby-versions'>Rossby waves across versions</a>, scored as the "
            "correlation between the model's dispersion and theory (M; 1 is perfect). <b>Alone</b>: only this group "
            "computes, the rest is mean-ablated. <b>Without</b>: this group is mean-ablated.</p>"
            + _table(["latitude", "group", "components", "M, full model", "M, group alone", "M, without group"], rows)
            + "<p>The law needs the MLPs (without them the correlation falls to −0.06 / −0.13) and the horizontal heads "
              "(without them, 0.35–0.36). It hardly needs the vertical heads: without all 16 of them the correlation is "
              "still 0.54 at 45°N and 0.84 at 45°S, against 0.83 and 0.92 intact. A barotropic wave is a "
              "within-level phenomenon, and v8 computes it mostly within levels. No head group reproduces the law on its "
              "own.</p>"
            + ("<p><b>And v8 has a law circuit.</b> LAWFUL's search over 24 groups (per block: its horizontal heads, its "
               "vertical heads, its MLP) finds nested τ-consistent circuits at both latitudes. For the 2D models the search was over whole blocks: v2 and "
               "v4 had no circuit and v3's was blocks 5–7 entire (<a href='#rossby-versions'>Rossby waves across versions</a>). Each row adds the groups listed to the row "
               "above:</p>" + _table(["latitude", "τ", "groups", "groups added"], srows)
               + "<p>The core, at τ = 0.3, is <b>block 7's horizontal heads plus the MLPs of the early blocks and blocks 6–7</b>. "
                 "Vertical heads enter only at τ = 0.9 at 45°N, and never at 45°S. Splitting heads by axis gives the search finer, physically named units "
                 "to choose from; in the 2D models every head mixes horizontally and vertically at once.</p>" if srows else ""))

    # 6. surface heating circuits
    sp = LAB / "surface_circuits.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        rows = []
        for region, d in s.items():
            for f, x in d.items():
                e = x["edges"]
                hs = [q for q in e if ".h" in q.split("->")[0]]
                vs = [q for q in hs if q.split("->")[0].endswith((".h4", ".h5"))]
                top = ", ".join(f"<code>{html.escape(k)}</code>" for k, _ in x["top_eap"][:3])
                rows.append([region, f, f"{x['corrupt'] - x['clean']:+.3f} K", x["n_edges"], f"{x['recovery']:.2f}",
                             f"{len(vs)} of {len(hs)}", top])
        parts.append(
            "<h3 id='v8-surface'>A circuit for surface heating</h3>"
            "<p>The skin temperature is raised 5 K within 8° of a point (Tibetan Plateau in June, Sahara in July) and the "
            "model takes one step. The metric is the mean temperature change at one level within 1000 km. Edges are "
            "scored by exact activation patching and pruned at τ = 0.02.</p>"
            + _table(["case", "level", "one-step response", "edges kept", "recovery", "head edges from vertical heads",
                      "largest single edges"], rows)
            + "<p>The lowest level (T_7) warms 0.2–0.4 K in one step, through about 100 edges that start at the embedding "
              "and the surface-reading head 5 of blocks 1 and 3. <b>About 70% of the head edges come from vertical heads</b>, "
              "which are a third of all heads: surface heating climbs the column through the vertical heads, then the MLPs "
              "turn it into a temperature tendency. Higher up, the one-step response is tiny (T_3: "
              "0.004 K over Tibet, −0.01 K over the Sahara), and the 233–389-edge circuits pruned for it fit noise: "
              "<b>treat them as no circuit</b>. In six hours, heat reaches one or two levels above the surface.</p>")

    # 7. error components and the intervention decade runs
    ep = LAB / "error_components.json"
    if ep.exists():
        ec = json.loads(ep.read_text())
        b0 = np.array(ec["baseline_bias"])
        comps = ec["components"]
        top = sorted(comps, key=lambda r: r["top_bias_change"])[:6]
        mid = sorted(comps, key=lambda r: -r["mid_bias_change"])[:6]
        erows = [[f"<code>{r['component']}</code>", {"h": "horizontal", "v": "vertical"}.get(r["kind"], r["kind"]),
                  f"{r['top_bias_change'] * 4 * 365:+.1f}", f"{r['mid_bias_change'] * 4 * 365:+.1f}", f"{r['rmse']:.3f}"]
                 for r in {x["component"]: x for x in top + mid}.values()]
        runs = [("seed 0", "long-f3d-rope-jit-f", PALETTE[0], "-"), ("seed 1", "long-f3d-rope-jit-f-seed1", PALETTE[0], "--"),
                ("seed 2", "long-f3d-rope-jit-f-seed2", PALETTE[0], ":"),
                ("L4H2 off (control)", "long-v8-abl-L4H2-control", PALETTE[2], "-"),
                ("L7H3 off", "long-v8-abl-L7H3", PALETTE[1], "-"), ("L7H3 at half", "long-v8-abl-L7H3-half", PALETTE[3], "-"),
                ("L6H2 off", "long-v8-abl-L6H2", PALETTE[5], "-")]
        drows, series = [], []
        for label, tag, col, ls in runs:
            f = DRIFT / f"{tag}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            n = list(z["names"]); d = z["g_e"] - z["g_t"]
            yrs = [d[i * 365:(i + 1) * 365] for i in range(int(np.ceil(len(d) / 365)))]
            t0, t3 = n.index("T_0"), n.index("T_3")
            drows.append([label, f"{_hold_days(d, n) / 365:.1f} y", f"{yrs[0][:, t0].mean():+.1f} / {yrs[0][:, t3].mean():+.1f}",
                          f"{yrs[-1][:, t0].mean():+.1f} / {yrs[-1][:, t3].mean():+.1f}",
                          f"{np.mean((abs(d[:, n.index('T_7')]) > 2) | (abs(d[:, n.index('PS')]) > 100) | (abs(d[:, n.index('U_3')]) > 3)):.1%}",
                          "none" if z["blow"].item() in ("null", None) else "yes"])
            series.append((label, np.arange(1, len(d) + 1) / 365, d[:, t0], d[:, t3], col, ls))
        if series:
            fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), facecolor=SURF, sharex=True)
            k = 30  # 30-day running mean
            sm = lambda y: np.convolve(y, np.ones(k) / k, mode="valid")  # noqa: E731
            for lab, tt, y0, y3, col, ls in series:
                axs[0].plot(tt[k - 1:], sm(y0), color=col, ls=ls, lw=1.3, label=lab)
                axs[1].plot(tt[k - 1:], sm(y3), color=col, ls=ls, lw=1.3, label=lab)
            for ax, ttl in zip(axs, ("model top (T_0)", "mid-troposphere (T_3)")):
                _style(ax, f"global-mean temperature error, {ttl} (K)")
                ax.axhline(0, color="#888", lw=0.8); ax.set_xlabel("years from 2000-01-01", color=INK2)
            axs[1].legend(frameon=False, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, fontsize=8.5)
            fig.savefig(out / "v8_interventions.png"); plt.close(fig)
        parts.append(
            "<h3 id='v8-drift'>Which components write the drift? One step versus ten years</h3>"
            "<p>Every v8 decade run drifts the same way in its last two years: the model top warms, the "
            "mid-troposphere cools (<a href='#decade'>Decade runs</a>). Per 6-h step over 24 states of 2010, v8's "
            f"temperature tendency has a small global cold bias at every level ({b0[0] * 4 * 365:+.0f} K/yr at the top, "
            f"{b0[3] * 4 * 365:+.0f} K/yr at level 3 if it accumulated unchecked; the correctors and the circulation take "
            "most of it back). Mean-ablating each of the 56 components one at a time shows which ones shift that bias. "
            "Columns are the change in the bias, in K per year of steps.</p>"
            + _table(["component", "kind", "Δ top bias (K/yr)", "Δ level-3 bias (K/yr)", "1-step T error (K)"], erows)
            + "<p>Three of these were then switched off (or halved) <b>at every step of a 10-year free run</b>, against a "
              "control head (L4H2) that hardly moves the bias, and the three noise seeds of the intact model.</p>"
            + (_img("v8_interventions.png", "decade runs with heads ablated, global T error at the top and mid-troposphere") if series else "")
            + _table(["run", "climate holds*", "year 1 error T_0 / T_3 (K)", "year 10 error T_0 / T_3 (K)", "days outside*", "blow-up"], drows)
            + "<p><b>One-step attribution does not predict multi-year drift.</b> First, the control matters: switching off "
              "L4H2, which hardly moves the one-step bias, also ends the 'climate holds' count after 2.2 years, just like L7H3 "
              "off (2.3 y) and L6H2 off (2.2 y). In all three, the first breach is a single day of jet-level wind more than "
              "3 m/s off, and at most 0.3% of days are ever outside the bounds: any full head ablation makes the run "
              "slightly noisier, not unstable. Against that control: removing L7H3, which at one step warms the model top, "
              "gives a run that is colder at every level from year 1 and ends with the worst mid-troposphere (−3.9 K in year "
              "10). Removing L6H2 leaves the model top about 6 K too cold from year 3 on. Halving L7H3 instead keeps the run "
              "inside the bounds for all 10 years. Yet every run, control and interventions alike, ends with the same "
              "late-decade signature: the model top warming by 2–3 K over the last two years while T_3 cools. "
              "So the drift is not written by any single head (and, below, not by the 2008–2009 forcing either).</p>"
            + "<p class='muted'>* climate holds: as in <a href='#decade'>Decade runs</a> (global T_7 within 2 K, PS within "
              "100 Pa, U_3 within 3 m/s, daily). Starting states: " + f"{ec['starts'][0]} … {ec['starts'][-1]}, every 15 days.</p>")

    # 7b. what drives the late drift
    arms = [("v8, jitter + filter (3 seeds)", ["long-f3d-rope-jit-f", "long-f3d-rope-jit-f-seed1", "long-f3d-rope-jit-f-seed2"], 2000),
            ("same, year-boundary step fixed", ["long-v8-jf-c16-fixyr"], 2000),
            ("fresh start in 2006", ["long-v8-jf-from2006"], 2006),
            ("fresh start in 2008", ["long-v8-jf-from2008"], 2008),
            ("jitter order not locked to the UTC hour (2 seeds)", ["long-v8-jf-c16d", "long-v8-jf-c16d-s1"], 2000),
            ("random jitter", ["long-v8-jf-rand"], 2000),
            ("energy corrector off", ["long-v8-jf-noenergy"], 2000),
            ("top 2 levels: increments × 0.9", ["long-v8-jf-damptop10"], 2000),
            ("top 2 levels: 25% of the small-scale increment removed", ["long-v8-jf-smalltop25"], 2000),
            ("<b>top 2 levels: 50% of the small-scale increment removed</b>",
             ["long-v8-jf-smalltop50", "long-v8-jf-smalltop50-s1", "long-v8-jf-smalltop50-s2"], 2000)]
    arows = []
    for label, tags, y0 in arms:
        fs = [DRIFT / f"{t}.npz" for t in tags if (DRIFT / f"{t}.npz").exists()]
        if not fs:
            continue
        vals = []
        for f in fs:
            z = np.load(f)
            n = list(z["names"]); d = z["g_e"] - z["g_t"]
            last = d[(2009 - y0) * 365:(2010 - y0) * 365]
            r = z["dd_e"] / z["dd_t"]
            vals.append([last[:, n.index(c)].mean() for c in ("T_0", "T_1", "T_3", "T_7")]
                        + [min(r[n.index("T_0")], r[n.index("T_1")]), max(r[n.index("T_0")], r[n.index("T_1")])])
        v = np.array(vals)
        rng = lambda j: (f"{v[:, j].mean():+.1f}" if len(v) == 1  # noqa: E731
                         else f"{v[:, j].min():+.1f} to {v[:, j].max():+.1f}")
        arows.append([label + (f" <span class='muted'>({len(fs)} run{'s' if len(fs) > 1 else ''})</span>" if len(tags) > 1 else ""),
                      rng(0), rng(1), rng(2), rng(3), f"{v[:, 4].min():.2f}–{v[:, 5].max():.2f}"])
    if arows:
        parts.append(
            "<h3 id='v8-drift-cause'>What drives the late drift?</h3>"
            "<p>Ten-year free runs (2000–2009, as in <a href='#decade'>Decade runs</a>), each changing one thing. Columns: "
            "global-mean error in <b>2009</b> (the last year) at the model top (T_0, T_1), mid-troposphere (T_3) and lowest "
            "level (T_7), and the day-to-day variability of the top two levels' temperature relative to E3SM's (1 = right; "
            "damping that only hides the drift would push it below 1).</p>"
            + _table(["run", "T_0 (K)", "T_1 (K)", "T_3 (K)", "T_7 (K)", "top variability ÷ E3SM"], arows)
            + "<ul><li><b>Not the forcing.</b> Runs started fresh in 2006 or 2008 are clean in 2009, the year the decade runs "
              "are worst: the drift builds up over about 8 years of free running.</li>"
              "<li><b>Not the jitter order.</b> Unlocking the jitter from the time of day (or randomizing it) makes the late "
              "warming worse, not better.</li>"
              "<li><b>Not the energy corrector.</b> Without it, the raw network is 1–2 K too cold through the column from year 1 "
              "(the corrector was hiding a cold bias), and the model top still warms by about 2 K at the end.</li>"
              "<li><b>Small scales at the top.</b> Damping all of the top two levels' increments does little; removing half of "
              "just their small-scale part (what a 30-wavenumber, 2-pass smoother takes out) removes most of the top "
              "warming and keeps E3SM's variability. The drift is fed by small-scale noise accumulating at the model top, where "
              "the atmosphere is thin and dynamical cores usually add extra damping (a sponge layer).</li></ul>"
            + _profile_response_html()
            + "<p class='muted'>The damping arms run outside the network, like the seam filter "
              "(<code>scripts/long_run.py --damp-small-top</code>). The first v9 fine-tune targets the raw cold bias; the top-level small scales are next.</p>")

    # 8. SAEs
    arch = ROOT / "archives"
    srows = []
    for name, d, lay in (("v2", "v2-2010-jan", (3, 5, 8)), ("v8 level 3", "v8-2010-jan-level3", (3, 6)),
                         ("v8 surface", "v8-2010-jan-level8", (3, 6))):
        for L in lay:
            p = arch / d / "bases" / f"sae-topk32-layer{L}.npz"
            if not p.exists():
                continue
            rec = json.loads(str(np.load(p, allow_pickle=True)["record"]))
            m = rec["meta"]["metrics"]
            srows.append([name, L, rec["meta"]["training"]["n_features"], f"{m['explained_variance']:.1%}",
                          f"{m['mean_active_features']:.0f}", f"{m['dead_fraction']:.1%}"])
    if srows:
        parts.append(
            "<h3 id='v8-sae'>Sparse autoencoders per level</h3>"
            "<p>Top-k sparse autoencoders (1024 features, k = 32) on the residual stream of January 2010, one token "
            "kind at a time: v8's level-3 tokens and its surface tokens, beside v2's whole-column tokens. Archives and "
            "dictionaries are in <code>archives/</code> (exege format).</p>"
            + _table(["tokens", "layer", "features", "variance explained", "active per token", "dead"], srows)
            + "<p>v8's surface tokens are the easiest to compress (96% at layer 3, 93% at layer 6); its level-3 tokens sit "
              "close to v2's whole-column tokens (85–91% against 88–93%). Both v8 token kinds lose some variance between "
              "layers 3 and 6, as the vertical heads mix levels into them.</p>")
    return "".join(parts)


# -- Emergent dynamics: physics the emulator was never told about ------------------------------

EMERGE = RUNS / "f3d-rope" / "lab" / "emergent"


def _ej(name):
    p = EMERGE / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _bc_growth(v):
    a = np.sqrt(np.array(v["E"]) / v["E"][0])
    i = int(np.argmin(a[:12]))
    return float(a[i]), i / 4.0, float(np.log(a[-1] / a[i]) / ((len(a) - 1 - i) / 4.0))


def _hyd_split(r):
    M = np.abs(np.array(r["model_V"]["re"]) + 1j * np.array(r["model_V"]["im"]))
    return float(np.triu(M, 1).sum()), float(np.tril(M, -1).sum()), float(np.trace(M))


def emergent_section(out: Path) -> str:
    if not EMERGE.exists():
        return ""
    parts = ["<h2 id='emergent'>Emergent dynamics: physics the emulator was never told about</h2>"
             "<p>The emulator is only ever trained to predict E3SM's next 6-hour state. Does it carry the atmosphere's "
             "dynamics inside it: waves, balances, responses to heating? Each probe plants an idealized perturbation on "
             "8 real 2010 states (as a ± pair, so the linear response is isolated), steps the raw network with its noise "
             "held at zero, and compares the response with a theoretical prediction that the model never saw. Code: "
             "<code>scripts/emergent.py</code>; probe ideas were developed with two AI reviewers (GPT-6.2 and Gemini 4) "
             "and every one was run and checked here.</p>"]
    rows = []
    adj = {c: _ej(f"adjust-{c}") for c in ("v8", "v2")}
    gill = {k: _ej(f"gill-{k}") for k in ("v8", "v2", "v8-lat45", "v8-novert")}
    hyd = {k: _ej(f"hydrostatic-{k}") for k in ("v8", "v2", "v8-novert")}
    ine = {k: _ej(f"inertial-{k}") for k in ("v8", "v2", "v8-novert")}
    lam = {c: _ej(f"lamb-{c}") for c in ("v8", "v2")}
    if gill["v8"]:
        g = gill["v8"]
        rows.append(["<b>Matsuno–Gill response</b> to steady equatorial heating (6 days)",
                     "easterlies east, westerlies west, twin cyclones west, upper flow reversed",
                     f"<b>yes</b>: U<sub>7</sub> east {g['u7_east_mean']:+.2f} / west {g['u7_west_mean']:+.2f} m/s, "
                     f"upper–lower correlation {g['corr_U2_U7']:+.2f}, cyclones N and S", "<b>yes</b>"])
    if adj["v8"]:
        r = {x["lat"]: x["step1"] for x in adj["v8"]["rows"]}
        rows.append(["<b>Geostrophic adjustment</b> of a pressure bump",
                     "balanced part kept grows with |f|; anticyclonic wind, sign flips across the equator",
                     f"<b>yes</b>: kept {r[0]['ps_kept']:.2f} at 0°, {r[45]['ps_kept']:.2f} at 45°N; wind "
                     f"{r[45]['vt_ccw']['7']:+.2f} (45°N) / {r[-45]['vt_ccw']['7']:+.2f} m/s (45°S)", "yes"])
    if hyd["v8"]:
        up, lo, dg = _hyd_split(hyd["v8"])
        rows.append(["<b>Hydrostatic integration</b>: warm one level, which levels' winds respond?",
                     "at and above the warming only, weighted by layer thickness",
                     f"<b>no</b>: the neighbouring levels respond, as much below as above (summed {up:.1f} vs {lo:.1f} m/s, "
                     f"{dg:.1f} at the warmed level), ~{hyd['v8']['slope_V']:.0%} of the inviscid size", "no"])
    if ine["v8"]:
        r = {x["lat"]: x for x in ine["v8"]["rows"]}
        rows.append(["<b>Inertial rotation</b> of a zonal wind band",
                     "turned by −sin(fΔt): about ∓0.8 at ±45°",
                     f"<b>no</b>: dV/A {r[45]['model']['3']['dV']:+.2f} at 45°N", "no"])
    if lam["v8"]:
        s1 = lam["v8"]["steps"]["1"]
        rows.append(["<b>Lamb pulse</b> (Hunga Tonga-like pressure bump)", "a ring ~6700 km out after 6 h (310 m/s)",
                     f"<b>no</b>: the bump decays in place ({s1['center_Pa']:.0f} of 200 Pa left), no ring", "no"])
    bal = {c: _ej(f"balance-{c}") for c in ("v8", "v2")}
    kel = {k: _ej(f"kelvin-{k}") for k in ("v8", "v2", "v8-novert")}
    bar = {k: _ej(f"baroclinic-{k}") for k in ("v8", "v2", "v8-novert")}
    inec = _ej("inertial-v8-clampvert")
    if bal["v8"]:
        b = bal["v8"]["rows"]
        rows.insert(1, ["<b>Balanced vs unbalanced</b> vortex, same size (600–2400 km)",
                        "the balanced vortex survives; the bare pressure bump adjusts away",
                        f"<b>yes</b>: pressure kept {min(r['balanced']['ps_kept'] for r in b):.2f}–{max(r['balanced']['ps_kept'] for r in b):.2f} "
                        f"balanced vs {min(r['ps_only']['ps_kept'] for r in b):.2f}–{max(r['ps_only']['ps_kept'] for r in b):.2f} bare", "yes"])
    if kel["v8"]:
        k8 = kel["v8"]["variants"]
        rows.insert(2, ["<b>Equatorial Kelvin wave</b> (4 days)", "eastward at 15–45 m/s; the wrongly polarized wave does not propagate",
                        f"<b>yes</b>: {k8['kelvin']['phase_speed_m_s']:+.0f} m/s; anti-Kelvin {k8['anti']['phase_speed_m_s']:+.1f} m/s",
                        f"yes ({kel['v2']['variants']['kelvin']['phase_speed_m_s']:+.0f} m/s)" if kel["v2"] else "–"])
    if bar["v8"]:
        b8 = bar["v8"]["variants"]
        rows.insert(3, ["<b>Baroclinic growth</b> (4 days)", "a wave tilted westward with height grows, an eastward-tilted one decays",
                        f"<b>yes</b>: after ~2 days of adjustment the westward-tilted wave grows at "
                        f"{_bc_growth(b8['west'])[2]:.2f}/day, an untilted one tilts westward and grows at "
                        f"{_bc_growth(b8['none'])[2]:.2f}/day (Eady: ~0.4–0.6), the eastward-tilted one at "
                        f"{_bc_growth(b8['east'])[2]:.2f}/day", "yes (weaker)"])
    parts.append(_table(["probe", "theory says", "v8", "v2 the same?"], rows))
    parts.append("<p><b>The pattern:</b> the emulator reproduces the <b>slow, balanced</b> atmosphere (the Gill "
                 "circulation, geostrophic balance, Kelvin waves, baroclinic growth) and damps "
                 "<b>fast, unbalanced</b> motion (Lamb pulses, inertial oscillations) instead of propagating it. That is "
                 "what learning from 6-hourly snapshots of a mostly balanced atmosphere should teach. One thing it does not do "
                 "is integrate a single warmed layer upward as hydrostatics requires. In v8 every multi-level response "
                 "runs through the <b>vertical heads</b>: switch them off and the Gill circulation, the Kelvin wave and "
                 "baroclinic growth all disappear.</p>")

    # Gill figure
    gz = EMERGE / "gill-v8.npz"
    if gz.exists():
        z = np.load(gz)
        lat, lon, m = z["lat"], z["lon"], z["maps"]
        nm = list(z["names"])
        sel = (lat > -40) & (lat < 40)
        lo = ((lon - 180 + 180) % 360 - 180)
        o = np.argsort(lo)
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=SURF, gridspec_kw={"width_ratios": [1.5, 1]})
        ax = axs[0]
        ps = m[nm.index("PS")][sel][:, o]
        vmax = np.abs(ps).max()
        im = ax.pcolormesh(lo[o], lat[sel], ps, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        u, v = m[nm.index("U_7")][sel][:, o], m[nm.index("V_7")][sel][:, o]
        ss = (slice(None, None, 4), slice(None, None, 6))
        ax.quiver(lo[o][ss[1]], lat[sel][ss[0]], u[ss], v[ss], scale=6, width=0.0025, color="#222")
        th = np.exp(-0.5 * (lo[o] / 12.0) ** 2)[None, :] * np.exp(-0.5 * (lat[sel] / 7.0) ** 2)[:, None]
        ax.contour(lo[o], lat[sel], th, levels=[0.5], colors=[PALETTE[1]], linewidths=1.6)
        ax.set_xlim(-120, 120)
        _style(ax, "v8, days 3–6: low-level wind and PS (Pa)")
        ax.set_xlabel("degrees east of the heating (orange: heating)", color=INK2)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015); cb.set_label("PS (Pa)", fontsize=8.5); cb.ax.tick_params(labelsize=8)
        ax = axs[1]
        for i, (k, lab) in enumerate((("v8", "v8"), ("v2", "v2"), ("v8-lat45", "v8, heating at 45°N"),
                                      ("v8-novert", "v8, vertical heads off"))):
            g = gill[k]
            if g:
                d = np.array(g["dlon"]); oo = np.argsort(d)
                ax.plot(d[oo], np.array(g["u7_eq"])[oo], color=PALETTE[i], lw=1.6, label=lab)
        ax.axhline(0, color="#888", lw=0.8); ax.axvline(0, color=PALETTE[1], lw=0.8, ls=":")
        ax.set_xlim(-150, 150)
        _style(ax, "low-level zonal wind along the heating's latitude (m/s)")
        ax.set_xlabel("degrees east of the heating", color=INK2)
        ax.legend(frameon=False, labelcolor=INK, fontsize=8)
        fig.savefig(out / "emergent_gill.png"); plt.close(fig)
        g = gill["v8"]
        parts.append("<h3 id='emergent-gill'>The Matsuno–Gill response</h3>"
                     "<p>A steady heating (0.25 K per 6 h at the peak, in the mid-troposphere, 12° × 7°) is added over the "
                     "equatorial Pacific at every step of 6-day paired rollouts (true forcings, prescribed ocean "
                     "temperature) from 8 starts in winter and summer 2010. Gill (1980) predicts low-level winds "
                     "converging on the heating, easterlies reaching far east (a Kelvin wave) and westerlies to the west "
                     "flanked by twin cyclones (Rossby waves), and upper-level winds reversed against the lower ones.</p>"
                     + _img("emergent_gill.png", "Gill response in v8")
                     + f"<p>v8 does all of it: low-level easterlies east of the heating ({g['u7_east_mean']:+.2f} m/s), "
                       f"westerlies west ({g['u7_west_mean']:+.2f} m/s), cyclones on both sides of the equator to the west, "
                       f"and an upper flow reversed against the lower (correlation {g['corr_U2_U7']:+.2f}). The eastern "
                       "response reaches farther than Gill's 3:1, and the western one decays within about 14°. The same "
                       "heating at 45°N gives none of this, and with its vertical heads off v8 does not respond at "
                       "all: the heating never reaches the winds.</p>")

    # geostrophic adjustment figure
    if adj["v8"]:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=SURF)
        for i, c in enumerate(("v8", "v2")):
            if not adj[c]:
                continue
            la = [x["lat"] for x in adj[c]["rows"]]
            axs[0].plot(la, [x["step1"]["ps_kept"] for x in adj[c]["rows"]], "-o", ms=3.5, color=PALETTE[i], label=c)
            axs[1].plot(la, [x["step1"]["vt_ccw"]["7"] for x in adj[c]["rows"]], "-o", ms=3.5, color=PALETTE[i], label=c)
        _style(axs[0], "pressure kept at the centre after 6 h (fraction)")
        _style(axs[1], "counter-clockwise wind around the high, level 7 (m/s)")
        for ax in axs:
            ax.set_xlabel("latitude of the bump", color=INK2); ax.axvline(0, color="#888", lw=0.8)
        axs[1].axhline(0, color="#888", lw=0.8)
        axs[0].legend(frameon=False, labelcolor=INK)
        fig.savefig(out / "emergent_adjust.png"); plt.close(fig)
        parts.append("<h3 id='emergent-adjust'>Geostrophic adjustment</h3>"
                     "<p>A 200 Pa surface-pressure bump (600 km) is planted at 11 latitudes over open ocean. Only the "
                     "balanced part of an unbalanced bump survives adjustment, and it grows as the deformation radius "
                     "shrinks with latitude; the wind it leaves behind circles the high anticyclonically: clockwise in the "
                     "north, counter-clockwise in the south, and not at all on the equator.</p>"
                     + _img("emergent_adjust.png", "geostrophic adjustment vs latitude")
                     + "<p>Both models keep more of the bump the farther from the equator, almost symmetrically, and spin "
                       "up anticyclonic winds whose sign flips at the equator, which no training target states.</p>")

    # hydrostatic figure
    if hyd["v8"]:
        mats = [("theory (inviscid)", hyd["v8"], "theory_V"), ("v8", hyd["v8"], "model_V")]
        if hyd["v8-novert"]:
            mats.append(("v8, vertical heads off", hyd["v8-novert"], "model_V"))
        fig, axs = plt.subplots(1, len(mats), figsize=(3.4 * len(mats) + 0.6, 3.4), facecolor=SURF)
        for ax, (lab, r, key) in zip(np.atleast_1d(axs), mats):
            M = np.abs(np.array(r[key]["re"]) + 1j * np.array(r[key]["im"]))
            ax.imshow(M / max(M.max(), 1e-12), cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"{lab}\n(max {M.max():.2g} m/s)", fontsize=9.5, color=INK)
            ax.set_xlabel("warmed level", fontsize=8.5, color=INK2); ax.set_xticks(range(8)); ax.set_yticks(range(8))
            ax.tick_params(labelsize=7.5)
        np.atleast_1d(axs)[0].set_ylabel("responding level (V)", fontsize=8.5, color=INK2)
        fig.savefig(out / "emergent_hydrostatic.png"); plt.close(fig)
        parts.append("<h3 id='emergent-hydrostatic'>Hydrostatic integration</h3>"
                     "<p>One level is warmed by a 1 K wave (wavenumber 4, at 45°N), nothing else changed. Geopotential is "
                     "not one of the model's variables, so to accelerate the winds correctly it has to add temperature up "
                     "from below: theory gives a response only at and above the warmed level (level 0 is the top), "
                     "weighted by layer thickness. Each panel is scaled to its own maximum.</p>"
                     + _img("emergent_hydrostatic.png", "hydrostatic response matrices")
                     + "<p><b>Not hydrostatic.</b> v8 spreads the response to the neighbouring levels, as much below the "
                       f"warmed level as above it (summed {_hyd_split(hyd['v8'])[0]:.1f} vs {_hyd_split(hyd['v8'])[1]:.1f} m/s, "
                       f"{_hyd_split(hyd['v8'])[2]:.1f} at the warmed level itself), at a few percent of the inviscid size; the "
                       f"correlation with theory ({hyd['v8']['corr_V']:.2f}) comes from the levels above. v2 is the same. "
                       "A single warmed layer with nothing else changed never occurs in the training data (real anomalies "
                       "are deep), and the model answers with a vertically smooth response instead of an upward integral. "
                       "With the vertical heads off only the warmed level responds: they carry all of the vertical spreading.</p>")
    if bal["v8"]:
        fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=SURF)
        for i, c in enumerate(("v8", "v2")):
            if not bal[c]:
                continue
            for j, la in enumerate((30.0, 45.0, -45.0)):
                rr = [r for r in bal[c]["rows"] if r["lat"] == la]
                w = [r["width_km"] for r in rr]
                ls = ("-", "--", ":")[j]
                ax.plot(w, [r["balanced"]["ps_kept"] for r in rr], ls, marker="o", ms=3.5, color=PALETTE[i],
                        label=f"{c} balanced" if j == 0 else None)
                ax.plot(w, [r["ps_only"]["ps_kept"] for r in rr], ls, marker="s", ms=3.5, color=PALETTE[i], alpha=0.45,
                        label=f"{c} pressure only" if j == 0 else None)
        ax.set_xscale("log"); ax.set_xticks([600, 1200, 2400]); ax.set_xticklabels(["600", "1200", "2400"])
        _style(ax, "pressure kept at the centre after 6 h (solid 30°N, dashed 45°N, dotted 45°S)")
        ax.set_xlabel("vortex width (km)", color=INK2); ax.set_ylim(0, 1)
        ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="lower right")
        fig.savefig(out / "emergent_balance.png"); plt.close(fig)
        parts.append("<h3 id='emergent-balance'>Learned balance, not smoothing</h3>"
                     "<p>Two perturbations of the same size: a bare surface-pressure bump, and the same bump with the "
                     "geostrophic wind that balances it at every level. A model that merely smooths would treat them alike; "
                     "a model that has learned balance keeps the balanced one.</p>"
                     + _img("emergent_balance.png", "balanced vs pressure-only vortex")
                     + "<p>At every width and latitude the balanced vortex keeps far more of its pressure (and 77–90% of its "
                       "wind). The bare bump keeps more as it widens, as adjustment theory says it should; the amount it keeps "
                       "(0.4–0.6 at 600 km in mid-latitudes) matches adjustment with the first <i>baroclinic</i> deformation "
                       "radius (~500 km), not the external one, which would keep only a few percent: the model treats a "
                       "pressure bump as a weather system, not as a Lamb disturbance.</p>")
    if kel["v8"]:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURF)
        for i, (k, lab) in enumerate((("v8", "v8"), ("v2", "v2"), ("v8-novert", "v8, vertical heads off"))):
            if not kel[k]:
                continue
            for var, ls in (("kelvin", "-"), ("anti", "--")):
                v = kel[k]["variants"][var]
                t = np.arange(len(v["phase"])) / 4.0
                ph = np.array(v["phase"]) - v["phase"][0]
                axs[0].plot(t, -np.rad2deg(ph) / kel[k]["s"], ls, color=PALETTE[i], lw=1.5,
                            label=f"{lab}, {'Kelvin' if var == 'kelvin' else 'anti-Kelvin'}")
                axs[1].plot(t, np.array(v["amp"]) / v["amp"][0], ls, color=PALETTE[i], lw=1.5)
        tt = np.linspace(0, 4, 5)
        for c, a in ((15, 0.35), (45, 0.35)):
            axs[0].plot(tt, np.rad2deg(c * tt * 86400 / 6371.0e3), color="#888", lw=0.9, ls=":", alpha=1)
        _style(axs[0], "eastward displacement (degrees)")
        _style(axs[1], "amplitude kept")
        for ax in axs:
            ax.set_xlabel("days", color=INK2)
        axs[0].legend(frameon=False, labelcolor=INK, fontsize=7.5, loc="upper left")
        fig.savefig(out / "emergent_kelvin.png"); plt.close(fig)
        k8 = kel["v8"]["variants"]
        parts.append("<h3 id='emergent-kelvin'>An equatorial Kelvin wave</h3>"
                     "<p>A first-baroclinic Kelvin wave (zonal wavenumber 2, trapped within ~10° of the equator, no "
                     "meridional wind, temperature and surface pressure in Kelvin polarization with the wind) is planted on "
                     "16 states and run 4 days in paired rollouts. Matsuno's theory: it moves east at c = 15–45 m/s "
                     "(convectively coupled to dry) and stays equatorially trapped. Control: the same wind with temperature "
                     "and pressure flipped, which is not a Kelvin mode. Dotted lines: 15 and 45 m/s.</p>"
                     + _img("emergent_kelvin.png", "Kelvin wave propagation")
                     + f"<p>v8's Kelvin wave moves east at {k8['kelvin']['phase_speed_m_s']:+.0f} m/s, keeps its meridional wind "
                       f"small (V/U {k8['kelvin']['v_over_u'][-1]:.2f} after 4 days) and {k8['kelvin']['amp_kept_end']:.0%} of its "
                       f"amplitude; the anti-Kelvin control does not propagate ({k8['anti']['phase_speed_m_s']:+.1f} m/s) and "
                       "decays faster. v2 does the same. With its vertical heads off v8 loses the wave within a day: the "
                       "coupling between the wind and the temperature that makes a Kelvin wave runs through them.</p>")
    if bar["v8"]:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURF)
        for i, var in enumerate(("west", "none", "east")):
            v = bar["v8"]["variants"][var]
            t = np.arange(len(v["E"])) / 4.0
            axs[0].plot(t, np.sqrt(np.array(v["E"]) / v["E"][0]), color=PALETTE[i], lw=1.6,
                        label={"west": "tilted west with height", "none": "untilted", "east": "tilted east"}[var])
            axs[1].plot(t, v["tilt_V2_V7"], color=PALETTE[i], lw=1.6)
            if bar["v8-novert"]:
                vv = bar["v8-novert"]["variants"][var]
                axs[0].plot(t, np.sqrt(np.array(vv["E"]) / vv["E"][0]), color=PALETTE[i], lw=1.0, ls=":")
        _style(axs[0], "v8: wave amplitude (dotted: vertical heads off)")
        _style(axs[1], "phase of V_2 relative to V_7 (rad; > 0 = westward tilt)")
        for ax in axs:
            ax.set_xlabel("days", color=INK2)
        axs[1].axhline(0, color="#888", lw=0.8)
        axs[0].legend(frameon=False, labelcolor=INK, fontsize=8.5)
        fig.savefig(out / "emergent_baroclinic.png"); plt.close(fig)
        b8 = bar["v8"]["variants"]
        parts.append("<h3 id='emergent-baroclinic'>Baroclinic structure</h3>"
                     "<p>A geostrophic, hydrostatic wave (zonal wavenumber 7, near the Eady maximum, at 45°S) is planted with "
                     "its phase tilted westward with height, untilted, or tilted eastward, and run 4 days in paired rollouts "
                     "on 16 states. Baroclinic theory: only the westward tilt draws energy from the vertical shear and grows "
                     "(carrying heat poleward); an eastward tilt gives energy back and decays; an untilted wave develops a "
                     "westward tilt as it starts to grow.</p>"
                     + _img("emergent_baroclinic.png", "baroclinic growth by tilt")
                     + f"<p>Every wave first sheds the part of it that is not a growing mode (amplitude falls to "
                       f"{_bc_growth(b8['west'])[0]:.2f}–{_bc_growth(b8['none'])[0]:.2f} over ~2 days), then grows. From that "
                       f"minimum to day 4 the westward-tilted wave grows at <b>{_bc_growth(b8['west'])[2]:.2f}/day</b> and the "
                       f"untilted one, which has meanwhile tilted westward by itself ({b8['none']['tilt_V2_V7'][0]:+.2f} → "
                       f"{b8['none']['tilt_V2_V7'][-1]:+.2f} rad), at <b>{_bc_growth(b8['none'])[2]:.2f}/day</b>, within the "
                       f"0.4–0.6/day Eady range; the eastward-tilted wave turns westward and grows only "
                       f"{_bc_growth(b8['east'])[2]:.2f}/day. All three carry heat poleward. v2 shows the same ordering at "
                       "somewhat lower rates. Without its vertical heads v8 loses every wave within two days: baroclinic "
                       "growth needs the upper and lower levels to talk.</p>")
    wv = RUNS / "f3d-rope" / "lab" / "waves"
    tw = json.loads((wv / "tropics-v8.json").read_text()) if (wv / "tropics-v8.json").exists() else None
    if tw:
        shutil.copy(wv / "wk-FLUT-v8.png", out / "emergent_wk_flut.png")
        sF, sU = tw["spectra"]["FLUT"], tw["spectra"]["U_6"]
        g = tw["gill_regression"]
        rows_e = [["Kelvin wave speed (outgoing longwave)", f"{sF['truth']['kelvin_speed_m_s']:.1f} m/s (h {sF['truth']['kelvin_h_m']:.0f} m)",
                   f"{sF['emu']['kelvin_speed_m_s']:.1f} m/s (h {sF['emu']['kelvin_h_m']:.0f} m)"],
                  ["Kelvin wave speed (wind at level 6, ~830 hPa)", f"{sU['truth']['kelvin_speed_m_s']:.1f} m/s", f"{sU['emu']['kelvin_speed_m_s']:.1f} m/s"],
                  ["Kelvin-band power above background (outgoing longwave)", f"{sF['truth']['kelvin_signal']:+.0%}", f"{sF['emu']['kelvin_signal']:+.0%}"],
                  ["equatorial Rossby band, power ÷ E3SM background", f"{sF['truth']['er_ratio']:.2f}", f"{sF['emu']['er_ratio']:.2f}"],
                  ["MJO band power (relative to E3SM)", "1.00", f"{sF['emu']['mjo_power'] / sF['truth']['mjo_power']:.2f}"],
                  ["Gill regression on equatorial Pacific rain: pattern correlation with E3SM (U low / U upper / V low)", "–",
                   " / ".join(f"{g['pattern_corr'][f]:.2f}" for f in ("U_6", "U_2", "V_6"))],
                  ["… amplitude relative to E3SM", "1", " / ".join(f"{g['amplitude_ratio'][f]:.2f}" for f in ("U_6", "U_2", "V_6"))]]
        parts.append("<h3 id='emergent-e3sm'>Against E3SM itself</h3>"
                     "<p>Theory is an idealized yardstick: E3SM has convection, friction and diffusion that it leaves out, so a "
                     "probe matching theory is not the same as matching E3SM. The E3SM reference for wave dynamics is its own "
                     f"variability. v8 runs freely over the held-out years {tw['years']} (true forcings, prescribed sea surface "
                     "temperature, correctors, jitter and seam filter), and its tropical fields are compared with E3SM's over "
                     "the same years: Wheeler–Kiladis wavenumber–frequency spectra (symmetric about the equator, divided by "
                     "E3SM's smoothed background; white lines: Kelvin and equatorial Rossby dispersion for equivalent depths "
                     "12, 25, 50 m), and the regression of low- and upper-level wind on equatorial Pacific rainfall (a "
                     "Gill-type heating response).</p>"
                     + _img("emergent_wk_flut.png", "Wheeler-Kiladis spectra, E3SM vs v8")
                     + _table(["", "E3SM", "v8 (free run)"], rows_e)
                     + "<p><b>v8's free-running tropical waves are E3SM's.</b> Its Kelvin waves travel at E3SM's convectively "
                       "coupled speed (~17–18 m/s, equivalent depth ~30 m), with a somewhat stronger signal; equatorial Rossby "
                       "and MJO power are within ~10%; and its wind response to Pacific rainfall has E3SM's pattern and "
                       "amplitude. The planted dry Kelvin wave above (seeded at 35 m/s) travelled at ~26 m/s, between the dry "
                       "and E3SM's coupled speed. v2 cannot be compared over five years (its free runs go unstable within "
                       "months); over 2011–2012 its rainfall–wind regression matched E3SM's less well than v8's (pattern "
                       "correlation 0.57 vs 0.86 for low-level zonal wind, 0.35 vs 0.82 for meridional wind). The probes with "
                       "no counterpart in E3SM's variability (Lamb pulse, inertial band, tilted baroclinic waves) still have "
                       "only theory as a yardstick; short E3SM runs from the same perturbed states would settle them.</p>")
    if ine["v8"] or lam["v8"]:
        parts.append("<h3 id='emergent-fast'>Fast, unbalanced motion is damped, not propagated</h3>"
                     "<p><b>Lamb pulse.</b> A 200 Pa bump should leave as a ring at 310 m/s (6700 km after one step, as after "
                     "the 2022 Hunga Tonga eruption). Both models instead let it decay where it is and smear outward over "
                     "about 2500 km. <b>Inertial rotation.</b> A zonal wind band with no pressure gradient to balance it "
                     "should turn by −sin(fΔt) in 6 h; intact v8 leaves it almost untouched. With its vertical heads "
                     "switched off (mean-ablated), v8 does turn it, with the right sign in each hemisphere, at about 20% of theory."
                     + (f" The clean test patches only the vertical heads' <i>response</i> to the band (their outputs replaced by "
                        f"those of the unperturbed run): the band then turns with both quadratures right at about 45% of theory "
                        f"(dV/A {inec['rows'][7]['model']['3']['dV']:+.2f} vs {inec['rows'][7]['theory']['dV']:+.2f} at 45°N, "
                        f"{inec['rows'][1]['model']['3']['dV']:+.2f} vs {inec['rows'][1]['theory']['dV']:+.2f} at 45°S). "
                        "So the horizontal machinery does compute Coriolis turning, and the vertical heads cancel it: a wind "
                        "band that is the same at every level looks like a balanced jet, and they supply the pressure-gradient "
                        "term such a jet would have.</p>" if inec else " Why is not yet understood.</p>"))
    return "".join(parts)


def _explorer_versions(out: Path) -> None:
    """Put the v1 -> v8 drawings into the architecture explorer (generated by build_arch.py),
    between markers so a rebuild replaces them."""
    p = out / "architecture.html"
    if not p.exists():
        return
    s = p.read_text()
    block = ("<!--versions--><h2 style=\"margin-top:2.4em\" id=\"versions\">How the architecture changed, v1 → v8</h2>"
             "<p>The explorer above dissects the 2D model (v2). Here is what changed across versions, and the inside of the 3D "
             "model v8, where each level is its own token and attention is split into heads across a level and heads up and "
             "down a column.</p><img class=\"fig\" src=\"arch/evolution.svg\" alt=\"v1 to v8 changes\" style=\"width:100%\">"
             "<img class=\"fig\" src=\"arch/v8.svg\" alt=\"inside v8\" style=\"width:100%;margin-top:14px\"><!--/versions-->")
    if "<!--versions-->" in s:
        i, j = s.index("<!--versions-->"), s.index("<!--/versions-->") + len("<!--/versions-->")
        s = s[:i] + block + s[j:]
    else:
        anchor = '<h2 style="margin-top:2.4em">What each design choice gives us</h2>'
        if anchor not in s:
            return
        s = s.replace(anchor, block + anchor, 1)
    p.write_text(s)


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
    import arch_svgs

    (out / "arch").mkdir(exist_ok=True)
    (out / "arch" / "evolution.svg").write_text(arch_svgs.evolution_svg())
    (out / "arch" / "v8.svg").write_text(arch_svgs.v8_svg())
    return f"""<h2 id="architecture">How we dissect the ViT</h2>
<figure style="margin:0 0 12px"><a href="architecture.html"><img class="fig" src="arch/schematic.svg" alt="architecture schematic: patches, embedding, 8 pre-LayerNorm transformer blocks, decoder; and the inside of one block"></a>
<figcaption class="muted">The 2D architecture (v1–v7): one 6-hour step with tensor shapes (top), and the inside of each of the 8 blocks with the
hook points the tools use (bottom). The 3D model v8 is drawn <a href="#arch-v8">below</a>.</figcaption></figure>
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
</div>
<h3 id="arch-versions">How the architecture changed, v1 → v8</h3>
<p>v1–v7 keep the same plain network and change what surrounds it: more data and diagnostics (v2), noise-conditioned norms
that make it stochastic (v3), longer rollouts and new losses (v4–v6), then STRATA's anti-aliased patch embedding with patch
jitter and the seam filter for long runs (v7). v8 changes the network itself: <b>one token per level</b>, and attention split
into heads that work across a level and heads that work up and down a column.</p>
<figure style="margin:0 0 12px"><img class="fig" src="arch/evolution.svg" alt="a chart of v1 to v8: which part of the network, its training and the per-step corrections changed in each version">
<figcaption class="muted">Each row is a part of the model; blue spans changed in that version and carry on to the right until the next change.</figcaption></figure>
<h3 id="arch-v8">Inside v8: tokens per level, horizontal and vertical heads</h3>
<figure style="margin:0 0 12px"><img class="fig" src="arch/v8.svg" alt="v8: 9 token levels over 4050 patches; in each block heads 0 to 3 attend across a level and heads 4 and 5 up and down a column">
<figcaption class="muted">In v8 the residual stream is 9 × 4050 × 384. Each block's heads 0–3 attend across their level, heads 4–5 up and down
their column, and the MLPs act token by token. Inside the network every exchange between levels goes through a vertical head; after it, the
3×3 output convolution mixes all 44 output channels, levels included (knocking out the vertical heads removes 94% of the
coupling between levels; the output convolution carries most of the rest: see <a href="#inside-v8">Inside v8</a>).</figcaption></figure>"""


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
    body = f"""<nav class="muted"><a href="#architecture">Architecture</a><a href="training.html">Training</a><a href="#versions">Versions</a><a href="#stability">Stability</a><a href="#decade">Decade runs</a><a href="#threed">3D model</a><a href="#rossby">Rossby waves</a><a href="#circuits">Circuits</a><a href="#inside-v8">Inside v8</a><a href="#emergent">Emergent dynamics</a><a href="#notebooks">Notebooks</a></nav>
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
{inside_v8_section(out)}
{emergent_section(out)}
<h2 id="notebooks">Hackathon notebooks (00–07 executed on {current["v"]["name"]}, 08 on v8)</h2><ul>{nb_list}</ul>
<p class="muted">Run them on Swing: <code>HACKATHON.md</code> in the vit-interp repository; the exege layers are on the
<code>mahf708/circuits</code> branch of exege.</p>"""
    _explorer_versions(out)
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
