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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("/lcrc/globalscratch/ac.ngmahfouz/vit-interp")
RUNS = ROOT / "runs"
NB_EXEC = ROOT / "notebooks-executed"
OUT = Path("/lcrc/group/e3sm/public_html/diagnostic_output/ac.ngmahfouz/vit-interp")
URL = "https://web.lcrc.anl.gov/public/e3sm/diagnostic_output/ac.ngmahfouz/vit-interp/"

# reference categorical palette, fixed order; one hue per version, never reused
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
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
                  ("+ 6 more epochs at 8 steps", "eval-ft8b")],
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
                  ("+ 6 more epochs at 8 steps (4-member ensemble)", "eval-ft8b")],
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
        "evals": [("long rollouts (4-member ensemble)", "eval-long")],
        "recipe": [
            "As v3 (stochastic, CRPS over 2 members, ACE-style correctors), starting from v3's one-step checkpoint, so it parallels v3's own short-rollout stages.",
            "Rollouts: each batch a random length between 4 and 16 six-hour steps (1–4 days); every step but the last runs without gradients, and the loss is taken on the last step only (the pushforward trick; ACE's optimize_last_step_only). Memory stays at one step's worth.",
            "Training: 4 epochs (lr 5e-5, global batch 8) from v3's one-step checkpoint (epoch 12). <a href=\"../training.html#v4\">Exact curriculum</a>.",
            "Why: rollout training was the biggest win in v1 and v2 and had not converged; with MSE, long horizons blur, but CRPS keeps the ensemble sharp.",
            "Load: <code>pg.load(\"v4\")</code>, <code>--source v4</code> (on Hugging Face as <code>v4/v4.pt</code>).",
        ],
        "changes": "long random-length rollouts with last-step gradient (stochastic, CRPS)",
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
    if (RUNS / v["checkpoint"]).exists():
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
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")


def fig_acc(series: list[tuple[str, dict, str, object]], path: Path) -> bool:
    """series: (label, eval metrics, color, linestyle). Persistence from the first."""
    if not series:
        return False
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), facecolor=SURF, sharey=True)
    x = [1, 3, 5, 10]
    for ax, v in zip(axes, ACC_VARS, strict=True):
        for label, m, c, ls in series:
            ax.plot(x, [m["skill"][v][k]["acc"] for k in LEADS], linestyle=ls, marker="o", color=c,
                    lw=2, ms=5, label=label)
        ax.plot(x, [series[0][1]["skill"][v][k]["acc_persistence"] for k in LEADS], "--", color=INK2,
                lw=1.2, label="persistence")
        _style(ax, v)
        ax.set_xticks(x)
        ax.set_xlabel("lead (days)", color=INK2, fontsize=9)
        ax.axhline(0.6, color=INK2, lw=0.6, ls=":")
    axes[0].set_ylabel("anomaly correlation (2010)", color=INK2)
    axes[0].set_ylim(-0.1, 1.02)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=min(4, len(series) + 1),
               frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def fig_training(v: dict, path: Path) -> bool:
    stages = [(label, _jsonl(RUNS / v["run"] / d / "metrics.jsonl")) for label, d in v["stages"]]
    stages = [(l, r) for l, r in stages if r]
    if not stages:
        return False
    fig, axes = plt.subplots(1, 1 + len(stages), figsize=(4.2 * (1 + len(stages)), 3.2), facecolor=SURF)
    one = stages[0][1]
    ax = axes[0]
    ax.plot([r["epoch"] for r in one], [r["val_loss"] for r in one], "-", color=PALETTE[0], lw=2)
    _style(ax, f"{stages[0][0]}: validation loss")
    ax.set_xlabel("epoch", color=INK2, fontsize=9)
    for ax, (label, rows) in zip(axes[1:], stages, strict=True):
        k = next((k for k in rows[0] if k.startswith("rmse_") and k != "rmse_1step"), None)
        if k:
            days = int(k.split("_")[1].replace("step", "")) / 4
            ax.plot([r["epoch"] for r in rows], [r[k]["U_3"] for r in rows], "-o", color=PALETTE[0], lw=2, ms=3)
            _style(ax, f"{label}: {days:g}-day U_3 RMSE (m/s)")
        ax.set_xlabel("epoch", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def fig_rossby(explore: Path, path: Path) -> list[dict]:
    phis = [p for p in (30, 45, 60, -45) if (explore / f"rossby_phi{p}.json").exists()]
    if not phis:
        return []
    fig, axes = plt.subplots(1, len(phis), figsize=(3.3 * len(phis), 3.2), facecolor=SURF)
    axes = np.atleast_1d(axes)
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
        ax.set_xlabel("zonal wavenumber s", color=INK2, fontsize=9)
        dop = explore / f"rossby_doppler_phi{phi}.json"
        out.append({"lat": lat, "corr": float(np.corrcoef(f, g)[0, 1]), "ratio": float(np.polyfit(g, f, 1)[0]),
                    "M": m["M"], "M_doppler": json.loads(dop.read_text())["measure"]["M"] if dop.exists() else None})
    axes[0].set_ylabel("intrinsic frequency ω − kŪ (rad/day)", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
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
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
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
        figs = {}
        for name in ("snapshots", "maps_time_mean", "zonal_mean_U_T", "global_mean_series"):
            src = RUNS / v["run"] / last / f"{name}.png"
            if src.exists():
                shutil.copy(src, d / f"{name}.png")
                figs[name] = f"{name}.png"
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
    head = next((m for _, m, _ in reversed(evals)), None) or _eval(v, v["headline_eval"])
    head_name = next((e for _, _, e in reversed(evals)), v["headline_eval"])
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
            steer = (f" At 45°N the shape correlation rises from {c3:.2f} to {c5:.2f} when the flow the law "
                     "Doppler-shifts by is taken lower (U_5 instead of U_3).")
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
<div class="card" style="display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:16px;align-items:center">
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
    evaluated = [(s["v"]["name"] + ": " + s["v"]["title"], s["eval"], PALETTE[i], "-")
                 for i, s in enumerate(summ) if s["eval"]]
    has_cmp = fig_acc(evaluated, out / "versions_acc.png")
    released = [s for s in summ if s["status"] == "released"]
    current = released[-1] if released else summ[0]
    m = current["eval"] or {}
    kpis = ""
    if m:
        sk = m["skill"]
        kpis = (f'<div class="grid"><div class="card"><div class="muted">current release</div><div class="kpi">'
                f'{current["v"]["name"]}</div><div class="muted">{html.escape(current["v"]["title"])}</div></div>'
                f'<div class="card"><div class="muted">5-day anomaly correlation, U_3</div><div class="kpi">'
                f'{sk["U_3"]["5d"]["acc"]:.2f}</div><div class="muted">persistence {sk["U_3"]["5d"]["acc_persistence"]:.2f}</div></div>'
                f'<div class="card"><div class="muted">free run on 2010</div><div class="kpi">{m["days_run"]:.0f} days</div>'
                f'<div class="muted">{"then drifts" if m.get("blowup") else "stable"}</div></div></div>')
    nbs = notebooks_html(out)
    nb_list = "".join(f'<li><a href="{h}">{s}</a>: {html.escape(NB_TITLES.get(s, ""))}</li>' for s, h in nbs)
    body = f"""<nav class="muted"><a href="#architecture">Architecture</a><a href="training.html">Training</a><a href="#versions">Versions</a><a href="#rossby">Rossby waves</a><a href="#circuits">Circuits</a><a href="#notebooks">Notebooks</a></nav>
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
{analysis_sections(out, current["v"])}
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
