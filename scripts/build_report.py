"""Build the public results page (static HTML + PNGs + the executed notebooks as HTML).

    .venv/bin/python scripts/build_report.py [--out /lcrc/group/e3sm/public_html/diagnostic_output/ac.ngmahfouz/vit-interp]

Everything is read from the run directory; rerun after new results land.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/lcrc/globalscratch/ac.ngmahfouz/vit-interp")
RUN = ROOT / "runs" / "vit-p4-d384-l8"
NB_EXEC = ROOT / "notebooks-executed"
OUT = Path("/lcrc/group/e3sm/public_html/diagnostic_output/ac.ngmahfouz/vit-interp")
URL = "https://web.lcrc.anl.gov/public/e3sm/diagnostic_output/ac.ngmahfouz/vit-interp/"

# reference categorical palette, fixed order (blue, orange, aqua); ink and grid tokens
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

STAGES = [  # (label, eval dir)
    ("one-step (v1 stage 1)", "eval-stage1-final"),
    ("one-step + dry-air/positivity correction", "eval-stage1-final-corrected"),
    ("rollout-tuned + correction (release v1)", "eval-ft8-corrected"),
]
SKILL_VARS = [("PS", "Pa"), ("T_7", "K"), ("U_3", "m/s"), ("V_3", "m/s"), ("specific_total_water_7", "kg/kg")]
LEADS = ["1d", "3d", "5d", "10d"]


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


def _metrics(name):
    p = RUN / name / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_skill(out: Path) -> str | None:
    evals = [(label, _metrics(d)) for label, d in STAGES]
    evals = [(l, m) for l, m in evals if m]
    if not evals:
        return None
    names = ["PS", "T_7", "U_3", "V_3"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), facecolor=SURF, sharey=True)
    x = [1, 3, 5, 10]
    for ax, v in zip(axes, names):
        for (label, m), c, ls in zip(evals, (C1, C2, C3), ("-", (0, (4, 3)), "-")):
            acc = [m["skill"][v][k]["acc"] for k in LEADS]
            ax.plot(x, acc, linestyle=ls, marker="o", color=c, lw=2, ms=5, label=label)
        pers = [evals[0][1]["skill"][v][k]["acc_persistence"] for k in LEADS]
        ax.plot(x, pers, "--", color=INK2, lw=1.2, label="persistence")
        _style(ax, v)
        ax.set_xticks(x)
        ax.set_xlabel("lead (days)", color=INK2, fontsize=9)
        ax.axhline(0.6, color=INK2, lw=0.6, ls=":")
    axes[0].set_ylabel("anomaly correlation", color=INK2)
    axes[0].set_ylim(-0.1, 1.02)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False,
               fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "skill_acc.png", dpi=150)
    plt.close(fig)
    return "skill_acc.png"


def fig_training(out: Path) -> str | None:
    s1 = RUN / "stage1" / "metrics.jsonl"
    if not s1.exists():
        return None
    rows = [json.loads(l) for l in s1.read_text().splitlines() if l.strip()]
    ft = []
    for d in ("ft2/ft4", "ft2/ft8"):
        p = RUN / d / "metrics.jsonl"
        if p.exists():
            ft += [(d.split("/")[-1], json.loads(l)) for l in p.read_text().splitlines() if l.strip()]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.2), facecolor=SURF)
    ax = axes[0]
    ax.plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows], "-", color=C1, lw=2)
    _style(ax, "one-step: validation loss (persistence = 1)")
    ax.set_xlabel("epoch", color=INK2, fontsize=9)
    key5 = next((k for k in rows[0] if k.startswith("rmse_") and k != "rmse_1step"), None)
    ax = axes[1]
    if key5:
        ax.plot([r["epoch"] for r in rows], [r[key5]["U_3"] for r in rows], "-", color=C1, lw=2)
    _style(ax, "one-step: 5-day U_3 RMSE (m/s)")
    ax.set_xlabel("epoch", color=INK2, fontsize=9)
    ax = axes[2]
    if ft:
        k = next(k for k in ft[0][1] if k.startswith("rmse_") and k != "rmse_1step")
        ax.plot(range(1, len(ft) + 1), [r[k]["U_3"] for _, r in ft], "-o", color=C1, lw=2, ms=4)
        n4 = sum(1 for tag, _ in ft if tag == "ft4")
        ax.axvline(n4 + 0.5, color=INK2, lw=0.8, ls=":")
        ax.text(n4 + 0.6, ax.get_ylim()[1], " 8-step", color=INK2, fontsize=8, va="top")
        ax.text(0.9, ax.get_ylim()[1], "4-step", color=INK2, fontsize=8, va="top")
    _style(ax, "rollout fine-tuning: 10-day U_3 RMSE (m/s)")
    ax.set_xlabel("fine-tuning epoch", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "training.png", dpi=150)
    plt.close(fig)
    return "training.png"


def fig_rossby(out: Path) -> tuple[str | None, list[dict]]:
    ex = RUN / "explore-final"
    phis = [30, 45, 60, -45]
    have = [p for p in phis if (ex / f"rossby_phi{p}.json").exists()]
    if not have:
        return None, []
    fig, axes = plt.subplots(1, len(have), figsize=(3.3 * len(have), 3.2), facecolor=SURF)
    summary = []
    import numpy as np

    for ax, phi in zip(np.atleast_1d(axes), have):
        m = json.loads((ex / f"rossby_phi{phi}.json").read_text())["measure"]
        s = [r["alpha"] for r in m["rows"]]
        ax.plot(s, [r["model"] for r in m["rows"]], "-o", color=C1, lw=2, ms=4, label="ViT")
        ax.plot(s, [r["law"] for r in m["rows"]], "-s", color=C2, lw=2, ms=4, label="barotropic theory")
        hemi = "N" if phi > 0 else "S"
        _style(ax, f"{abs(phi)}°{hemi}")
        ax.set_xlabel("zonal wavenumber s", color=INK2, fontsize=9)
        ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
        f = np.array([r["model"] for r in m["rows"]])
        g = np.array([r["law"] for r in m["rows"]])
        dop = ex / f"rossby_doppler_phi{phi}.json"
        summary.append({"lat": f"{abs(phi)}°{hemi}", "corr": float(np.corrcoef(f, g)[0, 1]),
                        "ratio": float(np.polyfit(g, f, 1)[0]), "M": m["M"],
                        "M_doppler": json.loads(dop.read_text())["measure"]["M"] if dop.exists() else None})
    axes = np.atleast_1d(axes)
    axes[0].set_ylabel("intrinsic frequency ω − kŪ (rad/day)", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "rossby.png", dpi=150)
    plt.close(fig)
    return "rossby.png", summary


def copy_eval_figs(out: Path) -> dict:
    figs = {}
    for name in ("snapshots", "maps_time_mean", "global_mean_series", "zonal_mean_U_T", "skill_vs_lead"):
        for tag, d in (("v1", "eval-ft8-corrected"), ("onestep", "eval-stage1-final")):
            p = RUN / d / f"{name}.png"
            if p.exists():
                shutil.copy(p, out / f"{tag}_{name}.png")
                figs[f"{tag}_{name}"] = f"{tag}_{name}.png"
    return figs


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


def _table(header, rows):
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def skill_table():
    rows = []
    for label, d in STAGES:
        m = _metrics(d)
        if not m:
            continue
        free = f"{m['days_run']:.0f} days" + (" (then drifts)" if m.get("blowup") else " (stable)")
        acc5 = " / ".join(f"{m['skill'][v]['5d']['acc']:.2f}" for v in ("PS", "T_7", "U_3", "V_3"))
        rmse5 = " / ".join(f"{m['skill'][v]['5d']['rmse']:.3g}" for v in ("PS", "T_7", "U_3", "V_3"))
        rows.append([html.escape(label), free, acc5, rmse5])
    return _table(["model", "free run (2010, true forcing, SST prescribed)",
                   "5-day ACC: PS / T_7 / U_3 / V_3", "5-day RMSE: PS (Pa) / T_7 (K) / U_3, V_3 (m/s)"], rows)


def circuits_section():
    ex = RUN / "explore-final"
    parts = []
    eap = ex / "eap_v3_npac.json"
    pr = ex / "prune_v3_npac.json"
    if eap.exists():
        a = json.loads(eap.read_text())
        top = list(a["scores"].items())[:10]
        parts.append("<h3>Edge attribution (EAP): which edges move the North Pacific jet-level V tendency</h3>"
                     f"<p>Metric: area-mean 6-h V_3 change within 2000 km of 45°N, 180°; base 2010-01-15, "
                     f"counterfactual 2010-01-20 (metric {a['clean']:.3f} vs {a['corrupt']:.3f}).</p>"
                     + _table(["edge", "metric change if this edge alone is patched"],
                              [[f"<code>{html.escape(e)}</code>", f"{s:+.3f}"] for e, s in top]))
    if pr.exists():
        c = json.loads(pr.read_text())
        from collections import Counter

        by = Counter(e.split("->")[0].split(".")[0] for e in c["edges"])
        parts.append(f"<h3>Greedy pruning (Vi-CD)</h3><p><b>{c['n_edges']} of {c['of']}</b> edges recover "
                     f"<b>{c['recovery']:.0%}</b> of the base-vs-counterfactual gap (τ = {c.get('tau')}). "
                     "Edges by sending component: "
                     + ", ".join(f"<code>{k}</code> {n}" for k, n in by.most_common(10)) + ".</p>")
    return "\n".join(parts)


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
"""


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    figs = copy_eval_figs(out)
    skill_png = fig_skill(out)
    train_png = fig_training(out)
    ross_png, ross = fig_rossby(out)
    nbs = notebooks_html(out)
    v1 = _metrics("eval-ft8-corrected") or {}
    one = _metrics("eval-stage1-final") or {}

    def img(name, alt):
        return f'<img class="fig" src="{name}" alt="{html.escape(alt)}" loading="lazy">' if name else ""

    nb_titles = {
        "00_quickstart": "one step and a 5-day forecast against E3SM",
        "01_attention_atlas": "where each of the 48 heads looks",
        "02_logit_lens": "decode every block: when does the model know the answer?",
        "03_ablation_lab": "knock out heads and MLPs",
        "04_direct_attribution": "who writes the jet-level wind over a box",
        "05_exege_latents": "latent archives, region analysis, sparse autoencoders (exege)",
        "06_rossby_probe": "plant a Rossby wave: does the model know β?",
        "07_exege_circuits_and_laws": "Vi-CD edge circuits and LAWFUL through exege",
    }
    nb_list = "".join(f'<li><a href="{h}">{s}</a>: {html.escape(nb_titles.get(s, ""))}</li>' for s, h in nbs)
    ross_rows = [[r["lat"], f"{r['corr']:.2f}", f"{r['ratio']:.1f}×", f"{r['M']:.0f}",
                  "" if r["M_doppler"] is None else f"{r['M_doppler']:.2f}"] for r in ross]
    ross_prose = ""
    if ross:
        corr = ", ".join(f"{r['corr']:+.2f} at {r['lat']}" for r in ross)
        ratio = ", ".join(f"{r['ratio']:.1f}× at {r['lat']}" for r in ross)
        ex = RUN / "explore-final"
        steer = ""
        u3, u5 = ex / "rossby_corr_phi45_U_3.json", ex / "rossby_corr_phi45_U_5.json"
        if u3.exists() and u5.exists():
            c3 = json.loads(u3.read_text())["measure"]["M"]
            c5 = json.loads(u5.read_text())["measure"]["M"]
            steer = (f" At 45°N the shape correlation rises from {c3:.2f} to {c5:.2f} when the flow the law "
                     "Doppler-shifts by is taken lower (U_5 instead of U_3): the model may steer waves with a "
                     "deeper-layer wind.")
        ross_prose = f"""<p>Everywhere, the model moves planted waves <b>westward relative to the U_3 flow</b> and faster than
barotropic theory (fitted magnitude {ratio}), so LAWFUL's misfit score M is strongly negative. How the frequency varies
with wavenumber (the <b>shape</b> of the dispersion relation) agrees with theory to varying degrees: correlation {corr}.{steer}</p>
<p>LAWFUL's circuit search over whole blocks finds no τ-consistent set: much of the response is carried by the
<b>direct path</b> (patch embedding → output), which LAWFUL by design does not patch. Steering level, baroclinicity
and that within-patch linear dynamics are open questions for the hackathon.</p>"""
    acc5 = v1.get("skill", {})
    kpis = ""
    if acc5:
        kpis = f"""<div class="grid">
<div class="card"><div class="muted">5-day anomaly correlation, U_3 (jet-level wind)</div><div class="kpi">{acc5['U_3']['5d']['acc']:.2f}</div><div class="muted">persistence {acc5['U_3']['5d']['acc_persistence']:.2f}</div></div>
<div class="card"><div class="muted">5-day anomaly correlation, surface pressure</div><div class="kpi">{acc5['PS']['5d']['acc']:.2f}</div><div class="muted">persistence {acc5['PS']['5d']['acc_persistence']:.2f}</div></div>
<div class="card"><div class="muted">stable free run (2010, corrected)</div><div class="kpi">{v1.get('days_run', 0):.0f} days</div><div class="muted">one-step model: {one.get('days_run', 0):.0f} days</div></div>
</div>"""

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ViT emulator internals</title><style>{CSS}</style></head>
<body><main>
<nav class="muted"><a href="#model">Model</a><a href="#skill">Skill</a><a href="#rossby">Rossby waves</a><a href="#circuits">Circuits</a><a href="#notebooks">Notebooks</a></nav>
<h1>Inside a ViT emulator of E3SM</h1>
<p class="lede">A close-to-vanilla Vision Transformer that steps E3SMv3 (1°, AMIP) forward 6 hours, built to be taken apart
for the interpretability hackathon: attention heads, circuits (Vi-CD), and physical laws (LAWFUL), with
<a href="https://github.com/mahf708/exege">exege</a>. Updated {datetime.now():%Y-%m-%d %H:%M %Z}.</p>
{kpis}

<h2 id="model">The model</h2>
<ul>
<li><b>Input:</b> 34 prognostic fields (PS, TS; T, U, V and total water on 8 levels) + 5 forcings (insolation, surface height, land/ocean/ice fractions).</li>
<li><b>Tokens:</b> 4°×4° patches → 45×90 = <b>4050 tokens</b> with a learned position embedding (the model's only knowledge of latitude).</li>
<li><b>Body:</b> 8 pre-LayerNorm blocks, 6 heads × 64, GELU MLP (16 M parameters); no windowing, no spectral layers: all horizontal mixing is attention.</li>
<li><b>Output:</b> the 6-h tendency. Trained on 2000–2009 (one-step, 40 epochs), then 4- and 8-step rollouts; everything here is on <b>2010</b>, unseen.</li>
<li><b>Outside the network:</b> optional ACE-style dry-air and water-positivity correction for free runs; it does not change the circuits.</li>
</ul>
{img(train_png, "training curves")}

<h2 id="skill">Skill on 2010</h2>
{skill_table()}
{img(skill_png, "anomaly correlation vs lead")}
<p class="muted">The dry-air/positivity correction leaves short-range skill unchanged (the blue and dashed orange curves
coincide); what it buys is a longer stable free run.</p>
<h3>Snapshots: one-step model (left column, day 30 seams) vs rollout-tuned release</h3>
<div class="grid"><div>{img(figs.get("onestep_snapshots"), "one-step snapshots")}<div class="muted">one-step model</div></div>
<div>{img(figs.get("v1_snapshots"), "release v1 snapshots")}<div class="muted">release v1 (rollout-tuned, corrected)</div></div></div>
<h3>Release v1: time means, zonal means, drift</h3>
{img(figs.get("v1_maps_time_mean"), "time-mean maps")}
{img(figs.get("v1_zonal_mean_U_T"), "zonal-mean U and T")}
{img(figs.get("v1_global_mean_series"), "global-mean series")}

<h2 id="rossby">Do Rossby waves emerge? (LAWFUL)</h2>
<p>We plant a small, geostrophically balanced, non-divergent wave of zonal wavenumber s = 2–14 on 8 real 2010 states,
take one step, and read the wave's frequency off the model's response. Barotropic theory predicts an intrinsic
frequency ω − kŪ = −βk/(k²+l²): westward, and largest in phase speed for the longest waves.</p>
{img(ross_png, "Rossby dispersion: model vs theory")}
{_table(["latitude", "correlation with theory (shape)", "model / theory magnitude", "LAWFUL M (eq. 10)", "M incl. Doppler shift"], ross_rows) if ross_rows else ""}
{ross_prose}

<h2 id="circuits">Circuits (Vi-CD) on the real model</h2>
{circuits_section()}

<h2 id="notebooks">Hackathon notebooks (executed on release v1)</h2>
<ul>{nb_list}</ul>
<p class="muted">Run them yourself on Swing: see <code>HACKATHON.md</code> in the vit-interp repository; the exege layers are on the
<code>mahf708/circuits</code> branch of exege. Checkpoint: <code>{RUN}/release/v1.pt</code>.</p>
</main></body></html>"""
    (out / "index.html").write_text(page)
    for p in out.rglob("*"):
        p.chmod(0o755 if p.is_dir() else 0o644)
    out.chmod(0o755)
    print(f"wrote {out}/index.html  ->  {URL}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    build(Path(ap.parse_args().out))
