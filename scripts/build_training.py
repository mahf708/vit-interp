"""The results site's training page: how every version was trained, stage by stage, and what each
method does. The curriculum tables are read from each stage's ``args.json``, ``metrics.jsonl`` and
checkpoints, so they show exactly what ran; the step times were measured from the PBS logs.

Used by ``build_report.py`` (``training_page``); not meant to run on its own.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

N_GPU = 4  # every stage ran data-parallel on one Swing node: 4 × A100 40 GB

# Median seconds per optimizer step (4 GPUs, excluding validation), read from the PBS logs.
STEP_SECONDS = {
    "vit-p4-d384-l8/stage1": 0.143, "vit-p4-d384-l8/stage2": 0.272,
    "vit-p4-d384-l8/ft2/ft4": 0.258, "vit-p4-d384-l8/ft2/ft8": 0.289,
    "vit2-p4-d384-l8/stage1": 0.295, "vit2-p4-d384-l8/ft4": 0.515,
    "vit2-p4-d384-l8/ft8": 0.538, "vit2-p4-d384-l8/ft8b": 0.547,
    "vit3-p4-d384-l8/stage1": 0.434, "vit3-p4-d384-l8/ft4": 0.900,
    "vit3-p4-d384-l8/ft8": 0.917, "vit3-p4-d384-l8/ft8b": 0.943,
    "vit4-p4-d384-l8/long": 0.664,
    "vit5a-p4-d384-l8/long": 1.05, "vit6-p4-d384-l8/long": 1.10, "vit7-p4-d384-l8/long": 1.18,
    "s3d-split-d384-rope/stage1": 0.672, "f3d-rope/long": 3.2,
}

# Each version's full lineage, oldest stage first: (stage dir, role). Roles: "own" (trained for this
# version), "inherited" (an earlier version's stage this one starts from), "extension" (trained
# after the released checkpoint, not part of the release).
LINEAGE = {
    "v1": [("vit-p4-d384-l8/stage1", "own"), ("vit-p4-d384-l8/stage2", "own"),
           ("vit-p4-d384-l8/ft2/ft4", "own"), ("vit-p4-d384-l8/ft2/ft8", "own")],
    "v2": [("vit2-p4-d384-l8/stage1", "own"), ("vit2-p4-d384-l8/ft4", "own"),
           ("vit2-p4-d384-l8/ft8", "own"), ("vit2-p4-d384-l8/ft8b", "extension")],
    "v3": [("vit2-p4-d384-l8/stage1", "inherited"), ("vit3-p4-d384-l8/stage1", "own"),
           ("vit3-p4-d384-l8/ft4", "own"), ("vit3-p4-d384-l8/ft8", "own"),
           ("vit3-p4-d384-l8/ft8b", "extension")],
    "v4": [("vit2-p4-d384-l8/stage1", "inherited"), ("vit3-p4-d384-l8/stage1", "inherited"),
           ("vit4-p4-d384-l8/long", "own")],
    "v7": [("vit2-p4-d384-l8/stage1", "inherited"), ("vit3-p4-d384-l8/stage1", "inherited"),
           ("vit4-p4-d384-l8/long", "inherited"), ("vit5a-p4-d384-l8/long", "own"),
           ("vit6-p4-d384-l8/long", "own"), ("vit7-p4-d384-l8/long", "own")],
    "3D": [("s3d-split-d384-rope/stage1", "own"), ("f3d-rope/long", "own")],
}
RELEASED_FROM = {"v1": "vit-p4-d384-l8/ft2/ft8", "v2": "vit2-p4-d384-l8/ft8",
                 "v3": "vit3-p4-d384-l8/ft8", "v4": "vit4-p4-d384-l8/long",
                 "v7": "vit7-p4-d384-l8/long", "3D": ""}

COLORS = {1: "#2a78d6", 4: "#1baf7a", 8: "#eda100", "rand": "#eb6834"}


def _stage(runs: Path, rel: str) -> dict | None:
    d = runs / rel
    if not (d / "args.json").exists():
        return None
    a = json.loads((d / "args.json").read_text())
    m = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()] \
        if (d / "metrics.jsonl").exists() else []
    steps = m[-1]["step"] if m else 0
    best = min(m, key=lambda r: r["val_loss"]) if m else None
    sec = STEP_SECONDS.get(rel)
    init = a.get("init")
    if init:
        p = Path(init)
        init = f"{p.parent.relative_to(runs)}/{p.name}" if p.is_relative_to(runs) else init
    # rows also come from mid-epoch walltime stops: steps per epoch from the end-of-epoch rows
    ratios = [r["step"] / r["epoch"] for r in m if r["epoch"] >= 1]
    per_epoch = int(round(min(ratios))) if ratios else (steps if m else None)
    epochs_run = round(steps / per_epoch, 1) if per_epoch else 0
    if epochs_run == int(epochs_run):
        epochs_run = int(epochs_run)
    return {"rel": rel, "a": a, "m": m, "steps": steps, "best": best, "init": init,
            "epochs_run": epochs_run, "gpu_h": steps * sec * N_GPU / 3600 if sec else None,
            "per_epoch": per_epoch}


def _length(a: dict) -> tuple[str, object]:
    n = a.get("n_steps", 1)
    if a.get("min_steps"):
        return f"{a['min_steps']}–{n} steps, random per batch", "rand"
    return ("1 step" if n == 1 else f"{n}-step rollout"), n


def _loss(a: dict) -> str:
    if a.get("loss") == "ensemble":
        s = f"fair CRPS, {a['n_ensemble']} members"
    else:
        s = "MSE"
    if a.get("min_steps") and a.get("grad_last_only"):
        s += "; last step only"
    elif a.get("n_steps", 1) > 1:
        s += f"; mean over {a['n_steps']} steps"
    return s


def _fmt_lr(x: float) -> str:
    return f"{x:.0e}".replace("e-0", "e-")


def curriculum_table(runs: Path, v: str) -> str:
    rows = []
    for rel, role in LINEAGE[v]:
        s = _stage(runs, rel)
        if s is None:
            continue
        a = s["a"]
        length, _ = _length(a)
        best = s["best"]
        epochs = f"{s['epochs_run']} of {a['epochs']}" if s["epochs_run"] != a["epochs"] else f"{a['epochs']}"
        kept = f"epoch {best['epoch']} (val {best['val_loss']:.4f})" if best else "–"
        if a.get("select") == "last" and s["m"]:
            kept = f"latest weights (val {s['m'][-1]['val_loss']:.4f})"
        tag = {"inherited": " <span class='tag'>from an earlier version</span>",
               "extension": " <span class='tag training'>after the release; not released</span>"}.get(role, "")
        if rel == RELEASED_FROM[v]:
            tag += " <span class='tag released'>release</span>"
        extras = []
        if a.get("corrector"):
            extras.append("correctors " + a["corrector"].replace(",", ", "))
        if a.get("ema"):
            extras.append(f"EMA {a['ema']}")
        if a.get("input_noise"):
            extras.append(f"input noise {a['input_noise']:g}σ")
        if a.get("noise_dim"):
            extras.append(f"noise-conditioned LN ({a['noise_dim']}-d {a.get('noise_type', 'gaussian')})")
        if a.get("head_conv"):
            extras.append(f"{a['head_conv']}×{a['head_conv']} output conv")
        if a.get("energy_weight"):
            extras.append(f"energy score {a['energy_weight']:g}")
        if a.get("train_jitter"):
            extras.append("patch jitter in training")
        if a.get("embed_filter"):
            extras.append("anti-aliased embedding")
        if a.get("arch") == "vit3d":
            extras.append(f"3D tokens, {a.get('attn3d')} attention ({a.get('heads_v')} vertical heads), {a.get('pos3d')} positions, dim {a.get('dim')}")
        rows.append([
            f"<code>{html.escape(rel)}</code>{tag}",
            f"<code>{html.escape(s['init'])}</code>" if s["init"] else "scratch",
            a["train_years"].replace("-", "–"),
            length, _loss(a), epochs,
            f"{s['per_epoch']:,} × {a['batch'] * N_GPU}" if s["per_epoch"] else "–",
            f"{_fmt_lr(a['lr'])}, warm-up {a['warmup']}",
            kept,
            f"{s['gpu_h']:.1f}" if s["gpu_h"] else "–",
            "; ".join(extras) or "–",
        ])
    head = ["stage", "starts from", "data", "trajectory", "loss", "epochs", "steps/epoch × global batch",
            "peak lr (cosine to 0)", "best checkpoint (2010 one-step val)", "GPU-h*", "other"]
    th = "".join(f"<th>{h}</th>" for h in head)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="curr"><tr>{th}</tr>{trs}</table></div>'


def curriculum_svg(runs: Path) -> str:
    """One row per version; each stage a bar whose width is its compute (GPU-hours)."""
    data = {v: [(s, role) for rel, role in LINEAGE[v] if (s := _stage(runs, rel))] for v in LINEAGE}
    total = max(sum(s["gpu_h"] or 0 for s, _ in rows) for rows in data.values())
    W, left, rowh, top = 1000, 46, 46, 8
    scale = (W - left - 10) / total
    out = [f'<svg viewBox="0 0 {W} {top + rowh * len(data) + 34}" role="img" class="currsvg" '
           f'aria-label="training curriculum of each version: stages as bars sized by compute">']
    for i, (v, rows) in enumerate(data.items()):
        y = top + i * rowh
        out.append(f'<text x="0" y="{y + 24}" class="lbl" font-weight="650">{v}</text>')
        x = left
        for s, role in rows:
            w = max((s["gpu_h"] or 0) * scale, 3)
            length, key = _length(s["a"])
            col = COLORS.get(key, "#888")
            op = "0.35" if role == "inherited" else "1"
            dash = ' stroke-dasharray="4 3" stroke="var(--fg2)" stroke-width="1.5" fill-opacity="0.3"' \
                if role == "extension" else ""
            title = (f"{s['rel']}: {length}, {_loss(s['a'])}, {s['epochs_run']} epochs, "
                     f"{s['gpu_h']:.1f} GPU-h" + (" (inherited)" if role == "inherited" else "")
                     + (" (after the release)" if role == "extension" else ""))
            out.append(f'<g><title>{html.escape(title)}</title><rect x="{x:.1f}" y="{y + 6}" width="{w - 2:.1f}" '
                       f'height="28" rx="4" fill="{col}" opacity="{op}"{dash}/>')
            short = {"rand": "4–16"}.get(key, str(key))
            label = f"{short}-step × {s['epochs_run']}" if w > 78 else (short if w > 22 else "")
            if label:
                out.append(f'<text x="{x + 6:.1f}" y="{y + 25}" class="bar">{label}</text>')
            out.append("</g>")
            if s["rel"] == RELEASED_FROM[v]:
                out.append(f'<path d="M{x + w - 2:.1f} {y + 2} l-6 -6 h12 z" fill="var(--fg)"><title>released checkpoint</title></path>')
            x += w
    ax = top + rowh * len(data) + 4
    out.append(f'<line x1="{left}" x2="{W - 10}" y1="{ax}" y2="{ax}" stroke="var(--line)"/>')
    for h in range(0, int(total) + 1, 10):
        xx = left + h * scale
        out.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{ax}" y2="{ax + 5}" stroke="var(--fg2)"/>'
                   f'<text x="{xx:.1f}" y="{ax + 20}" class="ax" text-anchor="middle">{h}</text>')
    out.append(f'<text x="{W - 10}" y="{ax + 20}" class="ax" text-anchor="end">GPU-hours (A100)</text></svg>')
    return "".join(out)


CSS_EXTRA = """
.currsvg{width:100%;height:auto}.currsvg .lbl{fill:var(--fg);font-size:15px}.currsvg .bar{fill:#fff;font-size:12.5px;font-weight:600}
.currsvg .ax{fill:var(--fg2);font-size:12px}table.curr td,table.curr th{font-size:.84rem;white-space:normal}
table.curr td:first-child{min-width:190px}.legend span{display:inline-block;margin-right:16px}
.legend i{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:5px}
.method{border-left:3px solid var(--acc);padding:2px 0 2px 14px;margin:1.4em 0}.method h3{margin:.2em 0 .3em}
.method .where{color:var(--fg2);font-size:.88rem}pre.eq{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 12px;overflow-x:auto;font-size:.88rem}
"""


def _method(id_, title, body, where) -> str:
    return (f'<div class="method" id="{id_}"><h3>{title}</h3>{body}'
            f'<p class="where">In the code: {where}</p></div>')


def explainer() -> str:
    m = []
    m.append(_method("m-target", "What the network predicts: a normalized 6-hour tendency", """
<p>The network never outputs the next state itself. It outputs the <b>change over 6 hours</b>, divided by that change's
standard deviation over the training years (computed per channel: <code>res_std</code>). The next state is
<code>x + tendency × res_std</code>, added outside the network. Two consequences: a network that outputs zeros is
persistence, a sensible starting point; and every channel's error counts on the same scale, so surface pressure (Pa)
and humidity (kg/kg) are weighed alike. Inputs are normalized the same way, by each channel's mean and standard deviation.</p>
<p>From v2 on, the network also outputs 10 <b>diagnostics</b> of the step that just happened (precipitation, latent and
sensible heat flux, surface and top-of-atmosphere radiation, moisture advection), normalized by their own statistics.
They are learned targets in the loss, they give the correctors the fluxes they need, and they are extra signals to
look for inside the network.</p>""", "<code>Emulator.physical</code>, <code>target_of</code> in <code>train.py</code>, <code>data.py</code> (stats)"))
    m.append(_method("m-mse", "One-step training with an area-weighted MSE", """
<p>Stage 1 of every version is one-step training: draw a random time from the training years, predict 6 hours ahead,
and minimize the mean squared error against E3SM's actual tendency. Each grid cell is weighted by cos(latitude), so the
crowded polar rows of the 1° grid count for their area, not their row count:</p>
<pre class="eq">L = mean over channels c, cells (i,j) of  w_i · (ŷ_cij − y_cij)²,   w_i = cos φ_i / mean(cos φ)</pre>
<p>This stage takes most of the epochs. It teaches the model the physics of one step, but nothing about what happens
when its own slightly wrong output becomes its next input.</p>""", "<code>loss_fn</code> in <code>train.py</code>; <code>area_w</code> in <code>emulator.py</code>"))
    m.append(_method("m-rollout", "Rollout fine-tuning: training on the model's own forecasts", """
<p>An emulator is used autoregressively: its output is its next input. Errors it never saw in training (slightly too
smooth fields, small biases) compound, and a purely one-step model drifts or blows up. Rollout fine-tuning feeds the model
its own predictions for <i>n</i> steps and puts the loss on every step against the true trajectory:</p>
<pre class="eq">x̂₀ = x₀,   x̂ₜ₊₁ = x̂ₜ + f(x̂ₜ, forcingₜ),   L = (1/n) Σₜ ℓ(x̂ₜ₊₁, xₜ₊₁)</pre>
<p>Gradients flow back through all <i>n</i> steps, so the model learns to make outputs that remain good inputs. The cost
is memory (n copies of every activation): we halve the per-GPU batch as n doubles (8 → 4 → 2) and lower the learning
rate at each stage (3e-4 → 5e-5 → 2e-5 or 3e-5), since the model is now being refined, not built. The forcings
(insolation, surface height, land, ocean and sea-ice fractions) come from the data at every step. This was the largest single improvement in v1 and v2.</p>""", "<code>loss_fn</code> in <code>train.py</code>; <code>--n-steps</code>, <code>--init</code>"))
    m.append(_method("m-corr", "ACE-style correctors inside the loss (v2 on)", """
<p>After every step, four small, differentiable fixes are applied to the network's output, in this order, mirroring Ai2's ACE
(<code>fme.core.corrector.atmosphere</code>); global means are area-weighted:</p>
<ul>
<li><b>positivity:</b> water cannot be negative: total water at every level (and the non-negative diagnostics) is clipped at zero;</li>
<li><b>dry air:</b> the global mass of dry air is constant: one global offset to every column's dry-air pressure
(surface pressure minus the weight of its water) pins its global mean to the forecast's starting value, and surface pressure is solved back from it;</li>
<li><b>moisture budget:</b> the global change in total water must equal evaporation minus precipitation: precipitation is rescaled to make the
global budget hold, and moisture advection is recomputed as each column's residual, so every column's budget closes exactly;</li>
<li><b>total energy:</b> the global change in column energy (c<sub>v</sub>T + L<sub>v</sub>q + gz) must equal the net of the predicted surface and
top-of-atmosphere fluxes plus 0.6 W/m² of heating the emulator cannot see: one temperature offset, the same at every level and column, closes it.</li>
</ul>
<p>The loss is taken <b>after</b> the correctors, so the network learns to produce states the corrections barely have to touch.
The correctors are outside the network: they never appear in hooks, circuits or LAWFUL measurements. They also explain the
long-run drift: with the energy budget enforced, a model that rains too little can only keep energy by warming less, so it
drifts moist and cold.</p>""", "<code>vit_interp/corrector.py</code>; <code>--corrector</code>, <code>--unaccounted-heating</code>"))
    m.append(_method("m-ema", "EMA weights and input noise (v2 on)", """
<p><b>Exponential moving average:</b> alongside the trained weights θ the trainer keeps θ̄ ← 0.999 θ̄ + 0.001 θ after every
step, an average over roughly the last thousand steps. Validation, checkpoints and every released model use θ̄: averaging
smooths away the last few noisy updates and typically gives steadier long rollouts.</p>
<p><b>Input noise:</b> before each step the model's input is perturbed by Gaussian noise of 2% of each field's standard
deviation (1% in rollout stages, none in the 8-step stage); the target is the true next state. This teaches the model to
pull slightly wrong states back toward the attractor, a cheap stand-in for the errors it will meet in its own rollouts.</p>""",
        "<code>--ema</code>, <code>--input-noise</code> in <code>train.py</code>"))
    m.append(_method("m-noise", "A stochastic model: noise-conditioned LayerNorms (v3, v4)", """
<p>A deterministic emulator, trained with MSE, learns the <i>average</i> of the futures consistent with its input. After a few
days that average is blurry, and the blur feeds back into the next steps. A stochastic emulator instead draws <i>one</i> plausible
future per run. We follow ACE2S: every LayerNorm in the network (two per block plus the final one) becomes</p>
<pre class="eq">LN(x) · (1 + W_scale z) + W_bias z,     z ~ N(0, I), one 32-dimensional vector per token, drawn fresh at every step</pre>
<p><code>W_scale</code> and <code>W_bias</code> start at zero, so the new model is exactly the deterministic one it was started from,
and the noise only gains influence as training finds it useful. The residual stream, attention and MLPs are unchanged, so every
interpretability tool works as before; the noise is one more input (<code>hook_noise</code>), and what it does to each norm is
visible at <code>hook_cond_scale</code>/<code>hook_cond_bias</code>.</p>""", "<code>ConditionalLayerNorm</code> in <code>model.py</code>; <code>--noise-dim</code>, <code>--noise-type</code>"))
    m.append(_method("m-crps", "Training an ensemble with the fair CRPS (v3, v4)", """
<p>MSE would teach the stochastic model to ignore its noise (the mean is the MSE-optimal answer). Instead each training sample is run
as <b>2 members</b> with independent noise, and the loss is the continuous ranked probability score, a proper score for a
whole distribution, evaluated per grid cell and channel and area-weighted:</p>
<pre class="eq">CRPS = E|X − y| − ½ E|X − X′|     (fair estimator: the second expectation over distinct members only)</pre>
<p>The first term rewards members close to the truth; the second rewards members that differ from each other. A model that collapses
its members loses the second term; one that spreads them too much pays in the first. Its minimum is reached when the members are
draws from the true distribution of outcomes, so it calibrates the ensemble: v3's spread/skill ratio is 0.92–0.98 (1 is perfect).
The "fair" form removes the bias of a 2-member estimate. ACE adds a small spectral energy score; we left it out (CRPS is the part
that calibrates, and our spectral transform is not ACE's). In rollout stages the members evolve independently from the same
start, and each step's CRPS is taken against the truth.</p>""", "<code>vit_interp/ensemble.py</code>; <code>--loss ensemble --n-ensemble 2 --crps-weight 1 --energy-weight 0</code>"))
    m.append(_method("m-last", "Long rollouts with the gradient on the last step only (v4)", """
<p>Backpropagating through all steps limits rollouts to about 8 steps (2 days) on our GPUs. v4 uses the <b>pushforward trick</b>
(ACE's <code>optimize_last_step_only</code>): each batch rolls out a <b>random length between 4 and 16 steps</b> (1–4 days),
running every step but the last without gradients, and puts the loss on the last step only. Memory stays at one step's
worth, whatever the length. The model still learns what its own day-3 states look like (they are its inputs), but it is
trained only to make the next step from them good, not to steer the earlier steps. Random lengths make every lead within
the range a training lead. With the stochastic model and CRPS, long horizons stay sharp rather than averaging to blur.
v4 starts from v3's one-step checkpoint and goes straight to these long rollouts, skipping v3's 4- and 8-step stages; it
holds a realistic climate longest (144 days).</p>""", "<code>loss_fn</code> in <code>train.py</code>; <code>--min-steps 4 --n-steps 16 --grad-last-only</code>"))
    m.append(_method("m-opt", "Optimizer, schedule and hardware (all versions)", """
<p>AdamW (β = 0.9, 0.95; weight decay 0.05), gradient norm clipped at 1.0, bfloat16 autocast. The learning rate warms up linearly
over the first <code>warmup</code> steps, then follows a cosine to zero over the stage's planned epochs; it restarts at each stage.
Every stage ran data-parallel on one Swing node (4 × A100 40 GB), so the global batch is 4× the per-GPU batch. An epoch is one
pass over all training start times (14,600 for 2000–2009; 43,800 for 1980–2009).</p>""", "<code>train.py</code> (<code>AdamW</code>, <code>LambdaLR</code>); <code>scripts/*.pbs</code>"))
    m.append(_method("m-select", "How checkpoints were chosen, and what that means for \"unseen\"", """
<p>After each epoch the model (EMA weights) is scored on <b>one-step</b> forecasts from every 7th time of <b>2010</b> (209 starts),
and a 5-day (v1) or 10-day rollout every 15 days of 2010 is logged; <code>best.pt</code> is the epoch with the lowest one-step validation
loss (MSE, or CRPS for v3/v4), and each stage starts from the previous stage's <code>best.pt</code>. 2010 never produced a
gradient, but it did pick checkpoints, and it is also the evaluation year, so the 2010 scores are mildly optimistic (a handful
of choices per stage, not thousands of parameters).</p>
<p>The one-step criterion also has a known bias: rollout training trades a little one-step accuracy for much better multi-day
behavior, so in rollout stages the best one-step epoch is usually the first (v2 and v3 kept epoch 1 of their 4-step stage).
That is why we judged v2's and v3's 8-step extensions by full evaluations, not by this number: v2's extension holds the climate
for 122 days (vs 114), and v3's for 115 (vs 88) and beats the release at every lead, despite a slightly worse validation CRPS.</p>""", "<code>validate</code> in <code>train.py</code>; <code>--val-stride</code>, <code>--rollout-steps</code>"))
    return "".join(m)


def training_page(runs: Path, out: Path, page, eval_lookup=None) -> None:
    """Write out/training.html. ``page(title, body, up)`` is the site's page wrapper."""
    legend = ('<p class="legend muted"><span><i style="background:#2a78d6"></i>one step</span>'
              '<span><i style="background:#1baf7a"></i>4-step rollouts</span><span><i style="background:#eda100"></i>8-step rollouts</span>'
              '<span><i style="background:#eb6834"></i>4–16 random steps, last-step gradient</span>'
              '<span><i style="background:#888;opacity:.35"></i>inherited from an earlier version</span>'
              '<span><i style="border:1.5px dashed #888;background:transparent"></i>after the release</span>'
              '<span>▼ released checkpoint</span></p>')
    tables = "".join(f'<h3 id="{v}">{v if v != "3D" else "3D (ViViT-style), in training"}</h3>{curriculum_table(runs, v)}' for v in LINEAGE)
    body = f"""<style>{CSS_EXTRA}</style>
<nav class="muted"><a href="#curriculum">Curriculum</a><a href="#methods">Methods</a><a href="#tables">Exact stages</a></nav>
<h1>How the models were trained</h1>
<p class="lede">Every version is the same network (4×4 patches, 8 pre-LayerNorm blocks × 6 heads, dim 384) trained differently.
This page gives the exact curriculum of each, read from the training runs themselves, and explains what each method does
and why it is there.</p>
<h2 id="curriculum">The curriculum at a glance</h2>
<p>Each bar is a training stage, sized by the compute it took. All versions start with long one-step training and then fine-tune
on rollouts of the model's own forecasts; v3 continues from v2's one-step stage, and v4 from v3's. Hover a bar for details.</p>
{curriculum_svg(runs)}
{legend}
<div class="grid">
<div class="card"><b>v1</b>: 2000–2009, MSE. One step × 40 epochs → 4-step rollouts × 8 epochs (in two stages) → 8-step × 4.</div>
<div class="card"><b>v2</b>: 1980–2009, + diagnostics, correctors, EMA, input noise, output conv. One step × 36 → 4-step × 3 → 8-step × 1 (+ 6 later, not released).</div>
<div class="card"><b>v3</b>: v2's one-step model made stochastic, CRPS on 2 members. + one step × 12 → 4-step × 3 → 8-step × 1 (+ 6 later, not released).</div>
<div class="card"><b>v4</b>: v3's one-step model → 4–16 random steps, last-step gradient × 4.</div>
<div class="card"><b>v7</b>: v4 → + spectral energy score, 4–24 steps × 4 (v5) → + patch jitter in training × 4 (v6) → + anti-aliased embedding × 4.</div>
<div class="card"><b>3D</b>: one token per level; split horizontal/vertical heads, rotary positions. One step × 20 → v7's rollout recipe × 8 (in training).</div>
</div>
<h2 id="methods">The methods, one by one</h2>
{explainer()}
<h2 id="tables">Exact stages</h2>
<p>Read from each stage's <code>args.json</code> and <code>metrics.jsonl</code> under <code>/lcrc/globalscratch/ac.ngmahfouz/vit-interp/runs/</code>.
"Starts from" is the checkpoint the stage was initialized from (the earlier stage's best, or a snapshot of a given epoch).
Epochs read "x of y" where a stage was stopped before its plan (the cosine schedule was set for y). Global batch = per-GPU × 4.
Every stage validated on 2010. Validation losses compare within a version's MSE stages or within its CRPS stages, not across them
(different scores), and v1's are on 34 channels while v2–v4's include the 10 diagnostics. v4 starts from <code>snap-for-v4.pt</code>, v3's one-step model at epoch 12 (the same weights as that stage's best).</p>
<p class="muted">* GPU-hours of training steps (4 A100s × median step time from the logs × steps), excluding validation, so the
wall-clock was ~10–25% longer. v1's 4-step training ran as two stages because the first (<code>stage2</code>) hit its 2-hour
limit after 2 of 4 epochs; an abandoned 4-step attempt from epoch 19 of stage 1 (<code>ft/ft4</code>) is not part of any release.</p>
{tables}"""
    (out / "training.html").write_text(page("How the models were trained", body, up="./"))
