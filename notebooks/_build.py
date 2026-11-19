"""Write the hackathon notebooks (output-free). Edit the cells here, then:

    .venv/bin/python notebooks/_build.py
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

SETUP = '''%matplotlib inline
import sys, os
from pathlib import Path
try:
    import vit_interp
except ImportError:  # not installed: use this checkout, $VIT_INTERP_REPO, or the shared one
    for repo in (Path.cwd().parent, Path(os.environ.get("VIT_INTERP_REPO", "/home/ac.ngmahfouz/vit-interp"))):
        if (repo / "vit_interp").is_dir():
            sys.path.insert(0, str(repo)); break
import numpy as np, torch, matplotlib.pyplot as plt
from vit_interp import playground as pg
em, meta = pg.load()      # or pg.load("/path/to/checkpoint.pt"); $PLAYGROUND_CKPT also works
FAST = pg.FAST            # PLAYGROUND_FAST=1 -> small settings (CPU tests)'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "vit-interp", "display_name": "vit-interp (ViT emulator)",
                                 "language": "python"}
    nbf.write(nb, HERE / name)


# ---------------------------------------------------------------------------------------------
write("00_quickstart.ipynb", [
    md("""
# 00 · Quickstart: meet the emulator

A close-to-vanilla **Vision Transformer** trained to step E3SMv3 (1°, AMIP) forward by 6 hours.

* **Input:** 34 prognostic fields (PS, TS, and T, U, V, total water on 8 levels; level 0 is the model top, 7 is near the surface) plus 5 forcings (SOLIN, PHIS, land/ocean/ice fractions), normalized.
* **Tokens:** 4×4-degree patches, so 45 × 90 = **4050 tokens**, each with a learned position embedding. The embedding is the only way the model knows where it is on the globe.
* **Body:** 8 pre-LayerNorm blocks with 6 heads each and a GELU MLP. There is no windowing and no spectral layer, so all horizontal mixing is attention.
* **Output:** the 6-h **tendency** of the 34 fields. Next state = current state + tendency.

Every head and MLP writes additively into the residual stream, which makes it a good subject for circuit hunting. Trained on 2000–2009; everything here uses **2010**, which it never saw.
"""),
    code(SETUP),
    md("## One step, against the truth"),
    code('''
t0 = "2010-01-15T00"
x = pg.state(t0)                                  # (1, 39, 180, 360), physical units
nxt, tend = pg.step(em, x)                        # emulated next state and 6-h tendency
tr = pg.truth(t0, steps=1)[0, :, :34]             # E3SM states at t0 and t0 + 6 h
true_tend = (tr[1] - tr[0]).to(tend.device)
pg.plot_maps({"ViT: V_3 tendency [m/s / 6 h]": tend[0, pg.ch("V_3")],
              "E3SM: V_3 tendency": true_tend[pg.ch("V_3")],
              "ViT − E3SM": tend[0, pg.ch("V_3")] - true_tend[pg.ch("V_3")]}, shared=True, cols=3);
'''),
    md("How much better than persistence (tendency = 0) is one step, field by field?"),
    code('''
w = torch.tensor(np.cos(np.deg2rad(meta["lat"])))[:, None].to(tend.device)
def rmse(a): return float(((a ** 2) * w).sum() / (w.sum() * a.shape[-1])) ** 0.5
rows = []
for name in ["PS", "TS", "T_7", "T_3", "U_3", "V_3", "U_6", "specific_total_water_7"]:
    c = pg.ch(name)
    rows.append((name, rmse(tend[0, c] - true_tend[c]), rmse(true_tend[c])))
import pandas as pd
pd.DataFrame(rows, columns=["field", "RMSE ViT", "RMSE persistence"]).assign(
    skill=lambda d: 1 - d["RMSE ViT"] / d["RMSE persistence"]).round(4)
'''),
    md("## A short forecast: where do errors grow?"),
    code('''
steps = 4 if FAST else 20                         # 20 steps = 5 days
emu, tru = pg.rollout(em, t0, steps)
lead_h = np.arange(steps + 1) * 6
fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for ax, name in zip(axes, ["PS", "T_7", "U_3", "V_3"]):
    c = pg.ch(name)
    err = [rmse((emu[k, c] - tru[k, c]).to(w.device)) for k in range(steps + 1)]
    pers = [rmse((tru[0, c] - tru[k, c]).to(w.device)) for k in range(steps + 1)]
    ax.plot(lead_h, err, label="ViT"); ax.plot(lead_h, pers, "--", label="persistence")
    ax.set_title(name); ax.set_xlabel("lead [h]")
axes[0].legend();
'''),
    code('''
k = steps
pg.plot_maps({f"ViT U_3 at +{6*k} h": emu[k, pg.ch("U_3")], f"E3SM U_3 at +{6*k} h": tru[k, pg.ch("U_3")],
              "difference": emu[k, pg.ch("U_3")] - tru[k, pg.ch("U_3")]}, cols=3, symmetric=False, cmap="viridis");
'''),
    md("""
## Is it Earth?

A full-year free run with climatology maps, zonal means, drift and skill against persistence and climatology lives in the evaluation folders next to the checkpoints (`runs/vit-p4-d384-l8/eval-*`). The cell below shows the latest summary.
"""),
    code('''
evals = sorted(pg.RUN.glob("eval-*/summary.md"), key=os.path.getmtime)
print(evals[-1].read_text() if evals else "no evaluation yet")
'''),
    md("""
## Try this

1. Pick another date (`pg.state("2010-07-20T12")`). Is the step worse in the summer hemisphere, or over the tropics?
2. Which fields does the ViT beat persistence on by the most, and which by the least? Look at the `skill` column for moisture and TS over land.
3. Plot the error at +5 days for `specific_total_water_7`. Where does it grow first: storm tracks, the ITCZ, or the poles?
4. 4°×4° patches: do you see **patch-edge artifacts** in the tendency maps? (Zoom in with `plt.xlim`/`plt.ylim`.)
"""),
])

# ---------------------------------------------------------------------------------------------
write("01_attention_atlas.ipynb", [
    md("""
# 01 · Attention atlas: where does each head look?

In a vanilla ViT, attention is the only way information moves between places. A head's **attention pattern** says, for every query token, how much it reads from every other token. This notebook does two things:

* builds an **atlas** of all 48 heads: how far they look, in which direction, and how focused they are;
* plots single patterns for chosen query points.

In the atlas, **dx < 0** means the head reads from the **west**. In midlatitude westerlies that is *upstream*, where tomorrow's weather comes from.
"""),
    code(SETUP),
    code('''
xs = torch.cat([pg.state(t) for t in (["2010-01-15"] if FAST else ["2010-01-15", "2010-04-15", "2010-07-15", "2010-10-15"])])
atlas = pg.head_atlas(em, xs)            # one row per (layer, head, latitude band)
atlas.round(2).head(12)
'''),
    md("A heat map per statistic, for the Northern Hemisphere midlatitudes (30–60°N):"),
    code('''
band = "NH mid"
fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
for ax, (stat, cmap) in zip(axes, [("dx", "RdBu_r"), ("dy", "RdBu_r"), ("dist", "viridis"), ("entropy", "magma")]):
    t = atlas[atlas.band == band].pivot(index="layer", columns="head", values=stat)
    v = np.abs(t.values).max()
    im = ax.imshow(t.values, cmap=cmap, **({"vmin": -v, "vmax": v} if cmap == "RdBu_r" else {}))
    for (i, j), val in np.ndenumerate(t.values):
        ax.text(j, i, f"{val:.0f}" if stat != "entropy" else f"{val:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"{stat} ({band})"); ax.set_xlabel("head"); ax.set_ylabel("layer"); plt.colorbar(im, ax=ax)
fig.tight_layout()
'''),
    md("Heads that look **upstream** (most negative dx) in the NH midlatitudes, and the most **local** ones:"),
    code('''
nh = atlas[atlas.band == "NH mid"]
display(nh.nsmallest(5, "dx")[["layer", "head", "dx", "dy", "dist", "entropy"]].round(2))
display(nh.nsmallest(5, "dist")[["layer", "head", "dx", "dy", "dist", "entropy"]].round(2))
'''),
    md("## One head, one query point"),
    code('''
west = nh.nsmallest(1, "dx").iloc[0]
L, H = int(west["layer"]), int(west["head"])
queries = {"North Atlantic (50N, 30W)": (50, 330), "Tropical Pacific (0, 180)": (0, 180),
           "Southern Ocean (55S, 90E)": (-55, 90), "Near the North Pole (85N, 0)": (85, 0)}
for title, q in queries.items():
    a = pg.attention(em, xs[:1], L, H, q)
    pg.plot_tokens(a, f"L{L}H{H} from {title}", mark=q, log=True)
'''),
    md("Is the pattern set by **position** (the learned embedding) or by **content** (today's weather)? Compare the same head and query on two dates:"),
    code('''
q = (50, 330)
a1 = pg.attention(em, pg.state("2010-01-15"), L, H, q)
a2 = pg.attention(em, pg.state("2010-01-20"), L, H, q)
pg.plot_maps({"Jan 15": np.log10(a1 + 1e-6), "Jan 20": np.log10(a2 + 1e-6)}, symmetric=False, cmap="viridis")
print("overlap (sum of min):", float(np.minimum(a1, a2).sum()))
'''),
    md("""
## Try this

1. Layer 0 heads have entropy close to log(4050) ≈ 8.3, so they spread attention almost uniformly. What could a global-average head be for? (Hint: global means of mass and energy, or the time of day through SOLIN.)
2. Find a head whose dx flips sign between the NH and SH midlatitudes, or between the midlatitudes and the tropics. Do the tropical easterlies show up as heads looking **east**?
3. Does an upstream-looking head look **farther** west in winter, when the jet is faster? Split `xs` into January and July and compare `dx`. If it does, the head is advecting with the flow, not looking at a fixed offset.
4. Plot `dy` near the poles. Do heads wrap across the pole, or treat the grid as a flat rectangle?
5. Pick a head and a query, and compute the attention-weighted mean wind at the tokens it reads. Is the displacement consistent with U × 6 h?
"""),
])

# ---------------------------------------------------------------------------------------------
write("02_logit_lens.ipynb", [
    md("""
# 02 · The lens: when does the model know the answer?

A transformer's residual stream is the same space at every depth. So we can take the stream **after any block** and push it through the final LayerNorm and output head, as if the model stopped there (the "logit lens" of language models).

If the decoded tendency already looks like the final one at block 3, the later blocks are refining it. If it only appears at block 6, that is where the work is done.
"""),
    code(SETUP),
    code('''
x = pg.state("2010-01-15T00")
decoded, table = pg.lens(em, x)
table.round(3)          # RMSE vs the final output, relative to its RMS (0 = already the answer)
'''),
    code('''
fig, ax = plt.subplots(figsize=(7, 3.5))
for g in ["PS/TS", "T", "U", "V", "water"]:
    ax.plot(table["after"], table[g], marker="o", label=g)
ax.set_ylabel("relative error vs final output"); ax.set_xticks(range(len(table))); ax.set_xticklabels(table["after"], rotation=45); ax.legend();
'''),
    md("What the V_3 tendency looks like as it is built up, block by block:"),
    code('''
c = pg.ch("V_3")
show = [3, 5, 6, 8]          # index 0 = embedding only, k = after block k-1
pg.plot_maps({table["after"][k]: decoded[k][0, c] for k in show}, shared=True, cols=2);
'''),
    md("""
## Try this

1. Which field group is "decided" earliest? Is moisture harder (later) than temperature?
2. Run the lens on a **summer** date and a **tropical cyclone** case (look for strong V_3 dipoles in the tropics). Does the depth at which the answer appears change?
3. The lens uses the final LayerNorm on an intermediate stream, which may be unfair. A "tuned lens" fits one linear map per layer. Fit one by least squares on a few dates: does block 2 then decode much better?
4. Compare the lens with ablation (notebook 03): a block whose removal hurts most should be where the lens error drops most. Does that hold?
"""),
])

# ---------------------------------------------------------------------------------------------
write("03_ablation_lab.ipynb", [
    md("""
# 03 · Ablation lab: knock components out

Remove a head or an MLP and see what breaks. Two flavors:

* **mean** ablation (the default): replace the component's output by its mean over all tokens. This keeps its average contribution and removes its spatial pattern.
* **zero** ablation: remove it entirely. This is harsher and can push the model off-distribution.

Effects are on the 6-h **tendency**, in physical units.
"""),
    code(SETUP),
    code('''
x = pg.state("2010-01-15T00")
a = pg.ablate(em, x, names=["L5H2"])           # also: heads=[(5, 2)], mlps=[3], mode="zero"
rc = a.rel_change()
fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(rc.keys(), rc.values()); ax.set_ylabel("|Δ tendency| / |tendency|"); ax.tick_params(axis="x", rotation=90)
'''),
    code('''
pg.plot_maps({"Δ V_3 tendency (L5H2 ablated)": a.delta[pg.ch("V_3")],
              "Δ T_7 tendency": a.delta[pg.ch("T_7")],
              "V_3 tendency (intact)": a.base[pg.ch("V_3")],
              "Δ water_7 tendency": a.delta[pg.ch("specific_total_water_7")]}, cols=2);
'''),
    md("## A sweep: which components matter for one quantity?\nThe metric here is the mean V_3 tendency over a North Atlantic box. Swap in your own."),
    code('''
mask = pg.box(35, 60, 280, 320)           # lat0, lat1, lon0, lon1 (0..360)
metric = lambda t: pg.box_mean(t[0, pg.ch("V_3")], mask)
which = ["L0H0", "L3H1", "L5H2", "L5H4", "L6MLP", "L7MLP"] if FAST else None   # None = all 56
sweep = pg.sweep_components(em, x, metric, which=which)
print("baseline:", round(sweep.attrs["baseline"], 3), "m/s per 6 h")
sweep.head(15).round(3)
'''),
    code('''
s = sweep.set_index("component").effect
order = pg.components(em)[1:]
s = s.reindex([c for c in order if c in s.index])
fig, ax = plt.subplots(figsize=(14, 3)); ax.bar(s.index, s.values); ax.tick_params(axis="x", rotation=90)
ax.set_ylabel("Δ metric"); ax.axhline(0, color="k", lw=0.5);
'''),
    md("""
## Try this

1. Zero vs mean ablation: rerun the sweep with `mode="zero"`. Which components look important only under zero ablation, and why might that be misleading?
2. Ablate **all heads of one layer** (`heads=[(4, h) for h in range(6)]`). Which layer's attention is indispensable?
3. Ablate the most **upstream-looking** head from notebook 01. Does the error appear **downstream** (east) of strong gradients, as if advection were broken?
4. Ablation in a **rollout**: write a hook that ablates L5H2 at every step (`em.rollout(..., hooks_at=lambda t: {...})`) and see how the forecast degrades over 2 days.
5. This is the node-level start of circuit discovery. The `circuits` module and exege's `exege.circuits` do it at the **edge** level (Vi-CD). Pick your metric and compare.
"""),
])

# ---------------------------------------------------------------------------------------------
write("04_direct_attribution.ipynb", [
    md("""
# 04 · Direct attribution: who writes the answer?

The final residual stream is a **sum**: embedding + 48 head outputs + 8 MLP outputs + biases. The output head is linear. The final LayerNorm is not, but freezing its per-token scale at its actual value makes the split linear, and exact for this input. So every component's **direct** contribution to a field's tendency can be drawn as a map, and the maps add up to the model's output.

*Direct* means what a component writes straight into the output, not what it does through later layers. Notebook 03's ablation measures the *total* effect, and comparing the two is informative.
"""),
    code(SETUP),
    code('''
x = pg.state("2010-01-15T00")
mask = pg.box(35, 60, 280, 320)            # North Atlantic
maps, table = pg.direct_attribution(em, x, "V_3", mask)
print("total in box:", round(table.attrs["total"], 3), " reconstruction error:", table.attrs["max_abs_reconstruction_error"])
table.head(10).round(3)
'''),
    code('''
top = table.component.head(6).tolist()
pg.plot_maps({f"{c}: {table.set_index('component').contribution[c]:+.2f}": maps[c] for c in top}, cols=3, shared=True);
'''),
    md("Sum per layer: are heads or MLPs doing most of the writing, and at which depth?"),
    code('''
t = table.set_index("component").contribution
layers = range(em.cfg.depth)
heads = [sum(t[f"L{i}H{h}"] for h in range(em.cfg.heads)) for i in layers]
mlps = [t[f"L{i}MLP"] for i in layers]
fig, ax = plt.subplots(figsize=(7, 3)); ax.bar(np.array(list(layers)) - 0.2, heads, 0.4, label="heads")
ax.bar(np.array(list(layers)) + 0.2, mlps, 0.4, label="MLP"); ax.axhline(0, color="k", lw=0.5); ax.legend(); ax.set_xlabel("layer");
'''),
    md("Direct (this notebook) vs total (ablation) effect, for the same components:"),
    code('''
names = table.component.head(10 if FAST else 20).tolist()
names = [n for n in names if n not in ("embed", "biases")]
metric = lambda tt: pg.box_mean(tt[0, pg.ch("V_3")], mask)
sw = pg.sweep_components(em, x, metric, which=names, mode="zero").set_index("component")
fig, ax = plt.subplots(figsize=(4.5, 4))
d = table.set_index("component").contribution[names]; tot = -sw.effect[names]
ax.scatter(d, tot); [ax.annotate(n, (d[n], tot[n]), fontsize=7) for n in names]
ax.axline((0, 0), slope=1, color="k", lw=0.5); ax.set_xlabel("direct contribution"); ax.set_ylabel("total effect (−Δ under zero ablation)");
'''),
    md("""
## Try this

1. Move the box to a **developing cyclone** or a **trough** (pick one from the V_3 tendency map in notebook 00). Do the same components show up?
2. Attribute **T_7** or **specific_total_water_7** instead. Is moisture written by MLPs (local physics) and winds by heads (transport)?
3. Off-diagonal points in the scatter are components that act **through other components**. Those are where circuits start. Take one, ablate it, and look at which later heads change their output (`em.step(..., cache=..., names={...})`).
4. The embedding term is the "persistence-like" linear map of each patch onto itself. How big is it compared with the heads?
"""),
])

# ---------------------------------------------------------------------------------------------
write("05_exege_latents.ipynb", [
    md("""
# 05 · Latents with exege: regions, PCA and a sparse autoencoder

[exege](https://github.com/mahf708/exege) reads a model's internal activations as a **latent archive**: one array per layer, `(times, nodes, channels)`, on a grid. Here each node is one of the 4050 patch tokens and the channels are the 384 residual-stream dimensions.

This notebook exports a small archive from the ViT, then asks exege:

* which channels respond in a region;
* where else the model "looks the same" (cosine similarity);
* what a sparse autoencoder finds.
"""),
    code(SETUP),
    code('''
import subprocess, shutil
REPO = Path(pg.__file__).resolve().parents[1]
EXEGE = str(Path(sys.executable).with_name("exege"))
OUT = Path("/lcrc/globalscratch") / os.environ["USER"] / "vit-interp-play" / "latents-2010-jan"
def run(*cmd):
    r = subprocess.run([str(c) for c in cmd], cwd=REPO, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:]); r.check_returncode()
ckpt = os.environ.get("PLAYGROUND_CKPT") or pg.default_checkpoint()
n = 12 if FAST else 40
run(sys.executable, "-m", "vit_interp.export", ckpt, OUT, "--n", n, "--layers", "1,4,7", "--embed", "--overwrite")
'''),
    code('''run(EXEGE, "latents", "info", OUT)'''),
    md("## A region: the North Atlantic storm track\nRank the channels that respond most within 1500 km of (50°N, 30°W) and map where else the model looks like this."),
    code('''
from exege.latents import Region, analyze_region, open_source
source = open_source(str(OUT))
layer = source.info().layers[-1].index          # the last block in the archive
res = analyze_region(source, time=0, layer=layer, region=Region(lat=50, lon=-30, radius_km=1500),
                     centered=True, n_components=3)
print("top channels:", res.ranking.channels[:10])
grid = source.grid()
pg.plot_maps({"cosine similarity to the region": grid.to_map(res.similarity),
              "PC 1 fitted in the region": grid.to_map(res.scores[:, 0])}, cols=2);
'''),
    md("The same from the command line, with a reproducible JSON summary:"),
    code('''run(EXEGE, "latents", "region", OUT, "--lat", 50, "--lon", -30, "--radius-km", 1500, "--centered", "--top", 6, "--pcs", 2)'''),
    md("## A sparse autoencoder on one layer\nThis one is tiny and quick, to see the workflow. For real features, export more times and train longer on a GPU."),
    code('''
(OUT / "bases").mkdir(exist_ok=True)
run(EXEGE, "nn", "sae", OUT, "--layer", layer, "--features", 128 if FAST else 512, "--k", 16,
    "--epochs", 1 if FAST else 3, "--out", OUT / "bases" / "sae-last.npz")
run(EXEGE, "latents", "region", OUT, "--lat", 50, "--lon", -30, "--radius-km", 1500,
    "--basis", OUT / "bases" / "sae-last.npz", "--features", 3)
'''),
    md("""
## Try this

1. Launch the app: `exege app --latents <OUT parent>` (on a compute node, then tunnel the port as in the README). Click around regions and layers.
2. Compare layers: is the North Atlantic "similar to" the North Pacific at layer 1 (geography-driven) and less so at layer 7 (weather-driven)?
3. Export a **month** (`--n 124`) and train an SAE on a GPU (`--device cuda --epochs 10`). Find a feature that tracks fronts, the ITCZ or sea ice.
4. Correlate a feature with a physical field: `exege latents fields OUT --basis ... --feature N`. See `exege latents --help`.
5. Hugging Face: archives can be read straight from `hf://datasets/<owner>/<repo>/<folder>`. See the exege docs.
"""),
])

# ---------------------------------------------------------------------------------------------
write("06_rossby_probe.ipynb", [
    md("""
# 06 · Do Rossby waves emerge? A LAWFUL-style probe

Following *LAWFUL* (Chen et al. 2026): pair the model with a physics law and test whether its response to controlled perturbations follows the law.

* **Law:** barotropic Rossby waves on a zonal flow U, with c − U = −β / (k² + l²). Long waves drift westward relative to the flow, short ones less.
* **Perturbation:** plant a small, geostrophically balanced, non-divergent wave of zonal wavenumber s on real states.
* **Bridge:** measure the phase speed of the planted wave in the model's one-step response.
* **Score:** M = 1 − Σ(c_model − c_theory)² / Σ(c_theory − mean)², where 1 is perfect agreement with the law.

The model was never told about β. Latitude reaches it only through the position embedding.
"""),
    code(SETUP),
    code('''
from vit_interp import rossby as R
lat, lon = np.asarray(meta["lat"]), np.asarray(meta["lon"])
X = torch.cat([pg.state(t) for t in (["2010-01-15"] if FAST else ["2010-01-05", "2010-01-15", "2010-01-25", "2010-02-05"])])
rows, M = R.sweep(em, X, lat, lon, R.Wave(phi0=45, sigma=10, v_amp=1.0, levels=(2, 3, 4, 5, 6)),
                  ss=range(2, 8) if FAST else range(2, 15), level=3)
import pandas as pd
df = pd.DataFrame(rows); print("consistency M on the intrinsic speed:", round(M, 3)); df.round(2)
'''),
    code('''
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(df.s, df.c_model, "o-", label="ViT"); ax[0].plot(df.s, df.c_theory, "s--", label="Rossby theory")
ax[0].axhline(df.U_bar[0], color="gray", lw=0.5, label="Ū (advection only)"); ax[0].set_xlabel("zonal wavenumber s"); ax[0].set_ylabel("phase speed [m/s]"); ax[0].legend()
ax[1].plot(df.s, df.growth_per_day, "o-"); ax[1].set_xlabel("s"); ax[1].set_ylabel("growth [1/day]"); ax[1].set_title("does the wave grow or decay?");
'''),
    md("""M can be very negative even when the model has the right **shape**: Rossby-like dispersion with the wrong magnitude or an offset. A linear fit of the model's intrinsic speed against theory separates the two. Slope 1, intercept 0 and r = 1 would be the law exactly; r near 1 with slope ≠ 1 means the right physics, mis-scaled."""),
    code('''
a, b = np.polyfit(df.intrinsic_theory, df.intrinsic_model, 1)
r = np.corrcoef(df.intrinsic_theory, df.intrinsic_model)[0, 1]
print(f"intrinsic speed: model = {a:.2f} x theory {b:+.2f} m/s, r = {r:.3f}")
'''),
    md("## Which components carry it?\nSufficiency: route the perturbation only through a set of components. Necessity: route it through everything else. Here we try the heads of one layer at a time. LAWFUL's τ-consistent circuits need M0 > 0. If M0 ≤ 0 at this checkpoint, read the columns relatively: a smaller `d_suf` means more of the model's own response is routed through that layer."),
    code('''
allc = R.components(em.cfg.depth, em.cfg.heads)
rows = []
for L in ([5] if FAST else range(em.cfg.depth)):
    comps = [("a", L, h) for h in range(em.cfg.heads)]
    r = R.sufficiency_necessity(em, X, lat, lon, comps, allc, R.Wave(phi0=45),
                                ss=range(2, 8) if FAST else range(2, 15))
    rows.append({"layer": L, **{k: round(v, 3) for k, v in r.items()}})
pd.DataFrame(rows)
'''),
    md("""
## Try this

1. Move the wave: `phi0=30` or `phi0=-50` (Southern Hemisphere), or a narrower `sigma`. Theory says β = 2Ω cos φ / a is weaker poleward. Does the model know?
2. Plant at the jet level only (`levels=(2, 3)`) vs through the depth. Is the model's response more barotropic or baroclinic?
3. LAWFUL's greedy search: grow a set of components one at a time until sufficiency ≥ 0.9·M0, then prune it. Use `R.sufficiency_necessity` in a loop. (exege is getting this as `exege.laws`.)
4. The **position embedding** is the only source of latitude. Swap the position embeddings of two latitude rows (a hook on `hook_pos_embed`) and see whether the westward drift follows.
5. Does the wave's growth rate tell you about baroclinic instability (growth for s ≈ 6–8 in winter)?
"""),
])

# ---------------------------------------------------------------------------------------------
write("07_exege_circuits_and_laws.ipynb", [
    md("""
# 07 · Circuits and laws with exege

Two papers, two tools, both run on the real ViT through exege's adapter (`vit-interp`). Nothing here is ViT-specific: exege only sees a system that can be **patched**.

* **Seeing Through Circuits (Vi-CD):** which **edges** (head → head, head → MLP, … → output) does a behaviour run through? We patch edges from a counterfactual run and keep the ones whose loss moves the metric. The graph has 57 senders, 17 receivers and **513 edges**.
* **LAWFUL:** does the model obey a **law** (here barotropic Rossby dispersion), and which components carry that agreement (sufficiency / necessity)?

Command-line equivalents are at the end: `exege circuits …` and `exege laws …`.
"""),
    code(SETUP),
    code("""
from exege.circuits import open_patchable, field_mean, attribute, prune, faithfulness_curve, core, jaccard
from exege.laws import Rossby, Envelope, measure, deviations, search, components
CKPT = os.environ.get("PLAYGROUND_CKPT") or str(pg.default_checkpoint())
system = open_patchable(CKPT, "vit-interp", year=2010)
g = system.graph(); grid = system.grid()
print(len(g.senders), "senders,", len(g.receivers), "receivers,", g.n_edges, "edges")
"""),
    md("""
## A behaviour and its counterfactual

The behaviour is the mean 6-h **V_3** (jet-level meridional wind) over the North Pacific. Its counterfactual is the same metric 5 days later (another weather state). Vi-CD inpaints the object out of an image; for an emulator *which* counterfactual you choose is the scientific question, so try others below.
"""),
    code("""
metric = field_mean(grid, "V_3", where=grid.within(45, 180, 2000))
base = system.states(["2010-01-15T00"])
source = system.states(["2010-01-20T00"])
"""),
    md("## Score every edge (EAP: one gradient), then prune (one patched step per edge)"),
    code("""
a = attribute(system, base, source, metric, method="eap")
print(f"clean {a.clean:.3f}, counterfactual {a.corrupt:.3f}")
import pandas as pd
pd.DataFrame(a.ranked(15), columns=["edge", "score (metric change if this edge alone is patched)"])
"""),
    code("""
c = prune(system, base, source, metric, tau=0.05 if FAST else 0.02)
print(f"{c.n_edges} of {g.n_edges} edges recover {c.recovery:.1%} of the clean-vs-counterfactual gap")
print("components in the circuit:", " ".join(c.components()))
"""),
    md("Does EAP's ranking find faithful circuits? Keep the top-k edges by EAP score and patch the rest (Vi-CD's first experiment):"),
    code("""
sizes = [5, 20, 60, 150] if FAST else [5, 10, 20, 40, 60, 100, 200, 513]
curve = faithfulness_curve(system, base, source, metric, a, sizes)
plt.plot(curve.sizes, curve.recoveries, "o-"); plt.axhline(1, color="k", lw=0.5)
plt.xscale("log"); plt.xlabel("edges kept (top-k by EAP)"); plt.ylabel("recovery"); plt.title("faithfulness vs sparsity");
"""),
    md("""
## Rossby waves: LAWFUL

Plant a small, balanced Rossby wave of zonal wavenumber s = 2…14 at 45°N on several real states, take one step, and read the wave's frequency off the model's response. Compare with ω − kŪ = −βk/(k²+l²).

* `score="eq10"` is LAWFUL's misfit score: 1 means the model matches the law, and negative means it is worse than the law's mean.
* `score="correlation"` asks only whether the model follows the law's **shape** across s.
"""),
    code("""
law = Rossby(u=("U_2", "U_3", "U_4"), v=("V_2", "V_3", "V_4"), observe_field="V_3", ps="PS",
             band=Envelope(45, 10), wavenumbers=range(2, 9) if FAST else range(2, 15))
X = system.states([0, 120] if FAST else [0, 60, 120, 180, 240, 300, 360, 420])
r10 = measure(system, X, law); rc = measure(system, X, law, score="correlation")
print(f"M (eq10) = {r10.M:.2f}    M (correlation) = {rc.M:.3f}")
df = pd.DataFrame(r10.rows())
plt.plot(df.alpha, df.model, "o-", label="ViT"); plt.plot(df.alpha, df.law, "s--", label="barotropic theory")
plt.xlabel("zonal wavenumber s"); plt.ylabel("intrinsic frequency ω − kŪ [rad/day]"); plt.legend();
"""),
    md("Which blocks carry the shape? Sufficiency (`d_suf` small = this block alone reproduces it) and necessity (`d_nec` large = the rest can't do without it), on the correlation score:"),
    code("""
rows = []
for L in ([4, 5, 6] if FAST else range(8)):
    comps = [f"a{L}.h{h}" for h in range(6)] + [f"m{L}"]
    d = deviations(system, X, law, comps, score="correlation")
    rows.append({"block": L, "M_suf": d.M_suf, "M_nec": d.M_nec, "d_suf": d.d_suf, "d_nec": d.d_nec})
pd.DataFrame(rows).round(3)
"""),
    md("LAWFUL's greedy search for τ-consistent circuits (nested, minimal). It is slow-ish: one patched sweep per candidate per step, so a few minutes on a GPU."),
    code("""
if not FAST:
    s = search(system, system.states([0, 120, 240, 360]), law,
               taus=(0.5, 0.7, 0.9), score="correlation", controls=[system.states([60, 180, 300, 420])])
    for circ in s.circuits:
        d = circ.deviations
        print(f"tau {circ.tau}: {len(circ.components)} components  d_suf {d.d_suf:.3f}  d_nec {d.d_nec:.3f}"
              f"  control d_suf {[round(x, 3) for x in d.to_dict().get('d_suf_random', [])]}")
        print("   ", " ".join(circ.components))
"""),
    md("""
## The same from a terminal

```console
$ exege circuits attribute --adapter vit-interp --source $CKPT --field V_3 --lat 45 --lon 180 \\
    --base 2010-01-15T00 --source-starts 2010-01-20T00 --method eap
$ exege circuits prune ... --tau 0.02 --out circuit.json
$ exege laws rossby --adapter vit-interp --source $CKPT --u U_2,U_3,U_4 --v V_2,V_3,V_4 \\
    --observe V_3 --ps PS --phi0 45 --starts 0,60,120,180 --score correlation --search
```

## Try this

1. **Counterfactuals.** Swap the source for the climatology, or for the same day with the trough removed (`system.perturb`), or for a day 6 hours later. Do circuits change?
2. **Stable cores.** Prune for 5 different January days, then `core([...])`. Which edges are always there? (Vi-CD RQ2)
3. **Regions.** Run circuits for tropics vs midlatitudes vs polar boxes and compare with `jaccard`.
4. **Steering level.** `Rossby(..., u_bar="U_5")`: does a lower steering wind close the gap with theory?
5. **Hemispheres and seasons.** `Envelope(-45, 10)`, and July states (`system.states(["2010-07-15T00", ...])`). β is the same, Ū is not.
6. **Errors** (the Nov 4 paper): pick a metric that is the model's *error* (ViT minus E3SM) in a region, and find the circuit that produces it.
"""),
])

print("wrote", sorted(p.name for p in HERE.glob("*.ipynb")))
