# vit-interp

A close-to-vanilla Vision Transformer emulator of E3SMv3 (1°, AMIP, 6-hourly), built to be
taken apart: edge-level circuits after *Seeing Through Circuits* (Vi-CD), law-aligned
patching after *LAWFUL* aimed at Rossby waves, and latent archives for
[exege](https://github.com/mahf708/exege).

## The model

`x_t` (34 prognostic + 5 forcing channels, normalized) → 4×4 patch embedding + learned
position embedding (4050 tokens on a 45×90 grid of 4° patches) → 8 pre-LN blocks
(6 heads × 64, MLP ×4, GELU) → LayerNorm → linear unpatchify → normalized 6-hour
tendency of the prognostic channels. There is no windowing, no relative position bias and
no spectral layer, so every token mixing happens in attention and every component writes
additively into the residual stream. Latitude reaches the model only through the
position embedding: where β lives is a question the circuits can answer.

Prognostic: `PS, TS, T_0..7, U_0..7, V_0..7, specific_total_water_0..7`. Forcing:
`SOLIN, PHIS, LANDFRAC, OCNFRAC, ICEFRAC`. In rollouts TS over the ocean is prescribed.

Data: `/lcrc/group/e3sm/ac.ngmahfouz/fme/amip_151/traindata` (monthly netCDF4). The
sibling `amip151_1951_2015.zarr` is incomplete: unwritten chunks read as NaN, so don't use it.
Train on 2000–2009, validate on 2010.

## Running (Swing, PBS)

```console
$ qsub -v RUN=vit-p4-d384-l8 scripts/train.pbs      # prep (once) + stage 1 (1-step) + stage 2 (4-step)
$ tail -f logs/*.OU
```

Runs land in `/lcrc/globalscratch/ac.ngmahfouz/vit-interp/runs/$RUN/stage{1,2}/{best,latest}.pt`,
with `metrics.jsonl` (1-step and 5-day RMSE for a few channels per epoch).

## Hooks

```python
from vit_interp.emulator import Emulator
em, _ = Emulator.load(ckpt, Stats(), lat)
nxt, d = em.step(x_phys, cache=cache := {}, names={"blocks.3.attn.hook_pattern"})
em.step(x_phys, hooks={"blocks.2.attn.hook_result": lambda r: r * mask})
```

Names follow TransformerLens (`blocks.{i}.attn.hook_{q,k,v,pattern,z,result}`,
`blocks.{i}.hook_{resid_pre,attn_out,resid_mid,mlp_out,resid_post}`,
`blocks.{i}.mlp.hook_{pre,post}`, `hook_embed`, `hook_pos_embed`). Asking for
`hook_pattern` switches that layer from flash attention to explicit attention (4050² per head).

## Circuits (Vi-CD) — `vit_interp/circuits.py`

The reduced graph uses 57 senders (embed, 48 heads, 8 MLPs) and 17 receivers (each block's
attention input, its MLP input, and the output), which gives 513 edges. `edge_forward` rebuilds every
receiver's input from clean/corrupted sender outputs; it is exact at both ends (tests).
`eap_scores` gives EAP, and `prune` gives the greedy activation-patching pass. A metric is any
scalar of the output tendency, e.g. `box_metric(PROG_NAMES.index("V_3"), lat_sl, lon_sl)`.
The open hackathon question is what "corrupted" means: a counterfactual weather state
(another day's state in the region, the climatology, the wave removed) plays the role of
Vi-CD's inpainted image.

## Rossby waves (LAWFUL) — `vit_interp/rossby.py`

Plant a small, balanced, non-divergent wave ψ = A·G(φ)·cos(sλ) on real states, step the
model, and read the phase speed of the response (projection on G·e^{isλ}). Compare the
intrinsic speed c − Ū with −β/(k² + l²) across s = 2..14:

```python
rows, M = rossby.sweep(em, X, lat, lon, rossby.Wave(phi0=45, levels=(2, 3, 4, 5, 6)))
rossby.sufficiency_necessity(em, X, lat, lon, comps=[("a", 5, 2), ("m", 5)],
                             all_comps=rossby.components(8, 6))
```

## exege

```console
$ python -m vit_interp.export RUN/stage2/best.pt /lcrc/.../archives/vit-2010-jan --n 120 --embed
$ exege latents info /lcrc/.../archives/vit-2010-jan
$ exege nn sae /lcrc/.../archives/vit-2010-jan --out .../bases/sae.npz
$ exege app --latents /lcrc/.../archives/
```

## Tests

`.venv/bin/python -m pytest -q tests`
