# vit-interp

A close-to-vanilla Vision Transformer that steps E3SMv3 (1°, AMIP) forward 6 hours, built to be taken
apart: attention heads, the residual stream, edge circuits (Vi-CD), and physical-law tests (LAWFUL), with
[exege](https://github.com/mahf708/exege).

**Start here:** [`HACKATHON.md`](HACKATHON.md) (what to do in the first ten minutes) ·
**Results, architecture explorer, executed notebooks:** https://web.lcrc.anl.gov/public/e3sm/diagnostic_output/ac.ngmahfouz/vit-interp/ ·
**Agents:** [`AGENTS.md`](AGENTS.md)

| What | Where |
|---|---|
| Checkpoints (v1, v2) + normalization stats | https://huggingface.co/mahf708/vit-interp-e3sm |
| 2010 input states (the evaluation year) | https://huggingface.co/datasets/mahf708/vit-interp-e3sm-2010 |
| Latent archives + sparse autoencoders (exege format) | https://huggingface.co/datasets/mahf708/vit-interp-e3sm-latents |
| exege with the `circuits` and `laws` layers | https://github.com/mahf708/exege, branch `mahf708/circuits` |

## Install

**On LCRC (Swing):** nothing to install. Use the shared environment and data:

```console
$ export PATH=/home/ac.ngmahfouz/vit-interp/.venv/bin:$PATH
$ /home/ac.ngmahfouz/vit-interp/.venv/bin/python -m ipykernel install --user --name vit-interp   # once, for Jupyter
```

**Anywhere else** (Python 3.11+, a GPU recommended; CPU works for single steps):

```console
$ git clone https://github.com/mahf708/vit-interp && cd vit-interp
$ pip install -e .            # pulls torch, exege (circuits/laws branch), huggingface_hub
```

The first `load()` downloads the released checkpoint (~260 MB) and the 2010 inputs (~15 GB, once) into
`~/.cache/vit-interp` (`$VIT_INTERP_CACHE` moves it; `$VIT_INTERP_DATA` points at a copy you already have).
Exege's latent archives need no download at all: `exege latents info hf://datasets/mahf708/vit-interp-e3sm-latents/v2-2010-jan`.

## Use

```python
from vit_interp import playground as pg
em, meta = pg.load()                       # release v2 (pg.load("v1") for the first model)
x = pg.state("2010-01-15T00")              # (1, 39, 180, 360) physical state
nxt, tend = pg.step(em, x)                 # next state and 6-h tendency
maps, table = pg.direct_attribution(em, x, "V_3", pg.box(30, 60, 150, 210))
```

```console
$ exege circuits prune --adapter vit-interp --source v2 --field V_3 --lat 45 --lon 180 \
    --base 2010-01-15T00 --source-starts 2010-01-20T00 --tau 0.02
$ exege laws rossby --adapter vit-interp --source v2 --u U_2,U_3,U_4 --v V_2,V_3,V_4 --observe V_3 \
    --ps PS --phi0 45 --starts 0,60,120,180 --score correlation
```

Notebooks: [`notebooks/`](notebooks/) (00 quickstart … 07 exege circuits and laws).

## Layout

| Path | Holds |
|---|---|
| `vit_interp/model.py` | the ViT, with TransformerLens-style hook points |
| `vit_interp/emulator.py` | normalization, steps, rollouts, checkpoints |
| `vit_interp/playground.py` | one-line helpers behind the notebooks |
| `vit_interp/circuits.py`, `rossby.py` | edge patching and the Rossby probe (in torch) |
| `vit_interp/exege_adapter.py` | the model as exege's `vit-interp` / `vit-interp-tokens` adapters |
| `vit_interp/data.py`, `train.py`, `evaluate.py`, `corrector.py` | data prep, training, evaluation, ACE-style correctors |
| `scripts/` | PBS jobs, the results-page and explorer builders |

## Stochastic models (v3, v4)

`pg.load("v3")` / `pg.load("v4")` load the stochastic models: every LayerNorm is noise-conditioned,
`LN(x) * (1 + W_scale(z)) + W_bias(z)` with per-token noise `z` (as ACE2S's ConditionalLayerNorm,
zero-initialized), trained with CRPS on 2-member ensembles. The residual stream and every hook are
unchanged; new hooks `hook_noise`, `{ln}.hook_cond_scale`, `{ln}.hook_cond_bias`.

- `em.step(x)` draws fresh noise on every call; pass `generator=` to make it reproducible, or
  `noise=torch.zeros(1, 4050, em.cfg.noise_dim)` for the noise-free path. `em.rollout(..., seed=s)`,
  `em.ensemble(x, n, seed)`; `python -m vit_interp.evaluate --members N`.
- exege adapters draw one noise field from their `seed` option and share it across every patched run,
  so patching stays exact and responses are measured under common random numbers.
- v4 adds `--min-steps` (random rollout length) and `--grad-last-only` (gradient through the last step).
