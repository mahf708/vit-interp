# Start here: inside a ViT weather emulator

A close-to-vanilla Vision Transformer that steps E3SMv3 (1°, AMIP) forward 6 hours: 4050
patch tokens, 8 blocks of 6 heads, every head and MLP writing into one residual stream.
It is small enough to take apart in an afternoon. Pick a door.

## 1. Notebooks (most people start here)

Eight notebooks, each a few lines per cell, each ending in a **Try this** list. Setup
(kernel, GPU node, tunnel) is in [`notebooks/README.md`](notebooks/README.md).

| | |
|---|---|
| `00_quickstart` | one step and a 5-day forecast against E3SM |
| `01_attention_atlas` | where each of the 48 heads looks: upstream? local? global? |
| `02_logit_lens` | decode every block: when does the model know the answer? |
| `03_ablation_lab` | knock out heads and MLPs; what breaks, where |
| `04_direct_attribution` | who writes the jet-level wind over a box |
| `05_exege_latents` | latent archives, region analysis, sparse autoencoders |
| `06_rossby_probe` | plant a Rossby wave; does the model know β? |
| `07_exege_circuits_and_laws` | Vi-CD edge circuits and LAWFUL, through exege |

Just want to look? Executed copies with every figure:
`/lcrc/globalscratch/ac.ngmahfouz/vit-interp/notebooks-executed/`.

## 2. One-liners (no Python)

On a GPU node, with `export PATH=/home/ac.ngmahfouz/vit-interp/.venv/bin:$PATH` and
`CKPT=/lcrc/globalscratch/ac.ngmahfouz/vit-interp/runs/vit-p4-d384-l8/release/v1.pt`:

```console
$ exege circuits graph --adapter vit-interp --source $CKPT
$ exege circuits attribute --adapter vit-interp --source $CKPT --field V_3 --lat 45 --lon 180 \
    --base 2010-01-15T00 --source-starts 2010-01-20T00 --method eap
$ exege circuits prune   --adapter vit-interp --source $CKPT --field V_3 --lat 45 --lon 180 \
    --base 2010-01-15T00 --source-starts 2010-01-20T00 --tau 0.02 --out circuit.json
$ exege laws rossby --adapter vit-interp --source $CKPT --u U_2,U_3,U_4 --v V_2,V_3,V_4 \
    --observe V_3 --ps PS --phi0 45 --starts 0,60,120,180 --score correlation
$ exege laws rossby ... --search --by-block --control 30,90,150,210   # which blocks carry it
```

Each runs in seconds to a minute on one A100 (a prune is 513 patched steps, about 45 s).
`--help` on any of them lists every option; `--out x.json` keeps the numbers.

## 3. Python, your own way

```python
from vit_interp import playground as pg                 # helpers behind the notebooks
em, meta = pg.load(CKPT)
from exege.circuits import open_patchable               # the paper methods, model-agnostic
system = open_patchable(CKPT, "vit-interp", year=2010)
from exege.latents import open_intervenable             # steering in token space
tokens = open_intervenable(CKPT, "vit-interp-tokens")
```

Hooks follow TransformerLens names (`blocks.{i}.attn.hook_pattern`, `hook_result`,
`blocks.{i}.hook_resid_post`, …), documented at the top of `vit_interp/model.py`.

## What is where

| | |
|---|---|
| checkpoints | `/lcrc/globalscratch/ac.ngmahfouz/vit-interp/runs/vit-p4-d384-l8/release/v1.pt` (frozen; 4- then 8-step rollout-tuned; the notebooks' default). `stage1/best.pt` is the one-step model, for comparisons |
| data | `/lcrc/globalscratch/ac.ngmahfouz/vit-interp/data/` (2000–2009 train, 2010 for everything here) |
| skill | `runs/vit-p4-d384-l8/eval-*/summary.md` (year-long free run, RMSE/ACC vs lead) |
| exege docs | `exege/docs/circuits.md`, `exege/docs/laws.md` on branch `mahf708/circuits` |
| open questions | the end of [`notebooks/README.md`](notebooks/README.md) |

## Known so far (one afternoon of poking; please overturn)

- Several heads in blocks 1–6 look **20–30° west** (upstream) in the midlatitudes; block-0
  heads average the whole globe; block 5–6 heads are sharply local.
- The answer forms in **blocks 5–6**; moisture is decided last.
- A North Pacific jet-level wind behaviour runs through **61 of 513 edges** (95% recovery),
  with a cluster in blocks 4–6.
- Planted Rossby waves move **westward relative to the flow and faster for the longest
  waves**, the right *shape* (r ≈ 0.98 with theory), but 3–4× the barotropic magnitude.
  Steering level? Baroclinicity? The patch size? That is a good afternoon.
- Skill (release/v1, year 2010): 5-day anomaly correlation 0.87 (PS), 0.84 (T_7), 0.83
  (U_3), 0.78 (V_3); free-running with dry-air/positivity correction it holds up for ~8
  months before drifting cold. Rollout fine-tuning took the free run from 58 to 248 days.
  Patch seams (a 4° checkerboard) grow in long runs: few-day questions give the cleanest
  answers. Full numbers: `runs/vit-p4-d384-l8/eval-ft8-corrected/summary.md`.
