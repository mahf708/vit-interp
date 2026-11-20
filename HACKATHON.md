# Start here: inside a ViT weather emulator

A close-to-vanilla Vision Transformer that steps E3SMv3 (1°, AMIP) forward 6 hours: 4050 patch tokens,
8 blocks of 6 heads, every head and MLP writing into one residual stream. It is small enough to take apart
in an afternoon, and the tools to do it are ready. The goal today: **find mechanisms**: which heads move
weather, which components carry a physical law, which circuits produce the model's errors.

**Results, the architecture explorer and every notebook already executed:**
https://web.lcrc.anl.gov/public/e3sm/diagnostic_output/ac.ngmahfouz/vit-interp/ — skim it first.

## The reading list, and where each paper lives here

| Paper | What it gives us | Where |
|---|---|---|
| [Seeing Through Circuits: Faithful Mechanistic Interpretability for Vision Transformers](https://arxiv.org/abs/2604.14477) (Vi-CD) | edge circuits in ViTs: a reduced graph (one attention-input node per block), greedy pruning by activation patching, stable cores across runs | `exege circuits`, notebook 07 |
| [LAWFUL: Law-Aligned Witness for Faithful Use of Latents](https://arxiv.org/abs/2607.28672) | does a model obey a physical law over a family of counterfactuals, and which components carry it (sufficiency, necessity, controls) | `exege laws` (Rossby waves), notebooks 06, 07 |
| [Rethinking Circuit Evaluation: Do Circuits Explain Model Errors?](https://arxiv.org/abs/2609.35686) | circuits for the model's *errors*, not just its predictions | any `exege circuits` metric built on ViT − E3SM; notebooks 03, 04, 07 |
| [A Mechanistic Analysis of Transformers for Dynamical Systems](https://arxiv.org/abs/2512.21113) | how transformers implement dynamics: attention as transport, layer-wise computation | notebooks 01 (attention atlas), 02 (logit lens), 06 |

Fifteen concrete open questions tied to these papers are at the end of [`notebooks/README.md`](notebooks/README.md).

## Get set up (5 minutes)

- **On LCRC (Swing):** nothing to install. `export PATH=/home/ac.ngmahfouz/vit-interp/.venv/bin:$PATH`, get a GPU
  node (`qsub -I -A E3SM -q gpu -l select=1:ngpus=1 -l walltime=4:00:00`), and follow the Jupyter steps in
  [`notebooks/README.md`](notebooks/README.md).
- **Anywhere else:** `git clone https://github.com/mahf708/vit-interp && cd vit-interp && pip install -e .`
  The first `pg.load()` downloads the released model and the 2010 inputs from Hugging Face (about 15 GB,
  once, into `~/.cache/vit-interp`). A GPU helps; a CPU is fine for single steps.
- **Coding agents:** point them at [`AGENTS.md`](AGENTS.md): shapes, hooks, tools, costs and traps in one page.

## 1. Notebooks (most people start here)

Each is a few lines per cell and ends with a **Try this** list.

| | |
|---|---|
| `00_quickstart` | one step and a 5-day forecast against E3SM |
| `01_attention_atlas` | where each of the 48 heads looks: upstream? local? global? |
| `02_logit_lens` | decode every block: when does the model know the answer? |
| `03_ablation_lab` | knock out heads and MLPs; what breaks, where |
| `04_direct_attribution` | who writes the jet-level wind over a box (exact decomposition) |
| `05_exege_latents` | latent archives, region analysis, sparse autoencoders |
| `06_rossby_probe` | plant a Rossby wave; does the model know β? |
| `07_exege_circuits_and_laws` | Vi-CD edge circuits and LAWFUL, through exege |

## 2. One-liners (no Python)

These work anywhere `vit-interp` is installed (`--source v2` is the released model; a path works too):

```console
$ exege circuits graph --adapter vit-interp --source v2
$ exege circuits attribute --adapter vit-interp --source v2 --field V_3 --lat 45 --lon 180 \
    --base 2010-01-15T00 --source-starts 2010-01-20T00 --method eap
$ exege circuits prune   --adapter vit-interp --source v2 --field V_3 --lat 45 --lon 180 \
    --base 2010-01-15T00 --source-starts 2010-01-20T00 --tau 0.02 --out circuit.json
$ exege laws rossby --adapter vit-interp --source v2 --u U_2,U_3,U_4 --v V_2,V_3,V_4 \
    --observe V_3 --ps PS --phi0 45 --starts 0,60,120,180 --score correlation
$ exege laws rossby ... --search --by-block --control 30,90,150,210   # which blocks carry it
```

On one A100: attribution ~10 s, a prune (513 patched steps) ~45 s, a Rossby sweep ~15 s, a block-level
LAWFUL search 2–4 min. `--help` lists every option; `--out x.json` keeps the numbers.

## 3. Precomputed latents (no GPU, no download)

The released model's residual stream (every block, every 6 hours, 30 days of January and of July 2010) as
[exege](https://github.com/mahf708/exege) latent archives on the 45×90 token grid, with sparse autoencoders
(1024 features, top-32; 87–93% of the variance explained) on blocks 2, 4 and 7, read straight from Hugging Face:

```console
$ A=hf://datasets/mahf708/vit-interp-e3sm-latents       # on LCRC also /lcrc/globalscratch/ac.ngmahfouz/vit-interp/archives
$ exege latents info $A/v2-2010-jan                      # layers: 0 = embedding, i + 1 = block i
$ exege latents region $A/v2-2010-jan --lat 45 --lon 180 --layer 5 --centered --pcs 3
$ hf download mahf708/vit-interp-e3sm-latents v2-2010-jan/bases/sae-topk32-layer5.npz --repo-type dataset --local-dir latents
$ exege latents region $A/v2-2010-jan --lat 45 --lon 180 --layer 5 \
    --features 5 --basis latents/v2-2010-jan/bases/sae-topk32-layer5.npz   # SAE features (bases must be local)
$ exege app --latents /lcrc/globalscratch/ac.ngmahfouz/vit-interp/archives   # the local web app (on LCRC)
```

January against July is a ready-made experiment: which features are seasonal, which follow the jet?

## 4. Python, your own way

```python
from vit_interp import playground as pg                  # the helpers behind the notebooks
em, meta = pg.load()                                     # release v2; pg.load("v1") for the first model
x = pg.state("2010-01-15T00")
from exege.circuits import open_patchable                # the paper methods, model-agnostic
system = open_patchable("v2", "vit-interp", year=2010)
from exege.latents import open_intervenable              # steering in token space
tokens = open_intervenable("v2", "vit-interp-tokens")
```

Hooks follow TransformerLens names (`blocks.{i}.attn.hook_pattern`, `hook_result`, `blocks.{i}.hook_resid_post`, …),
listed at the top of `vit_interp/model.py` and drawn on the architecture schematic.

## The models

| | What | Where |
|---|---|---|
| **v2 (default)** | 1980–2009 data, 10 extra diagnostic outputs (precipitation, fluxes), ACE-style correctors in the loss, 3×3 output conv, EMA; one-step then 4- and 8-step rollouts | `pg.load()`, `--source v2`; [Hugging Face](https://huggingface.co/mahf708/vit-interp-e3sm) |
| v1 | the first model: 2000–2009, MSE, rollout-tuned | `pg.load("v1")`, `--source v1` |
| v3 | v2 made stochastic (noise-conditioned LayerNorms), trained with CRPS: the best 10-day skill and calibrated ensembles | LCRC: `runs/vit3-p4-d384-l8/`, branch `v3-stochastic` |
| v4 | v3 trained on long random-length rollouts with last-step gradients: holds a realistic climate longest | LCRC: `runs/vit4-p4-d384-l8/`, branch `v4-longroll` |

Every number for every version (skill vs lead, free-run drift, days the climate holds) is on the results page.

## Known so far (release v2; a starting point, please overturn)

- **Skill on 2010:** 5-day anomaly correlation 0.92 (PS), 0.90 (T_7), 0.88 (U_3), 0.85 (V_3); the global
  climate stays close to E3SM's for ~4 months of free running, then drifts moist and cold (the energy budget is
  enforced; it trades temperature for latent energy). Patch seams grow in long runs: few-day questions give the
  cleanest answers.
- **Heads that look upstream:** L3H5 reads ~20° west of a midlatitude token on average; L1H0 and L6H4 also
  look west. Block-0 heads average the whole globe; later heads are sharply local. Head identities differ
  between checkpoints: recompute for the model you load.
- **The answer forms late:** the logit lens stays above 100% error until block 5 and falls to 0 only at the end.
- **A circuit:** the North Pacific jet-level wind tendency (base Jan 15, counterfactual Jan 20) runs through 88 of
  513 edges, which recover it fully; most of them leave block-4 heads and the direct embedding path.
- **Rossby waves:** planted waves move westward relative to the jet, with the right *shape* across wavenumbers at
  every latitude tested (correlation with barotropic theory 0.56 at 30°N, 0.79 at 45°N, 0.88 at 60°N, 0.87 at 45°S)
  but 2–11× the magnitude. LAWFUL finds no τ-consistent block set: much of the response runs through the direct
  embedding → output path, which LAWFUL does not patch. Steering level? Baroclinicity? Patch size?
