# AGENTS.md — working in vit-interp

You are helping a hackathon participant take apart a small ViT weather emulator of E3SMv3. Favor running
things and showing results (maps, tables) over writing new infrastructure: almost everything already exists.

## Orient in one minute

- `README.md` (install, links), `HACKATHON.md` (the menu of activities), `notebooks/README.md` (what each
  notebook does, open questions tied to the papers).
- Results and the architecture explorer, already computed:
  https://web.lcrc.anl.gov/public/e3sm/diagnostic_output/ac.ngmahfouz/vit-interp/ (read it before re-deriving
  anything: skill tables, Rossby results, circuits, which heads look upstream).

## The model (release v2)

- Input `x`: physical state `(B, 39, 180, 360)`: 34 prognostic channels (`PS, TS, T_0..7, U_0..7, V_0..7,
  specific_total_water_0..7`; level 0 = model top, 7 = near surface) then 5 forcings (`SOLIN, PHIS, LANDFRAC,
  OCNFRAC, ICEFRAC`). Names: `vit_interp.data.ALL_NAMES`; index one with `pg.ch("V_3")`.
- Tokens: 4°×4° patches → 45×90 = 4050 tokens (row-major lat, lon; `pg.token_of(lat, lon)`), dim 384, learned
  position embedding. 8 pre-LN blocks × 6 heads (64 dims), MLP ×4, then LN → linear head → un-patchify →
  3×3 conv (v2 only).
- Output: the normalized 6-h tendency. **v2 has 44 output channels** (34 tendencies + 10 ACE diagnostics:
  precipitation, LHFLX, SHFLX, radiative fluxes, moisture advection); v1 has 34. Always slice `[:, :34]`
  before multiplying by `em.res_std`. `em.step(x)` returns `(next_state, normalized_output)` and already does this.
- **v3/v4 are stochastic** (32-dim per-token noise into every LayerNorm, `em.cfg.noise_dim`): `em.step(x)`
  draws fresh noise per call. Pass `generator=` or `noise=` (zeros = noise-free path) for repeatable results;
  exege adapters fix one noise field per `seed`. v1/v2 are deterministic.
- The skip (`x + tendency`) and the conservation correctors are outside the network: they never enter hooks,
  circuits or LAWFUL measurements.

## Doing things

```python
from vit_interp import playground as pg
em, meta = pg.load()                 # "v1".."v4" / a path; downloads off LCRC
x = pg.state("2010-01-15T00")        # any 2010 time label prefix, or an index 0..1459
cache = {}; em.step(x.to(next(em.parameters()).device), cache=cache, names={"blocks.5.hook_resid_post"})
```

- Hooks (TransformerLens names, see the top of `vit_interp/model.py`): `hook_embed`, `hook_pos_embed`,
  `blocks.{i}.hook_resid_pre|mid|post`, `blocks.{i}.attn.hook_q|k|v|z|pattern|result` (`hook_result` is
  per head, `(B, N, H, D)`), `blocks.{i}.hook_attn_out|mlp_out`, `blocks.{i}.mlp.hook_pre|post`,
  `ln_final.hook_normalized`, `hook_out_tokens`, `hook_out_conv`. Pass `hooks={name: fn}` to edit, `cache`
  + `names` to record. Asking for `hook_pattern` computes explicit 4050×4050 attention (fine on GPU, slow on CPU).
- Playground helpers: `attention`, `head_atlas`, `lens`, `ablate`, `sweep_components`, `direct_attribution`
  (exact: the parts sum to the output), `rollout`, `plot_map(s)`, `box`, `box_mean`.
- Circuits and laws through exege (the model is the `vit-interp` adapter; graph = 57 senders, 17 receivers,
  513 edges): `exege.circuits.{open_patchable, field_mean, attribute, prune, faithfulness_curve, core}`,
  `exege.laws.{Rossby, measure, deviations, search, blocks}`; CLIs `exege circuits …`, `exege laws …`
  (`--help` on each). A prune is 513 patched steps (~45 s on an A100); search LAWFUL circuits with
  `--by-block` first.
- Latents: `exege latents …` on `hf://datasets/mahf708/vit-interp-e3sm-latents/v2-2010-jan` (or `-jul`);
  layer 0 = embedding, i + 1 = block i; SAE bases in each archive's `bases/`.

## Rules of the road

- **GPU:** on Swing, compute on a GPU node (`qsub -I -A E3SM -q gpu -l select=1:ngpus=1 -l walltime=4:00:00`),
  not the login node. CPU is fine for a single step or a small test.
- **Don't modify** `/lcrc/globalscratch/ac.ngmahfouz/...` (shared, read-only for you anyway) or the
  `release/` checkpoints. Write your outputs to your own directory.
- **Notebooks are generated** from `notebooks/_build.py`; to change one for everyone, edit that file and
  rerun it. For your own work, copy the notebooks and edit freely.
- **Time labels** are noleap-calendar strings (`2010-07-15T06:00:00`); unique prefixes work.
- **Head identities differ between checkpoints:** a finding about `L4H3` in v1 says nothing about `L4H3` in v2.
  Recompute (`pg.head_atlas`) for the model you load.
- **Known model limits:** good to ~5 days (5-day ACC 0.85–0.92); free runs drift after ~4 months; 4° patch
  seams appear in long runs. Prefer few-day questions.
- Tests: `python -m pytest -q tests` (CPU, a few minutes).
