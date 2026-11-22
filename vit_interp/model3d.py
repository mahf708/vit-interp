"""A ViViT-style 3D emulator: one token per (vertical level, horizontal patch).

The 2D ``ViT`` folds all 8 levels of a patch into one token, so vertical coupling happens inside
the patch embedding and the MLPs, out of reach of attention-based tools. Here each level is its
own token (4 channels: T, U, V, total water) and the surface is one more "level" (PS, TS and the
5 forcings), giving a residual stream of shape (B, L, N, D) with L = 9 levels and N = 4050
patches. Attention is factorized as in ViViT (Arnab et al. 2021) / NVlabs STRATA:

- ``attn3d="split"`` (ViViT model 4): in every block, ``heads_v`` heads attend up and down their
  column (over the 9 levels) and the rest attend horizontally (over all 4050 patches of their
  level). Every head has a declared job.
- ``attn3d="factorized"`` (ViViT model 3): every block has a horizontal attention sublayer, then a
  vertical one, then the MLP.
- ``attn3d="alternate"`` (STRATA): even blocks attend horizontally, odd blocks vertically.

Positions: ``pos3d="abs"`` adds a learned horizontal position embedding (shared by all levels) and
a learned level embedding; ``"rope"`` keeps the level embedding but encodes horizontal position
with rotary embeddings in the horizontal heads (periodic in longitude), so the model is
translation-equivariant in longitude by construction.

Inputs and outputs are exactly the 2D ViT's ((B, 39, H, W) normalized -> (B, out_chans, H, W)),
so ``Emulator``, training, correctors and evaluation are unchanged. Hooks keep TransformerLens
names with the level axis added: ``blocks.{i}.hook_resid_post`` is (B, L, N, D); attention hooks
are ``blocks.{i}.attn{_h,_v}.hook_{q,k,v,z}``, ``hook_pattern`` (on request), ``hook_result``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .model import MLP, LayerNorm, ViT, ViTConfig, _Ctx

N_LEV = 8
ATM_VARS = 4  # T, U, V, specific_total_water per level
SURF_IN = 7  # PS, TS, SOLIN, PHIS, LANDFRAC, OCNFRAC, ICEFRAC


def _channel_maps():
    """Indices (into the 39 inputs / the 34 prognostic outputs) of each level's channels."""
    from .data import ALL_NAMES, FORCING_NAMES, PROG_NAMES

    atm_in = [[ALL_NAMES.index(f"{v}_{lv}") for v in ("T", "U", "V", "specific_total_water")]
              for lv in range(N_LEV)]
    surf_in = [ALL_NAMES.index(n) for n in ["PS", "TS", *FORCING_NAMES]]
    atm_out = [[PROG_NAMES.index(f"{v}_{lv}") for v in ("T", "U", "V", "specific_total_water")]
               for lv in range(N_LEV)]
    surf_out = [PROG_NAMES.index("PS"), PROG_NAMES.index("TS")]
    return atm_in, surf_in, atm_out, surf_out


def _rope_tables(gh: int, gw: int, dh: int, device, dtype):
    """cos/sin (N, dh) for 2D rotary embeddings: half the pairs rotate with latitude index, half
    with longitude, longitude frequencies integer multiples of 2*pi/gw (exactly periodic)."""
    q = dh // 4  # pairs per axis
    lat_f = 1.0 / (100.0 ** (torch.arange(q, device=device, dtype=torch.float32) / q))
    lon_f = 2 * math.pi * torch.arange(1, q + 1, device=device, dtype=torch.float32) / gw
    i = torch.arange(gh, device=device, dtype=torch.float32)
    j = torch.arange(gw, device=device, dtype=torch.float32)
    a_lat = i[:, None, None] * lat_f  # (gh, 1, q)
    a_lon = j[None, :, None] * lon_f  # (1, gw, q)
    ang = torch.cat([a_lat.expand(gh, gw, q), a_lon.expand(gh, gw, q)], -1).reshape(gh * gw, 2 * q)
    ang = torch.repeat_interleave(ang, 2, dim=-1)  # (N, dh)
    return ang.cos().to(dtype), ang.sin().to(dtype)


def _rotate(x, cos, sin):
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack([-x2, x1], -1).flatten(-2)
    return x * cos + rot * sin


class Attention3D(nn.Module):
    """Multi-head attention over the (L, N) token grid; each head is horizontal or vertical."""

    def __init__(self, cfg: ViTConfig, axes: list[str]):
        super().__init__()
        H, dh, D = cfg.heads, cfg.d_head, cfg.dim
        self.cfg, self.axes = cfg, axes  # axes[h] in {"h", "v"}
        self.W_Q = nn.Parameter(torch.empty(H, D, dh))
        self.W_K = nn.Parameter(torch.empty(H, D, dh))
        self.W_V = nn.Parameter(torch.empty(H, D, dh))
        self.W_O = nn.Parameter(torch.empty(H, dh, D))
        self.b_Q = nn.Parameter(torch.zeros(H, dh))
        self.b_K = nn.Parameter(torch.zeros(H, dh))
        self.b_V = nn.Parameter(torch.zeros(H, dh))
        self.b_O = nn.Parameter(torch.zeros(D))
        for w in (self.W_Q, self.W_K, self.W_V, self.W_O):
            nn.init.trunc_normal_(w, std=0.02)
        self.hi = [h for h, a in enumerate(axes) if a == "h"]
        self.vi = [h for h, a in enumerate(axes) if a == "v"]

    def _attend(self, q, k, v, ctx, name):
        """q, k, v: (B, h, G, S, dh), attention over S within each of G groups."""
        if ctx.wants(name):
            p = ctx(name, (torch.einsum("...qe,...ke->...qk", q, k) / math.sqrt(q.shape[-1])).float().softmax(-1).to(v.dtype))
            return torch.einsum("...qk,...ke->...qe", p, v)
        B, h, G, S, dh = q.shape
        f = lambda a: a.transpose(1, 2).reshape(B * G, h, S, dh)  # noqa: E731  4-D for fused kernels
        o = F.scaled_dot_product_attention(f(q), f(k), f(v))
        return o.reshape(B, G, h, S, dh).transpose(1, 2)

    def forward(self, x, ctx: _Ctx, prefix: str, rope=None):
        # x: (B, L, N, D) -> q, k, v: (B, H, L, N, dh)
        def proj(W, b):
            y = torch.einsum("blnd,hde->bhlne", x, W)
            return y + b[:, None, None].to(y.dtype)  # keep autocast's dtype (fused attention kernels)

        q = ctx(f"{prefix}.hook_q", proj(self.W_Q, self.b_Q))
        k = ctx(f"{prefix}.hook_k", proj(self.W_K, self.b_K))
        v = ctx(f"{prefix}.hook_v", proj(self.W_V, self.b_V))
        parts = []
        if self.hi:
            qh, kh, vh = q[:, self.hi], k[:, self.hi], v[:, self.hi]
            if rope is not None:
                qh, kh = _rotate(qh, *rope), _rotate(kh, *rope)
            parts.append(self._attend(qh, kh, vh, ctx, f"{prefix}.hook_pattern_h"))  # over N
        if self.vi:
            t = lambda a: a[:, self.vi].transpose(2, 3)  # noqa: E731  (B, Hv, N, L, dh)
            parts.append(self._attend(t(q), t(k), t(v), ctx, f"{prefix}.hook_pattern_v").transpose(2, 3))
        dt = parts[0].dtype
        z = torch.cat([p_.to(dt) for p_ in parts], 1)
        order = self.hi + self.vi
        if order != sorted(order):  # back to head order
            z = z[:, [order.index(h) for h in range(len(order))]]
        z = ctx(f"{prefix}.hook_z", z)
        if ctx.wants(f"{prefix}.hook_result"):
            res = ctx(f"{prefix}.hook_result", torch.einsum("bhlne,hed->blnhd", z, self.W_O))
            return res.sum(3) + self.b_O
        return torch.einsum("bhlne,hed->blnd", z, self.W_O) + self.b_O


class Block3D(nn.Module):
    def __init__(self, cfg: ViTConfig, kind: str):
        super().__init__()
        H, Hv = cfg.heads, cfg.heads_v
        self.kind = kind  # "split", "factorized", "h", "v"
        self.ln1 = LayerNorm(cfg.dim, cfg.noise_dim)
        if kind == "split":
            self.attn = Attention3D(cfg, ["h"] * (H - Hv) + ["v"] * Hv)
        elif kind == "factorized":
            self.attn_h = Attention3D(cfg, ["h"] * H)
            self.ln_v = LayerNorm(cfg.dim, cfg.noise_dim)
            self.attn_v = Attention3D(cfg, ["v"] * H)
        else:
            self.attn = Attention3D(cfg, [kind] * H)
        self.ln2 = LayerNorm(cfg.dim, cfg.noise_dim)
        self.mlp = MLP(cfg)

    def forward(self, x, ctx: _Ctx, p: str, rope=None):
        x = ctx(f"{p}.hook_resid_pre", x)
        if self.kind == "factorized":
            a = ctx(f"{p}.hook_attn_h_out", self.attn_h(self.ln1.forward_ctx(x, ctx, f"{p}.ln1"), ctx, f"{p}.attn_h", rope))
            x = ctx(f"{p}.hook_resid_mid_h", x + a)
            a = ctx(f"{p}.hook_attn_v_out", self.attn_v(self.ln_v.forward_ctx(x, ctx, f"{p}.ln_v"), ctx, f"{p}.attn_v"))
        else:
            a = ctx(f"{p}.hook_attn_out", self.attn(self.ln1.forward_ctx(x, ctx, f"{p}.ln1"), ctx, f"{p}.attn", rope))
        x = ctx(f"{p}.hook_resid_mid", x + a)
        m = ctx(f"{p}.hook_mlp_out", self.mlp(self.ln2.forward_ctx(x, ctx, f"{p}.ln2"), ctx, f"{p}.mlp"))
        return ctx(f"{p}.hook_resid_post", x + m)


class ViT3D(ViT):
    """Same interface as ``ViT``; reuses its noise sampling, un-patchify and output convolution."""

    def __init__(self, cfg: ViTConfig):
        nn.Module.__init__(self)
        self.cfg = cfg
        p, D = cfg.patch, cfg.dim
        self.atm_in, self.surf_in, self.atm_out, self.surf_out = _channel_maps()
        self.n_diag = cfg.out_chans - (len(self.surf_out) + N_LEV * ATM_VARS)
        self.embed_atm = nn.Linear(ATM_VARS * p * p, D)
        self.embed_surf = nn.Linear(SURF_IN * p * p, D)
        self.level_embed = nn.Parameter(torch.zeros(1, N_LEV + 1, 1, D))
        nn.init.trunc_normal_(self.level_embed, std=0.02)
        if cfg.pos3d == "abs":
            self.pos_embed = nn.Parameter(torch.zeros(1, 1, cfg.n_tokens, D))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif cfg.pos3d != "rope":
            raise ValueError(f"pos3d must be abs or rope, not {cfg.pos3d!r}")
        kinds = {"split": lambda i: "split", "factorized": lambda i: "factorized",
                 "alternate": lambda i: "h" if i % 2 == 0 else "v"}[cfg.attn3d]
        self.blocks = nn.ModuleList(Block3D(cfg, kinds(i)) for i in range(cfg.depth))
        self.ln_final = LayerNorm(D, cfg.noise_dim)
        if cfg.noise_dim and cfg.noise_type == "isotropic":
            from .model import _token_harmonics

            self.register_buffer("_ylm", _token_harmonics(*cfg.grid), persistent=False)
        self.head_atm = nn.Linear(D, ATM_VARS * p * p)
        self.head_surf = nn.Linear(D, (len(self.surf_out) + self.n_diag) * p * p)
        for h in (self.head_atm, self.head_surf):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        if cfg.head_conv:
            k = cfg.head_conv
            self.out_conv = nn.Conv2d(cfg.out_chans, cfg.out_chans, k, bias=False)
            with torch.no_grad():
                self.out_conv.weight.zero_()
                self.out_conv.weight[torch.arange(cfg.out_chans), torch.arange(cfg.out_chans), k // 2, k // 2] = 1.0

    def _embed_group(self, x, lin: nn.Linear):
        """(B, C, H, W) -> (B, N, D) with this group's embedding (anti-aliased if configured)."""
        if not self.cfg.embed_filter:
            return lin(self.patchify(x))
        B, C, H, W = x.shape
        p, D = self.cfg.patch, self.cfg.dim
        w = lin.weight.view(D, C, p, p)
        s1 = F.conv2d(F.pad(x, (0, p - 1, 0, 0), mode="circular"), w.to(x.dtype), lin.bias.to(x.dtype))
        f = torch.as_tensor(self.cfg.embed_filter, dtype=s1.dtype, device=s1.device)
        f = f / f.sum()
        k = f.numel()
        lo, hi = (k - 1) // 2, (k - 1) - (k - 1) // 2
        s1 = F.pad(F.pad(s1, (lo, hi, 0, 0), mode="circular"), (0, 0, lo, hi), mode="replicate")
        e = F.conv2d(s1, (f[:, None] * f[None, :]).expand(D, 1, k, k), groups=D, stride=p)
        return e.flatten(2).transpose(1, 2)

    def embed(self, x, ctx: _Ctx):
        B = x.shape[0]
        atm = torch.stack([x[:, idx] for idx in self.atm_in], 1)  # (B, 8, 4, H, W)
        e_atm = self._embed_group(atm.flatten(0, 1), self.embed_atm).view(B, N_LEV, -1, self.cfg.dim)
        e_surf = self._embed_group(x[:, self.surf_in], self.embed_surf)[:, None]
        e = ctx("hook_embed", torch.cat([e_atm, e_surf], 1))  # (B, 9, N, D)
        e = e + ctx("hook_level_embed", self.level_embed)
        if self.cfg.pos3d == "abs":
            e = e + ctx("hook_pos_embed", self.pos_embed)
        return e

    def unembed(self, r, ctx: _Ctx):
        B = r.shape[0]
        r = self.ln_final.forward_ctx(r, ctx, "ln_final")
        t_atm = self.head_atm(r[:, :N_LEV])  # (B, 8, N, 4 p p)
        t_surf = self.head_surf(r[:, N_LEV])  # (B, N, (2 + n_diag) p p)
        ctx("hook_out_tokens", t_surf)
        y_atm = self.unpatchify(t_atm.flatten(0, 1), ATM_VARS).view(B, N_LEV, ATM_VARS, *self.cfg.img_size)
        y_surf = self.unpatchify(t_surf, len(self.surf_out) + self.n_diag)
        y = y_surf.new_zeros(B, self.cfg.out_chans, *self.cfg.img_size)
        for lv, idx in enumerate(self.atm_out):
            y[:, idx] = y_atm[:, lv]
        y[:, self.surf_out] = y_surf[:, : len(self.surf_out)]
        if self.n_diag:
            y[:, -self.n_diag:] = y_surf[:, len(self.surf_out):]
        if self.cfg.head_conv:
            h = self.cfg.head_conv // 2
            y = F.pad(F.pad(y, (h, h, 0, 0), mode="circular"), (0, 0, h, h), mode="replicate")
            y = ctx("hook_out_conv", self.out_conv(y))
        return y

    def forward(self, x, hooks=None, cache=None, names=None, noise=None, generator=None):
        ctx = _Ctx(hooks, cache, names)
        if self.cfg.noise_dim:
            if noise is None:
                noise = self.sample_noise(x.shape[0], x.device, generator)
            ctx.noise = ctx("hook_noise", noise.to(x.device))[:, None]  # broadcast over levels
        rope = None
        if self.cfg.pos3d == "rope":
            rope = _rope_tables(*self.cfg.grid, self.cfg.d_head, x.device, x.dtype)
        r = self.embed(x, ctx)
        ckpt = self.training and getattr(self, "grad_checkpoint", False) and not ctx.active
        for i, blk in enumerate(self.blocks):
            if ckpt:
                r = torch.utils.checkpoint.checkpoint(blk, r, ctx, f"blocks.{i}", rope, use_reentrant=False)
            else:
                r = blk(r, ctx, f"blocks.{i}", rope)
        return self.unembed(r, ctx)
