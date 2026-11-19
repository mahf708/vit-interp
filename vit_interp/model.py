"""A close-to-vanilla Vision Transformer emulator, written to be taken apart.

x_t (normalized, all channels) -> patch embed + learned position embedding
-> L pre-LN blocks [x += Attn(LN(x)); x += MLP(LN(x))] -> LN -> linear unpatchify
-> normalized residual tendency of the prognostic channels.

Every component writes additively into the residual stream, which is what edge-based
circuit discovery (Vi-CD) and LAWFUL patching need. The output projection is kept as a
(H, d_head, D) tensor so per-head contributions to the residual stream are exact.

Hook points (names follow TransformerLens):
    hook_embed, hook_pos_embed
    blocks.{i}.hook_resid_pre / hook_resid_mid / hook_resid_post
    blocks.{i}.ln1.hook_normalized, blocks.{i}.ln2.hook_normalized
    blocks.{i}.attn.hook_q / hook_k / hook_v           (B, H, N, d_head)
    blocks.{i}.attn.hook_pattern                        (B, H, N, N)  [forces explicit attention]
    blocks.{i}.attn.hook_z                              (B, H, N, d_head)
    blocks.{i}.attn.hook_result                         (B, N, H, D)  per-head output, no bias
    blocks.{i}.hook_attn_out, blocks.{i}.hook_mlp_out   (B, N, D)
    blocks.{i}.mlp.hook_pre / hook_post                 (B, N, D_mlp)
    ln_final.hook_normalized, hook_out_tokens           (B, N, C_out*p*p)
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

Hook = Callable[[torch.Tensor], torch.Tensor | None]


@dataclasses.dataclass
class ViTConfig:
    in_chans: int = 39
    out_chans: int = 34
    img_size: tuple[int, int] = (180, 360)
    patch: int = 4
    dim: int = 384
    depth: int = 8
    heads: int = 6
    mlp_ratio: float = 4.0

    @property
    def grid(self) -> tuple[int, int]:
        return self.img_size[0] // self.patch, self.img_size[1] // self.patch

    @property
    def n_tokens(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def d_head(self) -> int:
        return self.dim // self.heads


class _Ctx:
    """Per-forward hook/cache context; free when no hooks and no cache."""

    def __init__(self, hooks: dict[str, Hook] | None, cache: dict | None, names: set[str] | None):
        self.hooks = hooks or {}
        self.cache = cache
        self.names = names  # None = cache everything

    def __call__(self, name: str, t: torch.Tensor) -> torch.Tensor:
        fn = self.hooks.get(name)
        if fn is not None:
            r = fn(t)
            if r is not None:
                t = r
        if self.cache is not None and (self.names is None or name in self.names):
            self.cache[name] = t
        return t

    def wants(self, name: str) -> bool:
        return name in self.hooks or (
            self.cache is not None and (self.names is None or name in self.names)
        )

    @property
    def active(self) -> bool:
        return bool(self.hooks) or self.cache is not None


class LayerNorm(nn.LayerNorm):
    def forward_ctx(self, x, ctx: _Ctx, prefix: str):
        return ctx(f"{prefix}.hook_normalized", super().forward(x))


class Attention(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        H, dh, D = cfg.heads, cfg.d_head, cfg.dim
        self.cfg = cfg
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

    def forward(self, x, ctx: _Ctx, prefix: str):
        q = ctx(f"{prefix}.hook_q", torch.einsum("bnd,hde->bhne", x, self.W_Q) + self.b_Q[:, None])
        k = ctx(f"{prefix}.hook_k", torch.einsum("bnd,hde->bhne", x, self.W_K) + self.b_K[:, None])
        v = ctx(f"{prefix}.hook_v", torch.einsum("bnd,hde->bhne", x, self.W_V) + self.b_V[:, None])
        if ctx.wants(f"{prefix}.hook_pattern"):
            scores = torch.einsum("bhqe,bhke->bhqk", q, k) / math.sqrt(self.cfg.d_head)
            pattern = ctx(f"{prefix}.hook_pattern", scores.float().softmax(-1).to(v.dtype))
            z = torch.einsum("bhqk,bhke->bhqe", pattern, v)
        else:
            z = F.scaled_dot_product_attention(q, k, v)
        z = ctx(f"{prefix}.hook_z", z)
        if ctx.wants(f"{prefix}.hook_result"):
            result = ctx(f"{prefix}.hook_result", torch.einsum("bhne,hed->bnhd", z, self.W_O))
            return result.sum(2) + self.b_O
        return torch.einsum("bhne,hed->bnd", z, self.W_O) + self.b_O


class MLP(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.fc1 = nn.Linear(cfg.dim, hidden)
        self.fc2 = nn.Linear(hidden, cfg.dim)

    def forward(self, x, ctx: _Ctx, prefix: str):
        h = ctx(f"{prefix}.hook_pre", self.fc1(x))
        h = ctx(f"{prefix}.hook_post", F.gelu(h))
        return self.fc2(h)


class Block(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg.dim)
        self.mlp = MLP(cfg)

    def forward(self, x, ctx: _Ctx, p: str):
        x = ctx(f"{p}.hook_resid_pre", x)
        a = ctx(f"{p}.hook_attn_out", self.attn(self.ln1.forward_ctx(x, ctx, f"{p}.ln1"), ctx, f"{p}.attn"))
        x = ctx(f"{p}.hook_resid_mid", x + a)
        m = ctx(f"{p}.hook_mlp_out", self.mlp(self.ln2.forward_ctx(x, ctx, f"{p}.ln2"), ctx, f"{p}.mlp"))
        return ctx(f"{p}.hook_resid_post", x + m)


class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        p = cfg.patch
        self.patch_embed = nn.Linear(cfg.in_chans * p * p, cfg.dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.n_tokens, cfg.dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.ln_final = LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.out_chans * p * p)
        nn.init.zeros_(self.head.weight)  # start at persistence
        nn.init.zeros_(self.head.bias)

    # (B, C, H, W) <-> (B, N, C*p*p), tokens in row-major (lat, lon) order
    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.cfg.patch
        x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, (H // p) * (W // p), C * p * p)

    def unpatchify(self, t, C):
        B = t.shape[0]
        p = self.cfg.patch
        gh, gw = self.cfg.grid
        t = t.reshape(B, gh, gw, C, p, p).permute(0, 3, 1, 4, 2, 5)
        return t.reshape(B, C, gh * p, gw * p)

    def embed(self, x, ctx: _Ctx):
        e = ctx("hook_embed", self.patch_embed(self.patchify(x)))
        pe = ctx("hook_pos_embed", self.pos_embed.expand(x.shape[0], -1, -1))
        return e + pe

    def unembed(self, r, ctx: _Ctx):
        t = ctx("hook_out_tokens", self.head(self.ln_final.forward_ctx(r, ctx, "ln_final")))
        return self.unpatchify(t, self.cfg.out_chans)

    def forward(self, x, hooks: dict[str, Hook] | None = None, cache: dict | None = None,
                names: set[str] | None = None):
        """x: normalized inputs (B, in_chans, H, W) -> normalized residual (B, out_chans, H, W)."""
        ctx = _Ctx(hooks, cache, names)
        r = self.embed(x, ctx)
        for i, blk in enumerate(self.blocks):
            r = blk(r, ctx, f"blocks.{i}")
        return self.unembed(r, ctx)

    def run_with_cache(self, x, names: set[str] | None = None, hooks=None):
        cache: dict[str, torch.Tensor] = {}
        out = self.forward(x, hooks=hooks, cache=cache, names=names)
        return out, cache


def token_latlon(lat, lon, patch: int):
    """Centers of the patch grid (row-major), as flat arrays of length n_tokens."""
    import numpy as np

    lat = np.asarray(lat).reshape(-1, patch).mean(1)
    lon = np.asarray(lon).reshape(-1, patch).mean(1)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    return LA.ravel(), LO.ravel(), (len(lat), len(lon))
