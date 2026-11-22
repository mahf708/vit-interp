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
    hook_out_conv                                       (B, C_out, H, W)  [head_conv > 0 only]

``noise_dim = d > 0`` makes the model stochastic, as ACE2S's noise-conditioned SFNO is: every
LayerNorm (ln1, ln2 of each block, ln_final) becomes conditional,
``LN(x) * (1 + W_scale(z)) + W_bias(z)``, where ``z`` is per-token noise ``(B, N, d)`` and
``W_scale``, ``W_bias`` are bias-free linear maps initialized to zero (ACE's
ConditionalLayerNorm with a 1x1 convolution on its noise channels), so a fresh stochastic
model is the plain one. The noise is drawn per call (``noise_type`` "gaussian": i.i.d. per
token, ACE's default; "isotropic": white in spherical harmonics on the token grid, ACE's
other option) unless passed in. It only modulates what components *read*: the residual
stream is still embed + heads + MLPs, and every hook keeps its meaning. Extra hooks:
``hook_noise`` (B, N, d), and ``{prefix}.hook_cond_scale`` / ``hook_cond_bias`` (B, N, D) at
each conditional LayerNorm.

``head_conv = k > 0`` adds one k x k convolution after un-patchifying (circular in longitude,
edge-padded in latitude), initialized to the identity: it lets the output be smooth across
patch seams. It sits after the residual stream's last reader, so circuits are unchanged.
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
    head_conv: int = 0  # k > 0: a k x k convolution on the full-resolution output
    noise_dim: int = 0  # d > 0: noise-conditioned LayerNorms (a stochastic emulator)
    noise_type: str = "gaussian"  # or "isotropic"
    # anti-aliased patch embedding (after NVlabs/STRATA's DealiasedPatchEmbed): the same patch_embed
    # weights applied as a stride-1 convolution, then this low-pass filter and stride-p decimation;
    # empty = the plain non-overlapping patch embedding. (1, 4, 6, 4, 1) is STRATA's default.
    embed_filter: tuple = ()
    # architecture: "vit" (this file: one token per patch) or "vit3d" (model3d.py: one token per
    # level and patch, ViViT-style factorized attention)
    arch: str = "vit"
    attn3d: str = "split"  # vit3d: split (ViViT model 4), factorized (model 3), alternate (STRATA)
    heads_v: int = 2  # vit3d split: heads that attend vertically
    pos3d: str = "abs"  # vit3d: abs (learned) or rope (rotary, periodic in longitude)

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

    def __init__(self, hooks: dict[str, Hook] | None, cache: dict | None, names: set[str] | None,
                 noise: torch.Tensor | None = None):
        self.noise = noise  # (B, N, d) for noise-conditioned LayerNorms
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
    """A LayerNorm; with ``noise_dim > 0``, conditioned on per-token noise (zero-initialized)."""

    def __init__(self, dim: int, noise_dim: int = 0):
        super().__init__(dim)
        self.noise_dim = noise_dim
        if noise_dim:
            self.W_scale = nn.Linear(noise_dim, dim, bias=False)
            self.W_bias = nn.Linear(noise_dim, dim, bias=False)
            nn.init.zeros_(self.W_scale.weight)
            nn.init.zeros_(self.W_bias.weight)

    def forward_ctx(self, x, ctx: _Ctx, prefix: str):
        y = super().forward(x)
        if self.noise_dim:
            if ctx.noise is None:
                raise ValueError("a noise-conditioned LayerNorm needs noise in the context")
            z = ctx.noise.to(y.dtype)
            y = y * (1 + ctx(f"{prefix}.hook_cond_scale", self.W_scale(z))) + ctx(
                f"{prefix}.hook_cond_bias", self.W_bias(z))
        return ctx(f"{prefix}.hook_normalized", y)


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
        self.ln1 = LayerNorm(cfg.dim, cfg.noise_dim)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg.dim, cfg.noise_dim)
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
        self.ln_final = LayerNorm(cfg.dim, cfg.noise_dim)
        if cfg.noise_dim and cfg.noise_type not in ("gaussian", "isotropic"):
            raise ValueError(f"noise_type must be gaussian or isotropic, not {cfg.noise_type!r}")
        if cfg.noise_dim and cfg.noise_type == "isotropic":
            self.register_buffer("_ylm", _token_harmonics(*cfg.grid), persistent=False)
        self.head = nn.Linear(cfg.dim, cfg.out_chans * p * p)
        nn.init.zeros_(self.head.weight)  # start at persistence
        nn.init.zeros_(self.head.bias)
        if cfg.head_conv:
            k = cfg.head_conv
            self.out_conv = nn.Conv2d(cfg.out_chans, cfg.out_chans, k, bias=False)
            with torch.no_grad():  # identity: the plain head, until training says otherwise
                self.out_conv.weight.zero_()
                self.out_conv.weight[torch.arange(cfg.out_chans), torch.arange(cfg.out_chans),
                                     k // 2, k // 2] = 1.0

    # (B, C, H, W) <-> (B, N, C*p*p), tokens in row-major (lat, lon) order
    def embed_dealiased(self, x):
        """Shift-robust patch embedding: patch_embed as a stride-1 convolution (circular in
        longitude, valid in latitude), a separable low-pass, then stride-p decimation. Token
        (i, j) is centred on the same patch as the plain embedding's token (i, j)."""
        B, C, H, W = x.shape
        p, D = self.cfg.patch, self.cfg.dim
        w = self.patch_embed.weight.view(D, C, p, p)  # the (C, p_lat, p_lon) order of patchify
        s1 = F.conv2d(F.pad(x, (0, p - 1, 0, 0), mode="circular"), w.to(x.dtype), self.patch_embed.bias.to(x.dtype))
        f = torch.as_tensor(self.cfg.embed_filter, dtype=s1.dtype, device=s1.device)
        f = f / f.sum()
        k = f.numel()
        lo, hi = (k - 1) // 2, (k - 1) - (k - 1) // 2
        s1 = F.pad(s1, (lo, hi, 0, 0), mode="circular")  # longitude wraps
        s1 = F.pad(s1, (0, 0, lo, hi), mode="replicate")  # latitude: no zeros at the poles
        f2 = (f[:, None] * f[None, :]).expand(D, 1, k, k)
        e = F.conv2d(s1, f2, groups=D, stride=p)  # (B, D, H/p, W/p)
        return e.flatten(2).transpose(1, 2)

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
        e = ctx("hook_embed", self.embed_dealiased(x) if self.cfg.embed_filter else self.patch_embed(self.patchify(x)))
        pe = ctx("hook_pos_embed", self.pos_embed.expand(x.shape[0], -1, -1))
        return e + pe

    def unembed(self, r, ctx: _Ctx):
        t = ctx("hook_out_tokens", self.head(self.ln_final.forward_ctx(r, ctx, "ln_final")))
        y = self.unpatchify(t, self.cfg.out_chans)
        if self.cfg.head_conv:
            h = self.cfg.head_conv // 2
            y = F.pad(y, (h, h, 0, 0), mode="circular")  # longitude wraps
            y = F.pad(y, (0, 0, h, h), mode="replicate")  # latitude does not
            y = ctx("hook_out_conv", self.out_conv(y))
        return y

    def sample_noise(self, batch: int, device, generator: torch.Generator | None = None):
        """(batch, n_tokens, noise_dim) noise of the model's type; ``generator`` for determinism."""
        cfg = self.cfg
        if cfg.noise_type == "gaussian":
            return torch.randn((batch, cfg.n_tokens, cfg.noise_dim), generator=generator,
                               device=device)
        # isotropic: white spherical-harmonic coefficients up to L = n_lat_tokens, synthesized
        # at the token centres; scale sqrt(4 pi) / L gives unit variance (ACE's isotropic_noise)
        Y = self._ylm.to(device)  # (2, M, L, Hg): Re/Im parts of P_lm(lat) times the m-weights
        M, L = Y.shape[1], Y.shape[2]
        a = torch.randn((2, batch, cfg.noise_dim, L, M), generator=generator, device=device,
                        dtype=Y.dtype)
        a[1, ..., 0] = 0.0  # m = 0 is real
        a[..., 1:] = a[..., 1:] / math.sqrt(2.0)
        gh, gw = cfg.grid
        lon = torch.arange(gw, device=device, dtype=Y.dtype) * (2 * math.pi / gw) + math.pi / gw
        mm = torch.arange(M, device=device, dtype=Y.dtype)
        cos, sin = torch.cos(mm[:, None] * lon), torch.sin(mm[:, None] * lon)  # (M, W)
        f_m = torch.einsum("bdlm,mlh->bdhm", a[0], Y[0]), torch.einsum("bdlm,mlh->bdhm", a[1], Y[0])
        field = torch.einsum("bdhm,mw->bdhw", f_m[0], cos) - torch.einsum("bdhm,mw->bdhw", f_m[1], sin)
        field = field * (math.sqrt(4 * math.pi) / L)
        return field.permute(0, 2, 3, 1).reshape(batch, cfg.n_tokens, cfg.noise_dim)

    def forward(self, x, hooks: dict[str, Hook] | None = None, cache: dict | None = None,
                names: set[str] | None = None, noise: torch.Tensor | None = None,
                generator: torch.Generator | None = None):
        """x: normalized inputs (B, in_chans, H, W) -> normalized residual (B, out_chans, H, W).

        Stochastic models (noise_dim > 0) draw noise per call unless ``noise`` (B, N, d) is
        given; ``generator`` makes the draw deterministic."""
        ctx = _Ctx(hooks, cache, names)
        if self.cfg.noise_dim:
            if noise is None:
                noise = self.sample_noise(x.shape[0], x.device, generator)
            ctx.noise = ctx("hook_noise", noise.to(x.device))
        r = self.embed(x, ctx)
        for i, blk in enumerate(self.blocks):
            r = blk(r, ctx, f"blocks.{i}")
        return self.unembed(r, ctx)

    def run_with_cache(self, x, names: set[str] | None = None, hooks=None, noise=None, generator=None):
        cache: dict[str, torch.Tensor] = {}
        out = self.forward(x, hooks=hooks, cache=cache, names=names, noise=noise, generator=generator)
        return out, cache


def build_net(cfg: ViTConfig) -> nn.Module:
    """The network a config describes (``arch``: "vit" or "vit3d")."""
    if cfg.arch == "vit3d":
        from .model3d import ViT3D

        return ViT3D(cfg)
    if cfg.arch != "vit":
        raise ValueError(f"unknown arch {cfg.arch!r}")
    return ViT(cfg)


def token_latlon(lat, lon, patch: int):
    """Centers of the patch grid (row-major), as flat arrays of length n_tokens."""
    import numpy as np

    lat = np.asarray(lat).reshape(-1, patch).mean(1)
    lon = np.asarray(lon).reshape(-1, patch).mean(1)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    return LA.ravel(), LO.ravel(), (len(lat), len(lon))


def _token_harmonics(n_lat: int, n_lon: int) -> torch.Tensor:
    """(2, M, L, n_lat): orthonormal associated Legendre functions at the token-row latitudes
    (equal-angle centres), times 2 for m > 0 (the real synthesis weight); L = n_lat,
    M = min(L, n_lon // 2 + 1). Index 0 and 1 are equal (Re and Im share P_lm)."""
    import numpy as np

    L = n_lat
    M = min(L, n_lon // 2 + 1)
    lat = np.deg2rad(-90 + (np.arange(n_lat) + 0.5) * 180.0 / n_lat)
    x, s = np.sin(lat), np.cos(lat)
    P = np.zeros((M, L, n_lat))
    pmm = np.full(n_lat, math.sqrt(1 / (4 * math.pi)))
    for m in range(M):
        if m > 0:
            pmm = pmm * math.sqrt((2 * m + 1) / (2 * m)) * s
        P[m, m] = pmm
        if m + 1 < L:
            P[m, m + 1] = x * math.sqrt(2 * m + 3) * pmm
        for l in range(m + 2, L):
            a = math.sqrt((4 * l * l - 1) / (l * l - m * m))
            b = math.sqrt(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1))
            P[m, l] = a * (x * P[m, l - 1] - b * P[m, l - 2])
    P[1:] *= 2.0  # real synthesis: f = sum_l [a_l0 P_l0 + 2 sum_{m>0} Re(a_lm e^{im lon}) P_lm]
    P = torch.tensor(P, dtype=torch.float32)
    return torch.stack([P, P])
