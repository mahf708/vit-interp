"""Edge-level circuits on the ViT, after Vi-CD (Zukowska et al., "Seeing Through Circuits").

The graph is Vi-CD's reduced one. Senders are the input embedding (patch + position),
every attention head and every MLP. Receivers are each block's attention input (one node
for all heads, since q, k and v read the same LayerNorm'd stream), each block's MLP input,
and the output head. An edge (s, r) exists when s is upstream of r.

`edge_forward` runs the model with every receiver's input rebuilt as
    sum_s corrupt[s] + mask[s, r] * (clean_run[s] - corrupt[s])  (+ the O biases),
where clean_run[s] is the sender's output *in this patched run*. mask = 1 everywhere
reproduces the clean forward and mask = 0 the corrupted one exactly (tested). The mask may
be float and require grad: d metric / d mask at mask = 1 is edge attribution patching
(EAP), and `prune` is the ACDC/Vi-CD greedy pass with activation patching.

For a regression emulator there are no logits: a "metric" is any scalar function of the
normalized output tendency, e.g. the mean V_3 tendency over a box (see `box_metric`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch

from .model import ViT, _Ctx


@dataclasses.dataclass(frozen=True)
class Graph:
    depth: int
    heads: int

    @property
    def senders(self) -> list[str]:
        s = ["embed"]
        for i in range(self.depth):
            s += [f"a{i}.h{h}" for h in range(self.heads)] + [f"m{i}"]
        return s

    @property
    def receivers(self) -> list[str]:
        r = []
        for i in range(self.depth):
            r += [f"a{i}", f"m{i}"]
        return r + ["out"]

    def sender_index(self, layer: int, kind: str, head: int = 0) -> int:
        """kind 'a' (head) or 'm' (mlp); 'embed' is 0."""
        base = 1 + layer * (self.heads + 1)
        return base + (head if kind == "a" else self.heads)

    def receiver_index(self, name: str) -> int:
        return self.receivers.index(name)

    def upstream(self, r: str) -> list[int]:
        """Sender indices feeding receiver r."""
        if r == "out":
            return list(range(len(self.senders)))
        layer = int(r[1:])
        n = 1 + layer * (self.heads + 1)  # embed + all earlier layers
        if r[0] == "m":
            n += self.heads  # this layer's heads feed its MLP
        return list(range(n))

    def valid(self) -> torch.Tensor:
        """(n_senders, n_receivers) bool: which edges exist."""
        v = torch.zeros(len(self.senders), len(self.receivers), dtype=torch.bool)
        for j, r in enumerate(self.receivers):
            v[self.upstream(r), j] = True
        return v

    def edges(self) -> list[tuple[int, int]]:
        return [tuple(e) for e in self.valid().nonzero().tolist()]

    def edge_name(self, e: tuple[int, int]) -> str:
        return f"{self.senders[e[0]]}->{self.receivers[e[1]]}"


def graph_of(model: ViT) -> Graph:
    return Graph(model.cfg.depth, model.cfg.heads)


def _blank(noise=None) -> _Ctx:
    return _Ctx(None, None, None, noise)


def _noise_for(model: ViT, x_norm, noise):
    """Stochastic models need one noise field shared by every pass of a patched run."""
    if model.cfg.noise_dim and noise is None:
        raise ValueError("a stochastic model (noise_dim > 0) needs the noise to patch with: "
                         "pass the same noise to sender_outputs and edge_forward")
    return noise


def _attn_heads(model: ViT, i: int, x_in, noise=None) -> torch.Tensor:
    blk = model.blocks[i]
    cache: dict = {}
    ctx = _Ctx(None, cache, {f"blocks.{i}.attn.hook_result"})
    blk.attn(blk.ln1.forward_ctx(x_in, _blank(noise), ""), ctx, f"blocks.{i}.attn")
    return cache[f"blocks.{i}.attn.hook_result"]  # (B, N, H, D), no bias


def _mlp(model: ViT, i: int, x_in, noise=None) -> torch.Tensor:
    blk = model.blocks[i]
    return blk.mlp(blk.ln2.forward_ctx(x_in, _blank(noise), ""), _blank(), "")


@torch.no_grad()
def sender_outputs(model: ViT, x_norm, noise=None) -> torch.Tensor:
    """(n_senders, B, N, D) sender outputs of an ordinary forward pass (e.g. the corrupted one).
    Stochastic models: ``noise`` (B, N, d), the same as the patched run's."""
    noise = _noise_for(model, x_norm, noise)
    g = graph_of(model)
    names = {"hook_embed", "hook_pos_embed"}
    for i in range(g.depth):
        names |= {f"blocks.{i}.attn.hook_result", f"blocks.{i}.hook_mlp_out"}
    _, c = model.run_with_cache(x_norm, names=names, noise=noise)
    out = [c["hook_embed"] + c["hook_pos_embed"]]
    for i in range(g.depth):
        res = c[f"blocks.{i}.attn.hook_result"]
        out += [res[:, :, h] for h in range(g.heads)] + [c[f"blocks.{i}.hook_mlp_out"]]
    return torch.stack(out)


def edge_forward(model: ViT, x_norm, corrupt: torch.Tensor, mask: torch.Tensor,
                 noise=None) -> torch.Tensor:
    """Output (normalized tendency) of the clean input x_norm with edges patched.

    corrupt: sender_outputs(model, x_corrupt_norm). mask: (n_senders, n_receivers), 1 = edge
    carries the clean value, 0 = the corrupted one; entries for edges that do not exist are
    ignored.
    """
    noise = _noise_for(model, x_norm, noise)
    g = graph_of(model)
    mask = mask.to(corrupt.dtype)
    S: list[torch.Tensor] = [model.embed(x_norm, _blank())]
    bias = torch.zeros_like(S[0][:1, :1])

    def receive(r: str):
        j = g.receiver_index(r)
        up = g.upstream(r)
        clean = torch.stack(S[: len(up)])
        corr = corrupt[: len(up)]
        m = mask[up, j].view(-1, 1, 1, 1)
        return (corr + m * (clean - corr)).sum(0) + bias

    for i, blk in enumerate(model.blocks):
        res = _attn_heads(model, i, receive(f"a{i}"), noise)
        S += [res[:, :, h] for h in range(g.heads)]
        bias = bias + blk.attn.b_O
        S.append(_mlp(model, i, receive(f"m{i}"), noise))
    return model.unembed(receive("out"), _blank(noise))


def full_mask(model: ViT, value: float = 1.0, device=None) -> torch.Tensor:
    g = graph_of(model)
    return g.valid().to(device).float() * value


def eap_scores(model, x_norm, corrupt, metric: Callable[[torch.Tensor], torch.Tensor],
               noise=None) -> torch.Tensor:
    """Edge attribution patching: d metric / d mask at the clean run (first-order effect of
    corrupting each edge is minus this). Returns (n_senders, n_receivers), zero off-graph."""
    m = full_mask(model, device=x_norm.device).requires_grad_(True)
    metric(edge_forward(model, x_norm, corrupt, m, noise)).backward()
    return m.grad * graph_of(model).valid().to(m.device)


@torch.no_grad()
def prune(model, x_norm, corrupt, metric, tau: float, order: str = "acdc", verbose: bool = False,
          noise=None):
    """Greedy edge pruning with activation patching (ACDC; the Vi-CD criterion with a
    regression metric). Walks receivers from the output backwards and drops an edge when
    corrupting it moves the metric by less than tau (relative to the current circuit).

    Returns (mask, log) where log holds (edge, metric_after) for removed edges.
    """
    g = graph_of(model)
    mask = full_mask(model, device=x_norm.device)
    current = float(metric(edge_forward(model, x_norm, corrupt, mask, noise)))
    log = []
    for j in reversed(range(len(g.receivers))):
        for s in reversed(g.upstream(g.receivers[j])):
            mask[s, j] = 0.0
            new = float(metric(edge_forward(model, x_norm, corrupt, mask, noise)))
            if abs(new - current) < tau:
                current = new
                log.append(((s, j), new))
                if verbose:
                    print(f"drop {g.edge_name((s, j))}: {new:.4g}")
            else:
                mask[s, j] = 1.0
    return mask, log


def box_metric(channel: int, lat_slice: slice, lon_slice: slice):
    """Mean normalized tendency of one output channel over a lat/lon box (grid indices)."""

    def f(out):
        return out[:, channel, lat_slice, lon_slice].float().mean()

    return f
