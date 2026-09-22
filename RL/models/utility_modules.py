import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def _mlp(in_d: int, hid_d: int, out_d: int, depth: int = 2,
         out_gain: float = 0.01) -> nn.Sequential:
    """MLP with `depth` hidden layers and a pre-output LayerNorm.

    `out_gain` tags the OUTPUT Linear so `init_weights` gives it a small
    orthogonal gain instead of the hidden-layer gain. This is the standard PPO
    trick - a policy head that starts near-uniform explores instead of
    committing to an arbitrary action ordering - and tagging it here covers
    every `_mlp` call site in the codebase from one place.
    """
    assert depth >= 1
    layers: list = [nn.Linear(in_d, hid_d), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hid_d, hid_d), nn.Tanh()]
    out = nn.Linear(hid_d, out_d)
    out._out_gain = out_gain                     # read by init_weights()
    layers += [nn.LayerNorm(hid_d), out]         # layernorm after activation?
    return nn.Sequential(*layers)


# ==============================================================================
# Weight initialisation
# ==============================================================================
#
# Before sc-48 there was NO initialisation anywhere in this repo: every module
# relied on PyTorch's legacy `kaiming_uniform_(a=sqrt(5))` default, including
# every policy output layer and the critic's output layer.
#
# Two traps make a blanket `module.apply(fn)` the wrong tool here, both verified
# against the installed torch 2.9.1 / PyG 2.7.0:
#
#   * PyG's `TransformerConv` holds `lin_key/query/value/skip/beta`, and those
#     are `torch_geometric.nn.dense.linear.Linear` - NOT `nn.Linear`. An
#     isinstance check skips all five, so the attention layers keep the default
#     init while everything around them changes.
#   * `nn.MultiheadAttention` packs Q/K/V into a single bare `in_proj_weight`
#     Parameter (invisible to an nn.Linear check) while `out_proj` IS an
#     nn.Linear. A blanket apply therefore reinitialises out_proj and leaves
#     Q/K/V at the default - silently inconsistent across all 9 attention sites.
#
# Q/K deliberately get a SMALLER gain than V. Large Q/K gains produce sharp,
# low-entropy attention at initialisation (Zhai et al., ICML 2023), which in
# this network would reintroduce max-pool-like discontinuity through the
# readout. Diffuse attention at init is what we want.

def init_mha(m: nn.MultiheadAttention,
             qk_gain: float = 0.5, v_gain: float = 1.0) -> None:
    """Orthogonal init for nn.MultiheadAttention, slicing the packed Q/K/V."""
    d = m.embed_dim
    nn.init.orthogonal_(m.in_proj_weight[:d],        gain=qk_gain)   # Q
    nn.init.orthogonal_(m.in_proj_weight[d:2 * d],   gain=qk_gain)   # K
    nn.init.orthogonal_(m.in_proj_weight[2 * d:],    gain=v_gain)    # V
    nn.init.orthogonal_(m.out_proj.weight,           gain=v_gain)
    if m.in_proj_bias is not None:
        nn.init.zeros_(m.in_proj_bias)
    if m.out_proj.bias is not None:
        nn.init.zeros_(m.out_proj.bias)


def init_transformer_conv(conv, qk_gain: float = 0.5, v_gain: float = 1.0) -> None:
    """Orthogonal init for PyG TransformerConv's non-nn.Linear submodules."""
    conv.reset_parameters()                       # PyG Glorot baseline first
    for name, gain in (("lin_query", qk_gain), ("lin_key", qk_gain),
                       ("lin_value", v_gain),   ("lin_skip", v_gain)):
        lin = getattr(conv, name, None)
        if lin is None:
            continue
        nn.init.orthogonal_(lin.weight, gain=gain)
        if getattr(lin, "bias", None) is not None:
            nn.init.zeros_(lin.bias)
    # beta = sigmoid(0) = 0.5 -> attention and skip enter the residual equally.
    if getattr(conv, "lin_beta", None) is not None:
        nn.init.zeros_(conv.lin_beta.weight)
        if conv.lin_beta.bias is not None:
            nn.init.zeros_(conv.lin_beta.bias)


def init_weights(m: nn.Module, gain: float = math.sqrt(2)) -> None:
    """Recursively orthogonal-initialise a module tree.

    Layers tagged with `_out_gain` (see `_mlp`, and the critic's output layers)
    use that gain instead of `gain`, which is how output layers get their small
    near-uniform initialisation without a separate registry of names.
    """
    from torch_geometric.nn import TransformerConv      # local: avoids a cycle

    if isinstance(m, nn.MultiheadAttention):
        init_mha(m)
        return
    if isinstance(m, TransformerConv):
        init_transformer_conv(m)
        return
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain=getattr(m, "_out_gain", gain))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # nn.LayerNorm keeps PyTorch's default (weight=1, bias=0), which is correct.
    for child in m.children():
        init_weights(child, gain)


def _shannon_entropy(probs: torch.Tensor) -> torch.Tensor:
    """H = -sum(p * log p), numerically stable.  probs: (U,)."""
    return -(probs * probs.clamp(min=1e-8).log()).sum()   


def _build_grid_edge_index(Nx: int, Ny: int) -> torch.Tensor:
    """CPU edge-index tensor for the 8-connected (Moore) grid."""
    src, dst = [], []
    for i in range(Nx):
        for j in range(Ny):
            u = i * Ny + j
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < Nx and 0 <= nj < Ny:
                        src.append(u)
                        dst.append(ni * Ny + nj)
    return torch.tensor([src, dst], dtype=torch.long)



class MultiScaleConv(nn.Module):
    """Aggregate spatial context at multiple scales for a set of query tiles.

    Takes the full board node-embedding grid and, for a list of query tile IDs,
    returns a concatenation of multi-scale spatially-pooled feature vectors
    centred on each query tile.

    Each scale consists of `n_conv_layers` stacked Conv2d layers (same kernel
    size throughout the stack), applied sequentially to the board grid.  After
    the stack the feature map has the same spatial dimensions as the input — the
    query tile positions are then simply indexed to extract their features.

    Stacking convolutions expands the effective receptive field: a stack of
    n_conv_layers convolutions with kernel k covers a neighbourhood of radius
    n_conv_layers * (k // 2) hops, while keeping the parameter count modest.

    Parameters
    ──────────
    node_dim      : int            input (and output) channel width D
    kernel_sizes  : Sequence[int]  odd ints, one per scale e.g. (9, 7, 5, 3)
    n_conv_layers : int            number of stacked Conv2d per kernel size

    Input
    ─────
    node_emb  : Tensor (N_tiles, D)   full board node embeddings
    tile_ids  : list[int]             Q query positions to extract features for
    Nx, Ny    : int                   board dimensions

    Output
    ──────
    Tensor (Q, D * n_scales)   — one row per query tile, scales concatenated
    """

    def __init__(
        self,
        node_dim:      int,
        kernel_sizes:  Sequence[int] = (9, 7, 5, 3),
        n_conv_layers: int           = 2,
    ) -> None:
        super().__init__()

        assert all(k % 2 == 1 for k in kernel_sizes), \
            "All kernel_sizes must be odd integers."
        assert n_conv_layers >= 1, "n_conv_layers must be >= 1."

        self.node_dim     = node_dim
        self.kernel_sizes = list(kernel_sizes)

        # One sequential stack per kernel size.
        # Every Conv2d preserves spatial dimensions via padding = k // 2.
        # replicate padding avoids zero-boundary artefacts at board edges.
        self.conv_stacks = nn.ModuleList([
            nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(
                        node_dim, node_dim,
                        kernel_size=k,
                        padding=k // 2,
                        padding_mode="replicate",
                    ),
                    nn.Tanh(),
                )
                for _ in range(n_conv_layers)
            ])
            for k in kernel_sizes
        ])

    def forward(
        self,
        node_emb: torch.Tensor,   # (N_tiles, D)
        tile_ids: List[int],
        Nx:       int,
        Ny:       int,
    ) -> torch.Tensor:
        """Extract multi-scale features at query tile positions.

        Parameters
        ──────────
        node_emb : Tensor (N_tiles, D)
        tile_ids : list[int]   query tile indices (e.g. city or reachable tiles)
        Nx, Ny   : int

        Returns
        ───────
        Tensor (Q, D * n_scales)
        """
        node_emb = node_emb.float()
        D, dev   = node_emb.shape[-1], node_emb.device
        ids_t    = torch.tensor(tile_ids, dtype=torch.long, device=dev)

        # Reshape board to (1, D, Nx, Ny) for Conv2d
        grid = (
            node_emb
            .view(Nx, Ny, D)
            .permute(2, 0, 1)   # (D, Nx, Ny)
            .unsqueeze(0)       # (1, D, Nx, Ny)
            .contiguous()
        )

        scale_feats: List[torch.Tensor] = []
        for conv_stack in self.conv_stacks:
            out    = conv_stack(grid)          # (1, D, Nx, Ny) — same spatial size
            flat   = out.squeeze(0)            # (D, Nx, Ny)
            flat   = flat.view(D, -1).T        # (N_tiles, D)
            pooled = flat[ids_t]               # (Q, D)
            scale_feats.append(pooled)

        return torch.cat(scale_feats, dim=-1)  # (Q, D * n_scales)



