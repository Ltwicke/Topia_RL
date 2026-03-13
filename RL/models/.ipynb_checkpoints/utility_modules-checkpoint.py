from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def _mlp(in_d: int, hid_d: int, out_d: int, depth: int = 2) -> nn.Sequential:
    """MLP with `depth` hidden layers and a pre-output LayerNorm."""
    assert depth >= 1
    layers: list = [nn.Linear(in_d, hid_d), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hid_d, hid_d), nn.ReLU()]
    layers += [nn.LayerNorm(hid_d), nn.Linear(hid_d, out_d)]
    return nn.Sequential(*layers)


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
                    nn.ReLU(),
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



