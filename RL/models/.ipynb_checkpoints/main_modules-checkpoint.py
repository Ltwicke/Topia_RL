"""
graph_transformer.py
────────────────────────────────────────────────────────────────────────────────
Two standalone modules:

    GraphTransformerEncoder
        Converts a raw board observation (np.ndarray of node features) into
        per-tile node embeddings.  This tensor is the shared input for every
        decision head (movement, attack, create, capture) as well as the critic.

    CriticHead
        Consumes the node embeddings produced by the encoder and outputs a
        scalar state-value estimate V(s) via global mean pooling + MLP.

Typical usage
─────────────
    encoder = GraphTransformerEncoder(cfg)
    critic  = CriticHead(cfg)

    # Single board (inference / worker rollout)
    node_emb, global_emb = encoder.encode(graph_np, Nx, Ny)
    value                = critic(global_emb)

    # Minibatch (PPO update on GPU)
    node_embs, global_embs = encoder.encode_batch(graphs, board_sizes)
    values                 = critic(global_embs)   # (B,)
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import TransformerConv, global_mean_pool

from RL.models.utility_modules import _mlp, _build_grid_edge_index

# ── Constants ──────────────────────────────────────────────────────────────────

IN_FEATS: int = 26   # raw node-feature width from the game board

# ══════════════════════════════════════════════════════════════════════════════
# Module 1 — Graph Transformer Encoder
# ══════════════════════════════════════════════════════════════════════════════

class GraphTransformerEncoder(nn.Module):
    """Encode a raw board observation into per-tile node embeddings.

    Architecture
    ────────────
        input_proj  : Linear(in_feats → hidden_dim)
        depth ×     : TransformerConv  +  residual  +  LayerNorm
        → node_emb  : (N_tiles, hidden_dim)
        → global_emb: (1, hidden_dim)   mean-pooled over all tiles

    No positional encoding is applied.  Spatial awareness is built up
    implicitly through the message-passing layers.

    Edge indices for each board size are built once and cached.

    Parameters
    ──────────
    in_feats   : int   raw node-feature width            (default 26)
    hidden_dim : int   transformer hidden dimension
                       must be divisible by n_heads
    n_heads    : int   attention heads per TransformerConv layer
    depth      : int   number of TransformerConv layers  ← depth knob
    """

    def __init__(
        self,
        in_feats:   int = IN_FEATS,
        hidden_dim: int = 128,
        n_heads:    int = 4,
        depth:      int = 3,
    ) -> None:
        super().__init__()

        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"n_heads ({n_heads})"
            )

        self.hidden_dim = hidden_dim
        head_dim        = hidden_dim // n_heads

        self.input_proj = nn.Linear(in_feats, hidden_dim)

        # beta=True  — learned residual gate per layer
        # concat=True — head_dim × n_heads = hidden_dim at output
        self.tf_layers = nn.ModuleList([
            TransformerConv(
                hidden_dim, head_dim,
                heads=n_heads, concat=True, dropout=0.0, beta=True,
            )
            for _ in range(depth)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(depth)
        ])

        # Edge-index cache: (Nx, Ny) → CPU LongTensor
        self._edge_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        # Dummy buffer — tracks .device across .to() calls
        self.register_buffer("_dev_ref", torch.zeros(1))

    @property
    def device(self) -> torch.device:
        return self._dev_ref.device

    # ── Edge index ─────────────────────────────────────────────────────────

    def _get_edge_index(self, Nx: int, Ny: int) -> torch.Tensor:
        """Return (and lazily build) the cached CPU edge index for (Nx, Ny)."""
        key = (Nx, Ny)
        if key not in self._edge_cache:
            self._edge_cache[key] = _build_grid_edge_index(Nx, Ny)
        return self._edge_cache[key]

    # ── GNN forward ────────────────────────────────────────────────────────

    def _run_layers(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward through all TransformerConv layers with pre-norm residuals."""
        for layer, norm in zip(self.tf_layers, self.norms):
            x = norm(x + layer(x, edge_index))
        return x

    # ── Public API ─────────────────────────────────────────────────────────

    def encode(
        self,
        graph_np: np.ndarray,
        Nx:       int,
        Ny:       int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single board observation.

        Parameters
        ──────────
        graph_np : np.ndarray (N_tiles, in_feats)
        Nx, Ny   : int  board dimensions

        Returns
        ───────
        node_emb   : Tensor (N_tiles, hidden_dim)  — per-tile embeddings
        global_emb : Tensor (1, hidden_dim)         — mean-pooled board repr
        """
        dev        = self.device
        x          = torch.tensor(np.asarray(graph_np),
                                  dtype=torch.float32, device=dev)
        x          = self.input_proj(x)
        edge_index = self._get_edge_index(Nx, Ny).to(dev)
        x          = self._run_layers(x, edge_index)
        global_emb = x.mean(dim=0, keepdim=True)   # (1, hidden_dim)
        return x, global_emb

    def encode_batch(
        self,
        graphs:      List[np.ndarray],
        board_sizes: List[Tuple[int, int]],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Encode a batch of (possibly variable-sized) boards in one GNN pass.

        Boards are collated via PyG's Batch, which handles node-index
        offsetting automatically.

        Parameters
        ──────────
        graphs      : list of np.ndarray, each (N_i, in_feats)
        board_sizes : list of (Nx_i, Ny_i)

        Returns
        ───────
        node_embs   : list of Tensor (N_i, hidden_dim) — one per board
        global_embs : Tensor (B, hidden_dim)            — mean-pooled
        """
        dev = self.device

        data_list = [
            Data(
                x=torch.tensor(np.asarray(g), dtype=torch.float32),
                edge_index=self._get_edge_index(Nx, Ny),
            )
            for g, (Nx, Ny) in zip(graphs, board_sizes)
        ]

        big         = Batch.from_data_list(data_list).to(dev)
        x           = self.input_proj(big.x)
        x           = self._run_layers(x, big.edge_index)

        global_embs = global_mean_pool(x, big.batch)   # (B, hidden_dim)

        sizes     = [np.asarray(g).shape[0] for g in graphs]
        node_embs = list(torch.split(x, sizes, dim=0))

        return node_embs, global_embs


# ══════════════════════════════════════════════════════════════════════════════
# Module 2 — Critic Head
# ══════════════════════════════════════════════════════════════════════════════

class CriticHead(nn.Module):
    """Estimate state value V(s) from a global board embedding.

    Consumes the mean-pooled global embedding produced by
    GraphTransformerEncoder and maps it to a scalar via an MLP.

    Parameters
    ──────────
    hidden_dim : int   must match GraphTransformerEncoder.hidden_dim
    mlp_hidden : int   hidden width of the value MLP
    mlp_depth  : int   hidden layers inside the value MLP
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        mlp_hidden: int = 64,
        mlp_depth:  int = 2,
    ) -> None:
        super().__init__()
        self.value_mlp = _mlp(hidden_dim, mlp_hidden, 1, mlp_depth)

    def forward(self, global_emb: torch.Tensor) -> torch.Tensor:
        """Compute value estimate(s).

        Parameters
        ──────────
        global_emb : Tensor (1, hidden_dim)  or  (B, hidden_dim)

        Returns
        ───────
        Tensor ()   — scalar, if input was (1, hidden_dim)
        Tensor (B,) — batch of scalars, if input was (B, hidden_dim)
        """
        return self.value_mlp(global_emb).squeeze(-1)


# ── Parameter summary utility ──────────────────────────────────────────────────

def encoder_critic_summary(
    encoder: GraphTransformerEncoder,
    critic:  CriticHead,
) -> None:
    """Print a concise parameter count for the encoder and critic."""
    enc_params  = sum(p.numel() for p in encoder.parameters())
    crit_params = sum(p.numel() for p in critic.parameters())
    total       = enc_params + crit_params

    print("=" * 56)
    print(f"  {'Module':<32} {'Params':>10}")
    print("-" * 56)
    print(f"  {'GraphTransformerEncoder':<32} {enc_params:>10,}")
    print(f"    input_proj"
          f"{'':>20} "
          f"{sum(p.numel() for p in encoder.input_proj.parameters()):>10,}")
    for i, (layer, norm) in enumerate(zip(encoder.tf_layers, encoder.norms)):
        n = sum(p.numel() for p in layer.parameters()) + \
            sum(p.numel() for p in norm.parameters())
        print(f"    tf_layer[{i}] + norm{'':>14} {n:>10,}")
    print(f"  {'CriticHead':<32} {crit_params:>10,}")
    print("=" * 56)
    print(f"  {'TOTAL':<32} {total:>10,}")
    print(f"  Node embedding dim : {encoder.hidden_dim}")
    print(f"  Positional enc     : none")
    print("=" * 56)