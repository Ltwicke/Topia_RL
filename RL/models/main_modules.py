"""
graph_transformer.py
────────────────────────────────────────────────────────────────────────────────
Three standalone modules:

    GraphTransformerEncoder
        Converts a raw board observation (np.ndarray of node features) into
        per-tile node embeddings.  This tensor is the shared input for every
        decision head (movement, attack, create, capture, heal, upgrade-city,
        place-road, upgrade-2-vet) as well as the critic.

        V2.0: an optional `scalar_state` vector (own/opp stars, stars-per-turn,
        scores, normalised turn) is projected into the hidden_dim and *added*
        to the max-pooled global embedding.  Pass `scalar_state=None` to keep
        the V1 behaviour.

    CriticHead
        Consumes the global embedding produced by the encoder and outputs a
        scalar state-value estimate V(s) via an MLP.

    HiddenTileEstimator
        Per-tile FCNN that predicts the un-fogged node-feature vector for
        every tile from its node embedding.  Trained against the full board
        graph as an auxiliary objective; gradients flow back into the encoder.

Typical usage
─────────────
    encoder = GraphTransformerEncoder(cfg)
    critic  = CriticHead(cfg)

    # Single board (inference / worker rollout)
    node_emb, global_emb = encoder.encode(graph_np, Nx, Ny, scalar_state)
    value                = critic(global_emb)

    # Minibatch (PPO update on GPU)
    node_embs, global_embs = encoder.encode_batch(graphs, board_sizes, scalars)
    values                 = critic(global_embs)   # (B,)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import TransformerConv, global_max_pool

from RL.models.utility_modules import _mlp, _build_grid_edge_index
from game.enums import (
    NODE_FEAT_DIM,
    TILE_TYPE_SLICE,
    ROAD_SLICE,
    PLAYER_CTRL_SLICE,
    CITY_SLICE,
    UNIT_STATE_SLICE,
    OWN_TYPE_SLICE,
    OPP_TYPE_SLICE,
    N_CITY_TYPES,
    _PLAYER_CTRL_START,
    _CITY_START,
    _UNIT_START,
    REDUCED_TILE_TYPE_SLICE,
    REDUCED_ROAD_SLICE,
    REDUCED_OPP_CTRL_SLICE,
    REDUCED_CITY_SLICE,
    REDUCED_OPP_UNIT_SLICE,
    REDUCED_FEAT_DIM,
    MAX_CITY_LEVEL_HIDDEN,
)

# ── Constants ──────────────────────────────────────────────────────────────────

IN_FEATS:    int = NODE_FEAT_DIM   # raw node-feature width (live from enums)
SCALAR_DIM:  int = 5               # default width of `scalar_state`

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
        → global_emb: (1, hidden_dim)   max-pooled over all tiles
                                         + scalar_proj(scalar_state) if given

    No spatial positional encoding is applied — message-passing builds it up.
    Edge indices for each board size are built once and cached.

    Parameters
    ──────────
    in_feats   : int   raw node-feature width            (default NODE_FEAT_DIM)
    hidden_dim : int   transformer hidden dimension
                       must be divisible by n_heads
    n_heads    : int   attention heads per TransformerConv layer
    depth      : int   number of TransformerConv layers  ← depth knob
    scalar_dim : int   width of the optional scalar_state vector
    """

    def __init__(
        self,
        in_feats:   int = IN_FEATS,
        hidden_dim: int = 128,
        n_heads:    int = 4,
        depth:      int = 3,
        scalar_dim: int = SCALAR_DIM,
    ) -> None:
        super().__init__()

        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"n_heads ({n_heads})"
            )

        self.hidden_dim = hidden_dim
        self.scalar_dim = scalar_dim
        head_dim        = hidden_dim // n_heads

        self.input_proj = nn.Linear(in_feats, hidden_dim)

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

        # Scalar-state fusion: project per-env scalar features → hidden_dim and add
        # to the (max-pooled) global embedding.  Only used when scalar_state is given.
        self.scalar_proj = nn.Linear(scalar_dim, hidden_dim)

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
        graph_np:     np.ndarray,
        Nx:           int,
        Ny:           int,
        scalar_state: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single board observation.

        Parameters
        ──────────
        graph_np     : np.ndarray (N_tiles, in_feats)
        Nx, Ny       : int  board dimensions
        scalar_state : np.ndarray (scalar_dim,) or None
                       fused into the global embedding when given.

        Returns
        ───────
        node_emb   : Tensor (N_tiles, hidden_dim)  — per-tile embeddings
        global_emb : Tensor (1, hidden_dim)         — max-pooled board repr
                                                      + scalar_proj(scalar)
        """
        dev        = self.device
        x          = torch.tensor(np.asarray(graph_np),
                                  dtype=torch.float32, device=dev)
        x          = self.input_proj(x)
        edge_index = self._get_edge_index(Nx, Ny).to(dev)
        x          = self._run_layers(x, edge_index)
        global_emb = x.amax(dim=0, keepdim=True)   # (1, hidden_dim)

        if scalar_state is not None:
            scalar = torch.as_tensor(np.asarray(scalar_state),
                                     dtype=torch.float32, device=dev).reshape(1, -1)
            global_emb = global_emb + self.scalar_proj(scalar)

        return x, global_emb

    def encode_batch(
        self,
        graphs:        List[np.ndarray],
        board_sizes:   List[Tuple[int, int]],
        scalar_states: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Encode a batch of (possibly variable-sized) boards in one GNN pass.

        Boards are collated via PyG's Batch, which handles node-index
        offsetting automatically.

        Parameters
        ──────────
        graphs        : list of np.ndarray, each (N_i, in_feats)
        board_sizes   : list of (Nx_i, Ny_i)
        scalar_states : list of np.ndarray (scalar_dim,) or None
                        when given, must have length == len(graphs).

        Returns
        ───────
        node_embs   : list of Tensor (N_i, hidden_dim) — one per board
        global_embs : Tensor (B, hidden_dim)            — max-pooled
                                                          + scalar_proj(scalar)
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

        global_embs = global_max_pool(x, big.batch)   # (B, hidden_dim)

        if scalar_states is not None:
            scalars = torch.as_tensor(
                np.stack([np.asarray(s) for s in scalar_states], axis=0),
                dtype=torch.float32, device=dev,
            )
            global_embs = global_embs + self.scalar_proj(scalars)

        sizes     = [np.asarray(g).shape[0] for g in graphs]
        node_embs = list(torch.split(x, sizes, dim=0))

        return node_embs, global_embs


# ══════════════════════════════════════════════════════════════════════════════
# Module 2 — Critic Head
# ══════════════════════════════════════════════════════════════════════════════

class CriticHead(nn.Module):
    """Estimate state value V(s) from a global board embedding.

    Consumes the (max-pooled + scalar-fused) global embedding produced by
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


# ══════════════════════════════════════════════════════════════════════════════
# Module 3 — Hidden Tile Estimator (auxiliary pretraining head)
# ══════════════════════════════════════════════════════════════════════════════

# Reduced output layout — the estimator only predicts opponent-side info,
# since by the rules of the game hidden tiles can only contain opponent
# state (own units, own cities, own ctrl are mathematically zero on hidden
# tiles by construction).  All slice constants live in `game/enums.py`,
# derived dynamically from the enum sizes — adding a new TileType /
# UnitType / city level extends this layout automatically with no edits
# to this file.
#
#   tile_type — N_TILE_TYPES one-hot     → softmax + cross-entropy
#   road      — single binary bit        → sigmoid + BCE-with-logits
#   opp_ctrl  — single bit (1 = opp)     → sigmoid + BCE-with-logits
#   city      — {None, Village, L1..L_cap} softmax (10 dims today)
#               Empty class is explicit (idx 0), so no row-masking is
#               needed in the loss.
#   opp_unit  — {None, UnitType.*} softmax (9 dims today)
#               Empty class is explicit (idx 0), no row-masking either.
_REDUCED_GROUP_SLICES: List[Tuple[str, slice]] = [
    ("tile_type", REDUCED_TILE_TYPE_SLICE),
    ("road",      REDUCED_ROAD_SLICE),
    ("opp_ctrl",  REDUCED_OPP_CTRL_SLICE),
    ("city",      REDUCED_CITY_SLICE),
    ("opp_unit",  REDUCED_OPP_UNIT_SLICE),
]


def _full_to_reduced_target(full: torch.Tensor) -> torch.Tensor:
    """Vectorised transform from the full (N, NODE_FEAT_DIM) feature
    vector to the reduced (N, REDUCED_FEAT_DIM) one-hot target consumed
    by the HiddenTileEstimator loss.

    The input must already be in the current player's POV (i.e. with the
    P2 swap applied if applicable) — same convention as `partial_graph`
    and `EnvWrapper._full_graph_for_player(...)`.
    """
    N   = full.shape[0]
    out = full.new_zeros((N, REDUCED_FEAT_DIM))

    # tile_type, road — pass through.
    out[:, REDUCED_TILE_TYPE_SLICE] = full[:, TILE_TYPE_SLICE]
    out[:, REDUCED_ROAD_SLICE]      = full[:, ROAD_SLICE]

    # opp_ctrl — POV-space: idx 0 of PLAYER_CTRL = own, idx 1 = opp.
    out[:, REDUCED_OPP_CTRL_SLICE.start] = full[:, _PLAYER_CTRL_START + 1]

    # city — collapse to {None, Village, L1..L_cap}.
    village_bit    = full[:, _CITY_START]                                       # (N,)
    opp_city_block = full[:, _CITY_START + 1 + N_CITY_TYPES : _UNIT_START]      # (N, N_CITY_TYPES) — opp's per-level
    has_opp_city   = opp_city_block.sum(dim=-1) > 0.5
    raw_lvl        = opp_city_block.argmax(dim=-1) + 1                          # CityType levels start at 1
    capped_lvl     = torch.clamp(raw_lvl, max=MAX_CITY_LEVEL_HIDDEN)            # 1..L_cap

    none_rows    = (~has_opp_city) & (village_bit < 0.5)
    village_rows = (~has_opp_city) &  (village_bit > 0.5)
    out[none_rows,    REDUCED_CITY_SLICE.start + 0] = 1.0
    out[village_rows, REDUCED_CITY_SLICE.start + 1] = 1.0
    if bool(has_opp_city.any()):
        rows = torch.nonzero(has_opp_city, as_tuple=False).squeeze(-1)
        cols = REDUCED_CITY_SLICE.start + 1 + capped_lvl[has_opp_city]
        out[rows, cols] = 1.0

    # opp_unit — class 0 = None, classes 1.. = unit types.
    opp_unit_block = full[:, OPP_TYPE_SLICE]
    has_opp_unit   = opp_unit_block.sum(dim=-1) > 0.5
    unit_idx       = opp_unit_block.argmax(dim=-1)
    out[~has_opp_unit, REDUCED_OPP_UNIT_SLICE.start + 0] = 1.0
    if bool(has_opp_unit.any()):
        rows = torch.nonzero(has_opp_unit, as_tuple=False).squeeze(-1)
        cols = REDUCED_OPP_UNIT_SLICE.start + 1 + unit_idx[has_opp_unit]
        out[rows, cols] = 1.0

    return out


class HiddenTileEstimator(nn.Module):
    """Per-tile FCNN that predicts opponent-side info on hidden tiles.

    For every tile, the estimator takes the encoder's node embedding
    (D-dim) and emits raw scores of shape (REDUCED_FEAT_DIM,).  The
    output is split into independent groups (one-hot blocks + binary
    bits) which are normalised separately:

        - one-hot groups → softmax along the group axis
        - binary bits    → sigmoid

    The `loss` method computes the auxiliary objective: cross-entropy
    per softmax group + BCE on the road / opp_ctrl bits, summed.  Only
    tiles flagged as currently-hidden by the caller contribute
    (visible tiles already appear in the partial graph and are
    uninformative for this task).

    The full ground-truth target (NODE_FEAT_DIM-wide) is transformed
    internally to the reduced layout via `_full_to_reduced_target`,
    so callers can keep passing the un-fogged board graph without
    knowing about the reduced layout.

    Parameters
    ──────────
    node_dim   : int   width of the encoder's node embedding (== hidden_dim)
    mlp_hidden : int   width of the hidden FC layers
    mlp_depth  : int   number of hidden layers
    """

    OUT_DIM      = REDUCED_FEAT_DIM
    GROUP_SLICES = _REDUCED_GROUP_SLICES

    def __init__(
        self,
        node_dim:   int,
        mlp_hidden: int = 128,
        mlp_depth:  int = 2,
    ) -> None:
        super().__init__()
        self.predictor = _mlp(node_dim, mlp_hidden, REDUCED_FEAT_DIM, mlp_depth)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        """Return raw per-tile reduced-feature scores (no softmax/sigmoid).

        Parameters
        ──────────
        node_emb : Tensor (N, node_dim)

        Returns
        ───────
        Tensor (N, REDUCED_FEAT_DIM) — raw logits for every reduced dim.
        """
        return self.predictor(node_emb)

    def predict_proba(self, node_emb: torch.Tensor) -> torch.Tensor:
        """Apply per-group softmax / per-bit sigmoid to the raw forward."""
        raw = self.forward(node_emb)
        out = torch.zeros_like(raw)
        for name, sl in self.GROUP_SLICES:
            block = raw[:, sl]
            if name in ("road", "opp_ctrl"):
                out[:, sl] = torch.sigmoid(block)
            else:
                out[:, sl] = F.softmax(block, dim=-1)
        return out

    # ── Loss ───────────────────────────────────────────────────────────────

    def loss(
        self,
        pred:        torch.Tensor,
        target:      torch.Tensor,
        hidden_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Auxiliary cross-entropy / BCE loss restricted to hidden tiles.

        Parameters
        ──────────
        pred        : Tensor (N, REDUCED_FEAT_DIM)  — raw output of `forward`
        target      : Tensor (N, NODE_FEAT_DIM)    — un-fogged ground truth
                                                       in the current player's POV;
                                                       transformed internally to
                                                       the reduced layout.
        hidden_mask : Tensor (N,) bool             — True for tiles to learn on

        Returns
        ───────
        Tensor () — sum over hidden tiles of the per-tile group-loss sum.
                    Reduction is `sum` so callers can apply explicit per-tile
                    normalisation (divide by n_hidden) in their own
                    aggregation step.  Returns 0 when `hidden_mask` selects
                    nothing.
        """
        if hidden_mask.dtype != torch.bool:
            hidden_mask = hidden_mask.bool()

        n_hidden = int(hidden_mask.sum().item())
        if n_hidden == 0:
            return pred.sum() * 0.0   # zero, but keeps grad-graph alive

        sub_pred   = pred[hidden_mask]                                   # (M, REDUCED_FEAT_DIM)
        sub_target = _full_to_reduced_target(
            target[hidden_mask].to(sub_pred.dtype)
        )                                                                # (M, REDUCED_FEAT_DIM)

        total = pred.new_zeros(())
        for name, sl in self.GROUP_SLICES:
            logits = sub_pred[:, sl]
            tgt    = sub_target[:, sl]

            if name in ("road", "opp_ctrl"):
                total = total + F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), tgt.squeeze(-1), reduction="sum",
                )
                continue

            # city / opp_unit have an explicit "None" class at idx 0,
            # so every row is a valid target — no row-masking needed.
            class_idx = tgt.argmax(dim=-1)
            total = total + F.cross_entropy(
                logits, class_idx, reduction="sum",
            )

        return total


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
    print(f"    scalar_proj"
          f"{'':>19} "
          f"{sum(p.numel() for p in encoder.scalar_proj.parameters()):>10,}")
    print(f"  {'CriticHead':<32} {crit_params:>10,}")
    print("=" * 56)
    print(f"  {'TOTAL':<32} {total:>10,}")
    print(f"  Node embedding dim : {encoder.hidden_dim}")
    print(f"  Scalar state dim   : {encoder.scalar_dim}")
    print(f"  Pooling            : max")
    print(f"  Positional enc     : none")
    print("=" * 56)
