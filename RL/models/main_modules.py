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

from RL.models.utility_modules import (
    _mlp, _build_grid_edge_index, apply_rope_2d, grid_row_col,
)
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

# Bump whenever the parameter layout changes incompatibly. `_load_checkpoint`
# refuses an older blob outright rather than letting `load_state_dict` do a
# partial or silently-wrong load.
#   2 -> 3 (sc-48): encoder gains scalar_enc/fuse/fuse_norm and loses
#                   scalar_proj; CriticHead is rebuilt around a categorical
#                   V_TERM head, so its parameter names and output width both
#                   change. Nothing meaningful transfers - retrain from scratch.
MODEL_VERSION: int = 3

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
        in_feats:    int   = IN_FEATS,
        hidden_dim:  int   = 128,
        n_heads:     int   = 4,
        depth:       int   = 3,
        scalar_dim:  int   = SCALAR_DIM,
        score_tau:   float = 1342.0,
        scalar_mode: str   = "derived",
        fusion:      str   = "concat",
        attention:   str   = "global",
        readout:     str   = "global_node_mean",
    ) -> None:
        super().__init__()

        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"n_heads ({n_heads})"
            )

        self.hidden_dim = hidden_dim
        self.scalar_dim = scalar_dim
        self.attention  = attention
        self.readout    = readout
        head_dim        = hidden_dim // n_heads

        self.input_proj = nn.Linear(in_feats, hidden_dim)

        if attention == "global":
            # TRUE global self-attention (sc-48). PyG's TransformerConv, despite
            # the class name, masks attention to `edge_index` - it is a
            # message-passing GNN, so information moves one tile per layer and
            # depth bounds the receptive field at 2 tiles. A board valuation in
            # a territory-control game is inherently global, so that
            # representation was not reachable at any affordable depth.
            #
            # Measured (eval/probe_encoder_cost.py): global attention is
            # 1.4-2.3x FASTER and uses less memory than the local encoder here.
            # The O(N^2) vs O(8N) argument is real but irrelevant at N<=256,
            # D=48 - dense attention is two small GEMMs, while sparse scatter is
            # launch-overhead bound.
            if hidden_dim % 4 != 0:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must be divisible by 4 for 2-D "
                    f"RoPE (rows | cols, each rotated in pairs)"
                )
            self.attn_layers = nn.ModuleList([
                nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
                for _ in range(depth)
            ])
            self.norms1 = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(depth)])
            self.norms2 = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(depth)])
            self.ff_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, 2 * hidden_dim), nn.ReLU(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                )
                for _ in range(depth)
            ])
            self.out_norm = nn.LayerNorm(hidden_dim)
        else:
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

        # ── Scalar-state fusion (sc-48) ───────────────────────────────────────
        # `scalar_state` arrives RAW and unnormalised, and own/opp score are
        # O(1e2)-O(1e3) (100/city, 20/controlled tile, 250/park ...). The old
        # path projected that straight through a Linear and ADDED it to the
        # pooled board vector, which measured 133.8:1 in favour of the scalars -
        # a board share of 0.74%. The critic's first Tanh was saturated for 95%
        # of units before training started.
        #
        # ScalarEncoder bounds every feature first; `fusion="concat"` then
        # concatenates and projects instead of adding, and the trailing
        # LayerNorm is the hard guarantee that whatever happens upstream, the
        # critic's input stays unit-scale.
        self.scalar_mode = scalar_mode
        self.fusion      = fusion
        self.scalar_enc  = ScalarEncoder(
            hidden_dim, scalar_dim, score_tau=score_tau, mode=scalar_mode,
        )
        # Under global attention the scalar state is a TOKEN, not something
        # fused into a pooled vector, so `fuse` would be dead parameters that
        # never receive a gradient. Only build it on the local path.
        if attention != "global":
            if fusion == "concat":
                self.fuse      = nn.Linear(2 * hidden_dim, hidden_dim)
                self.fuse_norm = nn.LayerNorm(hidden_dim)
            elif fusion != "add":
                raise ValueError(f"unknown fusion mode {fusion!r}")

        # Readout projection: global-node embedding concatenated with the mean
        # over tiles. The global node alone is the obvious readout (it has
        # attended to every tile, like a [CLS] token), but a single token
        # collapses the whole board into one highly-parameterised weighted sum.
        # Concatenating a plain mean costs one Linear and guarantees a floor of
        # board information that does not depend on attention having learned
        # anything yet - which matters most at init, the regime sc-48 is about.
        if attention == "global" and readout == "global_node_mean":
            self.readout_proj = nn.Linear(2 * hidden_dim, hidden_dim)

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
        """Forward through all TransformerConv layers with POST-norm residuals."""
        for layer, norm in zip(self.tf_layers, self.norms):
            x = norm(x + layer(x, edge_index))
        return x

    # ── Global attention forward ───────────────────────────────────────────

    def _run_global(
        self,
        tiles:    torch.Tensor,             # (B, N, D)  projected tile tokens
        g_tok:    torch.Tensor,             # (B, 1, D)  global node token
        rows:     torch.Tensor,             # (B, N)     float row index
        cols:     torch.Tensor,             # (B, N)     float col index
        pad_mask: Optional[torch.Tensor],   # (B, 1+N)   True = ignore
    ) -> torch.Tensor:
        """Pre-norm transformer over [global_node, tiles]. -> (B, 1+N, D)

        2-D RoPE is applied to the TILE tokens' Q and K only:

          * V is left un-rotated, matching the convention already used by the
            selection heads (`attack_module.py`).
          * The global node carries no board position, so rotating it would
            invent one. It is excluded and attends positionally-neutrally.

        RoPE is not optional here. Without `edge_index` nothing else tells the
        model which tiles are adjacent, and the encoder would degenerate into a
        set transformer over tiles - strictly worse than what it replaced.
        """
        x = torch.cat([g_tok, tiles], dim=1)              # (B, 1+N, D)
        for attn, n1, n2, ff in zip(
            self.attn_layers, self.norms1, self.norms2, self.ff_layers
        ):
            h = n1(x)
            q = h.clone()
            q[:, 1:] = apply_rope_2d(h[:, 1:], rows, cols)
            # q is used for BOTH query and key; h (un-rotated) supplies V.
            x = x + attn(q, q, h, need_weights=False,
                         key_padding_mask=pad_mask)[0]
            x = x + ff(n2(x))
        return self.out_norm(x)

    def _global_token(
        self, scalar: Optional[torch.Tensor], batch: int
    ) -> torch.Tensor:
        """Scalar state as the global node's input token. -> (B, 1, D)

        This is where scalar context enters under global attention, replacing
        the old "project and add to the pooled vector" path. Two consequences
        beyond fixing defect (A): the scalar reaches `node_emb` and therefore
        every downstream selection head - a head choosing what to build can now
        see how many stars there are, which it never could before - and the
        global node's output doubles as the readout.
        """
        if scalar is None:
            return self.input_proj.weight.new_zeros(batch, 1, self.hidden_dim)
        return self.scalar_enc(scalar).reshape(batch, 1, self.hidden_dim)

    def _readout(
        self, tokens: torch.Tensor, pad_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """[global_node, tiles] -> (B, D) pooled board representation."""
        g_out = tokens[:, 0]                                        # (B, D)
        tiles = tokens[:, 1:]                                       # (B, N, D)
        if pad_mask is None:
            mean = tiles.mean(dim=1)
        else:
            valid = (~pad_mask[:, 1:]).unsqueeze(-1).to(tiles.dtype)
            mean  = (tiles * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        return self.readout_proj(torch.cat([g_out, mean], dim=-1))

    # ── Scalar fusion ──────────────────────────────────────────────────────

    def _fuse(
        self, pooled: torch.Tensor, scalar_t: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Combine the pooled board vector with the encoded scalar state.

        pooled : (B, D)     scalar_t : (B, scalar_dim) or None  ->  (B, D)

        Shared by `encode` and `encode_batch` so the two paths cannot drift
        apart; `tests/test_readout_parity.py` pins that they agree.
        """
        if scalar_t is None:
            s = torch.zeros_like(pooled)
        else:
            s = self.scalar_enc(scalar_t)
        if self.fusion == "add":                      # legacy, probes only
            return pooled + s
        return self.fuse_norm(self.fuse(torch.cat([pooled, s], dim=-1)))

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
        dev = self.device
        x   = torch.tensor(np.asarray(graph_np), dtype=torch.float32, device=dev)
        x   = self.input_proj(x)

        scalar = None
        if scalar_state is not None:
            scalar = torch.as_tensor(np.asarray(scalar_state),
                                     dtype=torch.float32, device=dev).reshape(1, -1)

        if self.attention == "global":
            n_tiles   = x.shape[0]
            rows, cols = grid_row_col(n_tiles, Ny, dev)
            g_tok  = self._global_token(scalar, batch=1)             # (1, 1, D)
            tokens = self._run_global(
                x.unsqueeze(0), g_tok,
                rows.unsqueeze(0), cols.unsqueeze(0), pad_mask=None,
            )
            return tokens[0, 1:], self._readout(tokens, None)

        edge_index = self._get_edge_index(Nx, Ny).to(dev)
        x          = self._run_layers(x, edge_index)
        pooled     = x.amax(dim=0, keepdim=True)   # (1, hidden_dim)
        return x, self._fuse(pooled, scalar)

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

        scalars = None
        if scalar_states is not None:
            scalars = torch.as_tensor(
                np.stack([np.asarray(s) for s in scalar_states], axis=0),
                dtype=torch.float32, device=dev,
            )

        if self.attention == "global":
            return self._encode_batch_global(graphs, board_sizes, scalars, dev)

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

        pooled      = global_max_pool(x, big.batch)   # (B, hidden_dim)
        global_embs = self._fuse(pooled, scalars)

        sizes     = [np.asarray(g).shape[0] for g in graphs]
        node_embs = list(torch.split(x, sizes, dim=0))

        return node_embs, global_embs

    def _encode_batch_global(
        self,
        graphs:      List[np.ndarray],
        board_sizes: List[Tuple[int, int]],
        scalars:     Optional[torch.Tensor],
        dev:         torch.device,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Batched global attention over right-padded boards.

        PyG's Batch handles variable board sizes by concatenating into one
        disconnected graph; dense attention cannot, so boards are padded to the
        largest in the minibatch and the padding is masked out of attention and
        out of the mean readout. Board sizes are drawn from `board_size_range`
        and are usually equal within a batch, in which case no padding happens.
        """
        B     = len(graphs)
        sizes = [int(np.asarray(g).shape[0]) for g in graphs]
        N_max = max(sizes)

        x = torch.zeros(B, N_max, self.input_proj.in_features,
                        dtype=torch.float32, device=dev)
        rows = torch.zeros(B, N_max, dtype=torch.float32, device=dev)
        cols = torch.zeros(B, N_max, dtype=torch.float32, device=dev)
        # +1 leading slot for the global node, which is never padding.
        pad_mask = torch.zeros(B, 1 + N_max, dtype=torch.bool, device=dev)

        for i, (g, (Nx, Ny), n) in enumerate(zip(graphs, board_sizes, sizes)):
            x[i, :n] = torch.as_tensor(np.asarray(g), dtype=torch.float32,
                                       device=dev)
            r, c = grid_row_col(n, Ny, dev)
            rows[i, :n], cols[i, :n] = r, c
            pad_mask[i, 1 + n:] = True

        tokens = self._run_global(
            self.input_proj(x), self._global_token(scalars, batch=B),
            rows, cols, pad_mask if N_max != min(sizes) else None,
        )
        node_embs = [tokens[i, 1:1 + n] for i, n in enumerate(sizes)]
        return node_embs, self._readout(
            tokens, pad_mask if N_max != min(sizes) else None
        )


# ══════════════════════════════════════════════════════════════════════════════
# Scalar-state conditioning
# ══════════════════════════════════════════════════════════════════════════════

DERIVED_SCALAR_DIM = 4


class ScalarEncoder(nn.Module):
    """Raw `scalar_state` -> bounded features -> hidden_dim.

    `scalar_state` is `[stars, stars_per_turn, own_score, opp_score, turn_norm]`
    straight out of the env, unnormalised. `player_score_official` is 100/city +
    50/upgrade + 20/controlled tile + 5/uncovered tile + 250/park, so the score
    entries are O(1e2)-O(1e3). Feeding those to a Linear and adding the result
    to a pooled board vector of magnitude ~8 is sc-48 defect (A).

    The derived features are all bounded to roughly [-1, 1]:

      turn                          already turn / max_turns
      tanh((own - opp) / 2*tau)     the reward-aligned score margin
      tanh(stars   / 100)
      tanh(spt     / 100)

    The margin feature is the important one. The terminal reward IS
    sigma(delta / tau) (see EnvWrapper._get_done_and_rewards), and
    tanh(d/2tau) == 2*sigma(d/tau) - 1, so V_TERM becomes very nearly LINEAR in
    an input feature instead of something the critic has to reconstruct from two
    saturating inputs.

    Absolute scores are deliberately NOT passed through separately: the margin
    already carries the value-relevant information, and re-introducing two
    O(1e3) quantities is exactly what caused the problem.

    KNOWN APPROXIMATION: the reward uses `_terminal_score` (official score x
    uncovered_ratio + 15*spt + 7.5*stars), not `player_score_official`, so the
    margin feature approximates the true reward margin. Closing that gap means
    adding `uncovered_ratio` to `scalar_state`, which changes the observation
    contract - its own story.

    `mode="raw"` reproduces the pre-sc-48 behaviour and exists only so the probe
    suite can demonstrate the difference; it is not a supported training config.
    """

    def __init__(
        self,
        hidden_dim: int,
        scalar_dim: int = SCALAR_DIM,
        score_tau:  float = 1342.0,
        mode:       str = "derived",
    ) -> None:
        super().__init__()
        if mode not in ("derived", "raw"):
            raise ValueError(f"unknown scalar mode {mode!r}")
        self.mode    = mode
        self.out_dim = DERIVED_SCALAR_DIM if mode == "derived" else scalar_dim
        # Buffer, not a constant: it travels with the checkpoint, so a
        # terminal_tau retune that does not match the trained model is visible.
        self.register_buffer("score_tau", torch.tensor(float(score_tau)))
        self.proj = nn.Linear(self.out_dim, hidden_dim)

    def features(self, s: torch.Tensor) -> torch.Tensor:
        """(B, scalar_dim) -> (B, DERIVED_SCALAR_DIM), each in ~[-1, 1]."""
        stars, spt, own, opp, turn = s.unbind(-1)
        return torch.stack([
            turn,
            torch.tanh((own - opp) / (2.0 * self.score_tau)),
            torch.tanh(stars / 100.0),
            torch.tanh(spt   / 100.0),
        ], dim=-1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(s) if self.mode == "derived" else s)


# ══════════════════════════════════════════════════════════════════════════════
# Module 2 — Critic Head
# ══════════════════════════════════════════════════════════════════════════════

# Value-stream indices. The critic predicts one value per reward stream so that
# dense shaping can be switched off mid-training without disturbing the terminal
# value function: V_TERM is regressed ONLY on terminal returns and is therefore
# already correct the moment the dense stream is dropped.
V_TERM, V_DENSE = 0, 1
N_VALUE_STREAMS = 2


class HLGaussHead(nn.Module):
    """Categorical value head - HL-Gauss (Farebrother et al. 2024, 2403.03950).

    Predicts a distribution over `n_bins` return bins instead of regressing a
    scalar, and recovers the value as the expectation over bins. Three reasons
    this is the right head here, not a fashionable one:

    1. The terminal return is TRIMODAL: a point mass at 0 (conquest loser), a
       band over (0, 1) (timeout, terminal_weight * sigma(delta/tau)), and a
       spike at conquest_reward. MSE regression against a trimodal target
       converges to the mean of the modes - a value that is never observed.
       That is sc-48's "predicts a constant near the middle" symptom, and a
       scalar head cannot do otherwise. A categorical head can put mass on all
       three.
    2. The support is CLOSED under the return operator. With gamma = 1.0 and
       r_term non-zero only at terminal steps, every n-step return is
       0 + ... + 0 + (r_term or V(s_n)); if V is bounded to [v_min, v_max] then
       so is every lambda-return. The bin range is therefore principled rather
       than tuned - which is exactly the condition the DreamerV3-tricks-for-PPO
       study found missing when two-hot underperformed.
    3. Cross-entropy in nats is comparable across streams, where two raw MSEs on
       differently-scaled returns are not.

    sigma/bin_width = 0.75 is the paper's recommendation: it spreads each
    target over ~6 neighbouring bins, which is what exploits the ordinal
    structure rather than treating bins as unrelated classes.
    """

    def __init__(
        self,
        in_dim:      int,
        v_min:       float,
        v_max:       float,
        n_bins:      int   = 51,
        sigma_ratio: float = 0.75,
        out_gain:    float = 0.01,
    ) -> None:
        super().__init__()
        if not v_max > v_min:
            raise ValueError(f"need v_max > v_min, got [{v_min}, {v_max}]")
        self.n_bins = n_bins
        edges   = torch.linspace(float(v_min), float(v_max), n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.register_buffer("edges", edges)             # (n_bins + 1,)
        self.register_buffer("bin_values", centers)      # (n_bins,)
        self.sigma = sigma_ratio * (float(v_max) - float(v_min)) / n_bins
        self.out   = nn.Linear(in_dim, n_bins)
        # Near-zero logits at init -> near-uniform categorical -> every state
        # maps to ~the support midpoint with a small, smooth, state-dependent
        # deviation. Narrowly distributed but NOT degenerate, which is exactly
        # what sc-48 asks for at initialisation.
        self.out._out_gain = out_gain

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) -> (B, n_bins) logits."""
        return self.out(h)

    def value(self, logits: torch.Tensor) -> torch.Tensor:
        """Logits -> scalar value, bounded to [v_min, v_max] by construction."""
        return (logits.softmax(-1) * self.bin_values).sum(-1)

    def target(self, y: torch.Tensor) -> torch.Tensor:
        """Scalar targets (B,) -> categorical targets (B, n_bins).

        Each target becomes a Gaussian centred on y, integrated over the bins
        via its CDF. Mass falling outside the support is renormalised back in,
        so a target above v_max saturates at the top bin instead of silently
        losing probability mass.
        """
        z   = y.clamp(float(self.edges[0]), float(self.edges[-1])).unsqueeze(-1)
        cdf = torch.special.ndtr((self.edges - z) / self.sigma)   # (B, n_bins+1)
        p   = cdf[..., 1:] - cdf[..., :-1]
        return p / p.sum(-1, keepdim=True).clamp_min(1e-8)

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, self.target(y))


class CriticHead(nn.Module):
    """Per-stream state values V(s) from a global board embedding.

    Stream 0 (V_TERM) is the terminal (win/margin) value; stream 1 (V_DENSE) is
    the dense-shaping value. They SHARE the trunk and split only at the output.

    The sc-74 guarantee - that setting `dense_beta = 0` must not force V_TERM to
    relearn - is a property of the ADVANTAGE weighting, not of the head
    topology: `dense_beta` appears only in the policy advantage, never in either
    value loss, so each stream is regressed on its own return regardless. A
    shared trunk preserves it exactly as well as separate trunks would, at half
    the parameters and with one set of statistics to verify.

    The two streams are deliberately NOT the same kind of head:

      V_TERM  categorical (HLGaussHead) over [0, conquest_reward]. The terminal
              return is trimodal and bounded - the case a categorical head
              handles and a scalar MSE head structurally cannot.
      V_DENSE scalar. The dense return is a sum over an unbounded-length event
              table, so with gamma = 1.0 it grows with episode length and has no
              fixed support to bin over.

    Pre-sc-48 this was `D -> 2D -> 4D -> n_streams` with Tanh, silently ignoring
    the `mlp_hidden`/`mlp_depth` it accepted. That made it 8*D^2 parameters -
    larger than the encoder feeding it, and quadratic in any future widening of
    the encoder. It is now genuinely `mlp_hidden`-wide, uses ReLU (no saturation
    region anywhere in the value path), and LayerNorms before each
    non-linearity so the input scale cannot matter again.

    Parameters
    ----------
    hidden_dim   : int    must match GraphTransformerEncoder.hidden_dim
    mlp_hidden   : int    trunk width (None -> 2 * hidden_dim)
    mlp_depth    : int    number of hidden layers in the trunk
    n_streams    : int    number of reward streams
    v_min, v_max : float  support of the V_TERM categorical head
    n_bins       : int    bins for the V_TERM head
    """

    def __init__(
        self,
        hidden_dim:  int = 128,
        mlp_hidden:  Optional[int] = None,
        mlp_depth:   int = 2,
        n_streams:   int = N_VALUE_STREAMS,
        v_min:       float = 0.0,
        v_max:       float = 2.0,
        n_bins:      int = 51,
        sigma_ratio: float = 0.75,
        out_gain:    float = 0.01,
    ) -> None:
        super().__init__()
        self.n_streams = n_streams
        self.v_min, self.v_max = float(v_min), float(v_max)
        H = int(mlp_hidden) if mlp_hidden else 2 * hidden_dim

        layers: list = [nn.LayerNorm(hidden_dim)]
        d_in = hidden_dim
        for _ in range(max(mlp_depth, 1)):
            layers += [nn.Linear(d_in, H), nn.LayerNorm(H), nn.ReLU()]
            d_in = H
        self.trunk = nn.Sequential(*layers)

        # out_gain=0.01 is the training default: near-uniform logits at init, so
        # every state maps to ~the support midpoint with a small, smooth,
        # state-dependent deviation. That is the requested "narrowly distributed
        # at initialisation" - but it also makes the value spread ~1e-4, which
        # is too close to degenerate for the smoothness probes to measure
        # anything. eval/smoke_critic_init.py therefore builds the policy with
        # out_gain=1.0: same architecture, readable dynamic range.
        self.term_head = HLGaussHead(
            H, v_min=v_min, v_max=v_max, n_bins=n_bins,
            sigma_ratio=sigma_ratio, out_gain=out_gain,
        )
        self.dense_head = nn.Linear(H, 1)
        self.dense_head._out_gain = 1.0

    def forward(self, global_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, D) -> (term_logits (B, n_bins), dense (B, 1)).

        Always batched. The old `(1, D) -> (n_streams,)` squeeze is gone: it
        returned a differently-shaped tensor for B == 1, which would broadcast
        silently when written into the value table. Callers wanting a single row
        index it themselves.
        """
        if global_emb.dim() == 1:
            global_emb = global_emb.unsqueeze(0)
        h = self.trunk(global_emb)
        return self.term_head(h), self.dense_head(h)

    def value(self, term_logits: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        """Head outputs -> (B, n_streams) values in RAW return space."""
        return torch.stack(
            [self.term_head.value(term_logits), dense.squeeze(-1)], dim=-1
        )

    def value_from_emb(self, global_emb: torch.Tensor) -> torch.Tensor:
        """(B, D) -> (B, n_streams) values in raw return space."""
        return self.value(*self(global_emb))

    @torch.no_grad()
    def report(self, global_emb: torch.Tensor, index: int = 0) -> dict:
        """Everything a human wants to see about one state's value.

        Returns numpy, detached, for row `index` of the batch:

            term_value  float   expectation of the categorical head
            term_probs  (n_bins,)  the full distribution
            bin_values  (n_bins,)  the support those probabilities sit on
            dense_value float
            term_mode   float   value of the single most likely bin
            term_std    float   spread of the distribution

        The scalar expectation is what GAE and the renderer's V-hat use, but it
        hides what the categorical head exists to express: the terminal return
        is trimodal, so a confident 1.0 and an even split between 0 and 2 are
        the same number and completely different beliefs. `term_std` and
        `term_mode` are the cheap summary of which one it is.
        """
        term_logits, dense = self(global_emb)
        p    = term_logits[index].softmax(-1)
        bins = self.term_head.bin_values
        mean = float((p * bins).sum())
        var  = float((p * (bins - mean) ** 2).sum())
        return {
            "term_value":  mean,
            "term_probs":  p.detach().cpu().numpy(),
            "bin_values":  bins.detach().cpu().numpy(),
            "dense_value": float(dense[index].squeeze(-1)),
            "term_mode":   float(bins[int(p.argmax())]),
            "term_std":    float(var ** 0.5),
        }

    def stream_losses(
        self, term_logits: torch.Tensor, dense: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """targets (B, n_streams) in RAW return space -> (term_loss, dense_loss).

        term is cross-entropy in nats, dense is MSE in squared return units.
        The two are NOT commensurable, which is why `value_dense_coef` exists
        rather than the plain sum the pre-sc-48 code used.
        """
        term_loss  = self.term_head.loss(term_logits, targets[:, V_TERM])
        dense_loss = F.mse_loss(dense.squeeze(-1), targets[:, V_DENSE])
        return term_loss, dense_loss


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
        #self.predictor = _mlp(node_dim, mlp_hidden, REDUCED_FEAT_DIM, mlp_depth)
        self.predictor    = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.Tanh(),
            nn.Linear(node_dim * 2, node_dim * 4),
            nn.LayerNorm(node_dim * 4),
            nn.Tanh(),
            nn.Linear(node_dim * 4, REDUCED_FEAT_DIM)
        )

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
    if getattr(encoder, "attention", "local") == "global":
        for i, (attn, n1, n2, ff) in enumerate(zip(
            encoder.attn_layers, encoder.norms1, encoder.norms2, encoder.ff_layers
        )):
            n = sum(sum(p.numel() for p in m.parameters())
                    for m in (attn, n1, n2, ff))
            print(f"    attn[{i}] + ff + norms{'':>10} {n:>10,}")
        for nm in ("out_norm", "readout_proj"):
            mod = getattr(encoder, nm, None)
            if mod is not None:
                print(f"    {nm:<20}{'':>6} "
                      f"{sum(p.numel() for p in mod.parameters()):>10,}")
    else:
        for i, (layer, norm) in enumerate(zip(encoder.tf_layers, encoder.norms)):
            n = (sum(p.numel() for p in layer.parameters())
                 + sum(p.numel() for p in norm.parameters()))
            print(f"    tf_layer[{i}] + norm{'':>14} {n:>10,}")
        print(f"    tf_layer[{i}] + norm{'':>14} {n:>10,}")
    print(f"    scalar_enc"
          f"{'':>20} "
          f"{sum(p.numel() for p in encoder.scalar_enc.parameters()):>10,}")
    if (getattr(encoder, "attention", "local") != "global" and getattr(encoder, "fusion", "add") == "concat"):
        n_fuse = (sum(p.numel() for p in encoder.fuse.parameters())
                  + sum(p.numel() for p in encoder.fuse_norm.parameters()))
        print(f"    fuse + norm{'':>19} {n_fuse:>10,}")
    print(f"  {'CriticHead':<32} {crit_params:>10,}")
    print(f"    trunk"
          f"{'':>25} "
          f"{sum(p.numel() for p in critic.trunk.parameters()):>10,}")
    print(f"    term_head (categorical)"
          f"{'':>7} "
          f"{sum(p.numel() for p in critic.term_head.parameters()):>10,}")
    print(f"    dense_head (scalar)"
          f"{'':>11} "
          f"{sum(p.numel() for p in critic.dense_head.parameters()):>10,}")
    print("=" * 56)
    print(f"  {'TOTAL':<32} {total:>10,}")
    print(f"  Node embedding dim : {encoder.hidden_dim}")
    print(f"  Scalar state dim   : {encoder.scalar_dim}"
          f"  (mode={getattr(encoder, 'scalar_mode', 'raw')})")
    is_global = getattr(encoder, "attention", "local") == "global"
    print(f"  Attention          : "
          f"{'global self-attention' if is_global else 'local (TransformerConv)'}")
    print(f"  Scalar fusion      : "
          f"{'global node token' if is_global else getattr(encoder, 'fusion', 'add')}")
    print(f"  Pooling            : "
          f"{'global node + mean tiles' if is_global else 'max'}")
    print(f"  Positional enc     : {'2-D RoPE on Q,K' if is_global else 'none'}")
    print(f"  V_TERM head        : {critic.term_head.n_bins} bins over "
          f"[{critic.v_min}, {critic.v_max}]  sigma={critic.term_head.sigma:.4f}")
    print("=" * 56)
