"""
movement_target_head.py
────────────────────────────────────────────────────────────────────────────────
Parallelised MovementTargetHead for the Polytopia RL project.

All eligible units are processed in three fully-batched stages:

  Stage 1 — TRANSFORMER (U, max_L, D)
      Sequences for every eligible unit are padded to the same length and
      passed through the attention + FF layer in a single forward call.
      A key_padding_mask prevents padded positions from being attended to.

  Stage 2 — MULTI-SCALE CONVOLUTIONS (U, D, Nx, Ny)
      Transformer-updated tokens are scattered back into full board grids,
      yielding one (N_tiles, D) grid per eligible unit.  The grids are
      stacked to (U, D, Nx, Ny) and F.unfold is applied once per kernel
      size across the whole batch.  Reachable-tile patches are gathered
      using batched index tensors.

  Stage 3 — MASKED SOFTMAX + ENTROPY (U, max_R)
      Logits for all units are computed in one score_mlp pass.  Padding
      positions receive −inf so softmax ignores them.  Entropies are
      computed in parallel over the masked probability vectors.

Scatter note
────────────
Scattering the per-unit context updates back into the board grids (step
between stage 1 and 2) still loops over U eligible units.  This is
intentional: scatter_ without a loop requires careful handling of
padding-index collisions that would make the code brittle.  U is bounded
by the number of game units (typically O(1)–O(10)), so the cost is
negligible compared to the batched neural-network operations.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from RL.models.utility_modules import _mlp, _shannon_entropy, MultiScaleConv

# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class MovementTargetResult:
    """
    All target-head outputs for every eligible unit, produced in one pass.

    Attributes
    ──────────
    unit_indices  : list[int]         original row indices in obs_units / mask
    tile_ids      : list[int]         current tile ID per eligible unit
    reachable_ids : list[list[int]]   per-unit reachable tile lists
    probs         : Tensor (U, max_R) softmax distributions, padded with 0
    entropies     : Tensor (U,)       per-unit Shannon entropies
    logits        : Tensor (U, max_R) raw score-MLP output, padded with −inf
    reach_mask    : Tensor (U, max_R) bool — True where the tile is real
    """
    unit_indices:  List[int]
    tile_ids:      List[int]
    reachable_ids: List[List[int]]
    probs:         torch.Tensor   # (U, max_R)
    entropies:     torch.Tensor   # (U,)
    logits:        torch.Tensor   # (U, max_R)  masked (-inf at padding)
    reach_mask:    torch.Tensor   # (U, max_R)  bool


@dataclass
class MovementUnitSelResult:
    """All unit-selection outputs for the movement action."""
    unit_indices: List[int]     # eligible unit indices (into obs_units / mask)
    tile_ids:     List[int]     # current tile ID per eligible unit
    probs:        torch.Tensor  # (U,)  softmax over eligible units
    entropy:      torch.Tensor  # ()    scalar — fed to ActionTypeHead
    logits:       torch.Tensor  # (U,)  raw, for log_prob in evaluate_actions


# ══════════════════════════════════════════════════════════════════════════════

class MovementTargetHead(nn.Module):
    """Score every reachable tile for every eligible unit in a single pass.

    Parameters
    ──────────
    node_dim     : int            width of incoming GNN node embeddings
    n_heads      : int            attention heads in the transformer layer
    kernel_sizes : Sequence[int]  odd ints, e.g. (7, 5, 3)
    context_bias : int            additive margin in the radius formula:
                                      radius = int(mvpts*2 + attack_range) + context_bias
                                  Must be >= max(kernel_sizes) // 2 so that
                                  convolution patches for the furthest reachable
                                  tile always lie within the transformer-updated
                                  region of the board grid.
    mlp_hidden   : int            hidden width of the score MLP
    mlp_depth    : int            hidden layers inside the score MLP
    """

    def __init__(
        self,
        node_dim:     int           = 128,
        n_heads:      int           = 4,
        kernel_sizes: Sequence[int] = (7, 5, 3),
        n_conv_layers:int           = 2,
        context_bias: int           = 4,
        mlp_hidden:   int           = 64,
        mlp_depth:    int           = 2,
    ) -> None:
        super().__init__()

        assert all(k % 2 == 1 for k in kernel_sizes), \
            "All kernel_sizes must be odd integers."
        assert context_bias >= max(kernel_sizes) // 2, (
            f"context_bias ({context_bias}) must be >= max(kernel_sizes)//2 "
            f"({max(kernel_sizes)//2}) to guarantee conv patches lie within "
            f"the transformer-updated region for every reachable tile."
        )

        self.node_dim     = node_dim
        self.kernel_sizes = list(kernel_sizes)
        self.context_bias = context_bias

        # ── Transformer layer (no positional encoding) ─────────────────────
        self.attn  = nn.MultiheadAttention(node_dim, n_heads, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.ReLU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)

        #--------------------

        self.convs = MultiScaleConv(node_dim, kernel_sizes, n_conv_layers)

        # ── Scorer: [transformer feat | conv scale × n_scales] → logit ─────
        score_in       = node_dim * (1 + len(kernel_sizes))
        self.score_mlp = _mlp(score_in, mlp_hidden, 1, mlp_depth)

    # ── Private helpers ────────────────────────────────────────────────────

    def _context_radius(self, unit) -> int:
        return int(float(unit.mvpts) * 2 + float(unit.attack_range)) + self.context_bias

    def _chebyshev_ids(
        self, tile_id: int, radius: int, Nx: int, Ny: int
    ) -> List[int]:
        """All tile IDs within Chebyshev distance `radius` of `tile_id`."""
        row, col = divmod(tile_id, Ny)
        ids: List[int] = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = row + dr, col + dc
                if 0 <= r < Nx and 0 <= c < Ny:
                    ids.append(r * Ny + c)
        return ids

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        node_emb:      torch.Tensor,
        movement_mask: np.ndarray,
        obs_units:     list,
        Nx:            int,
        Ny:            int,
    ) -> MovementTargetResult | None:
        """Process all eligible units in parallel.

        Parameters
        ──────────
        node_emb      : Tensor  (N_tiles, D)
        movement_mask : ndarray (n_units, N_tiles)  ← mask[1]
        obs_units     : list[Unit]                  ← obs["units"]
        Nx, Ny        : int

        Returns
        ───────
        MovementTargetResult  with per-unit probs, entropies, and logits.
        Returns None if no unit is eligible to move.
        """
        dev = node_emb.device
        D   = node_emb.shape[-1]

        # ── 0. Collect metadata for every eligible unit ────────────────────
        # Eligibility: mask row has at least one nonzero entry.
        eligible: List[Dict] = []
        for unit_idx, unit in enumerate(obs_units):
            reach_ids: List[int] = np.where(movement_mask[unit_idx] > 0)[0].tolist()
            if not reach_ids:
                continue
            radius  = self._context_radius(unit)
            ctx_ids = [
                cid for cid in self._chebyshev_ids(unit.tile.id, radius, Nx, Ny)
                if cid != unit.tile.id
            ]
            eligible.append(dict(
                unit_idx  = unit_idx,
                unit      = unit,
                tile_id   = unit.tile.id,
                reach_ids = reach_ids,
                ctx_ids   = ctx_ids,
            ))

        if not eligible:
            return None

        U     = len(eligible)
        max_C = max(len(e['ctx_ids'])   for e in eligible)
        max_R = max(len(e['reach_ids']) for e in eligible)
        max_L = 1 + max_C   # sequence length = unit token + context tokens

        # ─────────────────────────────────────────────────────────────────────
        # Stage 1 — BATCHED TRANSFORMER
        # Build (U, max_L, D) padded sequence tensor and run attention once.
        # ─────────────────────────────────────────────────────────────────────

        # seq_batch      : (U, max_L, D)  — unit token at position 0, then context
        # key_pad_mask   : (U, max_L) bool — True means "ignore this position"
        seq_batch    = torch.zeros(U, max_L, D, device=dev)
        key_pad_mask = torch.ones(U, max_L,    dtype=torch.bool, device=dev)

        for i, e in enumerate(eligible):
            C_i = len(e['ctx_ids'])
            seq_batch[i, 0]       = node_emb[e['tile_id']]
            key_pad_mask[i, 0]    = False   # unit token is always valid
            if C_i > 0:
                ctx_t = torch.tensor(e['ctx_ids'], dtype=torch.long, device=dev)
                seq_batch[i, 1:1 + C_i] = node_emb[ctx_t]
                key_pad_mask[i, 1:1 + C_i] = False

        # Single batched transformer forward pass
        attn_out, _ = self.attn(seq_batch, seq_batch, seq_batch,
                                key_padding_mask=key_pad_mask)
        seq_out  = self.norm1(seq_batch + attn_out)              # (U, max_L, D)
        seq_out  = self.norm2(seq_out   + self.ff(seq_out))      # (U, max_L, D)
        # seq_out[:, 0]   — updated unit token (discarded)
        # seq_out[:, 1:]  — updated context tokens (positions beyond C_i are garbage)

        # ─────────────────────────────────────────────────────────────────────
        # Scatter — write updated context tokens back into full board grids.
        # Tiles outside each unit's context window keep their original value.
        # ─────────────────────────────────────────────────────────────────────

        # updated_grids : (U, N_tiles, D)
        updated_grids = node_emb.unsqueeze(0).expand(U, -1, -1).clone()

        for i, e in enumerate(eligible):
            C_i   = len(e['ctx_ids'])
            ctx_t = torch.tensor(e['ctx_ids'], dtype=torch.long, device=dev)
            updated_grids[i, ctx_t] = seq_out[i, 1:1 + C_i]

        # ─────────────────────────────────────────────────────────────────────
        # Stage 2 — MULTI-SCALE CONVOLUTIONS via MultiScaleConv
        # Called once per eligible unit on its individual updated_grid.
        # ─────────────────────────────────────────────────────────────────────

        # Retrieve transformer-updated features at reachable tile positions
        # and run multi-scale convolutions, collecting results per unit.
        reach_feats_list:  List[torch.Tensor] = []   # each (R_i, D)
        multi_feats_list:  List[torch.Tensor] = []   # each (R_i, D * n_scales)

        for i, e in enumerate(eligible):
            reach_ids = e['reach_ids']
            # Transformer-updated features at reachable positions
            ids_t      = torch.tensor(reach_ids, dtype=torch.long, device=dev)
            reach_feats_list.append(updated_grids[i][ids_t])                # (R_i, D)
            # Multi-scale convolutions on this unit's updated grid
            multi_feats_list.append(
                self.convs(updated_grids[i], reach_ids, Nx, Ny)              # (R_i, D*n_scales)
            )

        # Pad to (U, max_R, ...) for batched score_mlp
        reach_ids_padded = torch.zeros(U, max_R, dtype=torch.long, device=dev)
        reach_mask       = torch.zeros(U, max_R, dtype=torch.bool,  device=dev)
        reach_feats_pad  = torch.zeros(U, max_R, D,                 device=dev)
        multi_feats_pad  = torch.zeros(U, max_R, D * len(self.convs.kernel_sizes), device=dev)

        for i, e in enumerate(eligible):
            R_i = len(e['reach_ids'])
            reach_ids_padded[i, :R_i] = torch.tensor(e['reach_ids'], dtype=torch.long, device=dev)
            reach_mask[i,       :R_i] = True
            reach_feats_pad[i,  :R_i] = reach_feats_list[i]
            multi_feats_pad[i,  :R_i] = multi_feats_list[i]

        # Concatenate: [transformer feat | scale_1 | scale_2 | …]
        combined = torch.cat([reach_feats_pad, multi_feats_pad], dim=-1)    # (U, max_R, D*(1+n_s))

        # Single score_mlp pass over all (unit, tile) pairs at once.
        logits_raw = self.score_mlp(combined).squeeze(-1)                   # (U, max_R)

        # ─────────────────────────────────────────────────────────────────────
        # Stage 3 — MASKED SOFTMAX AND ENTROPY (fully parallel over U)
        # ─────────────────────────────────────────────────────────────────────

        # Mask padding positions to -inf so softmax assigns them zero probability.
        logits_masked = logits_raw.masked_fill(~reach_mask, float("-inf"))
        probs         = F.softmax(logits_masked, dim=-1)              # (U, max_R)

        # Shannon entropy: H = -sum(p * log p), summed over valid tiles only.
        # Padding positions have prob=0 after softmax; clamp avoids log(0).
        safe_probs = probs.clamp(min=1e-8)
        entropies  = -(probs * safe_probs.log()).sum(dim=-1)          # (U,)

        # ── Assemble result ────────────────────────────────────────────────
        return MovementTargetResult(
            unit_indices  = [e['unit_idx']  for e in eligible],
            tile_ids      = [e['tile_id']   for e in eligible],
            reachable_ids = [e['reach_ids'] for e in eligible],
            probs         = probs,           # (U, max_R) — padded with 0
            entropies     = entropies,       # (U,)
            logits        = logits_masked,   # (U, max_R) — padded with -inf
            reach_mask    = reach_mask,      # (U, max_R) — bool
        )



class MovementUnitSelHead(nn.Module):
    """Select which eligible unit to move via pairwise attention with 2D RoPE.

    Parameters
    ──────────
    node_dim   : int   width of incoming GNN node embeddings
    n_heads    : int   attention heads per transformer block
    n_layers   : int   number of transformer blocks  (depth knob)
    mlp_hidden : int   hidden width for FF sub-layers and score head
    mlp_depth  : int   hidden layers inside the score head MLP
    """

    def __init__(
        self,
        node_dim:   int = 128,
        n_heads:    int = 4,
        n_layers:   int = 2,
        mlp_hidden: int = 64,
        mlp_depth:  int = 2,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim

        ## TODO: Find more physics inspired way to include the entropy
        # Fuse node embedding (D) + entropy scalar (1) → D
        self.input_proj = nn.Linear(node_dim + 1, node_dim)

        # Stack of transformer blocks
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(node_dim, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim, node_dim * 2), nn.ReLU(),
                nn.Linear(node_dim * 2, node_dim),
            )
            for _ in range(n_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(n_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(n_layers)])

        self.score_head = _mlp(node_dim, mlp_hidden, 1, mlp_depth)

    # ── 2D RoPE ────────────────────────────────────────────────────────────

    @staticmethod
    def _rope_1d(
        x:   torch.Tensor,   # (U, dim)
        pos: torch.Tensor,   # (U,)  float positions
    ) -> torch.Tensor:
        """Apply 1D rotary position embedding to feature tensor x.

        Rotates pairs of features using sin/cos at frequencies derived
        from the integer position.  dim must be even.
        """
        dim  = x.shape[-1]
        half = dim // 2
        dev  = x.device

        theta  = 1.0 / (10_000 ** (torch.arange(0, half, device=dev).float() / half))
        angles = pos.unsqueeze(-1) * theta.unsqueeze(0)   # (U, half)

        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([
            x1 * angles.cos() - x2 * angles.sin(),
            x1 * angles.sin() + x2 * angles.cos(),
        ], dim=-1)

    def _apply_rope_2d(
        self,
        x:        torch.Tensor,   # (1, U, D)
        tile_ids: List[int],
        Ny:       int,
    ) -> torch.Tensor:
        """Apply 2D RoPE: row rotation on the first half of dims,
        col rotation on the second half.

        Returns tensor of same shape (1, U, D).
        """
        D    = x.shape[-1]
        dev  = x.device
        half = D // 2

        rows = torch.tensor(
            [tid // Ny for tid in tile_ids], dtype=torch.float32, device=dev
        )
        cols = torch.tensor(
            [tid  % Ny for tid in tile_ids], dtype=torch.float32, device=dev
        )

        x_flat   = x.squeeze(0)                              # (U, D)
        x_row    = self._rope_1d(x_flat[..., :half], rows)   # (U, D//2)
        x_col    = self._rope_1d(x_flat[..., half:], cols)   # (U, D//2)
        return torch.cat([x_row, x_col], dim=-1).unsqueeze(0)  # (1, U, D)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        node_emb:      torch.Tensor,
        target_result: MovementTargetResult,
        Ny:            int,
    ) -> MovementUnitSelResult:
        """Select which eligible unit to move.

        Parameters
        ──────────
        node_emb      : Tensor (N_tiles, D)   — critic graph encoder output
        target_result : MovementTargetResult  — output of MovementTargetHead
        Ny            : int                   — board width (for RoPE col index)

        Returns
        ───────
        MovementUnitSelResult
        """
        node_emb   = node_emb.float()
        dev        = node_emb.device
        tile_ids   = target_result.tile_ids
        unit_idxs  = target_result.unit_indices
        entropies  = target_result.entropies   # (U,)
        U          = len(tile_ids)

        # ── 1. Build unit feature matrix ───────────────────────────────────
        # Retrieve node embeddings at each eligible unit's tile.
        ids_t   = torch.tensor(tile_ids, dtype=torch.long, device=dev)
        u_feats = node_emb[ids_t]                                    # (U, D)

        # Concatenate per-unit target entropy as an extra feature.
        ent_col = entropies.unsqueeze(-1).to(dev)                    # (U, 1)
        u_feats = self.input_proj(torch.cat([u_feats, ent_col], dim=-1))
        # (U, D)

        x = u_feats.unsqueeze(0)   # (1, U, D)  — batch dim for MHSA

        # ── 2. Transformer blocks with 2D RoPE on Q and K ──────────────────
        for attn, ff, n1, n2 in zip(
            self.attn_layers, self.ff_layers, self.norms1, self.norms2
        ):
            # RoPE is applied to Q and K only; V uses the un-rotated x.
            x_rope   = self._apply_rope_2d(x, tile_ids, Ny)         # (1, U, D)
            attn_out, _ = attn(x_rope, x_rope, x)                   # Q, K rotated; V plain
            x = n1(x + attn_out)
            x = n2(x + ff(x))

        # ── 3. Score, softmax, entropy ──────────────────────────────────────
        logits  = self.score_head(x.squeeze(0)).squeeze(-1)          # (U,)
        probs   = F.softmax(logits, dim=-1)                          # (U,)
        entropy = _shannon_entropy(probs)                            # scalar

        return MovementUnitSelResult(
            unit_indices = unit_idxs,
            tile_ids     = tile_ids,
            probs        = probs,
            entropy      = entropy,
            logits       = logits,
        )


    