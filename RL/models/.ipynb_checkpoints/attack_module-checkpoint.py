"""
attacking_module.py
────────────────────────────────────────────────────────────────────────────────
Two standalone modules for the attack action in the Polytopia RL project.

    AttackTargetHead
        Lower head.  For every eligible attacker, scores all reachable enemy
        units via a transformer and returns per-attacker softmax distributions
        plus entropy scalars.

    AttackUnitSelHead
        Upper head.  Consumes AttackTargetResult to select which attacker
        acts, using pairwise attention with 2D RoPE and the per-attacker
        entropy as an additional feature.

Attack mask conventions (from EnvWrapper.get_action_mask)
──────────────────────────────────────────────────────────
    mask[2] : ndarray (n_units_player, n_units_opponent)
        mask[2][i, j] = 1  iff  player unit i can attack enemy unit j.
    A player unit i is eligible to attack if any entry in mask[2][i] is 1.

Pipeline
────────
    node_emb, _ = encoder.encode(graph_np, Nx, Ny)

    attack_target_result = AttackTargetHead()(
        node_emb, mask[2], obs["units"], obs["enemy_units"], Ny
    )
    attack_sel_result = AttackUnitSelHead()(
        node_emb, attack_target_result, Ny
    )
    # attack_sel_result.entropy  →  ActionTypeHead

Imports
───────
    Helpers are imported from utility_modules to stay consistent with the
    rest of the codebase.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from RL.models.utility_modules import _mlp, _shannon_entropy


# ══════════════════════════════════════════════════════════════════════════════
# Result containers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AttackTargetResult:
    """
    All target-head outputs for every eligible attacker, produced in one pass.

    Attributes
    ──────────
    attacker_indices  : list[int]          row indices into obs_units / mask[2]
    attacker_tile_ids : list[int]          tile ID of each eligible attacker
    enemy_indices     : list[list[int]]    per-attacker reachable enemy indices
                                           (into obs["enemy_units"])
    probs             : Tensor (U, max_E)  softmax over reachable enemies,
                                           padded with 0
    entropies         : Tensor (U,)        per-attacker Shannon entropy
    logits            : Tensor (U, max_E)  raw score-MLP output,
                                           padded with -inf
    enemy_mask        : Tensor (U, max_E)  bool — True where enemy is real
    """
    attacker_indices:  List[int]
    attacker_tile_ids: List[int]
    enemy_indices:     List[List[int]]
    probs:             torch.Tensor   # (U, max_E)
    entropies:         torch.Tensor   # (U,)
    logits:            torch.Tensor   # (U, max_E)  masked (-inf at padding)
    enemy_mask:        torch.Tensor   # (U, max_E)  bool


@dataclass
class AttackUnitSelResult:
    """All unit-selection outputs for the attack action."""
    attacker_indices:  List[int]    # eligible attacker indices into obs_units
    attacker_tile_ids: List[int]    # tile ID of each eligible attacker
    probs:             torch.Tensor # (U,)  softmax over eligible attackers
    entropy:           torch.Tensor # ()    scalar — fed to ActionTypeHead
    logits:            torch.Tensor # (U,)  raw, for log_prob in evaluate_actions


# ══════════════════════════════════════════════════════════════════════════════
# Module 1 — Attack Target Head
# ══════════════════════════════════════════════════════════════════════════════

class AttackTargetHead(nn.Module):
    """Score every reachable enemy unit for every eligible attacker.

    Architecture (per eligible attacker, run in parallel across U)
    ──────────────────────────────────────────────────────────────
    1.  Build a sequence: [attacker_token | enemy_0_token | … | enemy_E_token]
        Sequences are padded to max_E + 1 across all eligible attackers.
        The attacker token acts as a CLS anchor at position 0.

    2.  Single batched transformer forward pass (U, max_L, D) with a
        key_padding_mask to neutralise padded enemy positions.

    3.  Discard position 0 (attacker anchor).  The updated enemy tokens
        at positions 1 … E_i represent each enemy as seen from attacker i.

    4.  score_mlp → logits (U, max_E) → masked softmax → probs + entropies.

    Note: unlike MovementTargetHead there are no convolutions here.  Enemy
    units are discrete tokens, not spatial tiles, so multi-scale spatial
    pooling has no meaningful analogue.  The transformer cross-attention
    between the attacker anchor and the enemy tokens is sufficient to
    capture relative tactical context.

    Parameters
    ──────────
    node_dim   : int   width of incoming GNN node embeddings
    n_heads    : int   attention heads in the transformer layer
    mlp_hidden : int   hidden width of the score MLP
    mlp_depth  : int   hidden layers inside the score MLP
    """

    def __init__(
        self,
        node_dim:   int = 128,
        n_heads:    int = 4,
        mlp_hidden: int = 64,
        mlp_depth:  int = 2,
    ) -> None:
        super().__init__()

        self.node_dim = node_dim

        # ── Transformer layer (no positional encoding) ─────────────────────
        self.attn  = nn.MultiheadAttention(node_dim, n_heads, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.ReLU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)

        # ── Scorer: updated enemy token → scalar logit ─────────────────────
        self.score_mlp = _mlp(node_dim, mlp_hidden, 1, mlp_depth)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        node_emb:    torch.Tensor,
        attack_mask: np.ndarray,
        obs_units:   list,
        obs_enemies: list,
        Ny:          int,
    ) -> AttackTargetResult | None:
        """Score reachable enemies for every eligible attacker in one pass.

        Parameters
        ──────────
        node_emb    : Tensor  (N_tiles, D)
        attack_mask : ndarray (n_units_player, n_units_opponent)  ← mask[2]
        obs_units   : list[Unit]   ← obs["units"]
        obs_enemies : list[Unit]   ← obs["enemy_units"]
        Ny          : int          board width (unused here, kept for API consistency)

        Returns
        ───────
        AttackTargetResult  or  None if no unit is eligible to attack.
        """
        node_emb = node_emb.float()
        dev      = node_emb.device
        D        = node_emb.shape[-1]

        # ── 0. Collect metadata for every eligible attacker ────────────────
        # Eligibility: at least one reachable enemy in the mask row.
        eligible: List[Dict] = []
        for unit_idx, unit in enumerate(obs_units):
            enemy_ids: List[int] = np.where(attack_mask[unit_idx] > 0)[0].tolist()
            if not enemy_ids:
                continue
            eligible.append(dict(
                unit_idx     = unit_idx,
                unit         = unit,
                tile_id      = unit.tile.id,
                enemy_ids    = enemy_ids,   # indices into obs_enemies
            ))

        if not eligible:
            return None

        U     = len(eligible)
        max_E = max(len(e['enemy_ids']) for e in eligible)
        max_L = 1 + max_E   # attacker token + enemy tokens

        # ── Stage 1 — BATCHED TRANSFORMER ─────────────────────────────────
        # seq_batch    : (U, max_L, D)
        # key_pad_mask : (U, max_L) bool — True = ignore this position
        seq_batch    = torch.zeros(U, max_L, D, device=dev)
        key_pad_mask = torch.ones( U, max_L,    dtype=torch.bool, device=dev)

        for i, e in enumerate(eligible):
            E_i = len(e['enemy_ids'])

            # Attacker anchor at position 0
            seq_batch[i, 0]    = node_emb[e['tile_id']]
            key_pad_mask[i, 0] = False

            # Enemy tokens at positions 1 … E_i
            enemy_tile_ids = [obs_enemies[eid].tile.id for eid in e['enemy_ids']]
            enemy_t        = torch.tensor(enemy_tile_ids, dtype=torch.long, device=dev)
            seq_batch[i, 1:1 + E_i] = node_emb[enemy_t]
            key_pad_mask[i, 1:1 + E_i] = False

        # Single batched transformer forward pass
        attn_out, _ = self.attn(seq_batch, seq_batch, seq_batch,
                                key_padding_mask=key_pad_mask)
        seq_out  = self.norm1(seq_batch + attn_out)         # (U, max_L, D)
        seq_out  = self.norm2(seq_out   + self.ff(seq_out)) # (U, max_L, D)

        # Drop attacker anchor at position 0 — keep only enemy token outputs
        enemy_out = seq_out[:, 1:, :]   # (U, max_E, D)

        # ── Stage 2 — SCORE MLP over all (attacker, enemy) pairs ──────────
        logits_raw = self.score_mlp(enemy_out).squeeze(-1)   # (U, max_E)

        # ── Stage 3 — MASKED SOFTMAX AND ENTROPY ──────────────────────────
        # Build enemy validity mask: True where the enemy slot is real
        enemy_mask = torch.zeros(U, max_E, dtype=torch.bool, device=dev)
        for i, e in enumerate(eligible):
            enemy_mask[i, :len(e['enemy_ids'])] = True

        logits_masked = logits_raw.masked_fill(~enemy_mask, float("-inf"))
        probs         = F.softmax(logits_masked, dim=-1)     # (U, max_E)

        safe_probs = probs.clamp(min=1e-8)
        entropies  = -(probs * safe_probs.log()).sum(dim=-1)  # (U,)

        # ── Assemble result ────────────────────────────────────────────────
        return AttackTargetResult(
            attacker_indices  = [e['unit_idx'] for e in eligible],
            attacker_tile_ids = [e['tile_id']  for e in eligible],
            enemy_indices     = [e['enemy_ids'] for e in eligible],
            probs             = probs,
            entropies         = entropies,
            logits            = logits_masked,
            enemy_mask        = enemy_mask,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Module 2 — Attack Unit Selection Head
# ══════════════════════════════════════════════════════════════════════════════

class AttackUnitSelHead(nn.Module):
    """Select which eligible attacker acts via pairwise attention with 2D RoPE.

    Mirrors MovementUnitSelHead exactly, adapted to AttackTargetResult.
    The per-attacker enemy-selection entropy is concatenated to each
    attacker's node embedding before the transformer, giving the selector
    access to "how contested is each attacker's best target choice?"

    Parameters
    ──────────
    node_dim   : int   width of incoming GNN node embeddings
                       must be divisible by 4 (required by 2D RoPE)
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

        assert node_dim % 4 == 0, (
            f"node_dim ({node_dim}) must be divisible by 4 for 2D RoPE."
        )
        self.node_dim = node_dim

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
    def _rope_1d(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """1D rotary position embedding.

        Parameters
        ──────────
        x   : (U, dim)   feature tensor — dim must be even
        pos : (U,)       float positions

        Returns tensor of same shape.
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
        """2D RoPE: row rotation on first half of dims, col on second half.

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

        x_flat = x.squeeze(0)                               # (U, D)
        x_row  = self._rope_1d(x_flat[..., :half], rows)    # (U, D//2)
        x_col  = self._rope_1d(x_flat[..., half:], cols)    # (U, D//2)
        return torch.cat([x_row, x_col], dim=-1).unsqueeze(0)  # (1, U, D)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        node_emb:      torch.Tensor,
        target_result: AttackTargetResult,
        Ny:            int,
    ) -> AttackUnitSelResult:
        """Select which eligible attacker acts.

        Parameters
        ──────────
        node_emb      : Tensor (N_tiles, D)   — graph encoder output
        target_result : AttackTargetResult    — output of AttackTargetHead
        Ny            : int                   — board width (for RoPE col index)

        Returns
        ───────
        AttackUnitSelResult
        """
        node_emb  = node_emb.float()
        dev       = node_emb.device

        tile_ids  = target_result.attacker_tile_ids
        unit_idxs = target_result.attacker_indices
        entropies = target_result.entropies           # (U,)
        U         = len(tile_ids)

        # ── 1. Build attacker feature matrix ───────────────────────────────
        ids_t   = torch.tensor(tile_ids, dtype=torch.long, device=dev)
        u_feats = node_emb[ids_t]                                     # (U, D)

        # Concatenate per-attacker enemy-selection entropy as a feature
        ent_col = entropies.unsqueeze(-1).to(dev)                     # (U, 1)
        u_feats = self.input_proj(torch.cat([u_feats, ent_col], dim=-1))
        # (U, D)

        x = u_feats.unsqueeze(0)   # (1, U, D)

        # ── 2. Transformer blocks with 2D RoPE on Q and K ──────────────────
        for attn, ff, n1, n2 in zip(
            self.attn_layers, self.ff_layers, self.norms1, self.norms2
        ):
            # RoPE applied to Q and K only; V uses the un-rotated x
            x_rope      = self._apply_rope_2d(x, tile_ids, Ny)       # (1, U, D)
            attn_out, _ = attn(x_rope, x_rope, x)
            x = n1(x + attn_out)
            x = n2(x + ff(x))

        # ── 3. Score, softmax, entropy ──────────────────────────────────────
        logits  = self.score_head(x.squeeze(0)).squeeze(-1)           # (U,)
        probs   = F.softmax(logits, dim=-1)                           # (U,)
        entropy = _shannon_entropy(probs)                             # scalar

        return AttackUnitSelResult(
            attacker_indices  = unit_idxs,
            attacker_tile_ids = tile_ids,
            probs             = probs,
            entropy           = entropy,
            logits            = logits,
        )