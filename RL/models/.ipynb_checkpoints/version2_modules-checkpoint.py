"""
version2_modules.py
────────────────────────────────────────────────────────────────────────────────
Decision heads added in V2.0 for the new action types: HealUnit, Upgrade2Vet,
UpgradeCity, and PlaceRoad.

Heads
─────
    HealUnitHead       — pairwise attention with 2D RoPE over eligible units
                         (mask[5]). Mirrors CaptureCityHead.

    Upgrade2VetHead    — pairwise attention with 2D RoPE over eligible units
                         (mask[8]). Mirrors CaptureCityHead.

    UpgradeCityHead    — per-city softmax over the 2 upgrade choices (mask[6]).
                         Lower head; the city itself is then selected by a
                         SequenceSelectionHead instantiated in policy.py
                         (with fuse_entropy controlled from cfg).

    PlaceRoadHead      — per-tile MLP over eligible road tiles (mask[7] > 0).
                         No multi-scale conv, no transformer — keeps with the
                         "simple head" intent in todos.md.

All heads return a `@dataclass` result container in the project style.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from RL.models.utility_modules import _mlp, _shannon_entropy


# ══════════════════════════════════════════════════════════════════════════════
# Result containers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HealUnitResult:
    """All outputs for the HealUnit action."""
    unit_indices: List[int]     # eligible unit indices into obs["units"]
    tile_ids:     List[int]     # tile ID of each eligible unit
    probs:        torch.Tensor  # (U,)  softmax over eligible units
    entropy:      torch.Tensor  # ()    scalar
    logits:       torch.Tensor  # (U,)  raw, for log_prob in evaluate_actions


@dataclass
class Upgrade2VetResult:
    """All outputs for the Upgrade2Vet action."""
    unit_indices: List[int]
    tile_ids:     List[int]
    probs:        torch.Tensor  # (U,)
    entropy:      torch.Tensor  # ()
    logits:       torch.Tensor  # (U,)


@dataclass
class UpgradeCityChoiceResult:
    """
    Per-city upgrade-choice outputs (lower head for UpgradeCity).

    Attributes
    ──────────
    city_indices : list[int]               row indices into obs["cities"] / mask[6]
    tile_ids     : list[int]               tile ID of each eligible city
    probs        : Tensor (C, 2)           softmax over the 2 upgrade choices,
                                           masked to 0 at unavailable choices
    entropies    : Tensor (C,)             per-city Shannon entropy over the 2 choices
    logits       : Tensor (C, 2)           raw, -inf at masked positions
    choice_mask  : Tensor (C, 2)  bool     True where the choice is available
    """
    city_indices: List[int]
    tile_ids:     List[int]
    probs:        torch.Tensor   # (C, 2)
    entropies:    torch.Tensor   # (C,)
    logits:       torch.Tensor   # (C, 2)  -inf at masked
    choice_mask:  torch.Tensor   # (C, 2)  bool


@dataclass
class PlaceRoadResult:
    """All outputs for the PlaceRoad action."""
    tile_ids: List[int]      # eligible tile IDs (rows where mask[7] > 0)
    probs:    torch.Tensor   # (T,)
    entropy:  torch.Tensor   # ()  scalar — eligible for entropy fusion
    logits:   torch.Tensor   # (T,)


# ══════════════════════════════════════════════════════════════════════════════
# Shared mixin: 2D RoPE for unit-selection-style heads
# ══════════════════════════════════════════════════════════════════════════════

def _rope_1d(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """1D rotary position embedding. x: (..., dim) where dim is even."""
    half = x.shape[-1] // 2
    dev  = x.device
    theta = 1.0 / (10_000 ** (torch.arange(0, half, device=dev).float() / half))
    ang   = pos.unsqueeze(-1) * theta.unsqueeze(0)   # (E, half)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([
        x1 * ang.cos() - x2 * ang.sin(),
        x1 * ang.sin() + x2 * ang.cos(),
    ], dim=-1)


def _apply_rope_2d(x: torch.Tensor, tile_ids: List[int], Ny: int) -> torch.Tensor:
    """2D RoPE: row rotation on first D//2 dims, col rotation on second D//2."""
    half = x.shape[-1] // 2
    dev  = x.device
    rows = torch.tensor([t // Ny for t in tile_ids], dtype=torch.float32, device=dev)
    cols = torch.tensor([t  % Ny for t in tile_ids], dtype=torch.float32, device=dev)
    xf   = x.squeeze(0)
    return torch.cat([
        _rope_1d(xf[..., :half], rows),
        _rope_1d(xf[..., half:], cols),
    ], dim=-1).unsqueeze(0)


class _UnitPairwiseSelector(nn.Module):
    """
    Shared core for HealUnitHead and Upgrade2VetHead.

    Pairwise attention with 2D RoPE across all eligible units, scored by a
    small MLP. Identical structure to CaptureCityHead; factored out so the
    two new heads stay short.
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

    def _score_units(
        self,
        node_emb: torch.Tensor,
        tile_ids: List[int],
        Ny:       int,
    ) -> torch.Tensor:
        """Run the pairwise transformer + score MLP. Returns logits (U,)."""
        dev   = node_emb.device
        ids_t = torch.tensor(tile_ids, dtype=torch.long, device=dev)
        x     = node_emb[ids_t].unsqueeze(0)   # (1, U, D)

        for attn, ff, n1, n2 in zip(
            self.attn_layers, self.ff_layers, self.norms1, self.norms2
        ):
            x_rope      = _apply_rope_2d(x, tile_ids, Ny)
            attn_out, _ = attn(x_rope, x_rope, x)
            x = n1(x + attn_out)
            x = n2(x + ff(x))

        return self.score_head(x.squeeze(0)).squeeze(-1)   # (U,)


# ══════════════════════════════════════════════════════════════════════════════
# Module 1 — Heal Unit Head
# ══════════════════════════════════════════════════════════════════════════════

class HealUnitHead(_UnitPairwiseSelector):
    """Select which eligible unit heals.

    A unit is eligible when `mask[5][i] == 1`.  Pairwise attention with 2D RoPE
    runs over all eligible units and a small MLP scores them.
    """

    def forward(
        self,
        node_emb:  torch.Tensor,
        heal_mask: np.ndarray,
        obs_units: list,
        Ny:        int,
    ) -> HealUnitResult | None:
        node_emb = node_emb.float()

        unit_indices = np.where(heal_mask > 0)[0].tolist()
        if not unit_indices:
            return None

        tile_ids = [obs_units[i].tile.id for i in unit_indices]
        logits   = self._score_units(node_emb, tile_ids, Ny)
        probs    = F.softmax(logits, dim=-1)
        entropy  = _shannon_entropy(probs)

        return HealUnitResult(
            unit_indices = unit_indices,
            tile_ids     = tile_ids,
            probs        = probs,
            entropy      = entropy,
            logits       = logits,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Module 2 — Upgrade2Vet Head
# ══════════════════════════════════════════════════════════════════════════════

class Upgrade2VetHead(_UnitPairwiseSelector):
    """Select which eligible unit promotes to veteran.

    A unit is eligible when `mask[8][i] == 1` (3+ kills, not yet veteran,
    not a Giant — see EnvWrapper.get_action_mask).
    """

    def forward(
        self,
        node_emb:    torch.Tensor,
        upgrade_mask: np.ndarray,
        obs_units:   list,
        Ny:          int,
    ) -> Upgrade2VetResult | None:
        node_emb = node_emb.float()

        unit_indices = np.where(upgrade_mask > 0)[0].tolist()
        if not unit_indices:
            return None

        tile_ids = [obs_units[i].tile.id for i in unit_indices]
        logits   = self._score_units(node_emb, tile_ids, Ny)
        probs    = F.softmax(logits, dim=-1)
        entropy  = _shannon_entropy(probs)

        return Upgrade2VetResult(
            unit_indices = unit_indices,
            tile_ids     = tile_ids,
            probs        = probs,
            entropy      = entropy,
            logits       = logits,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Module 3 — UpgradeCity (per-city choice) Head
# ══════════════════════════════════════════════════════════════════════════════

class UpgradeCityHead(nn.Module):
    """Score the 2 upgrade choices for every eligible city.

    A city is eligible if at least one of its two upgrade choices is affordable
    (i.e. `mask[6][c].sum() > 0`).  Output is a per-city softmax over the 2
    choices and a per-city entropy scalar that the city-selection head
    consumes when `fuse_entropy=True`.

    Architecture
    ────────────
    `node_emb[city_tile_ids]` is fed directly into a per-city MLP that scores
    the 2 choices.  Unavailable choices are masked to -inf before softmax.
    """

    def __init__(
        self,
        node_dim:   int = 128,
        mlp_hidden: int = 64,
        mlp_depth:  int = 2,
    ) -> None:
        super().__init__()
        self.node_dim  = node_dim
        self.score_mlp = _mlp(node_dim, mlp_hidden, 2, mlp_depth)

    def forward(
        self,
        node_emb:     torch.Tensor,
        upgrade_mask: np.ndarray,
        obs_cities:   list,
        Ny:           int,
    ) -> UpgradeCityChoiceResult | None:
        """
        Parameters
        ──────────
        node_emb     : Tensor  (N_tiles, D)
        upgrade_mask : ndarray (n_cities, 2)   ← mask[6]
        obs_cities   : list[City]              ← obs["cities"]
        Ny           : int  (kept for API symmetry; unused here)

        Returns
        ───────
        UpgradeCityChoiceResult  or  None if no city has an affordable choice.
        """
        node_emb = node_emb.float()
        dev      = node_emb.device

        eligible: List[dict] = []
        for c_idx, city in enumerate(obs_cities):
            if upgrade_mask[c_idx].sum() == 0:
                continue
            eligible.append(dict(city_idx=c_idx, tile_id=city.tile_id))

        if not eligible:
            return None

        C        = len(eligible)
        tile_ids = [e['tile_id'] for e in eligible]
        ids_t    = torch.tensor(tile_ids, dtype=torch.long, device=dev)

        feats      = node_emb[ids_t]                  # (C, D)
        logits_raw = self.score_mlp(feats)            # (C, 2)

        choice_mask = torch.zeros(C, 2, dtype=torch.bool, device=dev)
        for i, e in enumerate(eligible):
            choice_mask[i] = torch.tensor(
                upgrade_mask[e['city_idx']] > 0, dtype=torch.bool, device=dev
            )

        logits = logits_raw.masked_fill(~choice_mask, float("-inf"))
        probs  = F.softmax(logits, dim=-1)            # (C, 2)

        safe_probs = probs.clamp(min=1e-8)
        entropies  = -(probs * safe_probs.log()).sum(dim=-1)   # (C,)

        return UpgradeCityChoiceResult(
            city_indices = [e['city_idx'] for e in eligible],
            tile_ids     = tile_ids,
            probs        = probs,
            entropies    = entropies,
            logits       = logits,
            choice_mask  = choice_mask,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Module 4 — Place Road Head
# ══════════════════════════════════════════════════════════════════════════════

class PlaceRoadHead(nn.Module):
    """Score every eligible road tile with a small per-tile MLP.

    Eligible tile IDs are the rows where `mask[7][i] == 1` (any visible empty
    field tile under the player's control or unclaimed; affordability already
    enforced by the mask).  A per-tile MLP on `node_emb` produces a flat
    distribution over those tiles plus a scalar Shannon entropy that callers
    may fuse into the action-type head.

    No multi-scale conv and no transformer — kept intentionally lightweight
    per the "simple head" phrasing in todos.md.
    """

    def __init__(
        self,
        node_dim:   int = 128,
        mlp_hidden: int = 64,
        mlp_depth:  int = 2,
    ) -> None:
        super().__init__()
        self.node_dim  = node_dim
        self.score_mlp = _mlp(node_dim, mlp_hidden, 1, mlp_depth)

    def forward(
        self,
        node_emb: torch.Tensor,
        road_mask: np.ndarray,
    ) -> PlaceRoadResult | None:
        """
        Parameters
        ──────────
        node_emb  : Tensor  (N_tiles, D)
        road_mask : ndarray (N_tiles,)    ← mask[7]

        Returns
        ───────
        PlaceRoadResult  or  None if no tile is eligible.
        """
        node_emb = node_emb.float()
        dev      = node_emb.device

        tile_ids = np.where(road_mask > 0)[0].tolist()
        if not tile_ids:
            return None

        ids_t   = torch.tensor(tile_ids, dtype=torch.long, device=dev)
        feats   = node_emb[ids_t]                          # (T, D)
        logits  = self.score_mlp(feats).squeeze(-1)        # (T,)
        probs   = F.softmax(logits, dim=-1)                # (T,)
        entropy = _shannon_entropy(probs)                  # ()

        return PlaceRoadResult(
            tile_ids = tile_ids,
            probs    = probs,
            entropy  = entropy,
            logits   = logits,
        )
