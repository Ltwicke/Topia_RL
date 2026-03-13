"""
create_unit_module.py
────────────────────────────────────────────────────────────────────────────────
Two standalone modules for the CreateUnit action in the Polytopia RL project.

    CreateUnitTypeHead
        Lower head.  For every eligible city, aggregates spatial context
        from the graph embedding via multi-scale convolutions centred on the
        city tile, refines cross-city context through a transformer layer,
        then outputs a per-city softmax distribution over unit types + entropy.

    CreateCitySelHead
        Middle head.  Consumes CreateUnitTypeResult to select which city
        creates a unit, using pairwise attention with 2D RoPE and the
        per-city unit-type entropy as an additional input feature.

Mask conventions (from EnvWrapper.get_action_mask)
────────────────────────────────────────────────────
    mask[3] : ndarray (n_cities_player, N_UNIT_TYPES)
        mask[3][c, u] = 1  iff  city c can produce unit type u.
    A city c is eligible if any entry in mask[3][c] is 1.

Pipeline
────────
    node_emb, _ = encoder.encode(graph_np, Nx, Ny)

    unit_type_result = CreateUnitTypeHead()(
        node_emb, mask[3], obs["cities"], Nx, Ny
    )
    city_sel_result = CreateCitySelHead()(
        node_emb, unit_type_result, Ny
    )
    # city_sel_result.entropy  →  ActionTypeHead

Imports
───────
    Helpers imported from utility_modules for consistency with the codebase.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from RL.models.utility_modules import _mlp, _shannon_entropy, MultiScaleConv


# ══════════════════════════════════════════════════════════════════════════════
# Result containers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CreateUnitTypeResult:
    """
    All unit-type head outputs for every eligible city, produced in one pass.

    Attributes
    ──────────
    city_indices  : list[int]               row indices into obs["cities"] / mask[3]
    tile_ids      : list[int]               tile ID of each eligible city
    probs         : Tensor (C, N_UNIT_TYPES) softmax over unit types per city,
                                             masked to 0 for unavailable types
    entropies     : Tensor (C,)             per-city Shannon entropy
    logits        : Tensor (C, N_UNIT_TYPES) raw MLP output,
                                             unavailable types masked to -inf
    unit_type_mask: Tensor (C, N_UNIT_TYPES) bool — True where type is available
    """
    city_indices:   List[int]
    tile_ids:       List[int]
    probs:          torch.Tensor   # (C, N_UNIT_TYPES)
    entropies:      torch.Tensor   # (C,)
    logits:         torch.Tensor   # (C, N_UNIT_TYPES)  masked (-inf unavailable)
    unit_type_mask: torch.Tensor   # (C, N_UNIT_TYPES)  bool


@dataclass
class CreateCitySelResult:
    """All city-selection outputs for the CreateUnit action."""
    city_indices: List[int]     # eligible city indices into obs["cities"]
    tile_ids:     List[int]     # tile ID of each eligible city
    probs:        torch.Tensor  # (C,)  softmax over eligible cities
    entropy:      torch.Tensor  # ()    scalar — fed to ActionTypeHead
    logits:       torch.Tensor  # (C,)  raw, for log_prob in evaluate_actions


@dataclass
class CaptureCityResult:
    """All outputs for the capture city action."""
    unit_indices: List[int]     # eligible unit indices into obs["units"]
    tile_ids:     List[int]     # tile ID of each eligible unit
    probs:        torch.Tensor  # (U,)  softmax over eligible units
    entropy:      torch.Tensor  # ()    scalar — fed to ActionTypeHead
    logits:       torch.Tensor  # (U,)  raw, for log_prob in evaluate_actions


# ══════════════════════════════════════════════════════════════════════════════
# Module 1 — Create Unit Type Head
# ══════════════════════════════════════════════════════════════════════════════

class CreateUnitTypeHead(nn.Module):
    """Score every available unit type for every eligible city.

    Architecture (all eligible cities processed in parallel)
    ─────────────────────────────────────────────────────────
    1.  MULTI-SCALE CONVOLUTIONS (C, D, Nx, Ny)
        For each kernel size k in kernel_sizes, F.unfold extracts the k×k
        spatial neighbourhood centred on every city tile from node_emb.
        Each patch is mean-pooled to D dims and projected linearly.
        Outputs from all scales are concatenated: (C, D × n_scales).

    2.  INPUT PROJECTION  (C, D × n_scales) → (C, D)
        A linear layer compresses the concatenated multi-scale features back
        to the working dimension before the transformer.

    3.  CROSS-CITY TRANSFORMER  (1, C, D)
        A single MHSA + FF layer lets every city attend to all other cities,
        building a shared tactical context (e.g. two nearby cities should
        coordinate which unit types to produce).

    4.  UNIT TYPE SCORING  (C, D) → (C, N_UNIT_TYPES)
        An MLP maps each city's updated feature vector to logits over all
        unit types.  Unavailable types (mask[3][c] == 0) are masked to -inf
        before softmax.  Entropy is computed per city.

    Parameters
    ──────────
    node_dim      : int            width of incoming GNN node embeddings
    n_heads       : int            attention heads in the transformer layer
    kernel_sizes  : Sequence[int]  odd ints, largest first e.g. (9, 7, 5, 3)
    n_unit_types  : int            number of distinct unit types (= len(UnitType))
    mlp_hidden    : int            hidden width of the scoring MLP
    mlp_depth     : int            hidden layers inside the scoring MLP
    """

    def __init__(
        self,
        node_dim:     int           = 128,
        n_heads:      int           = 4,
        kernel_sizes: Sequence[int] = (9, 7, 5, 3),
        n_conv_layers:int           = 2,
        n_unit_types: int           = 5,
        mlp_hidden:   int           = 64,
        mlp_depth:    int           = 2,
    ) -> None:
        super().__init__()

        assert all(k % 2 == 1 for k in kernel_sizes), \
            "All kernel_sizes must be odd integers."

        self.node_dim     = node_dim
        self.kernel_sizes = list(kernel_sizes)
        self.n_unit_types = n_unit_types

        self.convs = MultiScaleConv(node_dim, kernel_sizes, n_conv_layers)

        # ── Cross-city transformer layer ───────────────────────────────────
        #self.attn  = nn.MultiheadAttention(node_dim, n_heads, batch_first=True)
        #self.ff    = nn.Sequential(
        #    nn.Linear(node_dim, node_dim * 2),
        #    nn.ReLU(),
        #    nn.Linear(node_dim * 2, node_dim),
        #)
        #self.norm1 = nn.LayerNorm(node_dim)
        #self.norm2 = nn.LayerNorm(node_dim)

        # ── Unit type scorer: city feature → logits over N_UNIT_TYPES ──────
        self.score_mlp = _mlp(node_dim * (1 + len(kernel_sizes)), mlp_hidden, n_unit_types, mlp_depth)


    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        node_emb:    torch.Tensor,
        create_mask: np.ndarray,
        obs_cities:  list,
        Nx:          int,
        Ny:          int,
    ) -> CreateUnitTypeResult | None:
        """Score available unit types for every eligible city.

        Parameters
        ──────────
        node_emb    : Tensor  (N_tiles, D)
        create_mask : ndarray (n_cities_player, N_UNIT_TYPES)  ← mask[3]
        obs_cities  : list[City]   ← obs["cities"]
        Nx, Ny      : int

        Returns
        ───────
        CreateUnitTypeResult  or  None if no city is eligible to create.
        """
        node_emb = node_emb.float()
        dev      = node_emb.device

        # ── 0. Collect eligible cities ─────────────────────────────────────
        # A city is eligible if at least one unit type is available for it.
        eligible: List[Dict] = []
        for city_idx, city in enumerate(obs_cities):
            if create_mask[city_idx].sum() == 0:
                continue
            eligible.append(dict(
                city_idx = city_idx,
                city     = city,
                tile_id  = city.tile_id,
            ))

        if not eligible:
            return None

        C        = len(eligible)
        tile_ids = [e['tile_id'] for e in eligible]
        ids_t    = torch.tensor(tile_ids, dtype=torch.long, device=dev)

        # ── Stage 1 — MULTI-SCALE CONVOLUTIONS ────────────────────────────
        city_feats = self.convs(node_emb, tile_ids, Nx, Ny) # (C, D * n_scales)
        combined = torch.cat([node_emb[ids_t], city_feats], dim=-1)
        

        # ── Stage 3 — CROSS-CITY TRANSFORMER ──────────────────────────────
        # Let cities attend to each other to build shared tactical context.
        #x = city_feats.unsqueeze(0)                 # (1, C, D)
        #attn_out, _ = self.attn(x, x, x)
        #x = self.norm1(x + attn_out)
        #x = self.norm2(x + self.ff(x))
        #city_feats = x.squeeze(0)                   # (C, D)

        # ── Stage 4 — UNIT TYPE SCORING ────────────────────────────────────
        logits_raw = self.score_mlp(combined)     # (C, N_UNIT_TYPES)

        # Build unit type availability mask from create_mask rows
        unit_type_mask = torch.zeros(C, self.n_unit_types, dtype=torch.bool, device=dev)
        for i, e in enumerate(eligible):
            row = create_mask[e['city_idx']]        # (N_UNIT_TYPES,)
            unit_type_mask[i] = torch.tensor(row > 0, dtype=torch.bool, device=dev)

        # Mask unavailable types to -inf, then softmax
        logits_masked = logits_raw.masked_fill(~unit_type_mask, float("-inf"))
        probs         = F.softmax(logits_masked, dim=-1)   # (C, N_UNIT_TYPES)

        safe_probs = probs.clamp(min=1e-8)
        entropies  = -(probs * safe_probs.log()).sum(dim=-1)   # (C,)

        return CreateUnitTypeResult(
            city_indices   = [e['city_idx'] for e in eligible],
            tile_ids       = tile_ids,
            probs          = probs,
            entropies      = entropies,
            logits         = logits_masked,
            unit_type_mask = unit_type_mask,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Module 2 — Create City Selection Head
# ══════════════════════════════════════════════════════════════════════════════

class CreateCitySelHead(nn.Module):
    """Select which eligible city creates a unit.

    Mirrors MovementUnitSelHead and AttackUnitSelHead: node embeddings at
    each city tile are fused with the per-city unit-type entropy from
    CreateUnitTypeHead, then refined by n_layers transformer blocks with
    2D RoPE applied to Q and K.

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

        # Fuse node embedding (D) + unit-type entropy scalar (1) → D
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
        x   : (C, dim)  feature tensor — dim must be even
        pos : (C,)      float positions

        Returns tensor of same shape.
        """
        dim  = x.shape[-1]
        half = dim // 2
        dev  = x.device

        theta  = 1.0 / (10_000 ** (torch.arange(0, half, device=dev).float() / half))
        angles = pos.unsqueeze(-1) * theta.unsqueeze(0)   # (C, half)

        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([
            x1 * angles.cos() - x2 * angles.sin(),
            x1 * angles.sin() + x2 * angles.cos(),
        ], dim=-1)

    def _apply_rope_2d(
        self,
        x:        torch.Tensor,   # (1, C, D)
        tile_ids: List[int],
        Ny:       int,
    ) -> torch.Tensor:
        """2D RoPE: row rotation on first half of dims, col on second half.

        Returns tensor of same shape (1, C, D).
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

        x_flat = x.squeeze(0)                                # (C, D)
        x_row  = self._rope_1d(x_flat[..., :half], rows)    # (C, D//2)
        x_col  = self._rope_1d(x_flat[..., half:], cols)    # (C, D//2)
        return torch.cat([x_row, x_col], dim=-1).unsqueeze(0)   # (1, C, D)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        node_emb:         torch.Tensor,
        unit_type_result: CreateUnitTypeResult,
        Ny:               int,
    ) -> CreateCitySelResult:
        """Select which eligible city creates a unit.

        Parameters
        ──────────
        node_emb         : Tensor (N_tiles, D)     — graph encoder output
        unit_type_result : CreateUnitTypeResult    — output of CreateUnitTypeHead
        Ny               : int                     — board width (for RoPE col index)

        Returns
        ───────
        CreateCitySelResult
        """
        node_emb   = node_emb.float()
        dev        = node_emb.device

        tile_ids   = unit_type_result.tile_ids
        city_idxs  = unit_type_result.city_indices
        entropies  = unit_type_result.entropies   # (C,)
        C          = len(tile_ids)

        # ── 1. Build city feature matrix ───────────────────────────────────
        # Retrieve node embeddings at each eligible city's tile.
        ids_t   = torch.tensor(tile_ids, dtype=torch.long, device=dev)
        c_feats = node_emb[ids_t]                                     # (C, D)

        # Concatenate per-city unit-type entropy as an additional feature.
        ent_col = entropies.unsqueeze(-1).to(dev)                     # (C, 1)
        c_feats = self.input_proj(torch.cat([c_feats, ent_col], dim=-1))
        # (C, D)

        x = c_feats.unsqueeze(0)   # (1, C, D)

        # ── 2. Transformer blocks with 2D RoPE on Q and K ──────────────────
        for attn, ff, n1, n2 in zip(
            self.attn_layers, self.ff_layers, self.norms1, self.norms2
        ):
            # RoPE applied to Q and K only; V uses the un-rotated x
            x_rope      = self._apply_rope_2d(x, tile_ids, Ny)       # (1, C, D)
            attn_out, _ = attn(x_rope, x_rope, x)
            x = n1(x + attn_out)
            x = n2(x + ff(x))

        # ── 3. Score, softmax, entropy ──────────────────────────────────────
        logits  = self.score_head(x.squeeze(0)).squeeze(-1)           # (C,)
        probs   = F.softmax(logits, dim=-1)                           # (C,)
        entropy = _shannon_entropy(probs)                             # scalar

        return CreateCitySelResult(
            city_indices = city_idxs,
            tile_ids     = tile_ids,
            probs        = probs,
            entropy      = entropy,
            logits       = logits,
        )



class CaptureCityHead(nn.Module):
    """Select which eligible unit captures a city.

    A unit is eligible if mask[4][i] == 1.  The head runs pairwise attention
    with 2D RoPE across all eligible units and scores them via a small MLP.

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

    # ── 2D RoPE (identical to other selection heads) ───────────────────────

    @staticmethod
    def _rope_1d(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        dim  = x.shape[-1]
        half = dim // 2
        dev  = x.device
        theta  = 1.0 / (10_000 ** (torch.arange(0, half, device=dev).float() / half))
        angles = pos.unsqueeze(-1) * theta.unsqueeze(0)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([
            x1 * angles.cos() - x2 * angles.sin(),
            x1 * angles.sin() + x2 * angles.cos(),
        ], dim=-1)

    def _apply_rope_2d(
        self, x: torch.Tensor, tile_ids: List[int], Ny: int
    ) -> torch.Tensor:
        D, dev, half = x.shape[-1], x.device, x.shape[-1] // 2
        rows = torch.tensor([tid // Ny for tid in tile_ids], dtype=torch.float32, device=dev)
        cols = torch.tensor([tid  % Ny for tid in tile_ids], dtype=torch.float32, device=dev)
        x_flat = x.squeeze(0)
        return torch.cat([
            self._rope_1d(x_flat[..., :half], rows),
            self._rope_1d(x_flat[..., half:], cols),
        ], dim=-1).unsqueeze(0)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        node_emb:     torch.Tensor,
        capture_mask: np.ndarray,
        obs_units:    list,
        Ny:           int,
    ) -> CaptureCityResult | None:
        """Select which unit captures a city.

        Parameters
        ──────────
        node_emb     : Tensor  (N_tiles, D)
        capture_mask : ndarray (n_units,)   ← mask[4]
        obs_units    : list[Unit]           ← obs["units"]
        Ny           : int

        Returns
        ───────
        CaptureCityResult  or  None if no unit is eligible.
        """
        node_emb = node_emb.float()
        dev      = node_emb.device

        # Eligible units: mask[4][i] == 1
        unit_indices = np.where(capture_mask > 0)[0].tolist()
        if not unit_indices:
            return None

        tile_ids = [obs_units[i].tile.id for i in unit_indices]

        # ── Build feature matrix and run transformer with RoPE ─────────────
        ids_t  = torch.tensor(tile_ids, dtype=torch.long, device=dev)
        x      = node_emb[ids_t].unsqueeze(0)   # (1, U, D)

        for attn, ff, n1, n2 in zip(
            self.attn_layers, self.ff_layers, self.norms1, self.norms2
        ):
            x_rope      = self._apply_rope_2d(x, tile_ids, Ny)
            attn_out, _ = attn(x_rope, x_rope, x)
            x = n1(x + attn_out)
            x = n2(x + ff(x))

        # ── Score, softmax, entropy ─────────────────────────────────────────
        logits  = self.score_head(x.squeeze(0)).squeeze(-1)   # (U,)
        probs   = F.softmax(logits, dim=-1)
        entropy = _shannon_entropy(probs)

        return CaptureCityResult(
            unit_indices = unit_indices,
            tile_ids     = tile_ids,
            probs        = probs,
            entropy      = entropy,
            logits       = logits,
        )


