"""
policy_network.py
────────────────────────────────────────────────────────────────────────────────
Full hierarchical policy network for the Polytopia RL project.

Architecture overview
─────────────────────
One shared GraphTransformerEncoder produces per-tile node embeddings used by
every decision head.  The CriticHead reads a mean-pooled global embedding to
estimate V(s).

The actor side is structured as a hierarchy of independent heads, one per
action type, running bottom-up:

    MoveUnit   :  MovementTargetHead  →  SequenceSelectionHead  →  AT head
    Attack     :  AttackTargetHead    →  SequenceSelectionHead  →  AT head
    CreateUnit :  CreateUnitTypeHead  →  SequenceSelectionHead  →  AT head
    CaptureCity:                          SequenceSelectionHead  →  AT head
    EndTurn    :                                                     AT head

SequenceSelectionHead is a single class shared by all four "middle" selection
steps.  It runs n_layers of transformer blocks with 2D RoPE and optionally
fuses a per-entity entropy scalar from the lower head.

Action sampling — joint trajectory distribution
───────────────────────────────────────────────
Instead of sampling top-down, the policy enumerates every valid action
trajectory τ = (action_type, entity, target), computes its joint probability

    P(τ) = P(AT=t) · P(entity=e | t) · P(target=r | e, t)

assembles a flat Categorical over all trajectories, and samples once.
This gives one complete action with a well-defined log-probability.

Log-probability is computed in factored form for numerical stability:
    log P(τ) = log P(AT) + log P(entity | AT) + log P(target | entity, AT)

Batching
────────
forward()           — single environment; used during worker rollout.
evaluate_actions()  — minibatch; GNN is batched (one pass for B samples),
                      decision heads loop per sample (masks are variable-shape).
compute_values_batch() — critic-only; fully batched; used for GAE refresh.

Snapshot format (make_snapshot)
────────────────────────────────
Stores the minimal information needed to re-score a transition:

    graph              : np.ndarray  (N_tiles, 26)
    Nx, Ny             : int
    player_id          : int
    unit_tile_ids      : list[int]   length n_units
    unit_mvpts         : list[float] length n_units  (needed by MovementTargetHead)
    unit_attack_ranges : list[float] length n_units
    enemy_tile_ids     : list[int]   length n_enemies
    city_tile_ids      : list[int]   length n_cities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game.enums import ActionTypes, UnitType

from RL.models.main_modules        import GraphTransformerEncoder, CriticHead
from RL.models.movement_module     import MovementTargetHead, MovementTargetResult
from RL.models.attack_module       import AttackTargetHead, AttackTargetResult
from RL.models.unit_generation_module import CreateUnitTypeHead, CreateUnitTypeResult
from RL.models.utility_modules     import _mlp, _shannon_entropy

N_ACTION_TYPES: int = len(ActionTypes)
N_UNIT_TYPES:   int = len(UnitType)


# ══════════════════════════════════════════════════════════════════════════════
# Proxy objects
# Used in evaluate_actions to reconstruct game objects from stored snapshots
# without requiring access to live game state.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _TileProxy:
    """Minimal tile interface: exposes only .id"""
    id: int


@dataclass
class _UnitProxy:
    """
    Minimal unit interface expected by MovementTargetHead and AttackTargetHead.
    Exposes .tile.id, .mvpts, .attack_range.
    """
    _tile_id:     int
    mvpts:        float
    attack_range: float

    @property
    def tile(self) -> _TileProxy:
        return _TileProxy(self._tile_id)


@dataclass
class _CityProxy:
    """
    Minimal city interface expected by CreateUnitTypeHead.
    Cities expose .tile_id directly (not .tile.id).
    """
    tile_id: int


# ══════════════════════════════════════════════════════════════════════════════
# Snapshot helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_snapshot(obs: dict, Nx: int, Ny: int, player_id: int) -> dict:
    """
    Capture all information needed to re-score a transition in evaluate_actions.

    Call this BEFORE env.step() so that tile IDs and graph reflect the
    decision state, not the post-step state.

    Parameters
    ──────────
    obs       : dict from EnvWrapper._get_obs()
    Nx, Ny    : board dimensions
    player_id : index of the acting player

    Returns (all lists are length-matched to the corresponding obs lists)
    ──────────────────────────────────────────────────────────────────────
    graph              : np.ndarray (N_tiles, 26)
    Nx, Ny             : int
    player_id          : int
    unit_tile_ids      : list[int]   (n_units,)
    unit_mvpts         : list[float] (n_units,)
    unit_attack_ranges : list[float] (n_units,)
    enemy_tile_ids     : list[int]   (n_enemies,)
    city_tile_ids      : list[int]   (n_cities,)
    """
    return {
        'graph':              np.asarray(obs['partial_graph']).copy(),
        'Nx':                 Nx,
        'Ny':                 Ny,
        'player_id':          player_id,
        'unit_tile_ids':      [u.tile.id          for u in obs['units']],
        'unit_mvpts':         [float(u.mvpts)      for u in obs['units']],
        'unit_attack_ranges': [float(u.attack_range) for u in obs['units']],
        'enemy_tile_ids':     [u.tile.id          for u in obs['enemy_units']],
        'city_tile_ids':      [c.tile_id          for c in obs['cities']],
    }


def _units_from_snap(snap: dict) -> List[_UnitProxy]:
    return [
        _UnitProxy(tid, mv, ar)
        for tid, mv, ar in zip(
            snap['unit_tile_ids'],
            snap['unit_mvpts'],
            snap['unit_attack_ranges'],
        )
    ]


def _enemies_from_snap(snap: dict) -> List[_UnitProxy]:
    """Enemy proxies only need .tile.id; movement stats are unused."""
    return [_UnitProxy(tid, 0.0, 0.0) for tid in snap['enemy_tile_ids']]


def _cities_from_snap(snap: dict) -> List[_CityProxy]:
    return [_CityProxy(tid) for tid in snap['city_tile_ids']]


# ══════════════════════════════════════════════════════════════════════════════
# Unified selection head
# ══════════════════════════════════════════════════════════════════════════════

class SequenceSelectionHead(nn.Module):
    """
    Unified middle head shared across all four action-type lines.

    Operates on a sequence of entity embeddings (units, cities, …) indexed
    at the provided tile IDs in node_emb, and outputs a softmax distribution +
    Shannon entropy over those entities.

    Optionally fuses a per-entity entropy scalar from the lower head, which
    provides "how uncertain is the best sub-action for this entity?" as a
    learned feature signal.

    2-D RoPE is applied to Q and K (not V) before every MHSA block, encoding
    each entity's (row, col) position on the board.

    Parameters
    ──────────
    node_dim     : int   GNN embedding width — must be divisible by 4 for RoPE
    n_heads      : int   attention heads per transformer block
    n_layers     : int   number of transformer blocks  ← depth knob
    mlp_hidden   : int   hidden width for FF sub-layers and the score head
    mlp_depth    : int   hidden layers inside the score head MLP
    fuse_entropy : bool  if True, prepends Linear(node_dim+1 → node_dim) and
                         expects an `entropies` tensor in forward()
    """

    def __init__(
        self,
        node_dim:     int  = 128,
        n_heads:      int  = 4,
        n_layers:     int  = 2,
        mlp_hidden:   int  = 64,
        mlp_depth:    int  = 2,
        fuse_entropy: bool = True,
    ) -> None:
        super().__init__()

        assert node_dim % 4 == 0, (
            f"node_dim ({node_dim}) must be divisible by 4 for 2D RoPE."
        )
        self.node_dim     = node_dim
        self.fuse_entropy = fuse_entropy

        if fuse_entropy: # TODO: Find a more physics involved way to include entropy!
            # Compress (node_emb ‖ entropy_scalar) → node_dim
            self.input_proj = nn.Linear(node_dim + 1, node_dim)

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
        """
        1D rotary position embedding.
        x   : (E, dim)   dim must be even
        pos : (E,)       float positions
        Returns tensor of same shape.
        """
        half  = x.shape[-1] // 2
        dev   = x.device
        theta = 1.0 / (10_000 ** (torch.arange(0, half, device=dev).float() / half))
        ang   = pos.unsqueeze(-1) * theta.unsqueeze(0)   # (E, half)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([
            x1 * ang.cos() - x2 * ang.sin(),
            x1 * ang.sin() + x2 * ang.cos(),
        ], dim=-1)

    def _rope_2d(
        self,
        x:        torch.Tensor,   # (1, E, D)
        tile_ids: List[int],
        Ny:       int,
    ) -> torch.Tensor:
        """
        2D RoPE: row rotation on first D//2 dims, col rotation on second D//2.
        Returns tensor of same shape (1, E, D).
        """
        half = x.shape[-1] // 2
        dev  = x.device
        rows = torch.tensor([t // Ny for t in tile_ids], dtype=torch.float32, device=dev)
        cols = torch.tensor([t  % Ny for t in tile_ids], dtype=torch.float32, device=dev)
        xf   = x.squeeze(0)   # (E, D)
        return torch.cat([
            self._rope_1d(xf[..., :half], rows),
            self._rope_1d(xf[..., half:], cols),
        ], dim=-1).unsqueeze(0)   # (1, E, D)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        node_emb:  torch.Tensor,                     # (N_tiles, D)
        tile_ids:  List[int],                        # entity tile IDs  length E
        Ny:        int,
        entropies: Optional[torch.Tensor] = None,    # (E,) from lower head
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ──────────
        node_emb  : (N_tiles, D)
        tile_ids  : entity tile IDs                  → length E
        Ny        : board width (for 2D RoPE col index)
        entropies : (E,)  per-entity entropy from lower head
                    required when fuse_entropy=True

        Returns
        ───────
        probs   : (E,)   softmax distribution over entities
        entropy : ()     Shannon entropy of the distribution (scalar)
        logits  : (E,)   raw pre-softmax logits (used in evaluate_actions)
        """
        node_emb = node_emb.float()
        dev      = node_emb.device

        ids_t = torch.tensor(tile_ids, dtype=torch.long, device=dev)
        feats = node_emb[ids_t]   # (E, D)

        if self.fuse_entropy and entropies is not None:
            ent_col = entropies.unsqueeze(-1).to(dev)                     # (E, 1)
            feats   = self.input_proj(torch.cat([feats, ent_col], dim=-1))
        # feats : (E, D)

        x = feats.unsqueeze(0)   # (1, E, D) — batch dim for MHSA

        for attn, ff, n1, n2 in zip(
            self.attn_layers, self.ff_layers, self.norms1, self.norms2
        ):
            # RoPE on Q and K only; V receives un-rotated x
            x_rope      = self._rope_2d(x, tile_ids, Ny)    # (1, E, D)
            attn_out, _ = attn(x_rope, x_rope, x)
            x = n1(x + attn_out)
            x = n2(x + ff(x))

        logits  = self.score_head(x.squeeze(0)).squeeze(-1)   # (E,)
        probs   = F.softmax(logits, dim=-1)                   # (E,)
        entropy = _shannon_entropy(probs)                     # scalar ()

        return probs, entropy, logits


# ══════════════════════════════════════════════════════════════════════════════
# Full Policy Network
# ══════════════════════════════════════════════════════════════════════════════

class PolicyNetwork(nn.Module):
    """
    Hierarchical actor-critic policy network for the Polytopia RL project.

    Hyperparameters are read from `cfg` with safe defaults so that the
    existing PPOConfig dataclass requires only the new fields to be added.

    Expected cfg attributes (all optional — defaults shown)
    ───────────────────────────────────────────────────────
    encoder_hidden_dim : int   = 128   GNN hidden width
    encoder_n_heads    : int   = 4     TransformerConv heads
    encoder_depth      : int   = 3     TransformerConv layers
    sel_n_heads        : int   = 4     selection-head MHSA heads
    sel_n_layers       : int   = 2     selection-head transformer depth
    mlp_hidden_dim     : int   = 64    MLP hidden width
    mlp_depth          : int   = 2     MLP hidden layers
    kernel_sizes       : tuple = (9,7,5,3)
    n_conv_layers      : int   = 2
    context_bias       : int   = 4
    """

    def __init__(self, cfg) -> None:
        super().__init__()

        # ── Hyperparameters ────────────────────────────────────────────────
        D          = getattr(cfg, 'encoder_hidden_dim', 128)
        enc_heads  = getattr(cfg, 'encoder_n_heads',    4)
        enc_depth  = getattr(cfg, 'encoder_depth',      3)
        sel_heads  = getattr(cfg, 'sel_n_heads',        4)
        sel_layers = getattr(cfg, 'sel_n_layers',       2)
        mlp_hid    = getattr(cfg, 'mlp_hidden_dim',     64)
        mlp_dep    = getattr(cfg, 'mlp_depth',          2)
        kernels    = getattr(cfg, 'kernel_sizes',       (9, 7, 5, 3))
        n_conv     = getattr(cfg, 'n_conv_layers',      2)
        ctx_bias   = getattr(cfg, 'context_bias',       4)

        # ── Encoder + Critic ───────────────────────────────────────────────
        self.encoder = GraphTransformerEncoder(
            hidden_dim = D,
            n_heads    = enc_heads,
            depth      = enc_depth,
        )
        self.critic = CriticHead(
            hidden_dim = D,
            mlp_hidden = mlp_hid,
            mlp_depth  = mlp_dep,
        )

        # ── Action type head ───────────────────────────────────────────────
        # Simple MLP on the global board embedding → logits over N_ACTION_TYPES.
        # The mask is applied via log-masking before softmax.
        self.action_type_head = _mlp(D, mlp_hid, N_ACTION_TYPES, mlp_dep)

        # ── Movement line ──────────────────────────────────────────────────
        self.move_target = MovementTargetHead(
            node_dim      = D,
            kernel_sizes  = kernels,
            n_conv_layers = n_conv,
            context_bias  = ctx_bias,
            mlp_hidden    = mlp_hid,
            mlp_depth     = mlp_dep,
        )
        # fuse_entropy=True: concatenates per-unit target entropy from move_target
        self.move_unit_sel = SequenceSelectionHead(
            node_dim     = D,
            n_heads      = sel_heads,
            n_layers     = sel_layers,
            mlp_hidden   = mlp_hid,
            mlp_depth    = mlp_dep,
            fuse_entropy = True,
        )

        # ── Attack line ────────────────────────────────────────────────────
        self.attack_target = AttackTargetHead(
            node_dim   = D,
            n_heads    = sel_heads,
            mlp_hidden = mlp_hid,
            mlp_depth  = mlp_dep,
        )
        # fuse_entropy=True: concatenates per-attacker enemy-selection entropy
        self.attack_unit_sel = SequenceSelectionHead(
            node_dim     = D,
            n_heads      = sel_heads,
            n_layers     = sel_layers,
            mlp_hidden   = mlp_hid,
            mlp_depth    = mlp_dep,
            fuse_entropy = True,
        )

        # ── Create unit line ───────────────────────────────────────────────
        self.create_type = CreateUnitTypeHead(
            node_dim      = D,
            n_heads       = sel_heads,
            kernel_sizes  = kernels,
            n_conv_layers = n_conv,
            n_unit_types  = N_UNIT_TYPES,
            mlp_hidden    = mlp_hid,
            mlp_depth     = mlp_dep,
        )
        # fuse_entropy=True: concatenates per-city unit-type entropy
        self.create_city_sel = SequenceSelectionHead(
            node_dim     = D,
            n_heads      = sel_heads,
            n_layers     = sel_layers,
            mlp_hidden   = mlp_hid,
            mlp_depth    = mlp_dep,
            fuse_entropy = True,
        )

        # ── Capture city line ──────────────────────────────────────────────
        # No lower head; fuse_entropy=False since there is nothing to fuse.
        self.capture_sel = SequenceSelectionHead(
            node_dim     = D,
            n_heads      = sel_heads,
            n_layers     = sel_layers,
            mlp_hidden   = mlp_hid,
            mlp_depth    = mlp_dep,
            fuse_entropy = False,
        )

    # ── Device helper ─────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    # ══════════════════════════════════════════════════════════════════════
    # Core shared computation: run all heads for one environment
    # ══════════════════════════════════════════════════════════════════════

    def _run_heads(
        self,
        node_emb:   torch.Tensor,   # (N_tiles, D)
        global_emb: torch.Tensor,   # (1, D)
        mask:       list,
        units:      list,           # real Unit objects or _UnitProxy
        enemies:    list,           # real Unit objects or _UnitProxy
        cities:     list,           # real City objects or _CityProxy
        Nx:         int,
        Ny:         int,
    ) -> dict:
        """
        Run all lower heads and selection heads for a single environment.

        Accepts both real game objects (forward pass) and proxy objects
        (evaluate_actions), since both expose the same attribute interface.

        Returned dict shape notes  (U_m = eligible movers,  U_a = eligible
        attackers,  C = eligible cities,  U_cap = eligible capturers)
        ───────────────────────────────────────────────────────────────────
        at_logits          : (N_ACTION_TYPES,)
        at_probs           : (N_ACTION_TYPES,)   masked + normalised
        avail              : np.ndarray (N_ACTION_TYPES,)  refined availability
        move_target        : MovementTargetResult | None
          .probs           : (U_m, max_R)
          .entropies       : (U_m,)
          .logits          : (U_m, max_R)  padded with -inf
          .reach_mask      : (U_m, max_R)  bool
        move_unit_probs    : (U_m,) | None
        move_unit_logits   : (U_m,) | None
        attack_target      : AttackTargetResult | None
          .probs           : (U_a, max_E)
          .entropies       : (U_a,)
          .logits          : (U_a, max_E)  padded with -inf
          .enemy_mask      : (U_a, max_E)  bool
        attack_unit_probs  : (U_a,) | None
        attack_unit_logits : (U_a,) | None
        create_type        : CreateUnitTypeResult | None
          .probs           : (C, N_UNIT_TYPES)
          .entropies       : (C,)
          .logits          : (C, N_UNIT_TYPES)  -inf at unavailable types
        create_city_probs  : (C,) | None
        create_city_logits : (C,) | None
        capture_unit_indices : list[int]   obs_units indices of eligible capturers
        capture_tile_ids     : list[int]   tile IDs of eligible capturers
        capture_probs      : (U_cap,) | None
        capture_logits     : (U_cap,) | None
        """
        dev = node_emb.device

        # ── 1. Lower heads ─────────────────────────────────────────────────
        move_target_result = (
            self.move_target(node_emb, mask[1], units, Nx, Ny)
            if mask[0][int(ActionTypes.MoveUnit)] else None
        )

        attack_target_result = (
            self.attack_target(node_emb, mask[2], units, enemies, Ny)
            if mask[0][int(ActionTypes.Attack)] else None
        )

        create_type_result = (
            self.create_type(node_emb, mask[3], cities, Nx, Ny)
            if mask[0][int(ActionTypes.CreateUnit)] else None
        )

        # Capture: eligible units are wherever mask[4] == 1
        capture_unit_indices: List[int] = (
            np.where(mask[4] > 0)[0].tolist()
            if mask[0][int(ActionTypes.CaptureCity)] else []
        )
        capture_tile_ids: List[int] = [units[i].tile.id for i in capture_unit_indices]

        # ── 2. Selection heads ─────────────────────────────────────────────
        move_unit_probs    = move_unit_logits    = None
        attack_unit_probs  = attack_unit_logits  = None
        create_city_probs  = create_city_logits  = None
        capture_probs      = capture_logits      = None

        if move_target_result is not None:
            move_unit_probs, _, move_unit_logits = self.move_unit_sel(
                node_emb,
                move_target_result.tile_ids,
                Ny,
                entropies = move_target_result.entropies,
            )
            # move_unit_probs  : (U_m,)
            # move_unit_logits : (U_m,)

        if attack_target_result is not None:
            attack_unit_probs, _, attack_unit_logits = self.attack_unit_sel(
                node_emb,
                attack_target_result.attacker_tile_ids,
                Ny,
                entropies = attack_target_result.entropies,
            )
            # attack_unit_probs  : (U_a,)
            # attack_unit_logits : (U_a,)

        if create_type_result is not None:
            create_city_probs, _, create_city_logits = self.create_city_sel(
                node_emb,
                create_type_result.tile_ids,
                Ny,
                entropies = create_type_result.entropies,
            )
            # create_city_probs  : (C,)
            # create_city_logits : (C,)

        if capture_tile_ids:
            capture_probs, _, capture_logits = self.capture_sel(
                node_emb,
                capture_tile_ids,
                Ny,
            )
            # capture_probs  : (U_cap,)
            # capture_logits : (U_cap,)

        # ── 3. Action type distribution ────────────────────────────────────
        # Refine the mask: zero out any AT whose head returned no valid entities.
        # This ensures the flat joint distribution always sums to 1.
        avail = mask[0].copy().astype(np.float32)
        if move_target_result   is None: avail[int(ActionTypes.MoveUnit)]    = 0.0
        if attack_target_result is None: avail[int(ActionTypes.Attack)]      = 0.0
        if create_type_result   is None: avail[int(ActionTypes.CreateUnit)]  = 0.0
        if not capture_tile_ids:         avail[int(ActionTypes.CaptureCity)] = 0.0
        # EndTurn is always kept if mask[0][EndTurn] was set

        avail_t   = torch.tensor(avail, dtype=torch.float32, device=dev)
        at_logits = self.action_type_head(global_emb.view(-1))           # (N_ACTION_TYPES,)
        at_probs  = F.softmax(
            at_logits + torch.log(avail_t.clamp(min=1e-12)), dim=-1
        )   # (N_ACTION_TYPES,)

        return dict(
            at_logits            = at_logits,
            at_probs             = at_probs,
            avail                = avail,
            move_target          = move_target_result,
            move_unit_probs      = move_unit_probs,
            move_unit_logits     = move_unit_logits,
            attack_target        = attack_target_result,
            attack_unit_probs    = attack_unit_probs,
            attack_unit_logits   = attack_unit_logits,
            create_type          = create_type_result,
            create_city_probs    = create_city_probs,
            create_city_logits   = create_city_logits,
            capture_unit_indices = capture_unit_indices,
            capture_tile_ids     = capture_tile_ids,
            capture_probs        = capture_probs,
            capture_logits       = capture_logits,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Joint trajectory distribution
    # ══════════════════════════════════════════════════════════════════════

    def _build_joint_distribution(
        self,
        heads: dict,
    ) -> Tuple[torch.Tensor, List[list]]:
        """
        Enumerate all valid action trajectories and compute joint probabilities.

        For a trajectory τ = (action_type t, entity e, target r):
            P(τ) = P(AT=t) · P(entity=e | t) · P(target=r | e, t)

        Single-level actions (EndTurn, CaptureCity) omit the target factor.

        The resulting tensor is re-normalised to absorb floating-point drift.

        Returns
        ───────
        joint_probs  : (N_traj,)    joint probabilities; differentiable
        traj_actions : list[list]   decoded action for each trajectory index
                                    aligned 1-to-1 with joint_probs
        """
        at_probs = heads['at_probs']   # (N_ACTION_TYPES,)

        joint_list:   List[torch.Tensor] = []
        traj_actions: List[list]         = []

        # ── EndTurn ────────────────────────────────────────────────────────
        # P(EndTurn)  —  always a single trajectory
        joint_list.append(at_probs[int(ActionTypes.EndTurn)].unsqueeze(0))
        traj_actions.append([ActionTypes.EndTurn])

        # ── MoveUnit  :  P(MOVE) · P(unit_i|MOVE) · P(tile_j|unit_i) ─────
        mt = heads['move_target']
        mu = heads['move_unit_probs']
        if mt is not None and mu is not None:
            p_at = at_probs[int(ActionTypes.MoveUnit)]
            for u_local, (u_idx, reach_ids) in enumerate(
                zip(mt.unit_indices, mt.reachable_ids)
            ):
                p_unit = mu[u_local]   # scalar
                for r_local, tile_id in enumerate(reach_ids):
                    p_tile = mt.probs[u_local, r_local]
                    joint_list.append((p_at * p_unit * p_tile).unsqueeze(0))
                    traj_actions.append([ActionTypes.MoveUnit, u_idx, tile_id])

        # ── Attack  :  P(ATK) · P(attacker_i|ATK) · P(enemy_j|attacker_i) -
        atr = heads['attack_target']
        au  = heads['attack_unit_probs']
        if atr is not None and au is not None:
            p_at = at_probs[int(ActionTypes.Attack)]
            for a_local, (a_idx, enemy_ids) in enumerate(
                zip(atr.attacker_indices, atr.enemy_indices)
            ):
                p_att = au[a_local]
                for e_local, enemy_idx in enumerate(enemy_ids):
                    p_enemy = atr.probs[a_local, e_local]
                    joint_list.append((p_at * p_att * p_enemy).unsqueeze(0))
                    traj_actions.append([ActionTypes.Attack, a_idx, enemy_idx])

        # ── CreateUnit  :  P(CRT) · P(city_c|CRT) · P(unit_type_k|city_c) -
        ct = heads['create_type']
        cc = heads['create_city_probs']
        if ct is not None and cc is not None:
            p_at = at_probs[int(ActionTypes.CreateUnit)]
            for c_local, c_idx in enumerate(ct.city_indices):
                p_city = cc[c_local]
                for ut_idx in range(N_UNIT_TYPES):
                    if not ct.unit_type_mask[c_local, ut_idx]:
                        continue
                    p_type = ct.probs[c_local, ut_idx]
                    joint_list.append((p_at * p_city * p_type).unsqueeze(0))
                    traj_actions.append([ActionTypes.CreateUnit, c_idx, ut_idx])

        # ── CaptureCity  :  P(CAP) · P(unit_i|CAP) ───────────────────────
        cap_p   = heads['capture_probs']
        cap_idx = heads['capture_unit_indices']
        if cap_p is not None and cap_idx:
            p_at = at_probs[int(ActionTypes.CaptureCity)]
            for u_local, u_idx in enumerate(cap_idx):
                p_unit = cap_p[u_local]
                joint_list.append((p_at * p_unit).unsqueeze(0))
                traj_actions.append([ActionTypes.CaptureCity, u_idx])

        joint_probs = torch.cat(joint_list, dim=0)   # (N_traj,)

        # Re-normalise to correct floating-point drift; all operations are
        # differentiable so gradients still flow for the entropy computation.
        joint_probs = joint_probs / joint_probs.sum().clamp(min=1e-12)

        return joint_probs, traj_actions

    # ══════════════════════════════════════════════════════════════════════
    # Factored log-probability (numerically stable, differentiable)
    # ══════════════════════════════════════════════════════════════════════

    def _action_log_prob(
        self,
        action: list,
        heads:  dict,
    ) -> torch.Tensor:
        """
        Compute log P(action) = log P(AT) + log P(entity|AT) + log P(target|…)
        using log_softmax applied directly to logits, which avoids the
        numerical instability of log(softmax(logits)).

        The padding convention in result containers is that logits[i][k] = -inf
        for padded (invalid) positions k, so log_softmax naturally assigns
        -inf to those slots and the valid slice is properly normalised.

        Parameters
        ──────────
        action : list  e.g. [ActionTypes.MoveUnit, unit_obs_idx, tile_id]
        heads  : dict  from _run_heads()

        Returns
        ───────
        log_prob : ()  scalar tensor (differentiable w.r.t. network weights)
        """
        dev   = heads['at_logits'].device
        atype = action[0]
        if not isinstance(atype, ActionTypes):
            atype = ActionTypes(int(atype))

        # ── Action type log-prob ───────────────────────────────────────────
        avail_t = torch.tensor(heads['avail'], dtype=torch.float32, device=dev)
        at_lsm  = F.log_softmax(
            heads['at_logits'] + torch.log(avail_t.clamp(min=1e-12)), dim=-1
        )   # (N_ACTION_TYPES,)
        lp = at_lsm[int(atype)]

        if atype == ActionTypes.EndTurn:
            return lp

        # ── MoveUnit ───────────────────────────────────────────────────────
        if atype == ActionTypes.MoveUnit:
            unit_obs_idx, tile_id = action[1], action[2]
            mt  = heads['move_target']
            mul = heads['move_unit_logits']   # (U_m,)
            # Position of this unit within the eligible list
            u_local = mt.unit_indices.index(unit_obs_idx)
            lp = lp + F.log_softmax(mul, dim=-1)[u_local]
            # Position of the target tile within the unit's reachable list.
            # mt.logits[u_local] : (max_R,) with -inf at padding positions;
            # r_local < R_i so it always indexes a valid slot.
            r_local = mt.reachable_ids[u_local].index(tile_id)
            lp = lp + F.log_softmax(mt.logits[u_local], dim=-1)[r_local]
            return lp

        # ── Attack ─────────────────────────────────────────────────────────
        if atype == ActionTypes.Attack:
            unit_obs_idx, enemy_obs_idx = action[1], action[2]
            atr = heads['attack_target']
            aul = heads['attack_unit_logits']   # (U_a,)
            a_local = atr.attacker_indices.index(unit_obs_idx)
            lp = lp + F.log_softmax(aul, dim=-1)[a_local]
            e_local = atr.enemy_indices[a_local].index(enemy_obs_idx)
            lp = lp + F.log_softmax(atr.logits[a_local], dim=-1)[e_local]
            return lp

        # ── CreateUnit ─────────────────────────────────────────────────────
        if atype == ActionTypes.CreateUnit:
            city_obs_idx    = action[1]
            unit_type_val   = int(action[2])   # integer index into UnitType
            ct  = heads['create_type']
            ccl = heads['create_city_logits']   # (C,)
            c_local = ct.city_indices.index(city_obs_idx)
            lp = lp + F.log_softmax(ccl, dim=-1)[c_local]
            # ct.logits[c_local] : (N_UNIT_TYPES,) with -inf at unavailable types
            lp = lp + F.log_softmax(ct.logits[c_local], dim=-1)[unit_type_val]
            return lp

        # ── CaptureCity ────────────────────────────────────────────────────
        if atype == ActionTypes.CaptureCity:
            unit_obs_idx = action[1]
            capl    = heads['capture_logits']          # (U_cap,)
            cap_idx = heads['capture_unit_indices']    # list[int]
            u_local = cap_idx.index(unit_obs_idx)
            lp = lp + F.log_softmax(capl, dim=-1)[u_local]
            return lp

        raise ValueError(f"Unknown action type: {atype}")

    # ══════════════════════════════════════════════════════════════════════
    # forward — single environment inference (worker rollout)
    # ══════════════════════════════════════════════════════════════════════

    def forward(
        self,
        obs:  dict,
        mask: list,
    ) -> Tuple[list, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample one complete action from the joint trajectory distribution.

        Called inside torch.no_grad() during worker rollout.

        Parameters
        ──────────
        obs  : dict  from EnvWrapper._get_obs()
        mask : list  from EnvWrapper.get_action_mask()

        Returns
        ───────
        action   : list     e.g. [ActionTypes.MoveUnit, 2, 45]
        log_prob : ()       log P(action) — factored, numerically stable
        entropy  : ()       H[joint trajectory distribution]
        value    : ()       V(s)
        """
        graph_np = np.asarray(obs['partial_graph'])
        N_tiles  = graph_np.shape[0]
        Nx = Ny  = int(round(N_tiles ** 0.5))

        # ── Encode ─────────────────────────────────────────────────────────
        node_emb, global_emb = self.encoder.encode(graph_np, Nx, Ny)
        # node_emb   : (N_tiles, D)
        # global_emb : (1, D)

        value = self.critic(global_emb)   # ()

        # ── Run all decision heads ──────────────────────────────────────────
        heads = self._run_heads(
            node_emb, global_emb, mask,
            obs['units'], obs['enemy_units'], obs['cities'],
            Nx, Ny,
        )

        # ── Build flat joint distribution ───────────────────────────────────
        joint_probs, traj_actions = self._build_joint_distribution(heads)
        # joint_probs  : (N_traj,)
        # traj_actions : list of N_traj action lists

        # ── Sample once from the joint distribution ─────────────────────────
        dist        = torch.distributions.Categorical(probs=joint_probs)
        sampled_idx = dist.sample()              # scalar — discrete, detached
        action      = traj_actions[sampled_idx.item()]

        # ── Log-prob via factored form (numerical stability) ─────────────────
        log_prob = self._action_log_prob(action, heads)   # ()

        # ── Entropy of joint distribution ────────────────────────────────────
        entropy = dist.entropy()   # ()

        return action, joint_probs, traj_actions, log_prob, entropy, value

    # ══════════════════════════════════════════════════════════════════════
    # compute_values_batch — critic-only, fully batched
    # ══════════════════════════════════════════════════════════════════════

    def compute_values_batch(
        self,
        obs_snaps: List[dict],
    ) -> torch.Tensor:
        """
        Critic-only forward pass over a stored minibatch of snapshots.
        Used at the start of every PPO epoch to refresh value targets for GAE.

        Parameters
        ──────────
        obs_snaps : list[dict]  length B — from make_snapshot()

        Returns
        ───────
        values : (B,)
        """
        graphs      = [s['graph'] for s in obs_snaps]
        board_sizes = [(s['Nx'], s['Ny']) for s in obs_snaps]
        _, global_embs = self.encoder.encode_batch(graphs, board_sizes)
        # global_embs : (B, D)
        return self.critic(global_embs)   # (B,)

    # ══════════════════════════════════════════════════════════════════════
    # evaluate_actions — PPO update re-scoring
    # ══════════════════════════════════════════════════════════════════════

    def evaluate_actions(
        self,
        obs_snaps: List[dict],
        actions:   List[list],
        masks:     List[list],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-score a minibatch of stored transitions under current network weights.

        Batching strategy
        ─────────────────
        The GNN encoder runs in one batched forward pass over all B samples
        (different board sizes are handled by PyG's Batch collation).

        The decision heads loop per sample: masks are variable-shape across
        environments (different n_units, n_enemies, n_cities per board), so
        they cannot be batched without non-trivial padding logic.  The heads
        themselves already parallelise over the small per-environment entity
        sets (see movement_module.py).

        Parameters
        ──────────
        obs_snaps : list[dict]   length B — from make_snapshot()
        actions   : list[list]   length B — stored from rollout
        masks     : list[list]   length B — stored from rollout

        Returns
        ───────
        log_probs : (B,)   log P(action_b) under current weights (differentiable)
        entropies : (B,)   H[joint distribution] for each sample  (differentiable)
        values    : (B,)   V(s_b) under current weights           (differentiable)
        """
        B = len(obs_snaps)

        # ── 1. Batched GNN encoder pass ─────────────────────────────────────
        graphs      = [s['graph'] for s in obs_snaps]
        board_sizes = [(s['Nx'], s['Ny']) for s in obs_snaps]

        node_embs, global_embs = self.encoder.encode_batch(graphs, board_sizes)
        # node_embs   : list of B tensors, each (N_b, D)
        # global_embs : (B, D)

        values = self.critic(global_embs)   # (B,)

        # ── 2. Per-sample decision heads + scoring ───────────────────────────
        log_probs_list: List[torch.Tensor] = []
        entropies_list: List[torch.Tensor] = []

        for b, (snap, action, mask) in enumerate(zip(obs_snaps, actions, masks)):
            node_emb   = node_embs[b]           # (N_b, D)
            global_emb = global_embs[b:b + 1]   # (1, D)  — slice preserves grad
            Nx, Ny     = snap['Nx'], snap['Ny']

            # Reconstruct lightweight unit/city objects from the stored snapshot
            units   = _units_from_snap(snap)
            enemies = _enemies_from_snap(snap)
            cities  = _cities_from_snap(snap)

            # Run all heads (no GNN re-computation; uses pre-computed node_emb)
            heads = self._run_heads(
                node_emb, global_emb, mask,
                units, enemies, cities, Nx, Ny,
            )

            # ── Log-prob of the stored action (factored, differentiable) ────
            lp = self._action_log_prob(action, heads)
            log_probs_list.append(lp)

            # ── Entropy of the joint distribution ────────────────────────────
            joint_probs, _ = self._build_joint_distribution(heads)
            ent = torch.distributions.Categorical(probs=joint_probs).entropy()
            entropies_list.append(ent)

        log_probs = torch.stack(log_probs_list)   # (B,)
        entropies = torch.stack(entropies_list)   # (B,)

        return log_probs, entropies, values


# ── Parameter summary ─────────────────────────────────────────────────────────

def model_summary(policy: PolicyNetwork) -> None:
    """Print a concise parameter breakdown for the full policy."""
    sections = {
        'GraphTransformerEncoder' : policy.encoder,
        'CriticHead'              : policy.critic,
        'Action type head'        : policy.action_type_head,
        'MovementTargetHead'      : policy.move_target,
        'Move unit selector'      : policy.move_unit_sel,
        'AttackTargetHead'        : policy.attack_target,
        'Attack unit selector'    : policy.attack_unit_sel,
        'CreateUnitTypeHead'      : policy.create_type,
        'Create city selector'    : policy.create_city_sel,
        'Capture selector'        : policy.capture_sel,
    }
    total     = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    print("=" * 60)
    print(f"  {'Module':<32} {'Params':>14}")
    print("-" * 60)
    for name, mod in sections.items():
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name:<32} {n:>14,}")
    print("=" * 60)
    print(f"  {'TOTAL':<32} {total:>14,}")
    print(f"  {'TRAINABLE':<32} {trainable:>14,}")
    print("=" * 60)