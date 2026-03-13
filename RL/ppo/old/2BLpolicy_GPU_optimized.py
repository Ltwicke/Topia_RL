"""
BLpolicy.py
────────────────────────────────────────────────────────────────────────────────
GraphSAGE-backed actor-critic for the Polytopia RL project.

GPU-batching redesign (evaluate_actions)
─────────────────────────────────────────
The old implementation looped over every sample individually, launching a
separate MPNN forward pass per graph.  With 256 samples and ~5 µs kernel-launch
overhead each, the GPU sat idle 99 % of the time.

New approach
  1. Single MPNN call on the whole minibatch.
     All B graphs share the same fixed board topology, so the edge_index is
     tiled with per-graph node offsets (block-diagonal trick) — no PyG Batch
     object overhead needed.
       x_all        : (B*N, 26)    → mpnn →  (B*N, mpnn_hidden)
       pe_all       : (B*N, PE_DIM)                        (repeated)
       node_embs    : (B*N, node_dim) → reshaped to (B, N, node_dim)

  2. Global heads run on (B, node_dim) in one matrix multiply.
       global_embs  = node_embs.mean(dim=1)   (B, node_dim)
       action logits: action_type_head(global_embs)  → (B, N_ACTION_TYPES)
       values       : critic_head(global_embs)       → (B,)
       action-type mask applied as log-mask → Categorical → log_prob + entropy

  3. Subheads grouped by action type.
     Samples are partitioned into ≤4 groups (MoveUnit / Attack / CreateUnit /
     CaptureCity).  Within each group the variable-length entity lists (units,
     enemies, cities) are zero-padded to max_count_in_group so that the Linear
     layers inside each MLP still run as a single batched matrix multiply.
     Pad positions are masked to −∞ before sampling distributions.
     move_target   uses ALL N nodes → no padding needed, fully batched.
"""

import torch
import torch.nn as nn
import numpy as np

from torch_geometric.nn import SAGEConv
from game.enums import ActionTypes, UnitType

# ── Dimensions (defaults — overridden by cfg where provided) ──────────────────
_MPNN_DIM_DEFAULT = 128
PE_DIM            = 16     # 8 per spatial axis
IN_FEATS          = 26     # raw node feature dimension

N_ACTION_TYPES = len(ActionTypes)
N_UNIT_TYPES   = len(UnitType)


# ── Building blocks ───────────────────────────────────────────────────────────

class _MPNN(nn.Module):
    """3-layer GraphSAGE: in_dim → hidden → hidden → hidden."""
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.c1  = SAGEConv(in_dim, hidden)
        self.c2  = SAGEConv(hidden, hidden)
        self.c3  = SAGEConv(hidden, hidden)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.act(self.c1(x, edge_index))
        x = self.act(self.c2(x, edge_index))
        x = self.act(self.c3(x, edge_index))
        return x   # (N, hidden)


def _mlp(in_d: int, hid_d: int, out_d: int, depth: int = 2) -> nn.Sequential:
    """
    Variable-depth MLP.
    depth = number of hidden layers (minimum 1).
      depth=1 → Linear(in→hid) ReLU Linear(hid→out)
      depth=2 → + one extra Linear(hid→hid) ReLU
      etc.
    """
    assert depth >= 1, "depth must be >= 1"
    layers: list[nn.Module] = [nn.Linear(in_d, hid_d), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hid_d, hid_d), nn.ReLU()]
    layers.append(nn.Linear(hid_d, out_d))
    return nn.Sequential(*layers)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _masked_categorical(logits: torch.Tensor,
                        mask:   torch.Tensor) -> torch.distributions.Categorical:
    """
    logits : (..., K)   float32 on device
    mask   : (..., K)   float32 on device, 0/1
    Returns a Categorical with invalid actions pushed to −∞.
    """
    return torch.distributions.Categorical(
        logits=logits + torch.log(mask.clamp(min=1e-8))
    )


def _tile_edge_index(edge_index: torch.Tensor,
                     B: int, N: int) -> torch.Tensor:
    """
    Given a single-graph edge_index (2, E), return the block-diagonal
    edge_index for B identical graphs: (2, B*E).

    This is exactly what PyG's Batch.from_data_list does internally, but
    without the Python-side overhead of constructing Data objects.
    """
    E       = edge_index.shape[1]
    offsets = torch.arange(B, device=edge_index.device) \
                   .repeat_interleave(E) * N          # (B*E,)
    tiled   = edge_index.repeat(1, B)                 # (2, B*E)
    return tiled + offsets.unsqueeze(0)               # (2, B*E)


# ── Main policy ───────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        Nx, Ny  = cfg.board_size
        self.Nx = Nx
        self.Ny = Ny

        mpnn_hidden = getattr(cfg, 'mpnn_hidden_dim', _MPNN_DIM_DEFAULT)
        mlp_hidden  = getattr(cfg, 'mlp_hidden_dim',  128)
        mlp_depth   = getattr(cfg, 'mlp_depth',        2)
        node_dim    = mpnn_hidden + PE_DIM

        self._node_dim   = node_dim
        self._mpnn_hidden = mpnn_hidden

        # ── Backbone ──────────────────────────────────────────────────────
        self.mpnn = _MPNN(IN_FEATS, mpnn_hidden)

        # ── Positional encoding ───────────────────────────────────────────
        self.row_emb = nn.Embedding(Nx, PE_DIM // 2)
        self.col_emb = nn.Embedding(Ny, PE_DIM // 2)

        # ── Global heads ──────────────────────────────────────────────────
        self.action_type_head = _mlp(node_dim, mlp_hidden, N_ACTION_TYPES, mlp_depth)
        self.critic_head      = _mlp(node_dim, mlp_hidden, 1,              mlp_depth)

        # ── Subheads (entity-level) ───────────────────────────────────────
        self.move_unit_sel    = _mlp(node_dim, mlp_hidden, 1,           mlp_depth)
        self.move_target      = _mlp(node_dim, mlp_hidden, 1,           mlp_depth)
        self.attack_unit_sel  = _mlp(node_dim, mlp_hidden, 1,           mlp_depth)
        self.attack_enemy_sel = _mlp(node_dim, mlp_hidden, 1,           mlp_depth)
        self.city_sel         = _mlp(node_dim, mlp_hidden, 1,           mlp_depth)
        self.unit_type_sel    = _mlp(node_dim, mlp_hidden, N_UNIT_TYPES, mlp_depth)
        self.capture_unit_sel = _mlp(node_dim, mlp_hidden, 1,           mlp_depth)

        # Fixed board topology — lives on whatever device the model is on
        self.register_buffer('edge_index', self._build_edge_index(Nx, Ny))

        # Pre-cache row/col index tensors for positional encoding
        # (avoids re-allocating every call)
        rows = torch.arange(Nx).repeat_interleave(Ny)   # (N,)
        cols = torch.arange(Ny).repeat(Nx)              # (N,)
        self.register_buffer('_pe_rows', rows)
        self.register_buffer('_pe_cols', cols)

    # ── Graph / encoding utilities ────────────────────────────────────────────

    def _build_edge_index(self, Nx: int, Ny: int) -> torch.Tensor:
        """Build (2, E) edge index for 8-connected (Moore) neighbourhood."""
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

    def _pos_enc(self) -> torch.Tensor:
        """(N, PE_DIM) positional encoding for a single graph."""
        return torch.cat([self.row_emb(self._pe_rows),
                          self.col_emb(self._pe_cols)], dim=-1)

    def _encode(self, partial_graph_np) -> torch.Tensor:
        """
        Single-graph encode used by CPU workers during rollout collection.
        partial_graph_np : np.ndarray (N, 26)
        Returns           : Tensor    (N, node_dim) on model device
        """
        dev = self.edge_index.device
        x   = torch.tensor(np.asarray(partial_graph_np),
                            dtype=torch.float32, device=dev)
        x   = self.mpnn(x, self.edge_index)
        pe  = self._pos_enc()
        return torch.cat([x, pe], dim=-1)

    def _encode_batch(self, graphs_np: np.ndarray) -> torch.Tensor:
        """
        Batched encode for the GPU PPO update.

        graphs_np : np.ndarray (B, N, 26)
        Returns   : Tensor    (B, N, node_dim) on model device

        Uses a single MPNN forward pass over the block-diagonal
        super-graph formed by tiling the shared edge_index.
        """
        dev   = self.edge_index.device
        B, N, _ = graphs_np.shape

        # (B*N, 26) — one contiguous transfer, no Python loop
        x_flat = torch.tensor(graphs_np.reshape(B * N, IN_FEATS),
                              dtype=torch.float32, device=dev)

        # Block-diagonal edge index (single call, no Data objects)
        ei_batch = _tile_edge_index(self.edge_index, B, N)  # (2, B*E)

        # Single MPNN forward pass: (B*N, 26) → (B*N, mpnn_hidden)
        h_flat = self.mpnn(x_flat, ei_batch)

        # Positional encoding: (N, PE_DIM) → repeat to (B*N, PE_DIM)
        pe_flat = self._pos_enc().repeat(B, 1)

        # Concatenate and reshape to (B, N, node_dim)
        return torch.cat([h_flat, pe_flat], dim=-1).reshape(B, N, self._node_dim)

    @property
    def _dev(self):
        return self.edge_index.device

    # ── Masking helper (single-sample, used in forward) ──────────────────────

    def _dist(self, logits: torch.Tensor,
              mask_np) -> torch.distributions.Categorical:
        dev  = logits.device
        mask = torch.tensor(np.asarray(mask_np), dtype=torch.float32,
                            device=dev).flatten()[:logits.shape[0]]
        return _masked_categorical(logits, mask)

    @staticmethod
    def _sizes(mask):
        return mask[1].shape[0], mask[2].shape[1], mask[3].shape[0]

    # ── forward : sample one action (CPU, single graph) ───────────────────────

    def forward(self, obs: dict, mask: list):
        """
        Called by CPU workers during rollout collection — one step at a time.
        Signature and logic unchanged from original.

        Returns (action, log_prob, entropy, value)
        """
        node_emb   = self._encode(obs['partial_graph'])
        global_emb = node_emb.mean(dim=0, keepdim=True)
        n_u, n_e, n_c = self._sizes(mask)

        value = self.critic_head(global_emb).squeeze()

        at_d  = self._dist(self.action_type_head(global_emb).squeeze(0), mask[0])
        at    = at_d.sample()
        log_p = at_d.log_prob(at)
        ent   = at_d.entropy()
        atype = ActionTypes(at.item())

        unit_ids  = [u.tile.id for u in obs['units']]
        enemy_ids = [u.tile.id for u in obs['enemy_units']]
        city_ids  = [c.tile_id for c in obs['cities']]

        if atype == ActionTypes.MoveUnit and n_u > 0:
            u_nodes = node_emb[unit_ids]
            ud      = self._dist(self.move_unit_sel(u_nodes).squeeze(-1),
                                 mask[1].max(axis=1))
            uid     = ud.sample()
            log_p  += ud.log_prob(uid);   ent += ud.entropy()

            td      = self._dist(self.move_target(node_emb).squeeze(-1),
                                 mask[1][uid.item()])
            tid     = td.sample()
            log_p  += td.log_prob(tid);   ent += td.entropy()
            action  = [atype, uid.item(), tid.item()]

        elif atype == ActionTypes.Attack and n_u > 0 and n_e > 0:
            u_nodes = node_emb[unit_ids]
            ud      = self._dist(self.attack_unit_sel(u_nodes).squeeze(-1),
                                 mask[2].max(axis=1))
            uid     = ud.sample()
            log_p  += ud.log_prob(uid);   ent += ud.entropy()

            e_nodes = node_emb[enemy_ids]
            ed      = self._dist(self.attack_enemy_sel(e_nodes).squeeze(-1),
                                 mask[2][uid.item()])
            eid     = ed.sample()
            log_p  += ed.log_prob(eid);   ent += ed.entropy()
            action  = [atype, uid.item(), eid.item()]

        elif atype == ActionTypes.CreateUnit and n_c > 0:
            c_nodes = node_emb[city_ids]
            cd      = self._dist(self.city_sel(c_nodes).squeeze(-1),
                                 mask[3].max(axis=1))
            cid     = cd.sample()
            log_p  += cd.log_prob(cid);   ent += cd.entropy()

            ut_logits = self.unit_type_sel(
                c_nodes[cid.item():cid.item()+1]).squeeze(0)
            utd       = self._dist(ut_logits, mask[3][cid.item()])
            utid      = utd.sample()
            log_p    += utd.log_prob(utid); ent += utd.entropy()
            action    = [atype, cid.item(), utid.item()]

        elif atype == ActionTypes.CaptureCity and n_u > 0:
            u_nodes = node_emb[unit_ids]
            ud      = self._dist(self.capture_unit_sel(u_nodes).squeeze(-1),
                                 mask[4])
            uid     = ud.sample()
            log_p  += ud.log_prob(uid);   ent += ud.entropy()
            action  = [atype, uid.item()]

        else:
            action = [ActionTypes.EndTurn]

        return action, log_p, ent, value

    # ── Batched subhead helpers ───────────────────────────────────────────────
    # Each helper receives:
    #   node_embs  : (B, N, node_dim)  — full minibatch node embeddings on GPU
    #   obs_snaps  : list[dict]        — full minibatch snapshots
    #   actions    : list[list]        — full minibatch actions
    #   masks      : list[list]        — full minibatch masks
    #   indices    : list[int]         — which samples in the minibatch this
    #                                    action type applies to
    # Returns (lp_delta, ent_delta) both shape (len(indices),) on GPU.

    def _eval_move(self, node_embs, obs_snaps, actions, masks,
                   indices: list) -> tuple[torch.Tensor, torch.Tensor]:
        """MoveUnit subheads — batched over the group."""
        dev  = node_embs.device
        B_g  = len(indices)
        N    = node_embs.shape[1]

        # ── Unit selection ────────────────────────────────────────────────
        unit_ids_list = [obs_snaps[i]['unit_ids'] for i in indices]
        max_u = max(len(u) for u in unit_ids_list)

        # Gather & pad unit embeddings: (B_g, max_u, node_dim)
        u_embs = torch.zeros(B_g, max_u, self._node_dim, device=dev)
        u_mask = torch.zeros(B_g, max_u, device=dev)
        for k, i in enumerate(indices):
            uids = unit_ids_list[k]
            n_u  = len(uids)
            u_embs[k, :n_u] = node_embs[i, uids]
            # valid-unit mask from rollout mask (max over target tiles)
            m = masks[i][1].max(axis=1)              # (n_u,)
            u_mask[k, :n_u] = torch.tensor(m, dtype=torch.float32, device=dev)

        # (B_g, max_u, 1) → (B_g, max_u)
        u_logits = self.move_unit_sel(u_embs).squeeze(-1)
        u_dist   = _masked_categorical(u_logits, u_mask)
        uid_t    = torch.tensor([actions[i][1] for i in indices],
                                dtype=torch.long, device=dev)
        lp_u     = u_dist.log_prob(uid_t)
        ent_u    = u_dist.entropy()

        # ── Target tile selection (all N tiles — no padding needed) ───────
        # node_embs[indices] : (B_g, N, node_dim)
        t_logits = self.move_target(node_embs[indices]).squeeze(-1)  # (B_g, N)
        t_mask   = torch.zeros(B_g, N, device=dev)
        for k, i in enumerate(indices):
            uid = actions[i][1]
            m   = masks[i][1][uid]                  # (N,)
            t_mask[k] = torch.tensor(m, dtype=torch.float32, device=dev)

        t_dist = _masked_categorical(t_logits, t_mask)
        tid_t  = torch.tensor([actions[i][2] for i in indices],
                              dtype=torch.long, device=dev)
        lp_t   = t_dist.log_prob(tid_t)
        ent_t  = t_dist.entropy()

        return lp_u + lp_t, ent_u + ent_t

    def _eval_attack(self, node_embs, obs_snaps, actions, masks,
                     indices: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Attack subheads — batched over the group."""
        dev  = node_embs.device
        B_g  = len(indices)

        # ── Attacker selection ────────────────────────────────────────────
        unit_ids_list = [obs_snaps[i]['unit_ids'] for i in indices]
        max_u = max(len(u) for u in unit_ids_list)

        u_embs = torch.zeros(B_g, max_u, self._node_dim, device=dev)
        u_mask = torch.zeros(B_g, max_u, device=dev)
        for k, i in enumerate(indices):
            uids = unit_ids_list[k]
            n_u  = len(uids)
            u_embs[k, :n_u] = node_embs[i, uids]
            m = masks[i][2].max(axis=1)              # (n_u,)
            u_mask[k, :n_u] = torch.tensor(m, dtype=torch.float32, device=dev)

        u_logits = self.attack_unit_sel(u_embs).squeeze(-1)
        u_dist   = _masked_categorical(u_logits, u_mask)
        uid_t    = torch.tensor([actions[i][1] for i in indices],
                                dtype=torch.long, device=dev)
        lp_u     = u_dist.log_prob(uid_t)
        ent_u    = u_dist.entropy()

        # ── Enemy selection ───────────────────────────────────────────────
        enemy_ids_list = [obs_snaps[i]['enemy_ids'] for i in indices]
        max_e = max(len(e) for e in enemy_ids_list)

        e_embs = torch.zeros(B_g, max_e, self._node_dim, device=dev)
        e_mask = torch.zeros(B_g, max_e, device=dev)
        for k, i in enumerate(indices):
            eids = enemy_ids_list[k]
            n_e  = len(eids)
            e_embs[k, :n_e] = node_embs[i, eids]
            uid = actions[i][1]
            m   = masks[i][2][uid]                   # (n_e,)
            e_mask[k, :n_e] = torch.tensor(m, dtype=torch.float32, device=dev)

        e_logits = self.attack_enemy_sel(e_embs).squeeze(-1)
        e_dist   = _masked_categorical(e_logits, e_mask)
        eid_t    = torch.tensor([actions[i][2] for i in indices],
                                dtype=torch.long, device=dev)
        lp_e     = e_dist.log_prob(eid_t)
        ent_e    = e_dist.entropy()

        return lp_u + lp_e, ent_u + ent_e

    def _eval_create(self, node_embs, obs_snaps, actions, masks,
                     indices: list) -> tuple[torch.Tensor, torch.Tensor]:
        """CreateUnit subheads — batched over the group."""
        dev  = node_embs.device
        B_g  = len(indices)

        # ── City selection ────────────────────────────────────────────────
        city_ids_list = [obs_snaps[i]['city_ids'] for i in indices]
        max_c = max(len(c) for c in city_ids_list)

        c_embs = torch.zeros(B_g, max_c, self._node_dim, device=dev)
        c_mask = torch.zeros(B_g, max_c, device=dev)
        for k, i in enumerate(indices):
            cids = city_ids_list[k]
            n_c  = len(cids)
            c_embs[k, :n_c] = node_embs[i, cids]
            m = masks[i][3].max(axis=1)              # (n_c,)
            c_mask[k, :n_c] = torch.tensor(m, dtype=torch.float32, device=dev)

        c_logits = self.city_sel(c_embs).squeeze(-1)
        c_dist   = _masked_categorical(c_logits, c_mask)
        cid_t    = torch.tensor([actions[i][1] for i in indices],
                                dtype=torch.long, device=dev)
        lp_c     = c_dist.log_prob(cid_t)
        ent_c    = c_dist.entropy()

        # ── Unit type selection ───────────────────────────────────────────
        # Gather the chosen city's embedding for each sample: (B_g, node_dim)
        chosen_city_embs = torch.stack([
            node_embs[i, obs_snaps[i]['city_ids'][actions[i][1]]]
            for i in indices
        ])                                                        # (B_g, node_dim)

        ut_logits = self.unit_type_sel(chosen_city_embs)         # (B_g, N_UNIT_TYPES)
        ut_mask   = torch.zeros(B_g, N_UNIT_TYPES, device=dev)
        for k, i in enumerate(indices):
            cid = actions[i][1]
            m   = masks[i][3][cid]                               # (N_UNIT_TYPES,)
            ut_mask[k] = torch.tensor(m, dtype=torch.float32, device=dev)

        ut_dist  = _masked_categorical(ut_logits, ut_mask)
        utid_t   = torch.tensor([actions[i][2] for i in indices],
                                dtype=torch.long, device=dev)
        lp_ut    = ut_dist.log_prob(utid_t)
        ent_ut   = ut_dist.entropy()

        return lp_c + lp_ut, ent_c + ent_ut

    def _eval_capture(self, node_embs, obs_snaps, actions, masks,
                      indices: list) -> tuple[torch.Tensor, torch.Tensor]:
        """CaptureCity subhead — batched over the group."""
        dev  = node_embs.device
        B_g  = len(indices)

        unit_ids_list = [obs_snaps[i]['unit_ids'] for i in indices]
        max_u = max(len(u) for u in unit_ids_list)

        u_embs = torch.zeros(B_g, max_u, self._node_dim, device=dev)
        u_mask = torch.zeros(B_g, max_u, device=dev)
        for k, i in enumerate(indices):
            uids = unit_ids_list[k]
            n_u  = len(uids)
            u_embs[k, :n_u] = node_embs[i, uids]
            m = masks[i][4]                                      # (n_u,)
            u_mask[k, :n_u] = torch.tensor(m, dtype=torch.float32, device=dev)

        u_logits = self.capture_unit_sel(u_embs).squeeze(-1)
        u_dist   = _masked_categorical(u_logits, u_mask)
        uid_t    = torch.tensor([actions[i][1] for i in indices],
                                dtype=torch.long, device=dev)
        lp_u     = u_dist.log_prob(uid_t)
        ent_u    = u_dist.entropy()

        return lp_u, ent_u

    # ── evaluate_actions : re-score stored transitions — GPU batched ──────────

    def evaluate_actions(self,
                         obs_snaps: list,
                         actions:   list,
                         masks:     list
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-score a minibatch of (obs, action, mask) triples under the current
        policy.  All heavy compute is fused into as few GPU kernel launches as
        possible.

        obs_snaps : list[dict]  keys: 'graph' (N,26), 'unit_ids', 'enemy_ids',
                                      'city_ids'
        actions   : list[list]
        masks     : list[list[np.ndarray]]

        Returns
          log_probs  : Tensor (B,)
          entropies  : Tensor (B,)
          values     : Tensor (B,)
        """
        dev = self._dev
        B   = len(obs_snaps)
        N   = self.Nx * self.Ny

        # ── 1. Single batched MPNN pass ───────────────────────────────────
        # Stack all node-feature matrices into one array before touching GPU
        graphs_np  = np.stack([s['graph'] for s in obs_snaps])  # (B, N, 26)
        node_embs  = self._encode_batch(graphs_np)              # (B, N, node_dim)

        # ── 2. Global embeddings → shared heads ───────────────────────────
        global_embs = node_embs.mean(dim=1)                     # (B, node_dim)
        values      = self.critic_head(global_embs).squeeze(-1) # (B,)

        at_logits_all = self.action_type_head(global_embs)      # (B, N_ACTION_TYPES)

        # ── 3. Action-type log_prob + entropy for whole batch ─────────────
        # Parse action types
        atypes = []
        for action in actions:
            at = action[0]
            if not isinstance(at, ActionTypes):
                at = ActionTypes(int(at))
            atypes.append(at)

        at_idx = torch.tensor([a.value for a in atypes],
                              dtype=torch.long, device=dev)     # (B,)

        # Stack action-type masks: (B, N_ACTION_TYPES)
        at_mask = torch.tensor(
            np.stack([m[0] for m in masks]),
            dtype=torch.float32, device=dev
        )
        at_dist    = _masked_categorical(at_logits_all, at_mask)
        log_probs  = at_dist.log_prob(at_idx)                   # (B,)
        entropies  = at_dist.entropy()                          # (B,)

        # ── 4. Subheads — one forward pass per action-type group ──────────
        # Group sample indices by action type, filtering invalid ones
        move_idx = [
            i for i, a in enumerate(atypes)
            if a == ActionTypes.MoveUnit
            and len(actions[i]) == 3
            and len(obs_snaps[i]['unit_ids']) > 0
        ]
        attack_idx = [
            i for i, a in enumerate(atypes)
            if a == ActionTypes.Attack
            and len(actions[i]) == 3
            and len(obs_snaps[i]['unit_ids']) > 0
            and len(obs_snaps[i]['enemy_ids']) > 0
        ]
        create_idx = [
            i for i, a in enumerate(atypes)
            if a == ActionTypes.CreateUnit
            and len(actions[i]) == 3
            and len(obs_snaps[i]['city_ids']) > 0
        ]
        capture_idx = [
            i for i, a in enumerate(atypes)
            if a == ActionTypes.CaptureCity
            and len(actions[i]) == 2
            and len(obs_snaps[i]['unit_ids']) > 0
        ]

        # Process each group — results are scatter-added into (B,) tensors
        if move_idx:
            lp_d, ent_d = self._eval_move(
                node_embs, obs_snaps, actions, masks, move_idx)
            log_probs[move_idx]    += lp_d
            entropies[move_idx]    += ent_d

        if attack_idx:
            lp_d, ent_d = self._eval_attack(
                node_embs, obs_snaps, actions, masks, attack_idx)
            log_probs[attack_idx]  += lp_d
            entropies[attack_idx]  += ent_d

        if create_idx:
            lp_d, ent_d = self._eval_create(
                node_embs, obs_snaps, actions, masks, create_idx)
            log_probs[create_idx]  += lp_d
            entropies[create_idx]  += ent_d

        if capture_idx:
            lp_d, ent_d = self._eval_capture(
                node_embs, obs_snaps, actions, masks, capture_idx)
            log_probs[capture_idx] += lp_d
            entropies[capture_idx] += ent_d

        return log_probs, entropies, values


# ── Utility ───────────────────────────────────────────────────────────────────

def model_summary(policy: PolicyNetwork):
    """Print a concise parameter breakdown."""
    total     = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    sections = {
        'MPNN backbone':     policy.mpnn,
        'Positional enc':    nn.ModuleList([policy.row_emb, policy.col_emb]),
        'Action type head':  policy.action_type_head,
        'Critic head':       policy.critic_head,
        'Move unit sel':     policy.move_unit_sel,
        'Move target':       policy.move_target,
        'Attack unit sel':   policy.attack_unit_sel,
        'Attack enemy sel':  policy.attack_enemy_sel,
        'City sel':          policy.city_sel,
        'Unit type sel':     policy.unit_type_sel,
        'Capture unit sel':  policy.capture_unit_sel,
    }

    print("=" * 56)
    print(f"  {'Module':<28} {'Params':>12}")
    print("-" * 56)
    for name, mod in sections.items():
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name:<28} {n:>12,}")
    print("=" * 56)
    print(f"  {'TOTAL':<28} {total:>12,}")
    print(f"  {'TRAINABLE':<28} {trainable:>12,}")
    print(f"  Node embedding dim : {policy._node_dim}")
    print("=" * 56)