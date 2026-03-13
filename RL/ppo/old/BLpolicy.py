"""
BaselineSAGEConv_policy_network.py
────────────────────────────────────────────────────────────────────────────────
GraphSAGE-backed actor-critic for the Polytopia RL project.

Changes vs. previous version
──────────────────────────────
• Board size is no longer fixed.  PolicyNetwork no longer stores Nx/Ny or a
  fixed edge_index buffer.  Instead, edge indices are built on demand and
  cached per (Nx, Ny) in a plain dict (_edge_cache).
• Learned row/col embeddings replaced with a parameter-free sinusoidal 2-D
  positional encoding that generalises to any square board size.
• _encode() now takes explicit (Nx, Ny) arguments.
• _encode_batch() runs the GNN for an entire minibatch in one forward pass
  using PyG's Batch utility, then splits results back per graph.
• evaluate_actions() uses _encode_batch() so all GNN work is parallelised on
  GPU; per-sample sub-decision scoring still loops (masks are variable-shape).
"""

import math
import torch
import torch.nn as nn
import numpy as np

from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch
from game.enums import ActionTypes, UnitType

# ── Dimensions (defaults — overridden by cfg where provided) ──────────────────
_MPNN_DIM_DEFAULT = 128
PE_DIM            = 16     # 8 per spatial axis (sinusoidal, no parameters)
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
    Build a fully-connected MLP.

    depth = number of hidden layers (each of size hid_d).
    Minimum depth = 1 → [Linear(in→hid), ReLU, Linear(hid→out)]
    depth = 2       → [Linear(in→hid), ReLU, Linear(hid→hid), ReLU, Linear(hid→out)]
    etc.
    """
    assert depth >= 1, "depth must be >= 1"
    layers: list[nn.Module] = [nn.Linear(in_d, hid_d), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hid_d, hid_d), nn.ReLU()]
    layers.append(nn.Linear(hid_d, out_d))
    return nn.Sequential(*layers)


# ── Main policy ───────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # ── Pull depth / width from cfg (with safe defaults) ──────────────
        mpnn_hidden  = getattr(cfg, 'mpnn_hidden_dim', _MPNN_DIM_DEFAULT)
        mlp_hidden   = getattr(cfg, 'mlp_hidden_dim',  128)
        mlp_depth    = getattr(cfg, 'mlp_depth',        2)
        node_dim     = mpnn_hidden + PE_DIM   # concatenated node embedding size

        self._node_dim = node_dim   # stored for external inspection

        # Edge index cache: (Nx, Ny) → CPU LongTensor.
        # Built lazily on first use for each board size encountered.
        self._edge_cache: dict = {}

        # Dummy buffer — only purpose is to give us a reliable .device reference
        # that moves correctly when policy.to(device) is called.
        self.register_buffer('_device_ref', torch.zeros(1))

        # ── Backbone ──────────────────────────────────────────────────────
        self.mpnn = _MPNN(IN_FEATS, mpnn_hidden)
        # Positional encoding is sinusoidal — no learned parameters needed.

        # ── Global heads ──────────────────────────────────────────────────
        self.action_type_head = _mlp(node_dim, mlp_hidden, N_ACTION_TYPES,  mlp_depth)
        self.critic_head      = _mlp(node_dim, mlp_hidden, 1,               mlp_depth)

        # ── Subheads (node-level) ─────────────────────────────────────────
        self.move_unit_sel    = _mlp(node_dim, mlp_hidden, 1,               mlp_depth)
        self.move_target      = _mlp(node_dim, mlp_hidden, 1,               mlp_depth)
        self.attack_unit_sel  = _mlp(node_dim, mlp_hidden, 1,               mlp_depth)
        self.attack_enemy_sel = _mlp(node_dim, mlp_hidden, 1,               mlp_depth)
        self.city_sel         = _mlp(node_dim, mlp_hidden, 1,               mlp_depth)
        self.unit_type_sel    = _mlp(node_dim, mlp_hidden, N_UNIT_TYPES,    mlp_depth)
        self.capture_unit_sel = _mlp(node_dim, mlp_hidden, 1,               mlp_depth)

    # ── Device helper ─────────────────────────────────────────────────────────

    @property
    def _dev(self):
        return self._device_ref.device

    # ── Graph / encoding utilities ────────────────────────────────────────────

    def _build_edge_index(self, Nx: int, Ny: int) -> torch.Tensor:
        """Build (2, E) edge index for the 8-connected (Moore) grid."""
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

    def _get_edge_index(self, Nx: int, Ny: int) -> torch.Tensor:
        """
        Return the (2, E) edge index for (Nx, Ny), building and caching it
        the first time it is requested.  Always returns a CPU tensor so it
        can be safely passed to PyG's Batch.from_data_list().
        """
        key = (Nx, Ny)
        if key not in self._edge_cache:
            self._edge_cache[key] = self._build_edge_index(Nx, Ny)  # CPU
        return self._edge_cache[key]

    def _pos_enc(self, Nx: int, Ny: int, device) -> torch.Tensor:
        """
        Parameter-free sinusoidal 2-D positional encoding.
        Works for any board size — no re-initialisation needed.

        Returns (Nx*Ny, PE_DIM) where PE_DIM = 8 (row dims) + 8 (col dims).
        """
        N = Nx * Ny
        d = PE_DIM // 2   # 8 dims per spatial axis

        rows = torch.arange(Nx, device=device).repeat_interleave(Ny).float()  # (N,)
        cols = torch.arange(Ny, device=device).repeat(Nx).float()              # (N,)

        # Frequencies: 4 sin/cos pairs per axis
        div_term = torch.exp(
            torch.arange(0, d, 2, device=device).float()
            * -(math.log(10000.0) / d)
        )   # (d//2,)

        pe_row = torch.zeros(N, d, device=device)
        pe_row[:, 0::2] = torch.sin(rows.unsqueeze(1) * div_term)
        pe_row[:, 1::2] = torch.cos(rows.unsqueeze(1) * div_term)

        pe_col = torch.zeros(N, d, device=device)
        pe_col[:, 0::2] = torch.sin(cols.unsqueeze(1) * div_term)
        pe_col[:, 1::2] = torch.cos(cols.unsqueeze(1) * div_term)

        return torch.cat([pe_row, pe_col], dim=-1)   # (N, PE_DIM)

    def _encode(self, partial_graph_np, Nx: int, Ny: int) -> torch.Tensor:
        """
        Encode a single board.
        partial_graph_np : np.ndarray (N_tiles, 26)
        Returns node_emb  : Tensor    (N_tiles, node_dim) on model device.
        """
        dev        = self._dev
        x          = torch.tensor(np.asarray(partial_graph_np),
                                  dtype=torch.float32, device=dev)
        edge_index = self._get_edge_index(Nx, Ny).to(dev)
        x          = self.mpnn(x, edge_index)       # (N_tiles, mpnn_hidden)
        pe         = self._pos_enc(Nx, Ny, dev)     # (N_tiles, PE_DIM)
        return torch.cat([x, pe], dim=-1)           # (N_tiles, node_dim)

    def _encode_batch(self, graphs: list, board_sizes: list):
        """
        Encode a list of boards in ONE batched GNN forward pass.

        graphs      : list of np.ndarray, each (N_i, 26)   (variable N_i)
        board_sizes : list of (Nx_i, Ny_i)

        Returns
        -------
        node_embs   : list of Tensors, each (N_i, node_dim)  on device
        global_embs : Tensor (B, node_dim)  — mean-pooled per graph, on device
        """
        dev = self._dev

        # Build PyG Data list on CPU; Batch.from_data_list handles offset logic.
        data_list = [
            Data(
                x          = torch.tensor(np.asarray(g), dtype=torch.float32),
                edge_index = self._get_edge_index(Nx, Ny),   # CPU
            )
            for g, (Nx, Ny) in zip(graphs, board_sizes)
        ]
        big_batch = Batch.from_data_list(data_list).to(dev)
        # big_batch.x          : (sum_N, 26)
        # big_batch.edge_index : (2, sum_E)  — correctly offset by PyG
        # big_batch.batch      : (sum_N,)    — graph index per node

        # ── Single GNN forward pass for all graphs ────────────────────────
        x_enc = self.mpnn(big_batch.x, big_batch.edge_index)   # (sum_N, mpnn_hidden)

        # ── Positional encodings (built per board size, concatenated) ─────
        pe_all = torch.cat(
            [self._pos_enc(Nx, Ny, dev) for Nx, Ny in board_sizes], dim=0
        )   # (sum_N, PE_DIM)

        node_emb_all = torch.cat([x_enc, pe_all], dim=-1)   # (sum_N, node_dim)

        # ── Global (graph-level) embeddings via mean pooling ──────────────
        global_embs = global_mean_pool(node_emb_all, big_batch.batch)   # (B, node_dim)

        # ── Split back to per-graph tensors ───────────────────────────────
        sizes     = [g.shape[0] for g in graphs]
        node_embs = list(torch.split(node_emb_all, sizes, dim=0))

        return node_embs, global_embs

    # ── Masking helper ────────────────────────────────────────────────────────

    def _dist(self, logits: torch.Tensor,
              mask_np) -> torch.distributions.Categorical:
        dev  = logits.device
        mask = torch.tensor(np.asarray(mask_np), dtype=torch.float32,
                            device=dev).flatten()[:logits.shape[0]]
        return torch.distributions.Categorical(
            logits=logits + torch.log(mask)#.clamp(min=1e-12))
        )

    @staticmethod
    def _sizes(mask):
        """Entity counts directly from mask shapes — never from live obs."""
        return mask[1].shape[0], mask[2].shape[1], mask[3].shape[0]

    # ── forward : sample one action ───────────────────────────────────────────

    def forward(self, obs: dict, mask: list):
        """
        obs keys:
          partial_graph  np.ndarray (N, 26)   — N = Nx*Ny (inferred)
          units          list[unit]  →  u.tile.id
          enemy_units    list[unit]  →  u.tile.id
          cities         list[city]  →  c.tile_id

        Returns (action, log_prob, entropy, value)
        """
        # Infer board dimensions from graph shape (always square).
        N_tiles    = np.asarray(obs['partial_graph']).shape[0]
        Nx = Ny    = int(round(N_tiles ** 0.5))

        node_emb   = self._encode(obs['partial_graph'], Nx, Ny)  # (N, node_dim)
        global_emb = node_emb.mean(dim=0, keepdim=True)          # (1, node_dim)
        n_u, n_e, n_c = self._sizes(mask)

        value = self.critic_head(global_emb).squeeze()

        at_d  = self._dist(self.action_type_head(global_emb).squeeze(0), mask[0])
        at    = at_d.sample()
        log_p = at_d.log_prob(at)
        ent   = at_d.entropy()
        atype = ActionTypes(at.item())

        # Tile IDs — read from live obs here (before env.step mutates the lists)
        unit_ids  = [u.tile.id for u in obs['units']]
        enemy_ids = [u.tile.id for u in obs['enemy_units']]
        city_ids  = [c.tile_id for c in obs['cities']]

        if atype == ActionTypes.MoveUnit and n_u > 0:
            u_nodes  = node_emb[unit_ids]
            ud       = self._dist(self.move_unit_sel(u_nodes).squeeze(-1),
                                  mask[1].max(axis=1))
            uid      = ud.sample()
            log_p   += ud.log_prob(uid);    ent += ud.entropy()

            td       = self._dist(self.move_target(node_emb).squeeze(-1),
                                  mask[1][uid.item()])
            tid      = td.sample()
            log_p   += td.log_prob(tid);    ent += td.entropy()
            action   = [atype, uid.item(), tid.item()]

        elif atype == ActionTypes.Attack and n_u > 0 and n_e > 0:
            u_nodes  = node_emb[unit_ids]
            ud       = self._dist(self.attack_unit_sel(u_nodes).squeeze(-1),
                                  mask[2].max(axis=1))
            uid      = ud.sample()
            log_p   += ud.log_prob(uid);    ent += ud.entropy()

            e_nodes  = node_emb[enemy_ids]
            ed       = self._dist(self.attack_enemy_sel(e_nodes).squeeze(-1),
                                  mask[2][uid.item()])
            eid      = ed.sample()
            log_p   += ed.log_prob(eid);    ent += ed.entropy()
            action   = [atype, uid.item(), eid.item()]

        elif atype == ActionTypes.CreateUnit and n_c > 0:
            c_nodes  = node_emb[city_ids]
            cd       = self._dist(self.city_sel(c_nodes).squeeze(-1),
                                  mask[3].max(axis=1))
            cid      = cd.sample()
            log_p   += cd.log_prob(cid);    ent += cd.entropy()

            ut_logits = self.unit_type_sel(
                c_nodes[cid.item():cid.item()+1]).squeeze(0)
            utd       = self._dist(ut_logits, mask[3][cid.item()])
            utid      = utd.sample()
            log_p    += utd.log_prob(utid); ent += utd.entropy()
            action    = [atype, cid.item(), utid.item()]

        elif atype == ActionTypes.CaptureCity and n_u > 0:
            u_nodes  = node_emb[unit_ids]
            ud       = self._dist(self.capture_unit_sel(u_nodes).squeeze(-1),
                                  mask[4])
            uid      = ud.sample()
            log_p   += ud.log_prob(uid);    ent += ud.entropy()
            action   = [atype, uid.item()]

        else:
            action = [ActionTypes.EndTurn]

        return action, log_p, ent, value

    # ── evaluate_actions : re-score stored transitions ────────────────────────

    def evaluate_actions(self, obs_snaps: list, actions: list, masks: list):
        """
        obs_snaps : list of dicts with keys
                      'graph'     : np.ndarray (N, 26)
                      'Nx', 'Ny'  : int  — board dimensions
                      'unit_ids'  : list[int]
                      'enemy_ids' : list[int]
                      'city_ids'  : list[int]

        Returns log_probs (T,), entropies (T,), values (T,)
        """
        dev = self._dev

        # ── 1. Batch GNN — single forward pass for the whole minibatch ────
        graphs      = [snap['graph'] for snap in obs_snaps]
        board_sizes = [(snap['Nx'], snap['Ny']) for snap in obs_snaps]
        node_embs, global_embs = self._encode_batch(graphs, board_sizes)
        # node_embs   : list of (N_i, node_dim) on device
        # global_embs : (B, node_dim)            on device

        # ── 2. Batch critic and action-type logits ─────────────────────────
        values        = self.critic_head(global_embs).squeeze(-1)   # (B,)
        at_logits_all = self.action_type_head(global_embs)           # (B, N_ACTION_TYPES)

        # ── 3. Per-sample sub-decision scoring (masks are variable-shape) ──
        log_probs, entropies = [], []

        for i, (snap, action, mask) in enumerate(zip(obs_snaps, actions, masks)):
            node_emb  = node_embs[i]

            unit_ids  = snap['unit_ids']
            enemy_ids = snap['enemy_ids']
            city_ids  = snap['city_ids']
            n_u, n_e, n_c = self._sizes(mask)

            atype = action[0]
            if not isinstance(atype, ActionTypes):
                atype = ActionTypes(int(atype))
            at_t  = torch.tensor(atype.value, dtype=torch.long, device=dev)

            # Use pre-computed logit row — avoids re-running the MLP head
            at_d  = self._dist(at_logits_all[i], mask[0])
            log_p = at_d.log_prob(at_t)
            ent   = at_d.entropy()

            if atype == ActionTypes.MoveUnit and len(action) == 3 and n_u > 0:
                uid, tid = action[1], action[2]
                u_nodes  = node_emb[unit_ids]
                ud       = self._dist(self.move_unit_sel(u_nodes).squeeze(-1),
                                      mask[1].max(axis=1))
                log_p   += ud.log_prob(torch.tensor(uid, device=dev)); ent += ud.entropy()

                td       = self._dist(self.move_target(node_emb).squeeze(-1),
                                      mask[1][uid])
                log_p   += td.log_prob(torch.tensor(tid, device=dev)); ent += td.entropy()

            elif atype == ActionTypes.Attack and len(action) == 3 and n_u > 0 and n_e > 0:
                uid, eid = action[1], action[2]
                u_nodes  = node_emb[unit_ids]
                ud       = self._dist(self.attack_unit_sel(u_nodes).squeeze(-1),
                                      mask[2].max(axis=1))
                log_p   += ud.log_prob(torch.tensor(uid, device=dev)); ent += ud.entropy()

                e_nodes  = node_emb[enemy_ids]
                ed       = self._dist(self.attack_enemy_sel(e_nodes).squeeze(-1),
                                      mask[2][uid])
                log_p   += ed.log_prob(torch.tensor(eid, device=dev)); ent += ed.entropy()

            elif atype == ActionTypes.CreateUnit and len(action) == 3 and n_c > 0:
                cid, utid = action[1], action[2]
                c_nodes   = node_emb[city_ids]
                cd        = self._dist(self.city_sel(c_nodes).squeeze(-1),
                                       mask[3].max(axis=1))
                log_p    += cd.log_prob(torch.tensor(cid,  device=dev)); ent += cd.entropy()

                ut_logits = self.unit_type_sel(c_nodes[cid:cid+1]).squeeze(0)
                utd       = self._dist(ut_logits, mask[3][cid])
                log_p    += utd.log_prob(torch.tensor(utid, device=dev)); ent += utd.entropy()

            elif atype == ActionTypes.CaptureCity and len(action) == 2 and n_u > 0:
                uid      = action[1]
                u_nodes  = node_emb[unit_ids]
                ud       = self._dist(self.capture_unit_sel(u_nodes).squeeze(-1),
                                      mask[4])
                log_p   += ud.log_prob(torch.tensor(uid, device=dev)); ent += ud.entropy()

            log_probs.append(log_p)
            entropies.append(ent)

        return torch.stack(log_probs), torch.stack(entropies), values


# ── Utility ───────────────────────────────────────────────────────────────────

def model_summary(policy: PolicyNetwork):
    """Print a concise parameter summary."""
    total     = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    # Per-module breakdown (positional encoding has no parameters — sinusoidal)
    sections = {
        'MPNN backbone':     policy.mpnn,
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
    print(f"  Positional enc     : sinusoidal (0 params, any board size)")
    print("=" * 56)
