import torch
import torch.nn as nn
import numpy as np

from torch_geometric.nn import SAGEConv
from game.enums import ActionTypes, UnitType

# ── Dimensions ────────────────────────────────────────────────────────────────
MPNN_DIM  = 128    # node embedding after message passing
PE_DIM    = 16    # 2-D positional encoding  (8 per spatial axis)
NODE_DIM  = MPNN_DIM + PE_DIM   # 80 — input to every subhead
MLP_HID   = 128
IN_FEATS  = 26    # raw node feature dimension from partial_graph

N_ACTION_TYPES = len(ActionTypes)
N_UNIT_TYPES   = len(UnitType)


# ── Building blocks ───────────────────────────────────────────────────────────

class _MPNN(nn.Module):
    """2-layer GraphSAGE: in_dim → hidden → hidden."""
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.c1  = SAGEConv(in_dim, hidden)
        self.c2  = SAGEConv(hidden, hidden)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.act(self.c1(x, edge_index))
        x = self.act(self.c2(x, edge_index))
        return x   # (N, hidden)


def _mlp(in_d: int, hid_d: int, out_d: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_d, hid_d), nn.ReLU(),
        nn.Linear(hid_d, out_d),
    )


# ── Main policy ───────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        Nx, Ny  = cfg.board_size
        self.Nx = Nx
        self.Ny = Ny

        # ── Backbone ──────────────────────────────────────────────────────────
        self.mpnn = _MPNN(IN_FEATS, MPNN_DIM)

        # ── 2-D positional encoding (learned, one embedding per row/col) ──────
        self.row_emb = nn.Embedding(Nx, PE_DIM // 2)
        self.col_emb = nn.Embedding(Ny, PE_DIM // 2)

        # ── Global heads (applied to mean-pooled node embedding) ──────────────
        self.action_type_head = _mlp(NODE_DIM, MLP_HID, N_ACTION_TYPES)
        self.critic_head      = _mlp(NODE_DIM, MLP_HID, 1)

        # ── Subheads (node-level, scalar output unless noted) ─────────────────
        self.move_unit_sel    = _mlp(NODE_DIM, MLP_HID, 1)
        self.move_target      = _mlp(NODE_DIM, MLP_HID, 1)
        self.attack_unit_sel  = _mlp(NODE_DIM, MLP_HID, 1)
        self.attack_enemy_sel = _mlp(NODE_DIM, MLP_HID, 1)
        self.city_sel         = _mlp(NODE_DIM, MLP_HID, 1)
        self.unit_type_sel    = _mlp(NODE_DIM, MLP_HID, N_UNIT_TYPES)
        self.capture_unit_sel = _mlp(NODE_DIM, MLP_HID, 1)

        # Fixed graph topology — registered as buffer so it moves with .to(device)
        self.register_buffer('edge_index', self._build_edge_index(Nx, Ny))

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

    def _pos_enc(self) -> torch.Tensor:
        """Returns (Nx*Ny, PE_DIM) positional encoding for all nodes."""
        dev  = self.edge_index.device
        rows = torch.arange(self.Nx, device=dev).repeat_interleave(self.Ny)
        cols = torch.arange(self.Ny, device=dev).repeat(self.Nx)
        return torch.cat([self.row_emb(rows), self.col_emb(cols)], dim=-1)

    def _encode(self, partial_graph_np) -> torch.Tensor:
        """
        partial_graph_np : np.ndarray (N, 26)
        Returns node_emb  : Tensor    (N, NODE_DIM) on model device
        """
        dev = self.edge_index.device
        x   = torch.tensor(np.asarray(partial_graph_np),
                            dtype=torch.float32, device=dev)
        x   = self.mpnn(x, self.edge_index)   # (N, MPNN_DIM)
        pe  = self._pos_enc()                  # (N, PE_DIM)
        return torch.cat([x, pe], dim=-1)      # (N, NODE_DIM)

    @property
    def _dev(self):
        return self.edge_index.device

    # ── Masking helper ────────────────────────────────────────────────────────

    def _dist(self, logits: torch.Tensor,
              mask_np) -> torch.distributions.Categorical:
        dev  = logits.device
        mask = torch.tensor(np.asarray(mask_np), dtype=torch.float32,
                            device=dev).flatten()[:logits.shape[0]]
        return torch.distributions.Categorical(
            logits=logits + torch.log(mask.clamp(min=1e-8))
        )

    @staticmethod
    def _sizes(mask):
        """Entity counts directly from mask shapes — never from live obs."""
        return mask[1].shape[0], mask[2].shape[1], mask[3].shape[0]

    # ── forward : sample one action ───────────────────────────────────────────

    def forward(self, obs: dict, mask: list):
        """
        obs keys:
          partial_graph  np.ndarray (N, 26)
          units          list[unit]  →  u.tile.id
          enemy_units    list[unit]  →  u.tile.id
          cities         list[city]  →  c.tile_id

        Returns (action, log_prob, entropy, value)
        """
        node_emb   = self._encode(obs['partial_graph'])         # (N, NODE_DIM)
        global_emb = node_emb.mean(dim=0, keepdim=True)         # (1, NODE_DIM)
        dev        = node_emb.device
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
            u_nodes  = node_emb[unit_ids]                              # (n_u, NODE_DIM)
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

            e_nodes  = node_emb[enemy_ids]                             # (n_e, NODE_DIM)
            ed       = self._dist(self.attack_enemy_sel(e_nodes).squeeze(-1),
                                  mask[2][uid.item()])
            eid      = ed.sample()
            log_p   += ed.log_prob(eid);    ent += ed.entropy()
            action   = [atype, uid.item(), eid.item()]

        elif atype == ActionTypes.CreateUnit and n_c > 0:
            c_nodes  = node_emb[city_ids]                              # (n_c, NODE_DIM)
            cd       = self._dist(self.city_sel(c_nodes).squeeze(-1),
                                  mask[3].max(axis=1))
            cid      = cd.sample()
            log_p   += cd.log_prob(cid);    ent += cd.entropy()

            ut_logits = self.unit_type_sel(
                c_nodes[cid.item():cid.item()+1]).squeeze(0)           # (N_UNIT_TYPES,)
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
                      'unit_ids'  : list[int]
                      'enemy_ids' : list[int]
                      'city_ids'  : list[int]

        Returns log_probs (T,), entropies (T,), values (T,)
        """
        dev = self._dev
        log_probs, entropies, values = [], [], []

        for snap, action, mask in zip(obs_snaps, actions, masks):
            node_emb   = self._encode(snap['graph'])
            global_emb = node_emb.mean(dim=0, keepdim=True)
            values.append(self.critic_head(global_emb).squeeze())

            unit_ids  = snap['unit_ids']
            enemy_ids = snap['enemy_ids']
            city_ids  = snap['city_ids']
            n_u, n_e, n_c = self._sizes(mask)

            atype = action[0]
            if not isinstance(atype, ActionTypes):
                atype = ActionTypes(int(atype))
            at_t  = torch.tensor(atype.value, dtype=torch.long, device=dev)

            at_d  = self._dist(self.action_type_head(global_emb).squeeze(0), mask[0])
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

        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)
