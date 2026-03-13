"""
DummyPolicy — actor + critic with pairwise-attention subheads.
Embeddings are random (no real obs encoding) to validate gradient flow only.
Swap _rand() calls later with real obs-conditioned encoders.
"""

import torch
import torch.nn as nn
from game.enums import ActionTypes, UnitType

N_ACTION_TYPES = len(ActionTypes)
N_UNIT_TYPES   = len(UnitType)


class PairwiseHead(nn.Module):
    """Produces logits of shape (n_q, n_k) via scaled dot-product attention."""
    def __init__(self, D: int):
        super().__init__()
        self.q     = nn.Linear(D, D, bias=False)
        self.k     = nn.Linear(D, D, bias=False)
        self.scale = D ** 0.5

    def forward(self, q_emb, k_emb):
        # q_emb: (n_q, D)  k_emb: (n_k, D)  →  (n_q, n_k)
        return self.q(q_emb) @ self.k(k_emb).T / self.scale


class DummyPolicy(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        D       = cfg.embed_dim
        n_tiles = cfg.board_size[0] * cfg.board_size[1]

        # Shared body (dummy: single learned context vector)
        self.global_ctx = nn.Parameter(torch.randn(1, D))

        # Fixed-vocab embeddings
        self.tile_emb      = nn.Embedding(n_tiles, D)
        self.unit_type_emb = nn.Embedding(N_UNIT_TYPES, D)

        # Actor heads
        self.action_type_head = nn.Linear(D, N_ACTION_TYPES)
        self.unit_sel         = PairwiseHead(D)
        self.tile_sel         = PairwiseHead(D)
        self.enemy_sel        = PairwiseHead(D)
        self.city_sel         = PairwiseHead(D)
        self.utype_sel        = PairwiseHead(D)

        # Critic head
        self.critic_head = nn.Linear(D, 1)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _D(self):
        return self.global_ctx.shape[1]

    def _rand(self, n: int) -> torch.Tensor:
        """Random entity embeddings — replace with real encoder later."""
        return torch.randn(n, self._D(), device=self.global_ctx.device)

    def _dist(self, logits: torch.Tensor, mask_np) -> torch.distributions.Categorical:
        mask = torch.tensor(mask_np, dtype=torch.float32, device=logits.device)
        return torch.distributions.Categorical(
            logits=logits + torch.log(mask.clamp(min=1e-8))
        )

    def _value(self) -> torch.Tensor:
        return self.critic_head(self.global_ctx).squeeze()   # scalar

    # ── forward: sample an action ─────────────────────────────────────────────

    def forward(self, obs: dict, mask: list):
        """
        Samples a hierarchical action.
        Returns (action: list, log_prob: Tensor, entropy: Tensor, value: Tensor)
        """
        ctx = self.global_ctx
        n_u = len(obs["units"])
        n_e = len(obs["enemy_units"])
        n_c = len(obs["cities"])

        value = self._value()

        at_d  = self._dist(self.action_type_head(ctx).squeeze(0), mask[0])
        at    = at_d.sample()
        log_p = at_d.log_prob(at)
        ent   = at_d.entropy()
        atype = ActionTypes(at.item())

        if atype == ActionTypes.MoveUnit and n_u > 0:
            u_emb = self._rand(n_u)
            ud    = self._dist(self.unit_sel(ctx, u_emb).squeeze(0), mask[1].max(axis=1))
            uid   = ud.sample();    log_p += ud.log_prob(uid);    ent += ud.entropy()
            td    = self._dist(self.tile_sel(u_emb[uid:uid+1], self.tile_emb.weight).squeeze(0), mask[1][uid.item()])
            tid   = td.sample();    log_p += td.log_prob(tid);    ent += td.entropy()
            action = [atype, uid.item(), tid.item()]

        elif atype == ActionTypes.Attack and n_u > 0 and n_e > 0:
            u_emb = self._rand(n_u);   e_emb = self._rand(n_e)
            ud    = self._dist(self.unit_sel(ctx, u_emb).squeeze(0), mask[2].max(axis=1))
            uid   = ud.sample();    log_p += ud.log_prob(uid);    ent += ud.entropy()
            ed    = self._dist(self.enemy_sel(u_emb[uid:uid+1], e_emb).squeeze(0), mask[2][uid.item()])
            eid   = ed.sample();    log_p += ed.log_prob(eid);    ent += ed.entropy()
            action = [atype, uid.item(), eid.item()]

        elif atype == ActionTypes.CreateUnit and n_c > 0:
            c_emb = self._rand(n_c)
            cd    = self._dist(self.city_sel(ctx, c_emb).squeeze(0), mask[3].max(axis=1))
            cid   = cd.sample();    log_p += cd.log_prob(cid);    ent += cd.entropy()
            utd   = self._dist(self.utype_sel(c_emb[cid:cid+1], self.unit_type_emb.weight).squeeze(0), mask[3][cid.item()])
            utid  = utd.sample();   log_p += utd.log_prob(utid);  ent += utd.entropy()
            action = [atype, cid.item(), utid.item()]

        elif atype == ActionTypes.CaptureCity and n_u > 0:
            u_emb = self._rand(n_u)
            ud    = self._dist(self.unit_sel(ctx, u_emb).squeeze(0), mask[4])
            uid   = ud.sample();    log_p += ud.log_prob(uid);    ent += ud.entropy()
            action = [atype, uid.item()]

        else:
            action = [ActionTypes.EndTurn]

        return action, log_p, ent, value

    # ── evaluate_actions: re-score stored actions under current weights ───────

    def evaluate_actions(self, obs_sizes: list, actions: list, masks: list):
        """
        Re-evaluates a mini-batch of stored transitions for the PPO loss.

        obs_sizes : list[(n_units, n_enemy, n_cities)]   length = T
        actions   : list[action_list]                    length = T
        masks     : list[mask_list]                      length = T

        Returns
        -------
        log_probs : Tensor (T,)
        entropies : Tensor (T,)
        values    : Tensor (T,)

        NOTE: random embeddings (_rand) mean log_probs here differ from those
        collected during the rollout — intentional for pipeline testing.
        Replace _rand() with a real encoder before meaningful training.
        """
        ctx = self.global_ctx
        log_probs, entropies, values = [], [], []

        for (n_u, n_e, n_c), action, mask in zip(obs_sizes, actions, masks):
            values.append(self._value())

            atype = action[0]
            atype = ActionTypes(int(atype)) if not isinstance(atype, ActionTypes) else atype
            at_t  = torch.tensor(atype.value, dtype=torch.long)

            at_d  = self._dist(self.action_type_head(ctx).squeeze(0), mask[0])
            log_p = at_d.log_prob(at_t)
            ent   = at_d.entropy()

            if atype == ActionTypes.MoveUnit and len(action) == 3 and n_u > 0:
                uid, tid = action[1], action[2]
                u_emb = self._rand(n_u)
                ud    = self._dist(self.unit_sel(ctx, u_emb).squeeze(0), mask[1].max(axis=1))
                log_p += ud.log_prob(torch.tensor(uid));    ent += ud.entropy()
                td    = self._dist(self.tile_sel(u_emb[uid:uid+1], self.tile_emb.weight).squeeze(0), mask[1][uid])
                log_p += td.log_prob(torch.tensor(tid));    ent += td.entropy()

            elif atype == ActionTypes.Attack and len(action) == 3 and n_u > 0 and n_e > 0:
                uid, eid  = action[1], action[2]
                u_emb = self._rand(n_u);   e_emb = self._rand(n_e)
                ud    = self._dist(self.unit_sel(ctx, u_emb).squeeze(0), mask[2].max(axis=1))
                log_p += ud.log_prob(torch.tensor(uid));    ent += ud.entropy()
                ed    = self._dist(self.enemy_sel(u_emb[uid:uid+1], e_emb).squeeze(0), mask[2][uid])
                log_p += ed.log_prob(torch.tensor(eid));    ent += ed.entropy()

            elif atype == ActionTypes.CreateUnit and len(action) == 3 and n_c > 0:
                cid, utid = action[1], action[2]
                c_emb = self._rand(n_c)
                cd    = self._dist(self.city_sel(ctx, c_emb).squeeze(0), mask[3].max(axis=1))
                log_p += cd.log_prob(torch.tensor(cid));    ent += cd.entropy()
                utd   = self._dist(self.utype_sel(c_emb[cid:cid+1], self.unit_type_emb.weight).squeeze(0), mask[3][cid])
                log_p += utd.log_prob(torch.tensor(utid));  ent += utd.entropy()

            elif atype == ActionTypes.CaptureCity and len(action) == 2 and n_u > 0:
                uid   = action[1]
                u_emb = self._rand(n_u)
                ud    = self._dist(self.unit_sel(ctx, u_emb).squeeze(0), mask[4])
                log_p += ud.log_prob(torch.tensor(uid));    ent += ud.entropy()

            log_probs.append(log_p)
            entropies.append(ent)

        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)
