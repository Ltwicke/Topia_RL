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
        return self.q(q_emb) @ self.k(k_emb).T / self.scale


class DummyPolicy(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        D       = cfg.embed_dim
        n_tiles = cfg.board_size[0] * cfg.board_size[1]

        self.global_ctx    = nn.Parameter(torch.randn(1, D))
        self.tile_emb      = nn.Embedding(n_tiles, D)
        self.unit_type_emb = nn.Embedding(N_UNIT_TYPES, D)

        self.action_type_head = nn.Linear(D, N_ACTION_TYPES)
        self.unit_sel         = PairwiseHead(D)
        self.tile_sel         = PairwiseHead(D)
        self.enemy_sel        = PairwiseHead(D)
        self.city_sel         = PairwiseHead(D)
        self.utype_sel        = PairwiseHead(D)

        self.critic_head = nn.Linear(D, 1)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _D(self):
        return self.global_ctx.shape[1]

    def _rand(self, n: int) -> torch.Tensor:
        return torch.randn(n, self._D(), device=self.global_ctx.device)

    def _dist(self, logits: torch.Tensor, mask_np) -> torch.distributions.Categorical:
        """
        Build a masked Categorical. Mask is sliced to logit length to handle
        any edge-case padding in the mask arrays.
        """
        mask = torch.tensor(mask_np, dtype=torch.float32,
                            device=logits.device).flatten()[:logits.shape[0]]
        return torch.distributions.Categorical(
            logits=logits + torch.log(mask.clamp(min=1e-8))
        )

    def _value(self) -> torch.Tensor:
        return self.critic_head(self.global_ctx).squeeze()

    def _sizes_from_mask(self, mask: list):
        """
        Read entity counts from mask shapes — the single source of truth.
        mask[1]: (n_u, n_tiles)   mask[2]: (n_u, n_e)   mask[3]: (n_c, N_UNIT_TYPES)
        """
        n_u = mask[1].shape[0]
        n_e = mask[2].shape[1]
        n_c = mask[3].shape[0]
        return n_u, n_e, n_c

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, obs: dict, mask: list):
        """
        Sample a hierarchical action. obs is used only if a real encoder is
        later plugged in; sizes are always read from mask.

        Returns: (action, log_prob, entropy, value)
        """
        ctx            = self.global_ctx
        n_u, n_e, n_c = self._sizes_from_mask(mask)
        value          = self._value()

        at_d  = self._dist(self.action_type_head(ctx).squeeze(0), mask[0])
        at    = at_d.sample()
        log_p = at_d.log_prob(at)
        ent   = at_d.entropy()
        atype = ActionTypes(at.item())

        if atype == ActionTypes.MoveUnit and n_u > 0:
            u_emb  = self._rand(n_u)
            ud     = self._dist(self.unit_sel(ctx, u_emb).squeeze(0),
                                mask[1].max(axis=1))
            uid    = ud.sample()
            log_p += ud.log_prob(uid);    ent += ud.entropy()

            td     = self._dist(
                self.tile_sel(u_emb[uid:uid+1], self.tile_emb.weight).squeeze(0),
                mask[1][uid.item()])
            tid    = td.sample()
            log_p += td.log_prob(tid);    ent += td.entropy()
            action = [atype, uid.item(), tid.item()]

        elif atype == ActionTypes.Attack and n_u > 0 and n_e > 0:
            u_emb  = self._rand(n_u)
            e_emb  = self._rand(n_e)
            ud     = self._dist(self.unit_sel(ctx, u_emb).squeeze(0),
                                mask[2].max(axis=1))
            uid    = ud.sample()
            log_p += ud.log_prob(uid);    ent += ud.entropy()

            ed     = self._dist(
                self.enemy_sel(u_emb[uid:uid+1], e_emb).squeeze(0),
                mask[2][uid.item()])
            eid    = ed.sample()
            log_p += ed.log_prob(eid);    ent += ed.entropy()
            action = [atype, uid.item(), eid.item()]

        elif atype == ActionTypes.CreateUnit and n_c > 0:
            c_emb  = self._rand(n_c)
            cd     = self._dist(self.city_sel(ctx, c_emb).squeeze(0),
                                mask[3].max(axis=1))
            cid    = cd.sample()
            log_p += cd.log_prob(cid);    ent += cd.entropy()

            utd    = self._dist(
                self.utype_sel(c_emb[cid:cid+1], self.unit_type_emb.weight).squeeze(0),
                mask[3][cid.item()])
            utid   = utd.sample()
            log_p += utd.log_prob(utid);  ent += utd.entropy()
            action = [atype, cid.item(), utid.item()]

        elif atype == ActionTypes.CaptureCity and n_u > 0:
            u_emb  = self._rand(n_u)
            ud     = self._dist(self.unit_sel(ctx, u_emb).squeeze(0), mask[4])
            uid    = ud.sample()
            log_p += ud.log_prob(uid);    ent += ud.entropy()
            action = [atype, uid.item()]

        else:
            action = [ActionTypes.EndTurn]

        return action, log_p, ent, value

    # ── evaluate_actions ──────────────────────────────────────────────────────

    def evaluate_actions(self, actions: list, masks: list):
        """
        Re-score stored transitions under current weights for the PPO loss.
        obs_sizes removed — sizes are derived from masks directly.

        Returns: log_probs (T,), entropies (T,), values (T,)
        """
        ctx = self.global_ctx
        dev = ctx.device          # single source of truth — works on CPU and GPU
        log_probs, entropies, values = [], [], []

        for action, mask in zip(actions, masks):
            n_u, n_e, n_c = self._sizes_from_mask(mask)
            values.append(self._value())

            atype = action[0]
            if not isinstance(atype, ActionTypes):
                atype = ActionTypes(int(atype))
            at_t  = torch.tensor(atype.value, dtype=torch.long, device=dev)

            at_d  = self._dist(self.action_type_head(ctx).squeeze(0), mask[0])
            log_p = at_d.log_prob(at_t)
            ent   = at_d.entropy()

            if atype == ActionTypes.MoveUnit and len(action) == 3 and n_u > 0:
                uid, tid = action[1], action[2]
                u_emb    = self._rand(n_u)
                ud       = self._dist(self.unit_sel(ctx, u_emb).squeeze(0),
                                      mask[1].max(axis=1))
                log_p   += ud.log_prob(torch.tensor(uid, device=dev));  ent += ud.entropy()
                td       = self._dist(
                    self.tile_sel(u_emb[uid:uid+1], self.tile_emb.weight).squeeze(0),
                    mask[1][uid])
                log_p   += td.log_prob(torch.tensor(tid, device=dev));  ent += td.entropy()

            elif atype == ActionTypes.Attack and len(action) == 3 and n_u > 0 and n_e > 0:
                uid, eid = action[1], action[2]
                u_emb    = self._rand(n_u)
                e_emb    = self._rand(n_e)
                ud       = self._dist(self.unit_sel(ctx, u_emb).squeeze(0),
                                      mask[2].max(axis=1))
                log_p   += ud.log_prob(torch.tensor(uid, device=dev));  ent += ud.entropy()
                ed       = self._dist(
                    self.enemy_sel(u_emb[uid:uid+1], e_emb).squeeze(0),
                    mask[2][uid])
                log_p   += ed.log_prob(torch.tensor(eid, device=dev));  ent += ed.entropy()

            elif atype == ActionTypes.CreateUnit and len(action) == 3 and n_c > 0:
                cid, utid = action[1], action[2]
                c_emb     = self._rand(n_c)
                cd        = self._dist(self.city_sel(ctx, c_emb).squeeze(0),
                                       mask[3].max(axis=1))
                log_p    += cd.log_prob(torch.tensor(cid,  device=dev));   ent += cd.entropy()
                utd       = self._dist(
                    self.utype_sel(c_emb[cid:cid+1], self.unit_type_emb.weight).squeeze(0),
                    mask[3][cid])
                log_p    += utd.log_prob(torch.tensor(utid, device=dev));  ent += utd.entropy()

            elif atype == ActionTypes.CaptureCity and len(action) == 2 and n_u > 0:
                uid   = action[1]
                u_emb = self._rand(n_u)
                ud    = self._dist(self.unit_sel(ctx, u_emb).squeeze(0), mask[4])
                log_p += ud.log_prob(torch.tensor(uid, device=dev));  ent += ud.entropy()

            log_probs.append(log_p)
            entropies.append(ent)

        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)
