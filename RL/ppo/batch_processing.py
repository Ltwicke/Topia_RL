"""
ppo/batch_processor.py
──────────────────────────────────────────────────────────────────────────────
Transforms a raw rollout batch (from EnvManager) into training-ready tensors.

Key responsibilities
────────────────────
1. Per-player GAE  — compute advantages and returns independently for each
   player's MDP stream, allowing credit assignment to span turn boundaries.
   Player p's "next state" is the next board state that player p observes
   (i.e., after the opponent has completed their turn), not the immediately
   following global step.

2. Normalisation  — RunningMeanStd (Welford online) for return targets;
   per-batch advantage whitening (zero mean, unit variance) before yielding.

3. Minibatch generation  — a generator function backed by PyTorch's
   BatchSampler that:
     • draws a fresh random permutation of all B sample indices each epoch,
     • retains only the first floor(B × train_fraction) indices, and
     • cuts them into fixed-size minibatches (drop_last=True).
   Calling the generator again in the next epoch re-draws the permutation,
   so the discarded fraction is different every epoch.

Per-player GAE — conceptual basis
──────────────────────────────────
Standard PPO for a single agent:
    δ_t  = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
    Â_t  = δ_t + γλ · (1 − done_t) · Â_{t+1}

For alternating-turn two-player self-play, instead of truncating Â at every
player switch, we index time by player-p's own decision points:

    Let {t_0 < t_1 < … < t_k} be all global steps where player p acted.
    Then:
        δ_i  = r_{t_i} + γ · V(s_{t_{i+1}}) · (1 − done_{t_i}) − V(s_{t_i})
        Â_i  = δ_i + γλ · (1 − done_{t_i}) · Â_{i+1}

    V(s_{t_{i+1}}) is the critic's value at the NEXT state player p observes
    — which already implicitly encodes the outcome of the opponent's turn as
    part of the transition dynamics.  No explicit reward signal is needed for
    the "waiting" period; the value function handles it.

    The discount γ is applied once per player-p decision (not once per global
    action), which is the standard convention in alternating-turn game RL
    (KataGo, OpenSpiel PPO, AlphaZero-style self-play).
"""

from __future__ import annotations

import time
from typing import Generator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler

from RL.ppo.game_manager import TrainConfig
from RL.models.main_modules import V_TERM, V_DENSE, N_VALUE_STREAMS


def _explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    """1 - Var(target - pred) / Var(target); 0 means no better than the mean.

    This is the metric that actually says whether a critic is learning. The
    value loss is computed on normalised returns, so its magnitude carries no
    information about fit quality — a critic stuck predicting a constant can
    show a very small v_loss.
    """
    if target.size == 0:
        return 0.0
    var_t = float(np.var(target))
    if var_t < 1e-12:
        return 0.0
    return float(1.0 - np.var(target - pred) / var_t)


# ══════════════════════════════════════════════════════════════════════════════
# Welford running mean / std
# ══════════════════════════════════════════════════════════════════════════════

class RunningMeanStd:
    """
    Online Welford estimator for return normalisation.

    Maintains a running mean and variance across all update cycles, so the
    normalisation adapts as the policy improves and reward magnitudes shift.
    """

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon   # small non-zero start avoids div-by-zero

    def update(self, x: np.ndarray) -> None:
        x           = np.asarray(x, dtype=np.float64).ravel()
        batch_mean  = x.mean()
        batch_var   = x.var()
        batch_count = x.size
        delta       = batch_mean - self.mean
        tot_count   = self.count + batch_count
        self.mean   = self.mean + delta * batch_count / tot_count
        M2 = (
            self.var    * self.count
            + batch_var * batch_count
            + delta**2  * self.count * batch_count / tot_count
        )
        self.var   = M2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / (np.sqrt(self.var) + 1e-8)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Per-player Generalised Advantage Estimation
# ══════════════════════════════════════════════════════════════════════════════

def compute_gae_per_player(
    rewards:     np.ndarray,   # (T, N)  float32
    values:      np.ndarray,   # (T, N)  float32
    dones:       np.ndarray,   # (T, N)  float32  1.0 = trajectory cut
    last_values: np.ndarray,   # (N, 2)  float32  bootstrap V(s_T) per seat
    player_ids:  np.ndarray,   # (T, N)  int32
    gamma:       float,
    gae_lam:     float,
    n_players:   int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE independently for each player's action subsequence.

    Unlike a single-agent rollout where t+1 is the globally next step, here
    t+1 for player p is the NEXT step where player p acts — skipping the
    opponent's intervening actions entirely.  The critic V(s_{t+1 for p})
    already encodes expected future value from player p's perspective
    conditioned on the opponent's behaviour.

    Parameters
    ──────────
    rewards     (T, N) float32  — immediate reward after each action
    values      (T, N) float32  — critic estimate V(s_t)
    dones       (T, N) float32  — 1.0 at trajectory cuts (the acting player's
                                  terminal step AND the other player's last
                                  decision step in the same game)
    last_values (N, 2) float32  — V(s_T) bootstrap per seat; used for that
                                  player's final in-rollout step
    player_ids  (T, N) int32    — player index who acted at each (t, e) slot
    gamma       float           — discount factor (applied per player-step)
    gae_lam     float           — GAE λ
    n_players   int             — number of players (default 2)

    Returns
    ───────
    advantages  (T, N) float32  — Â_t for every (t, e) slot
    returns     (T, N) float32  — advantages + values  (value-loss targets)

    Bootstrap approximation
    ───────────────────────
    For player p's last step in the rollout (index t_k), the true next state
    V(s_{t_{k+1}}) is beyond the rollout horizon.  We approximate it with
    last_values[e, p], which is V(s_T) evaluated at the post-rollout state from
    player p's own perspective — the post-rollout observation belongs to
    whoever moves next, so the other seat's value is its negation.
    When dones[t_k, e] == 1 the game has ended so the correct bootstrap is 0.

    Note (frozen-opponent self-play): advantages are computed for every player,
    but only the active seat's slots are kept downstream (filtered in
    BatchProcessor.process via the `is_active` mask). The non-active player's
    advantages are harmless and discarded.
    """
    T, N       = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)

    for p in range(n_players):
        for e in range(N):
            # Ordered global timesteps where player p acted in env e
            p_steps: np.ndarray = np.nonzero(player_ids[:, e] == p)[0]  # (k,)

            if p_steps.size == 0:
                continue

            k = p_steps.size

            # ── Vectorised next-value lookup ───────────────────────────────
            # next_vals[i] = V(s_{t_{i+1} for player p})
            next_vals = np.empty(k, dtype=np.float32)

            # All steps except the last: next value is at the following
            # player-p step within the rollout.
            if k > 1:
                next_vals[:-1] = values[p_steps[1:], e]

            # Last step: bootstrap from last_values (or 0 if game ended)
            last_t           = p_steps[-1]
            game_ended       = dones[last_t, e] > 0.5
            next_vals[-1]    = 0.0 if game_ended else last_values[e, p]

            # ── Backward GAE pass over player-p's own timeline ─────────────
            # not_done[i] = 1.0 unless the game terminated at p_steps[i].
            # (Turn end / player switch does NOT count as done here.)
            p_not_done = 1.0 - dones[p_steps, e]      # (k,)
            p_rewards  = rewards[p_steps, e]            # (k,)
            p_values   = values[p_steps, e]             # (k,)

            gae = 0.0
            for i in range(k - 1, -1, -1):
                delta = (
                    p_rewards[i]
                    + gamma * next_vals[i] * p_not_done[i]
                    - p_values[i]
                )
                gae = delta + gamma * gae_lam * p_not_done[i] * gae
                advantages[p_steps[i], e] = gae

    returns = advantages + values
    return advantages, returns


# ══════════════════════════════════════════════════════════════════════════════
# BatchProcessor
# ══════════════════════════════════════════════════════════════════════════════

class BatchProcessor:
    """
    Converts a raw rollout batch into processed training data and provides
    minibatch iterators for PPOTrainer.

    The processor is stateful: RunningMeanStd accumulates across all updates.

    Parameters
    ──────────
    cfg : TrainConfig
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg            = cfg
        # One normaliser per reward stream. Sharing a single one would couple
        # the terminal and dense scales, so switching dense off would shift the
        # statistics under V_TERM — the precise failure the split heads exist
        # to prevent.
        self.ret_normalizer       = RunningMeanStd()   # terminal stream
        self.ret_normalizer_dense = RunningMeanStd()   # dense stream

    # ── Main processing step ──────────────────────────────────────────────────

    def process(self, raw_batch: dict) -> Tuple[dict, float]:
        """
        Run per-player GAE, normalise returns, and flatten all arrays.

        The RunningMeanStd is updated ONCE per call using the current batch's
        return distribution, then used to normalise those same returns.  This
        matches the original implementation (update before normalising).

        Parameters
        ──────────
        raw_batch : dict — output of EnvManager.collect()

        Returns
        ───────
        processed_batch : dict
            ── Training tensors (numpy, moved to device inside PPOTrainer) ──
            flat_snaps    list[dict]        B = T × N_total snapshots
            flat_acts     list[list]        B action lists
            flat_masks    list[list]        B mask lists
            log_probs_np  np.ndarray (B,)   old log π(a|s) — float32
            adv_np        np.ndarray (B,)   per-player advantages — float32
            ret_norm_np   np.ndarray (B,)   normalised returns — float32

            ── Logging stats (extracted from raw_batch scalars) ─────────────
            n_finished    int     episodes that ended (done flag fired)
            n_won         int     episodes that ended by conquest
            total_reward  float   sum of all rewards in the batch
            avg_ep_len    float   average steps per completed episode

        t_gae : float — seconds spent on GAE computation
        """
        cfg = self.cfg
        t0  = time.time()

        rew_term   = raw_batch["rew_term"]     # (T, N)
        rew_dense  = raw_batch["rew_dense"]    # (T, N)
        values     = raw_batch["values"]       # (T, N, n_streams)
        dones      = raw_batch["dones"]
        player_ids = raw_batch["player_ids"]
        last_vals  = raw_batch["last_values"]  # (N, 2 seats, n_streams)
        T, N       = rew_term.shape

        # ── One GAE pass per reward stream ────────────────────────────────────
        # GAE is linear in the rewards given a matching value function, so the
        # per-stream advantages sum EXACTLY to the advantage of the combined
        # reward. That is what makes dense_beta a clean dial rather than an
        # approximation, and what lets V_TERM stay uncontaminated by shaping.
        adv_term, ret_term = compute_gae_per_player(
            rew_term, values[:, :, V_TERM], dones,
            last_vals[:, :, V_TERM], player_ids,
            gamma   = cfg.gamma,
            gae_lam = cfg.gae_lambda,
        )
        adv_dense, ret_dense = compute_gae_per_player(
            rew_dense, values[:, :, V_DENSE], dones,
            last_vals[:, :, V_DENSE], player_ids,
            gamma   = cfg.gamma,
            gae_lam = cfg.gae_lambda,
        )

        t_gae = time.time() - t0

        # ── Flatten (T, N) → (B,) time-major, then filter the training set ────
        # Active seat only, and forced steps dropped: a step with a single
        # possible trajectory has log pi == 0 and entropy == 0 under the hard
        # mask, so it contributes no gradient while still dragging the entropy
        # bonus down and inflating the advantage-whitening std.
        is_active_flat = raw_batch["is_active"].reshape(-1) > 0.5
        is_forced_flat = raw_batch.get(
            "is_forced", np.zeros_like(raw_batch["is_active"])
        ).reshape(-1) > 0.5
        train_mask = is_active_flat & (~is_forced_flat)
        active_idx = np.nonzero(train_mask)[0]
        n_forced_dropped = int((is_active_flat & is_forced_flat).sum())

        # Per-epoch recompute needs the full (T·N, n_streams) value table
        val_flat = values.reshape(-1, values.shape[-1]).astype(np.float32)

        adv_term_np  = adv_term.reshape(-1).astype(np.float32)[active_idx]
        adv_dense_np = adv_dense.reshape(-1).astype(np.float32)[active_idx]
        ret_term_np  = ret_term.reshape(-1).astype(np.float32)[active_idx]
        ret_dense_np = ret_dense.reshape(-1).astype(np.float32)[active_idx]
        lp_np        = raw_batch["log_probs"].reshape(-1).astype(np.float32)[active_idx]

        # The policy optimises the weighted sum; each critic head regresses its
        # own stream, so turning dense_beta to 0 changes the policy objective
        # without touching what V_TERM has learned.
        adv_np = adv_term_np + cfg.dense_beta * adv_dense_np

        # One normaliser PER STREAM. A shared one would couple the two scales,
        # so switching dense off would shift the statistics and drag V_TERM
        # with it — exactly the coupling this design exists to avoid.
        self.ret_normalizer.update(ret_term_np)
        ret_norm_np       = self.ret_normalizer.normalize(ret_term_np)
        self.ret_normalizer_dense.update(ret_dense_np)
        ret_dense_norm_np = self.ret_normalizer_dense.normalize(ret_dense_np)

        # ── Flatten list-of-lists (time-major), keep active only ──────────────
        all_snaps = [s for step in raw_batch["obs_snaps"] for s in step]
        all_acts  = [a for step in raw_batch["actions"]   for a in step]
        all_masks = [m for step in raw_batch["masks"]      for m in step]
        flat_snaps = [all_snaps[i] for i in active_idx]
        flat_acts  = [all_acts[i]  for i in active_idx]
        flat_masks = [all_masks[i] for i in active_idx]

        # ── Logging (game outcomes come from worker scalar counters) ──────────
        n_games           = int(raw_batch.get("n_games", 0))
        n_active_wins     = int(raw_batch.get("n_active_wins", 0))
        n_conquest        = int(raw_batch.get("n_conquest", 0))
        n_timeout         = int(raw_batch.get("n_timeout", 0))
        n_dropped_term    = int(raw_batch.get("n_dropped_terminal", 0))
        n_endturn         = int(raw_batch.get("n_endturn", 0))
        n_endturn_vol     = int(raw_batch.get("n_endturn_voluntary", 0))
        n_decisions_total = int(raw_batch.get("n_decisions_total", 0))
        active_reward_sum = float(
            (rew_term + rew_dense).reshape(-1)[active_idx].sum()
        )
        avg_ep_len        = active_idx.size / max(n_games, 1)
        # Anti-rush diagnostic: own decisions per own turn. Human-level play
        # should climb well above 2; a value near 1 means the policy is ending
        # its turn immediately to race the turn limit.
        decisions_per_turn = n_decisions_total / max(n_endturn, 1)
        # Fraction of games won outright rather than on a turn-30 score lead.
        conquest_rate      = n_conquest / max(n_games, 1)
        # Explained variance per head — the only honest read on critic quality.
        # v_loss is measured on NORMALISED returns and so cannot be compared to
        # the reward scale at all.
        ev_term  = _explained_variance(
            val_flat[active_idx, V_TERM], ret_term_np)
        ev_dense = _explained_variance(
            val_flat[active_idx, V_DENSE], ret_dense_np)

        processed_batch = {
            # Training data (active seat, real decisions only)
            "flat_snaps":   flat_snaps,
            "flat_acts":    flat_acts,
            "flat_masks":   flat_masks,
            "log_probs_np": lp_np,
            "adv_np":       adv_np,
            "ret_norm_np":       ret_norm_np,
            "ret_dense_norm_np": ret_dense_norm_np,
            # Inputs needed to recompute GAE each epoch (compute_values_batch)
            "rew_term_TN":            rew_term,
            "rew_dense_TN":           rew_dense,
            "dones_TN":               dones,
            "player_ids_TN":          player_ids,
            "last_values_N":          last_vals,
            "values_flat_collection": val_flat,
            "active_idx":             active_idx,
            "T":                      T,
            "N":                      N,
            # Logging
            "n_games":           n_games,
            "n_active_wins":     n_active_wins,
            "n_conquest":        n_conquest,
            "n_timeout":          n_timeout,
            "n_dropped_terminal": n_dropped_term,
            "n_forced_dropped":   n_forced_dropped,
            "decisions_per_turn": decisions_per_turn,
            "endturn_voluntary":  n_endturn_vol,
            "conquest_rate":      conquest_rate,
            "ev_term":            ev_term,
            "ev_dense":           ev_dense,
            "active_reward_sum":  active_reward_sum,
            "avg_ep_len":         avg_ep_len,
        }
        return processed_batch, t_gae

    # ── Per-epoch GAE recompute ───────────────────────────────────────────────

    def recompute_advantages(
        self,
        processed_batch:     dict,
        fresh_active_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recompute advantages/returns from refreshed critic values for the
        active seat. Called at the start of each PPO epoch (epoch ≥ 1).

        The running-mean/std stats are NOT updated here — only `normalize` is
        applied — so the return normalisation stays consistent across the
        update cycle (it was updated once in `process`).

        Parameters
        ──────────
        processed_batch     : dict — output of process()
        fresh_active_values : (n_active, n_streams) — per-stream V(s) for the
                              active snapshots, in the same order as
                              processed_batch["flat_snaps"].

        Returns
        ───────
        adv_np            (n_active,) float32 — fresh combined advantages
        ret_norm_np       (n_active,) float32 — fresh normalised terminal returns
        ret_dense_norm_np (n_active,) float32 — fresh normalised dense returns
        """
        cfg        = self.cfg
        active_idx = processed_batch["active_idx"]
        T, N       = processed_batch["T"], processed_batch["N"]
        last_vals  = processed_batch["last_values_N"]

        # Scatter fresh active values into the full (T·N, n_streams) table,
        # leaving non-active and forced slots at their collection values.
        vals_flat = processed_batch["values_flat_collection"].copy()
        vals_flat[active_idx] = np.asarray(fresh_active_values, dtype=np.float32)
        values_TNS = vals_flat.reshape(T, N, vals_flat.shape[-1])

        adv_term, ret_term = compute_gae_per_player(
            processed_batch["rew_term_TN"], values_TNS[:, :, V_TERM],
            processed_batch["dones_TN"], last_vals[:, :, V_TERM],
            processed_batch["player_ids_TN"],
            gamma   = cfg.gamma,
            gae_lam = cfg.gae_lambda,
        )
        adv_dense, ret_dense = compute_gae_per_player(
            processed_batch["rew_dense_TN"], values_TNS[:, :, V_DENSE],
            processed_batch["dones_TN"], last_vals[:, :, V_DENSE],
            processed_batch["player_ids_TN"],
            gamma   = cfg.gamma,
            gae_lam = cfg.gae_lambda,
        )

        adv_np = (adv_term.reshape(-1).astype(np.float32)[active_idx]
                  + cfg.dense_beta
                  * adv_dense.reshape(-1).astype(np.float32)[active_idx])
        ret_norm_np = self.ret_normalizer.normalize(
            ret_term.reshape(-1).astype(np.float32)[active_idx])
        ret_dense_norm_np = self.ret_normalizer_dense.normalize(
            ret_dense.reshape(-1).astype(np.float32)[active_idx])
        return adv_np, ret_norm_np, ret_dense_norm_np

    # ── Minibatch generator ───────────────────────────────────────────────────

    def minibatch_generator(
        self,
        processed_batch: dict,
        train_fraction:  float = 1.0,
    ) -> Generator[dict, None, None]:
        """
        Yield fixed-size minibatches for one PPO epoch.

        Algorithm
        ─────────
        1. Whiten advantages over ALL B samples (before sub-sampling), so
           that the normalisation is consistent regardless of fraction.
        2. Draw a uniformly random permutation of all B indices.
        3. Retain the first n_train = max(minibatch_size, floor(B × fraction))
           indices.  The max() guarantees at least one complete minibatch.
        4. Pass the retained indices through PyTorch's BatchSampler
           (SubsetRandomSampler → randomly ordered within the retained set,
           BatchSampler → cut into fixed-size batches, drop_last=True).

        Calling this method again (next epoch) re-draws step 2, so different
        samples are discarded each epoch, providing implicit coverage of the
        full batch over multiple epochs even when fraction < 1.

        Parameters
        ──────────
        processed_batch : dict — output of process()
        train_fraction  : float ∈ (0, 1] — fraction of B to use per epoch

        Yields
        ──────
        minibatch : dict
            snaps    list[dict]           minibatch_size snapshots
            acts     list[list]           minibatch_size actions
            masks    list[list]           minibatch_size masks
            log_old  torch.Tensor (mb,)   float32  — old log-probs (CPU)
            adv      torch.Tensor (mb,)   float32  — whitened advantages (CPU)
            ret_norm torch.Tensor (mb,)   float32  — normalised returns (CPU)

        All tensors are on CPU; PPOTrainer moves them to the target device
        inside _step() for maximum memory efficiency.
        """
        cfg = self.cfg
        # B is the number of ACTIVE-seat samples (≤ cfg.batch_size after the
        # frozen-opponent filter), derived from the array length itself.
        B   = processed_batch["adv_np"].shape[0]
        mb  = cfg.minibatch_size

        # ── Convert numpy → CPU tensors once per epoch call ──────────────────
        log_old   = torch.from_numpy(processed_batch["log_probs_np"])  # (B,)
        adv_full  = torch.from_numpy(processed_batch["adv_np"])        # (B,)
        ret_norm  = torch.from_numpy(processed_batch["ret_norm_np"])   # (B,)
        ret_dense = torch.from_numpy(processed_batch["ret_dense_norm_np"])  # (B,)

        # ── Whiten advantages over the full batch before sub-sampling ─────────
        # Normalisation is computed on all B samples so that the scale is
        # consistent; sub-sampling afterwards does not re-normalise.
        adv_full = (adv_full - adv_full.mean()) / (adv_full.std() + 1e-8)

        flat_snaps = processed_batch["flat_snaps"]
        flat_acts  = processed_batch["flat_acts"]
        flat_masks = processed_batch["flat_masks"]

        # ── Select training subset ─────────────────────────────────────────────
        n_train = max(mb, int(B * train_fraction))   # at least one full minibatch

        # Full random permutation, then take the first n_train
        perm      = torch.randperm(B)
        selected  = perm[:n_train]                   # (n_train,) — already random

        # BatchSampler gives us evenly-sized minibatches from `selected`.
        # SubsetRandomSampler re-shuffles within the selected set, which is
        # fine (guarantees random ordering of minibatches within the epoch).
        sampler = BatchSampler(
            SubsetRandomSampler(selected.tolist()),
            batch_size=mb,
            drop_last=True,
        )

        for idx_list in sampler:
            idx = torch.tensor(idx_list, dtype=torch.long)
            yield {
                "snaps":    [flat_snaps[i] for i in idx_list],
                "acts":     [flat_acts[i]  for i in idx_list],
                "masks":    [flat_masks[i] for i in idx_list],
                "log_old":        log_old[idx],    # (mb,) CPU tensor
                "adv":            adv_full[idx],   # (mb,) CPU tensor
                "ret_norm":       ret_norm[idx],   # (mb,) terminal-stream target
                "ret_dense_norm": ret_dense[idx],  # (mb,) dense-stream target
            }



# ══════════════════════════════════════════════════════════════════════════════
# EstimatorBatchProcessor — Phase A (HiddenTileEstimator pretraining)
# ══════════════════════════════════════════════════════════════════════════════

class EstimatorBatchProcessor:
    """
    Builds random minibatches of snapshot subsets for HiddenTileEstimator
    pretraining.  Each minibatch is a list[dict] (snapshots);
    `policy.estimator_loss` is the differentiable consumer.

    Unlike the PPO BatchProcessor this processor does NOT compute GAE,
    advantages, or return statistics — the estimator objective is purely
    supervised and only needs the raw obs snapshots.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

    def minibatch_generator(
        self,
        raw_batch:      dict,
        train_fraction: Optional[float] = None,
    ) -> Generator[List[dict], None, None]:
        """
        Yield list[dict] minibatches sampled uniformly without replacement
        from the full B = T × N_total rollout.

        Algorithm
        ─────────
        1. Flatten ``raw_batch["obs_snaps"]`` (list[T] × list[M_total]) into a
           single list of B snapshots.
        2. Draw a uniformly random permutation of all B indices and retain
           the first ``n_train = max(mb, floor(B × train_fraction))``.
        3. Pass through PyTorch's ``BatchSampler`` (``drop_last=True``).

        Calling this method again re-draws step 2 — call once per epoch
        to get a fresh permutation each epoch.

        Parameters
        ──────────
        raw_batch      : dict   — output of EnvManager.collect()
        train_fraction : float  — defaults to cfg.estimator_train_fraction

        Yields
        ──────
        snaps : list[dict] of length cfg.estimator_minibatch_size
        """
        flat_snaps = [s for step in raw_batch["obs_snaps"] for s in step]
        B    = len(flat_snaps)
        mb   = self.cfg.estimator_minibatch_size
        frac = (
            self.cfg.estimator_train_fraction
            if train_fraction is None
            else train_fraction
        )
        n_train = max(mb, int(B * frac))

        perm    = torch.randperm(B)[:n_train].tolist()
        sampler = BatchSampler(
            SubsetRandomSampler(perm),
            batch_size=mb,
            drop_last=True,
        )
        for idx_list in sampler:
            yield [flat_snaps[i] for i in idx_list]
