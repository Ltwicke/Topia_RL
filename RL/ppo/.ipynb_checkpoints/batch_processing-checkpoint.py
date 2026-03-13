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
from typing import Generator, Tuple

import numpy as np
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler

from ppo.env_manager import TrainConfig


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
    dones:       np.ndarray,   # (T, N)  float32  1.0 = game over
    last_values: np.ndarray,   # (N,)    float32  bootstrap V(s_T)
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
    values      (T, N) float32  — critic estimate V(s_t) at collection time
    dones       (T, N) float32  — 1.0 only at game termination (not turn end)
    last_values (N,)   float32  — V(s_T) bootstrap; used for each player's
                                  final in-rollout step
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
    last_values[e], which is V(s_T) evaluated at the post-rollout state.
    This is the same approximation used in single-agent PPO (V(s_{T+1}) ≈ 0
    or critic bootstrap).  When dones[t_k, e] == 1 the game has ended so the
    correct bootstrap is 0.
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
            next_vals[-1]    = 0.0 if game_ended else last_values[e]

            # ── Backward GAE pass over player-p's own timeline ─────────────
            # not_done[i] = 1.0 unless the game terminated at p_steps[i].
            # (Turning end / player switch does NOT count as done here.)
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
                gae               = delta + gamma * gae_lam * p_not_done[i] * gae
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
        self.ret_normalizer = RunningMeanStd()

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

        adv, ret = compute_gae_per_player(
            raw_batch["rewards"],
            raw_batch["values"],
            raw_batch["dones"],
            raw_batch["last_values"],
            raw_batch["player_ids"],
            gamma   = cfg.gamma,
            gae_lam = cfg.gae_lambda,
        )

        t_gae = time.time() - t0

        # ── Flatten (T, N) → (B,) row-major (time-major ordering preserved) ──
        adv_np = adv.reshape(-1).astype(np.float32)
        ret_np = ret.reshape(-1).astype(np.float32)
        lp_np  = raw_batch["log_probs"].reshape(-1).astype(np.float32)

        # Update running stats once per update cycle, then normalise
        self.ret_normalizer.update(ret_np)
        ret_norm_np = self.ret_normalizer.normalize(ret_np)

        # ── Flatten list-of-lists (time-major) ───────────────────────────────
        flat_snaps = [s for step in raw_batch["obs_snaps"] for s in step]
        flat_acts  = [a for step in raw_batch["actions"]   for a in step]
        flat_masks = [m for step in raw_batch["masks"]      for m in step]

        # ── Logging helpers ───────────────────────────────────────────────────
        n_finished   = int(raw_batch["dones"].sum())
        n_won        = int(raw_batch["won_flags"].sum())
        total_reward = float(raw_batch["rewards"].sum())
        avg_ep_len   = cfg.batch_size / max(n_finished, 1)

        processed_batch = {
            # Training data
            "flat_snaps":  flat_snaps,
            "flat_acts":   flat_acts,
            "flat_masks":  flat_masks,
            "log_probs_np": lp_np,
            "adv_np":       adv_np,
            "ret_norm_np":  ret_norm_np,
            # Logging
            "n_finished":   n_finished,
            "n_won":        n_won,
            "total_reward": total_reward,
            "avg_ep_len":   avg_ep_len,
        }
        return processed_batch, t_gae

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
        B   = cfg.batch_size
        mb  = cfg.minibatch_size

        # ── Convert numpy → CPU tensors once per epoch call ──────────────────
        log_old  = torch.from_numpy(processed_batch["log_probs_np"])  # (B,)
        adv_full = torch.from_numpy(processed_batch["adv_np"])        # (B,)
        ret_norm = torch.from_numpy(processed_batch["ret_norm_np"])   # (B,)

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
                "log_old":  log_old[idx],    # (mb,) CPU tensor
                "adv":      adv_full[idx],   # (mb,) CPU tensor
                "ret_norm": ret_norm[idx],   # (mb,) CPU tensor
            }





