"""
ppo/env_manager.py
──────────────────────────────────────────────────────────────────────────────
Contains:
  • TrainConfig  — single unified hyperparameter dataclass shared by all
                   modules (imported from here).
  • worker_fn()  — top-level (module-level) function executed in each worker
                   subprocess.  Must be top-level for multiprocessing pickling
                   under the 'spawn' start method.
  • EnvManager   — spawns the worker pool, distributes policy weights, and
                   assembles the raw rollout batch.

Design notes
────────────

The worker collects all T steps for all M environments without any
player-switch truncation: the full temporal structure is preserved so that
BatchProcessor can build independent per-player MDP streams.
"""

from __future__ import annotations

import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

# ── Project root ─────────────────────────────────────────────────────────────
# Must be set before any project-relative imports, both here (main process)
# and inside worker_fn (each spawned subprocess reimports this module).
_PROJECT_ROOT: str = r"C:\Users\laure\1own_projects\1polytopia_score"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env.wrapper import EnvWrapper
from game.enums  import BoardType, Tribes
from RL.models.policy import PolicyNetwork, make_snapshot


# ══════════════════════════════════════════════════════════════════════════════
# Unified hyperparameter config
# (imported by BatchProcessor, PPOTrainer, and train.py)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """
    Single source of truth for every hyperparameter in the training pipeline.

    Architecture fields are forwarded to PolicyNetwork.__init__().
    All other fields are consumed by EnvManager, BatchProcessor, or PPOTrainer.

    train_fraction : float ∈ (0, 1]
        Fraction of the assembled minibatches (per epoch) that are actually
        used for gradient updates.  Set < 1 to balance simulation vs. PPO
        update wall time when the PPO update is the bottleneck.
        1.0 = use the entire batch (standard PPO).
    """

    # ── Checkpoint / resume ───────────────────────────────────────────────────
    pretrained_ckpt: str = r"./checkpoints_training/policy_update_00002.pt"   # path to .pt; "" = train from scratch
    start_update:    int = 3    # first update index (set > 0 when resuming)

    # ── Encoder ───────────────────────────────────────────────────────────────
    encoder_hidden_dim: int = 48
    encoder_n_heads:    int = 4
    encoder_depth:      int = 2

    # ── Selection heads ───────────────────────────────────────────────────────
    sel_n_heads:  int = 4
    sel_n_layers: int = 2

    # ── MLP ───────────────────────────────────────────────────────────────────
    mlp_hidden_dim: int = 64
    mlp_depth:      int = 2

    # ── Multi-scale convolutions ──────────────────────────────────────────────
    kernel_sizes:  Tuple[int, ...] = (5,3)
    n_conv_layers: int             = 1

    # ── Movement context window ───────────────────────────────────────────────
    context_bias: int = 5

    # ── Parallelism ───────────────────────────────────────────────────────────
    n_processes:        int = 14
    n_envs_per_process: int = 2

    # ── Environment ───────────────────────────────────────────────────────────
    # board_type is randomised per env from board_type_pool — Dummy is dropped.
    board_config_dict: dict = field(default_factory=lambda: {
        "n_players":  2,
    })
    board_type_pool: tuple = field(
        default_factory=lambda: (
            BoardType.Drylands,
            BoardType.Lakes,
            BoardType.Archipelago,
        )
    )
    player_tribes:      list  = field(
        default_factory=lambda: [Tribes.Omaji, Tribes.Imperius]
    )
    max_turns_per_game: int   = 30
    board_size_range:   tuple = (11, 16)

    # ── Reward shaping ────────────────────────────────────────────────────────
    # terminal_reward_mode: "none" | "zero_sum" | "winner_only"
    #   winner_only — winner +diff, loser 0 (no punishment; current run)
    dense_reward:         bool  = True
    terminal_reward_mode: str   = "winner_only"
    win_reward:           float = 500.0   # flat bonus to the CONQUEROR's score
                                          # before the diff (timeout adds no bonus)

    # ── Frozen-opponent self-play ─────────────────────────────────────────────
    # The trained policy occupies seat `active_player_id`; the other seat is
    # played by a frozen copy refreshed every `opponent_refresh_interval`
    # updates. Only the active seat's transitions train PPO.
    self_play_frozen_opponent: bool = True
    active_player_id:          int  = 0
    opponent_refresh_interval: int  = 10

    # ── Estimator pretraining (Phase A of each update) ────────────────────────
    estimator_lr:             float = 3e-4
    estimator_n_epochs:       int   = 4
    estimator_minibatch_size: int   = 512
    estimator_train_fraction: float = 1.0    

    # ── Scenario eval (Phase C — runs after PPO update) ───────────────────────
    scenario_dir:             str   = "scenarios/scenarios"
    scenario_eval_interval:   int   = 1      # run scenarios every N updates; 0 disables
    scenario_names:           list  = field(
        default_factory=lambda: [
            "Knight_choice_no_village",
            "Rider_leapfrogging",
            "Simple_dash_dancing2",
            "road_for_kill",
            "estimate_lakes11",
            "Escaping_riders2",
            "Upgrade_city_order2",
            "Giant_Houdini",
            "Defender_ZoC",
            "Estimate_Drylands_endgame",
            "Rider_hit_and_run",
            "Dont_attack",
        ]
    )

    # ── Rollout ───────────────────────────────────────────────────────────────
    n_steps: int = 512

    # ── PPO epochs & batching ─────────────────────────────────────────────────
    n_epochs:       int   = 3
    n_minibatches:  int   = 128   # determines cfg.minibatch_size
    # Fraction ∈ (0,1]: what share of the assembled minibatches to train on
    # per epoch.  Reduces PPO update time without wasting simulation data.
    train_fraction: float = 1.0

    # ── PPO loss coefficients ─────────────────────────────────────────────────
    clip_eps:      float = 0.2
    vf_coef:       float = 0.5
    ent_coef:      float = 0.005
    max_grad_norm: float = 0.5

    # ── GAE / discount ────────────────────────────────────────────────────────
    gamma:                    float = 0.99
    gae_lambda:               float = 0.95   # single λ for all trajectories
    recompute_gae_each_epoch: bool  = True   # refresh values + GAE before each
                                              # PPO epoch (uses compute_values_batch)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr: float = 3e-4

    # ── Training loop ─────────────────────────────────────────────────────────
    n_updates:     int = 2500
    log_interval:  int = 1
    ckpt_interval:           int = 1     # rolling checkpoints (last MAX_CKPT kept)
    permanent_ckpt_interval: int = 100   # never-evicted snapshots every N updates

    # ── Speed / diagnostic ────────────────────────────────────────────────────
    use_amp:    bool = True   # AMP mixed precision training
    track_vram: bool = False   # print VRAM stats each update

    # ── Derived properties ────────────────────────────────────────────────────
    @property
    def n_envs_total(self) -> int:
        return self.n_processes * self.n_envs_per_process

    @property
    def batch_size(self) -> int:
        """Total samples per update = T × N_total."""
        return self.n_steps * self.n_envs_total

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.n_minibatches

    # ── Validation ───────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        assert self.encoder_hidden_dim % self.encoder_n_heads == 0, (
            f"encoder_hidden_dim ({self.encoder_hidden_dim}) must be divisible "
            f"by encoder_n_heads ({self.encoder_n_heads})."
        )
        assert self.encoder_hidden_dim % 4 == 0, (
            "encoder_hidden_dim must be divisible by 4 for 2-D RoPE."
        )
        assert all(k % 2 == 1 for k in self.kernel_sizes), \
            "All kernel_sizes must be odd."
        assert self.context_bias >= max(self.kernel_sizes) // 2, (
            f"context_bias ({self.context_bias}) must be >= "
            f"max(kernel_sizes)//2 ({max(self.kernel_sizes) // 2})."
        )
        assert self.mlp_depth     >= 1,   "mlp_depth must be >= 1."
        assert self.n_conv_layers >= 1,   "n_conv_layers must be >= 1."
        assert self.n_minibatches >= 1,   "n_minibatches must be >= 1."
        assert self.batch_size % self.n_minibatches == 0, (
            f"batch_size ({self.batch_size}) must be divisible by "
            f"n_minibatches ({self.n_minibatches})."
        )
        assert 0.0 < self.train_fraction <= 1.0, \
            "train_fraction must be in (0, 1]."


# ══════════════════════════════════════════════════════════════════════════════
# Environment factory helpers
# ══════════════════════════════════════════════════════════════════════════════

def _random_board_size(cfg: TrainConfig) -> tuple:
    n = random.randint(cfg.board_size_range[0], cfg.board_size_range[1])
    return (n, n)


def _make_env(cfg: TrainConfig) -> EnvWrapper:
    n          = random.randint(*cfg.board_size_range)
    board_type = random.choice(cfg.board_type_pool)
    board_config = {
        "board_size": [n, n],
        "board_type": board_type,
        "n_players":  cfg.board_config_dict["n_players"],
    }
    return EnvWrapper(
        board_config,
        cfg.player_tribes,
        max_turns_per_game=cfg.max_turns_per_game,
        win_reward=cfg.win_reward,
        dense_reward=cfg.dense_reward,
        terminal_reward_mode=cfg.terminal_reward_mode,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Worker function — must be module-level for pickle / mp.spawn
# ══════════════════════════════════════════════════════════════════════════════

def worker_fn(worker_id: int, cfg: TrainConfig, conn) -> None:
    """
    Collect T-step rollouts across M environments on CPU.

    Each spawned subprocess re-imports this module, so _PROJECT_ROOT is
    re-inserted into sys.path at module load time above — no extra setup
    needed here.

    Frozen-opponent self-play
    ──────────────────────────
    Two policies live in the worker: the active (trained) policy occupies seat
    `cfg.active_player_id`, a frozen older copy occupies the other seat. Both
    are run to drive realistic game dynamics, but only the active seat's
    transitions are flagged `is_active=1` and used for PPO downstream.

    Protocol
    ────────
    Main → Worker : ('collect', {'active': sd, 'frozen': sd})
    Worker → Main : ('data',    chunk_dict)

    Main → Worker : ('stop', None)
    Worker exits cleanly.

    Chunk arrays
    ────────────
    log_probs   (T, M)  float32     — log π_θ_old(a_t | s_t)
    values      (T, M)  float32     — V(s_t) at collection time
    rewards     (T, M)  float32     — immediate reward after action
    dones       (T, M)  float32     — 1.0 at trajectory cuts (the acting
                                      player's terminal step AND the other
                                      player's last decision step, so per-player
                                      GAE terminates both sides correctly)
    is_active   (T, M)  float32     — 1.0 where the active seat acted
    last_values (M,)    float32     — V(s_T) bootstrap (active critic)
    player_ids  (T, M)  int32       — which player acted at each step
    obs_snaps   list[T] × list[M]   — make_snapshot() dicts (for evaluate_actions)
    actions     list[T] × list[M]   — stored action lists
    masks       list[T] × list[M]   — stored action masks
    n_games / n_active_wins / n_conquest / n_timeout : int — logging scalars
    """
    # Ensure project root is on path in the worker process
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    envs    = [_make_env(cfg) for _ in range(cfg.n_envs_per_process)]
    obs_buf = [env.reset()    for env in envs]

    # Active (trained) policy + frozen opponent copy. Both run in eval mode;
    # only the active policy's transitions are used for PPO downstream.
    policy_active = PolicyNetwork(cfg)
    policy_frozen = PolicyNetwork(cfg)
    policy_active.eval()
    policy_frozen.eval()
    active_pid = cfg.active_player_id

    M = cfg.n_envs_per_process
    T = cfg.n_steps

    while True:
        cmd, payload = conn.recv()
        if cmd == "stop":
            break

        # Load latest weights (CPU tensors from main process).
        # payload = {'active': state_dict, 'frozen': state_dict}
        policy_active.load_state_dict(payload["active"])
        policy_frozen.load_state_dict(payload["frozen"])

        # Pre-allocate rollout buffers
        obs_snaps  = [[None] * M for _ in range(T)]
        actions    = [[None] * M for _ in range(T)]
        masks_buf  = [[None] * M for _ in range(T)]
        log_probs  = np.zeros((T, M), dtype=np.float32)
        values     = np.zeros((T, M), dtype=np.float32)
        rewards    = np.zeros((T, M), dtype=np.float32)
        dones      = np.zeros((T, M), dtype=np.float32)
        is_active  = np.zeros((T, M), dtype=np.float32)
        # acting-player track filled during rollout so we can back-fill the
        # opponent's last decision step at game termination
        player_ids = np.full((T, M), -1, dtype=np.int32)

        # Per-chunk logging scalars
        n_games = n_active_wins = n_conquest = n_timeout = 0

        t0 = time.time()
        with torch.no_grad():
            for t in range(T):
                for e, env in enumerate(envs):
                    obs  = obs_buf[e]
                    mask = env.get_action_mask()

                    # Player who is about to act at this global step
                    cur_pid = env.game.player_go_id

                    # Snapshot must be taken BEFORE env.step() so that the
                    # stored graph / tile IDs match the decision state.
                    snap = make_snapshot(
                        obs, env.Nx, env.Ny,
                        player_id=cur_pid,
                    )

                    # Dispatch on seat: active policy plays `active_pid`,
                    # frozen copy plays the other seat.
                    pol = policy_active if cur_pid == active_pid else policy_frozen

                    # forward() returns:
                    # action, joint_probs, traj_actions, log_prob, entropy, value
                    action, _, _, lp, _, val = pol(obs, mask)
                    next_obs, rew, done, info = env.step(action)

                    obs_snaps[t][e]   = snap
                    actions[t][e]     = action
                    masks_buf[t][e]   = mask
                    log_probs[t, e]   = lp.item()
                    values[t, e]      = val.item()
                    rewards[t, e]     = rew
                    dones[t, e]       = float(done)
                    is_active[t, e]   = 1.0 if cur_pid == active_pid else 0.0
                    player_ids[t, e]  = cur_pid

                    if done:
                        winner_id = info.get("winner_id", None)
                        is_conq   = bool(info.get("is_conquest", False))
                        r_opp     = float(info.get("reward_opp", 0.0))

                        # ── Back-fill the OTHER player's last decision step in
                        # the current game segment (since the most recent earlier
                        # `done` in this env, or 0). Mark that step `done=1` so
                        # per-player GAE terminates that trajectory, and add the
                        # other player's terminal reward (0 for the loser in
                        # winner_only mode).
                        opp_id = (cur_pid + 1) % 2
                        if t > 0:
                            prev_dones = np.nonzero(dones[:t, e] > 0.5)[0]
                            game_start = 0 if prev_dones.size == 0 else int(prev_dones[-1]) + 1
                            seg_pids = player_ids[game_start:t, e]
                            opp_steps = np.nonzero(seg_pids == opp_id)[0]
                            if opp_steps.size > 0:
                                last_opp_t = game_start + int(opp_steps[-1])
                                rewards[last_opp_t, e] += r_opp
                                dones[last_opp_t, e]    = 1.0

                        # ── Logging counters ──
                        n_games      += 1
                        n_active_wins += int(winner_id == active_pid)
                        n_conquest    += int(is_conq)
                        n_timeout     += int(not is_conq)

                        # Reset to a new random-size episode
                        envs[e]    = _make_env(cfg)
                        obs_buf[e] = envs[e].reset()
                    else:
                        obs_buf[e] = next_obs

            # ── Bootstrap V(s_T) for each env slot ───────────────────────────
            # obs_buf[e] now holds the first obs of a new episode (if done) or
            # the observation that follows the last collected step.
            last_values = np.zeros(M, dtype=np.float32)
            for e in range(M):
                snap_last = make_snapshot(
                    obs_buf[e], envs[e].Nx, envs[e].Ny,
                    player_id=envs[e].game.player_go_id,
                )
                _, global_emb  = policy_active.encoder.encode(
                    snap_last["graph"], snap_last["Nx"], snap_last["Ny"],
                )
                last_values[e] = policy_active.critic(global_emb).item()

        # `player_ids` is already filled in-loop (used for back-fill).

        print(
            f"  [worker {worker_id:02d}] rollout done — {time.time() - t0:.2f}s",
            flush=True,
        )

        conn.send(("data", {
            "obs_snaps":     obs_snaps,
            "actions":       actions,
            "masks":         masks_buf,
            "log_probs":     log_probs,
            "values":        values,
            "rewards":       rewards,
            "dones":         dones,
            "is_active":     is_active,
            "last_values":   last_values,
            "player_ids":    player_ids,
            "n_games":       n_games,
            "n_active_wins": n_active_wins,
            "n_conquest":    n_conquest,
            "n_timeout":     n_timeout,
        }))


# ══════════════════════════════════════════════════════════════════════════════
# EnvManager
# ══════════════════════════════════════════════════════════════════════════════

class EnvManager:
    """
    Manages the worker subprocess pool.

    Workflow per training update
    ────────────────────────────
    1. manager.distribute(active_sd, frozen_sd)  — push both policies to workers
    2. manager.collect()                   — block until all chunks arrive,
                                             assemble and return raw batch

    The raw batch is a dict with arrays of shape (T, N_total) (numeric data)
    or lists-of-lists (obs_snaps / actions / masks).  It is passed as-is to
    BatchProcessor.process().
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg           = cfg
        self._parent_conns: list = []
        self._workers:      list = []

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn all worker subprocesses.  Call once before the training loop."""
        pipes = [mp.Pipe() for _ in range(self.cfg.n_processes)]
        self._parent_conns = [p[0] for p in pipes]
        child_conns        = [p[1] for p in pipes]

        self._workers = [
            mp.Process(
                target=worker_fn,
                args=(i, self.cfg, child_conns[i]),
                daemon=True,
            )
            for i in range(self.cfg.n_processes)
        ]
        for w in self._workers:
            w.start()

    def shutdown(self) -> None:
        """Signal workers to stop and join them cleanly."""
        for conn in self._parent_conns:
            conn.send(("stop", None))
        for w in self._workers:
            w.join()

    # ── Per-update methods ────────────────────────────────────────────────────

    def distribute(self, active_sd: Dict, frozen_sd: Dict) -> float:
        """
        Send the active and frozen-opponent policy weights to every worker.

        Parameters
        ──────────
        active_sd : dict — CPU-side state dict of the trained policy
        frozen_sd : dict — CPU-side state dict of the frozen opponent

        Returns
        ───────
        t_dist : float — seconds spent serialising + sending
        """
        t0 = time.time()
        payload = {"active": active_sd, "frozen": frozen_sd}
        for conn in self._parent_conns:
            conn.send(("collect", payload))
        return time.time() - t0

    def collect(self) -> Tuple[dict, float]:
        """
        Block until every worker returns its rollout chunk, then assemble
        the full raw batch by concatenating along the environment axis.

        Returns
        ───────
        raw_batch : dict
            obs_snaps  list[T][N_total]        snapshot dicts
            actions    list[T][N_total]        action lists
            masks      list[T][N_total]        mask lists
            log_probs  np.ndarray (T, N_total) float32
            values     np.ndarray (T, N_total) float32
            rewards    np.ndarray (T, N_total) float32
            dones      np.ndarray (T, N_total) float32
            is_active  np.ndarray (T, N_total) float32
            last_values np.ndarray (N_total,)  float32
            player_ids np.ndarray (T, N_total) int32
            n_games / n_active_wins / n_conquest / n_timeout : int (summed)

        t_collect : float — seconds spent waiting for workers
        """
        t0     = time.time()
        chunks = [conn.recv()[1] for conn in self._parent_conns]
        t_collect = time.time() - t0

        T = self.cfg.n_steps
        raw_batch = {
            # List-of-lists: concatenate along env axis for each timestep
            "obs_snaps": [
                sum([c["obs_snaps"][t] for c in chunks], []) for t in range(T)
            ],
            "actions": [
                sum([c["actions"][t] for c in chunks], []) for t in range(T)
            ],
            "masks": [
                sum([c["masks"][t] for c in chunks], []) for t in range(T)
            ],
            # Numeric arrays: concatenate along env axis (axis=1)
            "log_probs":   np.concatenate([c["log_probs"]   for c in chunks], axis=1),
            "values":      np.concatenate([c["values"]      for c in chunks], axis=1),
            "rewards":     np.concatenate([c["rewards"]     for c in chunks], axis=1),
            "dones":       np.concatenate([c["dones"]       for c in chunks], axis=1),
            "is_active":   np.concatenate([c["is_active"]   for c in chunks], axis=1),
            # Bootstrap values: concatenate along env axis (axis=0, shape (N,))
            "last_values": np.concatenate([c["last_values"] for c in chunks]),
            "player_ids":  np.concatenate([c["player_ids"]  for c in chunks], axis=1),
            # Logging scalars: sum across worker chunks
            "n_games":       sum(c["n_games"]       for c in chunks),
            "n_active_wins": sum(c["n_active_wins"] for c in chunks),
            "n_conquest":    sum(c["n_conquest"]    for c in chunks),
            "n_timeout":     sum(c["n_timeout"]     for c in chunks),
        }
        return raw_batch, t_collect



        