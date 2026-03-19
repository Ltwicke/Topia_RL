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

Win-reward mode (this file)
────────────────────────────
Dense rewards are disabled.  The worker explicitly overrides the reward
buffer with:
    +1.0            at the winner's terminal step
    cfg.loss_reward at the loser's last step within the same episode
    0.0             everywhere else

The main training loop (train.py) gates each PPO update on the presence of
at least one winner in the collected batch; if no winner is found the workers
are re-triggered without advancing the update counter.
"""

from __future__ import annotations

import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    Single source of truth for every hyperparameter in the win-reward PPO
    training pipeline.

    Key differences from the dense-reward config
    ─────────────────────────────────────────────
    • minibatch_size  — direct hyperparameter (replaces n_minibatches).
                        The number of minibatches per epoch is computed
                        dynamically as batch_size // minibatch_size.
    • loss_reward     — small negative reward assigned to the loser's last
                        step in any episode that ends by conquest.
    • board_size_range — narrowed to (9, 11) to increase win frequency.
    • Dense rewards   — disabled (dense_reward=False in _make_env).

    Architecture fields are forwarded to PolicyNetwork.__init__().
    All other fields are consumed by EnvManager, BatchProcessor, or PPOTrainer.
    """

    # ── Checkpoint / resume ───────────────────────────────────────────────────
    pretrained_ckpt: str = r".\checkpoints_win_reward\policy_update_00029.pt"
    start_update:    int = 30

    # ── Encoder ───────────────────────────────────────────────────────────────
    encoder_hidden_dim: int = 64
    encoder_n_heads:    int = 4
    encoder_depth:      int = 4

    # ── Selection heads ───────────────────────────────────────────────────────
    sel_n_heads:  int = 4
    sel_n_layers: int = 2

    # ── MLP ───────────────────────────────────────────────────────────────────
    mlp_hidden_dim: int = 128
    mlp_depth:      int = 3

    # ── Multi-scale convolutions ──────────────────────────────────────────────
    kernel_sizes:  Tuple[int, ...] = (5, 3)
    n_conv_layers: int             = 1

    # ── Movement context window ───────────────────────────────────────────────
    context_bias: int = 4

    # ── Parallelism ───────────────────────────────────────────────────────────
    n_processes:        int = 16
    n_envs_per_process: int = 8

    # ── Environment ───────────────────────────────────────────────────────────
    board_config_dict: dict = field(default_factory=lambda: {
        "board_type": BoardType.Dummy,
        "n_players":  2,
    })
    player_tribes:      list  = field(
        default_factory=lambda: [Tribes.Omaji, Tribes.Imperius]
    )
    max_turns_per_game: int   = 1000
    # Narrowed from (10, 16) to increase conquest frequency per rollout.
    board_size_range:   tuple = (11,11)

    # ── Rollout ───────────────────────────────────────────────────────────────
    n_steps: int = 256

    # ── PPO epochs & batching ─────────────────────────────────────────────────
    n_epochs: int = 4
    # Direct minibatch size.  The number of minibatches per epoch is
    # batch_size // minibatch_size (computed dynamically, drop_last=True).
    minibatch_size: int = 128

    # ── Win / loss rewards ────────────────────────────────────────────────────
    # Winner's terminal step receives +1.0.
    # Loser's last step in the same episode receives loss_reward (≤ 0).
    win_reward: float = 30.
    loss_reward: float = -10.0

    # ── PPO loss coefficients ─────────────────────────────────────────────────
    clip_eps:      float = 0.2
    vf_coef:       float = 0.5
    ent_coef:      float = 0.02   # slightly higher than dense run to slow entropy collapse
    max_grad_norm: float = 0.5

    # ── GAE / discount ────────────────────────────────────────────────────────
    # Higher gamma (0.998) ensures the win signal propagates back through
    # the many steps of a strategy game episode.
    gamma:      float = 0.998
    gae_lambda: float = 0.95

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Lower than dense-reward run to preserve learned tactics.
    lr: float = 4e-5

    # ── Training loop ─────────────────────────────────────────────────────────
    n_updates:     int = 500
    log_interval:  int = 1
    ckpt_interval: int = 1

    # ── Speed / diagnostic ────────────────────────────────────────────────────
    use_amp:    bool = True
    track_vram: bool = False

    # ── Derived properties ────────────────────────────────────────────────────
    @property
    def n_envs_total(self) -> int:
        return self.n_processes * self.n_envs_per_process

    @property
    def batch_size(self) -> int:
        """Total samples per rollout = T × N_total."""
        return self.n_steps * self.n_envs_total

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
        assert self.mlp_depth     >= 1, "mlp_depth must be >= 1."
        assert self.n_conv_layers >= 1, "n_conv_layers must be >= 1."
        assert self.minibatch_size >= 1, "minibatch_size must be >= 1."
        assert self.loss_reward   <= 0.0, "loss_reward must be <= 0."


# ══════════════════════════════════════════════════════════════════════════════
# Environment factory helpers
# ══════════════════════════════════════════════════════════════════════════════

def _random_board_size(cfg: TrainConfig) -> tuple:
    n = random.randint(cfg.board_size_range[0], cfg.board_size_range[1])
    return (n, n)


def _make_env(cfg: TrainConfig) -> EnvWrapper:
    board_size   = _random_board_size(cfg)
    board_config = {
        "board_size": list(board_size),
        "board_type": cfg.board_config_dict["board_type"],
        "n_players":  cfg.board_config_dict["n_players"],
    }
    return EnvWrapper(
        board_config,
        cfg.player_tribes,
        max_turns_per_game=cfg.max_turns_per_game,
        dense_reward=False,   # ← win-reward mode: suppress all intermediate rewards
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

    Protocol
    ────────
    Main → Worker : ('collect', cpu_state_dict)
    Worker → Main : ('data',    chunk_dict)

    Main → Worker : ('stop', None)
    Worker exits cleanly.

    Reward override (win-reward mode)
    ─────────────────────────────────
    The environment reward is ignored.  Instead:
      • rewards[t, e] = +1.0          when env e terminates at step t by
                                       conquest (acting player wins).
      • rewards[last_t, e] = loss_reward  at the loser's last step within
                                           the same episode (if they acted
                                           at least once).
      • rewards[t, e] = 0.0           for all other steps.

    A per-(env, player) tracker `_last_step` records the most recent global
    timestep at which each player acted within the current episode.  It is
    reset to [None, None] whenever an episode ends (conquest or timeout).

    Chunk arrays
    ────────────
    log_probs   (T, M)  float32     — log π_θ_old(a_t | s_t)
    values      (T, M)  float32     — V(s_t) at collection time
    rewards     (T, M)  float32     — win/loss rewards only
    dones       (T, M)  float32     — 1.0 only on game termination
    won_flags   (T, M)  float32     — 1.0 when terminated by conquest
    last_values (M,)    float32     — V(s_T) bootstrap
    player_ids  (T, M)  int32       — which player acted at each step
    obs_snaps   list[T] × list[M]   — make_snapshot() dicts
    actions     list[T] × list[M]   — stored action lists
    masks       list[T] × list[M]   — stored action masks
    """
    # Ensure project root is on path in the worker process
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    envs    = [_make_env(cfg) for _ in range(cfg.n_envs_per_process)]
    obs_buf = [env.reset()    for env in envs]

    policy = PolicyNetwork(cfg)
    policy.eval()

    M = cfg.n_envs_per_process
    T = cfg.n_steps

    while True:
        cmd, payload = conn.recv()
        if cmd == "stop":
            break

        # Load latest weights (CPU tensors from main process)
        policy.load_state_dict(payload)

        # Pre-allocate rollout buffers
        obs_snaps = [[None] * M for _ in range(T)]
        actions   = [[None] * M for _ in range(T)]
        masks_buf = [[None] * M for _ in range(T)]
        log_probs = np.zeros((T, M), dtype=np.float32)
        values    = np.zeros((T, M), dtype=np.float32)
        rewards   = np.zeros((T, M), dtype=np.float32)   # overridden below
        dones     = np.zeros((T, M), dtype=np.float32)
        won_flags = np.zeros((T, M), dtype=np.float32)

        # Per-(env, player) tracker: last global timestep within the current
        # episode where that player acted.  None = player has not yet acted
        # in the current episode.  Reset to [None, None] on episode end.
        _last_step: List[List[Optional[int]]] = [[None, None] for _ in range(M)]

        t0 = time.time()
        with torch.no_grad():
            for t in range(T):
                for e, env in enumerate(envs):
                    obs  = obs_buf[e]
                    mask = env.get_action_mask()

                    # Capture acting player BEFORE env.step() changes the state.
                    acting_player: int = env.game.player_go_id

                    # Snapshot must be taken BEFORE env.step() so that the
                    # stored graph / tile IDs match the decision state.
                    snap = make_snapshot(
                        obs, env.Nx, env.Ny,
                        player_id=acting_player,
                    )

                    # forward() returns:
                    # action, joint_probs, traj_actions, log_prob, entropy, value
                    action, _, _, lp, _, val = policy(obs, mask)
                    next_obs, _rew, done, _  = env.step(action)   # env reward ignored

                    obs_snaps[t][e] = snap
                    actions[t][e]   = action
                    masks_buf[t][e] = mask
                    log_probs[t, e] = lp.item()
                    values[t, e]    = val.item()
                    dones[t, e]     = float(done)
                    # rewards[t, e] stays 0.0 unless terminal case below

                    # Update last-step tracker for acting player
                    _last_step[e][acting_player] = t

                    if done:
                        has_winner = env.winner is not None
                        won_flags[t, e] = float(has_winner)

                        if has_winner:
                            # Winner: acting player who just conquered
                            rewards[t, e] = cfg.win_reward

                            # Loser: assign loss_reward at their last step
                            # (assumes exactly 2 players, ids 0 and 1)
                            loser_id: int = 1 - acting_player
                            loser_t: Optional[int] = _last_step[e][loser_id]
                            if loser_t is not None:
                                rewards[loser_t, e] = cfg.loss_reward

                        # Reset episode tracker; create new env for next episode
                        _last_step[e] = [None, None]
                        envs[e]       = _make_env(cfg)
                        obs_buf[e]    = envs[e].reset()
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
                _, global_emb  = policy.encoder.encode(
                    snap_last["graph"], snap_last["Nx"], snap_last["Ny"],
                )
                last_values[e] = policy.critic(global_emb).item()

        # Build player_ids array from stored snapshots
        player_ids = np.array(
            [[obs_snaps[t][e]["player_id"] for e in range(M)]
             for t in range(T)],
            dtype=np.int32,
        )

        print(
            f"  [worker {worker_id:02d}] rollout done — {time.time() - t0:.2f}s",
            flush=True,
        )

        conn.send(("data", {
            "obs_snaps":   obs_snaps,
            "actions":     actions,
            "masks":       masks_buf,
            "log_probs":   log_probs,
            "values":      values,
            "rewards":     rewards,
            "dones":       dones,
            "won_flags":   won_flags,
            "last_values": last_values,
            "player_ids":  player_ids,
        }))


# ══════════════════════════════════════════════════════════════════════════════
# EnvManager
# ══════════════════════════════════════════════════════════════════════════════

class EnvManager:
    """
    Manages the worker subprocess pool.

    Workflow per training update
    ────────────────────────────
    1. manager.distribute(cpu_state_dict)  — push weights to all workers
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

    def distribute(self, state_dict: Dict) -> float:
        """
        Send the current policy weights to every worker.

        Parameters
        ──────────
        state_dict : dict — CPU-side state dict (tensors must already be .cpu())

        Returns
        ───────
        t_dist : float — seconds spent serialising + sending
        """
        t0 = time.time()
        for conn in self._parent_conns:
            conn.send(("collect", state_dict))
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
            rewards    np.ndarray (T, N_total) float32  win/loss only
            dones      np.ndarray (T, N_total) float32
            won_flags  np.ndarray (T, N_total) float32
            last_values np.ndarray (N_total,)  float32
            player_ids np.ndarray (T, N_total) int32

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
            "won_flags":   np.concatenate([c["won_flags"]   for c in chunks], axis=1),
            # Bootstrap values: concatenate along env axis (axis=0, shape (N,))
            "last_values": np.concatenate([c["last_values"] for c in chunks]),
            "player_ids":  np.concatenate([c["player_ids"]  for c in chunks], axis=1),
        }
        return raw_batch, t_collect

        