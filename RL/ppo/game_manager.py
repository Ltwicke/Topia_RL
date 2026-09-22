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
from game.enums  import ActionTypes, BoardType, Tribes
from RL.models.policy import PolicyNetwork, make_snapshot
from RL.models.main_modules import V_TERM, V_DENSE, N_VALUE_STREAMS


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
    pretrained_ckpt: str = r""   # path to .pt; "" = train from scratch
    start_update:    int = 0    # first update index (set > 0 when resuming)

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
    n_processes:        int = 4 #16
    n_envs_per_process: int = 1 #2

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
    # Two independent reward streams, each with its own critic head:
    #   terminal — the true objective (win by conquest, or by score at the limit)
    #   dense    — human-derived event shaping, a bootstrap toward sensible play
    # `dense_beta` weights dense in the POLICY advantage only. Set it to 0.0 to
    # switch shaping off mid-training: V_TERM is regressed solely on terminal
    # returns, so it stays valid and no value relearning is needed. Resume from
    # the checkpoint after flipping it.
    dense_beta:           float = 1.0
    dense_reward:         bool  = True
    # Scales the raw event table so an actively played episode totals ~1-2,
    # comparable to a timeout payout and well below the conquest bonus.
    dense_scale:          float = 0.04
    # Penalty for passing while other action types were legal. Forced
    # end-of-turns are never penalised. 0.0 = off; the dial for pass-pressure.
    endturn_voluntary_penalty: float = 1.0
    # terminal_reward_mode: "none" | "constant_sum"
    terminal_reward_mode: str   = "constant_sum"
    # Each seat is paid sigma(its own margin / terminal_tau) * terminal_weight,
    # so the two shares sum to terminal_weight and the loser tends to 0.
    # tau must be large enough that a NARROW win does not already saturate: at
    # tau=1000 a margin of 1500 paid 0.905 of maximum, which is why the agent
    # settled for scraping a turn-30 lead instead of pressing for a conquest.
    # At tau=2000 that same margin pays 0.679 and improving it keeps paying.
    # Still inferred rather than measured — eval/calibrate_terminal_tau.py.
    terminal_weight:      float = 1.0
    terminal_tau:         float = 1342.0
    # A conquest ends the game outright; a score lead only indicates a likely
    # win. Paid flat, so the terminal stream has the fixed, closed support
    # [0, conquest_reward] that the categorical value head bins over (sc-48).
    # Changing this REQUIRES re-deriving the head's bin support and invalidates
    # existing checkpoints; CriticHead asserts the two agree at construction.
    conquest_reward:      float = 2.0

    # ── Frozen-opponent self-play ─────────────────────────────────────────────
    # The trained policy occupies seat `active_player_id`; the other seat is
    # played by a frozen copy refreshed every `opponent_refresh_interval`
    # updates. Only the active seat's transitions train PPO.
    self_play_frozen_opponent: bool = True
    active_player_id:          int  = 0
    opponent_refresh_interval: int  = 1

    # ── Estimator pretraining (Phase A of each update) ────────────────────────
    estimator_lr:             float = 3e-4
    estimator_n_epochs:       int   = 3
    estimator_minibatch_size: int   = 32 #512
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
            "Get_defender_and_wall",
        ]
    )

    # ── Rollout ───────────────────────────────────────────────────────────────
    n_steps: int = 128 # 512

    # ── PPO epochs & batching ─────────────────────────────────────────────────
    n_epochs:       int   = 4 #2
    n_minibatches:  int   = 4 #64  # determines cfg.minibatch_size
    # Fraction ∈ (0,1]: what share of the assembled minibatches to train on
    # per epoch.  Reduces PPO update time without wasting simulation data.
    train_fraction: float = 1.0

    # ── PPO loss coefficients ─────────────────────────────────────────────────
    clip_eps:      float = 0.2
    vf_coef:       float = 0.5
    ent_coef:      float = 0.005
    max_grad_norm: float = 0.5

    # ── GAE / discount ────────────────────────────────────────────────────────
    # gamma=1.0 (undiscounted) is required by the terminal-only reward: γ is
    # applied per player-decision while the turn counter advances per round, so
    # any γ<1 pays a player for reaching the turn limit in fewer decisions —
    # i.e. for spamming EndTurn once ahead on score. Episodes are hard-bounded
    # by max_turns_per_game, so the undiscounted return is well defined.
    gamma:                    float = 1.0
    gae_lambda:               float = 0.95   # single λ for all trajectories
    recompute_gae_each_epoch: bool  = True   # refresh values + GAE before each
                                              # PPO epoch (uses compute_values_batch)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr: float = 3e-4

    # ── Training loop ─────────────────────────────────────────────────────────
    n_updates:     int = 1000
    log_interval:  int = 1
    ckpt_interval:           int = 1     # rolling checkpoints (last MAX_CKPT kept)
    permanent_ckpt_interval: int = 50   # never-evicted snapshots every N updates

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
        dense_reward=cfg.dense_reward,
        dense_scale=cfg.dense_scale,
        endturn_voluntary_penalty=cfg.endturn_voluntary_penalty,
        terminal_reward_mode=cfg.terminal_reward_mode,
        terminal_weight=cfg.terminal_weight,
        terminal_tau=cfg.terminal_tau,
        conquest_reward=cfg.conquest_reward,
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
    last_values (M, 2)  float32     — V(s_T) bootstrap per seat (active critic)
    player_ids  (T, M)  int32       — which player acted at each step
    obs_snaps   list[T] × list[M]   — make_snapshot() dicts (for evaluate_actions)
    actions     list[T] × list[M]   — stored action lists
    masks       list[T] × list[M]   — stored action masks
    n_games / n_active_wins / n_conquest / n_timeout : int — logging scalars
    n_dropped_terminal : int        — terminal shares that could not be delivered
                                      because the recipient had no decision in
                                      this chunk (expected 0 when T >> ep length)
    n_endturn / n_decisions_total : int — active-seat anti-rush diagnostic;
                                      decisions-per-turn = total / endturn
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
        # Per-stream values and rewards. Terminal and dense are never summed in
        # the buffers: the critic has one head per stream so that dense shaping
        # can be switched off (beta=0) without disturbing the terminal value
        # function. Index the last axis with V_TERM / V_DENSE.
        values     = np.zeros((T, M, N_VALUE_STREAMS), dtype=np.float32)
        rew_term   = np.zeros((T, M), dtype=np.float32)
        rew_dense  = np.zeros((T, M), dtype=np.float32)
        dones      = np.zeros((T, M), dtype=np.float32)
        is_active  = np.zeros((T, M), dtype=np.float32)
        # Steps where the policy had exactly one possible trajectory. They carry
        # no decision: log pi == 0 and entropy == 0 under the hard mask, so
        # training on them only dilutes the entropy bonus and inflates the
        # advantage-whitening std. Filtered out downstream.
        is_forced  = np.zeros((T, M), dtype=np.float32)
        # acting-player track filled during rollout so we can back-fill the
        # opponent's last decision step at game termination
        player_ids = np.full((T, M), -1, dtype=np.int32)

        # Index of each seat's most recent decision, per env, within THIS chunk.
        # -1 = that seat has not acted yet in this chunk, in which case its
        # terminal share cannot be delivered (its last decision belongs to an
        # already-shipped chunk) and is counted in n_dropped_terminal.
        last_idx = [[-1, -1] for _ in range(M)]

        # Per-chunk logging scalars
        n_games = n_active_wins = n_conquest = n_timeout = 0
        n_dropped_terminal = 0
        n_endturn = n_decisions_total = 0
        # EndTurn split into forced (no alternative existed) vs voluntary (the
        # agent passed while it could still have acted). Only the voluntary
        # count says anything about whether the policy is short-cutting.
        n_endturn_voluntary = n_forced = 0

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
                    action, joint_probs, _, lp, _, val = pol(obs, mask)

                    # A single enumerable trajectory means the agent had no
                    # choice at all — the exact definition of a forced step.
                    forced = int(joint_probs.numel()) <= 1

                    next_obs, rew, done, info = env.step(
                        action, n_valid_action_types=int(mask[0].sum()),
                    )

                    obs_snaps[t][e]   = snap
                    actions[t][e]     = action
                    masks_buf[t][e]   = mask
                    log_probs[t, e]   = lp.item()
                    values[t, e]      = val.detach().cpu().numpy()
                    rew_term[t, e]    = float(info["r_term"])
                    rew_dense[t, e]   = float(info["r_dense"])
                    dones[t, e]       = float(done)
                    is_active[t, e]   = 1.0 if cur_pid == active_pid else 0.0
                    is_forced[t, e]   = 1.0 if forced else 0.0
                    player_ids[t, e]  = cur_pid
                    last_idx[e][cur_pid] = t

                    # Anti-rush diagnostics, active seat only. decisions-per-turn
                    # is n_decisions_total / n_endturn (each EndTurn closes one
                    # of that seat's turns); human-level play should climb well
                    # above 2. n_endturn_voluntary isolates passes the agent
                    # actually chose, which is the number that reflects policy.
                    if cur_pid == active_pid:
                        n_decisions_total += 1
                        n_forced += int(forced)
                        if int(action[0]) == int(ActionTypes.EndTurn):
                            n_endturn += 1
                            if not forced:
                                n_endturn_voluntary += 1

                    if done:
                        winner_id = info.get("winner_id", None)
                        is_conq   = bool(info.get("is_conquest", False))
                        r_opp     = float(info.get("r_term_opp", 0.0))

                        # ── Deliver the OTHER player's terminal share onto its
                        # own last decision step and mark that step done=1, so
                        # per-player GAE terminates both sides' trajectories.
                        # Terminal stream only — dense is per-action and is
                        # never back-filled. The actor's own share already rode
                        # in on rew_term[t,e].
                        opp_id = (cur_pid + 1) % 2
                        j = last_idx[e][opp_id]
                        if j >= 0:
                            rew_term[j, e] += r_opp
                            dones[j, e]     = 1.0
                        else:
                            n_dropped_terminal += 1
                        last_idx[e] = [-1, -1]

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

            # ── Bootstrap V(s_T) per env slot AND per seat ───────────────────
            # obs_buf[e] now holds the first obs of a new episode (if done) or
            # the observation that follows the last collected step. That view
            # belongs to whoever acts next, so it is only a valid bootstrap for
            # that seat; the other seat's value is its negation, which is exact
            # because the terminal reward is antisymmetric (zero-sum).
            last_values = np.zeros((M, 2, N_VALUE_STREAMS), dtype=np.float32)
            for e in range(M):
                next_pid  = envs[e].game.player_go_id
                snap_last = make_snapshot(
                    obs_buf[e], envs[e].Nx, envs[e].Ny,
                    player_id=next_pid,
                )
                # scalar_state carries turn_norm — the most value-relevant
                # feature under a turn-limit terminal reward. Omitting it here
                # would make this bootstrap off-distribution versus every value
                # the critic is trained on.
                _, global_emb = policy_active.encoder.encode(
                    snap_last["graph"], snap_last["Nx"], snap_last["Ny"],
                    snap_last["scalar_state"],
                )
                v = policy_active.critic(global_emb).detach().cpu().numpy()
                other_pid = (next_pid + 1) % 2
                # The terminal stream is constant-sum, so the seat not moving
                # next is worth terminal_weight minus this seat's value — not
                # its negation, as it was under the old zero-sum reward.
                last_values[e, next_pid,  V_TERM] = v[V_TERM]
                last_values[e, other_pid, V_TERM] = cfg.terminal_weight - v[V_TERM]
                # Dense is an own-actions-only stream with no such symmetry, so
                # the opponent's dense value cannot be derived from this one.
                # Leaving it at 0 biases only the final in-rollout step of the
                # seat that is not about to move.
                last_values[e, next_pid,  V_DENSE] = v[V_DENSE]

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
            "rew_term":      rew_term,
            "rew_dense":     rew_dense,
            "dones":         dones,
            "is_active":     is_active,
            "is_forced":     is_forced,
            "last_values":   last_values,
            "player_ids":    player_ids,
            "n_games":       n_games,
            "n_active_wins": n_active_wins,
            "n_conquest":    n_conquest,
            "n_timeout":     n_timeout,
            "n_dropped_terminal":  n_dropped_terminal,
            "n_endturn":           n_endturn,
            "n_endturn_voluntary": n_endturn_voluntary,
            "n_forced":            n_forced,
            "n_decisions_total":   n_decisions_total,
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
            last_values np.ndarray (N_total, 2) float32  — per seat
            player_ids np.ndarray (T, N_total) int32
            n_games / n_active_wins / n_conquest / n_timeout : int (summed)
            n_dropped_terminal / n_endturn / n_decisions_total : int (summed)

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
            "rew_term":    np.concatenate([c["rew_term"]    for c in chunks], axis=1),
            "rew_dense":   np.concatenate([c["rew_dense"]   for c in chunks], axis=1),
            "dones":       np.concatenate([c["dones"]       for c in chunks], axis=1),
            "is_active":   np.concatenate([c["is_active"]   for c in chunks], axis=1),
            "is_forced":   np.concatenate([c["is_forced"]   for c in chunks], axis=1),
            # Bootstrap values: env axis is axis 0, shape (N, 2 seats, streams)
            "last_values": np.concatenate([c["last_values"] for c in chunks]),
            "player_ids":  np.concatenate([c["player_ids"]  for c in chunks], axis=1),
            # Logging scalars: sum across worker chunks
            "n_games":       sum(c["n_games"]       for c in chunks),
            "n_active_wins": sum(c["n_active_wins"] for c in chunks),
            "n_conquest":    sum(c["n_conquest"]    for c in chunks),
            "n_timeout":     sum(c["n_timeout"]     for c in chunks),
            "n_dropped_terminal":  sum(c["n_dropped_terminal"]  for c in chunks),
            "n_endturn":           sum(c["n_endturn"]           for c in chunks),
            "n_endturn_voluntary": sum(c["n_endturn_voluntary"] for c in chunks),
            "n_forced":            sum(c["n_forced"]            for c in chunks),
            "n_decisions_total":   sum(c["n_decisions_total"]   for c in chunks),
        }
        return raw_batch, t_collect



        