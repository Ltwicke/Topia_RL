"""
train_PPO_hierarchical.py
────────────────────────────────────────────────────────────────────────────────
PPO training script for the hierarchical Polytopia policy network.

Differences from 3train_PPO.py
────────────────────────────────
  • Imports PolicyNetwork, PolicyConfig, and make_snapshot from policy_network.
  • PPOConfig embeds PolicyConfig so all model hyperparameters live in one place.
  • Worker bootstrap uses policy.encoder.encode() + policy.critic() instead of
    the old private policy._encode() / policy.critic_head().
  • _snapshot() is replaced by make_snapshot() from policy_network, which
    additionally stores unit_mvpts and unit_attack_ranges needed by
    MovementTargetHead._context_radius() during evaluate_actions.
  • model_summary() imported from policy_network.
  • Everything else (GAE, PPO update, checkpoint logic, worker protocol) is
    unchanged from the original.
"""

import sys, os, time, random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm

sys.path.insert(0, r"C:\Users\laure\1own_projects\1polytopia_score")

from env.wrapper import EnvWrapper
from game.enums import BoardType, Tribes
from RL.models.policy import (
    PolicyNetwork,
    make_snapshot,
    model_summary,
)


# ══════════════════════════════════════════════════════════════════════════════
# Value-return normaliser
# ══════════════════════════════════════════════════════════════════════════════

class RunningMeanStd:
    """
    Welford online algorithm for tracking the running mean and variance of
    scalar return values.  Used to normalise targets for the value loss so
    that the critic always operates on a roughly unit-scale signal regardless
    of reward magnitude or discount horizon.
    """
    def __init__(self, epsilon: float = 1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x           = np.asarray(x, dtype=np.float64).ravel()
        batch_mean  = x.mean()
        batch_var   = x.var()
        batch_count = x.size
        delta       = batch_mean - self.mean
        tot_count   = self.count + batch_count
        self.mean   = self.mean + delta * batch_count / tot_count
        M2          = (self.var    * self.count
                       + batch_var * batch_count
                       + delta**2  * self.count * batch_count / tot_count)
        self.var    = M2 / tot_count
        self.count  = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint bookkeeping
# ══════════════════════════════════════════════════════════════════════════════

MAX_CKPT_KEEP = 3
CKPT_DIR      = "checkpoints_hierarchical"


def _save_checkpoint(policy: PolicyNetwork, update: int,
                     ckpt_queue: deque) -> None:
    """
    Save state_dict only and evict the oldest checkpoint when the rolling
    window is full.
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"policy_update_{update:05d}.pt")
    torch.save(policy.state_dict(), path)
    ckpt_queue.append(path)
    if len(ckpt_queue) > MAX_CKPT_KEEP:
        oldest = ckpt_queue.popleft()
        if os.path.exists(oldest):
            os.remove(oldest)
            print(f"  [ckpt] removed old checkpoint: {oldest}")
    print(f"  [ckpt] saved → {path}  (keeping last {MAX_CKPT_KEEP})")


# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class PolicyConfig:
    """
    Full hyperparameter config for PolicyNetwork and its training environment.

    Encoder
    ───────
    encoder_hidden_dim : hidden width of GraphTransformerEncoder; must be
                         divisible by encoder_n_heads AND by 4 (RoPE requirement
                         for all selection heads that share this dimension).
    encoder_n_heads    : attention heads per TransformerConv layer.
    encoder_depth      : number of TransformerConv layers  (receptive field knob).

    Selection heads  (SequenceSelectionHead — shared by all four action lines)
    ───────────────
    sel_n_heads  : MHSA heads per transformer block.
    sel_n_layers : number of transformer blocks per head  (depth knob).

    MLP  (used by every _mlp() call across all heads)
    ───
    mlp_hidden_dim : hidden width.
    mlp_depth      : number of hidden layers (minimum 1).

    Multi-scale convolutions  (MovementTargetHead + CreateUnitTypeHead)
    ────────────────────────
    kernel_sizes  : spatial pooling scales, all odd ints, largest first.
    n_conv_layers : stacked Conv2d layers per kernel size
                    (expands receptive field without growing output width).

    Movement context window
    ───────────────────────
    context_bias : added to per-unit radius = int(mvpts*2 + attack_range) + bias.
                   Must be >= max(kernel_sizes) // 2.
    """

    # ── Encoder ────────────────────────────────────────────────────────────
    encoder_hidden_dim: int   = 64
    encoder_n_heads:    int   = 4
    encoder_depth:      int   = 4

    # ── Selection heads ────────────────────────────────────────────────────
    sel_n_heads:  int = 4
    sel_n_layers: int = 2

    # ── MLP ────────────────────────────────────────────────────────────────
    mlp_hidden_dim: int = 128
    mlp_depth:      int = 3

    # ── Multi-scale convolutions ───────────────────────────────────────────
    kernel_sizes:  Tuple[int, ...] = (5, 3)
    n_conv_layers: int             = 1

    # ── Movement context window ────────────────────────────────────────────
    context_bias: int = 4

    # ── Derived / validation ───────────────────────────────────────────────
    def __post_init__(self) -> None:
        assert self.encoder_hidden_dim % self.encoder_n_heads == 0, (
            f"encoder_hidden_dim ({self.encoder_hidden_dim}) must be divisible "
            f"by encoder_n_heads ({self.encoder_n_heads})."
        )
        assert self.encoder_hidden_dim % 4 == 0, (
            f"encoder_hidden_dim ({self.encoder_hidden_dim}) must be divisible "
            f"by 4 for 2D RoPE in selection heads."
        )
        assert all(k % 2 == 1 for k in self.kernel_sizes), \
            "All kernel_sizes must be odd integers."
        assert self.context_bias >= max(self.kernel_sizes) // 2, (
            f"context_bias ({self.context_bias}) must be >= "
            f"max(kernel_sizes)//2 ({max(self.kernel_sizes)//2})."
        )
        assert self.mlp_depth >= 1, "mlp_depth must be >= 1."
        assert self.n_conv_layers >= 1, "n_conv_layers must be >= 1."


@dataclass
class PPOConfig(PolicyConfig):
    """
    Full training configuration.

    Inherits all model architecture fields from PolicyConfig so that a single
    config object is passed to both PolicyNetwork (for construction) and the
    training loop (for PPO hyperparameters).

    PolicyConfig fields (inherited)
    ───────────────────────────────
    encoder_hidden_dim, encoder_n_heads, encoder_depth,
    sel_n_heads, sel_n_layers,
    mlp_hidden_dim, mlp_depth,
    kernel_sizes, n_conv_layers, context_bias

    PPO-specific fields (defined here)
    ───────────────────────────────────
    See inline comments below.
    """

    # ── Parallelism ────────────────────────────────────────────────────────
    n_processes:        int   = 16
    n_envs_per_process: int   = 4

    # ── Environment ────────────────────────────────────────────────────────
    board_config_dict: dict = field(default_factory=lambda: {
        "board_type" : BoardType.Dummy,
        "n_players"  : 2,
    })
    player_tribes:      list  = field(
        default_factory=lambda: [Tribes.Omaji, Tribes.Imperius]
    )
    max_turns_per_game: int   = 100
    board_size_range:   tuple = (10, 16)

    # ── Rollout ────────────────────────────────────────────────────────────
    n_steps: int = 256

    # ── PPO epochs & batching ──────────────────────────────────────────────
    n_epochs:      int = 2
    n_minibatches: int = 128

    # ── PPO loss coefficients ──────────────────────────────────────────────
    clip_eps:      float = 0.2
    vf_coef:       float = 0.5
    ent_coef:      float = 0.01
    max_grad_norm: float = 0.5

    # ── GAE / discount ─────────────────────────────────────────────────────
    gamma:      float = 0.99
    gae_lambda: float = 0.95

    # ── Optimiser ─────────────────────────────────────────────────────────
    lr: float = 3e-4

    # ── Training ──────────────────────────────────────────────────────────
    n_updates:    int = 20
    log_interval: int = 1

    # ── Derived properties ─────────────────────────────────────────────────
    @property
    def n_envs_total(self) -> int:
        return self.n_processes * self.n_envs_per_process

    @property
    def batch_size(self) -> int:
        return self.n_steps * self.n_envs_total

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.n_minibatches


# ══════════════════════════════════════════════════════════════════════════════
# Environment helpers
# ══════════════════════════════════════════════════════════════════════════════

def _random_board_size(cfg: PPOConfig) -> tuple:
    """Sample a random square board size within the configured range."""
    n = random.randint(cfg.board_size_range[0], cfg.board_size_range[1])
    return (n, n)


def _make_env(cfg: PPOConfig) -> EnvWrapper:
    """Create a new EnvWrapper with a freshly sampled random board size."""
    board_size   = _random_board_size(cfg)
    board_config = {
        "board_size" : list(board_size),
        "board_type" : cfg.board_config_dict["board_type"],
        "n_players"  : cfg.board_config_dict["n_players"],
    }
    return EnvWrapper(
        board_config, cfg.player_tribes,
        max_turns_per_game = cfg.max_turns_per_game,
        dense_reward       = True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Worker process — CPU only
# ══════════════════════════════════════════════════════════════════════════════

def worker_fn(worker_id: int, cfg: PPOConfig, conn) -> None:
    """
    Collect T-step rollouts across M environments on CPU.

    Each environment is recreated with a new random board size whenever an
    episode ends, providing natural curriculum diversity.

    Protocol
    ────────
    Receives : ('collect', state_dict)
    Sends    : ('data', chunk_dict)

    Receives : ('stop', None)
    Exits.

    Chunk arrays — shapes
    ─────────────────────
    log_probs   : (T, M)  float32
    values      : (T, M)  float32
    rewards     : (T, M)  float32
    dones       : (T, M)  float32
    won_flags   : (T, M)  float32
    last_values : (M,)    float32
    player_ids  : (T, M)  int32
    obs_snaps   : list T × list M  — dicts from make_snapshot()
    actions     : list T × list M  — action lists
    masks       : list T × list M  — mask lists
    """
    envs    = [_make_env(cfg) for _ in range(cfg.n_envs_per_process)]
    obs_buf = [env.reset()    for env in envs]

    # CPU-only policy for rollout
    policy  = PolicyNetwork(cfg)
    policy.eval()

    M = cfg.n_envs_per_process
    T = cfg.n_steps

    while True:
        cmd, payload = conn.recv()
        if cmd == 'stop':
            break

        # ── Load latest weights from the main process ──────────────────────
        policy.load_state_dict(payload)

        obs_snaps = [[None] * M for _ in range(T)]
        actions   = [[None] * M for _ in range(T)]
        masks_buf = [[None] * M for _ in range(T)]
        log_probs = np.zeros((T, M), dtype=np.float32)
        values    = np.zeros((T, M), dtype=np.float32)
        rewards   = np.zeros((T, M), dtype=np.float32)
        dones     = np.zeros((T, M), dtype=np.float32)
        won_flags = np.zeros((T, M), dtype=np.float32)

        t0 = time.time()
        with torch.no_grad():

            # ── T-step rollout ─────────────────────────────────────────────
            for t in range(T):
                for e, env in enumerate(envs):
                    obs  = obs_buf[e]
                    mask = env.get_action_mask()

                    # Snapshot BEFORE step — captures decision-time state
                    snap = make_snapshot(
                        obs, env.Nx, env.Ny,
                        player_id = env.game.player_go_id,
                    )

                    action,_,_, lp, _, val = policy(obs, mask)
                    next_obs, rew, done, _info = env.step(action)

                    obs_snaps[t][e] = snap
                    actions[t][e]   = action
                    masks_buf[t][e] = mask
                    log_probs[t, e] = lp.item()
                    values[t, e]    = val.item()
                    rewards[t, e]   = rew
                    dones[t, e]     = float(done)

                    if done:
                        won_flags[t, e] = float(env.winner is not None)
                        # Recreate env with a new random board size
                        envs[e]    = _make_env(cfg)
                        obs_buf[e] = envs[e].reset()
                    else:
                        obs_buf[e] = next_obs

            # ── Bootstrap: V(s_T) for each active environment ──────────────
            # Uses the encoder + critic directly (no action heads needed).
            last_values = np.zeros(M, dtype=np.float32)
            for e in range(M):
                snap_last     = make_snapshot(
                    obs_buf[e], envs[e].Nx, envs[e].Ny,
                    player_id = envs[e].game.player_go_id,
                )
                node_emb, global_emb = policy.encoder.encode(
                    snap_last['graph'],
                    snap_last['Nx'],
                    snap_last['Ny'],
                )
                last_values[e] = policy.critic(global_emb).item()

        player_ids = np.array(
            [[obs_snaps[t][e]['player_id'] for e in range(M)] for t in range(T)],
            dtype=np.int32,
        )

        elapsed = time.time() - t0
        print(f"  [worker {worker_id:02d}] rollout done — {elapsed:.2f}s", flush=True)

        conn.send(('data', {
            'obs_snaps':   obs_snaps,
            'actions':     actions,
            'masks':       masks_buf,
            'log_probs':   log_probs,
            'values':      values,
            'rewards':     rewards,
            'dones':       dones,
            'won_flags':   won_flags,
            'last_values': last_values,
            'player_ids':  player_ids,
        }))


# ══════════════════════════════════════════════════════════════════════════════
# GAE
# ══════════════════════════════════════════════════════════════════════════════

def compute_gae(
    rewards:     np.ndarray,   # (T, N)
    values:      np.ndarray,   # (T, N)
    dones:       np.ndarray,   # (T, N)
    last_values: np.ndarray,   # (N,)
    gamma:       float,
    gae_lam:     float,
    player_ids:  np.ndarray,   # (T, N)  int32
) -> tuple:
    """
    Generalised Advantage Estimation with player-switch masking.

    When the active player switches between consecutive steps the bootstrap
    signal is zeroed out, preventing value estimates from one player
    contaminating the other player's advantage.

    Returns
    ───────
    advantages : (T, N)  float32
    returns    : (T, N)  float32  (advantages + values)
    """
    T, N       = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    gae        = np.zeros(N,      dtype=np.float32)

    for t in reversed(range(T)):
        next_val = last_values if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        if t < T - 1:
            player_switched = (player_ids[t] != player_ids[t + 1]).astype(np.float32)
            not_done        = not_done * (1.0 - player_switched)

        delta         = rewards[t] + gamma * next_val * not_done - values[t]
        gae           = delta + gamma * gae_lam * not_done * gae
        advantages[t] = gae

    return advantages, advantages + values


# ══════════════════════════════════════════════════════════════════════════════
# PPO update — GPU
# ══════════════════════════════════════════════════════════════════════════════

def ppo_update(
    policy:         PolicyNetwork,
    optimizer:      torch.optim.Optimizer,
    batch:          dict,
    cfg:            PPOConfig,
    device:         torch.device,
    ret_normalizer: RunningMeanStd,
) -> tuple:
    """
    PPO update with two additions:
      • Value-function normalisation: returns are normalised with a running
        mean/std before computing the MSE value loss.
      • Per-epoch return recomputation: at the start of every epoch, fresh
        value estimates are computed with torch.no_grad() and GAE is re-run.
        This reduces variance and improves sample efficiency.

    Returns
    ───────
    (mean_policy_loss, mean_value_loss, mean_entropy)  all float
    """
    B = cfg.batch_size

    flat_snaps = [s for step in batch['obs_snaps'] for s in step]
    flat_acts  = [a for step in batch['actions']   for a in step]
    flat_masks = [m for step in batch['masks']      for m in step]

    # Old log-probs from rollout — fixed throughout the PPO update
    log_old = torch.tensor(
        batch['log_probs'].reshape(-1), dtype=torch.float32
    ).to(device)   # (B,)

    # Seed adv/ret from rollout GAE; overwritten at the start of each epoch
    adv_np = batch['advantages'].reshape(-1).astype(np.float32)
    ret_np = batch['returns'].reshape(-1).astype(np.float32)

    # Update running return stats once per ppo_update call
    ret_normalizer.update(ret_np)

    indices = np.arange(B)
    pl_log, vl_log, el_log = [], [], []

    total_steps = cfg.n_epochs * cfg.n_minibatches
    pbar = tqdm(total=total_steps,
                desc="  PPO epochs × minibatches",
                leave=False, unit="mb")

    for epoch in range(cfg.n_epochs):

        # ── Recompute returns at the start of every epoch ──────────────────
        # Fresh critic values → re-run GAE → re-normalise advantages.
        with torch.no_grad():
            fresh_chunks = []
            for start in range(0, B, cfg.minibatch_size):
                mb_snaps = flat_snaps[start : start + cfg.minibatch_size]
                fresh_chunks.append(
                    policy.compute_values_batch(mb_snaps).cpu().numpy()
                )
            fresh_vals_flat = np.concatenate(fresh_chunks)   # (B,)

        # Reshape to (T, N_total) that compute_gae expects
        N_total       = cfg.n_envs_total
        fresh_vals_2d = fresh_vals_flat.reshape(cfg.n_steps, N_total)

        adv_2d, ret_2d = compute_gae(
            batch['rewards'],    fresh_vals_2d, batch['dones'],
            batch['last_values'], cfg.gamma,    cfg.gae_lambda,
            batch['player_ids'],
        )
        adv_np = adv_2d.reshape(-1).astype(np.float32)
        ret_np = ret_2d.reshape(-1).astype(np.float32)

        adv = torch.tensor(adv_np, dtype=torch.float32).to(device)  # (B,)
        ret = torch.tensor(ret_np, dtype=torch.float32).to(device)  # (B,)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        ret_norm = torch.tensor(
            ret_normalizer.normalize(ret_np), dtype=torch.float32
        ).to(device)   # (B,)

        np.random.shuffle(indices)

        for start in range(0, B, cfg.minibatch_size):
            mb = indices[start : start + cfg.minibatch_size]

            # Re-score stored transitions under the current network weights.
            # Returns:
            #   new_lp  : (mb,)  log P(action) — differentiable
            #   new_ent : (mb,)  H[joint dist]  — differentiable
            #   new_val : (mb,)  V(s)           — differentiable
            new_lp, new_ent, new_val = policy.evaluate_actions(
                [flat_snaps[i] for i in mb],
                [flat_acts[i]  for i in mb],
                [flat_masks[i] for i in mb],
            )

            ratio   = torch.exp(new_lp - log_old[mb])   # (mb,)
            mb_adv  = adv[mb]                            # (mb,)

            loss_clip = torch.min(
                ratio * mb_adv,
                ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
            ).mean()

            loss_val = F.mse_loss(new_val.squeeze(), ret_norm[mb])
            loss_ent = new_ent.mean()

            loss = -loss_clip + cfg.vf_coef * loss_val - cfg.ent_coef * loss_ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            pl_log.append(loss_clip.item())
            vl_log.append(loss_val.item())
            el_log.append(loss_ent.item())

            pbar.set_postfix(
                epoch   = epoch + 1,
                p_loss  = f"{loss_clip.item():.4f}",
                v_loss  = f"{loss_val.item():.4f}",
            )
            pbar.update(1)

    pbar.close()
    return np.mean(pl_log), np.mean(vl_log), np.mean(el_log)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg    = PPOConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build policy on GPU (or CPU if no GPU available)
    policy = PolicyNetwork(cfg).to(device)

    # Optional: resume from checkpoint
    pre_load = torch.load('./checkpoints_hierarchical/policy_update_00002.pt',
                           weights_only=True, map_location=device)
    policy.load_state_dict(pre_load)

    policy.train()
    optimizer      = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    ret_normalizer = RunningMeanStd()

    # ── Startup banner ─────────────────────────────────────────────────────────
    lo, hi = cfg.board_size_range
    print("\n" + "=" * 65)
    print("  POLYTOPIA RL — PPO TRAINING  (hierarchical policy)")
    print("=" * 65)
    print(f"  Device            : {device}")
    print(f"  Board size        : random square [{lo}×{lo} … {hi}×{hi}]")
    print(f"  Batch size        : {cfg.batch_size:,}  "
          f"({cfg.n_processes} proc × {cfg.n_envs_per_process} envs "
          f"× {cfg.n_steps} steps)")
    print(f"  Minibatch size    : {cfg.minibatch_size:,}  |  "
          f"Epochs: {cfg.n_epochs}  |  Updates: {cfg.n_updates}")
    print(f"  Encoder           : {cfg.encoder_depth} layers "
          f"× {cfg.encoder_hidden_dim}d  ({cfg.encoder_n_heads} heads)")
    print(f"  Selection heads   : {cfg.sel_n_layers} layers "
          f"× {cfg.encoder_hidden_dim}d  ({cfg.sel_n_heads} heads)")
    print(f"  MLP               : {cfg.mlp_depth} hidden layers "
          f"× {cfg.mlp_hidden_dim}d")
    print(f"  Conv kernels      : {cfg.kernel_sizes}  "
          f"× {cfg.n_conv_layers} stacked layers each")
    print(f"  Context bias      : {cfg.context_bias}")
    print()
    model_summary(policy)
    print()

    # ── Spawn workers ──────────────────────────────────────────────────────────
    parent_conns, child_conns = zip(*[mp.Pipe() for _ in range(cfg.n_processes)])
    workers = [
        mp.Process(
            target = worker_fn,
            args   = (i, cfg, child_conns[i]),
            daemon = True,
        )
        for i in range(cfg.n_processes)
    ]
    for w in workers:
        w.start()

    ckpt_queue = deque()

    # ── Outer update loop ──────────────────────────────────────────────────────
    outer_bar = tqdm(range(3, cfg.n_updates), desc="Updates", unit="upd")

    for update in outer_bar:
        t_update_start = time.time()

        # ── 1. Distribute latest weights to workers ────────────────────────
        t0         = time.time()
        state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
        for conn in parent_conns:
            conn.send(('collect', state_dict))
        t_dist = time.time() - t0
        print(f"\n[update {update:04d}] weights dispatched in {t_dist:.3f}s")

        # ── 2. Collect rollouts from all workers ───────────────────────────
        t0 = time.time()
        print(f"[update {update:04d}] waiting for {cfg.n_processes} workers …")
        chunks = []
        for i, conn in enumerate(tqdm(parent_conns,
                                       desc="  collecting workers",
                                       leave=False, unit="worker")):
            chunks.append(conn.recv()[1])
        t_collect = time.time() - t0
        print(f"[update {update:04d}] all workers done in {t_collect:.2f}s")

        # ── 3. Assemble full batch ─────────────────────────────────────────
        t0    = time.time()
        batch = {
            # Lists of lists — assembled time-step by time-step across workers
            'obs_snaps': [sum([c['obs_snaps'][t] for c in chunks], [])
                          for t in range(cfg.n_steps)],
            'actions':   [sum([c['actions'][t]   for c in chunks], [])
                          for t in range(cfg.n_steps)],
            'masks':     [sum([c['masks'][t]      for c in chunks], [])
                          for t in range(cfg.n_steps)],
            # (T, N_total) float32 arrays concatenated along the env dimension
            'log_probs':   np.concatenate([c['log_probs']   for c in chunks], axis=1),
            'values':      np.concatenate([c['values']      for c in chunks], axis=1),
            'rewards':     np.concatenate([c['rewards']     for c in chunks], axis=1),
            'dones':       np.concatenate([c['dones']       for c in chunks], axis=1),
            'won_flags':   np.concatenate([c['won_flags']   for c in chunks], axis=1),
            # (N_total,) float32 — bootstrap values at end of rollout
            'last_values': np.concatenate([c['last_values'] for c in chunks]),
            # (T, N_total) int32 — player index at each step
            'player_ids':  np.concatenate([c['player_ids']  for c in chunks], axis=1),
        }

        # ── 4. GAE ────────────────────────────────────────────────────────
        adv, ret = compute_gae(
            batch['rewards'], batch['values'], batch['dones'],
            batch['last_values'], cfg.gamma, cfg.gae_lambda,
            batch['player_ids'],
        )
        batch['advantages'] = adv   # (T, N_total)
        batch['returns']    = ret   # (T, N_total)
        t_gae = time.time() - t0
        print(f"[update {update:04d}] GAE computed in {t_gae:.3f}s")

        # ── 5. PPO update (GPU) ───────────────────────────────────────────
        t0 = time.time()
        pl, vl, el = ppo_update(policy, optimizer, batch, cfg, device, ret_normalizer)
        t_ppo = time.time() - t0
        print(f"[update {update:04d}] PPO update done in {t_ppo:.2f}s")

        # ── 6. Checkpoint ─────────────────────────────────────────────────
        _save_checkpoint(policy, update, ckpt_queue)

        # ── 7. Stats & logging ────────────────────────────────────────────
        n_finished   = int(batch['dones'].sum())
        n_won        = int(batch['won_flags'].sum())
        n_timeout    = n_finished - n_won
        total_reward = batch['rewards'].sum()
        avg_r        = total_reward / max(n_finished, 1)
        win_rate     = n_won / max(n_finished, 1)
        avg_ep_len   = cfg.batch_size / max(n_finished, 1)
        t_total      = time.time() - t_update_start

        if update % cfg.log_interval == 0:
            print()
            print(f"╔══ update {update:04d} ═══════════════════════════════════════════")
            print(f"║  Wall time     : {t_total:.1f}s  "
                  f"(dist {t_dist:.2f}s | collect {t_collect:.2f}s "
                  f"| GAE {t_gae:.3f}s | PPO {t_ppo:.2f}s)")
            print(f"║  Games finished: {n_finished:6d}  "
                  f"(conquest {n_won}, timeout {n_timeout})")
            print(f"║  Win rate      : {win_rate:.3f}")
            print(f"║  Avg ep length : {avg_ep_len:.1f} steps")
            print(f"║  Avg reward/ep : {avg_r:.3f}")
            print(f"║  p_loss        : {pl:.4f}")
            print(f"║  v_loss        : {vl:.4f}")
            print(f"║  entropy       : {el:.4f}")
            print(f"╚{'═' * 56}")
            print()

        outer_bar.set_postfix(
            fin = n_finished,
            win = f"{win_rate:.2f}",
            p   = f"{pl:.3f}",
            v   = f"{vl:.3f}",
            ent = f"{el:.3f}",
        )

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(CKPT_DIR, "policy_FINAL.pt")
    torch.save(policy.state_dict(), final_path)
    print(f"\nFinal policy saved → {final_path}")

    # ── Shutdown workers ──────────────────────────────────────────────────────
    for conn in parent_conns:
        conn.send(('stop', None))
    for w in workers:
        w.join()
    print("All workers shut down.  Training complete.")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()