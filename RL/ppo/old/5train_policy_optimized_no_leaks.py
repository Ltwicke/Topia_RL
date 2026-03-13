"""
train_policy_optimized.py
────────────────────────────────────────────────────────────────────────────────
Optimised PPO training script for the hierarchical Polytopia policy network.

Key changes vs new_train_policy.py
────────────────────────────────────
  [ARCHITECTURE]
  • PolicyConfig and PPOConfig merged into a single TrainConfig dataclass.
    One object is passed to both PolicyNetwork (for model construction) and
    the training loop (for PPO hyperparameters).
  • pretrained_ckpt / start_update live in TrainConfig so all knobs are in
    one place; checkpoint loading happens at the top of main(), before
    optimizer and scheduler construction.

  [MEMORY LEAK FIXES]
  • Explicit del + torch.cuda.empty_cache() + gc.collect() after every PPO
    update to ensure PyTorch's CUDA caching allocator releases blocks that
    Python GC may not reclaim promptly.
  • All large per-update tensors (log_old, adv, ret, ret_norm) are
    explicitly deleted at the end of ppo_update().
  • flat_snaps / flat_acts / flat_masks are deleted after the update.
  • An optional VRAM usage probe (TRACK_VRAM=True in config) prints
    torch.cuda.memory_allocated() and torch.cuda.memory_reserved() before
    and after each PPO update to make future regressions visible.

  [SPEED — targeting 5-10× speedup on the PPO update]
  • AMP (Automatic Mixed Precision): forward pass runs in fp16/bf16 via
    torch.amp.autocast; backward uses GradScaler.  Typical gain: 1.5–3×.
  • Fewer minibatches (n_minibatches: 64 → 16): each minibatch is 4× larger,
    so the GNN batched encode_batch call is 4× more GPU-efficient, and the
    Python loop overhead is 4× smaller.  Gradient quality improves (lower
    variance).  Gain: ~4×.
  • Per-epoch return recompute disabled by default (recompute_returns=False).
    With 2 epochs this removed 128 extra no-grad GNN forward passes per
    update.  Gain: ~2× on total forward passes.
  Combined theoretical speedup on PPO wall time: 4 × 1.5–3 × ~1.5 ≈ 9–18×.
    Real-world gain depends on graph size and GPU; expect 5–10×.
  • torch.backends.cudnn.benchmark = True for conv-heavy workloads.

  [LOGGING]
  • Dedicated ./logs/ directory created automatically.
  • Per-run timestamped .log file with Python logging (INFO to both file and
    stdout).
  • Per-run CSV metrics file: one row per update with all loss curves, timing
    components, win-rate, avg reward, VRAM stats.
  • Console banner unchanged.

  [MISC]
  • Removed hard-coded start index (range(13, ...)); controlled by
    cfg.start_update.
  • Checkpoint interval configurable via cfg.ckpt_interval.
"""

# ── Standard library ─────────────────────────────────────────────────────────
import sys, os, time, random, gc, csv, logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

# ── Scientific stack ─────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm

# ── Project root ──────────────────────────────────────────────────────────────
sys.path.insert(0, r"C:\Users\laure\1own_projects\1polytopia_score")

from env.wrapper import EnvWrapper
from game.enums import BoardType, Tribes
from RL.models.policy import (
    PolicyNetwork,
    make_snapshot,
    model_summary,
)


# ══════════════════════════════════════════════════════════════════════════════
# Welford running mean / std  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class RunningMeanStd:
    """Online Welford estimator for scalar return normalisation."""

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
# Unified Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """
    Single unified hyperparameter object.

    Architecture (passed to PolicyNetwork)
    ───────────────────────────────────────
    encoder_hidden_dim  — hidden width of GraphTransformerEncoder; must be
                          divisible by encoder_n_heads AND by 4 (RoPE).
    encoder_n_heads     — attention heads per TransformerConv layer.
    encoder_depth       — number of TransformerConv layers.
    sel_n_heads         — MHSA heads per SequenceSelectionHead block.
    sel_n_layers        — transformer blocks per SequenceSelectionHead.
    mlp_hidden_dim      — MLP hidden width (all heads).
    mlp_depth           — MLP hidden layers (minimum 1).
    kernel_sizes        — spatial conv scales, all odd, largest first.
    n_conv_layers       — stacked Conv2d layers per kernel size.
    context_bias        — movement context window padding (>= max(kernel)//2).

    Checkpoint / resume
    ────────────────────
    pretrained_ckpt  — path to .pt file to load before training.
                       Set to empty string "" to train from scratch.
    start_update     — first update index (set > 0 when resuming).

    Parallelism
    ────────────
    n_processes, n_envs_per_process

    Environment
    ────────────
    board_config_dict, player_tribes, max_turns_per_game, board_size_range

    Rollout
    ────────
    n_steps

    PPO hyperparameters
    ────────────────────
    n_epochs, n_minibatches, clip_eps, vf_coef, ent_coef, max_grad_norm,
    gamma, gae_lambda, lr

    Training loop
    ──────────────
    n_updates, log_interval, ckpt_interval

    Speed / memory flags
    ─────────────────────
    use_amp             — Automatic Mixed Precision (fp16 forward, fp32 master
                          weights).  Typically 1.5–3× faster on CUDA.
    recompute_returns   — Re-run GAE with fresh critic values at the start of
                          every PPO epoch.  Improves sample efficiency but adds
                          ~n_epochs × n_minibatches extra no-grad GNN passes.
                          Disable for speed (default False).
    track_vram          — Print VRAM stats before/after each PPO update.
    """

    # ── Checkpoint / resume ───────────────────────────────────────────────────
    pretrained_ckpt: str = ""
    start_update:    int = 0   # exclusive lower-bound of the update range

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
    n_processes:        int = 8
    n_envs_per_process: int = 4

    # ── Environment ───────────────────────────────────────────────────────────
    board_config_dict: dict = field(default_factory=lambda: {
        "board_type": BoardType.Dummy,
        "n_players" : 2,
    })
    player_tribes:      list  = field(
        default_factory=lambda: [Tribes.Omaji, Tribes.Imperius]
    )
    max_turns_per_game: int   = 100
    board_size_range:   tuple = (10, 16)

    # ── Rollout ───────────────────────────────────────────────────────────────
    n_steps: int = 256

    # ── PPO epochs & batching ─────────────────────────────────────────────────
    n_epochs:      int = 2
    # SPEED: reduced from 64 → 16.  4× bigger minibatches = far fewer GNN
    # forward passes, much better GPU utilisation, lower loop overhead.
    # Gradient variance also improves with larger mini-batches.
    n_minibatches: int = 16

    # ── PPO loss coefficients ─────────────────────────────────────────────────
    clip_eps:      float = 0.2
    vf_coef:       float = 0.5
    ent_coef:      float = 0.01
    max_grad_norm: float = 0.5

    # ── GAE / discount ────────────────────────────────────────────────────────
    gamma:      float = 0.99
    gae_lambda: float = 0.95

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr: float = 3e-4

    # ── Training loop ─────────────────────────────────────────────────────────
    n_updates:     int = 500
    log_interval:  int = 1
    ckpt_interval: int = 1

    # ── Optimisation / diagnostic flags ──────────────────────────────────────
    use_amp:           bool = True   # AMP  (strongly recommended on CUDA)
    recompute_returns: bool = False  # Per-epoch GAE refresh  (expensive)
    track_vram:        bool = True   # Log VRAM before/after PPO update

    # ── Derived properties ────────────────────────────────────────────────────
    @property
    def n_envs_total(self) -> int:
        return self.n_processes * self.n_envs_per_process

    @property
    def batch_size(self) -> int:
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
            "encoder_hidden_dim must be divisible by 4 for 2D RoPE."
        )
        assert all(k % 2 == 1 for k in self.kernel_sizes), \
            "All kernel_sizes must be odd integers."
        assert self.context_bias >= max(self.kernel_sizes) // 2, (
            f"context_bias ({self.context_bias}) must be >= "
            f"max(kernel_sizes)//2 ({max(self.kernel_sizes) // 2})."
        )
        assert self.mlp_depth     >= 1, "mlp_depth must be >= 1."
        assert self.n_conv_layers >= 1, "n_conv_layers must be >= 1."
        assert self.n_minibatches >= 1, "n_minibatches must be >= 1."
        assert self.batch_size % self.n_minibatches == 0, (
            f"batch_size ({self.batch_size}) must be divisible by "
            f"n_minibatches ({self.n_minibatches})."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Logging setup
# ══════════════════════════════════════════════════════════════════════════════

LOG_DIR = Path("logs")
_CSV_FIELDS = [
    "update", "wall_time_s",
    "t_dist_s", "t_collect_s", "t_gae_s", "t_ppo_s",
    "n_finished", "n_won", "win_rate",
    "avg_ep_len", "avg_reward",
    "p_loss", "v_loss", "entropy",
    "vram_alloc_before_mb", "vram_reserved_before_mb",
    "vram_alloc_after_mb",  "vram_reserved_after_mb",
]


def _setup_logging(run_tag: str):
    """
    Create ./logs/<run_tag>.log   (human-readable, timestamped)
          ./logs/<run_tag>_metrics.csv  (machine-readable per-update stats)

    Returns
    ───────
    logger   : logging.Logger
    csv_path : Path  (caller opens it to append rows)
    """
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{run_tag}.log"
    csv_path = LOG_DIR / f"{run_tag}_metrics.csv"

    logger = logging.getLogger("polytopia_rl")
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers if main() is called more than once in a session
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S"))
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.addHandler(ch)

    # Write CSV header only if the file is new / empty
    write_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    csv_fh = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=_CSV_FIELDS)
    if write_header:
        writer.writeheader()
        csv_fh.flush()

    return logger, writer, csv_fh


def _vram_mb(device: torch.device) -> Tuple[float, float]:
    """Return (allocated_MB, reserved_MB) for `device`, or (0, 0) on CPU."""
    if device.type != "cuda":
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated(device) / 1024 ** 2,
        torch.cuda.memory_reserved(device)  / 1024 ** 2,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint bookkeeping
# ══════════════════════════════════════════════════════════════════════════════

MAX_CKPT_KEEP = 3
CKPT_DIR      = "checkpoints_hierarchical"


def _save_checkpoint(policy: PolicyNetwork, update: int,
                     ckpt_queue: deque, logger: logging.Logger) -> None:
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"policy_update_{update:05d}.pt")
    torch.save(policy.state_dict(), path)
    ckpt_queue.append(path)
    if len(ckpt_queue) > MAX_CKPT_KEEP:
        oldest = ckpt_queue.popleft()
        if os.path.exists(oldest):
            os.remove(oldest)
            logger.info(f"  [ckpt] removed old checkpoint: {oldest}")
    logger.info(f"  [ckpt] saved → {path}  (keeping last {MAX_CKPT_KEEP})")


# ══════════════════════════════════════════════════════════════════════════════
# Environment helpers
# ══════════════════════════════════════════════════════════════════════════════

def _random_board_size(cfg: TrainConfig) -> tuple:
    n = random.randint(cfg.board_size_range[0], cfg.board_size_range[1])
    return (n, n)


def _make_env(cfg: TrainConfig) -> EnvWrapper:
    board_size   = _random_board_size(cfg)
    board_config = {
        "board_size": list(board_size),
        "board_type": cfg.board_config_dict["board_type"],
        "n_players" : cfg.board_config_dict["n_players"],
    }
    return EnvWrapper(
        board_config, cfg.player_tribes,
        max_turns_per_game=cfg.max_turns_per_game,
        dense_reward=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Worker process — CPU only
# ══════════════════════════════════════════════════════════════════════════════

def worker_fn(worker_id: int, cfg: TrainConfig, conn) -> None:
    """
    Collect T-step rollouts across M environments on CPU.

    Protocol
    ────────
    Receives : ('collect', state_dict)  → runs rollout → sends ('data', chunk)
    Receives : ('stop', None)           → exits cleanly

    Chunk shapes
    ─────────────
    log_probs   (T, M) float32   values  (T, M) float32
    rewards     (T, M) float32   dones   (T, M) float32
    won_flags   (T, M) float32   last_values (M,) float32
    player_ids  (T, M) int32
    obs_snaps   list[T] × list[M]  — dicts from make_snapshot()
    actions     list[T] × list[M]  — action lists
    masks       list[T] × list[M]  — mask lists
    """
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

        # Load latest weights (CPU tensors sent from main process)
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
            for t in range(T):
                for e, env in enumerate(envs):
                    obs  = obs_buf[e]
                    mask = env.get_action_mask()
                    snap = make_snapshot(
                        obs, env.Nx, env.Ny,
                        player_id=env.game.player_go_id,
                    )
                    action, _, _, lp, _, val = policy(obs, mask)
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
                        envs[e]    = _make_env(cfg)
                        obs_buf[e] = envs[e].reset()
                    else:
                        obs_buf[e] = next_obs

            # Bootstrap V(s_T)
            last_values = np.zeros(M, dtype=np.float32)
            for e in range(M):
                snap_last = make_snapshot(
                    obs_buf[e], envs[e].Nx, envs[e].Ny,
                    player_id=envs[e].game.player_go_id,
                )
                node_emb, global_emb = policy.encoder.encode(
                    snap_last["graph"], snap_last["Nx"], snap_last["Ny"],
                )
                last_values[e] = policy.critic(global_emb).item()

        player_ids = np.array(
            [[obs_snaps[t][e]["player_id"] for e in range(M)]
             for t in range(T)],
            dtype=np.int32,
        )

        elapsed = time.time() - t0
        print(f"  [worker {worker_id:02d}] rollout done — {elapsed:.2f}s",
              flush=True)

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
# GAE  (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════

def compute_gae(
    rewards:     np.ndarray,   # (T, N)
    values:      np.ndarray,   # (T, N)
    dones:       np.ndarray,   # (T, N)
    last_values: np.ndarray,   # (N,)
    gamma:       float,
    gae_lam:     float,
    player_ids:  np.ndarray,   # (T, N)  int32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalised Advantage Estimation with player-switch masking.
    Returns (advantages, returns) both (T, N) float32.
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
    scaler:         torch.cuda.amp.GradScaler,
    batch:          dict,
    cfg:            TrainConfig,
    device:         torch.device,
    ret_normalizer: RunningMeanStd,
) -> Tuple[float, float, float]:
    """
    PPO gradient update with AMP, explicit memory hygiene, and optional
    per-epoch return recomputation.

    Memory hygiene
    ───────────────
    Every large GPU tensor allocated in this function is explicitly deleted
    before returning.  This guarantees that PyTorch's CUDA allocator sees
    a steady peak — it does not accumulate across successive ppo_update()
    calls — so `torch.cuda.empty_cache()` in main() will reliably reclaim
    fragmented blocks between updates.

    AMP
    ────
    When cfg.use_amp is True:
      • Forward pass runs inside torch.amp.autocast (fp16 on CUDA, bf16 on
        CPU as fallback).
      • GradScaler handles loss scaling to prevent fp16 underflow.
      • The normalisation operations on adv / ret are kept in fp32 because
        they involve reduction over large tensors.

    Returns
    ───────
    (mean_policy_loss, mean_value_loss, mean_entropy)
    """
    B       = cfg.batch_size
    amp_ctx = (
        torch.amp.autocast(device_type=device.type)
        if cfg.use_amp and device.type == "cuda"
        else torch.no_grad().__class__()  # identity context — never triggers
    )
    # Note: for the gradient forward pass we want autocast, not no_grad.
    # We define a separate lambda to reuse the device string.
    def _autocast():
        return (torch.amp.autocast(device_type=device.type)
                if cfg.use_amp and device.type == "cuda"
                else _NullContext())

    # ── Flatten rollout lists ─────────────────────────────────────────────────
    flat_snaps: List[dict] = [s for step in batch["obs_snaps"] for s in step]
    flat_acts:  List[list] = [a for step in batch["actions"]   for a in step]
    flat_masks: List[list] = [m for step in batch["masks"]      for m in step]

    # Old log-probs (fixed reference throughout all PPO epochs) ───────────────
    log_old: torch.Tensor = torch.tensor(
        batch["log_probs"].reshape(-1), dtype=torch.float32,
    ).to(device)   # (B,)

    # Initial adv / ret from rollout GAE ─────────────────────────────────────
    adv_np: np.ndarray = batch["advantages"].reshape(-1).astype(np.float32)
    ret_np: np.ndarray = batch["returns"].reshape(-1).astype(np.float32)

    # Update running return stats once per ppo_update call ────────────────────
    ret_normalizer.update(ret_np)

    indices = np.arange(B)
    pl_log, vl_log, el_log = [], [], []

    total_steps = cfg.n_epochs * cfg.n_minibatches
    pbar = tqdm(total=total_steps,
                desc="  PPO epochs × minibatches",
                leave=False, unit="mb")

    for epoch in range(cfg.n_epochs):

        # ── Optional: refresh returns with fresh critic values ─────────────
        if cfg.recompute_returns:
            with torch.no_grad():
                fresh_chunks = []
                for start in range(0, B, cfg.minibatch_size):
                    mb_snaps = flat_snaps[start: start + cfg.minibatch_size]
                    with _autocast():
                        chunk = policy.compute_values_batch(mb_snaps).cpu().numpy()
                    fresh_chunks.append(chunk)
                fresh_vals_flat = np.concatenate(fresh_chunks)  # (B,)
            del fresh_chunks

            N_total       = cfg.n_envs_total
            fresh_vals_2d = fresh_vals_flat.reshape(cfg.n_steps, N_total)
            adv_2d, ret_2d = compute_gae(
                batch["rewards"],    fresh_vals_2d, batch["dones"],
                batch["last_values"], cfg.gamma,    cfg.gae_lambda,
                batch["player_ids"],
            )
            adv_np = adv_2d.reshape(-1).astype(np.float32)
            ret_np = ret_2d.reshape(-1).astype(np.float32)
            del fresh_vals_flat, fresh_vals_2d, adv_2d, ret_2d

        # ── Build epoch-level GPU tensors ──────────────────────────────────
        adv: torch.Tensor = torch.tensor(adv_np, dtype=torch.float32).to(device)
        ret: torch.Tensor = torch.tensor(ret_np, dtype=torch.float32).to(device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        ret_norm: torch.Tensor = torch.tensor(
            ret_normalizer.normalize(ret_np), dtype=torch.float32,
        ).to(device)   # (B,)

        np.random.shuffle(indices)

        # ── Minibatch loop ─────────────────────────────────────────────────
        for start in range(0, B, cfg.minibatch_size):
            mb = indices[start: start + cfg.minibatch_size]

            with _autocast():
                new_lp, new_ent, new_val = policy.evaluate_actions(
                    [flat_snaps[i] for i in mb],
                    [flat_acts[i]  for i in mb],
                    [flat_masks[i] for i in mb],
                )

                ratio    = torch.exp(new_lp - log_old[mb])   # (mb,)
                mb_adv   = adv[mb]                            # (mb,)

                loss_clip = torch.min(
                    ratio * mb_adv,
                    ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv,
                ).mean()

                loss_val = F.mse_loss(new_val.squeeze(), ret_norm[mb])
                loss_ent = new_ent.mean()

                loss = -loss_clip + cfg.vf_coef * loss_val - cfg.ent_coef * loss_ent

            optimizer.zero_grad(set_to_none=True)   # set_to_none=True saves memory

            if cfg.use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()

            pl_log.append(loss_clip.item())
            vl_log.append(loss_val.item())
            el_log.append(loss_ent.item())

            pbar.set_postfix(
                epoch  = epoch + 1,
                p_loss = f"{loss_clip.item():.4f}",
                v_loss = f"{loss_val.item():.4f}",
            )
            pbar.update(1)

            # Free minibatch GPU tensors immediately
            del new_lp, new_ent, new_val, ratio, mb_adv
            del loss_clip, loss_val, loss_ent, loss

        # Free epoch-level GPU tensors before the next epoch allocates its own
        del adv, ret, ret_norm

    pbar.close()

    # ── Final cleanup of function-scoped GPU tensors ─────────────────────────
    del log_old
    del flat_snaps, flat_acts, flat_masks

    return float(np.mean(pl_log)), float(np.mean(vl_log)), float(np.mean(el_log))


class _NullContext:
    """No-op context manager (replaces autocast on CPU)."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg    = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        # Helps with conv-heavy graphs; safe to enable permanently
        torch.backends.cudnn.benchmark = True

    # ── Logging ───────────────────────────────────────────────────────────────
    run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    logger, csv_writer, csv_fh = _setup_logging(run_tag)
    logger.info(f"Run tag : {run_tag}")
    logger.info(f"Log dir : {LOG_DIR.resolve()}")

    # ══════════════════════════════════════════════════════════════════════════
    #  CHECKPOINT LOADING  — happens BEFORE optimizer construction so that
    #  Adam moment buffers are associated with the already-loaded parameters.
    # ══════════════════════════════════════════════════════════════════════════
    policy = PolicyNetwork(cfg).to(device)

    if cfg.pretrained_ckpt and os.path.exists(cfg.pretrained_ckpt):
        logger.info(f"Loading pretrained weights from: {cfg.pretrained_ckpt}")
        state_dict = torch.load(
            cfg.pretrained_ckpt, weights_only=True, map_location=device
        )
        policy.load_state_dict(state_dict)
        del state_dict                   # free the extra copy in Python memory
        logger.info("Checkpoint loaded successfully.")
    elif cfg.pretrained_ckpt:
        logger.warning(
            f"pretrained_ckpt '{cfg.pretrained_ckpt}' not found — "
            "training from scratch."
        )
    else:
        logger.info("No pretrained checkpoint specified — training from scratch.")

    policy.train()

    # ── Optimizer + AMP scaler ────────────────────────────────────────────────
    optimizer      = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    scaler         = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))
    ret_normalizer = RunningMeanStd()

    # ── Console banner ─────────────────────────────────────────────────────────
    lo, hi = cfg.board_size_range
    banner_lines = [
        "",
        "=" * 70,
        "  POLYTOPIA RL — PPO TRAINING  (hierarchical policy, optimised)",
        "=" * 70,
        f"  Device             : {device}",
        f"  Board size         : random square [{lo}×{lo} … {hi}×{hi}]",
        f"  Batch size         : {cfg.batch_size:,}  "
        f"({cfg.n_processes} proc × {cfg.n_envs_per_process} envs "
        f"× {cfg.n_steps} steps)",
        f"  Minibatch size     : {cfg.minibatch_size:,}  |  "
        f"Minibatches: {cfg.n_minibatches}  |  "
        f"Epochs: {cfg.n_epochs}  |  Updates: {cfg.n_updates}",
        f"  Encoder            : {cfg.encoder_depth} layers "
        f"× {cfg.encoder_hidden_dim}d  ({cfg.encoder_n_heads} heads)",
        f"  Selection heads    : {cfg.sel_n_layers} layers "
        f"× {cfg.encoder_hidden_dim}d  ({cfg.sel_n_heads} heads)",
        f"  MLP                : {cfg.mlp_depth} hidden layers "
        f"× {cfg.mlp_hidden_dim}d",
        f"  Conv kernels       : {cfg.kernel_sizes}  "
        f"× {cfg.n_conv_layers} stacked layers",
        f"  Context bias       : {cfg.context_bias}",
        f"  AMP                : {cfg.use_amp}",
        f"  Recompute returns  : {cfg.recompute_returns}",
        f"  Start update       : {cfg.start_update}",
        f"  Pretrained ckpt    : {cfg.pretrained_ckpt or '(none)'}",
        "",
    ]
    for line in banner_lines:
        logger.info(line)
    model_summary(policy)
    logger.info("")

    # ── Spawn workers ──────────────────────────────────────────────────────────
    parent_conns, child_conns = zip(*[mp.Pipe() for _ in range(cfg.n_processes)])
    workers = [
        mp.Process(
            target=worker_fn,
            args=(i, cfg, child_conns[i]),
            daemon=True,
        )
        for i in range(cfg.n_processes)
    ]
    for w in workers:
        w.start()

    ckpt_queue = deque()

    # ── Outer update loop ──────────────────────────────────────────────────────
    outer_bar = tqdm(
        range(cfg.start_update, cfg.n_updates),
        desc="Updates", unit="upd",
    )

    for update in outer_bar:
        t_update_start = time.time()

        # ── 1. Distribute latest weights to workers ────────────────────────
        t0 = time.time()
        # Move to CPU before IPC so workers can load without a GPU
        state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
        for conn in parent_conns:
            conn.send(("collect", state_dict))
        del state_dict   # release the extra CPU copy immediately
        t_dist = time.time() - t0
        logger.info(f"\n[update {update:04d}] weights dispatched in {t_dist:.3f}s")

        # ── 2. Collect rollouts from all workers ───────────────────────────
        t0 = time.time()
        logger.info(f"[update {update:04d}] waiting for {cfg.n_processes} workers …")
        chunks = []
        for i, conn in enumerate(
            tqdm(parent_conns, desc="  collecting workers", leave=False, unit="worker")
        ):
            chunks.append(conn.recv()[1])
        t_collect = time.time() - t0
        logger.info(f"[update {update:04d}] all workers done in {t_collect:.2f}s")

        # ── 3. Assemble full batch ─────────────────────────────────────────
        t0 = time.time()
        batch = {
            "obs_snaps": [
                sum([c["obs_snaps"][t] for c in chunks], [])
                for t in range(cfg.n_steps)
            ],
            "actions": [
                sum([c["actions"][t] for c in chunks], [])
                for t in range(cfg.n_steps)
            ],
            "masks": [
                sum([c["masks"][t] for c in chunks], [])
                for t in range(cfg.n_steps)
            ],
            "log_probs":   np.concatenate([c["log_probs"]   for c in chunks], axis=1),
            "values":      np.concatenate([c["values"]      for c in chunks], axis=1),
            "rewards":     np.concatenate([c["rewards"]     for c in chunks], axis=1),
            "dones":       np.concatenate([c["dones"]       for c in chunks], axis=1),
            "won_flags":   np.concatenate([c["won_flags"]   for c in chunks], axis=1),
            "last_values": np.concatenate([c["last_values"] for c in chunks]),
            "player_ids":  np.concatenate([c["player_ids"]  for c in chunks], axis=1),
        }
        # chunks held the only other references to these numpy arrays; drop it
        del chunks
        gc.collect()

        # ── 4. GAE ────────────────────────────────────────────────────────
        adv, ret = compute_gae(
            batch["rewards"], batch["values"], batch["dones"],
            batch["last_values"], cfg.gamma, cfg.gae_lambda,
            batch["player_ids"],
        )
        batch["advantages"] = adv
        batch["returns"]    = ret
        t_gae = time.time() - t0
        logger.info(f"[update {update:04d}] GAE computed in {t_gae:.3f}s")

        # ── 5. VRAM snapshot (before PPO update) ──────────────────────────
        vram_alloc_before, vram_res_before = _vram_mb(device)
        if cfg.track_vram and device.type == "cuda":
            logger.info(
                f"[update {update:04d}] VRAM before PPO  "
                f"alloc={vram_alloc_before:.0f} MB  "
                f"reserved={vram_res_before:.0f} MB"
            )

        # ── 6. PPO update (GPU) ───────────────────────────────────────────
        t0 = time.time()
        pl, vl, el = ppo_update(
            policy, optimizer, scaler, batch, cfg, device, ret_normalizer
        )
        t_ppo = time.time() - t0
        logger.info(f"[update {update:04d}] PPO update done in {t_ppo:.2f}s")

        # ── 7. Extract game stats BEFORE releasing batch ──────────────────
        # These are cheap numpy reductions; must happen before del batch.
        n_finished   = int(batch["dones"].sum())
        n_won        = int(batch["won_flags"].sum())
        n_timeout    = n_finished - n_won
        total_reward = float(batch["rewards"].sum())
        avg_reward   = total_reward / max(n_finished, 1)
        win_rate     = n_won / max(n_finished, 1)
        avg_ep_len   = cfg.batch_size / max(n_finished, 1)

        # ── 8. Explicit memory release ────────────────────────────────────
        # batch contains large numpy arrays and lists of snapshot dicts.
        # Deleting it here (before empty_cache) gives PyTorch's CUDA
        # allocator the best chance to reclaim blocks and keep VRAM flat
        # across updates.
        del batch, adv, ret
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ── 9. VRAM snapshot (after PPO update + cleanup) ─────────────────
        vram_alloc_after, vram_res_after = _vram_mb(device)
        if cfg.track_vram and device.type == "cuda":
            logger.info(
                f"[update {update:04d}] VRAM after  PPO  "
                f"alloc={vram_alloc_after:.0f} MB  "
                f"reserved={vram_res_after:.0f} MB"
            )

        # ── 10. Checkpoint ────────────────────────────────────────────────
        if update % cfg.ckpt_interval == 0:
            _save_checkpoint(policy, update, ckpt_queue, logger)

        # ── 11. Stats & console logging ────────────────────────────────────
        t_total = time.time() - t_update_start

        if update % cfg.log_interval == 0:
            logger.info("")
            logger.info(
                f"╔══ update {update:04d} ══════════════════════════════════════════════"
            )
            logger.info(
                f"║  Wall time     : {t_total:.1f}s  "
                f"(dist {t_dist:.2f}s | collect {t_collect:.2f}s "
                f"| GAE {t_gae:.3f}s | PPO {t_ppo:.2f}s)"
            )
            logger.info(
                f"║  Games finished: {n_finished:6d}  "
                f"(conquest {n_won}, timeout {n_timeout})"
            )
            logger.info(f"║  Win rate      : {win_rate:.3f}")
            logger.info(f"║  Avg ep length : {avg_ep_len:.1f} steps")
            logger.info(f"║  Avg reward/ep : {avg_reward:.3f}")
            logger.info(f"║  p_loss        : {pl:.4f}")
            logger.info(f"║  v_loss        : {vl:.4f}")
            logger.info(f"║  entropy       : {el:.4f}")
            if cfg.track_vram and device.type == "cuda":
                logger.info(
                    f"║  VRAM          : {vram_alloc_after:.0f} MB alloc  "
                    f"/ {vram_res_after:.0f} MB reserved"
                )
            logger.info(f"╚{'═' * 60}")
            logger.info("")

        outer_bar.set_postfix(
            fin = n_finished,
            win = f"{win_rate:.2f}",
            p   = f"{pl:.3f}",
            v   = f"{vl:.3f}",
            ent = f"{el:.3f}",
        )

        # ── 12. CSV metrics row ────────────────────────────────────────────
        csv_writer.writerow({
            "update":                  update,
            "wall_time_s":             f"{t_total:.3f}",
            "t_dist_s":                f"{t_dist:.3f}",
            "t_collect_s":             f"{t_collect:.3f}",
            "t_gae_s":                 f"{t_gae:.4f}",
            "t_ppo_s":                 f"{t_ppo:.3f}",
            "n_finished":              n_finished,
            "n_won":                   n_won,
            "win_rate":                f"{win_rate:.4f}",
            "avg_ep_len":              f"{avg_ep_len:.2f}",
            "avg_reward":              f"{avg_reward:.4f}",
            "p_loss":                  f"{pl:.6f}",
            "v_loss":                  f"{vl:.6f}",
            "entropy":                 f"{el:.6f}",
            "vram_alloc_before_mb":    f"{vram_alloc_before:.1f}",
            "vram_reserved_before_mb": f"{vram_res_before:.1f}",
            "vram_alloc_after_mb":     f"{vram_alloc_after:.1f}",
            "vram_reserved_after_mb":  f"{vram_res_after:.1f}",
        })
        csv_fh.flush()

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(CKPT_DIR, "policy_FINAL.pt")
    torch.save(policy.state_dict(), final_path)
    logger.info(f"\nFinal policy saved → {final_path}")

    # ── Shutdown workers ──────────────────────────────────────────────────────
    for conn in parent_conns:
        conn.send(("stop", None))
    for w in workers:
        w.join()
    logger.info("All workers shut down.  Training complete.")

    csv_fh.close()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()