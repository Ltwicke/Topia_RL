"""
train.py
──────────────────────────────────────────────────────────────────────────────
Entry point for Polytopia RL PPO training with per-player MDP streams.

Architecture
────────────
This script orchestrates three cooperating modules from the ppo/ package:

  EnvManager     — spawns worker processes, distributes policy weights,
                   assembles the raw (T × N) rollout batch.

  BatchProcessor — computes per-player GAE (advantages can now propagate
                   across turn boundaries), normalises returns, and produces
                   minibatch iterators via PyTorch's BatchSampler.

  PPOTrainer     — holds the Adam optimiser + AMP GradScaler, executes the
                   clipped PPO loss for each minibatch, and handles explicit
                   GPU memory cleanup.

All logging (console + CSV + .log file) lives here in main().

Per-update loop
───────────────
  1. Distribute latest policy weights to workers           [EnvManager]
  2. Collect T-step rollouts from all workers              [EnvManager]
  3. Compute per-player GAE + normalisation                [BatchProcessor]
  4. Run n_epochs × (train_fraction × n_minibatches) steps [PPOTrainer]
  5. Log, checkpoint.

Per-player GAE (key change vs previous pipeline)
──────────────────────────────────────────────────
This new BatchProcessor computes GAE independently for each player's own decision
stream: player p's "next state" is the state p observes AFTER the opponent's
turn, so advantages can reflect multi-turn strategic consequences.
"""

from __future__ import annotations

import csv
import gc
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# ── Project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = r"C:\Users\laure\1own_projects\1polytopia_score"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.policy import PolicyNetwork, model_summary

# ── PPO modules ───────────────────────────────────────────────────────────────
from ppo.game_manager    import TrainConfig, EnvManager
from ppo.batch_processing import BatchProcessor
from ppo.ppo   import PPOTrainer


# ══════════════════════════════════════════════════════════════════════════════
# Logging helpers
# ══════════════════════════════════════════════════════════════════════════════

LOG_DIR   = Path("logs")
CKPT_DIR  = "checkpoints_training"
MAX_CKPT  = 3

_CSV_FIELDS = [
    "update",
    "wall_time_s", "t_dist_s", "t_collect_s", "t_gae_s", "t_ppo_s",
    "n_finished", "n_won", "win_rate",
    "avg_ep_len", "avg_reward",
    "p_loss", "v_loss", "entropy",
    "vram_alloc_before_mb", "vram_reserved_before_mb",
    "vram_alloc_after_mb",  "vram_reserved_after_mb",
]


def _setup_logging(run_tag: str):
    """
    Create per-run .log and _metrics.csv files in ./logs/.

    Returns
    ───────
    logger     : logging.Logger
    csv_writer : csv.DictWriter
    csv_fh     : file handle  (caller must call csv_fh.close() at the end)
    """
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{run_tag}.log"
    csv_path = LOG_DIR / f"{run_tag}_metrics.csv"

    logger = logging.getLogger("polytopia_rl")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.addHandler(ch)

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


def _save_checkpoint(
    policy:     PolicyNetwork,
    update:     int,
    ckpt_queue: deque,
    logger:     logging.Logger,
) -> None:
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"policy_update_{update:05d}.pt")
    torch.save(policy.state_dict(), path)
    ckpt_queue.append(path)
    if len(ckpt_queue) > MAX_CKPT:
        oldest = ckpt_queue.popleft()
        if os.path.exists(oldest):
            os.remove(oldest)
            logger.info(f"  [ckpt] removed old checkpoint: {oldest}")
    logger.info(f"  [ckpt] saved → {path}  (keeping last {MAX_CKPT})")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg    = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── Logging setup ─────────────────────────────────────────────────────────
    run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    logger, csv_writer, csv_fh = _setup_logging(run_tag)
    logger.info(f"Run tag : {run_tag}")
    logger.info(f"Log dir : {LOG_DIR.resolve()}")

    # ── Policy construction ───────────────────────────────────────────────────
    policy = PolicyNetwork(cfg).to(device)

    if cfg.pretrained_ckpt and os.path.exists(cfg.pretrained_ckpt):
        logger.info(f"Loading pretrained weights: {cfg.pretrained_ckpt}")
        state = torch.load(
            cfg.pretrained_ckpt, weights_only=True, map_location=device
        )
        policy.load_state_dict(state)
        del state
        logger.info("Checkpoint loaded.")
    elif cfg.pretrained_ckpt:
        logger.warning(
            f"pretrained_ckpt '{cfg.pretrained_ckpt}' not found — "
            "training from scratch."
        )
    else:
        logger.info("No pretrained checkpoint — training from scratch.")

    policy.train()

    # ── Module construction ───────────────────────────────────────────────────
    env_manager    = EnvManager(cfg)
    batch_proc     = BatchProcessor(cfg)
    trainer        = PPOTrainer(policy, cfg, device)

    # ── Console banner ─────────────────────────────────────────────────────────
    lo, hi = cfg.board_size_range
    n_train_per_epoch = max(cfg.minibatch_size,
                            int(cfg.batch_size * cfg.train_fraction))
    n_mb_per_epoch = n_train_per_epoch // cfg.minibatch_size

    banner_lines = [
        "",
        "=" * 72,
        "  POLYTOPIA RL — PPO  (per-player MDP streams)",
        "=" * 72,
        f"  Device            : {device}",
        f"  Board size        : random square [{lo}×{lo} … {hi}×{hi}]",
        f"  Workers × envs    : {cfg.n_processes} × {cfg.n_envs_per_process}"
        f"  =  {cfg.n_envs_total} parallel envs",
        f"  Rollout steps     : {cfg.n_steps}",
        f"  Full batch        : {cfg.batch_size:,} samples "
        f"({cfg.n_steps} steps × {cfg.n_envs_total} envs)",
        f"  Train fraction    : {cfg.train_fraction:.2f}  "
        f"→  {n_train_per_epoch:,} samples / epoch  "
        f"({n_mb_per_epoch} minibatches × {cfg.minibatch_size})",
        f"  PPO epochs        : {cfg.n_epochs}  |  Updates: {cfg.n_updates}",
        f"  clip_eps / vf / ent: {cfg.clip_eps} / {cfg.vf_coef} / {cfg.ent_coef}",
        f"  γ / λ             : {cfg.gamma} / {cfg.gae_lambda}",
        f"  LR                : {cfg.lr}",
        f"  AMP               : {cfg.use_amp}",
        f"  Encoder           : {cfg.encoder_depth} layers "
        f"× {cfg.encoder_hidden_dim}d ({cfg.encoder_n_heads} heads)",
        f"  Selection heads   : {cfg.sel_n_layers} layers "
        f"× {cfg.encoder_hidden_dim}d ({cfg.sel_n_heads} heads)",
        f"  MLP               : {cfg.mlp_depth} layers × {cfg.mlp_hidden_dim}d",
        f"  Conv kernels      : {cfg.kernel_sizes} × {cfg.n_conv_layers} layers",
        f"  Context bias      : {cfg.context_bias}",
        f"  Start update      : {cfg.start_update}",
        f"  Pretrained        : {cfg.pretrained_ckpt or '(none)'}",
        "",
        "  GAE mode          : per-player MDP (cross-turn credit assignment)",
        "",
    ]
    for line in banner_lines:
        logger.info(line)
    model_summary(policy)
    logger.info("")

    # ── Spawn workers ──────────────────────────────────────────────────────────
    env_manager.start()
    logger.info(f"Spawned {cfg.n_processes} worker processes.")

    ckpt_queue = deque()
    outer_bar  = tqdm(
        range(cfg.start_update, cfg.n_updates),
        desc="Updates", unit="upd",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Outer update loop
    # ══════════════════════════════════════════════════════════════════════════
    for update in outer_bar:
        t_update_start = time.time()

        # ── 1. Distribute latest weights to workers ────────────────────────
        # Move to CPU before IPC (workers have no GPU)
        cpu_state = {k: v.cpu() for k, v in policy.state_dict().items()}
        t_dist    = env_manager.distribute(cpu_state)
        del cpu_state   # release extra CPU copy immediately
        logger.info(f"\n[update {update:04d}] weights dispatched in {t_dist:.3f}s")

        # ── 2. Collect rollouts ────────────────────────────────────────────
        logger.info(
            f"[update {update:04d}] waiting for {cfg.n_processes} workers …"
        )
        raw_batch, t_collect = env_manager.collect()
        logger.info(f"[update {update:04d}] all workers done in {t_collect:.2f}s")

        # ── 3. Per-player GAE + normalisation ─────────────────────────────
        processed_batch, t_gae = batch_proc.process(raw_batch)
        logger.info(f"[update {update:04d}] per-player GAE in {t_gae:.4f}s")

        # Release the raw batch now — large numpy arrays no longer needed.
        del raw_batch
        gc.collect()

        # ── 4. VRAM snapshot (before PPO) ─────────────────────────────────
        vram_alloc_before, vram_res_before = _vram_mb(device)
        if cfg.track_vram and device.type == "cuda":
            logger.info(
                f"[update {update:04d}] VRAM before PPO  "
                f"alloc={vram_alloc_before:.0f} MB  "
                f"reserved={vram_res_before:.0f} MB"
            )

        # ── 5. PPO gradient updates ────────────────────────────────────────
        t0    = time.time()
        stats = trainer.update(processed_batch, batch_proc)
        t_ppo = time.time() - t0
        logger.info(f"[update {update:04d}] PPO update done in {t_ppo:.2f}s")

        # ── 6. Extract game stats from processed_batch ────────────────────
        n_finished   = processed_batch["n_finished"]
        n_won        = processed_batch["n_won"]
        n_timeout    = n_finished - n_won
        total_reward = processed_batch["total_reward"]
        avg_reward   = total_reward / max(n_finished, 1)
        win_rate     = n_won / max(n_finished, 1)
        avg_ep_len   = processed_batch["avg_ep_len"]

        pl = stats["p_loss"]
        vl = stats["v_loss"]
        el = stats["entropy"]

        # ── 7. Release processed batch ────────────────────────────────────
        del processed_batch
        gc.collect()

        # ── 8. VRAM snapshot (after PPO) ──────────────────────────────────
        vram_alloc_after, vram_res_after = _vram_mb(device)
        if cfg.track_vram and device.type == "cuda":
            logger.info(
                f"[update {update:04d}] VRAM after  PPO  "
                f"alloc={vram_alloc_after:.0f} MB  "
                f"reserved={vram_res_after:.0f} MB"
            )

        # ── 9. Checkpoint ──────────────────────────────────────────────────
        if update % cfg.ckpt_interval == 0:
            _save_checkpoint(policy, update, ckpt_queue, logger)

        # ── 10. Console logging ────────────────────────────────────────────
        t_total = time.time() - t_update_start

        if update % cfg.log_interval == 0:
            logger.info("")
            logger.info(
                f"╔══ update {update:04d} ════════════════════════════════════════════"
            )
            logger.info(
                f"║  Wall time     : {t_total:.1f}s  "
                f"(dist {t_dist:.2f}s | collect {t_collect:.2f}s "
                f"| GAE {t_gae:.4f}s | PPO {t_ppo:.2f}s)"
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

        # ── 11. CSV metrics row ────────────────────────────────────────────
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

    # ══════════════════════════════════════════════════════════════════════════
    # Finalise
    # ══════════════════════════════════════════════════════════════════════════
    final_path = os.path.join(CKPT_DIR, "policy_FINAL.pt")
    torch.save(policy.state_dict(), final_path)
    logger.info(f"\nFinal policy saved → {final_path}")

    env_manager.shutdown()
    logger.info("All workers shut down.  Training complete.")
    csv_fh.close()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()



