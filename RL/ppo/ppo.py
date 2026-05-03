"""
ppo/ppo_trainer.py
──────────────────────────────────────────────────────────────────────────────
PPOTrainer — owns the Adam optimiser and AMP GradScaler and executes all
             gradient update steps.

Design
──────
• update() drives the outer epoch loop and delegates per-minibatch steps to
  _step().
• Minibatches are produced by BatchProcessor.minibatch_generator(), which is
  called fresh every epoch so that the random permutation / train_fraction
  selection is re-drawn each time.
• All large intermediate GPU tensors are explicitly deleted inside _step()
  to keep VRAM usage flat across minibatches.
• Memory hygiene (empty_cache + gc.collect) runs once at the end of update(),
  NOT inside _step() — calling empty_cache inside a tight loop is expensive.

Loss formulation
────────────────
    L = −L_clip + c_v · L_val − c_e · L_ent

    L_clip = E[ min( r·Â,  clip(r, 1−ε, 1+ε)·Â ) ]        PPO-Clip
    L_val  = MSE( V_θ(s), G̃_t )                            value regression
    L_ent  = E[ H[π_θ(·|s)] ]                               entropy bonus

where r = π_θ(a|s) / π_θ_old(a|s), Â are whitened advantages, and G̃_t are
running-mean-std-normalised returns (from BatchProcessor).

AMP
───
When cfg.use_amp=True and device is CUDA:
  • The forward pass (evaluate_actions) runs in torch.amp.autocast (fp16/bf16).
  • GradScaler handles loss scaling to prevent fp16 underflow.
  • grad clipping uses the unscaled gradients (scaler.unscale_ before clip).
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ppo.game_manager import TrainConfig

if TYPE_CHECKING:
    # Avoid circular import at runtime; type-checker only.
    from ppo.batch_processing import BatchProcessor


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

class _NullContext:
    """No-op context manager — replaces autocast on CPU / non-AMP paths."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ══════════════════════════════════════════════════════════════════════════════
# PPOTrainer
# ══════════════════════════════════════════════════════════════════════════════

class PPOTrainer:
    """
    Executes PPO gradient updates over a processed batch.

    Parameters
    ──────────
    policy : nn.Module       — PolicyNetwork instance (already on `device`)
    cfg    : TrainConfig
    device : torch.device

    Public attributes
    ─────────────────
    optimizer : torch.optim.Adam
    scaler    : torch.cuda.amp.GradScaler

    Usage
    ─────
        trainer = PPOTrainer(policy, cfg, device)
        stats   = trainer.update(processed_batch, batch_processor)
        # stats: {'p_loss': float, 'v_loss': float, 'entropy': float}
    """

    def __init__(
        self,
        policy: nn.Module,
        cfg:    TrainConfig,
        device: torch.device,
    ) -> None:
        self.policy = policy
        self.cfg    = cfg
        self.device = device

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        self.scaler    = torch.cuda.amp.GradScaler(
            enabled=(cfg.use_amp and device.type == "cuda")
        )

    # ── Context helpers ───────────────────────────────────────────────────────

    def _autocast(self):
        """Return the appropriate forward-pass context manager."""
        if self.cfg.use_amp and self.device.type == "cuda":
            return torch.amp.autocast(device_type=self.device.type)
        return _NullContext()

    # ── Single minibatch gradient step ────────────────────────────────────────

    def _step(self, minibatch: dict) -> tuple:
        """
        Execute one forward + backward pass on a single minibatch.

        The tensors in `minibatch` (log_old, adv, ret_norm) arrive on CPU
        from the generator.  They are moved to self.device here and deleted
        immediately after the scalar values are captured, keeping GPU memory
        usage proportional to a single minibatch rather than the full batch.

        Parameters
        ──────────
        minibatch : dict — keys: snaps, acts, masks, log_old, adv, ret_norm
                           (log_old / adv / ret_norm are CPU float32 tensors)

        Returns
        ───────
        (p_loss_item, v_loss_item, ent_item) — Python floats, fully detached
        """
        device = self.device
        cfg    = self.cfg

        # Move per-minibatch tensors to the compute device
        log_old  = minibatch["log_old"].to(device)    # (mb,)
        adv      = minibatch["adv"].to(device)         # (mb,)
        ret_norm = minibatch["ret_norm"].to(device)    # (mb,)

        with self._autocast():

            new_lp, new_ent, new_val = self.policy.evaluate_actions(
                minibatch["snaps"],
                minibatch["acts"],
                minibatch["masks"],
            )

            ratio = torch.exp(new_lp - log_old.detach())   # (mb,)

            clipped_ratio = ratio.clamp(
                1.0 - cfg.clip_eps,
                1.0 + cfg.clip_eps,
            )
            loss_clip = torch.min(ratio * adv, clipped_ratio * adv).mean()

            loss_val = F.mse_loss(new_val.squeeze(-1), ret_norm)

            loss_ent = new_ent.mean()

            loss = -loss_clip + cfg.vf_coef * loss_val - cfg.ent_coef * loss_ent

        # ── Backward + optimiser step ─────────────────────────────────────────
        self.optimizer.zero_grad(set_to_none=True)   # set_to_none saves memory

        if cfg.use_amp and device.type == "cuda":
            # AMP path: scale → backward → unscale → clip → step → update
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)   # must unscale before clipping
            nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard path
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            self.optimizer.step()

        # ── Capture scalars BEFORE releasing tensors ──────────────────────────
        p_item   = loss_clip.item()
        v_item   = loss_val.item()
        ent_item = loss_ent.item()

        # ── Explicit GPU tensor cleanup ───────────────────────────────────────
        # Deleting here (not at end of update) keeps peak VRAM flat across
        # minibatches: the allocator can reuse these blocks for the next mb.
        del log_old, adv, ret_norm
        del new_lp, new_ent, new_val
        del ratio, clipped_ratio
        del loss_clip, loss_val, loss_ent, loss

        return p_item, v_item, ent_item

    # ── Full update (n_epochs × minibatches) ─────────────────────────────────

    def update(
        self,
        processed_batch: dict,
        batch_processor: "BatchProcessor",
    ) -> dict:
        """
        Run cfg.n_epochs of PPO updates.

        For each epoch, batch_processor.minibatch_generator() is called with
        cfg.train_fraction to draw a fresh random subset of the batch.
        This means:
          • Each epoch sees a different random subset (if fraction < 1.0).
          • The full shuffle guarantees that no systematic bias is introduced.
          • Over multiple epochs the full batch is approximately covered.

        Parameters
        ──────────
        processed_batch : dict         — output of BatchProcessor.process()
        batch_processor : BatchProcessor — used to generate minibatches

        Returns
        ───────
        stats : dict
            p_loss  float — mean clipped surrogate loss (all epochs, all mb)
            v_loss  float — mean value loss
            entropy float — mean entropy
        """
        cfg = self.cfg

        pl_log: list = []
        vl_log: list = []
        el_log: list = []

        # Estimate total minibatches for the progress bar
        n_train_per_epoch = max(
            cfg.minibatch_size,
            int(cfg.batch_size * cfg.train_fraction),
        )
        n_mb_per_epoch = n_train_per_epoch // cfg.minibatch_size
        total_mb       = cfg.n_epochs * n_mb_per_epoch

        pbar = tqdm(
            total=total_mb,
            desc="  PPO epochs × minibatches",
            leave=False,
            unit="mb",
        )

        self.policy.train()

        for epoch in range(cfg.n_epochs):
            # Fresh permutation + fraction selection each epoch
            for minibatch in batch_processor.minibatch_generator(
                processed_batch,
                train_fraction=cfg.train_fraction,
            ):
                p_item, v_item, ent_item = self._step(minibatch)

                pl_log.append(p_item)
                vl_log.append(v_item)
                el_log.append(ent_item)

                pbar.set_postfix(
                    epoch  = epoch + 1,
                    p_loss = f"{p_item:.4f}",
                    v_loss = f"{v_item:.4f}",
                )
                pbar.update(1)

        pbar.close()

        # ── Compute stats before releasing the log lists ──────────────────────
        stats = {
            "p_loss":  float(np.mean(pl_log)),
            "v_loss":  float(np.mean(vl_log)),
            "entropy": float(np.mean(el_log)),
        }

        # ── Post-update memory cleanup ────────────────────────────────────────
        del pl_log, vl_log, el_log
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return stats





