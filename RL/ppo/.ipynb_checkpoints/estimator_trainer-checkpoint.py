"""
ppo/estimator_trainer.py
──────────────────────────────────────────────────────────────────────────────
Phase A of the dual-objective training update — supervised pretraining of
the HiddenTileEstimator.

Owns its own Adam optimiser over (encoder.* + hidden_estimator.*).  The PPO
trainer's optimiser is unchanged; the two optimisers share encoder
parameters but maintain independent moment estimates ("Separate Adam for
estimator" per the user-locked design choice).

Per-update flow
───────────────
1. EnvManager.collect()              produces raw_batch with obs_snaps
2. EstimatorPretrainer.update(batch) runs cfg.estimator_n_epochs over
                                     EstimatorBatchProcessor.minibatch_generator,
                                     stepping its own optimiser per minibatch.
3. PPOTrainer.update(...)            runs the standard PPO update, untouched.

Loss
────
    L = mean_b ( sum_t∈hidden(b) sum_g group_loss_g(t) / n_hidden(b) )

where group_loss_g is the cross-entropy or BCE for the g-th feature group
(road, player-ctrl, city, unit-state, P0/P1 unit-type) computed by
``HiddenTileEstimator.loss``.  Per-sample normalisation is done in
``PolicyNetwork.estimator_loss`` (see [RL/models/policy.py]); the inner
``HiddenTileEstimator.loss`` reduction is ``'sum'`` so the explicit divide
by ``n_hidden`` outside gives the intended per-tile-averaged semantic.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ppo.game_manager import TrainConfig
from ppo.ppo          import _NullContext

if TYPE_CHECKING:
    from ppo.batch_processing import EstimatorBatchProcessor


# ══════════════════════════════════════════════════════════════════════════════
# EstimatorPretrainer
# ══════════════════════════════════════════════════════════════════════════════

class EstimatorPretrainer:
    """
    Pretrains the HiddenTileEstimator on the most recent rollout.

    Parameters
    ──────────
    policy : nn.Module       — PolicyNetwork instance (already on `device`)
    cfg    : TrainConfig
    device : torch.device

    Public attributes
    ─────────────────
    optimizer : torch.optim.Adam   — Adam over encoder + hidden_estimator
    scaler    : torch.cuda.amp.GradScaler
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

        # Encoder + estimator only — heads + critic are not on this loss path.
        self._params = (
            list(policy.encoder.parameters())
            + list(policy.hidden_estimator.parameters())
        )
        self.optimizer = torch.optim.Adam(self._params, lr=cfg.estimator_lr)
        self.scaler    = torch.cuda.amp.GradScaler(
            enabled=(cfg.use_amp and device.type == "cuda")
        )

    # ── Context helpers ───────────────────────────────────────────────────────

    def _autocast(self):
        if self.cfg.use_amp and self.device.type == "cuda":
            return torch.amp.autocast(device_type=self.device.type)
        return _NullContext()

    # ── Single minibatch gradient step ────────────────────────────────────────

    def _step(self, snaps: list) -> float:
        """Forward + backward on a single estimator minibatch.

        Returns the scalar loss (Python float).
        """
        self.optimizer.zero_grad(set_to_none=True)

        with self._autocast():
            loss = self.policy.estimator_loss(snaps)

        if self.cfg.use_amp and self.device.type == "cuda":
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self._params, self.cfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self._params, self.cfg.max_grad_norm)
            self.optimizer.step()

        item = loss.item()
        del loss
        return item

    # ── Full pretraining update (n_epochs × minibatches) ──────────────────────

    def update(
        self,
        raw_batch:     dict,
        est_batch_proc: "EstimatorBatchProcessor",
    ) -> dict:
        """
        Run cfg.estimator_n_epochs over the rollout, one gradient step per
        minibatch.  Each epoch redraws the random permutation so subsequent
        epochs see different sample orderings.

        Parameters
        ──────────
        raw_batch      : dict — output of EnvManager.collect()
        est_batch_proc : EstimatorBatchProcessor

        Returns
        ───────
        stats : dict
            est_loss : float — mean per-minibatch loss across all epochs
            n_steps  : int   — number of gradient steps taken
        """
        cfg = self.cfg

        # ── Estimate total minibatches for the progress bar ───────────────────
        flat_n  = sum(len(step) for step in raw_batch["obs_snaps"])
        n_train = max(
            cfg.estimator_minibatch_size,
            int(flat_n * cfg.estimator_train_fraction),
        )
        mb_per_epoch = n_train // cfg.estimator_minibatch_size
        total_mb     = cfg.estimator_n_epochs * mb_per_epoch

        pbar = tqdm(
            total=total_mb,
            desc="  Estimator epochs × minibatches",
            leave=False,
            unit="mb",
        )

        self.policy.train()

        losses: list = []
        for epoch in range(cfg.estimator_n_epochs):
            for snaps in est_batch_proc.minibatch_generator(raw_batch):
                item = self._step(snaps)
                losses.append(item)
                pbar.set_postfix(epoch=epoch + 1, est_loss=f"{item:.4f}")
                pbar.update(1)

        pbar.close()

        stats = {
            "est_loss": float(np.mean(losses)) if losses else 0.0,
            "n_steps":  len(losses),
        }

        # ── Post-update memory cleanup ────────────────────────────────────────
        del losses
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return stats
