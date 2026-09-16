"""
smoke_train_one_update.py
──────────────────────────
End-to-end integration of ONE training update with the real multiprocessing
pipeline, on a tiny config. Exercises:

  • EnvManager two-state-dict distribute(active, frozen) + collect()
  • worker is_active / scalar-counter path
  • EstimatorPretrainer.update (unchanged, both seats)
  • BatchProcessor.process active-seat filtering
  • PPOTrainer.update with per-epoch GAE recompute (compute_values_batch)
  • rolling + permanent checkpoint save

Run from project root:
  python eval/smoke_train_one_update.py
"""

from __future__ import annotations

import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    # RL/ on path so the inner `from ppo...`, `from models...` imports resolve
    sys.path.insert(0, str(_PROJECT_ROOT / "RL"))

    from RL.ppo.game_manager      import TrainConfig, EnvManager
    from RL.ppo.batch_processing  import BatchProcessor, EstimatorBatchProcessor
    from RL.ppo.ppo               import PPOTrainer
    from RL.ppo.estimator_trainer import EstimatorPretrainer
    from RL.models.policy         import PolicyNetwork
    import RL.train as train_mod

    cfg = TrainConfig()
    # ── Tiny, fast overrides ──
    cfg.pretrained_ckpt    = ""        # from scratch
    cfg.start_update       = 0
    cfg.n_updates          = 2
    cfg.n_processes        = 2
    cfg.n_envs_per_process = 2
    cfg.n_steps            = 64
    cfg.n_epochs           = 2
    cfg.n_minibatches      = 4
    cfg.train_fraction     = 0.5
    cfg.estimator_n_epochs = 1
    cfg.estimator_minibatch_size = 64
    cfg.max_turns_per_game = 6
    cfg.board_size_range   = (11, 11)
    cfg.scenario_eval_interval = 0     # skip scenarios
    cfg.opponent_refresh_interval = 1  # exercise refresh
    cfg.permanent_ckpt_interval   = 1  # exercise permanent save
    cfg.use_amp            = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    policy = PolicyNetwork(cfg).to(device)
    policy.train()

    env_manager    = EnvManager(cfg)
    ppo_batch_proc = BatchProcessor(cfg)
    est_batch_proc = EstimatorBatchProcessor(cfg)
    ppo_trainer    = PPOTrainer(policy, cfg, device)
    est_pretrainer = EstimatorPretrainer(policy, cfg, device)

    # Point checkpoint dir at a temp folder so we don't pollute the repo
    tmp_ckpt = tempfile.mkdtemp(prefix="smoke_ckpt_")
    train_mod.CKPT_DIR = tmp_ckpt

    env_manager.start()
    print(f"spawned {cfg.n_processes} workers")
    ckpt_queue: deque = deque()

    frozen_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

    try:
        for update in range(cfg.n_updates):
            cpu_state = {k: v.cpu() for k, v in policy.state_dict().items()}
            t_dist = env_manager.distribute(cpu_state, frozen_state)
            del cpu_state

            raw_batch, t_collect = env_manager.collect()
            print(f"[u{update}] collect {t_collect:.2f}s  "
                  f"n_games={raw_batch['n_games']}  "
                  f"active_wins={raw_batch['n_active_wins']}  "
                  f"conquest={raw_batch['n_conquest']}  "
                  f"timeout={raw_batch['n_timeout']}")

            # is_active sanity vs player_ids == active seat
            ia = raw_batch["is_active"]
            pid = raw_batch["player_ids"]
            assert np.array_equal(ia > 0.5, pid == cfg.active_player_id), \
                "is_active mask disagrees with player_ids"

            est_stats = est_pretrainer.update(raw_batch, est_batch_proc)
            assert np.isfinite(est_stats["est_loss"]), "est_loss not finite"

            processed_batch, t_gae = ppo_batch_proc.process(raw_batch)
            n_active = processed_batch["adv_np"].shape[0]
            total    = raw_batch["rewards"].size
            print(f"[u{update}] active samples = {n_active} / {total} "
                  f"(~{100*n_active/total:.0f}%)  gae {t_gae:.3f}s")
            assert 0 < n_active < total, "active filter produced degenerate count"

            del raw_batch
            stats = ppo_trainer.update(processed_batch, ppo_batch_proc)
            assert np.isfinite(stats["p_loss"]), "p_loss not finite"
            assert np.isfinite(stats["v_loss"]), "v_loss not finite"
            assert np.isfinite(stats["entropy"]), "entropy not finite"
            print(f"[u{update}] p_loss={stats['p_loss']:.4f}  "
                  f"v_loss={stats['v_loss']:.4f}  entropy={stats['entropy']:.4f}")

            # Refresh frozen + checkpoints
            frozen_state = {k: v.detach().cpu().clone()
                            for k, v in policy.state_dict().items()}
            train_mod._save_checkpoint(policy, ppo_trainer, est_pretrainer,
                                       update, ckpt_queue, train_mod.logging.getLogger("smoke"))
            train_mod._save_checkpoint(policy, ppo_trainer, est_pretrainer,
                                       update, ckpt_queue, train_mod.logging.getLogger("smoke"),
                                       permanent=True)

            del processed_batch
    finally:
        env_manager.shutdown()

    perm = list(Path(tmp_ckpt).glob("policy_permanent_*.pt"))
    assert perm, "no permanent checkpoint written"
    print(f"permanent checkpoints: {[p.name for p in perm]}")
    print("\nEnd-to-end one-update smoke passed.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
