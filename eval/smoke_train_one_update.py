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

            # ── Terminal-share delivery invariants (sc-41) ─────────────────
            # A "dropped" share is a game whose loser last acted in an
            # already-shipped chunk. That trajectory was cut by the rollout
            # horizon and bootstrapped rather than terminated, which is normal
            # PPO truncation — but it must stay rare, so it is counted and the
            # two invariants below are stated exactly in terms of it.
            rew_t  = raw_batch["rew_term"]
            rew_d  = raw_batch["rew_dense"]
            dn     = raw_batch["dones"]
            n_games = raw_batch["n_games"]
            n_drop  = raw_batch["n_dropped_terminal"]

            assert n_drop <= n_games, "more drops than games — counter is wrong"
            drop_rate = n_drop / max(n_games, 1)
            print(f"[u{update}] terminal shares dropped: {n_drop}/{n_games} "
                  f"({drop_rate:.0%})")
            assert drop_rate <= 0.34, (
                f"terminal shares dropped at {drop_rate:.0%} of games — far "
                f"above the chunk-boundary rate; delivery is broken"
            )

            # Every finished game cuts BOTH seats' trajectories exactly once
            # (the actor's own step + the loser's back-fill), minus the drops.
            expected_cuts = 2 * n_games - n_drop
            assert int(dn.sum()) == expected_cuts, (
                f"trajectory cuts = {int(dn.sum())}, expected "
                f"{expected_cuts} (= 2×{n_games} games − {n_drop} dropped)"
            )

            # Constant-sum: each fully delivered game pays the two seats shares
            # totalling terminal_weight (or conquest_reward on a conquest), so
            # the terminal buffer sums to that per game, give or take the
            # undelivered shares.
            term_sum = float(rew_t.sum())
            max_per_game = cfg.conquest_reward
            assert -1e-4 <= term_sum <= n_games * max_per_game + 1e-4, (
                f"terminal stream sums to {term_sum:.4f}, outside [0, "
                f"{n_games * max_per_game:.4f}] for {n_games} games"
            )

            # The terminal stream must ONLY ever be non-zero on a trajectory cut
            # — that is what keeps dense shaping out of the terminal head.
            assert np.all(rew_t[dn <= 0.5] == 0.0), (
                "terminal reward on a step that is not a trajectory cut"
            )
            # Conversely the dense stream is per-action and must be zero when
            # dense shaping is disabled.
            if not cfg.dense_reward:
                assert np.all(rew_d == 0.0), (
                    "dense reward emitted with dense_reward=False"
                )

            est_stats = est_pretrainer.update(raw_batch, est_batch_proc)
            assert np.isfinite(est_stats["est_loss"]), "est_loss not finite"

            processed_batch, t_gae = ppo_batch_proc.process(raw_batch)
            n_active = processed_batch["adv_np"].shape[0]
            total    = raw_batch["rew_term"].size
            print(f"[u{update}] active samples = {n_active} / {total} "
                  f"(~{100*n_active/total:.0f}%)  "
                  f"forced dropped={processed_batch['n_forced_dropped']}  "
                  f"gae {t_gae:.3f}s")
            assert 0 < n_active < total, "active filter produced degenerate count"
            print(f"[u{update}] decisions/turn={processed_batch['decisions_per_turn']:.2f}  "
                  f"conquest_rate={processed_batch['conquest_rate']:.3f}  "
                  f"ev_term={processed_batch['ev_term']:+.3f}  "
                  f"ev_dense={processed_batch['ev_dense']:+.3f}")

            # Every training sample must be a real decision: forced steps carry
            # no gradient and only distort the entropy and whitening statistics.
            assert processed_batch["n_forced_dropped"] >= 0
            assert len(processed_batch["flat_snaps"]) == n_active, \
                "snapshot list and advantage array disagree after filtering"
            assert processed_batch["ret_dense_norm_np"].shape[0] == n_active, \
                "dense return targets misaligned with the training batch"

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
