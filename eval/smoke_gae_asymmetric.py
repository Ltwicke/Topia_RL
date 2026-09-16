"""
smoke_gae_asymmetric.py  (now: symmetric GAE + per-epoch recompute)
───────────────────────────────────────────────────────────────────
The asymmetric winner/loser λ was reverted to a single symmetric λ. This
smoke test verifies:

  1. The reverted `compute_gae_per_player` matches a fresh reference
     symmetric-GAE implementation to float epsilon.
  2. `BatchProcessor.recompute_advantages` is idempotent: when handed the
     collection values it reproduces the advantages of a direct
     `compute_gae_per_player` (filtered to the active seat).

Run from project root:
  python eval/smoke_gae_asymmetric.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RL.ppo.batch_processing import compute_gae_per_player, BatchProcessor
from RL.ppo.game_manager import TrainConfig


# ── Reference symmetric GAE (independent re-implementation) ─────────────────
def reference_gae(rewards, values, dones, last_values, player_ids,
                  gamma, gae_lam, n_players=2):
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    for p in range(n_players):
        for e in range(N):
            p_steps = np.nonzero(player_ids[:, e] == p)[0]
            if p_steps.size == 0:
                continue
            k = p_steps.size
            next_vals = np.empty(k, dtype=np.float32)
            if k > 1:
                next_vals[:-1] = values[p_steps[1:], e]
            last_t        = p_steps[-1]
            next_vals[-1] = 0.0 if dones[last_t, e] > 0.5 else last_values[e]
            p_not_done = 1.0 - dones[p_steps, e]
            p_rewards  = rewards[p_steps, e]
            p_values   = values[p_steps, e]
            gae = 0.0
            for i in range(k - 1, -1, -1):
                delta = p_rewards[i] + gamma * next_vals[i] * p_not_done[i] - p_values[i]
                gae = delta + gamma * gae_lam * p_not_done[i] * gae
                advantages[p_steps[i], e] = gae
    return advantages, advantages + values


def make_fake_batch(T=20, N=2, done_t=15):
    rng = np.random.default_rng(0)
    rewards     = rng.standard_normal((T, N)).astype(np.float32)
    values      = rng.standard_normal((T, N)).astype(np.float32)
    dones       = np.zeros((T, N), dtype=np.float32)
    last_values = rng.standard_normal(N).astype(np.float32)
    player_ids  = np.zeros((T, N), dtype=np.int32)
    is_active   = np.zeros((T, N), dtype=np.float32)
    for e in range(N):
        for t in range(T):
            player_ids[t, e] = t % 2
            is_active[t, e]  = 1.0 if (t % 2) == 0 else 0.0   # active seat = player 0
        # one game cut per seat (winner step + loser back-fill)
        dones[done_t, e]     = 1.0
        dones[done_t - 1, e] = 1.0
    return dict(rewards=rewards, values=values, dones=dones,
                last_values=last_values, player_ids=player_ids,
                is_active=is_active)


def test_matches_reference():
    b = make_fake_batch()
    gamma, lam = 0.999, 0.99
    adv_new, ret_new = compute_gae_per_player(
        b["rewards"], b["values"], b["dones"], b["last_values"],
        b["player_ids"], gamma=gamma, gae_lam=lam,
    )
    adv_ref, ret_ref = reference_gae(
        b["rewards"], b["values"], b["dones"], b["last_values"],
        b["player_ids"], gamma=gamma, gae_lam=lam,
    )
    da = np.abs(adv_new - adv_ref).max()
    dr = np.abs(ret_new - ret_ref).max()
    assert da < 1e-5 and dr < 1e-5, f"diverged: adv {da:.2e} ret {dr:.2e}"
    print(f"  [symmetric] adv max|diff|={da:.2e}  ret max|diff|={dr:.2e}  OK")


def test_recompute_idempotent():
    b = make_fake_batch()
    cfg = TrainConfig()
    gamma, lam = cfg.gamma, cfg.gae_lambda
    T, N = b["rewards"].shape

    # Direct GAE, filtered to active seat (the expected adv)
    adv, _ = compute_gae_per_player(
        b["rewards"], b["values"], b["dones"], b["last_values"],
        b["player_ids"], gamma=gamma, gae_lam=lam,
    )
    is_active_flat = b["is_active"].reshape(-1) > 0.5
    active_idx     = np.nonzero(is_active_flat)[0]
    expected_adv   = adv.reshape(-1)[active_idx]

    # Minimal processed_batch carrying just the recompute inputs
    bp = BatchProcessor(cfg)
    processed = {
        "rewards_TN":             b["rewards"],
        "dones_TN":               b["dones"],
        "player_ids_TN":          b["player_ids"],
        "last_values_N":          b["last_values"],
        "values_flat_collection": b["values"].reshape(-1).astype(np.float32),
        "active_idx":             active_idx,
        "T": T, "N": N,
    }
    # Feed the collection values as the "fresh" values → must reproduce adv
    fresh_active_vals = b["values"].reshape(-1)[active_idx].astype(np.float32)
    adv_re, _ = bp.recompute_advantages(processed, fresh_active_vals)

    d = np.abs(adv_re - expected_adv).max()
    assert d < 1e-5, f"recompute not idempotent: max|diff|={d:.2e}"
    print(f"  [recompute] idempotent with collection values: max|diff|={d:.2e}  OK")


def main():
    print("test_matches_reference ...")
    test_matches_reference()
    print("test_recompute_idempotent ...")
    test_recompute_idempotent()
    print("\nAll GAE tests passed.")


if __name__ == "__main__":
    main()
