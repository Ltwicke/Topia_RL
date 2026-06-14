"""
smoke_gae_asymmetric.py
─────────────────────────
Sanity check for the new asymmetric-λ per-player GAE.

Tests:
  1. With lambda_winner == lambda_loser, output matches the reference
     symmetric-λ GAE (the prior implementation, copied below).
  2. With lambda_winner > lambda_loser, the loser-side advantage trace
     decays faster going back from the terminal step than the winner-side
     trace.
  3. Bootstrap behavior at done=1 is independent of λ (terminal advantage
     == delta).

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

from RL.ppo.batch_processing import compute_gae_per_player


# ── Reference implementation (the prior symmetric-λ version) ───────────────
def reference_gae_symmetric(
    rewards, values, dones, last_values, player_ids, gamma, gae_lam,
    n_players=2,
):
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
            game_ended    = dones[last_t, e] > 0.5
            next_vals[-1] = 0.0 if game_ended else last_values[e]
            p_not_done = 1.0 - dones[p_steps, e]
            p_rewards  = rewards[p_steps, e]
            p_values   = values[p_steps, e]
            gae = 0.0
            for i in range(k - 1, -1, -1):
                delta = (
                    p_rewards[i]
                    + gamma * next_vals[i] * p_not_done[i]
                    - p_values[i]
                )
                gae = delta + gamma * gae_lam * p_not_done[i] * gae
                advantages[p_steps[i], e] = gae
    returns = advantages + values
    return advantages, returns


# ── Synthetic batch ────────────────────────────────────────────────────────
def make_fake_batch(T=20, N=2, done_t=15, winner_per_env=(0, 1)):
    """Two envs, alternating players, one game each, ending at done_t."""
    rewards     = np.zeros((T, N), dtype=np.float32)
    values      = np.full((T, N), 0.5, dtype=np.float32)
    dones       = np.zeros((T, N), dtype=np.float32)
    last_values = np.full((N,), 0.0, dtype=np.float32)
    player_ids  = np.zeros((T, N), dtype=np.int32)
    winners     = np.full((T, N), -1, dtype=np.int8)

    for e in range(N):
        for t in range(T):
            player_ids[t, e] = t % 2

        # Place a +1 reward at done_t (winner side) and -1 on opponent's
        # last decision step — simulates zero-sum terminal.
        winner = winner_per_env[e]
        loser  = 1 - winner
        # Find the step at done_t and the opp's last step before it
        # — we simply mark dones at done_t for the winner step, and at
        # the previous step for the loser (matching the worker back-fill).
        # done_t belongs to the winner if done_t % 2 == winner; otherwise
        # we shift by one. Keep it deterministic.
        # For simplicity, place dones at done_t (winner) and done_t-1 (loser).
        rewards[done_t, e]     = +100.0 if (done_t % 2) == winner else -100.0
        rewards[done_t - 1, e] = +100.0 if ((done_t - 1) % 2) == winner else -100.0

        dones[done_t, e]       = 1.0
        dones[done_t - 1, e]   = 1.0
        winners[done_t, e]     = winner
        winners[done_t - 1, e] = winner

    return dict(
        rewards=rewards,
        values=values,
        dones=dones,
        last_values=last_values,
        player_ids=player_ids,
        winners=winners,
    )


def test_symmetric_matches_reference():
    batch = make_fake_batch()
    gamma  = 0.99
    lam    = 0.95
    adv_new, ret_new = compute_gae_per_player(
        batch["rewards"], batch["values"], batch["dones"],
        batch["last_values"], batch["player_ids"], batch["winners"],
        gamma=gamma, lambda_winner=lam, lambda_loser=lam,
    )
    adv_ref, ret_ref = reference_gae_symmetric(
        batch["rewards"], batch["values"], batch["dones"],
        batch["last_values"], batch["player_ids"],
        gamma=gamma, gae_lam=lam,
    )
    diff_a = np.abs(adv_new - adv_ref).max()
    diff_r = np.abs(ret_new - ret_ref).max()
    assert diff_a < 1e-5, f"advantages diverge: max |diff|={diff_a}"
    assert diff_r < 1e-5, f"returns diverge: max |diff|={diff_r}"
    print(f"  [symmetric] adv max|diff| = {diff_a:.2e}  "
          f"ret max|diff| = {diff_r:.2e}  OK")


def test_asymmetric_loser_decays_faster():
    batch = make_fake_batch()
    gamma = 0.99
    lam_w, lam_l = 0.95, 0.5
    adv, _ = compute_gae_per_player(
        batch["rewards"], batch["values"], batch["dones"],
        batch["last_values"], batch["player_ids"], batch["winners"],
        gamma=gamma, lambda_winner=lam_w, lambda_loser=lam_l,
    )
    # In env 0 the winner is player 0. Winner steps: 0,2,4,...
    # In env 1 the winner is player 1. Winner steps: 1,3,5,...
    # Look at the early-game advantage magnitudes — losers should decay
    # toward zero faster than winners.
    pids = batch["player_ids"]
    for e in range(2):
        winner = (0, 1)[e]
        winner_steps = np.nonzero(pids[:14, e] == winner)[0]
        loser_steps  = np.nonzero(pids[:14, e] != winner)[0]
        # Compare the t=0/1 step magnitudes (furthest from terminal)
        early_w = abs(float(adv[winner_steps[0], e]))
        early_l = abs(float(adv[loser_steps[0], e]))
        print(f"  env {e}: winner={winner}  "
              f"|adv[early]|  winner-side={early_w:.4f}  loser-side={early_l:.4f}")
        assert early_l < early_w, (
            f"loser-side advantage at t=0 ({early_l}) should decay faster "
            f"than winner-side ({early_w}) when lambda_loser < lambda_winner"
        )


def test_terminal_invariance():
    """At the terminal step (done=1), advantage should equal `delta` and be
    independent of λ."""
    batch = make_fake_batch()
    gamma = 0.99
    adv1, _ = compute_gae_per_player(
        batch["rewards"], batch["values"], batch["dones"],
        batch["last_values"], batch["player_ids"], batch["winners"],
        gamma=gamma, lambda_winner=0.95, lambda_loser=0.95,
    )
    adv2, _ = compute_gae_per_player(
        batch["rewards"], batch["values"], batch["dones"],
        batch["last_values"], batch["player_ids"], batch["winners"],
        gamma=gamma, lambda_winner=0.95, lambda_loser=0.10,
    )
    dones = batch["dones"]
    for e in range(dones.shape[1]):
        for t in np.nonzero(dones[:, e] > 0.5)[0]:
            # Should be identical at done steps regardless of λ
            d = abs(float(adv1[t, e] - adv2[t, e]))
            assert d < 1e-5, (
                f"env={e} t={t}: terminal adv differs by {d} between λ "
                f"settings (should be independent)"
            )
    print("  [terminal] advantage at done steps is lambda-invariant  OK")


def main():
    print("test_symmetric_matches_reference …")
    test_symmetric_matches_reference()
    print("test_asymmetric_loser_decays_faster …")
    test_asymmetric_loser_decays_faster()
    print("test_terminal_invariance …")
    test_terminal_invariance()
    print()
    print("All GAE tests passed.")


if __name__ == "__main__":
    main()
