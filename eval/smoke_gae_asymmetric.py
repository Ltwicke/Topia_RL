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
from RL.models.main_modules import V_TERM, V_DENSE, N_VALUE_STREAMS


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
            next_vals[-1] = 0.0 if dones[last_t, e] > 0.5 else last_values[e, p]
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
    # Two independent reward streams, each with its own value channel.
    rew_term    = rng.standard_normal((T, N)).astype(np.float32)
    rew_dense   = rng.standard_normal((T, N)).astype(np.float32)
    values      = rng.standard_normal((T, N, N_VALUE_STREAMS)).astype(np.float32)
    dones       = np.zeros((T, N), dtype=np.float32)
    # Bootstrap per env, per seat, per stream.
    last_values = rng.standard_normal(
        (N, 2, N_VALUE_STREAMS)).astype(np.float32)
    player_ids  = np.zeros((T, N), dtype=np.int32)
    is_active   = np.zeros((T, N), dtype=np.float32)
    is_forced   = np.zeros((T, N), dtype=np.float32)
    for e in range(N):
        for t in range(T):
            player_ids[t, e] = t % 2
            is_active[t, e]  = 1.0 if (t % 2) == 0 else 0.0   # active seat = player 0
        # one game cut per seat (winner step + loser back-fill)
        dones[done_t, e]     = 1.0
        dones[done_t - 1, e] = 1.0
    return dict(rew_term=rew_term, rew_dense=rew_dense, values=values,
                dones=dones, last_values=last_values, player_ids=player_ids,
                is_active=is_active, is_forced=is_forced)


def test_matches_reference():
    b = make_fake_batch()
    gamma, lam = 0.999, 0.99
    adv_new, ret_new = compute_gae_per_player(
        b["rew_term"], b["values"][:, :, V_TERM], b["dones"],
        b["last_values"][:, :, V_TERM], b["player_ids"], gamma=gamma, gae_lam=lam,
    )
    adv_ref, ret_ref = reference_gae(
        b["rew_term"], b["values"][:, :, V_TERM], b["dones"],
        b["last_values"][:, :, V_TERM], b["player_ids"], gamma=gamma, gae_lam=lam,
    )
    da = np.abs(adv_new - adv_ref).max()
    dr = np.abs(ret_new - ret_ref).max()
    assert da < 1e-5 and dr < 1e-5, f"diverged: adv {da:.2e} ret {dr:.2e}"
    print(f"  [symmetric] adv max|diff|={da:.2e}  ret max|diff|={dr:.2e}  OK")


def test_stream_decomposition_is_exact():
    """
    The load-bearing property of the two-headed critic: per-stream GAE summed
    with beta=1 must equal single-stream GAE on the summed reward, exactly.

    If this drifts, `dense_beta` stops being a clean dial and switching dense
    off silently changes the terminal advantages too — the whole point of the
    decomposition is that it does not.
    """
    b = make_fake_batch()
    gamma, lam = 1.0, 0.95

    adv_t, _ = compute_gae_per_player(
        b["rew_term"], b["values"][:, :, V_TERM], b["dones"],
        b["last_values"][:, :, V_TERM], b["player_ids"], gamma=gamma, gae_lam=lam)
    adv_d, _ = compute_gae_per_player(
        b["rew_dense"], b["values"][:, :, V_DENSE], b["dones"],
        b["last_values"][:, :, V_DENSE], b["player_ids"], gamma=gamma, gae_lam=lam)

    # Single-stream reference on the combined reward and combined value.
    adv_c, _ = compute_gae_per_player(
        b["rew_term"] + b["rew_dense"],
        b["values"][:, :, V_TERM] + b["values"][:, :, V_DENSE],
        b["dones"],
        b["last_values"][:, :, V_TERM] + b["last_values"][:, :, V_DENSE],
        b["player_ids"], gamma=gamma, gae_lam=lam)

    d = np.abs((adv_t + adv_d) - adv_c).max()
    assert d < 1e-5, f"stream decomposition is not exact: max|diff|={d:.2e}"

    # beta=0 must leave exactly the terminal advantages, untouched by dense.
    d0 = np.abs((adv_t + 0.0 * adv_d) - adv_t).max()
    assert d0 == 0.0
    print(f"  [decomposition] A_term + A_dense == A_combined: max|diff|={d:.2e}  OK")


def test_dense_switch_off_leaves_terminal_untouched():
    """
    The guarantee the whole two-headed design exists for: setting dense_beta
    to 0 mid-training must change the POLICY objective without disturbing
    anything the terminal value head has learned.

    Runs the real BatchProcessor.process() twice on one batch, once at beta=1
    and once at beta=0, and checks that:
      • the terminal return targets are bit-identical
      • the advantage difference is exactly beta * A_dense, nothing else
    If the two streams shared a normaliser this would fail, because dropping
    dense would shift the shared statistics under the terminal targets.
    """
    import dataclasses

    T, N = 20, 2
    b = make_fake_batch(T=T, N=N)
    B = T * N
    raw = {
        **b,
        "obs_snaps": [[None] * N for _ in range(T)],
        "actions":   [[None] * N for _ in range(T)],
        "masks":     [[None] * N for _ in range(T)],
        "log_probs": np.zeros((T, N), dtype=np.float32),
        "n_games": 2, "n_active_wins": 1, "n_conquest": 0, "n_timeout": 2,
        "n_dropped_terminal": 0, "n_endturn": 10,
        "n_endturn_voluntary": 3, "n_decisions_total": 20,
    }

    cfg_on  = dataclasses.replace(TrainConfig(), dense_beta=1.0)
    cfg_off = dataclasses.replace(TrainConfig(), dense_beta=0.0)
    p_on,  _ = BatchProcessor(cfg_on).process(raw)
    p_off, _ = BatchProcessor(cfg_off).process(raw)

    # The terminal head's regression targets must not move at all.
    dt = np.abs(p_on["ret_norm_np"] - p_off["ret_norm_np"]).max()
    assert dt == 0.0, (
        f"terminal return targets shifted when dense was switched off "
        f"(max|diff|={dt:.2e}) — the streams are coupled somewhere"
    )

    # The advantage must differ by exactly the dense contribution.
    adv_dense_only = p_on["adv_np"] - p_off["adv_np"]
    assert np.abs(adv_dense_only).max() > 0.0, "beta had no effect at all"
    p_half, _ = BatchProcessor(
        dataclasses.replace(TrainConfig(), dense_beta=0.5)).process(raw)
    expected_half = p_off["adv_np"] + 0.5 * adv_dense_only
    dh = np.abs(p_half["adv_np"] - expected_half).max()
    assert dh < 1e-5, f"advantage is not linear in dense_beta: max|diff|={dh:.2e}"
    print(f"  [beta switch] terminal targets identical (diff={dt:.1e}), "
          f"advantage linear in beta (diff={dh:.1e})  OK")


def test_recompute_idempotent():
    b = make_fake_batch()
    cfg = TrainConfig()
    gamma, lam = cfg.gamma, cfg.gae_lambda
    T, N = b["rew_term"].shape

    # Direct per-stream GAE, combined the way process() does it
    adv_t, _ = compute_gae_per_player(
        b["rew_term"], b["values"][:, :, V_TERM], b["dones"],
        b["last_values"][:, :, V_TERM], b["player_ids"], gamma=gamma, gae_lam=lam)
    adv_d, _ = compute_gae_per_player(
        b["rew_dense"], b["values"][:, :, V_DENSE], b["dones"],
        b["last_values"][:, :, V_DENSE], b["player_ids"], gamma=gamma, gae_lam=lam)
    adv = adv_t + cfg.dense_beta * adv_d

    is_active_flat = b["is_active"].reshape(-1) > 0.5
    active_idx     = np.nonzero(is_active_flat)[0]
    expected_adv   = adv.reshape(-1)[active_idx]

    # Minimal processed_batch carrying just the recompute inputs
    bp = BatchProcessor(cfg)
    val_flat = b["values"].reshape(-1, N_VALUE_STREAMS).astype(np.float32)
    processed = {
        "rew_term_TN":            b["rew_term"],
        "rew_dense_TN":           b["rew_dense"],
        "dones_TN":               b["dones"],
        "player_ids_TN":          b["player_ids"],
        "last_values_N":          b["last_values"],
        "values_flat_collection": val_flat,
        "active_idx":             active_idx,
        "T": T, "N": N,
    }
    # Feed the collection values as the "fresh" values → must reproduce adv
    adv_re, _, _ = bp.recompute_advantages(processed, val_flat[active_idx])

    d = np.abs(adv_re - expected_adv).max()
    assert d < 1e-5, f"recompute not idempotent: max|diff|={d:.2e}"
    print(f"  [recompute] idempotent with collection values: max|diff|={d:.2e}  OK")


def test_opposing_terminal_rewards_give_opposing_advantages():
    """
    A real invariant, not a re-implementation check.

    Build a zero-reward game whose two seats receive equal and opposite
    terminal payouts (+z and -z) on their own last decisions, both cut with
    done=1. With a symmetric (all-zero) critic, the two seats' terminal
    advantages must come out exact opposites.

    The live reward scheme is constant-sum rather than zero-sum, so this is a
    test of the GAE machinery on synthetic input, not of the payout formula.

    This is what catches a wrong seat attribution or a wrong-perspective
    bootstrap — test_matches_reference cannot, because it compares the
    implementation against a copy of the same formula.
    """
    T, N, z = 12, 1, 0.75
    rewards    = np.zeros((T, N), dtype=np.float32)
    values     = np.zeros((T, N), dtype=np.float32)
    dones      = np.zeros((T, N), dtype=np.float32)
    last_values = np.zeros((N, 2), dtype=np.float32)
    player_ids = np.array([[t % 2] for t in range(T)], dtype=np.int32)

    p0_last, p1_last = T - 2, T - 1        # seat 0 acts on even t, seat 1 on odd
    rewards[p0_last, 0] = +z
    rewards[p1_last, 0] = -z
    dones[p0_last, 0] = dones[p1_last, 0] = 1.0

    adv, ret = compute_gae_per_player(
        rewards, values, dones, last_values, player_ids,
        gamma=1.0, gae_lam=0.95,
    )

    a0, a1 = adv[p0_last, 0], adv[p1_last, 0]
    assert abs(a0 + a1) < 1e-6, (
        f"terminal advantages are not antisymmetric: {a0:+.6f} and {a1:+.6f} "
        f"(sum {a0 + a1:+.6f}) — seat attribution or bootstrap perspective is wrong"
    )
    assert a0 > 0 > a1, f"winner/loser advantage signs inverted: {a0:+.4f} {a1:+.4f}"

    # With gamma=1 the credit must reach the opening at full strength, decayed
    # only by lambda — the property that removes the rush-to-timeout incentive.
    first0 = adv[0, 0]
    assert first0 > 0.0, f"winner's opening advantage not positive: {first0:+.4f}"
    print(f"  [opposing] terminal adv {a0:+.4f} / {a1:+.4f} (sum {a0+a1:+.1e}), "
          f"opening adv {first0:+.4f}  OK")


def test_done_cuts_cross_episode_leak():
    """Two games in one column: the first game's cut must block credit flowing
    backwards from the second game into the first."""
    T, N = 12, 1
    rewards    = np.zeros((T, N), dtype=np.float32)
    values     = np.zeros((T, N), dtype=np.float32)
    dones      = np.zeros((T, N), dtype=np.float32)
    last_values = np.zeros((N, 2), dtype=np.float32)
    player_ids = np.array([[t % 2] for t in range(T)], dtype=np.int32)

    # Game 1 ends at t=4/5; game 2 pays seat 0 a big reward at t=10.
    dones[4, 0] = dones[5, 0] = 1.0
    rewards[10, 0] = 100.0

    adv, _ = compute_gae_per_player(
        rewards, values, dones, last_values, player_ids,
        gamma=1.0, gae_lam=1.0,
    )
    leaked = np.abs(adv[:5, 0]).max()
    assert leaked < 1e-6, (
        f"game 2's reward leaked back into game 1: max|adv|={leaked:.4f}"
    )
    assert adv[6, 0] > 0.0, "credit did not reach game 2's earlier steps"
    print(f"  [episode cut] pre-cut max|adv|={leaked:.1e}  OK")


def main():
    print("test_matches_reference ...")
    test_matches_reference()
    print("test_stream_decomposition_is_exact ...")
    test_stream_decomposition_is_exact()
    print("test_dense_switch_off_leaves_terminal_untouched ...")
    test_dense_switch_off_leaves_terminal_untouched()
    print("test_recompute_idempotent ...")
    test_recompute_idempotent()
    print("test_opposing_terminal_rewards_give_opposing_advantages ...")
    test_opposing_terminal_rewards_give_opposing_advantages()
    print("test_done_cuts_cross_episode_leak ...")
    test_done_cuts_cross_episode_leak()
    print("\nAll GAE tests passed.")


if __name__ == "__main__":
    main()
