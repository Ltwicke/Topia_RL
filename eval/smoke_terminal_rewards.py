"""
smoke_terminal_rewards.py
─────────────────────────
Sanity check for the constant-sum terminal reward path in env/wrapper.py.

Asserts per game:
  • the winner's share lands on the WINNER's seat, not the actor's (the sc-41
    regression: the seat was previously read back off game.player_go_id AFTER
    EndTurn had already swapped it)
  • the two seats' terminal shares sum to terminal_weight
  • no share is ever negative — the loser tends to 0 rather than being punished
  • a conquest strictly outranks the best possible timeout result
  • dense and terminal arrive on separate streams and dense stays off the
    terminal stream entirely

Run from project root:
  python eval/smoke_terminal_rewards.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env.wrapper import EnvWrapper
from game.enums import ActionTypes, BoardType, Tribes


def random_valid_action(env: EnvWrapper) -> list[int]:
    """Sample a uniformly-random valid action from the env's action mask."""
    masks = env.get_action_mask()
    type_mask = masks[0]
    valid_types = np.nonzero(type_mask)[0]
    atype = int(np.random.choice(valid_types))

    if atype == ActionTypes.MoveUnit:
        sub = masks[1]                              # (n_units, n_tiles)
        idxs = np.argwhere(sub > 0)
        u, tile = idxs[np.random.randint(len(idxs))]
        return [atype, int(u), int(tile)]

    if atype == ActionTypes.Attack:
        sub = masks[2]                              # (n_units, n_visible_enemies)
        idxs = np.argwhere(sub > 0)
        u, d = idxs[np.random.randint(len(idxs))]
        return [atype, int(u), int(d)]

    if atype == ActionTypes.CreateUnit:
        sub = masks[3]                              # (n_cities, n_unit_types)
        idxs = np.argwhere(sub > 0)
        c, ut = idxs[np.random.randint(len(idxs))]
        return [atype, int(c), int(ut)]

    if atype == ActionTypes.CaptureCity:
        sub = masks[4]
        idxs = np.nonzero(sub)[0]
        return [atype, int(np.random.choice(idxs))]

    if atype == ActionTypes.HealUnit:
        sub = masks[5]
        idxs = np.nonzero(sub)[0]
        return [atype, int(np.random.choice(idxs))]

    if atype == ActionTypes.UpgradeCity:
        sub = masks[6]                              # (n_cities, 2)
        idxs = np.argwhere(sub > 0)
        c, ch = idxs[np.random.randint(len(idxs))]
        return [atype, int(c), int(ch)]

    if atype == ActionTypes.PlaceRoad:
        sub = masks[7]
        idxs = np.nonzero(sub)[0]
        return [atype, int(np.random.choice(idxs))]

    if atype == ActionTypes.Upgrade2Vet:
        sub = masks[8]
        idxs = np.nonzero(sub)[0]
        return [atype, int(np.random.choice(idxs))]

    # EndTurn
    return [atype]


def play_one_game(seed: int, mode: str, dense: bool,
                  terminal_weight: float = 1.0,
                  terminal_tau: float = 2000.0,
                  conquest_reward: float = 3.0,
                  conquest_early_bonus: float = 0.5) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    board_config = {
        "board_size": [11, 11],
        "board_type": BoardType.Drylands,
        "n_players":  2,
    }
    env = EnvWrapper(
        board_config,
        [Tribes.Omaji, Tribes.Imperius],
        max_turns_per_game=12,
        dense_reward=dense,
        terminal_reward_mode=mode,
        terminal_weight=terminal_weight,
        terminal_tau=terminal_tau,
        conquest_reward=conquest_reward,
        conquest_early_bonus=conquest_early_bonus,
    )
    env.reset()

    dense_log: list[float] = []
    term_log:  list[float] = []
    n_steps = 0
    while True:
        a = random_valid_action(env)
        # Seat that is about to act. At a turn-limit timeout this is always
        # player 1 (the timeout fires on its EndTurn), while the score leader
        # may be either seat — which is exactly the inversion this guards.
        actor_id = env.game.player_go_id
        obs, rew, done, info = env.step(
            a, n_valid_action_types=int(env.get_action_mask()[0].sum()),
        )
        dense_log.append(float(info["r_dense"]))
        term_log.append(float(info["r_term"]))
        n_steps += 1
        if done:
            return {
                "n_steps":     n_steps,
                "dense":       dense_log,
                "term":        term_log,
                "winner_id":   info["winner_id"],
                "is_conquest": info["is_conquest"],
                "actor_id":    actor_id,
                "terminal_r":  float(info["r_term"]),
                "terminal_ro": float(info["r_term_opp"]),
            }
        if n_steps > 5000:
            raise RuntimeError("Game did not terminate — likely a bug")


def main() -> None:
    n_games = 24
    W, CONQ, EARLY = 1.0, 3.0, 0.5
    CONQ_MAX = CONQ * (1.0 + EARLY)

    # ── constant_sum: seat attribution, sum, non-negativity ───────────────
    # The attribution half is the sc-41 regression: _get_done_and_rewards used
    # to read the seat back off game.player_go_id AFTER apply_action had
    # swapped it, so at a timeout the winner's share went to the loser.
    print("=== constant_sum: terminal share reaches the WINNER's seat ===")
    seen_actor_won = seen_other_won = seen_conquest = seen_timeout = 0
    for seed in range(n_games):
        r = play_one_game(seed, mode="constant_sum", dense=False)
        wid, actor = r["winner_id"], r["actor_id"]

        assert wid in (0, 1, None), f"[seed {seed}] bad winner_id={wid}"

        # Never punish: both shares are non-negative, the loser tends to 0.
        assert r["terminal_r"] >= -1e-9 and r["terminal_ro"] >= -1e-9, (
            f"[seed {seed}] negative terminal share: "
            f"r_actor={r['terminal_r']:+.6f} r_other={r['terminal_ro']:+.6f}"
        )

        # THE attribution check: whichever seat won must hold the LARGER share.
        # terminal_r belongs to the actor, terminal_ro to the other seat.
        if wid is not None:
            actor_share, other_share = r["terminal_r"], r["terminal_ro"]
            winner_share = actor_share if wid == actor else other_share
            loser_share  = other_share if wid == actor else actor_share
            assert winner_share > loser_share, (
                f"[seed {seed}] winner P{wid} got {winner_share:.4f} but the "
                f"loser got {loser_share:.4f} (actor was P{actor}) — terminal "
                f"reward is attributed to the wrong seat"
            )
            if wid == actor: seen_actor_won += 1
            else:            seen_other_won += 1

        # dense off → the dense stream is identically zero
        assert all(d == 0.0 for d in r["dense"]), (
            f"[seed {seed}] dense stream non-zero with dense_reward=False"
        )
        # …and the terminal stream is zero on every non-terminal step
        for i, tv in enumerate(r["term"][:-1]):
            assert tv == 0.0, f"[seed {seed}] non-terminal r_term at {i}: {tv}"

        if r["is_conquest"]:
            seen_conquest += 1
            # Flat bonus plus the early-win multiplier, so it lands in
            # [CONQ, CONQ*(1+EARLY)] and the loser gets nothing.
            assert CONQ - 1e-6 <= r["terminal_r"] <= CONQ_MAX + 1e-6, (
                f"[seed {seed}] conquest paid {r['terminal_r']:.4f}, expected "
                f"[{CONQ}, {CONQ_MAX}]"
            )
            assert r["terminal_ro"] == 0.0
        else:
            seen_timeout += 1
            # Timeout is constant-sum and strictly below any conquest.
            s = r["terminal_r"] + r["terminal_ro"]
            assert abs(s - W) < 1e-6, (
                f"[seed {seed}] timeout shares sum to {s:.6f}, expected {W}"
            )
            assert r["terminal_r"] < CONQ, (
                f"[seed {seed}] a timeout paid {r['terminal_r']:.4f}, which "
                f"reaches the conquest reward {CONQ} — conquest must dominate"
            )
        print(f"  seed {seed:2d}: steps={r['n_steps']:4d}  winner={wid}  "
              f"actor=P{actor}  conquest={r['is_conquest']}  "
              f"r_actor={r['terminal_r']:.4f}  r_other={r['terminal_ro']:.4f}")

    # Coverage: the sweep is only a regression test if it actually drove BOTH
    # attribution branches — a sweep where the actor always wins proves nothing.
    assert seen_actor_won > 0, "sweep never produced a game won by the actor"
    assert seen_other_won > 0, (
        "sweep never produced a game won by the NON-acting seat, so this run "
        "does not exercise the attribution fix"
    )
    assert seen_timeout > 0, "sweep never produced a turn-limit timeout"
    print(f"\n  coverage: actor-won={seen_actor_won}  other-won={seen_other_won}  "
          f"conquest={seen_conquest}  timeout={seen_timeout}")

    # ── dense on: separate stream, correctly scaled ───────────────────────
    print("\n=== dense stream (bootstrap shaping) ===")
    for seed in range(6):
        r = play_one_game(seed, mode="constant_sum", dense=True)
        dense_total = sum(r["dense"])
        # Dense must guide without outranking the objective: an episode's total
        # shaping should stay in the neighbourhood of a single terminal payout.
        assert abs(dense_total) < CONQ_MAX, (
            f"[seed {seed}] dense total {dense_total:.3f} rivals the conquest "
            f"reward {CONQ_MAX} — dense_scale is too large"
        )
        # Terminal stream must be untouched by dense shaping.
        assert all(t == 0.0 for t in r["term"][:-1]), (
            f"[seed {seed}] dense shaping leaked onto the terminal stream"
        )
        print(f"  seed {seed}: dense total={dense_total:+.3f}  "
              f"terminal={r['terminal_r']:.4f}  steps={r['n_steps']}")

    # ── none: no terminal reward at all ───────────────────────────────────
    print("\n=== none (pure dense) ===")
    for seed in range(2):
        r = play_one_game(seed, mode="none", dense=True)
        assert r["terminal_r"] == 0.0 and r["terminal_ro"] == 0.0, (
            f"[seed {seed}] mode=none must pay nothing terminal, got "
            f"{r['terminal_r']} / {r['terminal_ro']}"
        )
        print(f"  seed {seed}: winner={r['winner_id']}  "
              f"dense total={sum(r['dense']):+.3f} (no terminal)")

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
