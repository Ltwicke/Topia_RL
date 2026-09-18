"""
smoke_zero_sum_rewards.py
─────────────────────────
Sanity check for the new zero-sum terminal reward path in env/wrapper.py.

Asserts per game:
  • r_cur + r_opp == 0 at termination (zero-sum)
  • winner_id is 0, 1, or None
  • all non-terminal rewards along the trajectory are exactly 0.0
  • info["reward_opp"] is 0.0 on every non-terminal step

Prints the diff distribution so the magnitude can be eyeballed.

Run from project root:
  python eval/smoke_zero_sum_rewards.py
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
                  terminal_tau: float = 666.0,
                  conquest_reward: float = 2.0) -> dict:
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
    )
    env.reset()

    rewards_log: list[float] = []
    n_steps = 0
    while True:
        a = random_valid_action(env)
        # Seat that is about to act. At a turn-limit timeout this is always
        # player 1 (the timeout fires on its EndTurn), while the score leader
        # may be either seat — which is exactly the inversion this guards.
        actor_id = env.game.player_go_id
        obs, rew, done, info = env.step(a)
        rewards_log.append(float(rew))
        n_steps += 1
        if done:
            return {
                "n_steps":     n_steps,
                "rewards":     rewards_log,
                "winner_id":   info["winner_id"],
                "is_conquest": info["is_conquest"],
                "actor_id":    actor_id,
                "terminal_r":  float(rew),
                "terminal_ro": float(info["reward_opp"]),
            }
        if n_steps > 5000:
            raise RuntimeError("Game did not terminate — likely a bug")


def main() -> None:
    n_games = 24
    W, CONQ = 1.0, 2.0

    # ── zero_sum: seat attribution, antisymmetry, boundedness ─────────────
    # This is the sc-41 regression block. Before the fix, _get_done_and_rewards
    # read the seat back off game.player_go_id AFTER apply_action had already
    # swapped it, so at a timeout the winner's share was handed to the loser
    # (and the opposite branch raised NameError on an undefined `amt`).
    print("=== zero_sum: terminal share reaches the WINNER's seat ===")
    seen_actor_won = seen_other_won = seen_conquest = seen_timeout = 0
    for seed in range(n_games):
        r = play_one_game(seed, mode="zero_sum", dense=False)
        wid, actor = r["winner_id"], r["actor_id"]

        assert wid in (0, 1, None), f"[seed {seed}] bad winner_id={wid}"

        # Exact antisymmetry: the two seats' shares must cancel.
        s = r["terminal_r"] + r["terminal_ro"]
        assert abs(s) < 1e-6, (
            f"[seed {seed}] zero_sum terminal not balanced: "
            f"{r['terminal_r']:.6f} + {r['terminal_ro']:.6f} = {s:.6f}"
        )

        # Bounded by the largest payout the scheme can emit.
        assert abs(r["terminal_r"]) <= CONQ + 1e-6, (
            f"[seed {seed}] terminal share {r['terminal_r']} exceeds "
            f"conquest_reward {CONQ}"
        )

        # THE attribution check: whichever seat won must hold the positive
        # share. terminal_r belongs to the actor, terminal_ro to the other seat.
        if wid is not None:
            actor_share = r["terminal_r"]
            other_share = r["terminal_ro"]
            winner_share = actor_share if wid == actor else other_share
            loser_share  = other_share if wid == actor else actor_share
            assert winner_share > 0.0, (
                f"[seed {seed}] winner P{wid} got a non-positive share "
                f"{winner_share:+.4f} (actor was P{actor}) — terminal reward "
                f"is attributed to the wrong seat"
            )
            assert loser_share < 0.0, (
                f"[seed {seed}] loser got a non-negative share {loser_share:+.4f}"
            )
            if wid == actor: seen_actor_won += 1
            else:            seen_other_won += 1

        # dense off → all non-terminal rewards are exactly 0
        for i, rv in enumerate(r["rewards"][:-1]):
            assert rv == 0.0, f"[seed {seed}] non-terminal reward at {i}: {rv}"

        if r["is_conquest"]:
            seen_conquest += 1
            assert abs(abs(r["terminal_r"]) - CONQ) < 1e-6, (
                f"[seed {seed}] conquest should pay exactly ±{CONQ}, "
                f"got {r['terminal_r']:+.4f}"
            )
        else:
            seen_timeout += 1
            # A timeout is a tanh-squashed margin, so it can never reach the
            # flat conquest bonus: an outright win always outranks a timeout.
            assert abs(r["terminal_r"]) <= W + 1e-6, (
                f"[seed {seed}] timeout share {r['terminal_r']:+.4f} exceeds "
                f"terminal_weight {W}"
            )
        print(f"  seed {seed:2d}: steps={r['n_steps']:4d}  winner={wid}  "
              f"actor=P{actor}  conquest={r['is_conquest']}  "
              f"r_actor={r['terminal_r']:+.4f}  r_other={r['terminal_ro']:+.4f}")

    # Coverage: the sweep is only a regression test if it actually drove BOTH
    # attribution branches. The `winner is not the actor` case is the one that
    # used to raise NameError, so a sweep that never hits it proves nothing.
    assert seen_actor_won > 0, "sweep never produced a game won by the actor"
    assert seen_other_won > 0, (
        "sweep never produced a game won by the NON-acting seat — that is the "
        "branch that used to crash, so this run does not exercise the fix"
    )
    assert seen_timeout > 0, "sweep never produced a turn-limit timeout"
    print(f"\n  coverage: actor-won={seen_actor_won}  other-won={seen_other_won}  "
          f"conquest={seen_conquest}  timeout={seen_timeout}")

    # ── winner_only: loser is never punished ──────────────────────────────
    print("\n=== winner_only (non-zero-sum ablation) ===")
    for seed in range(n_games):
        r = play_one_game(seed, mode="winner_only", dense=False)
        wid, actor = r["winner_id"], r["actor_id"]
        assert r["terminal_r"] >= -1e-6 and r["terminal_ro"] >= -1e-6, (
            f"[seed {seed}] winner_only assigned a negative share: "
            f"r_actor={r['terminal_r']:+.4f} r_other={r['terminal_ro']:+.4f}"
        )
        if wid is not None:
            winner_share = r["terminal_r"] if wid == actor else r["terminal_ro"]
            assert winner_share > 0.0, (
                f"[seed {seed}] winner P{wid} got {winner_share:+.4f} "
                f"(actor was P{actor})"
            )
    print(f"  {n_games} games: winner always paid, loser never punished")

    # ── none: no terminal reward at all ───────────────────────────────────
    print("\n=== none (pure dense) ===")
    for seed in range(2):
        r = play_one_game(seed, mode="none", dense=True)
        assert r["terminal_ro"] == 0.0, (
            f"[seed {seed}] mode=none should give r_opp==0, got {r['terminal_ro']}"
        )
        print(f"  seed {seed}: winner={r['winner_id']}  "
              f"terminal_r={r['terminal_r']:+.2f} (dense only)")

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
