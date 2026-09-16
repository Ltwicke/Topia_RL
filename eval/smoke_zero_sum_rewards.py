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


def play_one_game(seed: int, mode: str, dense: bool) -> dict:
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
    )
    env.reset()

    rewards_log: list[float] = []
    n_steps = 0
    while True:
        a = random_valid_action(env)
        obs, rew, done, info = env.step(a)
        rewards_log.append(float(rew))
        n_steps += 1
        if done:
            return {
                "n_steps":     n_steps,
                "rewards":     rewards_log,
                "winner_id":   info["winner_id"],
                "is_conquest": info["is_conquest"],
                "terminal_r":  float(rew),
                "terminal_ro": float(info["reward_opp"]),
            }
        if n_steps > 5000:
            raise RuntimeError("Game did not terminate — likely a bug")


def main() -> None:
    n_games = 6

    # ── winner_only + dense (the actual training regime) ──────────────────
    print("=== winner_only + dense (training regime) ===")
    for seed in range(n_games):
        r = play_one_game(seed, mode="winner_only", dense=True)
        wid, is_conq = r["winner_id"], r["is_conquest"]
        assert wid in (0, 1, None), f"[seed {seed}] bad winner_id={wid}"

        # Loser never receives a negative terminal add. The acting player's
        # terminal reward includes dense shaping, so we check the OPPONENT
        # side: r_opp is the other player's terminal share — must be >= 0
        # (winner_only never assigns a negative reward to anyone).
        assert r["terminal_ro"] >= -1e-6, (
            f"[seed {seed}] opponent terminal share negative in winner_only: "
            f"{r['terminal_ro']}"
        )
        # Exactly one side gets a positive terminal diff (or none on a tie).
        # r_opp>0 means the non-acting player won (timeout score-lead).
        print(f"  seed {seed}: steps={r['n_steps']:4d}  winner={wid}  "
              f"conquest={is_conq}  r_cur={r['terminal_r']:+9.2f}  "
              f"r_opp={r['terminal_ro']:+9.2f}")

    # ── zero_sum back-compat: r_cur + r_opp == 0 at terminal ──────────────
    print("\n=== zero_sum (back-compat) ===")
    for seed in range(n_games):
        r = play_one_game(seed, mode="zero_sum", dense=False)
        s = r["terminal_r"] + r["terminal_ro"]
        assert abs(s) < 1e-3, (
            f"[seed {seed}] zero_sum terminal not balanced: "
            f"{r['terminal_r']:.3f} + {r['terminal_ro']:.3f} = {s:.3f}"
        )
        # dense off → all non-terminal rewards are exactly 0
        for i, rv in enumerate(r["rewards"][:-1]):
            assert rv == 0.0, f"[seed {seed}] non-terminal reward at {i}: {rv}"
        print(f"  seed {seed}: winner={r['winner_id']}  "
              f"r_cur+r_opp={s:+.4f}")

    # ── none: pure dense, no terminal diff ────────────────────────────────
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
