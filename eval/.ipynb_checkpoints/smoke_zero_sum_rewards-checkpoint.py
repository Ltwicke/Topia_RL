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


def play_one_game(seed: int) -> dict:
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
        dense_reward=False,
        zero_sum_terminal=True,
    )
    env.reset()

    rewards_log: list[float] = []
    info_opp_log: list[float] = []
    n_steps = 0
    while True:
        a = random_valid_action(env)
        obs, rew, done, info = env.step(a)
        rewards_log.append(float(rew))
        info_opp_log.append(float(info["reward_opp"]))
        n_steps += 1
        if done:
            return {
                "n_steps":     n_steps,
                "rewards":     rewards_log,
                "info_opps":   info_opp_log,
                "winner_id":   info["winner_id"],
                "terminal_r":  float(rew),
                "terminal_ro": float(info["reward_opp"]),
            }
        if n_steps > 5000:
            raise RuntimeError("Game did not terminate — likely a bug")


def main() -> None:
    n_games = 5
    diffs: list[float] = []
    for seed in range(n_games):
        result = play_one_game(seed)

        # zero-sum at terminal
        s = result["terminal_r"] + result["terminal_ro"]
        assert abs(s) < 1e-4, (
            f"[seed {seed}] r_cur + r_opp != 0 at terminal: {result['terminal_r']:.4f} + "
            f"{result['terminal_ro']:.4f} = {s:.4f}"
        )

        # winner_id is 0, 1, or None
        wid = result["winner_id"]
        assert wid in (0, 1, None), f"[seed {seed}] unexpected winner_id={wid}"

        # all non-terminal rewards are zero
        for i, r in enumerate(result["rewards"][:-1]):
            assert r == 0.0, (
                f"[seed {seed}] non-terminal reward at step {i}: {r}"
            )
            assert result["info_opps"][i] == 0.0, (
                f"[seed {seed}] non-terminal info reward_opp at step {i}: "
                f"{result['info_opps'][i]}"
            )

        diffs.append(result["terminal_r"])
        print(
            f"  seed {seed}: steps={result['n_steps']:4d}  winner={wid}  "
            f"r_cur={result['terminal_r']:+10.2f}  r_opp={result['terminal_ro']:+10.2f}"
        )

    arr = np.array(diffs, dtype=np.float64)
    print()
    print(f"diff stats over {n_games} games:")
    print(f"  mean = {arr.mean():.2f}")
    print(f"  std  = {arr.std():.2f}")
    print(f"  min  = {arr.min():.2f}")
    print(f"  max  = {arr.max():.2f}")
    print()
    print("All assertions passed.")


if __name__ == "__main__":
    main()
