"""
calibrate_terminal_tau.py
─────────────────────────
Measure the empirical distribution of |Δ _terminal_score| at a turn-limit
timeout, so `TrainConfig.terminal_tau` can be set from data instead of guessed.

Why this matters
────────────────
The timeout terminal reward is `W * tanh(Δscore / τ)`. If τ is much smaller
than a typical |Δ|, tanh saturates and every win pays the same ±W — the margin
information the squashing was meant to preserve is destroyed, and the reward
degenerates into a pure win/loss signal. If τ is much larger than a typical
|Δ|, tanh stays in its linear region and the reward is effectively unbounded
again.

Setting τ ≈ median(|Δ|) puts the median game at tanh(1) ≈ 0.76·W, which keeps
most games on the responsive part of the curve while still bounding blowouts.

Caveat: run with a trained policy when you have one. Random play produces a
different (usually wider) score spread than real play, so this is a starting
point to be re-measured once the policy is non-trivial.

Run from project root (venv active — see CLAUDE.md):
  python eval/calibrate_terminal_tau.py [n_games] [max_turns]
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
from game.enums import BoardType, Tribes

from smoke_zero_sum_rewards import random_valid_action


def collect_margins(n_games: int, max_turns: int) -> tuple[list[float], int]:
    """Play random games; return |Δ terminal_score| for each timeout finish."""
    margins: list[float] = []
    n_conquest = 0

    for seed in range(n_games):
        random.seed(seed)
        np.random.seed(seed)
        n = random.randint(11, 16)
        env = EnvWrapper(
            {"board_size": [n, n],
             "board_type": random.choice([BoardType.Drylands, BoardType.Lakes]),
             "n_players": 2},
            [Tribes.Omaji, Tribes.Imperius],
            max_turns_per_game=max_turns,
            dense_reward=False,
            terminal_reward_mode="zero_sum",
        )
        env.reset()

        for _ in range(20000):
            actor_id = env.game.player_go_id
            _, _, done, info = env.step(random_valid_action(env))
            if done:
                if info["is_conquest"]:
                    n_conquest += 1
                else:
                    other_id = (actor_id + 1) % 2
                    margins.append(abs(
                        env._terminal_score(env.game.players[actor_id])
                        - env._terminal_score(env.game.players[other_id])
                    ))
                break
        else:
            raise RuntimeError(f"[seed {seed}] game did not terminate")

    return margins, n_conquest


def main() -> None:
    n_games   = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print(f"Playing {n_games} random games at max_turns_per_game={max_turns} ...")
    margins, n_conquest = collect_margins(n_games, max_turns)

    if not margins:
        print(f"No timeout finishes in {n_games} games ({n_conquest} conquests).")
        return

    m = np.array(margins, dtype=np.float64)
    pct = {p: float(np.percentile(m, p)) for p in (10, 25, 50, 75, 90, 99)}

    print(f"\n|Δ terminal_score| over {m.size} timeout games "
          f"({n_conquest} conquests excluded)")
    print(f"  mean   {m.mean():10.2f}    std  {m.std():10.2f}")
    for p, v in pct.items():
        print(f"  p{p:<3d}  {v:10.2f}")

    tau = pct[50]
    print(f"\nSuggested  terminal_tau = {tau:.1f}   (median |Δ|)")

    # Saturation check: how much of the margin range would survive this τ?
    for cand in (pct[25], pct[50], pct[75]):
        z = np.tanh(m / cand)
        frac_sat = float((np.abs(z) > 0.99).mean())
        print(f"  τ={cand:8.1f} → {frac_sat*100:5.1f}% of games saturate "
              f"(|tanh| > 0.99), mean |z| = {np.abs(z).mean():.3f}")
    print("\nPrefer the τ whose saturated fraction is small (< ~10%): a saturated "
          "game pays the same as a blowout, which is the information loss the "
          "tanh squashing exists to avoid.")


if __name__ == "__main__":
    main()
