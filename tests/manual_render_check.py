"""
Manual script to exercise the new EnvWrapper.render() across a random game.
Saves PNGs to render_debug/ for eyeball inspection.

Run:
    python -m tests.manual_render_check
"""
import os
import sys
import numpy as np

from game.enums import BoardType, Tribes, ActionTypes
from env.wrapper import EnvWrapper
from tests.test_random_games import _random_action_from_mask

BOARD_CONFIG = {"board_size": (9, 9), "board_type": BoardType.Dummy, "n_players": 2}
TRIBES = [Tribes.Omaji, Tribes.Yaddak]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "render_debug")


def main(seed=0, max_steps=60, show_overlay=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(seed)

    env = EnvWrapper(BOARD_CONFIG, TRIBES)
    env.reset()

    # Stage A: initial board, no action yet.
    env.render(show_action_overlay=False,
               save_path=os.path.join(OUT_DIR, "step_000_initial.png"),
               show=False)
    print("Saved step_000_initial.png")

    for step in range(1, max_steps + 1):
        mask = env.get_action_mask()
        action = _random_action_from_mask(mask)
        obs, reward, done, info = env.step(action)

        atype = ActionTypes(action[0]).name
        env.render(show_action_overlay=show_overlay,
                   save_path=os.path.join(OUT_DIR, f"step_{step:03d}_{atype}.png"),
                   show=False)
        print(f"step {step:3d}  action={atype:12s}  "
              f"p_go={env.game.player_go_id}  turn={env.game.turn}  done={done}")

        if done:
            break

    print(f"\nAll renders saved to {OUT_DIR}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    main(seed=seed, max_steps=steps)
