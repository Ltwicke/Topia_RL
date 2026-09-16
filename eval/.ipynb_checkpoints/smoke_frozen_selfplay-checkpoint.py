"""
smoke_frozen_selfplay.py
─────────────────────────
Stripped single-process version of the worker rollout to validate the
frozen-opponent self-play bookkeeping (no PPO, no multiprocessing):

  1. Player-1 (frozen) steps are flagged is_active == 0; player-0 steps == 1.
  2. The active policy (seat 0) and frozen policy (seat 1) are dispatched by
     env.game.player_go_id.
  3. Each finished game contributes exactly one done-cut among PLAYER-0 slots
     (its own terminal move, or a back-filled cut when it loses).
  4. n_games == n_conquest + n_timeout, and n_active_wins <= n_games.

Run from project root:
  python eval/smoke_frozen_selfplay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RL.ppo.game_manager import TrainConfig, _make_env
from RL.models.policy import PolicyNetwork, make_snapshot


def run_rollout(T=1500, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = TrainConfig()
    # Shrink the env so games terminate quickly under the untrained policy.
    cfg.max_turns_per_game = 6
    cfg.board_size_range   = (11, 11)
    active_pid = cfg.active_player_id

    env = _make_env(cfg)
    obs = env.reset()

    pol_active = PolicyNetwork(cfg); pol_active.eval()
    pol_frozen = PolicyNetwork(cfg); pol_frozen.eval()

    dones      = np.zeros(T, dtype=np.float32)
    is_active  = np.zeros(T, dtype=np.float32)
    player_ids = np.full(T, -1, dtype=np.int32)

    n_games = n_active_wins = n_conquest = n_timeout = 0

    with torch.no_grad():
        for t in range(T):
            mask = env.get_action_mask()
            cur_pid = env.game.player_go_id
            _ = make_snapshot(obs, env.Nx, env.Ny, player_id=cur_pid)

            pol = pol_active if cur_pid == active_pid else pol_frozen
            action, _, _, _, _, _ = pol(obs, mask)
            next_obs, rew, done, info = env.step(action)

            is_active[t]  = 1.0 if cur_pid == active_pid else 0.0
            player_ids[t] = cur_pid
            dones[t]      = float(done)

            if done:
                winner_id = info["winner_id"]
                is_conq   = info["is_conquest"]
                opp_id = (cur_pid + 1) % 2
                if t > 0:
                    prev_dones = np.nonzero(dones[:t] > 0.5)[0]
                    game_start = 0 if prev_dones.size == 0 else int(prev_dones[-1]) + 1
                    seg = player_ids[game_start:t]
                    opp_steps = np.nonzero(seg == opp_id)[0]
                    if opp_steps.size > 0:
                        last_opp_t = game_start + int(opp_steps[-1])
                        dones[last_opp_t] = 1.0

                n_games       += 1
                n_active_wins += int(winner_id == active_pid)
                n_conquest    += int(is_conq)
                n_timeout     += int(not is_conq)

                env = _make_env(cfg)
                obs = env.reset()
            else:
                obs = next_obs

    return dict(dones=dones, is_active=is_active, player_ids=player_ids,
                n_games=n_games, n_active_wins=n_active_wins,
                n_conquest=n_conquest, n_timeout=n_timeout)


def main():
    r = run_rollout()

    # 1. is_active matches player_ids == 0
    active_expected = (r["player_ids"] == 0).astype(np.float32)
    assert np.array_equal(r["is_active"], active_expected), \
        "is_active does not match player-0 slots"

    # 2. n_games == n_conquest + n_timeout
    assert r["n_games"] == r["n_conquest"] + r["n_timeout"], \
        f"game count split mismatch: {r['n_games']} != " \
        f"{r['n_conquest']} + {r['n_timeout']}"
    assert r["n_active_wins"] <= r["n_games"], "active wins exceed games"

    # 3. Each finished game contributes exactly one done among player-0 slots.
    #    Count done markers that land on player-0 steps; should equal n_games
    #    (player 0 always has exactly one terminal cut per game: its winning
    #     move OR a back-filled cut when it loses).
    p0_dones = int(((r["dones"] > 0.5) & (r["player_ids"] == 0)).sum())
    # Allow the trailing in-progress game (no done yet) — so p0_dones is either
    # n_games or n_games (the last game may be unfinished and contributes 0).
    assert p0_dones == r["n_games"], (
        f"player-0 done-cuts ({p0_dones}) != n_games ({r['n_games']}); "
        "each finished game must cut player 0's trajectory exactly once"
    )

    print(f"  games={r['n_games']}  active_wins={r['n_active_wins']}  "
          f"conquest={r['n_conquest']}  timeout={r['n_timeout']}")
    print(f"  player-0 done-cuts={p0_dones}  (== n_games)")
    print("\nAll frozen self-play checks passed.")


if __name__ == "__main__":
    main()
