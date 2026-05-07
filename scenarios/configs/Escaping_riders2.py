"""
Escaping_riders2 — three riders are in `escaping` state and need to move
to safe tiles. The active player has up to 3 decisions (rollout stops on
EndTurn). Success metric: count of riders that end up on UNSAFE tiles —
ideally 0.

UNSAFE tile_ids cover the danger zone laid out in the scenario:
    {106, 108, 120, 121, 122, 123, 124, 125,
     134, 135, 136, 138, 139, 140, 141, 154, 155}
"""

from __future__ import annotations

from typing import List

import numpy as np

from game.enums              import ActionTypes, UnitType
from scenarios.eval.adapter  import GameEnvAdapter
from scenarios.eval.runner   import ScenarioRunner, RunnerResult


_UNSAFE_TILES = frozenset({
    106, 108, 120, 121, 122, 123, 124, 125,
    134, 135, 136, 138, 139, 140, 141, 154, 155,
})


class Runner(ScenarioRunner):
    n_samples      = 20
    n_decisions    = 3        # max — stops on EndTurn too
    render_enabled = False

    # POV is the player whose riders we're tracking. The YAML's
    # `current_player` is the active player; we read it dynamically.
    def play(self, policy, scenario, device) -> RunnerResult:
        adapter = GameEnvAdapter(scenario)
        end_turn_atype = int(ActionTypes.EndTurn)
        pov_pid = int(scenario.current_player)

        unsafe_counts: List[int] = []
        decisions_taken: List[int] = []

        for _ in range(self.n_samples):
            adapter.reset()
            decisions_before = adapter.env.n_decisions

            # Rollout up to n_decisions, breaking on EndTurn.
            for _d in range(self.n_decisions):
                rec = self._one_forward(adapter, policy)
                if int(rec.action[0]) == end_turn_atype:
                    break
                _obs, _r, done, _info = adapter.step(rec.action)
                if done:
                    break

            decisions_taken.append(adapter.env.n_decisions - decisions_before)

            # Count active player's riders that ended on UNSAFE tiles.
            player = adapter.game.players[pov_pid]
            n_unsafe = sum(
                1 for u in player.units_under_control.values()
                if u.unit_type == UnitType.Rider and int(u.tile.id) in _UNSAFE_TILES
            )
            unsafe_counts.append(n_unsafe)

        unsafe_arr = np.asarray(unsafe_counts, dtype=np.int32)
        n          = self.n_samples
        all_safe   = int((unsafe_arr == 0).sum())

        return RunnerResult(
            metrics = {
                "unsafe_rider_count_mean":  float(unsafe_arr.mean()),
                "unsafe_rider_count_max":   int(unsafe_arr.max()),
                "unsafe_rider_count_min":   int(unsafe_arr.min()),
                "all_safe_rate":            float(all_safe / max(n, 1)),
                "n_all_safe":               all_safe,
                "n_samples":                n,
                "avg_decisions_taken":      float(np.mean(decisions_taken)),
                "n_decisions_max":          int(self.n_decisions),
            },
            metrics_extra = {"unsafe_counts": unsafe_counts},
            title = (
                f"Escaping_riders2 — all-safe in {all_safe}/{n} rollouts "
                f"(mean unsafe={unsafe_arr.mean():.2f})"
            ),
        )
