"""
Defender_ZoC — within 2 decisions (rollout stops on EndTurn) the active
player must position units to occupy at least 2 of 3 key tiles, building
a zone-of-control wall:

    KEY_TILES = {82, 83, 92}

Success metric: ≥2 of those 3 tiles are occupied by an active-player unit
after the rollout. The exact count (0/1/2/3) is also logged.
"""

from __future__ import annotations

from typing import List

import numpy as np

from game.enums              import ActionTypes
from scenarios.eval.adapter  import GameEnvAdapter
from scenarios.eval.runner   import ScenarioRunner, RunnerResult


_KEY_TILES = frozenset({82, 83, 92})


class Runner(ScenarioRunner):
    n_samples      = 20
    n_decisions    = 2
    render_enabled = False

    def play(self, policy, scenario, device) -> RunnerResult:
        adapter = GameEnvAdapter(scenario)
        end_turn_atype = int(ActionTypes.EndTurn)
        pov_pid = int(scenario.current_player)

        occupied_counts: List[int]  = []
        successes:       List[bool] = []
        decisions_taken: List[int]  = []

        for _ in range(self.n_samples):
            adapter.reset()
            decisions_before = adapter.env.n_decisions

            for _d in range(self.n_decisions):
                rec = self._one_forward(adapter, policy)
                if int(rec.action[0]) == end_turn_atype:
                    break
                _obs, _r, done, _info = adapter.step(rec.action)
                if done:
                    break

            decisions_taken.append(adapter.env.n_decisions - decisions_before)

            player = adapter.game.players[pov_pid]
            occupied_tiles = {
                int(u.tile.id)
                for u in player.units_under_control.values()
            }
            n_occupied = len(occupied_tiles & _KEY_TILES)
            occupied_counts.append(n_occupied)
            successes.append(n_occupied >= 2)

        n           = self.n_samples
        n_success   = int(sum(successes))
        occ_arr     = np.asarray(occupied_counts, dtype=np.int32)

        return RunnerResult(
            metrics = {
                "two_of_three_rate":      float(n_success / max(n, 1)),
                "n_two_of_three":         n_success,
                "occupied_count_mean":    float(occ_arr.mean()),
                "occupied_count_max":     int(occ_arr.max()),
                "occupied_count_min":     int(occ_arr.min()),
                "n_samples":              n,
                "avg_decisions_taken":    float(np.mean(decisions_taken)),
                "n_decisions_max":        int(self.n_decisions),
            },
            metrics_extra = {
                "occupied_counts": occupied_counts,
                "successes":       successes,
            },
            title = (
                f"Defender_ZoC — 2-of-3 in {n_success}/{n}  "
                f"(mean occupied={occ_arr.mean():.2f}/3)"
            ),
        )
