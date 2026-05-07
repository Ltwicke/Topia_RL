"""
Upgrade_city_order2 — P0 (active player) has multiple cities and enough
stars to upgrade them. Up to 2 decisions per rollout (stops on EndTurn).

Success metric: were ALL of P0's cities upgraded during the rollout?
A city counts as "upgraded" if its `times_upgraded` strictly increased
from its initial value at scenario reset. Partial success (only one of
two cities upgraded) is logged as well.
"""

from __future__ import annotations

from typing import List

import numpy as np

from game.enums              import ActionTypes
from scenarios.eval.adapter  import GameEnvAdapter
from scenarios.eval.runner   import ScenarioRunner, RunnerResult


class Runner(ScenarioRunner):
    n_samples      = 20
    n_decisions    = 2        # max — stops on EndTurn too
    render_enabled = False

    def play(self, policy, scenario, device) -> RunnerResult:
        adapter = GameEnvAdapter(scenario)
        end_turn_atype = int(ActionTypes.EndTurn)
        pov_pid = int(scenario.current_player)

        successes:                List[bool] = []
        n_upgraded_per_sample:    List[int]  = []
        decisions_taken:          List[int]  = []
        n_initial_cities = 0  # captured once below

        for _ in range(self.n_samples):
            adapter.reset()
            player = adapter.game.players[pov_pid]
            initial_times = {
                int(c.tile_id): int(c.times_upgraded)
                for c in player.cities_under_control
            }
            n_initial_cities = len(initial_times)
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

            # Count cities whose times_upgraded strictly increased.
            player_after = adapter.game.players[pov_pid]
            n_upgraded = sum(
                1 for c in player_after.cities_under_control
                if int(c.tile_id) in initial_times
                and int(c.times_upgraded) > initial_times[int(c.tile_id)]
            )
            n_upgraded_per_sample.append(n_upgraded)
            successes.append(
                n_initial_cities > 0 and n_upgraded == n_initial_cities
            )

        n            = self.n_samples
        n_success    = int(sum(successes))
        n_upgraded_a = np.asarray(n_upgraded_per_sample, dtype=np.int32)

        return RunnerResult(
            metrics = {
                "both_upgraded_rate":     float(n_success / max(n, 1)),
                "n_both_upgraded":        n_success,
                "n_initial_cities":       int(n_initial_cities),
                "avg_cities_upgraded":    float(n_upgraded_a.mean()),
                "max_cities_upgraded":    int(n_upgraded_a.max()),
                "n_samples":              n,
                "avg_decisions_taken":    float(np.mean(decisions_taken)),
                "n_decisions_max":        int(self.n_decisions),
            },
            metrics_extra = {
                "n_upgraded_per_sample": n_upgraded_per_sample,
                "successes":             successes,
            },
            title = (
                f"Upgrade_city_order2 — all-upgraded in {n_success}/{n} "
                f"({100 * n_success / max(n, 1):.0f}%); "
                f"avg upgraded {n_upgraded_a.mean():.2f}/"
                f"{n_initial_cities}"
            ),
        )
