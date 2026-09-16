"""
Giant_Houdini — the active player must, within 2 decisions:
  1. Move a Rider onto the key tile (id=118).
  2. Upgrade one of its cities.
  3. Choose the superunit branch on that upgrade (any new lvl whose name
     contains '_su': lvl5_su, lvl6_su, lvl7_su).

If all three apply, the upcoming auto-spawn ends up deleting the opposing
Giant ("Houdini"). Each sub-condition is logged individually so we can see
which step the policy is failing at; success = (1) ∧ (2) ∧ (3).

Rollout stops on EndTurn. No render — metrics-only.
"""

from __future__ import annotations

from typing import List

import numpy as np

from game.enums              import ActionTypes, UnitType
from scenarios.eval.adapter  import GameEnvAdapter
from scenarios.eval.runner   import ScenarioRunner, RunnerResult


_KEY_TILE = 118


class Runner(ScenarioRunner):
    n_samples      = 20
    n_decisions    = 2
    render_enabled = False

    def play(self, policy, scenario, device) -> RunnerResult:
        adapter = GameEnvAdapter(scenario)
        end_turn_atype = int(ActionTypes.EndTurn)
        pov_pid = int(scenario.current_player)

        rider_on_key:    List[bool] = []
        city_upgraded:   List[bool] = []
        su_chosen:       List[bool] = []
        all_three:       List[bool] = []
        decisions_taken: List[int]  = []

        for _ in range(self.n_samples):
            adapter.reset()
            player = adapter.game.players[pov_pid]
            initial_lvls = {
                int(c.tile_id): c.lvl for c in player.cities_under_control
            }
            initial_times = {
                int(c.tile_id): int(c.times_upgraded)
                for c in player.cities_under_control
            }
            decisions_before = adapter.env.n_decisions

            for _d in range(self.n_decisions):
                rec = self._one_forward(adapter, policy)
                if int(rec.action[0]) == end_turn_atype:
                    break
                _obs, _r, done, _info = adapter.step(rec.action)
                if done:
                    break

            decisions_taken.append(adapter.env.n_decisions - decisions_before)

            # ── Sub-condition 1: rider on key tile ────────────────────────────
            player_after = adapter.game.players[pov_pid]
            rider_hit = any(
                u.unit_type == UnitType.Rider and int(u.tile.id) == _KEY_TILE
                for u in player_after.units_under_control.values()
            )

            # ── Sub-condition 2: any city's times_upgraded strictly increased ─
            upgraded_tile_ids = {
                int(c.tile_id) for c in player_after.cities_under_control
                if int(c.tile_id) in initial_times
                and int(c.times_upgraded) > initial_times[int(c.tile_id)]
            }
            upgraded = bool(upgraded_tile_ids)

            # ── Sub-condition 3: superunit branch chosen on an upgrade ────────
            # An upgrade hits the SU branch when the new lvl name contains
            # '_su'. Restrict to cities that actually got upgraded so that
            # cities already at lvl*_su pre-rollout don't count.
            su = any(
                "_su" in c.lvl.name
                and int(c.tile_id) in upgraded_tile_ids
                and c.lvl != initial_lvls.get(int(c.tile_id))
                for c in player_after.cities_under_control
            )

            rider_on_key .append(rider_hit)
            city_upgraded.append(upgraded)
            su_chosen    .append(su)
            all_three    .append(rider_hit and upgraded and su)

        n         = self.n_samples
        n_rider   = int(sum(rider_on_key))
        n_upg     = int(sum(city_upgraded))
        n_su      = int(sum(su_chosen))
        n_success = int(sum(all_three))

        return RunnerResult(
            metrics = {
                "rider_on_118_rate":      float(n_rider   / max(n, 1)),
                "city_upgraded_rate":     float(n_upg     / max(n, 1)),
                "superunit_chosen_rate":  float(n_su      / max(n, 1)),
                "all_three_rate":         float(n_success / max(n, 1)),
                "n_rider_on_118":         n_rider,
                "n_city_upgraded":        n_upg,
                "n_superunit_chosen":     n_su,
                "n_all_three":            n_success,
                "n_samples":              n,
                "avg_decisions_taken":    float(np.mean(decisions_taken)),
                "n_decisions_max":        int(self.n_decisions),
            },
            metrics_extra = {
                "rider_on_key":  rider_on_key,
                "city_upgraded": city_upgraded,
                "su_chosen":     su_chosen,
                "all_three":     all_three,
            },
            title = (
                f"Giant_Houdini — all-three in {n_success}/{n}  "
                f"(rider={n_rider}, upgrade={n_upg}, su={n_su})"
            ),
        )
