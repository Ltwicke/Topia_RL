"""
Agent plays 2 actions; we seek to see upgrade to wall and get defender
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from game.enums              import ActionTypes, UnitType

from env.renderer            import BoardRenderer
from scenarios.eval.adapter  import GameEnvAdapter
from scenarios.eval.runner   import ScenarioRunner, RunnerResult


class Runner(ScenarioRunner):
    n_samples      = 20
    n_decisions    = 2
    render_enabled = True

    # POV player whose uncovered count we track.
    pov_player_id = 0

    def play(self, policy, scenario, device) -> RunnerResult:

        end_turn_atype = int(ActionTypes.EndTurn)
        records = []
        adapter = GameEnvAdapter(scenario)

        wall_chosen: List[bool] = []
        defender_created:  List[bool] = []
        any_unit_created: List[bool] = []
        decisions_taken: List[int]  = []
        both: List[bool] = []

        player = adapter.game.players[self.pov_player_id]

        for _ in range(self.n_samples):
            adapter.reset()

            for _d in range(self.n_decisions):
                rec = self._one_forward(adapter, policy)
                records.append(rec)
                if int(rec.action[0]) == end_turn_atype:
                    break
                _obs, _r, done, _info= adapter.step(rec.action)
                if done: # does not happen
                    break

            decisions_taken.append(adapter.env.n_decisions)

            ## check condition 1
            walled = any("_wall" in c.lvl.name for c in player.cities_under_control)
            wall_chosen.append(walled)

            ## check condition 2
            any_unit = adapter.env.game.game_board.board[83].unit != None
            any_unit_created.append(any_unit)

            if any_unit:
                defender_chosen = int(adapter.env.game.game_board.board[83].unit.unit_type) == int(UnitType.Defender)
                defender_created.append(defender_chosen)

            both.append(walled and defender_chosen)

        n_wall_chosen = int(sum(wall_chosen))
        n_defenders_created = int(sum(defender_created))
        n_any_unit_created = int(sum(any_unit_created))

        success_both = int(sum(both))

        # Average joint_probs across samples for the overlay.
        avg_probs, traj_actions = self._average_joint_probs(records)
        last_action = records[-1].action
        renderer    = BoardRenderer(adapter.env)
        prob_overlay, atype_probs = renderer.compute_prob_overlay(
            last_action, torch.from_numpy(avg_probs), traj_actions,
        )

        return RunnerResult(
            prob_overlay   = prob_overlay,
            atype_probs    = atype_probs,
            sampled_action = last_action,
            metrics        = {
                "wall_chosen_rate":     float(n_wall_chosen/self.n_samples),
                "unit_creation_rate":   float(n_any_unit_created/self.n_samples),
                "defenders_rate":       float(n_defenders_created/max(n_any_unit_created, 1)),
                "success_both":         float(success_both/self.n_samples),
                "n_wall_chosen":        n_wall_chosen,
                "n_defenders_created":  n_defenders_created,
                "n_any_unit_created":   n_any_unit_created,
                "n_samples":            int(self.n_samples),
                "n_decisions":          int(self.n_decisions),
                "avg_decisions_taken":    float(np.mean(decisions_taken)),
            },
            title = (
                f"Defender_and_wall — success_rate={float(success_both/self.n_samples)}  "
                f"(any_units_made={n_any_unit_created}, wall_chosen={n_wall_chosen}, defender_chosen={n_defenders_created})"
            ),
        )
