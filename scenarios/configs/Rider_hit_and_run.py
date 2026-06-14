"""
Rider hit and run.
POV: P0
1 decision
simplest case
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from game.enums              import ActionTypes
from env.renderer            import BoardRenderer
from scenarios.eval.adapter  import GameEnvAdapter
from scenarios.eval.runner   import ScenarioRunner, RunnerResult


class Runner(ScenarioRunner):
    n_samples      = 20
    n_decisions    = 1
    render_enabled = True

    def play(self, policy, scenario, device) -> RunnerResult:
        records          = []
        adapter = GameEnvAdapter(scenario)

        move_atype = int(ActionTypes.MoveUnit)

        for _ in range(self.n_samples):
            adapter.reset()
            rec = self._one_forward(adapter, policy)
            records.append(rec)

            atype = int(rec.action[0])

        # Average joint_probs across samples for the overlay.
        avg_probs, traj_actions = self._average_joint_probs(records)
        last_action = records[-1].action
        renderer    = BoardRenderer(adapter.env)
        prob_overlay, atype_probs = renderer.compute_prob_overlay(
            last_action, torch.from_numpy(avg_probs), traj_actions,
        )

        n = self.n_samples
        return RunnerResult(
            prob_overlay   = prob_overlay,
            atype_probs    = atype_probs,
            sampled_action = last_action,
            metrics        = {
                "dummy":  1,
            },
            title = (
                f"{scenario.name}"
            ),
        )
