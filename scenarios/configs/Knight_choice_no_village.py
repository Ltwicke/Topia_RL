"""
Knight_chain_choice — P1's red Knight is the only `ready` unit. The agent
makes one decision; we render the averaged probability overlay across N=20
forward passes AND log which "side" of the chain the policy preferred.

Choice tracking
───────────────
The Knight has two viable move corridors. Each forward pass samples a target
tile; we count how often the sampled tile falls into:

    LEFT   = {147, 149, 163, 164, 165}
    RIGHT  = {151, 167, 168, 169}

The metric `choice_left_rate` (left_count / n_samples) is what tracks
training progress on this scenario.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from game.enums              import ActionTypes
from env.renderer            import BoardRenderer
from scenarios.eval.adapter  import GameEnvAdapter
from scenarios.eval.runner   import ScenarioRunner, RunnerResult


_LEFT_TILES  = {147, 149, 163, 164, 165}
_RIGHT_TILES = {151, 167, 168, 169}


class Runner(ScenarioRunner):
    n_samples      = 20
    n_decisions    = 1
    render_enabled = True

    def play(self, policy, scenario, device) -> RunnerResult:
        adapter = GameEnvAdapter(scenario)

        records          = []
        n_left           = 0
        n_right          = 0
        n_other_move     = 0
        n_non_move       = 0

        move_atype = int(ActionTypes.MoveUnit)

        for _ in range(self.n_samples):
            adapter.reset()
            rec = self._one_forward(adapter, policy)
            records.append(rec)

            atype = int(rec.action[0])
            if atype != move_atype or len(rec.action) < 3:
                n_non_move += 1
                continue
            target_tile = int(rec.action[2])
            if   target_tile in _LEFT_TILES:  n_left  += 1
            elif target_tile in _RIGHT_TILES: n_right += 1
            else:                             n_other_move += 1

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
                "choice_left_rate":  float(n_left  / max(n, 1)),
                "choice_right_rate": float(n_right / max(n, 1)),
                "n_left":            int(n_left),
                "n_right":           int(n_right),
                "n_other_move":      int(n_other_move),
                "n_non_move":        int(n_non_move),
                "n_samples":         int(n),
            },
            title = (
                f"{scenario.name} — left={n_left}/{n}  right={n_right}/{n}  "
                f"other={n_other_move + n_non_move}/{n}"
            ),
        )
