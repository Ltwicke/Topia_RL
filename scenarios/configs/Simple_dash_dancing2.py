"""
Simple_dash_dancing2 — the agent (P1) gets exactly 2 decisions per rollout.
Sub-optimal play (attack the warrior directly) uncovers fewer tiles than
"dash-dance" play (move into open ground first → attack from cover).

The Δ(uncovered tiles) of P1 across the 2 decisions is the readout. We also
average the first-decision joint_probs over N rollouts for a movement
heatmap the renderer overlays.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from env.renderer            import BoardRenderer
from scenarios.eval.adapter  import GameEnvAdapter
from scenarios.eval.runner   import ScenarioRunner, RunnerResult


class Runner(ScenarioRunner):
    n_samples      = 20
    n_decisions    = 2
    render_enabled = True

    # POV player whose uncovered count we track.
    pov_player_id = 1

    def play(self, policy, scenario, device) -> RunnerResult:
        adapter = GameEnvAdapter(scenario)

        first_records: List = []
        deltas:        List[int] = []

        for _ in range(self.n_samples):
            adapter.reset()
            unc_before = len(
                adapter.game.players[self.pov_player_id].uncovered_tile_ids
            )

            # Decision 1 — capture probs, then apply.
            rec1 = self._one_forward(adapter, policy)
            first_records.append(rec1)
            _obs, _r, done, _info = adapter.step(rec1.action)

            # Decision 2 — only if game still active.
            if not done:
                rec2 = self._one_forward(adapter, policy)
                _obs, _r, done, _info = adapter.step(rec2.action)

            unc_after = len(
                adapter.game.players[self.pov_player_id].uncovered_tile_ids
            )
            deltas.append(int(unc_after - unc_before))

        deltas_np = np.asarray(deltas, dtype=np.int32)

        # Average first-decision joint_probs for the overlay.
        avg_probs, traj_actions = self._average_joint_probs(first_records)
        last_action = first_records[-1].action
        renderer    = BoardRenderer(adapter.env)
        prob_overlay, atype_probs = renderer.compute_prob_overlay(
            last_action, torch.from_numpy(avg_probs), traj_actions,
        )

        return RunnerResult(
            prob_overlay   = prob_overlay,
            atype_probs    = atype_probs,
            sampled_action = last_action,
            metrics        = {
                "uncovered_delta_mean": float(deltas_np.mean()),
                "uncovered_delta_std":  float(deltas_np.std()),
                "uncovered_delta_max":  int(deltas_np.max()),
                "uncovered_delta_min":  int(deltas_np.min()),
                "n_samples":            int(self.n_samples),
                "n_decisions":          int(self.n_decisions),
            },
            title = (
                f"{scenario.name} — Δuncovered  "
                f"mean={deltas_np.mean():.1f}  "
                f"min={deltas_np.min()}  max={deltas_np.max()}"
            ),
        )
