"""
estimate_lakes11 — single-shot evaluation of the HiddenTileEstimator.

For this scenario we do NOT sample any rollouts. We just:
  1. Reset the scenario to its YAML state.
  2. Call `policy.estimate_hidden_dual(obs)` once → (est_a, est_b).
  3. Compute `policy.estimator_loss([snap])` for the same state.
  4. Render the dual-POV figure with `show_hidden=True` so both players'
     hidden-tile estimates are overlaid on the board.

Metric: the scalar estimator loss (per-tile-normalised cross-entropy + BCE
across the hidden-feature groups). Logged to summary.csv as
`estimator_loss` so it can be plotted across updates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from RL.models.policy        import make_snapshot
from scenarios.eval.adapter  import GameEnvAdapter
from scenarios.eval.runner   import ScenarioRunner, RunnerResult


class Runner(ScenarioRunner):
    n_samples      = 1            # single forward pass; no averaging
    n_decisions    = 0            # no rollout, no actions
    render_enabled = True         # full dual-POV figure with hidden overlays

    def play(self, policy, scenario, device) -> RunnerResult:
        adapter = GameEnvAdapter(scenario)
        obs     = adapter.reset()

        # ── Estimator loss (uses a List[snap] for batch shape) ────────────────
        snap = make_snapshot(obs, adapter.Nx, adapter.Ny, adapter.game.player_go_id)
        loss_t   = policy.estimator_loss([snap])
        loss_val = float(loss_t.item())

        # ── Hidden-tile inference for the dual-POV render ────────────────────
        est_a, est_b = policy.estimate_hidden_dual(obs)
        est_pair = (np.asarray(est_a), np.asarray(est_b))

        return RunnerResult(
            metrics = {
                "estimator_loss": loss_val,
                "n_samples":      1,
            },
            metrics_extra = {"hidden_estimate": est_pair},
            title         = f"estimate_lakes11 — estimator loss = {loss_val:.4f}",
        )

    # ── render() ─────────────────────────────────────────────────────────────

    def render(
        self,
        scenario,
        result:   RunnerResult,
        out_path: Path,
    ) -> None:
        """Dual-POV render with the hidden-tile estimator overlay.

        Rebuilds the adapter so EnvWrapper.render has a live game state that
        matches the same scenario the estimate was computed against. The
        estimate itself is reused verbatim — it doesn't depend on the
        rebuilt game.
        """
        est_pair = result.metrics_extra.get("hidden_estimate")
        if est_pair is None:
            return  # nothing to draw

        adapter = GameEnvAdapter(scenario)
        adapter.reset()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        adapter.env.render(
            show_hidden     = True,
            hidden_estimate = est_pair,
            save_path       = str(out_path),
            show            = False,
        )
