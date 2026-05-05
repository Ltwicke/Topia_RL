"""
Rider_leapfrogging — P0 has two riders. The "leapfrog" pattern is: rider A
moves forward (covering new ground); rider B then moves past A, uncovering
even more. Sub-optimal play uncovers fewer tiles.

Readout is a histogram of Δ(uncovered tiles) across N=20 two-decision
rollouts. No board render — the histogram captures the entire signal.
"""

from __future__ import annotations

from pathlib import Path
from typing  import List

import numpy as np
import matplotlib.pyplot as plt

from scenarios.eval.adapter import GameEnvAdapter
from scenarios.eval.runner  import ScenarioRunner, RunnerResult


class Runner(ScenarioRunner):
    n_samples      = 20
    n_decisions    = 2
    render_enabled = False        # metrics-only — no histogram PNG

    # Player whose uncovered count we track. Hard-coded to 0 because the
    # scenario is authored for P0 to act. (If we ever want this configurable
    # we can lift it onto the YAML, but YOLO for now.)
    pov_player_id = 0

    def play(self, policy, scenario, device) -> RunnerResult:
        adapter = GameEnvAdapter(scenario)

        deltas: List[int] = []
        for _ in range(self.n_samples):
            adapter.reset()
            unc_before = len(
                adapter.game.players[self.pov_player_id].uncovered_tile_ids
            )
            self._one_rollout(adapter, policy, n_decisions=self.n_decisions)
            unc_after = len(
                adapter.game.players[self.pov_player_id].uncovered_tile_ids
            )
            deltas.append(int(unc_after - unc_before))

        deltas_np = np.asarray(deltas, dtype=np.int32)
        return RunnerResult(
            metrics = {
                "uncovered_delta_mean":   float(deltas_np.mean()),
                "uncovered_delta_std":    float(deltas_np.std()),
                "uncovered_delta_max":    int(deltas_np.max()),
                "uncovered_delta_min":    int(deltas_np.min()),
                "n_samples":              int(self.n_samples),
                "n_decisions":            int(self.n_decisions),
            },
            metrics_extra = {"deltas": deltas},
            title = (
                f"Rider leapfrogging — Δuncovered  "
                f"mean={deltas_np.mean():.1f}  "
                f"min={deltas_np.min()}  max={deltas_np.max()}"
            ),
        )

    def render(
        self,
        scenario,
        result:   RunnerResult,
        out_path: Path,
    ) -> None:
        deltas = result.metrics_extra.get("deltas", [])
        fig, ax = plt.subplots(figsize=(7, 4))
        if deltas:
            # Integer bins, one per observed value, centred on the integer.
            lo, hi = min(deltas), max(deltas)
            bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
            ax.hist(deltas, bins=bins, edgecolor="black", alpha=0.85,
                    color="#4a7eb6")
            ax.axvline(np.mean(deltas), color="crimson", linestyle="--",
                       linewidth=1.5, label=f"mean={np.mean(deltas):.1f}")
            ax.legend(loc="upper right", fontsize=9)
        ax.set_xlabel("Δ uncovered tiles after 2 decisions")
        ax.set_ylabel("# rollouts")
        ax.set_title(result.title or scenario.name, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
