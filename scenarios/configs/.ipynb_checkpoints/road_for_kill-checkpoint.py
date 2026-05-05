"""
road_for_kill — P1 has a road network leading toward P0's lone Warrior at
(8, 4). The agent (playing as P1) makes up to 8 decisions OR stops as soon
as it selects EndTurn; success = P0's Warrior dies.

Readout: a single stacked bar showing the kill / no-kill ratio across
N=20 rollouts. No board render.
"""

from __future__ import annotations

from pathlib import Path
from typing  import List

import numpy as np
import matplotlib.pyplot as plt

from game.enums              import ActionTypes, UnitType
from scenarios.eval.adapter  import GameEnvAdapter
from scenarios.eval.runner   import ScenarioRunner, RunnerResult


class Runner(ScenarioRunner):
    n_samples   = 20
    n_decisions = 8       # max — rollout also stops on EndTurn

    # Whose Warrior we're trying to kill.
    target_player_id = 0

    # ── Custom rollout ────────────────────────────────────────────────────────

    def _play_one_rollout(self, adapter: GameEnvAdapter, policy) -> None:
        """Up to `self.n_decisions` decisions, breaking on EndTurn (without
        applying it) or on game termination."""
        end_turn_atype = int(ActionTypes.EndTurn)
        for _d in range(self.n_decisions):
            rec = self._one_forward(adapter, policy)
            if int(rec.action[0]) == end_turn_atype:
                return
            _obs, _r, done, _info = adapter.step(rec.action)
            if done:
                return

    def _p0_warrior_ids(self, adapter: GameEnvAdapter) -> set:
        """Collect the unit_ids of every Warrior owned by `target_player_id`
        in the live game. Re-captured every reset because unit_ids are
        randomised by `Game._new_unit_id`."""
        player = adapter.game.players[self.target_player_id]
        return {
            uid for uid, unit in player.units_under_control.items()
            if unit.unit_type == UnitType.Warrior
        }

    # ── play() ───────────────────────────────────────────────────────────────

    def play(self, policy, scenario, device) -> RunnerResult:
        adapter = GameEnvAdapter(scenario)
        successes: List[bool] = []
        decisions_taken: List[int] = []

        for _ in range(self.n_samples):
            adapter.reset()
            initial_uids = self._p0_warrior_ids(adapter)
            decisions_before = adapter.env.n_decisions

            self._play_one_rollout(adapter, policy)

            decisions_taken.append(adapter.env.n_decisions - decisions_before)
            survivors = self._p0_warrior_ids(adapter) & initial_uids
            killed    = len(survivors) < len(initial_uids) # in this scenario, there is only one enemy unit
            successes.append(killed)

        n_killed   = int(sum(successes))
        n          = len(successes)
        successes_arr = np.asarray(successes, dtype=np.int32)

        return RunnerResult(
            metrics = {
                "success_rate":          float(n_killed / max(n, 1)),
                "n_killed":              n_killed,
                "n_samples":             n,
                "avg_decisions_taken":   float(np.mean(decisions_taken)),
                "n_decisions_max":       int(self.n_decisions),
            },
            metrics_extra = {"successes": successes},
            title = (
                f"road_for_kill — P0 Warrior killed in "
                f"{n_killed}/{n} rollouts ({100 * n_killed / max(n, 1):.0f}%)"
            ),
        )

    # ── render() ─────────────────────────────────────────────────────────────

    def render(
        self,
        scenario,
        result:   RunnerResult,
        out_path: Path,
    ) -> None:
        """Single horizontal stacked bar: killed (crimson) | alive (gray)."""
        successes = result.metrics_extra.get("successes", [])
        n         = len(successes)
        n_killed  = int(sum(successes))
        n_alive   = n - n_killed

        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.barh([0], [n_killed], color="crimson",   edgecolor="black",
                label=f"killed  (n={n_killed})")
        ax.barh([0], [n_alive],  left=[n_killed], color="#bcbcbc",
                edgecolor="black",
                label=f"alive   (n={n_alive})")

        # Annotate counts inside the bars when there's enough room.
        if n_killed > 0:
            ax.text(n_killed / 2, 0, f"{n_killed}",
                    ha="center", va="center", color="white", fontweight="bold")
        if n_alive > 0:
            ax.text(n_killed + n_alive / 2, 0, f"{n_alive}",
                    ha="center", va="center", color="black", fontweight="bold")

        ax.set_xlim(0, max(n, 1))
        ax.set_yticks([])
        ax.set_xlabel(f"# rollouts (N = {n})")
        ax.set_title(result.title or scenario.name, fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
