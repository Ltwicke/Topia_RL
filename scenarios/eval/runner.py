"""
scenarios/eval/runner.py
──────────────────────────────────────────────────────────────────────────────
`ScenarioRunner` — base class for per-scenario eval logic.

Subclass per scenario in `scenarios/configs/<name>.py`:

    from scenarios.eval.runner import ScenarioRunner

    class Runner(ScenarioRunner):
        n_samples   = 20      # how many forward-pass samples to aggregate
        n_decisions = 1       # decisions per single rollout
        # Override play() / render() if the defaults aren't enough.

The base class ships:
  • `play(...)`   — default: N forward passes from the original state, average
                    `joint_probs`, return a RunnerResult with prob_overlay /
                    atype_probs ready for the renderer.
  • `render(...)` — default: BoardRenderer.draw onto a single-figure PNG with
                    the averaged overlays, save to disk.

Helpers that subclasses can mix-and-match:
  • `_one_forward(adapter, policy)`     — single policy.forward call, returns a
                                          DecisionRecord.
  • `_one_rollout(adapter, policy, n)`  — n sequential decisions on the same
                                          adapter, returning list[DecisionRecord].
  • `_average_joint_probs(records)`     — average across decisions whose
                                          (atype, traj_actions) match.

Future scenarios that need anything fancier (custom histograms, per-rollout
metrics, side-by-side renders, …) override `play()` and/or `render()`
fully — the helpers are optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional, Tuple

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt

from env.renderer       import BoardRenderer
from scenarios.eval.adapter import GameEnvAdapter
from scenarios.scenario import Scenario


# ══════════════════════════════════════════════════════════════════════════════
# Result type
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DecisionRecord:
    """One policy.forward call's output."""
    action:        list                         # sampled action, e.g. [Move, uid, tid]
    joint_probs:   np.ndarray                   # shape (n_traj,) probabilities
    traj_actions:  list                         # parallel list of trajectories
    log_prob:      float
    value:         float


@dataclass
class RunnerResult:
    """Anything the runner wants to surface for rendering + logging.

    Everything beyond `metrics` is render-only payload — `metrics` is the only
    field the harness writes to the per-update summary CSV.
    """
    prob_overlay:   Optional[dict]    = None     # tile_id → alpha (renderer-ready)
    atype_probs:    Optional[dict]    = None     # ActionType.name → float
    final_snapshot: Optional[dict]    = None     # for inspect-style readouts
    sampled_action: Optional[list]    = None     # last sampled action
    metrics:        dict              = field(default_factory=dict)
                                                 # scalar values → summary.csv
    metrics_extra:  dict              = field(default_factory=dict)
                                                 # arbitrary payload for render()
    title:          str               = ""


# ══════════════════════════════════════════════════════════════════════════════
# Base runner
# ══════════════════════════════════════════════════════════════════════════════

class ScenarioRunner:
    """Default behaviour: N-sample joint_probs averaging + board+overlay render.

    Subclasses override class attrs for simple knob changes and override the
    methods for arbitrary logic.
    """

    # ── Knobs (subclass-overridable) ──────────────────────────────────────────

    n_samples:      int  = 20     # forward-pass samples to aggregate
    n_decisions:    int  = 1      # sequential decisions per rollout
    render_enabled: bool = True   # if False, ScenarioBank skips render() entirely
                                  # and no PNG is written. The metrics dict still
                                  # flows into summary.csv as usual.

    # ── play() ───────────────────────────────────────────────────────────────

    def play(
        self,
        policy:   torch.nn.Module,
        scenario: Scenario,
        device:   torch.device,
    ) -> RunnerResult:
        """
        Default play: from the scenario's starting state, run policy.forward
        N=`self.n_samples` times (resetting between each call). Average the
        per-trajectory probabilities; condition the prob_overlay on the most
        recent sampled action.

        Suitable for `n_decisions=1` "what would the policy do here" scenarios
        like Knight_chain_choice. Override for multi-decision rollouts or
        readouts beyond the prob heatmap.
        """
        adapter = GameEnvAdapter(scenario)
        records: List[DecisionRecord] = []
        for _ in range(self.n_samples):
            adapter.reset()
            rec = self._one_forward(adapter, policy)
            records.append(rec)

        # Average across calls (game state is identical each time, so
        # traj_actions enumerations match).
        avg_probs, traj_actions = self._average_joint_probs(records)
        last_action = records[-1].action

        # Build renderer-ready overlays from the averaged distribution.
        renderer = BoardRenderer(adapter.env)
        avg_probs_t = torch.from_numpy(avg_probs)
        prob_overlay, atype_probs = renderer.compute_prob_overlay(
            last_action, avg_probs_t, traj_actions,
        )

        return RunnerResult(
            prob_overlay   = prob_overlay,
            atype_probs    = atype_probs,
            sampled_action = last_action,
            metrics        = {
                "n_samples":    int(self.n_samples),
                "n_decisions":  int(self.n_decisions),
                "n_trajectories": int(len(traj_actions)),
            },
            title          = (
                f"{scenario.name} — averaged over {self.n_samples} samples"
            ),
        )

    # ── render() ─────────────────────────────────────────────────────────────

    def render(
        self,
        scenario: Scenario,
        result:   RunnerResult,
        out_path: Path,
    ) -> None:
        """
        Default render: a single board figure with the averaged probability
        overlay + atype-probability bar chart in the side info panel.

        Subclasses with metrics-only readouts can override this to a no-op or
        produce any matplotlib figure (histogram, scatter, side-by-side, …).
        """
        adapter  = GameEnvAdapter(scenario)
        renderer = BoardRenderer(adapter.env)

        fig = plt.figure(figsize=(13, 7))
        gs  = fig.add_gridspec(1, 2, width_ratios=[adapter.Ny, 4.2], wspace=0.08)
        ax_board = fig.add_subplot(gs[0])
        ax_info  = fig.add_subplot(gs[1])

        # Render from the active player's POV.
        active_pid = adapter.game.player_go_id
        uncovered  = set(adapter.game.players[active_pid].uncovered_tile_ids)
        renderer.draw(
            ax           = ax_board,
            ax_info      = ax_info,
            uncovered    = uncovered,
            prob_overlay = result.prob_overlay or None,
            atype_probs  = result.atype_probs  or None,
            title        = result.title or scenario.name,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)

    # ── Helpers (protected — for subclasses to mix) ───────────────────────────

    @staticmethod
    def _one_forward(
        adapter: GameEnvAdapter,
        policy:  torch.nn.Module,
    ) -> DecisionRecord:
        """Run one policy.forward on the adapter's current state. Does NOT
        advance the adapter — caller decides whether to call adapter.step()."""
        obs  = adapter.get_obs()
        mask = adapter.get_action_mask()
        action, joint_probs, traj_actions, log_prob, _entropy, value = policy(obs, mask)
        # Normalise: detach+cpu+numpy for joint_probs so subclasses can do math.
        try:
            jp_np = joint_probs.detach().cpu().numpy()
        except AttributeError:
            jp_np = np.asarray(joint_probs)
        try:
            lp_f = float(log_prob.item())
        except AttributeError:
            lp_f = float(log_prob)
        try:
            v_f  = float(value.item())
        except AttributeError:
            v_f  = float(value)
        return DecisionRecord(
            action       = list(action),
            joint_probs  = jp_np,
            traj_actions = list(traj_actions),
            log_prob     = lp_f,
            value        = v_f,
        )

    def _one_rollout(
        self,
        adapter:     GameEnvAdapter,
        policy:      torch.nn.Module,
        n_decisions: Optional[int] = None,
    ) -> List[DecisionRecord]:
        """Execute `n_decisions` sequential decisions, advancing the adapter
        with `adapter.step` after each. Returns one DecisionRecord per
        decision in order. Caller is expected to have called `adapter.reset()`
        beforehand."""
        n = self.n_decisions if n_decisions is None else n_decisions
        out: List[DecisionRecord] = []
        for _d in range(n):
            rec = self._one_forward(adapter, policy)
            out.append(rec)
            _obs, _r, done, _info = adapter.step(rec.action)
            if done:
                break
        return out

    @staticmethod
    def _average_joint_probs(
        records: List[DecisionRecord],
    ) -> Tuple[np.ndarray, list]:
        """Average `joint_probs` across records that share the same trajectory
        enumeration. Returns (avg_probs, traj_actions). Falls back to the
        first record's probs if the enumerations diverge (shouldn't happen
        when the game state is identical, but cheap to guard)."""
        if not records:
            return np.zeros(0, dtype=np.float32), []
        first = records[0]
        n_traj = len(first.traj_actions)
        # Sanity check: every record should have the same trajectory list.
        consistent = all(
            len(r.traj_actions) == n_traj
            and r.joint_probs.shape == first.joint_probs.shape
            for r in records
        )
        if not consistent:
            return first.joint_probs, first.traj_actions
        stacked = np.stack([r.joint_probs for r in records], axis=0)  # (N, n_traj)
        avg     = stacked.mean(axis=0).astype(np.float32)
        return avg, first.traj_actions
