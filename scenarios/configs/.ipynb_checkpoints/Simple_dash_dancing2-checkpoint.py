"""
Simple_dash_dancing2 — the policy makes one decision; we render the board
with an averaged probability overlay across 20 samples.

Uses the default ScenarioRunner behaviour.
"""

from scenarios.eval.runner import ScenarioRunner


class Runner(ScenarioRunner):
    n_samples   = 20
    n_decisions = 1
