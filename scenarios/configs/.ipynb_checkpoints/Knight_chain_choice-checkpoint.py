"""
Knight_chain_choice — P1's red Knight is the only `ready` unit. The policy
makes one decision; we render the board with an averaged probability overlay
showing the most-likely move targets across 20 forward-pass samples.

Uses the default ScenarioRunner behaviour (n_samples=20, n_decisions=1,
board+overlay render).
"""

from scenarios.eval.runner import ScenarioRunner


class Runner(ScenarioRunner):
    n_samples   = 20
    n_decisions = 1
