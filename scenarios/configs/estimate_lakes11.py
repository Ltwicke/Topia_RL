"""
estimate_lakes11 — single-shot evaluation of the HiddenTileEstimator on a
Lakes 11×11 board.

Behaviour comes from `EstimatorRunner` (in scenarios/eval/runner.py):
  1. Reset the scenario to its YAML state.
  2. `policy.estimate_hidden_dual(obs)` → (est_a, est_b).
  3. `policy.estimator_loss([snap])` → scalar loss.
  4. Render the dual-POV figure with `show_hidden=True`.

Metric: `estimator_loss` (per-tile-normalised cross-entropy + BCE).
"""

from scenarios.eval.runner import EstimatorRunner


class Runner(EstimatorRunner):
    pass
