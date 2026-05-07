"""
Estimate_Drylands_endgame — single-shot evaluation of the
HiddenTileEstimator on a 16×16 Drylands endgame state from P0's POV.

Same behaviour as `estimate_lakes11` (and any other 'estimate_*' scenario):
  1. Reset → run estimator inference + loss once.
  2. Render dual-POV with `show_hidden=True` so both players' hidden-tile
     predictions are overlaid on the board.

Metric: `estimator_loss`.
"""

from scenarios.eval.runner import EstimatorRunner


class Runner(EstimatorRunner):
    pass
