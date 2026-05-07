"""
scenarios.eval — periodic-eval harness for trained policies.

Public surface:
  ScenarioRunner   — base class; subclass per scenario in scenarios/configs/<name>.py
  RunnerResult     — what `play()` returns and `render()` consumes
  DecisionRecord   — one policy.forward call's output (used by helpers)
  GameEnvAdapter   — thin wrapper letting a scenario reset cheaply between samples
  ScenarioBank     — loads N scenarios, dispatches them, manages output
"""

from scenarios.eval.adapter import GameEnvAdapter
from scenarios.eval.bank    import ScenarioBank
from scenarios.eval.runner  import (
    DecisionRecord,
    EstimatorRunner,
    RunnerResult,
    ScenarioRunner,
)

__all__ = [
    "DecisionRecord",
    "EstimatorRunner",
    "GameEnvAdapter",
    "RunnerResult",
    "ScenarioBank",
    "ScenarioRunner",
]
