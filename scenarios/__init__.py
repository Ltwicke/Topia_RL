"""
scenarios — interactive scenario editor and (later) eval harness.

Public surface:
  Scenario                 — in-memory dataclass for a complete game state
  ScenarioPlayer / City / Unit — nested dataclasses
"""

from scenarios.scenario import (
    Scenario,
    ScenarioPlayer,
    ScenarioCity,
    ScenarioUnit,
)

__all__ = [
    "Scenario",
    "ScenarioPlayer",
    "ScenarioCity",
    "ScenarioUnit",
]
