"""
scenarios.configs — per-scenario Runner classes.

Each file `<name>.py` defines a `class Runner(ScenarioRunner): ...` matching
the YAML at `scenarios/scenarios/<name>.yaml`. Discovery is by filename:
`scenarios.eval.bank.ScenarioBank` imports `scenarios.configs.<name>` and
uses its `Runner` class. A scenario without a config falls back to the
default `ScenarioRunner` (N-sample joint_probs averaging, board+overlay
render).
"""
