"""
scenarios/eval/bank.py
──────────────────────────────────────────────────────────────────────────────
ScenarioBank — discovers per-scenario configs, dispatches them, manages output.

Lifecycle
─────────
  bank = ScenarioBank("scenarios/scenarios", ["Knight_chain_choice", ...])
  metrics = bank.run(policy, device, output_dir=Path("logs/scenarios/update_42"))

`metrics` is `{scenario_name: {metric_key: value, ...}}` — the harness's caller
also writes a per-update `summary.csv` row by calling
`bank.append_summary_csv(csv_path, update, metrics)`.

Discovery
─────────
For each name in `scenario_names`, the bank loads:
  scenarios/scenarios/<name>.yaml         — required (the world state)
  scenarios/configs/<name>.py             — optional (custom Runner). If
                                            absent or import fails, the
                                            default `ScenarioRunner` is used.

Robustness
──────────
A scenario whose play() / render() raises is logged + skipped, not allowed
to crash training. The harness records `{name: {"_error": "<traceback>"}}`
for diagnostics.
"""

from __future__ import annotations

import csv
import importlib
import logging
import time
import traceback
from pathlib import Path
from typing  import Dict, List, Tuple

import torch

from scenarios.scenario        import Scenario
from scenarios.eval.runner     import ScenarioRunner

_log = logging.getLogger("polytopia_rl")


# ══════════════════════════════════════════════════════════════════════════════
# ScenarioBank
# ══════════════════════════════════════════════════════════════════════════════

class ScenarioBank:
    """Loads N scenarios + their runners; runs them sequentially against a
    policy; saves render PNGs + returns a metrics dict for CSV logging."""

    def __init__(
        self,
        scenario_dir:   str | Path,
        scenario_names: List[str],
    ) -> None:
        self.scenario_dir = Path(scenario_dir)
        self.entries: List[Tuple[str, Scenario, ScenarioRunner]] = []
        for name in scenario_names:
            self._load_one(name)

    def _load_one(self, name: str) -> None:
        yaml_path = self.scenario_dir / f"{name}.yaml"
        if not yaml_path.exists():
            _log.warning(f"[scenarios] missing YAML: {yaml_path} — skipping")
            return
        try:
            scenario = Scenario.from_yaml(yaml_path)
        except Exception:
            _log.error(
                f"[scenarios] failed to load {yaml_path}:\n{traceback.format_exc()}"
            )
            return

        runner = self._load_runner(name)
        self.entries.append((name, scenario, runner))

    @staticmethod
    def _load_runner(name: str) -> ScenarioRunner:
        """Import scenarios.configs.<name>; return its `Runner()` instance.
        Falls back to the default ScenarioRunner if the module / class is
        missing."""
        mod_path = f"scenarios.configs.{name}"
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            return ScenarioRunner()
        runner_cls = getattr(mod, "Runner", None)
        if runner_cls is None:
            _log.warning(
                f"[scenarios] {mod_path} has no `Runner` class — using default"
            )
            return ScenarioRunner()
        try:
            return runner_cls()
        except Exception:
            _log.error(
                f"[scenarios] {mod_path}.Runner() failed:\n{traceback.format_exc()}"
            )
            return ScenarioRunner()

    # ── Public dispatch ──────────────────────────────────────────────────────

    def run(
        self,
        policy:     torch.nn.Module,
        device:     torch.device,
        output_dir: Path,
    ) -> Dict[str, Dict]:
        """
        Run every loaded scenario sequentially. For each, save a render PNG
        into `output_dir/<name>.png` and collect its `metrics` dict.

        Returns: {scenario_name: metrics_dict}. On failure: {"_error": "..."}.

        Sets the policy to eval() during the call and restores train() at the
        end. All forward passes run under torch.no_grad().
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Dict] = {}
        was_training = policy.training
        policy.eval()
        try:
            with torch.no_grad():
                for name, scenario, runner in self.entries:
                    t0 = time.time()
                    try:
                        result = runner.play(policy, scenario, device)
                        if getattr(runner, "render_enabled", True):
                            runner.render(scenario, result,
                                          output_dir / f"{name}.png")
                    except Exception:
                        tb = traceback.format_exc()
                        _log.error(f"[scenarios] {name} failed:\n{tb}")
                        results[name] = {"_error": tb.splitlines()[-1]}
                        continue
                    dt = time.time() - t0
                    metrics = dict(result.metrics)
                    metrics["t_sec"] = float(dt)
                    results[name] = metrics
                    _log.info(
                        f"[scenarios] {name:30s} done in {dt:.2f}s — "
                        f"metrics={metrics}"
                    )
        finally:
            if was_training:
                policy.train()

        return results

    # ── Per-update summary CSV ───────────────────────────────────────────────

    @staticmethod
    def append_summary_csv(
        csv_path: Path,
        update:   int,
        metrics:  Dict[str, Dict],
    ) -> None:
        """
        Append one row per scenario per update. Column set is the union of all
        scenarios' metric keys ever seen — sparse cells are blank. We rewrite
        the header on every call so newly-added metrics automatically appear.
        """
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing rows + header to build a complete column set.
        existing_rows: List[Dict[str, str]] = []
        existing_cols: List[str] = []
        if csv_path.exists() and csv_path.stat().st_size > 0:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_cols = list(reader.fieldnames or [])
                existing_rows = [row for row in reader]

        # New rows for this update.
        new_rows: List[Dict[str, str]] = []
        for name, m in metrics.items():
            row: Dict[str, str] = {"update": str(update), "scenario": name}
            for k, v in m.items():
                row[k] = "" if v is None else str(v)
            new_rows.append(row)

        # Union of columns. Always start with update + scenario.
        col_set = set(existing_cols) | {"update", "scenario"}
        for row in new_rows:
            col_set.update(row.keys())
        # Stable column order: update, scenario, then other keys sorted.
        ordered = ["update", "scenario"] + sorted(col_set - {"update", "scenario"})

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ordered)
            writer.writeheader()
            for row in existing_rows + new_rows:
                writer.writerow({k: row.get(k, "") for k in ordered})
