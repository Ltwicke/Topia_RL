"""
scenarios/eval/adapter.py
──────────────────────────────────────────────────────────────────────────────
Thin wrapper that lets a scenario evaluation reset to its original state
without rebuilding the EnvWrapper from scratch on every "try".

Building an EnvWrapper allocates its own Board / Game and runs map
generation. For 20-sample averaging this would dominate cost. Instead we
build the EnvWrapper once and swap `self.game` on each reset.
"""

from __future__ import annotations

from typing import Tuple

from env.wrapper       import EnvWrapper
from scenarios.scenario import Scenario


class GameEnvAdapter:
    """
    Wraps an EnvWrapper so its `.game` attribute can be replaced with a
    fresh `Scenario.to_game()` instance on every `reset()`.

    The adapter delegates `step` / `_get_obs` / `get_action_mask` /
    `last_action` / `_overlay_ctx` to the underlying EnvWrapper, but `reset`
    rebuilds the Game from the source scenario rather than re-running map
    generation.

    The renderer (BoardRenderer) reads `env.game`, `env.Nx`, `env.Ny`,
    `env.n_players`, `env.n_decisions`, `env.last_action`, and
    `env._overlay_ctx`. All of those live on the underlying EnvWrapper.
    """

    def __init__(
        self,
        scenario:           Scenario,
        max_turns_per_game: int = 999,
    ) -> None:
        self.scenario = scenario
        # Build the wrapper once with whatever board_type fits — the terrain
        # gets overwritten on reset() anyway.
        from game.enums import BoardType
        board_cfg = {
            "board_size": list(scenario.board_size),
            "board_type": BoardType.Drylands,
            "n_players":  scenario.n_players,
        }
        self._env = EnvWrapper(
            board_cfg,
            list(scenario.player_tribes),
            max_turns_per_game = max_turns_per_game,
            dense_reward       = True,
        )
        # First reset puts the scenario state in.
        self.reset()

    # ── Public surface — feed a renderer / policy ────────────────────────

    @property
    def env(self) -> EnvWrapper:
        return self._env

    @property
    def game(self):
        return self._env.game

    @property
    def Nx(self) -> int: return self._env.Nx
    @property
    def Ny(self) -> int: return self._env.Ny
    @property
    def n_players(self) -> int: return self._env.n_players

    # ── Reset / step ─────────────────────────────────────────────────────

    def reset(self) -> dict:
        """Rebuild the Game from the source scenario; return the obs dict."""
        self._env.game        = self.scenario.to_game()
        self._env.last_action = None
        self._env.n_decisions = 0
        self._env._overlay_ctx = {}
        self._env.winner      = None
        return self._env._get_obs()

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        """Apply an action via the underlying EnvWrapper.step path."""
        return self._env.step(action)

    def get_action_mask(self):
        return self._env.get_action_mask()

    def get_obs(self) -> dict:
        return self._env._get_obs()
