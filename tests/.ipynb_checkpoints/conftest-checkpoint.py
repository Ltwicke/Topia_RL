import pytest
import numpy as np
from game.enums import BoardType, Tribes
from game.game import Game

BOARD_CONFIG = {"board_size": (8, 8), "board_type": BoardType.Dummy, "n_players": 2}
TRIBES = [Tribes.Omaji, Tribes.Yaddak]


@pytest.fixture
def fresh_game():
    np.random.seed(42)
    g = Game(BOARD_CONFIG, TRIBES)
    g.reset_game()
    return g


@pytest.fixture
def fresh_env():
    from env.wrapper import EnvWrapper
    env = EnvWrapper(BOARD_CONFIG, TRIBES)
    env.reset()
    return env
