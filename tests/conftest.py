import random
import pytest
import numpy as np
from game.enums import BoardType, Tribes
from game.game import Game

BOARD_CONFIG = {"board_size": (8, 8), "board_type": BoardType.Dummy, "n_players": 2}
TRIBES = [Tribes.Omaji, Tribes.Yaddak]


def _seed_all(seed=42):
    np.random.seed(seed)
    random.seed(seed)


@pytest.fixture
def fresh_game():
    _seed_all()
    g = Game(BOARD_CONFIG, TRIBES)
    g.reset_game()
    return g


@pytest.fixture
def fresh_env():
    from env.wrapper import EnvWrapper
    _seed_all()
    env = EnvWrapper(BOARD_CONFIG, TRIBES)
    env.reset()
    return env
