import numpy as np
import pytest

from game.enums import BoardType, Tribes, ActionTypes
from env.wrapper import EnvWrapper

BOARD_CONFIG_9x9 = {"board_size": (9, 9), "board_type": BoardType.Dummy, "n_players": 2}
TRIBES = [Tribes.Omaji, Tribes.Yaddak]
N_GAMES = 5
MAX_DECISIONS = 200


def _make_env(seed=None):
    if seed is not None:
        np.random.seed(seed)
    env = EnvWrapper(BOARD_CONFIG_9x9, TRIBES)
    env.reset()
    return env


def _select_random_matrix_element(mat):
    nonzero_rows = np.where(mat.any(axis=1))[0]
    row_idx = np.random.choice(nonzero_rows)
    col_idx = np.random.choice(np.flatnonzero(mat[row_idx]))
    return int(row_idx), int(col_idx)


def _select_random_array_element(arr):
    return int(np.random.choice(np.flatnonzero(arr)))


def _random_action_from_mask(mask):
    """Build a valid random action list from a fresh action mask."""
    action_type = ActionTypes(int(np.random.choice(np.flatnonzero(mask[0]))))

    if action_type == ActionTypes.MoveUnit:
        unit_pos, tile_id = _select_random_matrix_element(mask[1])
        return [ActionTypes.MoveUnit.value, unit_pos, tile_id]

    elif action_type == ActionTypes.Attack:
        attacker_pos, defender_pos = _select_random_matrix_element(mask[2])
        return [ActionTypes.Attack.value, attacker_pos, defender_pos]

    elif action_type == ActionTypes.CreateUnit:
        city_idx, unit_type = _select_random_matrix_element(mask[3])
        return [ActionTypes.CreateUnit.value, city_idx, unit_type]

    elif action_type == ActionTypes.CaptureCity:
        unit_pos = _select_random_array_element(mask[4])
        return [ActionTypes.CaptureCity.value, unit_pos]

    else:  # EndTurn
        return [ActionTypes.EndTurn.value]


@pytest.mark.parametrize("game_seed", range(N_GAMES))
def test_random_game_no_exception(game_seed):
    """Play a full random game (mask refreshed every step) and assert no exceptions."""
    env = _make_env(seed=game_seed)
    decisions = 0

    while decisions < MAX_DECISIONS:
        mask = env.get_action_mask()  # refresh mask EVERY step
        action = _random_action_from_mask(mask)
        obs, reward, done, info = env.step(action)
        decisions += 1
        if done:
            break

    assert decisions <= MAX_DECISIONS
