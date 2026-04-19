"""
Mask observation tests — verify that the action mask is internally consistent
and coupled to the partial graph / board state.
"""
import numpy as np
import pytest

from game.enums import ActionTypes, BoardType, Tribes, UnitType, UnitState
from env.wrapper import EnvWrapper

BOARD_CONFIG = {"board_size": (8, 8), "board_type": BoardType.Dummy, "n_players": 2}
TRIBES = [Tribes.Omaji, Tribes.Yaddak]


@pytest.fixture
def fresh_env():
    np.random.seed(42)
    env = EnvWrapper(BOARD_CONFIG, TRIBES)
    env.reset()
    return env


# ---------------------------------------------------------------------------
# Structural / sum tests
# ---------------------------------------------------------------------------

def test_mask_has_correct_components(fresh_env):
    mask = fresh_env.get_action_mask()
    assert len(mask) == len(ActionTypes)


def test_end_turn_always_valid(fresh_env):
    mask = fresh_env.get_action_mask()
    assert mask[0][ActionTypes.EndTurn] == 1.0


def test_action_type_mask_is_not_all_zero(fresh_env):
    mask = fresh_env.get_action_mask()
    assert mask[0].sum() >= 1  # at minimum EndTurn is valid


def test_move_mask_shape(fresh_env):
    mask = fresh_env.get_action_mask()
    player = fresh_env.game.players[fresh_env.game.player_go_id]
    n_units = len(player.units_under_control)
    n_tiles = fresh_env.n_tiles
    assert mask[1].shape == (n_units, n_tiles)


def test_attack_mask_shape_equals_visible_enemies(fresh_env):
    mask = fresh_env.get_action_mask()
    player = fresh_env.game.players[fresh_env.game.player_go_id]
    n_units = len(player.units_under_control)
    n_visible = len(fresh_env._visible_enemy_units())
    assert mask[2].shape == (n_units, n_visible)


def test_create_mask_shape(fresh_env):
    mask = fresh_env.get_action_mask()
    player = fresh_env.game.players[fresh_env.game.player_go_id]
    n_cities = len(player.cities_under_control)
    assert mask[3].shape == (n_cities, len(UnitType))


def test_capture_mask_shape(fresh_env):
    mask = fresh_env.get_action_mask()
    player = fresh_env.game.players[fresh_env.game.player_go_id]
    n_units = len(player.units_under_control)
    assert mask[4].shape == (n_units,)


# ---------------------------------------------------------------------------
# Coupling: mask values consistent with partial graph
# ---------------------------------------------------------------------------

def test_attack_targets_are_visible_in_partial_graph(fresh_env):
    """Every column in the attack mask must correspond to a unit visible
    in the current player's uncovered tiles."""
    mask = fresh_env.get_action_mask()
    visible_enemies = fresh_env._visible_enemy_units()

    # If there are any nonzero entries in the attack mask,
    # the indexed enemies must actually be visible.
    if mask[2].sum() > 0:
        reachable_defender_positions = np.unique(np.where(mask[2] > 0)[1])
        player = fresh_env.game.players[fresh_env.game.player_go_id]
        uncovered = player.uncovered_tile_ids

        for def_pos in reachable_defender_positions:
            enemy = visible_enemies[def_pos]
            assert enemy.tile.id in uncovered, (
                f"Attack target at position {def_pos} (tile {enemy.tile.id}) "
                f"is not in the player's uncovered tiles."
            )


def test_move_targets_are_valid_tiles(fresh_env):
    """Every tile set to 1 in the move mask must be a tile ID within range."""
    mask = fresh_env.get_action_mask()
    n_tiles = fresh_env.n_tiles

    nonzero = np.where(mask[1] > 0)
    for tile_id in nonzero[1]:
        assert 0 <= tile_id < n_tiles, f"Move target tile_id {tile_id} out of range."


def test_move_mask_zeros_for_idle_units(fresh_env):
    """Units in idle or can_hit state must have no move targets in the mask."""
    mask = fresh_env.get_action_mask()
    player = fresh_env.game.players[fresh_env.game.player_go_id]
    player_units = list(player.units_under_control.values())

    for pos, unit in enumerate(player_units):
        if unit.turn_state not in (UnitState.ready, UnitState.escaping):
            assert mask[1][pos].sum() == 0, (
                f"Unit at pos {pos} (state {unit.turn_state}) should have no move targets."
            )


def test_attack_mask_zero_when_no_visible_enemies(fresh_env):
    """If no enemy units are visible, the attack mask must be all zeros."""
    player = fresh_env.game.players[fresh_env.game.player_go_id]
    # Manually clear uncovered tiles so no enemy is visible
    original = player.uncovered_tile_ids.copy()
    player.uncovered_tile_ids = {fresh_env.game.game_board.capital_tile_ids[fresh_env.game.player_go_id]}

    mask = fresh_env.get_action_mask()
    assert mask[2].sum() == 0, "Attack mask should be all zeros when no enemies visible."

    player.uncovered_tile_ids = original  # restore


def test_attack_mask_col_count_matches_visible_enemy_count(fresh_env):
    """Number of columns in attack mask == number of visible enemy units."""
    visible = fresh_env._visible_enemy_units()
    mask = fresh_env.get_action_mask()
    assert mask[2].shape[1] == len(visible)


def test_attack_mask_does_not_leak_hidden_unit_count(fresh_env):
    """Attack mask column count must NOT equal total opponent unit count
    when some opponent units are hidden (i.e., outside the player's vision)."""
    player = fresh_env.game.players[fresh_env.game.player_go_id]
    opponent = fresh_env.game.players[(fresh_env.game.player_go_id + 1) % 2]

    # Shrink vision so no enemies are visible
    capital_id = fresh_env.game.game_board.capital_tile_ids[fresh_env.game.player_go_id]
    player.uncovered_tile_ids = {capital_id}

    mask = fresh_env.get_action_mask()
    visible_count = len(fresh_env._visible_enemy_units())
    total_opponent = len(opponent.units_under_control)

    # With hidden enemies: visible_count < total_opponent
    # and mask column count == visible_count (NOT total_opponent)
    assert mask[2].shape[1] == visible_count
    if visible_count < total_opponent:
        assert mask[2].shape[1] != total_opponent


def test_create_unit_mask_zero_when_city_full(fresh_env):
    """If a city is full, its row in the create-unit mask must be all zeros."""
    player = fresh_env.game.players[fresh_env.game.player_go_id]
    mask = fresh_env.get_action_mask()

    for city_idx, city in enumerate(player.cities_under_control):
        if city.current_n_units >= city.max_unit_cap:
            assert mask[3][city_idx].sum() == 0, (
                f"City {city_idx} is full but has nonzero create-unit mask."
            )


def test_mask_sums_after_end_turn(fresh_env):
    """After an EndTurn, mask is recomputed for the other player and must still be valid."""
    fresh_env.step([ActionTypes.EndTurn.value])
    mask = fresh_env.get_action_mask()

    assert mask[0][ActionTypes.EndTurn] == 1.0
    assert all(m.sum() >= 0 for m in mask)

    player = fresh_env.game.players[fresh_env.game.player_go_id]
    n_units = len(player.units_under_control)
    n_visible = len(fresh_env._visible_enemy_units())
    assert mask[1].shape[0] == n_units
    assert mask[2].shape == (n_units, n_visible)
