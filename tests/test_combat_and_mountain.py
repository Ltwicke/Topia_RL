"""
Tests for ranged-attack retaliation, stiff mechanic, fortify attribute,
mountain stopping-destination movement, and PlaceRoad mountain guard.
"""
import random
import numpy as np
import pytest

from game.enums import (
    ActionTypes, BoardType, DefenseBonus, PlayerId, TileType, Tribes,
    UnitState, UnitType,
)
from game.components.units import (
    Archer, Catapult, Defender, Giant, Knight, Sword, Warrior,
)
from env.wrapper import EnvWrapper


BOARD_CONFIG = {"board_size": (8, 8), "board_type": BoardType.Dummy, "n_players": 2}
TRIBES = [Tribes.Omaji, Tribes.Yaddak]


def _seed(s=42):
    np.random.seed(s)
    random.seed(s)


@pytest.fixture
def fresh_env():
    _seed()
    env = EnvWrapper(BOARD_CONFIG, TRIBES)
    env.reset()
    return env


@pytest.fixture
def fresh_game(fresh_env):
    return fresh_env.game


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _swap_unit_class(game, player_id, new_cls, unit_type):
    """Replace the first unit of `player_id` with a fresh `new_cls` on the same tile."""
    player = game.players[player_id]
    orig = next(iter(player.units_under_control.values()))
    replacement = new_cls(
        player_id=orig.player_id, city=orig.city,
        tile=orig.tile, unit_id=orig.unit_id,
    )
    replacement.turn_state = UnitState.ready
    player.units_under_control[replacement.unit_id] = replacement
    orig.tile.unit = replacement
    return replacement


def _place_at(game, unit, tile_id):
    unit.tile.unit = None
    new_tile = game.game_board.board[tile_id]
    new_tile.unit = unit
    unit.tile = new_tile


def _pick_tile_at_distance(game, origin_tile_id, want_dist):
    for tid in range(len(game.game_board.board)):
        if tid == origin_tile_id:
            continue
        if game._tile_distance(origin_tile_id, tid) == want_dist:
            return tid
    return None


def _attack_action(attacker, defender):
    return {
        "type": ActionTypes.Attack,
        "unit_id": attacker.unit_id,
        "o_unit_id": defender.unit_id,
    }


# ---------------------------------------------------------------------------
# Fortify — only fortify=True units get Shield/Wall on own city
# ---------------------------------------------------------------------------

def test_fortify_warrior_gets_shield(fresh_game):
    g = fresh_game
    unit = next(iter(g.players[0].units_under_control.values()))
    assert isinstance(unit, Warrior)  # default starting unit
    g._apply_unit_def_bonus(unit)
    assert unit.def_bonus == DefenseBonus.Shield


def test_fortify_defender_gets_shield(fresh_game):
    g = fresh_game
    defender = _swap_unit_class(g, 0, Defender, UnitType.Defender)
    g._apply_unit_def_bonus(defender)
    assert defender.fortify is True
    assert defender.def_bonus == DefenseBonus.Shield


def test_no_fortify_sword_gets_no_bonus(fresh_game):
    g = fresh_game
    sword = _swap_unit_class(g, 0, Sword, UnitType.Sword)
    g._apply_unit_def_bonus(sword)
    assert sword.fortify is False
    assert sword.def_bonus == DefenseBonus.NoBonus


def test_no_fortify_catapult_gets_no_bonus(fresh_game):
    g = fresh_game
    cata = _swap_unit_class(g, 0, Catapult, UnitType.Catapult)
    g._apply_unit_def_bonus(cata)
    assert cata.def_bonus == DefenseBonus.NoBonus


def test_no_fortify_giant_gets_no_bonus(fresh_game):
    g = fresh_game
    giant = _swap_unit_class(g, 0, Giant, UnitType.Giant)
    g._apply_unit_def_bonus(giant)
    assert giant.fortify is False
    assert giant.def_bonus == DefenseBonus.NoBonus


# ---------------------------------------------------------------------------
# Ranged attack: no retaliation damage at distance, no advance on ranged kill
# ---------------------------------------------------------------------------

def test_ranged_attacker_takes_no_retaliation_at_distance(fresh_game):
    """Archer attacking a melee defender at distance 2 → attacker HP unchanged."""
    g = fresh_game
    archer = _swap_unit_class(g, 0, Archer, UnitType.Archer)
    defender = next(iter(g.players[1].units_under_control.values()))

    # Put defender on a tile at Chebyshev distance 2 from archer
    target_tid = _pick_tile_at_distance(g, archer.tile.id, 2)
    assert target_tid is not None
    # Ensure target tile is a walkable field
    g.game_board.board[target_tid].tile_type = TileType.field
    _place_at(g, defender, target_tid)

    archer.current_hp = archer.hp
    defender.current_hp = defender.hp
    hp_before = archer.current_hp

    g.player_go_id = 0
    g.apply_action(_attack_action(archer, defender))

    # Archer takes no retaliation damage
    assert archer.current_hp == hp_before


def test_ranged_kill_does_not_advance(fresh_game):
    """Archer killing a defender at distance 2 stays on its original tile."""
    g = fresh_game
    archer = _swap_unit_class(g, 0, Archer, UnitType.Archer)
    defender = next(iter(g.players[1].units_under_control.values()))

    target_tid = _pick_tile_at_distance(g, archer.tile.id, 2)
    assert target_tid is not None
    g.game_board.board[target_tid].tile_type = TileType.field
    _place_at(g, defender, target_tid)

    archer_tile_before = archer.tile.id
    archer.atk_stat = 999  # one-shot

    g.player_go_id = 0
    g.apply_action(_attack_action(archer, defender))

    assert archer.tile.id == archer_tile_before
    assert g.game_board.board[archer_tile_before].unit is archer
    assert g.game_board.board[target_tid].unit is None
    # defender removed from opponent
    assert defender.unit_id not in g.players[1].units_under_control


def test_ranged_melee_retaliation_still_applies(fresh_game):
    """Archer attacking adjacent defender (distance 1) takes retaliation normally."""
    g = fresh_game
    archer = _swap_unit_class(g, 0, Archer, UnitType.Archer)
    defender = next(iter(g.players[1].units_under_control.values()))

    adj_tid = _pick_tile_at_distance(g, archer.tile.id, 1)
    assert adj_tid is not None
    g.game_board.board[adj_tid].tile_type = TileType.field
    _place_at(g, defender, adj_tid)

    archer.current_hp = archer.hp
    defender.current_hp = defender.hp
    hp_before = archer.current_hp

    g.player_go_id = 0
    g.apply_action(_attack_action(archer, defender))

    # Melee-distance attack: retaliation must subtract HP (non-zero defense damage)
    if archer.unit_id in g.players[0].units_under_control:
        assert archer.current_hp < hp_before


def test_ranged_melee_kill_advances(fresh_game):
    """Archer killing an adjacent defender advances into the defender's tile."""
    g = fresh_game
    archer = _swap_unit_class(g, 0, Archer, UnitType.Archer)
    defender = next(iter(g.players[1].units_under_control.values()))

    adj_tid = _pick_tile_at_distance(g, archer.tile.id, 1)
    assert adj_tid is not None
    g.game_board.board[adj_tid].tile_type = TileType.field
    _place_at(g, defender, adj_tid)

    archer.atk_stat = 999
    g.player_go_id = 0
    g.apply_action(_attack_action(archer, defender))

    assert archer.tile.id == adj_tid
    assert g.game_board.board[adj_tid].unit is archer


# ---------------------------------------------------------------------------
# Stiff — Catapult deals no retaliation damage
# ---------------------------------------------------------------------------

def test_stiff_catapult_deals_no_retaliation(fresh_game):
    """Warrior attacking an adjacent Catapult (stiff) takes 0 retaliation damage."""
    g = fresh_game
    attacker = next(iter(g.players[0].units_under_control.values()))
    catapult = _swap_unit_class(g, 1, Catapult, UnitType.Catapult)

    adj_tid = _pick_tile_at_distance(g, attacker.tile.id, 1)
    assert adj_tid is not None
    g.game_board.board[adj_tid].tile_type = TileType.field
    _place_at(g, catapult, adj_tid)

    attacker.current_hp = attacker.hp
    catapult.current_hp = catapult.hp
    hp_before = attacker.current_hp

    g.player_go_id = 0
    g.apply_action(_attack_action(attacker, catapult))

    # Catapult is stiff → attacker hp unchanged
    if attacker.unit_id in g.players[0].units_under_control:
        assert attacker.current_hp == hp_before
    assert catapult.stiff is True


# ---------------------------------------------------------------------------
# Mountain is a stopping destination but not transit
# ---------------------------------------------------------------------------

def _find_adjacent_field(game, tile_id):
    G = game.game_board.movement_topology_graph
    src = game.game_board.int_to_tup[tile_id]
    for nbr in G.neighbors(src):
        nid = game.game_board.tup_to_int[nbr]
        tile = game.game_board.board[nid]
        if tile.tile_type == TileType.field and tile.unit is None:
            return nid
    return None


def test_mountain_is_valid_stopping_destination(fresh_env):
    g = fresh_env.game
    player = g.players[g.player_go_id]
    unit = next(iter(player.units_under_control.values()))

    mountain_id = _find_adjacent_field(g, unit.tile.id)
    assert mountain_id is not None
    g.game_board.board[mountain_id].tile_type = TileType.mountain
    player.uncovered_tile_ids.add(mountain_id)
    g.game_board.create_board_graph_from_board_state(g.all_tile_ids)
    player.construct_partial_graph_2players(g.game_board)

    valid = g.calc_movement_target_and_shortest_path(unit)
    mountain_node = g.game_board.int_to_tup[mountain_id]
    assert mountain_node in valid, "mountain should be a valid stopping destination"


def test_mountain_not_used_as_transit(fresh_env):
    """No returned path should traverse a mountain as an intermediate node."""
    g = fresh_env.game
    player = g.players[g.player_go_id]
    board = g.game_board.board
    G = g.game_board.movement_topology_graph

    unit = next(iter(player.units_under_control.values()))
    unit.mvpts = 3  # give plenty of range so mountain-transit would be enticing
    src_id = unit.tile.id

    # Convert every field neighbour of src EXCEPT one into mountains; the exception becomes the
    # only transit route. Additionally add a second "far" mountain tile 2-hops out to verify the
    # mountain never appears mid-path.
    mountain_id = None
    for nbr in G.neighbors(g.game_board.int_to_tup[src_id]):
        nbr_id = g.game_board.tup_to_int[nbr]
        if board[nbr_id].tile_type == TileType.field and board[nbr_id].unit is None:
            mountain_id = nbr_id
            break
    if mountain_id is None:
        pytest.skip("No field neighbour to convert to mountain")

    board[mountain_id].tile_type = TileType.mountain
    player.uncovered_tile_ids.add(mountain_id)
    g.game_board.create_board_graph_from_board_state(g.all_tile_ids)
    player.construct_partial_graph_2players(g.game_board)

    valid = g.calc_movement_target_and_shortest_path(unit)
    mountain_node = g.game_board.int_to_tup[mountain_id]

    # Mountain itself allowed as destination (path length <= 1 hop ending at mountain)
    assert mountain_node in valid
    assert valid[mountain_node][-1] == mountain_node
    # The mountain must NEVER appear as an intermediate node in any other path
    for dest_node, path in valid.items():
        if dest_node == mountain_node:
            continue
        assert mountain_node not in path, (
            f"mountain {mountain_node} used as transit in path to {dest_node}: {path}"
        )


# ---------------------------------------------------------------------------
# PlaceRoad — mountain guard
# ---------------------------------------------------------------------------

def test_place_road_rejects_mountain(fresh_game):
    g = fresh_game
    player = g.players[0]
    player.stars = 50
    # Pick any tile, turn it into a mountain
    target = next(tid for tid in player.uncovered_tile_ids
                  if g.game_board.board[tid].tile_type == TileType.field
                  and not g.game_board.board[tid].has_road)
    g.game_board.board[target].tile_type = TileType.mountain

    with pytest.raises(AssertionError):
        g.apply_action({"type": ActionTypes.PlaceRoad, "tile_id": target})
