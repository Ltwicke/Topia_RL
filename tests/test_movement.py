"""
Movement tests — verify the new three-phase movement algorithm:
  1. Destinations: no occupied / hidden / impassable tile appears as target
  2. Friendly transit: unit can reach tiles through friendly-occupied intermediate tiles
  3. Enemy blocking: enemy-occupied tiles are neither transit nor destination
  4. Zone of control: ZoC tiles are valid one-step destinations but block transit beyond them
  5. Road mechanic: Dijkstra respects edge weights; weight=0.5 edges extend range
"""
import numpy as np
import pytest

from game.enums import BoardType, Tribes, UnitType, UnitState, TileType
from game.components.units import Warrior
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
# General destination-validity properties
# ---------------------------------------------------------------------------

def test_move_mask_never_targets_occupied_tile(fresh_env):
    """Any tile set to 1 in the move mask must have no unit on it."""
    mask = fresh_env.get_action_mask()
    board = fresh_env.game.game_board.board
    for tile_id in set(np.where(mask[1] > 0)[1]):
        assert board[tile_id].unit is None, (
            f"Tile {tile_id} has a unit but appears as a move target"
        )


def test_move_targets_are_field_tiles(fresh_env):
    """Every tile in the move mask must be a walkable field tile."""
    mask = fresh_env.get_action_mask()
    board = fresh_env.game.game_board.board
    for tile_id in set(np.where(mask[1] > 0)[1]):
        assert board[tile_id].tile_type == TileType.field, (
            f"Tile {tile_id} (type {board[tile_id].tile_type}) is in move mask but not a field"
        )


def test_move_targets_are_in_uncovered_tiles(fresh_env):
    """Every tile in the move mask must be visible to the current player."""
    mask = fresh_env.get_action_mask()
    player = fresh_env.game.players[fresh_env.game.player_go_id]
    for tile_id in set(np.where(mask[1] > 0)[1]):
        assert tile_id in player.uncovered_tile_ids, (
            f"Tile {tile_id} is in move mask but not in uncovered tiles"
        )


# ---------------------------------------------------------------------------
# Friendly transit
# ---------------------------------------------------------------------------

def test_friendly_unit_in_path_does_not_block_transit(fresh_env):
    """A unit should reach tiles on the far side of a friendly-occupied tile."""
    game = fresh_env.game
    player = game.players[game.player_go_id]
    board = game.game_board.board
    G = game.game_board.movement_topology_graph

    uid = list(player.units_under_control.keys())[0]
    unit = player.units_under_control[uid]
    src_node = game.game_board.int_to_tup[unit.tile.id]

    # Find a chain: src → mid (adjacent field, empty) → far (adjacent to mid, empty, != src)
    mid_id = far_id = None
    for nbr in G.neighbors(src_node):
        nbr_id = game.game_board.tup_to_int[nbr]
        if board[nbr_id].tile_type != TileType.field or board[nbr_id].unit is not None:
            continue
        for nbr2 in G.neighbors(nbr):
            nbr2_id = game.game_board.tup_to_int[nbr2]
            if nbr2_id == unit.tile.id:
                continue
            if board[nbr2_id].tile_type == TileType.field and board[nbr2_id].unit is None:
                mid_id, far_id = nbr_id, nbr2_id
                break
        if mid_id is not None:
            break

    if mid_id is None:
        pytest.skip("No two-hop field chain found on this board")

    # Expose both tiles
    player.uncovered_tile_ids |= {mid_id, far_id}
    game.game_board.create_board_graph_from_board_state(game.all_tile_ids)
    player.construct_partial_graph_2players(game.game_board)

    far_node = game.game_board.int_to_tup[far_id]
    mid_node = game.game_board.int_to_tup[mid_id]

    # Verify far tile reachable without any blocking
    valid_empty = game.calc_movement_target_and_shortest_path(unit)
    if far_node not in valid_empty:
        pytest.skip("Far tile not reachable even without blocking unit (board geometry)")

    # Place a FRIENDLY unit at mid tile
    new_uid = game._new_unit_id()
    friendly = Warrior(player_id=unit.player_id, city=unit.city,
                       tile=board[mid_id], unit_id=new_uid)
    friendly.turn_state = UnitState.idle
    board[mid_id].unit = friendly
    player.units_under_control[new_uid] = friendly
    game.game_board.create_board_graph_from_board_state(game.all_tile_ids)
    player.construct_partial_graph_2players(game.game_board)

    valid_with_friendly = game.calc_movement_target_and_shortest_path(unit)

    assert far_node in valid_with_friendly, (
        "Far tile must be reachable through a friendly-occupied intermediate tile"
    )
    assert mid_node not in valid_with_friendly, (
        "Friendly-occupied tile must not appear as a move destination"
    )

    # Cleanup
    board[mid_id].unit = None
    del player.units_under_control[new_uid]
    game.game_board.create_board_graph_from_board_state(game.all_tile_ids)
    player.construct_partial_graph_2players(game.game_board)


# ---------------------------------------------------------------------------
# Enemy blocking
# ---------------------------------------------------------------------------

def test_enemy_tile_not_a_move_destination(fresh_env):
    """A tile occupied by a visible enemy must not appear in the move mask."""
    game = fresh_env.game
    player = game.players[game.player_go_id]
    opponent = game.players[(game.player_go_id + 1) % 2]

    if not opponent.units_under_control:
        pytest.skip("Opponent has no units")

    o_unit = next(iter(opponent.units_under_control.values()))
    o_tile_id = o_unit.tile.id

    # Make the enemy tile visible
    player.uncovered_tile_ids.add(o_tile_id)
    game.game_board.create_board_graph_from_board_state(game.all_tile_ids)
    player.construct_partial_graph_2players(game.game_board)

    mask = fresh_env.get_action_mask()

    for row in mask[1]:
        assert row[o_tile_id] == 0, (
            f"Enemy-occupied tile {o_tile_id} appears as a move target"
        )


# ---------------------------------------------------------------------------
# Zone of control
# ---------------------------------------------------------------------------

def test_zoc_tile_is_reachable_as_direct_destination(fresh_env):
    """A ZoC tile (adjacent to enemy) directly within 1 step of the unit is a valid destination."""
    game = fresh_env.game
    player = game.players[game.player_go_id]
    opponent = game.players[(game.player_go_id + 1) % 2]
    board = game.game_board.board
    G = game.game_board.movement_topology_graph

    uid = list(player.units_under_control.keys())[0]
    unit = player.units_under_control[uid]
    src_node = game.game_board.int_to_tup[unit.tile.id]

    if not opponent.units_under_control:
        pytest.skip("Opponent has no units")

    o_unit = next(iter(opponent.units_under_control.values()))

    # Find geometry: src → zoc_tile and enemy adjacent to zoc_tile
    # (zoc_tile is 1 step from src, enemy is adjacent to zoc_tile but != src)
    zoc_id = enemy_id = None
    for nbr in G.neighbors(src_node):
        nbr_id = game.game_board.tup_to_int[nbr]
        if board[nbr_id].tile_type != TileType.field or board[nbr_id].unit is not None:
            continue
        # Place enemy adjacent to this candidate zoc tile
        for nbr2 in G.neighbors(nbr):
            nbr2_id = game.game_board.tup_to_int[nbr2]
            if nbr2_id == unit.tile.id:
                continue
            if board[nbr2_id].tile_type == TileType.field and board[nbr2_id].unit is None:
                zoc_id, enemy_id = nbr_id, nbr2_id
                break
        if zoc_id is not None:
            break

    if zoc_id is None:
        pytest.skip("No suitable ZoC geometry found on this board")

    # Move enemy unit to enemy_id
    old_enemy_tile = o_unit.tile
    old_enemy_tile.unit = None
    board[enemy_id].unit = o_unit
    o_unit.tile = board[enemy_id]

    player.uncovered_tile_ids |= {zoc_id, enemy_id}
    game.game_board.create_board_graph_from_board_state(game.all_tile_ids)
    player.construct_partial_graph_2players(game.game_board)

    valid_paths = game.calc_movement_target_and_shortest_path(unit)
    zoc_node = game.game_board.int_to_tup[zoc_id]

    assert zoc_node in valid_paths, (
        "ZoC tile 1 step from the unit should be reachable as a stopping destination"
    )

    # Cleanup
    board[enemy_id].unit = None
    old_enemy_tile.unit = o_unit
    o_unit.tile = old_enemy_tile
    game.game_board.create_board_graph_from_board_state(game.all_tile_ids)
    player.construct_partial_graph_2players(game.game_board)


def test_zoc_blocks_transit_to_far_tile(fresh_env):
    """A tile reachable only through a ZoC tile (transit blocked) must not appear in results."""
    game = fresh_env.game
    player = game.players[game.player_go_id]
    opponent = game.players[(game.player_go_id + 1) % 2]
    board = game.game_board.board
    G = game.game_board.movement_topology_graph

    uid = list(player.units_under_control.keys())[0]
    unit = player.units_under_control[uid]
    src_node = game.game_board.int_to_tup[unit.tile.id]

    if not opponent.units_under_control or unit.mvpts < 2:
        pytest.skip("Need enemy and unit with mvpts >= 2")

    o_unit = next(iter(opponent.units_under_control.values()))

    # Find chain: src → zoc_tile → far_tile, enemy adjacent to zoc_tile
    # such that far_tile is ONLY reachable via zoc_tile
    zoc_id = far_id = enemy_id = None
    for nbr in G.neighbors(src_node):
        nbr_id = game.game_board.tup_to_int[nbr]
        if board[nbr_id].tile_type != TileType.field or board[nbr_id].unit is not None:
            continue
        for nbr2 in G.neighbors(nbr):
            nbr2_id = game.game_board.tup_to_int[nbr2]
            if nbr2_id == unit.tile.id:
                continue
            if board[nbr2_id].tile_type == TileType.field and board[nbr2_id].unit is None:
                # candidate: zoc=nbr_id, far=nbr2_id; place enemy adjacent to nbr_id
                for nbr3 in G.neighbors(nbr):
                    nbr3_id = game.game_board.tup_to_int[nbr3]
                    if nbr3_id in (unit.tile.id, nbr_id, nbr2_id):
                        continue
                    if board[nbr3_id].tile_type == TileType.field and board[nbr3_id].unit is None:
                        zoc_id, far_id, enemy_id = nbr_id, nbr2_id, nbr3_id
                        break
            if zoc_id is not None:
                break
        if zoc_id is not None:
            break

    if zoc_id is None:
        pytest.skip("No suitable ZoC-blocking geometry found on this board")

    # Move enemy to enemy_id (makes zoc_id a ZoC tile)
    old_enemy_tile = o_unit.tile
    old_enemy_tile.unit = None
    board[enemy_id].unit = o_unit
    o_unit.tile = board[enemy_id]

    player.uncovered_tile_ids |= {zoc_id, far_id, enemy_id}
    game.game_board.create_board_graph_from_board_state(game.all_tile_ids)
    player.construct_partial_graph_2players(game.game_board)

    valid_paths = game.calc_movement_target_and_shortest_path(unit)
    far_node = game.game_board.int_to_tup[far_id]

    # far_tile requires transit through zoc_tile → should not be reachable
    # (The ZoC tile itself can be a destination, but anything beyond it can't be reached via transit)
    # This only holds if far_tile has no other non-ZoC path from src
    far_neighbors = {game.game_board.tup_to_int[n]
                     for n in G.neighbors(game.game_board.int_to_tup[far_id])}
    # If far_tile is adjacent to src directly, this test can't isolate ZoC transit
    if unit.tile.id in far_neighbors:
        pytest.skip("Far tile is directly adjacent to src; ZoC isolation not possible")

    assert far_node not in valid_paths, (
        "Far tile reachable only through ZoC-blocked transit must not appear in results"
    )

    # Cleanup
    board[enemy_id].unit = None
    old_enemy_tile.unit = o_unit
    o_unit.tile = old_enemy_tile
    game.game_board.create_board_graph_from_board_state(game.all_tile_ids)
    player.construct_partial_graph_2players(game.game_board)


# ---------------------------------------------------------------------------
# Road mechanic
# ---------------------------------------------------------------------------

def test_road_edge_extends_movement_range(fresh_env):
    """Road edges (weight=0.5) allow movement where weight=1.5 edges would block it.

    Strategy: set all edges from the unit's tile to weight=1.5 (> 1 mvpt → no movement),
    then reset one specific edge to weight=0.5 (< 1 mvpt → reachable).
    This cleanly isolates the road mechanic without relying on board geometry.
    """
    import networkx as nx
    game = fresh_env.game
    player = game.players[game.player_go_id]
    board = game.game_board.board
    G = game.game_board.movement_topology_graph

    uid = list(player.units_under_control.keys())[0]
    unit = player.units_under_control[uid]
    assert unit.mvpts == 1, "Warrior must have mvpts=1 for this test"

    src_node = game.game_board.int_to_tup[unit.tile.id]

    # Find an adjacent uncovered field tile to use as the road target
    road_target_id = None
    for nbr in G.neighbors(src_node):
        nbr_id = game.game_board.tup_to_int[nbr]
        if (board[nbr_id].tile_type == TileType.field
                and board[nbr_id].unit is None
                and nbr_id in player.uncovered_tile_ids):
            road_target_id = nbr_id
            break

    if road_target_id is None:
        pytest.skip("No uncovered adjacent field tile available for road test")

    road_target_node = game.game_board.int_to_tup[road_target_id]

    # Set ALL edges from src to weight=1.5 → unit can't move anywhere (> 1 mvpt)
    original_weights = {}
    for nbr in list(G.neighbors(src_node)):
        original_weights[(src_node, nbr)] = G[src_node][nbr]['weight']
        G[src_node][nbr]['weight'] = 1.5
        G[nbr][src_node]['weight'] = 1.5
        original_weights[(nbr, src_node)] = original_weights[(src_node, nbr)]

    valid_blocked = game.calc_movement_target_and_shortest_path(unit)
    assert len(valid_blocked) == 0, (
        "Unit should not be able to move when all edges have weight=1.5 > mvpts=1"
    )

    # Set the road target edge to weight=0.5 → that tile becomes reachable (0.5 ≤ 1.0)
    G[src_node][road_target_node]['weight'] = 0.5
    G[road_target_node][src_node]['weight'] = 0.5

    valid_road = game.calc_movement_target_and_shortest_path(unit)
    assert road_target_node in valid_road, (
        "Road target must be reachable via weight=0.5 edge when mvpts=1"
    )

    # Restore all original weights
    for (u, v), w in original_weights.items():
        G[u][v]['weight'] = w
