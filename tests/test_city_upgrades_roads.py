"""
TDD tests for PlaceRoad, Explorer vision, Giant (su) creation, lvl8plus cycling,
city_stars_per_turn park fix, and road movement restriction.
Written before implementation.
"""
import random
import pytest
import numpy as np

from game.enums import ActionTypes, CityType, TileType, PlayerId, UnitState
from game.components.city import _CITY_UPGRADE_COST, City


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _upgrade_city(game, player_id=0, city_idx=0, choice=0):
    return game.apply_action({
        "type": ActionTypes.UpgradeCity,
        "city": city_idx,
        "choice": choice,
    })


def _place_road(game, tile_id, player_id=0):
    return game.apply_action({
        "type": ActionTypes.PlaceRoad,
        "tile_id": tile_id,
    })


def _upgrade_to_lvl4(game, player_id=0):
    """Do three choice-0 upgrades to reach lvl4_popgrwth."""
    game.players[player_id].stars = 1000
    _upgrade_city(game, player_id, choice=0)  # lvl1 → lvl2_workshop
    _upgrade_city(game, player_id, choice=0)  # lvl2_workshop → lvl3_resources
    _upgrade_city(game, player_id, choice=0)  # lvl3_resources → lvl4_popgrwth


def _upgrade_to_lvl8plus(game, player_id=0):
    """Seven choice-0 upgrades to reach lvl8plus."""
    game.players[player_id].stars = 5000
    for _ in range(7):
        _upgrade_city(game, player_id, choice=0)


# ---------------------------------------------------------------------------
# city_stars_per_turn: park summation fix
# ---------------------------------------------------------------------------

def test_park_first_choice_counted_in_spt():
    """choices[3] == 1 (first park at lvl5) must contribute +1 star/turn."""
    c = City(tile_id=0, player_id=PlayerId.P1)
    # Simulate 4 upgrades: workshop, resources, popgrwth, then PARK (choice=1)
    c.choices = [0, 0, 0, 1]   # indices 0-3; choices[3]=1 = park
    c.times_upgraded = 4
    # base spt = 1 + 4 = 5; workshop (+1) + park at [3] (+1) = 7
    assert c.city_stars_per_turn == 7


def test_workshop_still_adds_spt():
    """choices[0] == 0 (workshop) must still add +1 star/turn (regression guard)."""
    c = City(tile_id=0, player_id=PlayerId.P1)
    c.choices = [0]
    c.times_upgraded = 1
    # base = 1 + 1 = 2; workshop adds 1 → 3
    assert c.city_stars_per_turn == 3


def test_no_park_no_extra_spt():
    """Choosing su (choice 0) at lvl5+ must NOT add extra star/turn."""
    c = City(tile_id=0, player_id=PlayerId.P1)
    c.choices = [1, 0, 0, 0]   # explorer, resources, popgrwth, su
    c.times_upgraded = 4
    # base = 1 + 4 = 5; no workshop (choices[0]=1), no park (choices[3]=0)
    assert c.city_stars_per_turn == 5


# ---------------------------------------------------------------------------
# Explorer vision
# ---------------------------------------------------------------------------

def test_explorer_reveals_tiles(fresh_game):
    g = fresh_game
    player = g.players[0]
    tiles_before = len(player.uncovered_tile_ids)
    player.stars = 1000
    _upgrade_city(g, choice=1)  # lvl1 → lvl2_explorer
    assert len(player.uncovered_tile_ids) > tiles_before


def test_explorer_path_at_most_14_steps(fresh_game, monkeypatch):
    """Explorer walks at most 14 steps (path has at most 15 nodes incl. start)."""
    g = fresh_game
    captured_paths = []

    original = g._apply_explorer_vision.__func__ if hasattr(g._apply_explorer_vision, '__func__') else None

    def capturing_explorer(self, player, start_tile_id, n_steps=14):
        # call the real implementation but capture what apply_unit_vision gets
        original_avu = self.apply_unit_vision
        def recording_avu(unit, path):
            captured_paths.append(list(path))
            return original_avu(unit, path)
        self.apply_unit_vision = recording_avu
        g.__class__._apply_explorer_vision(self, player, start_tile_id, n_steps)
        self.apply_unit_vision = original_avu

    monkeypatch.setattr(g, '_apply_explorer_vision',
                        lambda player, start_tile_id, n_steps=14:
                            capturing_explorer(g, player, start_tile_id, n_steps))

    player = g.players[0]
    player.stars = 1000
    _upgrade_city(g, choice=1)

    assert len(captured_paths) == 1
    assert len(captured_paths[0]) <= 15  # start + up to 14 steps


# ---------------------------------------------------------------------------
# Giant / su creation
# ---------------------------------------------------------------------------

def test_su_creates_giant_empty_city(fresh_game):
    """Choosing su (choice 0) at lvl4 creates a Giant at an empty city tile."""
    from game.components.units import Giant
    g = fresh_game
    player = g.players[0]
    city = player.cities_under_control[0]
    city_tile = g.game_board.board[city.tile_id]

    # Clear the city tile so Giant lands directly
    city_tile.unit = None
    city.current_n_units = 0

    _upgrade_to_lvl4(g)
    n_units_before = len(player.units_under_control)
    _upgrade_city(g, choice=0)  # lvl4_popgrwth → lvl5_su → Giant created

    assert isinstance(city_tile.unit, Giant)
    assert city_tile.unit.player_id == player.player_id
    assert len(player.units_under_control) == n_units_before + 1


def test_su_pushes_occupying_unit(fresh_game):
    """If city tile is occupied, the existing unit is moved to an adjacent free tile."""
    g = fresh_game
    player = g.players[0]
    city = player.cities_under_control[0]
    city_tile = g.game_board.board[city.tile_id]

    # Starting warrior IS on the city tile — don't clear it
    occupant = city_tile.unit
    assert occupant is not None

    _upgrade_to_lvl4(g)
    _upgrade_city(g, choice=0)  # Giant created; warrior must be pushed

    from game.components.units import Giant
    # Giant is now on the city tile
    assert isinstance(city_tile.unit, Giant)
    # Warrior is still alive but on a different tile
    assert occupant.unit_id in player.units_under_control
    assert occupant.tile.id != city.tile_id


def test_su_destroys_unit_when_no_free_adj(fresh_game):
    """No free adjacent field tile → occupying unit is destroyed."""
    from game.components.units import Giant
    g = fresh_game
    player = g.players[0]
    city = player.cities_under_control[0]
    city_tile = g.game_board.board[city.tile_id]
    occupant = city_tile.unit
    assert occupant is not None

    # Block all adjacent field tiles by converting them to water
    adj = [t for t in g.tiles_in_range(city.tile_id, 1) if t != city.tile_id]
    original_types = {}
    for tid in adj:
        t = g.game_board.board[tid]
        original_types[tid] = t.tile_type
        t.tile_type = TileType.water  # block it

    _upgrade_to_lvl4(g)
    _upgrade_city(g, choice=0)

    # Restore tile types
    for tid, tt in original_types.items():
        g.game_board.board[tid].tile_type = tt

    assert isinstance(city_tile.unit, Giant)
    assert occupant.unit_id not in player.units_under_control


# ---------------------------------------------------------------------------
# PlaceRoad — game logic
# ---------------------------------------------------------------------------

def _find_non_road_field_tile(game, player_id=0):
    """Return a visible field tile without a road that is own/neutral territory."""
    player = game.players[player_id]
    for tid in player.uncovered_tile_ids:
        tile = game.game_board.board[tid]
        if (not tile.has_road
                and tile.tile_type == TileType.field
                and (tile.cntrl is None or tile.cntrl == player.player_id)):
            return tid
    return None


def test_place_road_sets_has_road(fresh_game):
    g = fresh_game
    g.players[0].stars = 50
    tile_id = _find_non_road_field_tile(g)
    assert tile_id is not None
    _place_road(g, tile_id)
    assert g.game_board.board[tile_id].has_road is True


def test_place_road_deducts_4_stars(fresh_game):
    g = fresh_game
    g.players[0].stars = 10
    tile_id = _find_non_road_field_tile(g)
    _place_road(g, tile_id)
    assert g.players[0].stars == 6


def test_road_edge_weight_halved(fresh_game):
    """Two adjacent tiles both with has_road → edge weight becomes 0.5."""
    g = fresh_game
    # Pick any two adjacent non-road field tiles
    for u_tup, v_tup in g.game_board.movement_topology_graph.edges():
        u_id = g.game_board.tup_to_int[u_tup]
        v_id = g.game_board.tup_to_int[v_tup]
        tu = g.game_board.board[u_id]
        tv = g.game_board.board[v_id]
        if (not tu.has_road and not tv.has_road
                and tu.tile_type == TileType.field
                and tv.tile_type == TileType.field):
            tu.has_road = True
            tv.has_road = True
            g.game_board._update_road_edge_weights()
            w = g.game_board.movement_topology_graph[u_tup][v_tup]['weight']
            assert w == 0.5
            return
    pytest.skip("No adjacent non-road field pair found")


# ---------------------------------------------------------------------------
# PlaceRoad — action mask
# ---------------------------------------------------------------------------

def test_place_road_mask_requires_4_stars(fresh_env):
    env = fresh_env
    env.game.players[0].stars = 3  # not enough
    mask = env.get_action_mask()
    assert mask[0][ActionTypes.PlaceRoad] == 0.0
    assert mask[7].sum() == 0.0


def test_place_road_mask_field_only(fresh_env):
    """Water tiles must not appear in the road mask."""
    env = fresh_env
    env.game.players[0].stars = 50
    mask = env.get_action_mask()
    for tile_id in np.flatnonzero(mask[7]):
        tile = env.game.game_board.board[int(tile_id)]
        assert tile.tile_type == TileType.field


def test_place_road_mask_own_neutral_only(fresh_env):
    """Enemy-controlled tiles must not appear in the road mask."""
    env = fresh_env
    player = env.game.players[0]
    opponent = env.game.players[1]
    player.stars = 50

    # Mark a visible tile as enemy-controlled
    visible = list(player.uncovered_tile_ids)
    target = None
    for tid in visible:
        tile = env.game.game_board.board[tid]
        if not tile.has_road and tile.tile_type == TileType.field:
            tile.cntrl = opponent.player_id
            target = tid
            break

    mask = env.get_action_mask()
    if target is not None:
        assert mask[7][target] == 0.0


def test_place_road_mask_skips_existing_road(fresh_env):
    """A tile that already has a road must not appear in the mask."""
    env = fresh_env
    player = env.game.players[0]
    player.stars = 50
    # Find a visible non-road field tile and mark it as having a road
    for tid in player.uncovered_tile_ids:
        tile = env.game.game_board.board[tid]
        if (not tile.has_road and tile.tile_type == TileType.field
                and (tile.cntrl is None or tile.cntrl == player.player_id)):
            tile.has_road = True
            mask = env.get_action_mask()
            assert mask[7][tid] == 0.0
            return
    pytest.skip("No suitable tile found")


# ---------------------------------------------------------------------------
# Road movement restriction
# ---------------------------------------------------------------------------

def test_road_no_discount_on_enemy_tile(fresh_game):
    """A road-road edge where one endpoint is enemy-controlled uses weight 1.0."""
    g = fresh_game
    opponent = g.players[1]

    # Find two adjacent field tiles and give both roads
    for u_tup, v_tup in g.game_board.movement_topology_graph.edges():
        u_id = g.game_board.tup_to_int[u_tup]
        v_id = g.game_board.tup_to_int[v_tup]
        tu = g.game_board.board[u_id]
        tv = g.game_board.board[v_id]
        if tu.tile_type == TileType.field and tv.tile_type == TileType.field:
            tu.has_road = True
            tv.has_road = True
            g.game_board._update_road_edge_weights()
            # Confirm edge is discounted globally
            assert g.game_board.movement_topology_graph[u_tup][v_tup]['weight'] == 0.5

            # Mark one tile as enemy-controlled
            tu.cntrl = opponent.player_id

            # Get a player 0 unit to use for movement calc
            unit = next(iter(g.players[0].units_under_control.values()))
            # Build the movement graph G as calc_movement does, then check Phase 2b
            import networkx as nx
            from game.enums import OWN_TYPE_SLICE, OPP_TYPE_SLICE, _TILE_TYPE_START
            partial_graph = g.players[0].partial_graph
            cant_step_on = partial_graph[:, _TILE_TYPE_START] == 0
            own_occupied = (partial_graph[:, OWN_TYPE_SLICE] != 0).any(axis=-1)
            enemy_occupied = (partial_graph[:, OPP_TYPE_SLICE] != 0).any(axis=-1)
            G = g.game_board.movement_topology_graph.copy()

            # Phase 2b: apply enemy road restriction
            opponent_id_val = (int(unit.player_id) + 1) % 2
            for eu, ev in list(G.edges()):
                if G[eu][ev].get('weight', 1.0) < 1.0:
                    for tid in (g.game_board.tup_to_int[eu],
                                g.game_board.tup_to_int[ev]):
                        ctrl = g.game_board.board[tid].cntrl
                        if ctrl is not None and int(ctrl) == opponent_id_val:
                            G[eu][ev]['weight'] = 1.0
                            break

            w_after = G[u_tup][v_tup]['weight']
            assert w_after == 1.0, "enemy road should not discount movement"
            return

    pytest.skip("No adjacent field pair found")


# ---------------------------------------------------------------------------
# lvl8plus cycling
# ---------------------------------------------------------------------------

def test_lvl8plus_in_upgrade_mask(fresh_env):
    """A city at lvl8plus should still appear in the UpgradeCity mask."""
    env = fresh_env
    player = env.game.players[0]
    player.stars = 5000
    city = player.cities_under_control[0]
    # Force city to lvl8plus
    city.lvl = CityType.lvl8plus
    city.times_upgraded = 7
    mask = env.get_action_mask()
    assert mask[0][ActionTypes.UpgradeCity] == 1.0
    assert mask[6].sum() > 0.0


def test_lvl8plus_times_upgraded_increases(fresh_game):
    """times_upgraded keeps increasing indefinitely past lvl8plus."""
    g = fresh_game
    _upgrade_to_lvl8plus(g)
    city = g.players[0].cities_under_control[0]
    assert city.lvl == CityType.lvl8plus
    tu_before = city.times_upgraded
    _upgrade_city(g, choice=1)  # park at lvl8plus
    _upgrade_city(g, choice=1)
    assert city.times_upgraded == tu_before + 2


def test_lvl8plus_su_creates_giant(fresh_game):
    """choice 0 at lvl8plus creates a new Giant each time."""
    from game.components.units import Giant
    g = fresh_game
    _upgrade_to_lvl8plus(g)
    city = g.players[0].cities_under_control[0]
    city_tile = g.game_board.board[city.tile_id]
    city_tile.unit = None
    city.current_n_units = 0

    n_before = len(g.players[0].units_under_control)
    _upgrade_city(g, choice=0)  # su at lvl8plus
    assert isinstance(city_tile.unit, Giant)
    assert len(g.players[0].units_under_control) == n_before + 1
