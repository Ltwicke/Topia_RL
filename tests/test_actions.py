"""
TDD tests for HealUnit, UpgradeCity, Upgrade2Vet, kill tracking, and star costs.
Written before implementation — expected to fail until game.py / wrapper.py are updated.
"""
import pytest
from game.enums import ActionTypes, CityType, DefenseBonus, PlayerId, UnitState, UnitType
from game.components.city import _CITY_UPGRADE_COST, City


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _player_unit(game, player_id=0):
    """Return the first unit for the given player."""
    player = game.players[player_id]
    return next(iter(player.units_under_control.values()))


def _heal_action(game, player_id=0):
    player = game.players[player_id]
    uid = next(iter(player.units_under_control.keys()))
    return {"type": ActionTypes.HealUnit, "unit_id": uid}


def _upgrade_city_action(game, player_id=0, city_idx=0, choice=0):
    return {"type": ActionTypes.UpgradeCity, "city": city_idx, "choice": choice}


def _upgrade2vet_action(game, player_id=0):
    player = game.players[player_id]
    uid = next(iter(player.units_under_control.keys()))
    return {"type": ActionTypes.Upgrade2Vet, "unit_id": uid}


def _create_unit_action(game, player_id=0, city_idx=0, unit_type=UnitType.Warrior):
    return {"type": ActionTypes.CreateUnit, "city": city_idx, "unit_type": unit_type}


# ---------------------------------------------------------------------------
# HealUnit — game logic
# ---------------------------------------------------------------------------

def test_heal_inside_territory(fresh_game):
    g = fresh_game
    unit = _player_unit(g, 0)
    unit.current_hp = 4.0  # well below max (10.0), room for 4.0 heal without capping
    unit.turn_state = UnitState.ready
    tile = unit.tile
    tile.cntrl = g.players[0].player_id  # own territory
    g.apply_action(_heal_action(g))
    assert unit.current_hp == 8.0


def test_heal_outside_territory(fresh_game):
    g = fresh_game
    unit = _player_unit(g, 0)
    unit.current_hp = unit.hp - 5.0
    unit.turn_state = UnitState.ready
    unit.tile.cntrl = None  # neutral territory
    hp_before = unit.current_hp
    g.apply_action(_heal_action(g))
    assert unit.current_hp == hp_before + 2.0


def test_heal_capped_at_max_hp(fresh_game):
    g = fresh_game
    unit = _player_unit(g, 0)
    unit.current_hp = unit.hp - 1.0  # only 1 below max
    unit.turn_state = UnitState.ready
    unit.tile.cntrl = g.players[0].player_id
    g.apply_action(_heal_action(g))
    assert unit.current_hp == unit.hp


def test_heal_sets_idle(fresh_game):
    g = fresh_game
    unit = _player_unit(g, 0)
    unit.current_hp = unit.hp - 3.0
    unit.turn_state = UnitState.ready
    unit.tile.cntrl = g.players[0].player_id
    g.apply_action(_heal_action(g))
    assert unit.turn_state == UnitState.idle


# ---------------------------------------------------------------------------
# HealUnit — action mask
# ---------------------------------------------------------------------------

def test_heal_mask_ready_only(fresh_env):
    env = fresh_env
    unit = _player_unit(env.game, 0)
    unit.current_hp = unit.hp - 3.0

    unit.turn_state = UnitState.idle
    mask = env.get_action_mask()
    assert mask[0][ActionTypes.HealUnit] == 0.0

    unit.turn_state = UnitState.ready
    mask = env.get_action_mask()
    assert mask[0][ActionTypes.HealUnit] == 1.0


def test_heal_mask_excludes_full_hp(fresh_env):
    env = fresh_env
    unit = _player_unit(env.game, 0)
    unit.current_hp = unit.hp  # already full
    unit.turn_state = UnitState.ready
    mask = env.get_action_mask()
    assert mask[0][ActionTypes.HealUnit] == 0.0


# ---------------------------------------------------------------------------
# UpgradeCity — game logic
# ---------------------------------------------------------------------------

def test_upgrade_city_deducts_stars(fresh_game):
    g = fresh_game
    player = g.players[0]
    city = player.cities_under_control[0]
    player.stars = 50
    expected_cost = _CITY_UPGRADE_COST[CityType.lvl2_workshop]  # choice 0 from lvl1
    g.apply_action(_upgrade_city_action(g, choice=0))
    assert player.stars == 50 - expected_cost


def test_upgrade_city_changes_level(fresh_game):
    g = fresh_game
    player = g.players[0]
    player.stars = 50
    g.apply_action(_upgrade_city_action(g, choice=0))
    city = player.cities_under_control[0]
    assert city.lvl == CityType.lvl2_workshop


def test_upgrade_city_resources_refund(fresh_game):
    g = fresh_game
    player = g.players[0]
    player.stars = 50
    # First upgrade to lvl2 (choice 0 = workshop)
    g.apply_action(_upgrade_city_action(g, choice=0))
    stars_after_lvl2 = player.stars
    # Second upgrade: from lvl2_workshop, choice 0 = lvl3_resources (+5 stars refund)
    cost_to_lvl3 = _CITY_UPGRADE_COST[CityType.lvl3_resources]
    g.apply_action(_upgrade_city_action(g, choice=0))
    assert player.stars == stars_after_lvl2 - cost_to_lvl3 + 5


def test_upgrade_city_wall_grants_wall_bonus(fresh_game):
    g = fresh_game
    player = g.players[0]
    player.stars = 100
    # Upgrade to lvl2 first (choice 0 = workshop)
    g.apply_action(_upgrade_city_action(g, choice=0))
    # From lvl2_workshop, choice 1 = lvl3_wall
    g.apply_action(_upgrade_city_action(g, choice=1))
    city = player.cities_under_control[0]
    assert city.lvl == CityType.lvl3_wall
    # Unit on the city tile should now have Wall defense bonus
    city_tile = g.game_board.board[city.tile_id]
    if city_tile.unit is not None and city_tile.unit.player_id == player.player_id:
        assert city_tile.unit.def_bonus == DefenseBonus.Wall


def test_upgrade_city_popgrowth_sets_discount(fresh_game):
    g = fresh_game
    player = g.players[0]
    player.stars = 100
    g.apply_action(_upgrade_city_action(g, choice=0))  # lvl2_workshop
    g.apply_action(_upgrade_city_action(g, choice=0))  # lvl3_resources
    # lvl4 choice 0 = popgrowth
    g.apply_action(_upgrade_city_action(g, choice=0))
    city = player.cities_under_control[0]
    assert city.lvl == CityType.lvl4_popgrwth
    assert city.pending_discount == 6


def test_upgrade_city_discount_applied(fresh_game):
    g = fresh_game
    player = g.players[0]
    player.stars = 200
    g.apply_action(_upgrade_city_action(g, choice=0))  # lvl2_workshop
    g.apply_action(_upgrade_city_action(g, choice=0))  # lvl3_resources
    g.apply_action(_upgrade_city_action(g, choice=0))  # lvl4_popgrowth (discount=6)
    stars_before_lvl5 = player.stars
    expected_cost_lvl5 = _CITY_UPGRADE_COST[CityType.lvl5_su]  # choice 0
    discounted_cost = max(0, expected_cost_lvl5 - 6)
    g.apply_action(_upgrade_city_action(g, choice=0))
    assert player.stars == stars_before_lvl5 - discounted_cost


def test_upgrade_city_bordergrwth_controls_tiles(fresh_game):
    g = fresh_game
    player = g.players[0]
    player.stars = 100
    city = player.cities_under_control[0]
    g.apply_action(_upgrade_city_action(g, choice=0))  # lvl2_workshop
    g.apply_action(_upgrade_city_action(g, choice=0))  # lvl3_resources
    # lvl4 choice 1 = bordergrwth
    g.apply_action(_upgrade_city_action(g, choice=1))
    assert city.lvl == CityType.lvl4_bordergrwth
    # All tiles the city controls must now belong to this player
    for tid in city.controlled_tile_ids:
        assert g.game_board.board[tid].cntrl == player.player_id
    # No tile in range 2 should be unclaimed (either ours or opponent's pre-existing)
    for tid in g.tiles_in_range(city.tile_id, distance=2):
        assert g.game_board.board[tid].cntrl is not None


# ---------------------------------------------------------------------------
# UpgradeCity — action mask
# ---------------------------------------------------------------------------

def test_upgrade_city_mask_requires_stars(fresh_env):
    env = fresh_env
    player = env.game.players[0]
    player.stars = 0  # can't afford anything
    mask = env.get_action_mask()
    assert mask[0][ActionTypes.UpgradeCity] == 0.0
    assert mask[6].sum() == 0.0


def test_upgrade_city_mask_max_level(fresh_env):
    env = fresh_env
    player = env.game.players[0]
    player.stars = 999
    city = player.cities_under_control[0]
    # lvl8plus cities are always upgradeable (self-loop in upgrade tree)
    city.lvl = CityType.lvl8plus
    mask = env.get_action_mask()
    assert mask[6].sum() > 0.0
    assert mask[0][ActionTypes.UpgradeCity] == 1.0


def test_upgrade_city_mask_valid_when_affordable(fresh_env):
    env = fresh_env
    player = env.game.players[0]
    player.stars = 50
    mask = env.get_action_mask()
    assert mask[0][ActionTypes.UpgradeCity] == 1.0
    assert mask[6].sum() > 0.0


# ---------------------------------------------------------------------------
# Upgrade2Vet — game logic
# ---------------------------------------------------------------------------

def test_upgrade2vet_sets_is_vet(fresh_game):
    g = fresh_game
    unit = _player_unit(g, 0)
    unit.kills = 3
    g.apply_action(_upgrade2vet_action(g))
    assert unit.is_vet is True


def test_upgrade2vet_increases_hp(fresh_game):
    g = fresh_game
    unit = _player_unit(g, 0)
    unit.kills = 3
    hp_before = unit.hp
    g.apply_action(_upgrade2vet_action(g))
    assert unit.hp == hp_before + 5


def test_upgrade2vet_full_heal(fresh_game):
    g = fresh_game
    unit = _player_unit(g, 0)
    unit.kills = 3
    unit.current_hp = 3.0  # damaged
    g.apply_action(_upgrade2vet_action(g))
    assert unit.current_hp == unit.hp


def test_upgrade2vet_does_not_change_turn_state(fresh_game):
    g = fresh_game
    unit = _player_unit(g, 0)
    unit.kills = 3
    unit.turn_state = UnitState.ready
    g.apply_action(_upgrade2vet_action(g))
    assert unit.turn_state == UnitState.ready  # turn state unchanged by design


# ---------------------------------------------------------------------------
# Upgrade2Vet — action mask
# ---------------------------------------------------------------------------

def test_upgrade2vet_mask_requires_3_kills(fresh_env):
    env = fresh_env
    unit = _player_unit(env.game, 0)
    unit.kills = 2
    mask = env.get_action_mask()
    assert mask[0][ActionTypes.Upgrade2Vet] == 0.0

    unit.kills = 3
    mask = env.get_action_mask()
    assert mask[0][ActionTypes.Upgrade2Vet] == 1.0


def test_upgrade2vet_mask_excludes_already_vet(fresh_env):
    env = fresh_env
    unit = _player_unit(env.game, 0)
    unit.kills = 3
    unit.is_vet = True
    mask = env.get_action_mask()
    assert mask[0][ActionTypes.Upgrade2Vet] == 0.0


# ---------------------------------------------------------------------------
# Kill tracking & knight re-attack
# ---------------------------------------------------------------------------

def test_kill_increments_kills(fresh_game):
    """Unit.kills increments when it destroys an enemy unit."""
    g = fresh_game
    attacker = _player_unit(g, 0)
    defender = _player_unit(g, 1)

    # Place them adjacent
    attacker.tile.unit = None
    defender.tile.unit = None
    adj_tile = g.game_board.board[attacker.tile.id + 1]
    adj_tile.unit = defender
    defender.tile = adj_tile

    # Make attacker one-shot the defender
    attacker.atk_stat = 999
    attacker.turn_state = UnitState.ready
    g.apply_action({
        "type": ActionTypes.Attack,
        "unit_id": attacker.unit_id,
        "o_unit_id": defender.unit_id,
    })
    assert attacker.kills == 1


def _setup_knight(g, player_id=0):
    """Replace player's unit with a Knight at its current tile."""
    from game.components.units import Knight
    player = g.players[player_id]
    orig = next(iter(player.units_under_control.values()))
    knight = Knight(player_id=orig.player_id, city=orig.city,
                    tile=orig.tile, unit_id=orig.unit_id)
    player.units_under_control[knight.unit_id] = knight
    orig.tile.unit = knight
    return knight


def _place_unit_at(g, unit, tile_id):
    """Move a unit to tile_id, clearing its old tile reference."""
    unit.tile.unit = None
    new_tile = g.game_board.board[tile_id]
    new_tile.unit = unit
    unit.tile = new_tile


def test_knight_can_reattack_after_kill(fresh_game):
    """Knight stays in can_hit state after a kill when an enemy remains adjacent."""
    from game.components.units import Warrior
    g = fresh_game
    player = g.players[0]
    opponent = g.players[1]

    knight = _setup_knight(g)
    knight.atk_stat = 999
    knight.turn_state = UnitState.can_hit

    # Find two adjacent tiles using tiles_in_range (guaranteed neighbours)
    adj = [t for t in g.tiles_in_range(knight.tile.id, 1) if t != knight.tile.id]
    assert len(adj) >= 2, "Knight needs at least 2 adjacent tiles"
    first_tid, second_tid = adj[0], adj[1]

    # Place first defender at first_tid (the kill target)
    defender = _player_unit(g, 1)
    _place_unit_at(g, defender, first_tid)

    # Create a second enemy at second_tid so enemy_adjacent stays True after the kill
    uid2 = g._new_unit_id()
    second_enemy = Warrior(player_id=opponent.player_id,
                           city=opponent.cities_under_control[0],
                           tile=g.game_board.board[second_tid],
                           unit_id=uid2)
    g.game_board.board[second_tid].unit = second_enemy
    opponent.units_under_control[uid2] = second_enemy

    g.player_go_id = 0
    g.apply_action({
        "type": ActionTypes.Attack,
        "unit_id": knight.unit_id,
        "o_unit_id": defender.unit_id,
    })
    # Knight moved to first_tid; second_enemy at second_tid is adjacent → stays can_hit
    assert knight.turn_state == UnitState.can_hit


def test_knight_idle_after_no_kill(fresh_game):
    """Knight becomes idle after a non-lethal attack."""
    g = fresh_game
    player = g.players[0]

    knight = _setup_knight(g)
    knight.atk_stat = 0.01  # tiny — won't kill
    knight.turn_state = UnitState.can_hit

    adj = [t for t in g.tiles_in_range(knight.tile.id, 1) if t != knight.tile.id]
    defender = _player_unit(g, 1)
    _place_unit_at(g, defender, adj[0])
    defender.current_hp = defender.hp  # full HP — won't die

    g.player_go_id = 0
    g.apply_action({
        "type": ActionTypes.Attack,
        "unit_id": knight.unit_id,
        "o_unit_id": defender.unit_id,
    })
    assert knight.turn_state == UnitState.idle


# ---------------------------------------------------------------------------
# CreateUnit — star deduction
# ---------------------------------------------------------------------------

def test_create_unit_deducts_stars(fresh_game):
    g = fresh_game
    player = g.players[0]
    city = player.cities_under_control[0]

    # Ensure city tile is free
    city_tile = g.game_board.board[city.tile_id]
    city_tile.unit = None
    city.current_n_units = 0

    player.stars = 20
    g.apply_action(_create_unit_action(g, unit_type=UnitType.Warrior))
    assert player.stars == 20 - 2  # Warrior costs 2


def test_create_unit_mask_excludes_unaffordable(fresh_env):
    env = fresh_env
    player = env.game.players[0]
    player.stars = 2  # only Warriors (cost 2) are affordable

    mask = env.get_action_mask()
    # Warrior column (index 0) should be set for eligible cities
    warrior_col = int(UnitType.Warrior)
    knight_col = int(UnitType.Knight)
    city_count = len(player.cities_under_control)

    if city_count > 0:
        # At least one city should have Warrior affordable
        assert mask[3][:, warrior_col].max() == 1.0 or mask[3].sum() == 0
        # Knight (cost 13) should NOT be affordable
        assert mask[3][:, knight_col].sum() == 0.0
