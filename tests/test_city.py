import dataclasses
import pytest
from game.enums import CityType, PlayerId
from game.components.city import City


def test_city_is_dataclass():
    assert dataclasses.is_dataclass(City)


def test_city_has_no_unit_field():
    c = City(tile_id=0, player_id=None)
    assert not hasattr(c, "unit")


def test_village_player_id_is_none():
    c = City(tile_id=0, player_id=None)
    assert c.player_id is None
    assert c.lvl == CityType.village


def test_village_max_unit_cap_is_zero():
    c = City(tile_id=0, player_id=None)
    assert c.max_unit_cap == 0


def test_city_max_unit_cap_is_three():
    c = City(tile_id=0, player_id=PlayerId.P1)
    assert c.max_unit_cap == 3


def test_upgrade_increases_max_unit_cap():
    c = City(tile_id=0, player_id=PlayerId.P1)
    c.upgrade()
    assert c.lvl == CityType.lvl2_city
    assert c.max_unit_cap == 6


def test_capture_changes_owner_and_sets_city_lvl():
    c = City(tile_id=0, player_id=None)
    c.capture(PlayerId.P1)
    assert c.player_id == PlayerId.P1
    assert c.lvl == CityType.city


def test_capture_clears_siege():
    c = City(tile_id=0, player_id=PlayerId.P1)
    c.under_siege = True
    c.capture(PlayerId.P2)
    assert not c.under_siege


def test_current_n_units_present(fresh_game):
    player = fresh_game.players[0]
    city = fresh_game.game_board.board[player.capital_id].city
    assert hasattr(city, "current_n_units")
    assert city.current_n_units >= 1


def test_tile_is_authoritative_for_unit_occupancy(fresh_game):
    player = fresh_game.players[0]
    tile = fresh_game.game_board.board[player.capital_id]
    assert tile.unit is not None
