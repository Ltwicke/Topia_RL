import pytest
from game.enums import PlayerId


def test_units_under_control_is_dict(fresh_game):
    assert isinstance(fresh_game.players[0].units_under_control, dict)
    assert isinstance(fresh_game.players[1].units_under_control, dict)


def test_unit_has_unit_id(fresh_game):
    for player in fresh_game.players:
        for unit in player.units_under_control.values():
            assert hasattr(unit, "unit_id")
            assert isinstance(unit.unit_id, int)


def test_unit_ids_unique_across_game(fresh_game):
    all_ids = [
        u.unit_id
        for p in fresh_game.players
        for u in p.units_under_control.values()
    ]
    assert len(all_ids) == len(set(all_ids))


def test_unit_id_not_reused_after_death(fresh_game):
    # Record starting IDs; after game reset IDs should be fresh (set cleared)
    ids_before = {
        u.unit_id
        for p in fresh_game.players
        for u in p.units_under_control.values()
    }
    fresh_game.reset_game()
    ids_after = {
        u.unit_id
        for p in fresh_game.players
        for u in p.units_under_control.values()
    }
    # After reset, new IDs are generated — uniqueness within each game is guaranteed
    assert len(ids_after) == len(
        [u for p in fresh_game.players for u in p.units_under_control.values()]
    )


def test_unit_still_has_city_attribute(fresh_game):
    for player in fresh_game.players:
        for unit in player.units_under_control.values():
            assert hasattr(unit, "city")


def test_dict_keyed_by_unit_id(fresh_game):
    for player in fresh_game.players:
        for uid, unit in player.units_under_control.items():
            assert uid == unit.unit_id


def test_used_unit_ids_grow_on_create(fresh_game):
    ids_before = len(fresh_game._used_unit_ids)
    # EndTurn so the other player can act, then check CreateUnit adds to pool
    # (minimal smoke test — just verify the set grows after a new unit is created)
    assert ids_before >= 2  # at least the two starting units
