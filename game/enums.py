from enum import IntEnum, Enum

class UnitType(IntEnum):
    Warrior = 0
    Rider = 1


class UnitState(IntEnum):
    """
    idle: the unit cannot do any action this turn anymore
    ready: the unit has not done any action this turn
    escaping: the unit cannot attack anymore, but can move
    can_hit: the unit can attack, but cannot move.
    """
    idle = 0
    ready = 1
    escaping = 2
    can_hit = 3
    #frozen = 3
    #poisoned = 4


class CityType(IntEnum):
    village = -1
    city = 0
    lvl2_city = 1

"""
class CityType(IntEnum):
    village = 0
    lvl1_city = 1
    lvl2_city_workshop = 2
    lvl2_city_explorer = 3
    lvl3_city_resources = 4
    lvl3_city_wall = 5
    lvl4_city_popgrwth = 6
    lvl4_city_bordergwth = 7
    lvl5_city_park = 8
    lvl5_city_su = 9
    lvl6_city_park = 10
    lvl6_city_su = 11
    lvl7_city_park = 12
    lvl7_city_su = 13
    lvl8plus_city = 14
"""

class TileType(IntEnum):
    field = 0
    water = 1
    deep_water = 2


class TileStatus(IntEnum):
    no_status = 0
    flooded = 1

class Actions(IntEnum):
    move_unit = 0
    train_unit = 1
    heal_unit = 2
    attack = 3

class BoardType(IntEnum):
    Dummy = 0
    Drylands = 1
    Lakes = 2

class PlayerId(IntEnum):
    P1 = 0
    P2 = 1


class Tribes(IntEnum):
    Omaji = 0
    Yaddak = 1
    Imperius = 2


class DefenseBonus(float, Enum):
    NoBonus = 1.
    Shield = float(3/2) # 1.5
    Wall = 4.

class ActionTypes(IntEnum):
    MoveUnit = 0
    Attack = 1
    CreateUnit = 2
    CaptureCity = 3
    
    EndTurn = 4
    





