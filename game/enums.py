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


# ---------------------------------------------------------------------------
# Derived dimension constants
# ---------------------------------------------------------------------------
N_TILE_TYPES  = len(TileType)
N_PLAYERS     = len(PlayerId)
N_UNIT_TYPES  = len(UnitType)
N_UNIT_STATES = len(UnitState)
N_TILE_STATI  = len(TileStatus)
N_CITY_TYPES  = len(CityType) - 1   # excludes village from the city one-hot

# ---------------------------------------------------------------------------
# Feature vector boundary integers
# Layout: [tile_type | player_ctrl | city | unit_state | P0_type | P1_type]
# ---------------------------------------------------------------------------
_TILE_TYPE_START   = 0
_PLAYER_CTRL_START = _TILE_TYPE_START   + N_TILE_TYPES                    # 3
_CITY_START        = _PLAYER_CTRL_START + N_PLAYERS                       # 5
_UNIT_START        = _CITY_START        + 1 + N_CITY_TYPES * N_PLAYERS    # 10
_UNIT_TYPE_START   = _UNIT_START        + N_UNIT_STATES                   # 14
NODE_FEAT_DIM      = _UNIT_TYPE_START   + N_UNIT_TYPES * N_PLAYERS        # 18

# ---------------------------------------------------------------------------
# Named slices into any node feature vector
# ---------------------------------------------------------------------------
TILE_TYPE_SLICE   = slice(_TILE_TYPE_START,   _PLAYER_CTRL_START)   # [0:3]
PLAYER_CTRL_SLICE = slice(_PLAYER_CTRL_START, _CITY_START)          # [3:5]
CITY_SLICE        = slice(_CITY_START,        _UNIT_START)           # [5:10]
UNIT_STATE_SLICE  = slice(_UNIT_START,        _UNIT_TYPE_START)      # [10:14] — shared across players

# partial_graph slices (after P2 swap: own player's type block is always first)
OWN_TYPE_SLICE = slice(_UNIT_TYPE_START,                    _UNIT_TYPE_START + N_UNIT_TYPES)  # [14:16]
OPP_TYPE_SLICE = slice(_UNIT_TYPE_START + N_UNIT_TYPES,     NODE_FEAT_DIM)                    # [16:18]


def player_type_slice(player_idx: int) -> slice:
    """Unit-type slice for player_idx in the raw board_graph (absolute order, no P2 swap)."""
    s = _UNIT_TYPE_START + player_idx * N_UNIT_TYPES
    return slice(s, s + N_UNIT_TYPES)





