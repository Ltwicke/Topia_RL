from enum import IntEnum, Enum

class UnitType(IntEnum):
    Warrior = 0
    Rider = 1
    Archer = 2
    Knight = 3
    Catapult = 4
    Giant = 5
    Sword = 6
    Defender = 7


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


""" OLD:
class CityType(IntEnum):
    village = -1
    city = 0
    lvl2_city = 1
"""


class CityType(IntEnum):
    village = 0
    lvl1 = 1
    lvl2_workshop = 2
    lvl2_explorer = 3
    lvl3_resources = 4
    lvl3_wall = 5
    lvl4_popgrwth = 6
    lvl4_bordergrwth = 7
    lvl5_su = 8
    lvl5_park = 9
    lvl6_su = 10
    lvl6_park = 11
    lvl7_su = 12
    lvl7_park = 13
    lvl8plus = 14


class TileType(IntEnum):
    field = 0
    water = 1
    deep_water = 2
    mountain = 3


class TileStatus(IntEnum):
    no_status = 0
    flooded = 1
    

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
    HealUnit = 4
    UpgradeCity = 5
    PlaceRoad = 6
    Upgrade2Vet = 7

    EndTurn = 8


# ---------------------------------------------------------------------------
# Derived dimension constants
# ---------------------------------------------------------------------------
N_TILE_TYPES  = len(TileType)
N_PLAYERS     = len(PlayerId)
N_UNIT_TYPES  = len(UnitType)
N_UNIT_STATES = len(UnitState)
N_TILE_STATI  = len(TileStatus)
N_CITY_TYPES  = len(CityType) -1  # excludes village from the city one-hot

# ---------------------------------------------------------------------------
# Feature vector boundary integers
# Layout: [tile_type | player_ctrl | city | unit_state | P0_type | P1_type]
# ---------------------------------------------------------------------------
_TILE_TYPE_START   = 0
_ROAD_START        = _TILE_TYPE_START   + N_TILE_TYPES                    # 4
_PLAYER_CTRL_START = _ROAD_START        + 1                               # 5
_CITY_START        = _PLAYER_CTRL_START + N_PLAYERS                       # 7
_UNIT_START        = _CITY_START        + 1 + N_CITY_TYPES * N_PLAYERS    # 36, the +1 is for the single village type, which is player-agnostic
_UNIT_TYPE_START   = _UNIT_START        + N_UNIT_STATES                   # 40
NODE_FEAT_DIM      = _UNIT_TYPE_START   + N_UNIT_TYPES * N_PLAYERS        # 54

# ---------------------------------------------------------------------------
# Named slices into any node feature vector
# ---------------------------------------------------------------------------
TILE_TYPE_SLICE   = slice(_TILE_TYPE_START,   _ROAD_START)           # [0:4]
ROAD_SLICE        = slice(_ROAD_START,         _PLAYER_CTRL_START)    # [4:5]
PLAYER_CTRL_SLICE = slice(_PLAYER_CTRL_START, _CITY_START)            # [5:7]
CITY_SLICE        = slice(_CITY_START,        _UNIT_START)             # [7:36]
UNIT_STATE_SLICE  = slice(_UNIT_START,        _UNIT_TYPE_START)        # [36:40] — shared across players

# partial_graph slices (after P2 swap: own player's type block is always first)
OWN_TYPE_SLICE = slice(_UNIT_TYPE_START,                    _UNIT_TYPE_START + N_UNIT_TYPES)  # [40:47]
OPP_TYPE_SLICE = slice(_UNIT_TYPE_START + N_UNIT_TYPES,     NODE_FEAT_DIM)                    # [47:54]


def player_type_slice(player_idx: int) -> slice:
    """Unit-type slice for player_idx in the raw board_graph (absolute order, no P2 swap)."""
    s = _UNIT_TYPE_START + player_idx * N_UNIT_TYPES
    return slice(s, s + N_UNIT_TYPES)


# ---------------------------------------------------------------------------
# Partial-graph swap descriptors (P2 perspective)
# Each (p0_slice, p1_slice) pair has equal length. P2 swap exchanges P0 ↔ P1 data.
# Add a row here whenever a new player-specific feature block is introduced.
# ---------------------------------------------------------------------------
PARTIAL_GRAPH_SWAPS: list[tuple[slice, slice]] = [
    # player_ctrl: 1 bit per player
    (slice(_PLAYER_CTRL_START,                   _PLAYER_CTRL_START + 1),
     slice(_PLAYER_CTRL_START + 1,               _CITY_START)),
    # city: N_CITY_TYPES dims per player (village flag at _CITY_START is player-agnostic)
    (slice(_CITY_START + 1,                      _CITY_START + 1 + N_CITY_TYPES),
     slice(_CITY_START + 1 + N_CITY_TYPES,       _UNIT_START)),
    # unit type: N_UNIT_TYPES dims per player
    (slice(_UNIT_TYPE_START,                     _UNIT_TYPE_START + N_UNIT_TYPES),
     slice(_UNIT_TYPE_START + N_UNIT_TYPES,      NODE_FEAT_DIM)),
]





