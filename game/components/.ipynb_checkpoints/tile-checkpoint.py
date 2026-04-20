import numpy as np
from game.enums import (
    TileType, TileStatus, PlayerId, UnitType, CityType, UnitState,
    N_UNIT_TYPES, N_PLAYERS, N_UNIT_STATES, N_TILE_TYPES, N_TILE_STATI, N_CITY_TYPES
)
from game.components.city import City
from game.components.units import Warrior

def one_hot_field_type(member: TileType):
    """member is an integer. Move to components file"""
    return np.eye(N_TILE_TYPES)[member] 


def one_hot_tile_status(member: TileStatus):
    return np.eye(N_TILE_STATI)[member]


def player_controls_tile(member: PlayerId):
    if member == None:
        return np.zeros(N_PLAYERS)
    else:
        return np.eye(N_PLAYERS)[member]


def city_featurizer(city: City):
    """
    Here are some examples: See enums for dimensions
    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 
    0 1 0 1 0 1 1 0 1 0 0 0 0 0 0      = city lvl5 with explorer, resources,  bordergrwth and superunit
    0 1 1 0 0 1 0 1 0 1 1 0 1 0 1      = city lvl8plus with 1 park and 2 superunits chosen (keep this for global board, conceal for partial graph)
    because village only exists once, it must not be multiplied by N_PLAYERS... thats where the +1 comes from
    """

    if city == None: # there is just no city on this tile
        return np.zeros(N_CITY_TYPES * N_PLAYERS + 1)

    if city.player_id == None: # Village case
        village_vec = np.zeros(N_CITY_TYPES * N_PLAYERS + 1)
        village_vec[0] = 1
        return village_vec


    else: # City case
        city_vecs = [np.zeros(N_CITY_TYPES) for n in range(N_PLAYERS)] # P1 P2 P3 P4 ...
        city_vecs[city.player_id][0] = 1 # because its at least a city lvl1, DONT FORGET WE PAD LATER

        ##### choice encoding block
        choices = city.choices[:6] # no more choices required, because we cut at lvl8

        for start_index, choice in zip(range(1, len(choices)*2, 2), choices): # 1 3 5 7 9 11
            city_vecs[city.player_id][start_index:start_index+2] = np.eye(2)[choice] # assigns [1 0] for choice 0 and [0 1] for choice 1

        if city.lvl == CityType.lvl8plus:
            city_vecs[city.player_id][-1] = 1 
        
        return np.pad(np.hstack(city_vecs), pad_width=(1,0), mode="constant", constant_values=0)

    return "should not get here"


def unit_featurizer(unit):
    """
    New compact layout: [unit_state (4) | P0_type (2) | P1_type (2)] = 8 dims.
    Unit state stores HP fraction in the active state slot; type is one-hot per player.
    Since at most one unit can occupy a tile, state is player-agnostic.
    """
    if unit is None:
        return np.zeros(N_UNIT_STATES + N_UNIT_TYPES * N_PLAYERS)

    state_vec = np.zeros(N_UNIT_STATES)
    state_vec[int(unit.turn_state)] = float(unit.current_hp) / float(unit.hp)

    type_vecs = [np.zeros(N_UNIT_TYPES) for _ in range(N_PLAYERS)]
    type_vecs[int(unit.player_id)][int(unit.unit_type)] = 1.0

    return np.hstack([state_vec, *type_vecs])




class Tile(object):
    """
    This class holds all the information that a polytopia tile can hold (use enums for this). The tiles are instantiated based on the map creation 
    logic in board.py in the initialize function. It includes
    """
    def __init__(
            self,
            id: int,
            tile_type: TileType,
            city: City,
            tile_status: TileStatus,
            unit: Unit,
            player_controls: PlayerId
            ):
        
        self.id = id
        self.tile_type = tile_type
        self.city = city
        self.tile_status = tile_status
        self.unit = unit
        self.cntrl = player_controls
        self.has_road: bool = False

        self.is_edge = False # is this useful?


    def transform_to_node_features(self):
        """
        Transform a tile object into the vector node representation, where the ordering is as follows:
        
        """
        tile_type_feats = one_hot_field_type(self.tile_type)
        road_feat = np.array([float(self.has_road)])
        player_control_feats = player_controls_tile(self.cntrl)
        city_feats = city_featurizer(self.city)
        unit_feats = unit_featurizer(self.unit)

        return np.hstack([
            tile_type_feats,
            road_feat,
            player_control_feats,
            city_feats,
            unit_feats,
        ])


    def __eq__(self, other):
        if self.id == other.id:
            return True
        else:
            False
    

