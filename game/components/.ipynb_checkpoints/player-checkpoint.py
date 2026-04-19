from dataclasses import dataclass, field
import numpy as np
from copy import copy

from game.enums import PlayerId, BoardType, Tribes
from game.components.board import Board


_SCORE_PER_CITY = {
    
}

_SCORE_PER_UNIT = {
    
}



@dataclass
class Player(object):
    """
    All objects such as units and cities will be handled in the board class. The functions of the player are held minimal to better translate into RL.
    Call self.reset to instantiate a bunch of self variables, when a board is given.

    This class may later also include some form of "suspicion" for example, by counting the score of the other players and estimating their
    unit count from turns or something...
    """
    def __init__(self, id: PlayerId, tribe: Tribes):
        self.player_id = id
        self.tribe = tribe

    def reset(self, board):
        """
        This is basically a second constructor. Sets up the game for turn 0.
        Specifically:   cover everything except capital vision,
                        only starting unit under control, 
                        Set spt to Tribe starting spt,
        """

        self.partial_graph = np.empty(shape=board.board_graph.shape)

        self.uncovered_tile_ids = set()
        self.units_under_control: dict = {}  # unit_id -> Unit
        self.cities_under_control = []
        self.stars = 5

        self.capital_id = board.capital_tile_ids[self.player_id.value]
        x, y = board.int_to_tup[self.capital_id]
        ## get indices centered at capital_id for 2 tiles in every direction:
        capital_vision_indices = [
            (x + dx, y + dy)
            for dx in range(-2, 2+1)
            for dy in range(-2, 2+1)
            if 0 <= x + dx <= board.board_size[0]-1 and 0 <= y + dy <= board.board_size[1]-1
        ]
        self.uncovered_tile_ids.update([board.tup_to_int[tup] for tup in capital_vision_indices])

        ## modify partial graph here:
        self.construct_partial_graph_2players(board)

        ## collect starting unit:
        unit = board.board[self.capital_id].unit
        unit.set_ready() # only at the start of the game!
        self.units_under_control[unit.unit_id] = unit
        self.cities_under_control.append(board.board[self.capital_id].city)

        ## stars per turn based on capital:

    @property
    def current_stars_per_turn(self) -> int:
        spt = 0 
        for city in self.cities_under_control:
            spt += city.city_stars_per_turn

    @property
    def current_score(self) -> int:
        score = 0
        


    def construct_partial_graph_2players(self, board):
        """
        Use uncovered tiles to create the partial graph from the board graph.
        Switch dimensions based on self.id:
        P1 view: P1 P2 P3 ...
        P2 view: P2 P1 P3 ...
        P3 view: P3 P1 P2 ...
        BIG TODO: Need IntEnum class to store the starting dimensions of each thing.
        """
        self.partial_graph = copy(board.board_graph) 

        # switch only for player P2:
        if self.player_id.value == 1:  ## PlayerId.P2
            self.partial_graph[:, [3, 4]] = self.partial_graph[:, [4, 3]]              ## player control [3:5]
            self.partial_graph[:, [6, 7, 8, 9]] = self.partial_graph[:, [8, 9, 6, 7]] ## city blocks [6:10]
            ## unit type blocks only [14:18] — state [10:14] is player-agnostic, never swapped
            self.partial_graph[:, [14, 15, 16, 17]] = self.partial_graph[:, [16, 17, 14, 15]]

        # conceal:
        self.partial_graph[~np.isin(np.arange(self.partial_graph.shape[0]), list(self.uncovered_tile_ids))] = 0


