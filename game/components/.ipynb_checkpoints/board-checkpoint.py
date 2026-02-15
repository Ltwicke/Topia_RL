## This will define how the board behaves -> one big graph structure
import numpy as np
import networkx as nx
from random import sample

from game.enums import UnitType, TileType, BoardType, PlayerId, TileStatus
from game.components.tile import Tile
from game.components.city import City
from game.components.units import Warrior

NODE_FEAT_DIM = 26  ## this needs to be adjusted, when enums get bigger...
N_PLAYERS = len(PlayerId)

def board_generating_logic(board_size, board_type, n_players):
    """This function builds a plan how the graph should be created. This function will be moved to components or get its own file."""

    if board_type == BoardType.Dummy:
        water_matrix = (np.random.rand(board_size[0], board_size[1]) < 0.1).astype(int) 
        village_matrix = (np.random.rand(board_size[0], board_size[1]) < 0.1).astype(int) * (1-water_matrix)
        chosen_capitals = sample(list(np.argwhere(village_matrix == 1)), len(PlayerId))

        capital_matrix = np.zeros_like(village_matrix)
        capital_matrix[tuple(zip(*chosen_capitals))] = 1

        return np.stack([water_matrix, village_matrix, capital_matrix], axis=0) # shape (3, board_size)

class Board(object):
    """
    The board is holding all necessary objects to represent the full, visible game board.
    Individual board views will be handled in the player.py file. 
    """
    def __init__(self, board_size, board_type, n_players):
        self.board_size = board_size
        self.board_type = board_type
        self.n_players = n_players

        self.board = []
        self.board_graph = np.empty(shape=(board_size[0] * board_size[1], NODE_FEAT_DIM))

        self.movement_topology_graph = nx.grid_2d_graph(board_size[0], board_size[1])
        ## add diagonal edges:
        diagonals = []
        for x in range(board_size[0] - 1):
            for y in range(board_size[1] - 1):
                # Diagonale: Unten-Rechts
                diagonals.append(((x, y), (x + 1, y + 1)))
                # Diagonale: Unten-Links
                diagonals.append(((x + 1, y), (x, y + 1)))
        self.movement_topology_graph.add_edges_from(diagonals)

        self.initialize()

    def initialize(self):
        """This function creates an empty board based on the creation logic.
        It does not create the graph yet, the graph is steadily created from the board from the enum objects in a one-hot-encoded way."""
        board_matrix = board_generating_logic(self.board_size, self.board_type, self.n_players)
        capital_assign_counter = 0

        self.board = []
        self.capital_tile_ids = {}
        self.int_to_tup = {}
        self.tup_to_int = {}

        for ind, (i, j) in enumerate(np.ndindex(board_matrix.shape[1:])):
            ## also collect some dics here, such as capital P1 : ind
            design_vec = board_matrix[:, i, j] # holds [water, village, capital] currently

            self.int_to_tup[ind] = (i, j)
            self.tup_to_int[(i,j)] = ind

            city = None
            unit = None
            tile_status = TileStatus.no_status
            player_control = None

            field_type = TileType(design_vec[0])

            if (design_vec[1] and not design_vec[2]):
                city = City(None, ind, is_capital=False)

            elif (design_vec[1] and design_vec[2]):
                city = City(PlayerId(capital_assign_counter), ind, is_capital=True)
                self.capital_tile_ids[PlayerId(capital_assign_counter).value] = ind
                capital_assign_counter += 1 ## assign next capital to next player

            tile = Tile(
                id=ind,
                tile_type=field_type,
                city=city,
                tile_status=tile_status,
                unit=unit,
                player_controls=player_control
                )
            
            self.board.append(tile)

        ## place starting units: 
        for player_id, capital_id in self.capital_tile_ids.items():
            unit = Warrior(
                player_id=PlayerId(player_id),
                city=self.board[capital_id].city,
                tile=self.board[capital_id]
                )
            self.board[capital_id].unit = unit
            self.board[capital_id].city.unit = unit


    def create_board_graph_from_board_state(self, active_tile_inds):
        """
        Uses self.board to create a one-hot encoded graph based on the current board state.
        TODO: A matrix mask would potentially be faster for uncovered_tile_ids...
        """
        for tile in self.board:
            if tile.id in active_tile_inds: # only update the tiles that have been involved in the action
                self.board_graph[tile.id] = tile.transform_to_node_features()




