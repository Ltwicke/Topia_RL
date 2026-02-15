# Here will be the entire logic for the game, which will then be wrapped in env for the RL task
import numpy as np
import math
import networkx as nx

from game.enums import UnitType, TileType, BoardType, PlayerId, TileStatus, Tribes, ActionTypes, UnitState
from game.components.board import Board
from game.components.player import Player
from game.components.units import Warrior, Rider


class Game(object):

    def __init__(self, board_config={}, player_tribes=[], debug_mode=False):

        self.game_board = Board(**board_config)
        self.all_tile_ids = np.arange(np.prod(self.game_board.board_size))
        self.n_players = board_config["n_players"]
        self.players = []

        for player_id in range(self.n_players):
            self.players.append(
                Player(PlayerId(player_id), player_tribes[player_id])
            )

    def reset_game(self):
        self.game_board.initialize()
        self.game_board.create_board_graph_from_board_state(self.all_tile_ids)
                                                            
        for player in self.players:
            player.reset(self.game_board)

        self.player_go_id = 0



    def apply_action(self, action: dict, return_message=False):
        """
        The big one! Takes an action-dictionary and then modifies the game_board according to the action
        """
        
        player = self.players[self.player_go_id]
        opponent = self.players[(self.player_go_id + 1) % 2] ## works only for 2 player mode

        if action["type"] == ActionTypes.MoveUnit:
            unit = player.units_under_control[action["unit"]]
            
            self.move_unit(unit, action["target_id"])

            self.apply_unit_vision(unit, action["path"])

            self.advance_unit_turn_state(unit)


        elif action["type"] == ActionTypes.Attack:
            unit = player.units_under_control[action["unit"]]  ## "unit" is an integer here
            o_unit = opponent.units_under_control[action["o_unit_index"]]
            unit_tile = unit.tile
            o_unit_tile = o_unit.tile

            attackResult, defenseResult = self.attack_retaliate_calc(unit, o_unit)

            unit_result_hp = unit.current_hp - defenseResult
            o_unit_result_hp = o_unit.current_hp - attackResult

            if o_unit_result_hp <= 0: ## attacker deletes defender --> No current_hp change
                attack_path = [unit_tile.id, o_unit_tile.id]
                unit_tile.unit = None ## attacker moves tile!
                o_unit_tile.unit = unit ## former defender tile now points to attacker. TODO: Include invalid movements
                
                ## update tile in unit:
                unit.tile = o_unit_tile
                
                if o_unit_tile.city != None:
                    o_unit_tile.city.unit = unit
                if unit_tile.city != None:
                    unit_tile.city.unit = None # because unit leaves city

                self.apply_unit_vision(unit, attack_path)

                del opponent.units_under_control[action["o_unit_index"]] ## remove defender pointer from opponent
                o_unit.city.current_n_units -= 1

                self.advance_unit_turn_state(unit)
                
                self.game_board.create_board_graph_from_board_state(self.all_tile_ids)
                player.construct_partial_graph_2players(self.game_board)
                return 1

            if unit_result_hp <= 0: ## attacker vanishes due to defender
                unit_tile.unit = None ## Delete unit pointer from tile
                if unit_tile.city != None:
                    unit_tile.city.unit = None

                del player.units_under_control[action["unit"]] # remove unit pointer from player
                unit.city.current_n_units -= 1
                o_unit.current_hp = o_unit_result_hp ## set new hp
                
                self.game_board.create_board_graph_from_board_state(self.all_tile_ids)
                player.construct_partial_graph_2players(self.game_board)
                return 1
            
            unit.current_hp = unit_result_hp
            o_unit.current_hp = o_unit_result_hp

            self.advance_unit_turn_state(unit)


        elif action["type"] == ActionTypes.CreateUnit:
            city = player.cities_under_control[action["city"]]
            #assert city.unit == None, "the city is not empty" ## Unecessary, because createUnit can only be selected, if conditions apply!
            city_tile = self.game_board.board[city.tile_id]

            if action["unit_type"] == UnitType.Warrior:
                unit = Warrior(
                    player_id=self.player_go_id,
                    city=city,
                    tile=city_tile
                )

            elif action["unit_type"] == UnitType.Rider:
                unit = Rider(
                    player_id=self.player_go_id,
                    city=city,
                    tile=city_tile
                )
                
            city_tile.unit = unit ## on city TILE
            city.unit = unit # on city object... THIS NEEDS FIXING IN GENERAL
            city.current_n_units += 1
            
            player.units_under_control.append(unit)

        
        elif action["type"] == ActionTypes.CaptureCity:
            unit = player.units_under_control[action["unit"]]  ## "unit" is an integer here
            unit_tile = unit.tile
            former_unit_city = unit.city
            city = unit_tile.city

            ## capture city:
            city.capture(player.player_id) # also sets current_n_units of city to 1

            former_unit_city.current_n_units -= 1

            unit.turn_state = UnitState.idle
            


        elif action["type"] == ActionTypes.EndTurn:
            self.player_go_id = (self.player_go_id + 1) % 2

            for unit in self.players[self.player_go_id].units_under_control:
                unit.set_ready() # set turn state to ready

            ## player_go_id gets his stars for the turn

        ## Create a new board_graph AND players partial graph:
        self.game_board.create_board_graph_from_board_state(self.all_tile_ids)
        player.construct_partial_graph_2players(self.game_board)



    def tiles_in_range(self, loc_ind: Int, distance: Int):
        x, y = self.game_board.int_to_tup[loc_ind] # returns (x, y)

        target_tiles_indices = [
            self.game_board.tup_to_int[(x + dx, y + dy)]
            for dx in range(-distance, distance+1)
            for dy in range(-distance, distance+1)
            if 0 <= x + dx <= self.game_board.board_size[0]-1 and 0 <= y + dy <= self.game_board.board_size[1]-1
        ]

        return target_tiles_indices # INCLUDING loc_ind


    def _has_reachable_node(self, G, source, cutoff):
        for u, v in nx.bfs_edges(G, source, depth_limit=cutoff):
            # if there is a single edge found:
            return 1 
        return 0

    
    def calc_movement_target_and_shortest_path(self, unit, target_tile=None, greedy_search=False):
        """Given the unit.mvpts and unit.tile.id, calculate all valid target ids and the shortest (valid) path to them.
        The road mechanic may be introduced by modifying edges between nodes to be different from trivial.
        TODO: Make this calculation a two step process; currently, i remove all nodes where a unit is standing on. While this is correct,
        that no other unit can walk onto the specific tile, its incorrect, because riders for example can jump over it. Therefore, nodes
        need to be removed for final target location but NOT for path finding!
        """
        G = self.game_board.movement_topology_graph.copy() # copy the graph structure for every function call

        partial_graph = self.players[unit.player_id].partial_graph 

        cant_step_on = partial_graph[:,0] == 0 # True if its not a field tile
        hidden_nodes = (partial_graph[:,0:3] == 0).all(axis=-1) # True if tile is not uncovered yet
        occupied = (partial_graph[:, 10:25] == 1).any(axis=-1) # True if any unit on tile TODO: Currently also blocks own units to jump over, this is WRONG!
        ## TODO: Enemy zone of control: Remove all nodes, where enemy units are adjacent

        invalid_mask = cant_step_on | hidden_nodes | occupied
        nodes_to_remove = [self.game_board.int_to_tup[index] for index in np.argwhere(invalid_mask).flatten()]
        unit_location_node = self.game_board.int_to_tup[unit.tile.id]
        try:
            nodes_to_remove.remove(unit_location_node)
        except:
            pass
        
        G.remove_nodes_from(nodes_to_remove)

        if greedy_search:
            return self._has_reachable_node(G, unit_location_node, unit.mvpts)

        paths_dict = nx.single_source_shortest_path(G, unit_location_node, cutoff=unit.mvpts)

        if target_tile != None:
            return paths_dict[self.game_board.int_to_tup[target_tile]]
        
        return paths_dict
        

    def move_unit(self, unit, target_tile_id):

        ## update source_tile:
        if unit.tile.city != None:
            unit.tile.city.unit = None
        
        unit.tile.unit = None

        ## update target tile:
        target_tile = self.game_board.board[target_tile_id]
        target_tile.unit = unit
        if target_tile.city != None:
            target_tile.city.unit = unit

        ## update unit tile pointer:
        unit.tile = target_tile


    def apply_unit_vision(self, unit, path):
        
        for tile_id in path:
            visioned_tile_ids = self.tiles_in_range(tile_id, distance=unit.vision_range)
            self.players[unit.player_id].uncovered_tile_ids.update(visioned_tile_ids)


    def attack_retaliate_calc(self, unit, o_unit, splash=False):
        """
        Calculate the resulting hp of both units and returns the result to be handled in the apply_action function
        """
        attackForce = unit.atk_stat * (unit.current_hp / unit.hp)
        defenseForce = o_unit.def_stat * (o_unit.current_hp / o_unit.hp) * o_unit.def_bonus 
        totalDamage = attackForce + defenseForce 
        attackResult = math.ceil((attackForce / totalDamage) * unit.atk_stat * 4.5) 
        defenseResult = math.ceil((defenseForce / totalDamage) * o_unit.def_stat * 4.5)

        if splash:
            attackResult /= 2

        return attackResult, defenseResult


    def advance_unit_turn_state(self, unit):
        """
        This function includes all the logic about unit turn_states. This necessitates for the surrounding of the unit.
        idle: the unit cannot do any action this turn anymore
        ready: the unit has not done any action this turn
        escaping: the unit cannot attack anymore, but can move
        can_hit: the unit can attack, but cannot move.
        """
        player = self.player_go_id # currently either 0 or 1
        opponent = (player + 1) % 2 ## 1 + 1 % 2 = 0, 0 + 1 % 2 = 1 WORKS ONLY FOR 2 PLAYERS
        surr_units = [
                    self.game_board.board[id].unit.player_id for id in self.tiles_in_range(unit.tile.id, unit.attack_range) \
                    if self.game_board.board[id].unit != None
                    ]
        current_state = unit.turn_state

        if unit.unit_type == UnitType.Warrior:

            if current_state == UnitState.ready:
                if opponent in surr_units:
                    unit.turn_state = UnitState.can_hit
                else:
                    unit.turn_state = UnitState.idle

            elif current_state == UnitState.can_hit:
                unit.turn_state = UnitState.idle

        elif unit.unit_type == UnitType.Rider:

            if current_state == UnitState.ready:
                if opponent in surr_units:
                    unit.turn_state = UnitState.can_hit
                else:
                    unit.turn_state = UnitState.escaping

            elif current_state == UnitState.escaping:
                unit.turn_state = UnitState.idle

            elif current_state == UnitState.can_hit:
                unit.turn_state = UnitState.escaping




