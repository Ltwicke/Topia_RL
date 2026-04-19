# Here will be the entire logic for the game, which will then be wrapped in env for the RL task
import random
import numpy as np
import math
import networkx as nx

from game.enums import (
    UnitType, TileType, BoardType, PlayerId, TileStatus, Tribes, ActionTypes, UnitState, DefenseBonus,
    OWN_TYPE_SLICE, OPP_TYPE_SLICE, _TILE_TYPE_START,
)
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
        self._used_unit_ids: set[int] = set()
        self.game_board.initialize(self)
        self.game_board.create_board_graph_from_board_state(self.all_tile_ids)

        for player in self.players:
            player.reset(self.game_board)

        self.player_go_id = 0
        self.turn = 0

    def _new_unit_id(self) -> int:
        """Generate a random integer ID (0-9999) unique for this game session.
        IDs are never removed from _used_unit_ids, so dead units' IDs are never recycled."""
        uid = random.randint(0, 9999)
        while uid in self._used_unit_ids:
            uid = random.randint(0, 9999)
        self._used_unit_ids.add(uid)
        return uid


    def apply_action(self, action: dict, return_message=False):
        """
        The big one! Takes an action-dictionary and then modifies the game_board according to the action
        """
        
        player = self.players[self.player_go_id]
        opponent = self.players[(self.player_go_id + 1) % 2] ## works only for 2 player mode
        message = {}
        message["action_type"] = action["type"] # stores ActionTypes

        if action["type"] == ActionTypes.MoveUnit:
            unit = player.units_under_control[action["unit_id"]]
            
            self.move_unit(unit, action["target_id"])

            n_tiles_uncovered = self.apply_unit_vision(unit, action["path"])

            self.advance_unit_turn_state(unit, action)

            if unit.tile.city != None:
                self._apply_unit_def_bonus(unit)
            else:
                unit.def_bonus = DefenseBonus.NoBonus

            message["tiles_uncovered"] = n_tiles_uncovered


        elif action["type"] == ActionTypes.Attack:
            unit = player.units_under_control[action["unit_id"]]
            o_unit = opponent.units_under_control[action["o_unit_id"]]
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
                    self._apply_unit_def_bonus(unit) # You had to change to .value for the enum!
                else:
                    unit.def_bonus = DefenseBonus.NoBonus

                self.apply_unit_vision(unit, attack_path)

                del opponent.units_under_control[action["o_unit_id"]] ## remove defender pointer from opponent
                o_unit.city.current_n_units -= 1

                self.advance_unit_turn_state(unit, action)
                
                self.game_board.create_board_graph_from_board_state(self.all_tile_ids)
                player.construct_partial_graph_2players(self.game_board)

                message["killed_unit"] = 1
                
                return message ## return here, because due to the calculation, unit_result_hp could also be 0

            if unit_result_hp <= 0: ## attacker vanishes due to defender
                unit_tile.unit = None ## Delete unit pointer from tile

                del player.units_under_control[action["unit_id"]] # remove unit pointer from player
                unit.city.current_n_units -= 1
                o_unit.current_hp = o_unit_result_hp ## set new hp
                
                self.game_board.create_board_graph_from_board_state(self.all_tile_ids)
                player.construct_partial_graph_2players(self.game_board)

                message["killed_unit"] = 0
                
                return message
            
            unit.current_hp = unit_result_hp
            o_unit.current_hp = o_unit_result_hp

            self.advance_unit_turn_state(unit, action)

            message["killed_unit"] = 0


        elif action["type"] == ActionTypes.CreateUnit:
            city = player.cities_under_control[action["city"]]
            #assert city.unit == None, "the city is not empty" ## Unecessary, because createUnit can only be selected, if conditions apply!
            city_tile = self.game_board.board[city.tile_id]

            new_uid = self._new_unit_id()
            if action["unit_type"] == UnitType.Warrior:
                unit = Warrior(
                    player_id=PlayerId(self.player_go_id),
                    city=city,
                    tile=city_tile,
                    unit_id=new_uid,
                )
            elif action["unit_type"] == UnitType.Rider:
                unit = Rider(
                    player_id=PlayerId(self.player_go_id),
                    city=city,
                    tile=city_tile,
                    unit_id=new_uid,
                )

            city_tile.unit = unit ## on city TILE
            city.current_n_units += 1

            unit.def_bonus = DefenseBonus.Shield

            player.units_under_control[unit.unit_id] = unit

            message["unit_type"] = action["unit_type"]

        
        elif action["type"] == ActionTypes.CaptureCity:
            unit = player.units_under_control[action["unit_id"]]
            unit_tile = unit.tile
            former_unit_city = unit.city
            city = unit_tile.city
            city_tile_id = city.tile_id
            former_player_id = city.player_id

            ## capture city:
            city.capture(player.player_id) # also sets current_n_units of city to 1

            if former_player_id != None: #meaning the city belonged to someone
                opponent_city_tile_ids = [city.tile_id for city in opponent.cities_under_control]
                del opponent.cities_under_control[opponent_city_tile_ids.index(city_tile_id)] ## removes the correct city

            player.cities_under_control.append(city)

            former_unit_city.current_n_units -= 1

            unit.turn_state = UnitState.idle

            unit.def_bonus = DefenseBonus.Shield
            

        elif action["type"] == ActionTypes.EndTurn:
            self.turn += self.player_go_id % 2 # 0 1 0 1 0 1 0 1 ...
            self.player_go_id = (self.player_go_id + 1) % 2

            for unit in self.players[self.player_go_id].units_under_control.values():
                unit.set_ready() # set turn state to ready

            ## player_go_id gets his stars for the turn

        ## Create a new board_graph AND players partial graph:
        self.game_board.create_board_graph_from_board_state(self.all_tile_ids)
        player.construct_partial_graph_2players(self.game_board)

        return message



    def tiles_in_range(self, loc_ind: Int, distance: Int):
        x, y = self.game_board.int_to_tup[loc_ind] # returns (x, y)

        target_tiles_indices = [
            self.game_board.tup_to_int[(x + dx, y + dy)]
            for dx in range(-distance, distance+1)
            for dy in range(-distance, distance+1)
            if 0 <= x + dx <= self.game_board.board_size[0]-1 and 0 <= y + dy <= self.game_board.board_size[1]-1
        ]

        return target_tiles_indices # INCLUDING loc_ind


    def _apply_unit_def_bonus(self, unit):
        if unit.tile.city.player_id == None:
            unit.def_bonus = DefenseBonus.NoBonus
        elif unit.tile.city.player_id.value == self.player_go_id: # players unit on his own city
            unit.def_bonus = DefenseBonus.Shield
        else:
            unit.def_bonus = DefenseBonus.NoBonus # in normal map generation this is not even possible to reach here
            
    
    def calc_movement_target_and_shortest_path(self, unit, target_tile=None, greedy_search=False):
        """Calculate valid movement destinations and shortest paths for unit.

        Transit rules:
          - Hidden tiles and non-field tiles block ALL movement through them.
          - Enemy-occupied tiles and their ZoC neighbors block transit (unit must stop before).
          - Friendly-occupied tiles are passable as intermediate nodes.
          - ZoC tiles (adjacent to enemies) can be stopping destinations but not transit nodes.

        Road mechanic: edge weight < 1.0 reduces movement cost; Dijkstra respects weights.
        """
        partial_graph = self.players[unit.player_id].partial_graph

        # Phase 0 — classify tiles
        cant_step_on   = partial_graph[:, _TILE_TYPE_START] == 0               # not a field (hidden tiles are also 0)
        own_occupied   = (partial_graph[:, OWN_TYPE_SLICE] != 0).any(axis=-1)
        enemy_occupied = (partial_graph[:, OPP_TYPE_SLICE] != 0).any(axis=-1)
        any_occupied   = own_occupied | enemy_occupied
        destination_blocked = cant_step_on | any_occupied

        # Phase 1 — ZoC set: tiles adjacent to (or occupied by) visible enemy units
        enemy_tile_ids = set(np.argwhere(enemy_occupied).flatten())
        zoc_ids: set[int] = set(enemy_tile_ids)
        for eid in enemy_tile_ids:
            for nbr in self.game_board.movement_topology_graph.neighbors(
                    self.game_board.int_to_tup[eid]):
                zoc_ids.add(self.game_board.tup_to_int[nbr])

        # Phase 2 — build transit graph
        G = self.game_board.movement_topology_graph.copy()
        zoc_arr = np.zeros(len(partial_graph), dtype=bool)
        for i in zoc_ids:
            zoc_arr[i] = True

        transit_blocked = cant_step_on | enemy_occupied | zoc_arr
        nodes_to_remove = [self.game_board.int_to_tup[i]
                           for i in np.argwhere(transit_blocked).flatten()]
        unit_loc = self.game_board.int_to_tup[unit.tile.id]
        if unit_loc in nodes_to_remove:
            nodes_to_remove.remove(unit_loc)
        G.remove_nodes_from(nodes_to_remove)

        if unit_loc not in G:
            return False if greedy_search else ({} if target_tile is None else [])

        # Phase 3 — Dijkstra (respects road edge weights)
        lengths, paths = nx.single_source_dijkstra(
            G, unit_loc, cutoff=unit.mvpts, weight='weight')

        if greedy_search:
            if any(not destination_blocked[self.game_board.tup_to_int[n]]
                   for n in lengths if n != unit_loc):
                return True
            # also check ZoC tiles reachable as one-step stopping points
            for node, cost in lengths.items():
                for nbr in self.game_board.movement_topology_graph.neighbors(node):
                    nbr_id = self.game_board.tup_to_int[nbr]
                    if nbr_id not in zoc_ids or destination_blocked[nbr_id]:
                        continue
                    step = self.game_board.movement_topology_graph \
                               .get_edge_data(node, nbr).get('weight', 1.0)
                    if cost + step <= unit.mvpts:
                        return True
            return False

        # Extend: ZoC tiles reachable as stopping destinations (one step from transit nodes)
        for node, cost in list(lengths.items()):
            for nbr in self.game_board.movement_topology_graph.neighbors(node):
                nbr_id = self.game_board.tup_to_int[nbr]
                if nbr_id not in zoc_ids or destination_blocked[nbr_id]:
                    continue
                step = self.game_board.movement_topology_graph \
                           .get_edge_data(node, nbr).get('weight', 1.0)
                if cost + step <= unit.mvpts and nbr not in lengths:
                    lengths[nbr] = cost + step
                    paths[nbr] = paths[node] + [nbr]

        valid_paths = {
            node: path for node, path in paths.items()
            if node != unit_loc
            and not destination_blocked[self.game_board.tup_to_int[node]]
        }

        if target_tile is not None:
            return valid_paths[self.game_board.int_to_tup[target_tile]]
        return valid_paths
        

    def move_unit(self, unit, target_tile_id):

        ## update source_tile: move away from city
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

        delta_uncovered_tiles = 0
        player_uncovered_tiles = self.players[unit.player_id].uncovered_tile_ids
        delta_uncovered_tiles -= len(player_uncovered_tiles)
        
        for tile_id in path:
            visioned_tile_ids = self.tiles_in_range(tile_id, distance=unit.vision_range)
            player_uncovered_tiles.update(visioned_tile_ids)
            
        delta_uncovered_tiles += len(player_uncovered_tiles)
        return delta_uncovered_tiles


    def attack_retaliate_calc(self, unit, o_unit, splash=False):
        """
        Calculate the resulting hp of both units and returns the result to be handled in the apply_action function
        """
        attackForce = unit.atk_stat * (unit.current_hp / unit.hp)
        defenseForce = o_unit.def_stat * (o_unit.current_hp / o_unit.hp) * o_unit.def_bonus.value 
        totalDamage = attackForce + defenseForce 
        attackResult = math.ceil((attackForce / totalDamage) * unit.atk_stat * 4.5) 
        defenseResult = math.ceil((defenseForce / totalDamage) * o_unit.def_stat * 4.5)

        if splash:
            attackResult /= 2

        return attackResult, defenseResult


    def advance_unit_turn_state(self, unit, action):
        """
        This function includes all the logic about unit turn_states. This necessitates for the surrounding of the unit.
        idle: the unit cannot do any action this turn anymore
        ready: the unit has not done any action this turn
        escaping: the unit cannot attack anymore, but can move
        can_hit: the unit can attack, but cannot move.

        TODO: This has to be reconsidered; There are too many situations that depend on factors outside of unit. OR include the action
        and make it a little bit more ordered.
        """
        player = self.player_go_id # currently either 0 or 1
        opponent = (player + 1) % 2 ## 1 + 1 % 2 = 0, 0 + 1 % 2 = 1 WORKS ONLY FOR 2 PLAYERS
        surr_units = [
                    self.game_board.board[id].unit.player_id for id in self.tiles_in_range(unit.tile.id, unit.attack_range) \
                    if self.game_board.board[id].unit != None # no else statement? Does it default to None?
                    ]
        current_state = unit.turn_state
        action_type = action["type"] # one of the enums

        if unit.unit_type == UnitType.Warrior:
            # some action has happened; e.g. warrior was moved, warrior attacked, now change turn_state on the unit:
            if action_type == ActionTypes.MoveUnit: # only possible from ready state
                if opponent in surr_units:
                    unit.turn_state = UnitState.can_hit 
                else:
                    unit.turn_state = UnitState.idle
            # action_type == Attack: 
            elif action_type == ActionTypes.Attack:
                unit.turn_state = UnitState.idle

        elif unit.unit_type == UnitType.Rider:

            if action_type == ActionTypes.MoveUnit:
                if current_state == UnitState.ready:
                    if opponent in surr_units:
                        unit.turn_state = UnitState.can_hit # same here as above
                    else:
                        unit.turn_state = UnitState.idle
                elif current_state == UnitState.escaping: # an escaping rider can only move, so we will be in this if
                    unit.turn_state = UnitState.idle
            # action_type == Attack
            elif action_type == ActionTypes.Attack:
                unit.turn_state = UnitState.escaping




