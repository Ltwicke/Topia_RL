# Here will be the entire logic for the game, which will then be wrapped in env for the RL task
import random
import numpy as np
import math
import networkx as nx
from types import SimpleNamespace

from game.enums import (
    UnitType, TileType, BoardType, CityType, PlayerId, TileStatus, Tribes, ActionTypes, UnitState, DefenseBonus,
    OWN_TYPE_SLICE, OPP_TYPE_SLICE, _TILE_TYPE_START, _ROAD_START,
)
from game.components.board import Board
from game.components.city import _CITY_UPGRADES, _CITY_UPGRADE_COST
from game.components.player import Player
from game.components.units import Warrior, Rider, Knight, Giant, Archer, Catapult, Sword, Defender


_TURN_STATE_TRANSITIONS = [
    # (unit_type, from_state, action_type, enemy_adjacent, new_state)
    (UnitType.Warrior, UnitState.ready,    ActionTypes.MoveUnit, False, UnitState.idle),
    (UnitType.Warrior, UnitState.ready,    ActionTypes.MoveUnit, True,  UnitState.can_hit),
    (UnitType.Warrior, None,               ActionTypes.Attack,   None,  UnitState.idle),
    ## ARCHER
    (UnitType.Archer, UnitState.ready,    ActionTypes.MoveUnit, False, UnitState.idle),
    (UnitType.Archer, UnitState.ready,    ActionTypes.MoveUnit, True,  UnitState.can_hit), 
    (UnitType.Archer, None,               ActionTypes.Attack,   None,  UnitState.idle),
    ## SWORDSMAN
    (UnitType.Sword, UnitState.ready,    ActionTypes.MoveUnit, False, UnitState.idle),
    (UnitType.Sword, UnitState.ready,    ActionTypes.MoveUnit, True,  UnitState.can_hit),
    (UnitType.Sword, None,               ActionTypes.Attack,   None,  UnitState.idle),
    ## KNIGHT
    (UnitType.Knight, UnitState.ready,    ActionTypes.MoveUnit, False, UnitState.idle),
    (UnitType.Knight, UnitState.ready,    ActionTypes.MoveUnit, True,  UnitState.can_hit),
    # Knight re-attack handled in advance_unit_turn_state directly (needs kill info from message)
    ## CATAPULT
    (UnitType.Catapult, UnitState.ready,    ActionTypes.MoveUnit, False, UnitState.idle),
    (UnitType.Catapult, UnitState.ready,    ActionTypes.MoveUnit, True,  UnitState.idle),
    (UnitType.Catapult, None,               ActionTypes.Attack,   None,  UnitState.idle),
    ## GIANT
    (UnitType.Giant, UnitState.ready,    ActionTypes.MoveUnit, False, UnitState.idle),
    (UnitType.Giant, UnitState.ready,    ActionTypes.MoveUnit, True,  UnitState.idle),
    (UnitType.Giant, None,               ActionTypes.Attack,   None,  UnitState.idle),
    ## DEFENDER
    (UnitType.Defender, UnitState.ready,    ActionTypes.MoveUnit, False, UnitState.idle),
    (UnitType.Defender, UnitState.ready,    ActionTypes.MoveUnit, True,  UnitState.idle),
    (UnitType.Defender, None,               ActionTypes.Attack,   None,  UnitState.idle),
    ## RIDER
    (UnitType.Rider, UnitState.ready,    ActionTypes.MoveUnit, False, UnitState.idle),
    (UnitType.Rider, UnitState.ready,    ActionTypes.MoveUnit, True,  UnitState.can_hit),
    (UnitType.Rider, UnitState.escaping, ActionTypes.MoveUnit, None,  UnitState.idle),
    (UnitType.Rider, None,               ActionTypes.Attack,   None,  UnitState.escaping),
]


def col_round(x):
    frac = x - math.floor(x)
    if frac < 0.5: return math.floor(x)
    return math.ceil(x)


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
        self.game_board.create_board_graph_from_board_state(self.all_tile_ids) # once before player.reset

        for player in self.players:
            player.reset(self.game_board)

        for player in self.players:
            for city in player.cities_under_control:
                self._claim_city_territory(city, player.player_id, distance=1)

        self.game_board.create_board_graph_from_board_state(self.all_tile_ids) # again after distributing control

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

            self.advance_unit_turn_state(unit, action, message)

            """ Already handled in move_unit
            if unit.tile.city != None:
                self._apply_unit_def_bonus(unit)
            else:
                unit.def_bonus = DefenseBonus.NoBonus
            """

            message["tiles_uncovered"] = n_tiles_uncovered


        elif action["type"] == ActionTypes.Attack:
            unit = player.units_under_control[action["unit_id"]]
            o_unit = opponent.units_under_control[action["o_unit_id"]]
            unit_tile = unit.tile
            o_unit_tile = o_unit.tile

            # retaliation gate: defender retaliates only if (a) not stiff AND (b) attacker is within its reach
            dist = self._tile_distance(unit_tile.id, o_unit_tile.id)
            retaliates = (not o_unit.stiff) and (dist <= o_unit.attack_range)

            attackResult, defenseResult = self.attack_retaliate_calc(unit, o_unit)
            if not retaliates:
                defenseResult = 0

            unit_result_hp = unit.current_hp - defenseResult
            o_unit_result_hp = o_unit.current_hp - attackResult

            if o_unit_result_hp <= 0: ## attacker deletes defender --> No current_hp change
                is_ranged_attack = dist > 1  # no advance into defender tile on ranged kill
                if not is_ranged_attack:
                    attack_path = [unit_tile.id, o_unit_tile.id]
                    unit_tile.unit = None ## attacker moves tile!
                    o_unit_tile.unit = unit ## former defender tile now points to attacker
                    unit.tile = o_unit_tile
                    self._apply_unit_def_bonus(unit)
                    self.apply_unit_vision(unit, attack_path)
                else:
                    # ranged kill: attacker stays put; just clear the dead defender from its tile
                    o_unit_tile.unit = None

                del opponent.units_under_control[action["o_unit_id"]] ## remove defender key from opponent
                o_unit.city.current_n_units -= 1
                unit.kills += 1

                message["killed_unit"] = 1

                self.advance_unit_turn_state(unit, action, message)

                self.game_board.create_board_graph_from_board_state(self.all_tile_ids)
                player.construct_partial_graph_2players(self.game_board)

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

            self.advance_unit_turn_state(unit, action, message)

            message["killed_unit"] = 0


        elif action["type"] == ActionTypes.CreateUnit:
            city = player.cities_under_control[action["city"]]
            #assert city.unit == None, "the city is not empty" ## Unecessary, because createUnit can only be selected, if conditions apply!
            city_tile = self.game_board.board[city.tile_id]

            new_uid = self._new_unit_id()
            _UNIT_CLASSES = {
                UnitType.Warrior:  Warrior,
                UnitType.Rider:    Rider,
                UnitType.Knight:   Knight,
                UnitType.Giant:    Giant,
                UnitType.Archer:   Archer,
                UnitType.Catapult: Catapult,
                UnitType.Sword:    Sword,
                UnitType.Defender: Defender,
            }
            unit = _UNIT_CLASSES[action["unit_type"]](
                player_id=PlayerId(self.player_go_id),
                city=city,
                tile=city_tile,
                unit_id=new_uid,
            )

            city_tile.unit = unit ## on city TILE
            city.current_n_units += 1

            self._apply_unit_def_bonus(unit)

            player.units_under_control[unit.unit_id] = unit
            player.stars -= unit.cost

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
            if city.controlled_tile_ids:
                self._transfer_city_control(city, player.player_id)
            else:
                self._claim_city_territory(city, player.player_id, distance=1)

            if former_player_id != None: #meaning the city belonged to someone
                opponent_city_tile_ids = [city.tile_id for city in opponent.cities_under_control]
                del opponent.cities_under_control[opponent_city_tile_ids.index(city_tile_id)] ## removes the correct city

            player.cities_under_control.append(city)

            former_unit_city.current_n_units -= 1

            unit.turn_state = UnitState.idle

            self._apply_unit_def_bonus(unit)

        elif action["type"] == ActionTypes.HealUnit:
            unit = player.units_under_control[action["unit_id"]]
            heal_amount = 4.0 if unit.tile.cntrl == player.player_id else 2.0
            unit.current_hp = min(unit.hp, unit.current_hp + heal_amount)
            unit.turn_state = UnitState.idle
            message["heal_amount"] = heal_amount

        elif action["type"] == ActionTypes.UpgradeCity:
            city = player.cities_under_control[action["city"]]
            choice = action["choice"]
            next_lvl = _CITY_UPGRADES[city.lvl][choice]
            cost = max(0, _CITY_UPGRADE_COST[next_lvl] - city.pending_discount) #increase cost by 2 stars for every enemy unit inside city borders
            city.pending_discount = 0
            player.stars -= cost

            # capture pre-upgrade state before city.upgrade() mutates it
            will_explore = (city.lvl == CityType.lvl1 and choice == 1)
            will_create_giant = (city.times_upgraded >= 3 and choice == 0)

            city.upgrade(choice)

            if city.lvl == CityType.lvl3_resources:
                player.stars += 5
            elif city.lvl == CityType.lvl4_popgrwth:
                city.pending_discount = 6
            elif city.lvl == CityType.lvl4_bordergrwth:
                self._claim_city_territory(city, player.player_id, distance=2)
            elif city.lvl == CityType.lvl8plus:
                city.pending_discount = - 10 * (city.times_upgraded-7) # increase the cost of upgrading cities more and more

            if will_explore:
                self._apply_explorer_vision(player, city.tile_id)

            if will_create_giant:
                self._create_giant_at_city(player, opponent, city)

            city_tile = self.game_board.board[city.tile_id]
            if city_tile.unit is not None and city_tile.unit.player_id == player.player_id:
                self._apply_unit_def_bonus(city_tile.unit)

            message["new_lvl"] = city.lvl
            message["choice"] = choice

        elif action["type"] == ActionTypes.PlaceRoad:
            tile = self.game_board.board[action["tile_id"]]
            if tile.tile_type == TileType.field:
                tile.has_road = True
                player.stars -= 5
            elif tile.tile_type == TileType.water:
                assert self._bridge_axis(tile) is not None, "invalid bridge placement"
                tile.has_road = True
                player.stars -= 9
            self.game_board._update_road_edge_weights()

        elif action["type"] == ActionTypes.Upgrade2Vet:
            unit = player.units_under_control[action["unit_id"]]
            old_curr_hp = unit.current_hp
            unit.is_vet = True
            unit.hp += 5
            unit.current_hp = unit.hp
            #UNIT TURN STATE DOES NOT CHANGE, THIS IS CORRECT!
            message["hp_diff"] = unit.current_hp - old_curr_hp

        elif action["type"] == ActionTypes.EndTurn:
            for unit in self.players[self.player_go_id].units_under_control.values():
                unit.set_idle() # set all units of turn ending player to idle to make the networks life easier
            
            self.turn += self.player_go_id % 2 # 0 1 0 1 0 1 0 1 ...
            self.player_go_id = (self.player_go_id + 1) % 2
            new_player = self.players[self.player_go_id] # after ending the turn

            for unit in new_player.units_under_control.values():
                unit.set_ready() # set turn state to ready

            self._refresh_siege_states()
            ## player_go_id gets his stars for the turn
            if self.turn > 0:
                new_player.stars += new_player.current_stars_per_turn

            self.game_board.create_board_graph_from_board_state(self.all_tile_ids)
            new_player.construct_partial_graph_2players(self.game_board)            # important: create for new player
            return message

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


    def _refresh_siege_states(self):
        for player in self.players:
            for city in player.cities_under_control:
                tile = self.game_board.board[city.tile_id]
                city.under_siege = (
                    tile.unit is not None
                    and tile.unit.player_id != city.player_id
                )

    def _claim_city_territory(self, city, player_id, distance):
        for tid in self.tiles_in_range(city.tile_id, distance):
            tile = self.game_board.board[tid]
            if tile.cntrl is None:
                tile.cntrl = player_id
                city.controlled_tile_ids.append(tid)

    def _transfer_city_control(self, city, new_player_id):
        for tid in city.controlled_tile_ids:
            self.game_board.board[tid].cntrl = new_player_id

    def _tile_distance(self, tid_a, tid_b):
        ax, ay = self.game_board.int_to_tup[tid_a]
        bx, by = self.game_board.int_to_tup[tid_b]
        return max(abs(ax - bx), abs(ay - by))

    def _bridge_axis(self, tile):
        """Return 'NS', 'WE', or None for the valid bridge orientation of a water tile."""
        H, W = self.game_board.board_size
        r, c = self.game_board.int_to_tup[tile.id]
        walkable = {TileType.field, TileType.mountain}

        def neighbour_type(dr, dc):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                return self.game_board.board[self.game_board.tup_to_int[(nr, nc)]].tile_type
            return None

        if neighbour_type(-1, 0) in walkable and neighbour_type(+1, 0) in walkable:
            return 'NS'
        if neighbour_type(0, -1) in walkable and neighbour_type(0, +1) in walkable:
            return 'WE'
        return None

    def _apply_unit_def_bonus(self, unit):
        if unit.tile.tile_type == TileType.mountain:
            unit.def_bonus = DefenseBonus.Shield
            return
        city = unit.tile.city
        if (city is None
                or city.player_id is None
                or city.player_id.value != self.player_go_id
                or not unit.fortify):
            unit.def_bonus = DefenseBonus.NoBonus
            return
        has_wall = len(city.choices) >= 2 and city.choices[1] == 1
        unit.def_bonus = DefenseBonus.Wall if has_wall else DefenseBonus.Shield
            
    
    def _apply_explorer_vision(self, player, start_tile_id, n_steps=14):
        current = start_tile_id
        path = [current]
        uncovered = player.uncovered_tile_ids

        for _ in range(n_steps):
            tup = self.game_board.int_to_tup[current]
            walkable = [
                self.game_board.tup_to_int[n]
                for n in self.game_board.movement_topology_graph.neighbors(tup)
                if self.game_board.board[self.game_board.tup_to_int[n]].tile_type
                   not in (TileType.water, TileType.deep_water)
            ]
            if not walkable:
                break
            scores = {tid: sum(1 for v in self.tiles_in_range(tid, 1) if v not in uncovered)
                      for tid in walkable}
            best = max(scores.values())
            current = random.choice([t for t, s in scores.items() if s == best])
            path.append(current)

        dummy = SimpleNamespace(player_id=player.player_id, vision_range=1)
        self.apply_unit_vision(dummy, path)

    def _create_giant_at_city(self, player, opponent, city):
        city_tile = self.game_board.board[city.tile_id]
        if city_tile.unit is not None:
            occupant = city_tile.unit
            free_adj = [
                t for t in self.tiles_in_range(city.tile_id, 1)
                if t != city.tile_id # not the city itself
                and self.game_board.board[t].unit is None # no other unit on it
                and (self.game_board.board[t].tile_type != TileType.water or self.game_board.board[t].has_road == True) # not pure water
            ]
            if free_adj:
                self.move_unit(occupant, random.choice(free_adj))
            else:
                city_tile.unit = None
                occupant.city.current_n_units -= 1
                if occupant.player_id == player.player_id:
                    del player.units_under_control[occupant.unit_id]
                else:
                    del opponent.units_under_control[occupant.unit_id]

        new_uid = self._new_unit_id()
        giant = Giant(player_id=player.player_id, city=city,
                      tile=city_tile, unit_id=new_uid)
        city_tile.unit = giant
        city.current_n_units += 1
        self._apply_unit_def_bonus(giant)
        player.units_under_control[giant.unit_id] = giant

    def calc_movement_target_and_shortest_path(self, unit, target_tile=None, greedy_search=False):
        """Calculate valid movement destinations and shortest paths for unit.

        Transit rules:
          - Hidden tiles, water and mountain tiles block ALL transit.
          - Mountains CAN be stopping destinations; entering one ends movement.
          - Enemy-occupied tiles and their ZoC neighbors block transit (unit must stop before).
          - Friendly-occupied tiles are passable as intermediate nodes.
          - ZoC tiles (adjacent to enemies) can be stopping destinations but not transit nodes.

        Road mechanic: edge weight < 1.0 reduces movement cost; Dijkstra respects weights.
        """
        partial_graph = self.players[unit.player_id].partial_graph

        # Phase 0 — classify tiles
        field_bit      = partial_graph[:, _TILE_TYPE_START]                    # 1 if field, else 0
        mountain_bit   = partial_graph[:, _TILE_TYPE_START + int(TileType.mountain)]
        water_bit      = partial_graph[:, _TILE_TYPE_START + int(TileType.water)]
        road_bit       = partial_graph[:, _ROAD_START]
        bridge_bit     = (water_bit > 0) & (road_bit > 0)                     # water tile with a bridge
        cant_transit   = (field_bit == 0) & ~bridge_bit                        # fields and bridges can be traversed
        cant_stop      = (field_bit == 0) & (mountain_bit == 0) & ~bridge_bit  # bridges are valid stopping tiles
        own_occupied   = (partial_graph[:, OWN_TYPE_SLICE] != 0).any(axis=-1)
        enemy_occupied = (partial_graph[:, OPP_TYPE_SLICE] != 0).any(axis=-1)
        any_occupied   = own_occupied | enemy_occupied
        destination_blocked = cant_stop | any_occupied

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

        transit_blocked = cant_transit | enemy_occupied | zoc_arr
        nodes_to_remove = [self.game_board.int_to_tup[i]
                           for i in np.argwhere(transit_blocked).flatten()]
        unit_loc = self.game_board.int_to_tup[unit.tile.id]
        if unit_loc in nodes_to_remove:
            nodes_to_remove.remove(unit_loc)
        G.remove_nodes_from(nodes_to_remove)

        if unit_loc not in G:
            return False if greedy_search else ({} if target_tile is None else [])

        # Phase 2b — road discount disabled on enemy-controlled edges
        opponent_id_val = (int(unit.player_id) + 1) % 2
        for eu, ev in list(G.edges()):
            if G[eu][ev].get('weight', 1.0) < 1.0:
                for tid in (self.game_board.tup_to_int[eu],
                            self.game_board.tup_to_int[ev]):
                    ctrl = self.game_board.board[tid].cntrl
                    if ctrl is not None and int(ctrl) == opponent_id_val:
                        G[eu][ev]['weight'] = 1.0
                        break

        # Phase 3 — Dijkstra (respects road edge weights)
        lengths, paths = nx.single_source_dijkstra(
            G, unit_loc, cutoff=unit.mvpts, weight='weight')

        if greedy_search:
            if any(not destination_blocked[self.game_board.tup_to_int[n]]
                   for n in lengths if n != unit_loc):
                return True
            # also check ZoC / mountain tiles reachable as one-step stopping points
            for node, cost in lengths.items():
                for nbr in self.game_board.movement_topology_graph.neighbors(node):
                    nbr_id = self.game_board.tup_to_int[nbr]
                    is_zoc_stop      = (nbr_id in zoc_ids)
                    is_mountain_stop = bool(mountain_bit[nbr_id])
                    if (not (is_zoc_stop or is_mountain_stop)) or destination_blocked[nbr_id]:
                        continue
                    base_w = self.game_board.movement_topology_graph \
                                 .get_edge_data(node, nbr).get('weight', 1.0)
                    if base_w < 1.0:
                        for tid in (self.game_board.tup_to_int[node],
                                    self.game_board.tup_to_int[nbr]):
                            ctrl = self.game_board.board[tid].cntrl
                            if ctrl is not None and int(ctrl) == opponent_id_val:
                                base_w = 1.0
                                break
                    if cost + base_w <= unit.mvpts:
                        return True
            return False

        # Extend: ZoC / mountain tiles reachable as stopping destinations (one step from transit nodes)
        for node, cost in list(lengths.items()):
            for nbr in self.game_board.movement_topology_graph.neighbors(node):
                nbr_id = self.game_board.tup_to_int[nbr]
                is_zoc_stop      = (nbr_id in zoc_ids)
                is_mountain_stop = bool(mountain_bit[nbr_id])
                if (not (is_zoc_stop or is_mountain_stop)) or destination_blocked[nbr_id]:
                    continue
                base_w = self.game_board.movement_topology_graph \
                             .get_edge_data(node, nbr).get('weight', 1.0)
                if base_w < 1.0:
                    for tid in (self.game_board.tup_to_int[node],
                                self.game_board.tup_to_int[nbr]):
                        ctrl = self.game_board.board[tid].cntrl
                        if ctrl is not None and int(ctrl) == opponent_id_val:
                            base_w = 1.0
                            break
                step = base_w
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

        ## update defense bonus (important if unit is being pushed by giant creation)
        self._apply_unit_def_bonus(unit)


    def apply_unit_vision(self, unit, path):

        delta_uncovered_tiles = 0
        player_uncovered_tiles = self.players[unit.player_id].uncovered_tile_ids
        delta_uncovered_tiles -= len(player_uncovered_tiles)

        for tile_id in path:
            is_mountain = self.game_board.board[tile_id].tile_type == TileType.mountain
            dist = 2 if is_mountain else unit.vision_range
            player_uncovered_tiles.update(self.tiles_in_range(tile_id, distance=dist))

        delta_uncovered_tiles += len(player_uncovered_tiles)
        return delta_uncovered_tiles


    def attack_retaliate_calc(self, unit, o_unit, splash=False):
        """
        Calculate the resulting hp of both units and returns the result to be handled in the apply_action function
        """
        attackForce = unit.atk_stat * (unit.current_hp / unit.hp)
        defenseForce = o_unit.def_stat * (o_unit.current_hp / o_unit.hp) * o_unit.def_bonus.value 
        totalDamage = attackForce + defenseForce 
        attackResult = col_round((attackForce / totalDamage) * unit.atk_stat * 4.5) 
        defenseResult = col_round((defenseForce / totalDamage) * o_unit.def_stat * 4.5)

        if splash:
            attackResult /= 2

        return attackResult, defenseResult


    def advance_unit_turn_state(self, unit, action, message):
        opponent_id = (self.player_go_id + 1) % 2
        enemy_adjacent = any(
            self.game_board.board[tid].unit is not None
            and self.game_board.board[tid].unit.player_id == opponent_id
            for tid in self.tiles_in_range(unit.tile.id, unit.attack_range)
        )
        action_type = action["type"]

        # Knight re-attack: stays can_hit only if it made a kill AND an enemy is still adjacent
        if (unit.unit_type == UnitType.Knight
                and unit.turn_state in (UnitState.ready, UnitState.can_hit)
                and action_type == ActionTypes.Attack):
            killed = message.get("killed_unit", 0) == 1
            unit.turn_state = UnitState.can_hit if (killed and enemy_adjacent) else UnitState.idle
            return

        for ut, fs, at, ea, new_state in _TURN_STATE_TRANSITIONS:
            if (unit.unit_type == ut
                    and (fs is None or unit.turn_state == fs)
                    and action_type == at
                    and (ea is None or enemy_adjacent == ea)):
                unit.turn_state = new_state
                return
        raise ValueError(
            f"No turn-state transition for {unit.unit_type}, {unit.turn_state}, {action_type}, {enemy_adjacent}"
        )




