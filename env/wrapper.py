## This wraps the entire game folder to be used as the environment for RL

import numpy as np
from game.game import Game
from game.enums import BoardType, Tribes, ActionTypes, UnitType, UnitState


class EnvWrapper(object):

    def __init__(self, board_config, player_tribes, max_turns_per_game=30, win_reward=500, dense_reward=False):

        self.Nx, self.Ny = board_config["board_size"][0], board_config["board_size"][1]
        self.n_tiles = self.Nx * self.Ny
        self.n_players = len(player_tribes)
        
        self.game = Game(board_config, player_tribes)
        self.win_reward = win_reward
        self.dense_reward = dense_reward

    
    def reset(self):
        self.game.reset_game()
        self.winner = None
        return None

    
    def step(self, action):
        """
        Return the tuple for RL training in the 'gymnasium' setting
        """
        translated_action = self._translate_action(action)

        message = self.game.apply_action(translated_action)

        obs = self._get_obs()

        done, reward = self._get_done_and_rewards(action)

        info = {"log": message}

        return obs, reward, done, info

    
    def _get_obs(self):
        """
        This is the input to the policynetwork
        I assume here I combine the partial graph with the position embedding and everything else I need to give the NN the full information...
        obs is a python dictionary, including not only the board information but also the next player, the next next player etc... it will be converted to torch.tensor to be handled in RL!
        """
        player = self.game.players[self.game.player_go_id]
        opponent = self.game.players[(self.game.player_go_id + 1) % 2]
        
        obs = {
            "partial_graph" : player.partial_graph, ## is it already constructed??
            "units" : player.units_under_control,
            "cities" : player.cities_under_control,
            #"opponent_score" : opponent.score, 
        }

        return obs
        

    def _get_done_and_rewards(self, action):
        """
        Win, if you capture the opponents capital
        Positive Reward for creating units, capturing cities, killing opponent units, clearing fog, ...
        Negative Reward for loosing units, loosing cities, ...
        """
        pass


    def _translate_action(self, action):
        """
        action is a simple list of integer indices, specific to each possible action type, for example:
        action = [0, 2, 55] --> Move (0) unit (2) to tile (55)
        action = [2, 0, 1] ---> Create (2) in city (0) a rider (1)
        action = [4] ---> current player ends his turn
        """
        player = self.game.players[self.game.player_go_id]
        translated_action = {}
        action_type = ActionTypes(action[0])
        translated_action["type"] = action_type
    
        if action_type == ActionTypes.MoveUnit:
            translated_action["unit"] = action[1]
            translated_action["target_id"] = action[2]
            
            ## i have to recalculate the path...
            unit = player.units_under_control[action[1]]
            path_in_ids = [self.game.game_board.tup_to_int[x] for x in self.game.calc_movement_target_and_shortest_path(unit, target_tile=action[2])]
            translated_action["path"] = path_in_ids
    
        elif action_type == ActionTypes.Attack:
            translated_action["unit"] = action[1]
            translated_action["o_unit_index"] = action[2]
    
        elif action_type == ActionTypes.CreateUnit:
            translated_action["city"] = action[1]
            translated_action["unit_type"] = UnitType(action[2])
    
        elif action_type == ActionTypes.CaptureCity:
            translated_action["unit"] = action[1]
    
        elif action_type == ActionTypes.EndTurn:
            pass # do nothing here atm
    
        
        return translated_action
        


    def get_action_mask(self):
        """
        Another big one; gets all possible actions and subactions at any time
        """
        player = self.game.players[self.game.player_go_id]
        opponent = self.game.players[(self.game.player_go_id + 1) % 2]
        num_actions = len(ActionTypes)
        num_units_player = len(player.units_under_control)
        num_units_opponent = len(opponent.units_under_control)
        num_cities_player = len(player.cities_under_control)
    
        valid_actions = [ ## the order is hard-coded...
            np.zeros((num_actions,)),
            np.zeros((num_units_player, N_TILES)), #move unit
            np.zeros((num_units_player, num_units_opponent)), # attack enemy unit
            np.zeros((num_cities_player, N_UNIT_TYPES)), #create unit
            np.zeros((num_units_player,)), # capture village/ seige city
            
        ]
    
        # move unit
        unit_can_move = np.zeros((num_units_player,))
        for unit_id, unit in enumerate(player.units_under_control):
            can_move_bool = (unit.turn_state in (UnitState.ready, UnitState.escaping) and _get_valid_move_locations(self.game, unit, greedy_search=True))
            if can_move_bool:
                unit_can_move[unit_id] = 1
    
        if sum(unit_can_move) > 0: # there is a unit that can move
            valid_actions[0][ActionTypes.MoveUnit] = 1.0
            for unit_id, unit_valid in enumerate(unit_can_move):
                if unit_valid == 1: # the specific unit can move; set the rows with possible targets
                    target_tile_ids = _get_valid_move_locations(self.game, player.units_under_control[unit_id]) # here is a problem for riders in escape...
                    valid_actions[1][unit_id][target_tile_ids] = 1.0 # smart indexing
    
        # attack
        ## PART 1: Which units are eligible to attack (decide, wether action is possible)
        unit_can_hit = np.zeros((num_units_player,))
        for unit_id, unit in enumerate(player.units_under_control):
            surrounding_unit_player_ids = [self.game.game_board.board[id].unit.player_id for id in self.game.tiles_in_range(unit.tile.id, unit.attack_range) \
                                           if self.game.game_board.board[id].unit != None]
        
            can_hit_bool = (unit.turn_state in (UnitState.ready, UnitState.can_hit) and opponent.player_id in surrounding_unit_player_ids)    
            if can_hit_bool:
                unit_can_hit[unit_id] = 1
        
        if num_units_player == 0 or num_units_opponent == 0:
            unit_can_hit = np.zeros(1) # if there is no defender to attack; skip this action
    
        ## PART 2: given a unit can attack, which unit can attack which defenders?
        if sum(unit_can_hit) > 0:
            valid_actions[0][ActionTypes.Attack] = 1.0
            for attacker_id, can_hit in enumerate(unit_can_hit):
                if can_hit == 1: # the specific unit can hit
                    unit = player.units_under_control[attacker_id]
                    reachable_tiles = self.game.tiles_in_range(unit.tile.id, unit.attack_range) # where can the specifics units attack reach
                    for defender_id, defender in enumerate(opponent.units_under_control):
                        if defender.tile.id in reachable_tiles:
                            valid_actions[2][attacker_id][defender_id] = 1.0 # means: defender is in range of attacker
    
        # create unit
        can_create_unit = np.array([1 if ((city.current_n_units < city.max_unit_cap) and city.unit == None) \
                                    else 0 for city in player.cities_under_control])
        if sum(can_create_unit) > 0: #TODO: include unit cost here too
            valid_actions[0][ActionTypes.CreateUnit] = 1.0
            for city_id, city_valid in enumerate(can_create_unit):
                if city_valid == 1: #the city can hold more units
                    valid_actions[3][city_id] = np.ones((N_UNIT_TYPES,)) ## this will be dependent on current stars
                
        # capture city
        can_capture_city = np.zeros((num_units_player,))
        for unit_id, unit in enumerate(player.units_under_control):
            city = unit.tile.city
            city_bool = (city != None and (city.player_id != player.player_id or city.player_id == None) and unit.turn_state == UnitState.ready)
            if city_bool:
                can_capture_city[unit_id] = 1
                
        if sum(can_capture_city) > 0:
            valid_actions[0][ActionTypes.CaptureCity] = 1.0
            for unit_id, unit_valid in enumerate(can_capture_city):
                if unit_valid == 1:
                    valid_actions[4][unit_id] = 1.0
    
        # end turn
        if sum(valid_actions[0]) == 0: # if there is nothing more to do 
            valid_actions[0][ActionTypes.EndTurn] = 1.0
            
        
        return valid_actions
    
        

    def _get_valid_move_locations(self, unit, greedy_search=False):
    
        unit_loc_key = self.game.game_board.int_to_tup[unit.tile.id]
        if greedy_search:
            can_reach = self.game.calc_movement_target_and_shortest_path(unit, greedy_search=greedy_search)
            return can_reach
            
        possible_targets_dict_w_path = self.game.calc_movement_target_and_shortest_path(unit)
        possible_targets_dict_w_path.pop(unit_loc_key)
        
        target_tile_ids = [self.game.game_board.tup_to_int[x] for x in possible_targets_dict_w_path.keys()]
    
        return target_tile_ids



    def render(self, figsize=(4,4)):
        """
        Render the board game state.
        
        Parameters:
        state_graph: numpy array of shape (Nx * Ny, 26)
        Nx: number of tiles in x direction
        Ny: number of tiles in y direction
        figsize: figure size tuple
        """
        Nx, Ny = self.Nx, self.Ny
        state_graph = self.game.game_board.board_graph
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        state_grid = state_graph.reshape(Nx, Ny, 26)
        
        for i in range(Nx):
            for j in range(Ny):
                tile = state_grid[i, j]
                x, y = j, Nx - 1 - i
                
                if tile[2] > 0:
                    rect = Rectangle((x, y), 1, 1, facecolor='#00008B', edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)
                elif tile[1] > 0:
                    rect = Rectangle((x, y), 1, 1, facecolor='#4169E1', edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)
                elif tile[0] > 0:
                    rect = Rectangle((x, y), 1, 1, facecolor='#90EE90', edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)
                else:
                    rect = Rectangle((x, y), 1, 1, facecolor='white', edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)
                
                if tile[3] > 0:
                    shine = Rectangle((x, y), 1, 1, facecolor='#ADD8E6', alpha=0.4, edgecolor='none')
                    ax.add_patch(shine)
                
                if tile[4] > 0:
                    shine = Rectangle((x, y), 1, 1, facecolor='#FFB6C1', alpha=0.4, edgecolor='none')
                    ax.add_patch(shine)
                
                if tile[5] > 0:
                    circle = Circle((x + 0.5, y + 0.5), 0.15, facecolor='#8B4513', edgecolor='black', linewidth=1)
                    ax.add_patch(circle)
                
                if tile[6] > 0:
                    circle = Circle((x + 0.5, y + 0.5), 0.15, facecolor='blue', edgecolor='black', linewidth=1)
                    ax.add_patch(circle)
                
                if tile[8] > 0:
                    circle = Circle((x + 0.5, y + 0.5), 0.15, facecolor='red', edgecolor='black', linewidth=1)
                    ax.add_patch(circle)
        
        for i in range(Nx):
            for j in range(Ny):
                tile = state_grid[i, j]
                x, y = j, Nx - 1 - i
                
                if np.any(tile[10:14] > 0):
                    pawn_points = np.array([
                        [x + 0.5, y + 0.7],
                        [x + 0.4, y + 0.3],
                        [x + 0.6, y + 0.3]
                    ])
                    pawn = Polygon(pawn_points, facecolor='blue', edgecolor='darkblue', linewidth=1.5)
                    ax.add_patch(pawn)
                    head = Circle((x + 0.5, y + 0.75), 0.08, facecolor='blue', edgecolor='darkblue', linewidth=1.5)
                    ax.add_patch(head)
                
                if np.any(tile[14:18] > 0):
                    body = Rectangle((x + 0.35, y + 0.35), 0.3, 0.25, facecolor='blue', edgecolor='darkblue', linewidth=1.5)
                    ax.add_patch(body)
                    head_points = np.array([
                        [x + 0.5, y + 0.75],
                        [x + 0.4, y + 0.6],
                        [x + 0.6, y + 0.6]
                    ])
                    head = Polygon(head_points, facecolor='blue', edgecolor='darkblue', linewidth=1.5)
                    ax.add_patch(head)
                    ear = Circle((x + 0.65, y + 0.7), 0.06, facecolor='blue', edgecolor='darkblue', linewidth=1.5)
                    ax.add_patch(ear)
                
                if np.any(tile[18:22] > 0):
                    pawn_points = np.array([
                        [x + 0.5, y + 0.7],
                        [x + 0.4, y + 0.3],
                        [x + 0.6, y + 0.3]
                    ])
                    pawn = Polygon(pawn_points, facecolor='red', edgecolor='darkred', linewidth=1.5)
                    ax.add_patch(pawn)
                    head = Circle((x + 0.5, y + 0.75), 0.08, facecolor='red', edgecolor='darkred', linewidth=1.5)
                    ax.add_patch(head)
                
                if np.any(tile[22:26] > 0):
                    body = Rectangle((x + 0.35, y + 0.35), 0.3, 0.25, facecolor='red', edgecolor='darkred', linewidth=1.5)
                    ax.add_patch(body)
                    head_points = np.array([
                        [x + 0.5, y + 0.75],
                        [x + 0.4, y + 0.6],
                        [x + 0.6, y + 0.6]
                    ])
                    head = Polygon(head_points, facecolor='red', edgecolor='darkred', linewidth=1.5)
                    ax.add_patch(head)
                    ear = Circle((x + 0.65, y + 0.7), 0.06, facecolor='red', edgecolor='darkred', linewidth=1.5)
                    ax.add_patch(ear)
        
        ax.set_xlim(0, Ny)
        ax.set_ylim(0, Nx)
        ax.set_aspect('equal')
        ax.set_xticks(range(Ny + 1))
        ax.set_yticks(range(Nx + 1))
        ax.grid(True, alpha=0.3)
        ax.set_title('Board Game State', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.show()









