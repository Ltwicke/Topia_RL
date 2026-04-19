## This wraps the entire game folder to be used as the environment for RL

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

from game.game import Game
from game.enums import (
    BoardType, Tribes, ActionTypes, UnitType, UnitState,
    NODE_FEAT_DIM, N_UNIT_TYPES, N_CITY_TYPES,
    _TILE_TYPE_START, _PLAYER_CTRL_START, _CITY_START,
    UNIT_STATE_SLICE, player_type_slice,
    TileType,
)


class EnvWrapper(object):

    def __init__(self, board_config, player_tribes, max_turns_per_game=30, win_reward=60, dense_reward=False):

        self.Nx, self.Ny = board_config["board_size"][0], board_config["board_size"][1]
        self.n_tiles = self.Nx * self.Ny
        self.n_players = len(player_tribes)
        
        self.game = Game(board_config, player_tribes)
        self.win_reward = win_reward
        self.dense_reward = dense_reward
        self.max_turns_per_game = max_turns_per_game

    
    def reset(self):
        self.game.reset_game()
        self.winner = None
        return self._get_obs()

    
    def step(self, action):
        """
        Return the tuple for RL training in the 'gymnasium' setting
        """
        translated_action = self._translate_action(action)

        message = self.game.apply_action(translated_action)

        obs = self._get_obs()

        done, reward = self._get_done_and_rewards(message)

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
            "enemy_units" : opponent.units_under_control, # thats technically cheating a bit, bc network knows how many units the opponent has, even if hidden.
            #"opponent_score" : opponent.score, 
        }

        return obs
        

    def _get_done_and_rewards(self, message):
        """
        Win, if you capture the opponents capital
        Positive Reward for creating units, capturing cities, killing opponent units, clearing fog, ...
        Negative Reward for loosing units, loosing cities, ...
        """
        done = False
        opponent = self.game.players[(self.game.player_go_id + 1) % 2]
        reward = 0.0

        if len(opponent.cities_under_control) == 0: # game terminates, if opponent has no cities anymore.
            done = True
            self.winner = self.game.player_go_id
        if self.game.turn >= self.max_turns_per_game:
            done = True
            # winner stays None

        if self.dense_reward:
            if message["action_type"] == ActionTypes.MoveUnit:
                reward += 0.3 * message["tiles_uncovered"] # for uncovering tiles
            elif message["action_type"] == ActionTypes.Attack:
                if message["killed_unit"] == 1:
                    reward += 1.0 # for killing a unit
                elif message["killed_unit"] == 0:
                    pass
            elif message["action_type"] == ActionTypes.CreateUnit:
                reward += 1.0 # for creating a unit
            elif message["action_type"] == ActionTypes.CaptureCity:
                reward += 5.0
            elif message["action_type"] == ActionTypes.EndTurn:
                reward -= 1.5
                
        if done and self.winner != None:
            reward += self.win_reward # biiig reward for winning
        
        return (done, reward)


    def _player_unit_ids(self) -> list:
        """Ordered list of unit_ids for the current player (stable mask index -> unit_id)."""
        return list(self.game.players[self.game.player_go_id].units_under_control.keys())

    def _visible_enemy_units(self) -> list:
        """Enemy units visible in the current player's partial graph (fixes info leakage)."""
        player = self.game.players[self.game.player_go_id]
        opp_id = (self.game.player_go_id + 1) % 2
        return [
            self.game.game_board.board[tid].unit
            for tid in player.uncovered_tile_ids
            if self.game.game_board.board[tid].unit is not None
            and int(self.game.game_board.board[tid].unit.player_id) == opp_id
        ]

    def _translate_action(self, action):
        """
        action is a simple list of integer indices, specific to each possible action type:
        action = [0, pos, tile_id] --> MoveUnit: unit at position pos in player's unit list, to tile_id
        action = [1, pos, def_pos]  --> Attack: unit at pos attacks visible enemy at def_pos
        action = [2, city_idx, unit_type] --> CreateUnit
        action = [3, pos]           --> CaptureCity
        action = [4]                --> EndTurn
        Returns: translated_action dict with unit_id keys for game.py
        """
        player = self.game.players[self.game.player_go_id]
        player_unit_ids = self._player_unit_ids()
        translated_action = {}
        action_type = ActionTypes(action[0])
        translated_action["type"] = action_type

        if action_type == ActionTypes.MoveUnit:
            uid = player_unit_ids[action[1]]
            translated_action["unit_id"] = uid
            translated_action["target_id"] = action[2]
            unit = player.units_under_control[uid]
            path_in_ids = [self.game.game_board.tup_to_int[x] for x in self.game.calc_movement_target_and_shortest_path(unit, target_tile=action[2])]
            translated_action["path"] = path_in_ids

        elif action_type == ActionTypes.Attack:
            translated_action["unit_id"] = player_unit_ids[action[1]]
            visible_enemies = self._visible_enemy_units()
            translated_action["o_unit_id"] = visible_enemies[action[2]].unit_id

        elif action_type == ActionTypes.CreateUnit:
            translated_action["city"] = action[1]
            translated_action["unit_type"] = UnitType(action[2])

        elif action_type == ActionTypes.CaptureCity:
            translated_action["unit_id"] = player_unit_ids[action[1]]

        elif action_type == ActionTypes.EndTurn:
            pass

        return translated_action
        


    def get_action_mask(self):
        """
        Gets all possible actions and subactions at any time.
        Player units are indexed by position in units_under_control.items() order.
        Attack targets are visible enemy units only (no info leakage about hidden units).
        """
        player = self.game.players[self.game.player_go_id]
        opponent = self.game.players[(self.game.player_go_id + 1) % 2]
        num_actions = len(ActionTypes)
        player_units = list(player.units_under_control.values())  # positional list for mask indexing
        num_units_player = len(player_units)
        visible_enemies = self._visible_enemy_units()  # only enemies visible in partial graph
        num_cities_player = len(player.cities_under_control)

        valid_actions = [
            np.zeros((num_actions,)),
            np.zeros((num_units_player, self.n_tiles)),           # move unit
            np.zeros((num_units_player, len(visible_enemies))),   # attack (visible enemies only)
            np.zeros((num_cities_player, N_UNIT_TYPES)),          # create unit
            np.zeros((num_units_player,)),                        # capture city
        ]

        # move unit
        unit_can_move = np.zeros((num_units_player,))
        for pos, unit in enumerate(player_units):
            if unit.turn_state in (UnitState.ready, UnitState.escaping) and self._get_valid_move_locations(unit, greedy_search=True):
                unit_can_move[pos] = 1

        if unit_can_move.sum() > 0:
            valid_actions[0][ActionTypes.MoveUnit] = 1.0
            for pos, can_move in enumerate(unit_can_move):
                if can_move:
                    target_tile_ids = self._get_valid_move_locations(player_units[pos])
                    valid_actions[1][pos][target_tile_ids] = 1.0

        # attack — only against visible enemies, no info leakage
        unit_can_hit = np.zeros((num_units_player,))
        for pos, unit in enumerate(player_units):
            surrounding_player_ids = [
                self.game.game_board.board[tid].unit.player_id
                for tid in self.game.tiles_in_range(unit.tile.id, unit.attack_range)
                if self.game.game_board.board[tid].unit is not None
            ]
            if unit.turn_state in (UnitState.ready, UnitState.can_hit) and opponent.player_id in surrounding_player_ids:
                unit_can_hit[pos] = 1

        if num_units_player == 0 or len(visible_enemies) == 0:
            unit_can_hit = np.zeros(1)

        if unit_can_hit.sum() > 0:
            valid_actions[0][ActionTypes.Attack] = 1.0
            for attacker_pos, can_hit in enumerate(unit_can_hit):
                if can_hit:
                    reachable = self.game.tiles_in_range(player_units[attacker_pos].tile.id, player_units[attacker_pos].attack_range)
                    for def_pos, defender in enumerate(visible_enemies):
                        if defender.tile.id in reachable:
                            valid_actions[2][attacker_pos][def_pos] = 1.0

        # create unit
        can_create_unit = np.array([
            1 if (city.current_n_units < city.max_unit_cap
                  and self.game.game_board.board[city.tile_id].unit is None) else 0
            for city in player.cities_under_control
        ])
        if can_create_unit.sum() > 0:
            valid_actions[0][ActionTypes.CreateUnit] = 1.0
            for city_id, city_valid in enumerate(can_create_unit):
                if city_valid:
                    valid_actions[3][city_id] = np.ones((N_UNIT_TYPES,))

        # capture city
        can_capture_city = np.zeros((num_units_player,))
        for pos, unit in enumerate(player_units):
            city = unit.tile.city
            if city is not None:
                if (city.player_id != player.player_id or city.player_id is None) and unit.turn_state == UnitState.ready:
                    can_capture_city[pos] = 1

        if can_capture_city.sum() > 0:
            valid_actions[0][ActionTypes.CaptureCity] = 1.0
            for pos, unit_valid in enumerate(can_capture_city):
                if unit_valid:
                    valid_actions[4][pos] = 1.0
    
        # end turn
        #if sum(valid_actions[0]) == 0: # if there is nothing more to do TODO: Change this once the RL actually learns
        valid_actions[0][ActionTypes.EndTurn] = 1.0
            
        
        return valid_actions
    
        

    def _get_valid_move_locations(self, unit, greedy_search=False):

        if greedy_search:
            return self.game.calc_movement_target_and_shortest_path(unit, greedy_search=True)

        valid_paths = self.game.calc_movement_target_and_shortest_path(unit)
        return [self.game.game_board.tup_to_int[node] for node in valid_paths]


    def render(self, figsize=(10, 5), shared_fog=True, critic_value=None, translated_action=None):
        Nx, Ny = self.Nx, self.Ny
        state_graph = self.game.game_board.board_graph
        state_grid  = state_graph.reshape(Nx, Ny, NODE_FEAT_DIM)
    
        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(1, 2, width_ratios=[Ny, 3.5], wspace=0.05)
        ax      = fig.add_subplot(gs[0])
        ax_info = fig.add_subplot(gs[1])
    
        if shared_fog:
            uncovered = set()
            for i in range(self.n_players):
                uncovered |= self.game.players[i].uncovered_tile_ids
        else:
            uncovered = set(range(self.n_tiles))
    
        def tile_center(tile_id):
            row = tile_id // Ny
            col = tile_id % Ny
            return col + 0.5, (Nx - 1 - row) + 0.5
    
        def draw_sword(mx, my, angle, s=0.22):
            bx0 = mx + np.cos(angle) * s * 0.6
            by0 = my + np.sin(angle) * s * 0.6
            bx1 = mx - np.cos(angle) * s * 0.6
            by1 = my - np.sin(angle) * s * 0.6
            ax.plot([bx0, bx1], [by0, by1],
                    color='#DAA520', lw=3, solid_capstyle='round', zorder=8)
            perp = angle + np.pi / 2
            gx0  = mx + np.cos(perp) * s * 0.35
            gy0  = my + np.sin(perp) * s * 0.35
            gx1  = mx - np.cos(perp) * s * 0.35
            gy1  = my - np.sin(perp) * s * 0.35
            ax.plot([gx0, gx1], [gy0, gy1],
                    color='#C0C0C0', lw=2.5, solid_capstyle='round', zorder=8)
            for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
                bx = mx + np.cos(ang) * s * 0.55
                by = my + np.sin(ang) * s * 0.55
                ax.plot([mx, bx], [my, by],
                        color='orange', lw=1, alpha=0.7, zorder=7)
    
        # ── Pass 1: terrain + fog ─────────────────────────────────────────────
        for i in range(Nx):
            for j in range(Ny):
                tile    = state_grid[i, j]
                tile_id = i * Ny + j
                x, y    = j, Nx - 1 - i
    
                if tile_id not in uncovered:
                    ax.add_patch(Rectangle((x, y), 1, 1,
                        facecolor='#707070', edgecolor='#404040', linewidth=0.5))
                    continue
    
                if   tile[_TILE_TYPE_START + int(TileType.deep_water)] > 0: fc = '#00008B'
                elif tile[_TILE_TYPE_START + int(TileType.water)]      > 0: fc = '#4169E1'
                elif tile[_TILE_TYPE_START + int(TileType.field)]      > 0: fc = '#90EE90'
                else:                                                         fc = '#F5F5DC'
                ax.add_patch(Rectangle((x, y), 1, 1,
                    facecolor=fc, edgecolor='black', linewidth=0.5))

                if tile[_PLAYER_CTRL_START] > 0:
                    ax.add_patch(Rectangle((x, y), 1, 1,
                        facecolor='#ADD8E6', alpha=0.4, edgecolor='none'))
                if tile[_PLAYER_CTRL_START + 1] > 0:
                    ax.add_patch(Rectangle((x, y), 1, 1,
                        facecolor='#FFB6C1', alpha=0.4, edgecolor='none'))
                if tile[_CITY_START] > 0:
                    ax.add_patch(Circle((x+0.5, y+0.5), 0.15,
                        facecolor='#8B4513', edgecolor='black', linewidth=1))
                if tile[_CITY_START + 1] > 0:
                    ax.add_patch(Circle((x+0.5, y+0.5), 0.15,
                        facecolor='blue', edgecolor='black', linewidth=1))
                if tile[_CITY_START + 1 + N_CITY_TYPES] > 0:
                    ax.add_patch(Circle((x+0.5, y+0.5), 0.15,
                        facecolor='red', edgecolor='black', linewidth=1))

        # ── Pass 2: units ─────────────────────────────────────────────────────
        _P0_TYPE = player_type_slice(0)
        _P1_TYPE = player_type_slice(1)
        for i in range(Nx):
            for j in range(Ny):
                tile    = state_grid[i, j]
                tile_id = i * Ny + j
                if tile_id not in uncovered:
                    continue
                hp_val = tile[UNIT_STATE_SLICE].max()
                if hp_val <= 0:
                    continue
                x, y = j, Nx - 1 - i

                if tile[_P0_TYPE].any():
                    fc, ec = 'blue', 'darkblue'
                    shape = 'warrior' if np.argmax(tile[_P0_TYPE]) == int(UnitType.Warrior) else 'rider'
                else:
                    fc, ec = 'red', 'darkred'
                    shape = 'warrior' if np.argmax(tile[_P1_TYPE]) == int(UnitType.Warrior) else 'rider'

                if shape == 'warrior':
                    pts = np.array([[x+0.5,y+0.70],[x+0.40,y+0.30],[x+0.60,y+0.30]])
                    ax.add_patch(Polygon(pts, facecolor=fc, edgecolor=ec, linewidth=1.5))
                    ax.add_patch(Circle((x+0.5, y+0.75), 0.08,
                        facecolor=fc, edgecolor=ec, linewidth=1.5))
                else:
                    ax.add_patch(Rectangle((x+0.35, y+0.35), 0.30, 0.25,
                        facecolor=fc, edgecolor=ec, linewidth=1.5))
                    pts = np.array([[x+0.50,y+0.75],[x+0.40,y+0.60],[x+0.60,y+0.60]])
                    ax.add_patch(Polygon(pts, facecolor=fc, edgecolor=ec, linewidth=1.5))
                    ax.add_patch(Circle((x+0.65, y+0.70), 0.06,
                        facecolor=fc, edgecolor=ec, linewidth=1.5))
    
        # ── Pass 3: action overlays ───────────────────────────────────────────
        if translated_action is not None:
            t_act  = translated_action
            atype  = t_act["type"]
            player = self.game.players[self.game.player_go_id]
            pcolor = 'royalblue' if self.game.player_go_id == 0 else 'crimson'
    
            if atype == ActionTypes.MoveUnit:
                path_ids = t_act["path"]
                for tid in path_ids:
                    cx, cy = tile_center(tid)
                    ax.plot(cx, cy, 'o',
                            color=pcolor, markersize=8,
                            markeredgecolor='white', markeredgewidth=1.0,
                            zorder=10)
    
            elif atype == ActionTypes.Attack:

                opponent = self.game.players[(self.game.player_go_id + 1) % 2]
                uid = t_act["unit_id"]
                oid = t_act["o_unit_id"]
                ax0, ay0 = tile_center(player.units_under_control[uid].tile.id)
                ax1, ay1 = tile_center(opponent.units_under_control[oid].tile.id)
                mx, my   = (ax0 + ax1) / 2, (ay0 + ay1) / 2
                angle    = np.arctan2(ay1 - ay0, ax1 - ax0)
                draw_sword(mx, my, angle)
    
            elif atype == ActionTypes.CaptureCity:
                uid = t_act["unit_id"]
                cx, cy = tile_center(player.units_under_control[uid].tile.id)
                ax.add_patch(Rectangle((cx - 0.22, cy + 0.02), 0.44, 0.12,
                              facecolor='gold', edgecolor='darkorange',
                              linewidth=1.5, zorder=8))
                for px, ph in [(cx - 0.15, 0.18), (cx, 0.23), (cx + 0.15, 0.18)]:
                    ax.add_patch(Polygon(
                        [[px - 0.06, cy + 0.12],
                         [px,         cy + 0.12 + ph],
                         [px + 0.06, cy + 0.12]],
                        closed=True, facecolor='gold',
                        edgecolor='darkorange', linewidth=1.5, zorder=8
                    ))
    
        # ── Board axis ────────────────────────────────────────────────────────
        ax.set_xlim(0, Ny); ax.set_ylim(0, Nx)
        ax.set_aspect('equal')
        ax.set_xticks(range(Ny + 1)); ax.set_yticks(range(Nx + 1))
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_title('Board State', fontsize=11, fontweight='bold', pad=4)
    
        # ── Info panel ────────────────────────────────────────────────────────
        ax_info.axis('off')
        ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
    
        pid       = self.game.player_go_id
        pcolor_hx = '#1E6FD9' if pid == 0 else '#D92B1E'
    
        badge = FancyBboxPatch((0.05, 0.80), 0.90, 0.13,
                                boxstyle="round,pad=0.02",
                                facecolor=pcolor_hx, edgecolor='none', alpha=0.85)
        ax_info.add_patch(badge)
        ax_info.text(0.50, 0.865, f"▶  Player {pid}'s Turn",
                     ha='center', va='center', fontsize=11, fontweight='bold',
                     color='white', transform=ax_info.transAxes)
    
        def _info_row(y, label, value, vc='#222222'):
            ax_info.text(0.08, y, label, ha='left',  va='top', fontsize=9,
                         color='#555555', transform=ax_info.transAxes)
            ax_info.text(0.92, y, str(value), ha='right', va='top', fontsize=9,
                         fontweight='bold', color=vc, transform=ax_info.transAxes)
    
        _info_row(0.74, "Turn", self.game.turn)
    
        if critic_value is not None:
            v  = critic_value.item() if hasattr(critic_value, 'item') else float(critic_value)
            vc = '#1a7a1a' if v >= 0 else '#cc2200'
            _info_row(0.64, "Critic V̂", f"{v:+.3f}", vc=vc)
    
        if translated_action is not None:
            atype_str = translated_action["type"].name
            _info_row(0.54, "Last action", atype_str, vc=pcolor_hx)
    
        ax_info.plot([0.05, 0.95], [0.38, 0.38],
                     color='#CCCCCC', linewidth=0.8,
                     transform=ax_info.transAxes)
    
        plt.tight_layout()
        plt.show()




    def render_with_trajs(self, figsize=(10, 5), shared_fog=True, critic_value=None,
                          action=None, joint_probs=None, traj_actions=None):
        Nx, Ny = self.Nx, self.Ny
        state_graph = self.game.game_board.board_graph
        state_grid  = state_graph.reshape(Nx, Ny, NODE_FEAT_DIM)
    
        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(1, 2, width_ratios=[Ny, 3.5], wspace=0.05)
        ax      = fig.add_subplot(gs[0])
        ax_info = fig.add_subplot(gs[1])
    
        # ── Fog of war ────────────────────────────────────────────────────────
        if shared_fog:
            uncovered = set()
            for i in range(self.n_players):
                uncovered |= self.game.players[i].uncovered_tile_ids
        else:
            uncovered = set(range(self.n_tiles))
    
        # ── Helpers ───────────────────────────────────────────────────────────
        def tile_center(tile_id):
            row = tile_id // Ny
            col = tile_id % Ny
            return col + 0.5, (Nx - 1 - row) + 0.5
    
        def draw_sword(mx, my, angle, s=0.22):
            bx0 = mx + np.cos(angle) * s * 0.6
            by0 = my + np.sin(angle) * s * 0.6
            bx1 = mx - np.cos(angle) * s * 0.6
            by1 = my - np.sin(angle) * s * 0.6
            ax.plot([bx0, bx1], [by0, by1],
                    color='#DAA520', lw=3, solid_capstyle='round', zorder=8)
            perp = angle + np.pi / 2
            gx0 = mx + np.cos(perp) * s * 0.35
            gy0 = my + np.sin(perp) * s * 0.35
            gx1 = mx - np.cos(perp) * s * 0.35
            gy1 = my - np.sin(perp) * s * 0.35
            ax.plot([gx0, gx1], [gy0, gy1],
                    color='#C0C0C0', lw=2.5, solid_capstyle='round', zorder=8)
            for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
                bx = mx + np.cos(ang) * s * 0.55
                by = my + np.sin(ang) * s * 0.55
                ax.plot([mx, bx], [my, by],
                        color='orange', lw=1, alpha=0.7, zorder=7)
    
        # ── Pre-compute trajectory overlays ───────────────────────────────────
        prob_overlay: dict[int, float] = {}
        pcolor_rgb = (0.25, 0.41, 0.88) if self.game.player_go_id == 0 else (0.85, 0.15, 0.15)
        atype_probs: dict[str, float] = {at.name: 0.0 for at in ActionTypes}
    
        if joint_probs is not None and traj_actions is not None:
            probs_np = joint_probs.detach().cpu().numpy()
    
            for traj, p in zip(traj_actions, probs_np):
                atype_probs[ActionTypes(traj[0]).name] += float(p)
    
            if action is not None:
                sampled_atype = ActionTypes(action[0])
    
                if sampled_atype == ActionTypes.MoveUnit:
                    sampled_uid = action[1]
                    move_mask   = np.array([
                        (ActionTypes(t[0]) == ActionTypes.MoveUnit and t[1] == sampled_uid)
                        for t in traj_actions
                    ], dtype=bool)
                    if move_mask.any():
                        move_probs   = probs_np[move_mask]
                        move_targets = [traj_actions[i][2] for i in np.where(move_mask)[0]]
                        total = move_probs.sum()
                        if total > 0:
                            move_probs = move_probs / total
                        for tile_id, alpha in zip(move_targets, move_probs):
                            prob_overlay[tile_id] = float(alpha)
    
        # ── Pass 1: terrain + fog + probability overlay ───────────────────────
        for i in range(Nx):
            for j in range(Ny):
                tile    = state_grid[i, j]
                tile_id = i * Ny + j
                x, y    = j, Nx - 1 - i
    
                if tile_id not in uncovered:
                    ax.add_patch(Rectangle((x, y), 1, 1,
                        facecolor='#707070', edgecolor='#404040', linewidth=0.5))
                    continue
    
                if   tile[_TILE_TYPE_START + int(TileType.deep_water)] > 0: fc = '#00008B'
                elif tile[_TILE_TYPE_START + int(TileType.water)]      > 0: fc = '#4169E1'
                elif tile[_TILE_TYPE_START + int(TileType.field)]      > 0: fc = '#90EE90'
                else:                                                         fc = '#F5F5DC'
                ax.add_patch(Rectangle((x, y), 1, 1,
                    facecolor=fc, edgecolor='black', linewidth=0.5))

                if tile[_PLAYER_CTRL_START] > 0:
                    ax.add_patch(Rectangle((x, y), 1, 1,
                        facecolor='#ADD8E6', alpha=0.4, edgecolor='none'))
                if tile[_PLAYER_CTRL_START + 1] > 0:
                    ax.add_patch(Rectangle((x, y), 1, 1,
                        facecolor='#FFB6C1', alpha=0.4, edgecolor='none'))
                if tile[_CITY_START] > 0:
                    ax.add_patch(Circle((x+0.5, y+0.5), 0.15,
                        facecolor='#8B4513', edgecolor='black', linewidth=1))
                if tile[_CITY_START + 1] > 0:
                    ax.add_patch(Circle((x+0.5, y+0.5), 0.15,
                        facecolor='blue', edgecolor='black', linewidth=1))
                if tile[_CITY_START + 1 + N_CITY_TYPES] > 0:
                    ax.add_patch(Circle((x+0.5, y+0.5), 0.15,
                        facecolor='red', edgecolor='black', linewidth=1))

                if tile_id in prob_overlay:
                    alpha = np.clip(prob_overlay[tile_id], 0.0, 0.92)
                    ax.add_patch(Rectangle((x, y), 1, 1,
                        facecolor=pcolor_rgb, alpha=alpha,
                        edgecolor='none', zorder=3))

        # ── Pass 2: units ─────────────────────────────────────────────────────
        _P0_TYPE = player_type_slice(0)
        _P1_TYPE = player_type_slice(1)
        for i in range(Nx):
            for j in range(Ny):
                tile    = state_grid[i, j]
                tile_id = i * Ny + j
                if tile_id not in uncovered:
                    continue
                hp_val = tile[UNIT_STATE_SLICE].max()
                if hp_val <= 0:
                    continue
                x, y = j, Nx - 1 - i

                if tile[_P0_TYPE].any():
                    fc, ec = 'blue', 'darkblue'
                    shape = 'warrior' if np.argmax(tile[_P0_TYPE]) == int(UnitType.Warrior) else 'rider'
                else:
                    fc, ec = 'red', 'darkred'
                    shape = 'warrior' if np.argmax(tile[_P1_TYPE]) == int(UnitType.Warrior) else 'rider'

                if shape == 'warrior':
                    pts = np.array([[x+0.5, y+0.70],
                                    [x+0.40, y+0.30],
                                    [x+0.60, y+0.30]])
                    ax.add_patch(Polygon(pts, facecolor=fc, edgecolor=ec,
                                         linewidth=1.5, zorder=4))
                    ax.add_patch(Circle((x+0.5, y+0.75), 0.08, facecolor=fc,
                                        edgecolor=ec, linewidth=1.5, zorder=4))
                else:
                    ax.add_patch(Rectangle((x+0.35, y+0.35), 0.30, 0.25,
                                           facecolor=fc, edgecolor=ec,
                                           linewidth=1.5, zorder=4))
                    pts = np.array([[x+0.50, y+0.75],
                                    [x+0.40, y+0.60],
                                    [x+0.60, y+0.60]])
                    ax.add_patch(Polygon(pts, facecolor=fc, edgecolor=ec,
                                         linewidth=1.5, zorder=4))
                    ax.add_patch(Circle((x+0.65, y+0.70), 0.06, facecolor=fc,
                                        edgecolor=ec, linewidth=1.5, zorder=4))

                # HP label at top-right of tile
                ax.text(x + 0.92, y + 0.82, f"{hp_val:.1f}",
                        ha='right', va='top', fontsize=5.5,
                        fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.1', fc=ec,
                                  ec='none', alpha=0.7),
                        zorder=6)
    
        # ── Pass 3: action overlays ───────────────────────────────────────────
        if action is not None:
            t_act  = self._translate_action(action)
            atype  = t_act["type"]
            player = self.game.players[self.game.player_go_id]
            pcolor = 'royalblue' if self.game.player_go_id == 0 else 'crimson'
    
            if atype == ActionTypes.MoveUnit:
                path_ids = t_act.get("path", [])
                for tid in path_ids:
                    cx, cy = tile_center(tid)
                    ax.plot(cx, cy, 'o',
                            color=pcolor, markersize=8,
                            markeredgecolor='white', markeredgewidth=1.0,
                            zorder=10)
            
            elif atype == ActionTypes.Attack:
                pass
                #opponent = self.game.players[(self.game.player_go_id + 1) % 2]
                #uid = t_act["unit_id"]
                #oid = t_act["o_unit_index"]
                #ax0, ay0 = tile_center(player.units_under_control[uid].tile.id)
                #ax1, ay1 = tile_center(opponent.units_under_control[oid].tile.id)
                #mx, my   = (ax0 + ax1) / 2, (ay0 + ay1) / 2
                #angle    = np.arctan2(ay1 - ay0, ax1 - ax0)
                #draw_sword(mx, my, angle)
            
    
            elif atype == ActionTypes.CaptureCity:
                uid = t_act["unit_id"]
                cx, cy = tile_center(player.units_under_control[uid].tile.id)
                ax.add_patch(Rectangle((cx - 0.22, cy + 0.02), 0.44, 0.12,
                              facecolor='gold', edgecolor='darkorange',
                              linewidth=1.5, zorder=8))
                for px, ph in [(cx - 0.15, 0.18), (cx, 0.23), (cx + 0.15, 0.18)]:
                    ax.add_patch(Polygon(
                        [[px - 0.06, cy + 0.12],
                         [px,         cy + 0.12 + ph],
                         [px + 0.06, cy + 0.12]],
                        closed=True, facecolor='gold',
                        edgecolor='darkorange', linewidth=1.5, zorder=8
                    ))
    
        # ── Board axis ────────────────────────────────────────────────────────
        ax.set_xlim(0, Ny); ax.set_ylim(0, Nx)
        ax.set_aspect('equal')
        ax.set_xticks(range(Ny + 1)); ax.set_yticks(range(Nx + 1))
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_title('Board State', fontsize=11, fontweight='bold', pad=4)
    
        # ── Info panel ────────────────────────────────────────────────────────
        ax_info.axis('off')
        ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
    
        pid       = self.game.player_go_id
        pcolor_hx = '#1E6FD9' if pid == 0 else '#D92B1E'
    
        badge = FancyBboxPatch((0.05, 0.87), 0.90, 0.11,
                                boxstyle="round,pad=0.02",
                                facecolor=pcolor_hx, edgecolor='none', alpha=0.85)
        ax_info.add_patch(badge)
        ax_info.text(0.50, 0.925, f"▶  Player {pid}'s Turn",
                     ha='center', va='center', fontsize=11, fontweight='bold',
                     color='white', transform=ax_info.transAxes)
    
        def _info_row(y, label, value, vc='#222222'):
            ax_info.text(0.08, y, label, ha='left',  va='top', fontsize=9,
                         color='#555555', transform=ax_info.transAxes)
            ax_info.text(0.92, y, str(value), ha='right', va='top', fontsize=9,
                         fontweight='bold', color=vc, transform=ax_info.transAxes)
    
        _info_row(0.82, "Turn", self.game.turn)
    
        if critic_value is not None:
            v  = critic_value.item() if hasattr(critic_value, 'item') else float(critic_value)
            vc = '#1a7a1a' if v >= 0 else '#cc2200'
            _info_row(0.74, "Critic V̂", f"{v:+.3f}", vc=vc)
    
        if action is not None:
            try:
                atype_str = ActionTypes(action[0]).name
            except Exception:
                atype_str = str(action[0])
            _info_row(0.66, "Last action", atype_str, vc=pcolor_hx)
    
        # ── Action-type probability breakdown ─────────────────────────────────
        ax_info.plot([0.05, 0.95], [0.60, 0.60],
                     color='#CCCCCC', linewidth=0.8,
                     transform=ax_info.transAxes)
        ax_info.text(0.50, 0.57, "Action Probabilities",
                     ha='center', va='top', fontsize=8.5, fontweight='bold',
                     color='#333333', transform=ax_info.transAxes)
    
        row_y = 0.50
        for at in ActionTypes:
            p = atype_probs.get(at.name, 0.0)
            bar_w = p * 0.60
            ax_info.add_patch(FancyBboxPatch(
                (0.08, row_y - 0.030), 0.60, 0.028,
                boxstyle="round,pad=0.002",
                facecolor='#EEEEEE', edgecolor='none',
                transform=ax_info.transAxes, clip_on=False, zorder=2
            ))
            if bar_w > 0:
                ax_info.add_patch(FancyBboxPatch(
                    (0.08, row_y - 0.030), bar_w, 0.028,
                    boxstyle="round,pad=0.002",
                    facecolor=pcolor_hx, edgecolor='none', alpha=0.75,
                    transform=ax_info.transAxes, clip_on=False, zorder=3
                ))
            ax_info.text(0.08, row_y - 0.001, at.name, ha='left', va='center',
                         fontsize=7, color='#333333',
                         transform=ax_info.transAxes, zorder=4)
            ax_info.text(0.70, row_y - 0.001, f"{p*100:.1f}%", ha='left', va='center',
                         fontsize=7, fontweight='bold', color='#333333',
                         transform=ax_info.transAxes, zorder=4)
            row_y -= 0.072
    
        plt.tight_layout()
        plt.show()




    