## This wraps the entire game folder to be used as the environment for RL

import numpy as np
import matplotlib.pyplot as plt

from game.game import Game
from game.enums import (
    BoardType, CityType, Tribes, ActionTypes, UnitType, UnitState,
    NODE_FEAT_DIM, N_UNIT_TYPES,
    TileType,
    PARTIAL_GRAPH_SWAPS,
)
from game.components.city import _CITY_UPGRADES, _CITY_UPGRADE_COST
from game.components.units import _UNIT_COSTS


class EnvWrapper(object):

    def __init__(self, board_config, player_tribes, max_turns_per_game=999, win_reward=60, dense_reward=False):

        self.Nx, self.Ny = board_config["board_size"][0], board_config["board_size"][1]
        self.n_tiles = self.Nx * self.Ny
        self.n_players = len(player_tribes)
        
        self.game = Game(board_config, player_tribes)
        self.win_reward = win_reward
        self.dense_reward = dense_reward
        self.max_turns_per_game = max_turns_per_game

        self.last_action = None
        self.n_decisions = 0
        self._overlay_ctx = {}


    def reset(self):
        self.game.reset_game()
        self.winner = None
        self.last_action = None
        self.n_decisions = 0
        self._overlay_ctx = {}
        return self._get_obs()


    def step(self, action):
        """
        Return the tuple for RL training in the 'gymnasium' setting
        """
        translated_action = self._translate_action(action)
        self._snapshot_overlay_ctx(translated_action)

        message = self.game.apply_action(translated_action)

        self._finalize_overlay_ctx(translated_action)
        self.last_action = translated_action
        self.n_decisions += 1

        obs = self._get_obs()

        done, reward = self._get_done_and_rewards(message)

        info = {"log": message}

        return obs, reward, done, info


    def _snapshot_overlay_ctx(self, translated):
        """Capture pre-apply state needed by the renderer's action overlay."""
        ctx = {}
        atype = translated["type"]
        player = self.game.players[self.game.player_go_id]
        opponent = self.game.players[(self.game.player_go_id + 1) % 2]

        if atype == ActionTypes.MoveUnit:
            uid = translated["unit_id"]
            unit = player.units_under_control[uid]
            ctx["src_tile_id"] = unit.tile.id
            ctx["dst_tile_id"] = translated["target_id"]
            ctx["path"] = translated.get("path", [])

        elif atype == ActionTypes.Attack:
            uid = translated["unit_id"]
            oid = translated["o_unit_id"]
            attacker = player.units_under_control[uid]
            defender = opponent.units_under_control[oid]
            ctx["attacker_tile_id"] = attacker.tile.id
            ctx["defender_tile_id"] = defender.tile.id
            ctx["attacker_range"] = attacker.attack_range
            ctx["o_unit_id"] = oid

        elif atype == ActionTypes.HealUnit:
            ctx["healed_unit_id"] = translated["unit_id"]

        self._overlay_ctx = ctx

    def _finalize_overlay_ctx(self, translated):
        """Fill in post-apply fields (e.g. whether a defender died in Attack)."""
        if translated["type"] == ActionTypes.Attack:
            opponent = self.game.players[(self.game.player_go_id + 1) % 2]
            oid = self._overlay_ctx.get("o_unit_id")
            self._overlay_ctx["defender_died"] = oid not in opponent.units_under_control

    
    def _get_obs(self):
        """
        Observation dict consumed by the policy network.

        All views are taken from the current player's perspective. The P2
        unit/city/control swap is already applied to `partial_graph` by
        `Player.construct_partial_graph_2players` and is applied here to
        `full_graph` so own/opp slots line up across both views.

        Keys
        ────
        partial_graph      : np.ndarray  (N_tiles, NODE_FEAT_DIM)
            Fog-of-war view; hidden tiles are zeroed.
        full_graph         : np.ndarray  (N_tiles, NODE_FEAT_DIM)
            Un-fogged ground truth in the same channel layout as
            partial_graph; used as the HiddenTileEstimator label.
        units              : list[Unit]   current player's units (stable order)
        cities             : list[City]   current player's cities
        enemy_units        : list[Unit]   only enemies visible to the player
        scalar_state       : np.ndarray  (5,) float32 — own stars, spt,
            own/opp scores, normalised turn count.  Fed into the encoder's global emb.
        uncovered_tile_ids : np.ndarray  (k,) int64 — sorted uncovered tile IDs;
            used by HiddenTileEstimator to mask visible tiles out of the loss.
        opp_partial_graph,
        opp_scalar_state,
        opp_uncovered_tile_ids
            Same three keys but built from the opponent's POV — populated every
            turn so a single obs dict can drive `policy.estimate_hidden_dual()`.
        """
        player   = self.game.players[self.game.player_go_id]
        opponent = self.game.players[(self.game.player_go_id + 1) % 2]

        own_score = player.current_score   if player.current_score   is not None else 0
        opp_score = opponent.current_score if opponent.current_score is not None else 0
        turn_norm = float(self.game.turn) / max(1.0, float(self.max_turns_per_game))

        return {
            "partial_graph":      player.partial_graph,
            "full_graph":         self._full_graph_for_player(player.player_id.value),
            "units":              list(player.units_under_control.values()),
            "cities":             player.cities_under_control,
            "enemy_units":        self._visible_enemy_units(),
            "scalar_state":       np.array([
                float(player.stars),
                float(player.current_stars_per_turn),
                float(own_score),
                float(opp_score),
                turn_norm,
            ], dtype=np.float32),
            "uncovered_tile_ids": np.array(sorted(player.uncovered_tile_ids), dtype=np.int64),

            # — opponent's POV (consumed by HiddenTileEstimator dual-POV path) —
            "opp_partial_graph":      opponent.partial_graph,
            "opp_scalar_state":       np.array([
                float(opponent.stars),
                float(opponent.current_stars_per_turn),
                float(opp_score),
                float(own_score),
                turn_norm,
            ], dtype=np.float32),
            "opp_uncovered_tile_ids": np.array(sorted(opponent.uncovered_tile_ids), dtype=np.int64),
        }


    def _full_graph_for_player(self, pid: int) -> np.ndarray:
        """Un-fogged board graph in `pid`'s channel order (P2 swap applied for pid==1)."""
        full = np.copy(self.game.game_board.board_graph)
        if pid == 1:
            for s0, s1 in PARTIAL_GRAPH_SWAPS:
                tmp = full[:, s0].copy()
                full[:, s0] = full[:, s1]
                full[:, s1] = tmp
        return full
        

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

            elif message["action_type"] == ActionTypes.CreateUnit:
                if message["unit_type"] == UnitType.Rider:
                    reward += 0.5
                elif message["unit_type"] == UnitType.Sword:
                    reward += 1.0
                elif message["unit_type"] == UnitType.Knight:
                    reward += 2.5
                elif message["unit_type"] == UnitType.Catapult:
                    reward += 1.0 # because they are hard to place
                elif message["unit_type"] == UnitType.Defender:
                    reward -= .5 # discourage a passive playstyle

            elif message["action_type"] == ActionTypes.HealUnit:
                if message["heal_amount"] == 4.0:
                    reward += .5
                else:
                    reward -= .5 # healing outside of own city border almost never is worth it..
                 
            elif message["action_type"] == ActionTypes.UpgradeCity:
                pass
                ## TODO: implement the rewards for city upgrades; must be done by a human!

            elif message["action_type"] == ActionTypes.Upgrade2Vet:
                reward += message["hp_diff"] * 0.1 # reward greater healing with the upgrade2vet mechanic

            elif message["action_type"] == ActionTypes.PlaceRoad:
                #reward -= 0.2 # discourage IF used too much!
                pass
                
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

        elif action_type == ActionTypes.HealUnit:
            translated_action["unit_id"] = player_unit_ids[action[1]]

        elif action_type == ActionTypes.UpgradeCity:
            translated_action["city"] = action[1]
            translated_action["choice"] = action[2]

        elif action_type == ActionTypes.PlaceRoad:
            translated_action["tile_id"] = action[1]

        elif action_type == ActionTypes.Upgrade2Vet:
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
            np.zeros((num_actions,)),                             # action types
            np.zeros((num_units_player, self.n_tiles)),           # move unit
            np.zeros((num_units_player, len(visible_enemies))),   # attack (visible enemies only)
            np.zeros((num_cities_player, N_UNIT_TYPES)),          # create unit
            np.zeros((num_units_player,)),                        # capture city
            np.zeros((num_units_player,)),                        # heal unit
            np.zeros((num_cities_player, 2)),                     # upgrade city
            np.zeros((self.n_tiles,)),                            # place road
            np.zeros((num_units_player,)),                        # upgrade unit to veteran
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
            for attacker_pos, can_hit in enumerate(unit_can_hit):
                if can_hit:
                    reachable = self.game.tiles_in_range(player_units[attacker_pos].tile.id, player_units[attacker_pos].attack_range)
                    for def_pos, defender in enumerate(visible_enemies):
                        if defender.tile.id in reachable:
                            valid_actions[2][attacker_pos][def_pos] = 1.0
            if valid_actions[2].sum() > 0:
                valid_actions[0][ActionTypes.Attack] = 1.0

        # create unit
        can_create_unit = np.array([
            1 if (city.current_n_units < city.max_unit_cap
                  and self.game.game_board.board[city.tile_id].unit is None) else 0
            for city in player.cities_under_control
        ])
        if can_create_unit.sum() > 0:
            for city_id, city_valid in enumerate(can_create_unit):
                if city_valid:
                    for unit_type in UnitType:
                        if player.stars >= _UNIT_COSTS[unit_type]:
                            valid_actions[3][city_id, int(unit_type)] = 1.0
            if valid_actions[3].sum() > 0:
                valid_actions[0][ActionTypes.CreateUnit] = 1.0

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
    
        # heal unit
        for pos, unit in enumerate(player_units):
            if unit.turn_state == UnitState.ready and unit.current_hp < unit.hp:
                valid_actions[5][pos] = 1.0
        if valid_actions[5].sum() > 0:
            valid_actions[0][ActionTypes.HealUnit] = 1.0

        # upgrade city
        for city_idx, city in enumerate(player.cities_under_control):
            for choice in range(2):
                next_lvl = _CITY_UPGRADES[city.lvl][choice]
                cost = max(0, _CITY_UPGRADE_COST[next_lvl] - city.pending_discount)
                if player.stars >= cost:
                    valid_actions[6][city_idx, choice] = 1.0
        if valid_actions[6].sum() > 0:
            valid_actions[0][ActionTypes.UpgradeCity] = 1.0

        # place road
        if player.stars >= 5:                                   ## ROAD PRICE
            for tile_id in player.uncovered_tile_ids:
                tile = self.game.game_board.board[tile_id]
                if (not tile.has_road
                        and tile.tile_type == TileType.field
                        and (tile.cntrl is None or tile.cntrl == player.player_id)):
                    valid_actions[7][tile_id] = 1.0
            if valid_actions[7].sum() > 0:
                valid_actions[0][ActionTypes.PlaceRoad] = 1.0

        # upgrade unit to veteran
        for pos, unit in enumerate(player_units):
            if unit.kills >= 3 and not unit.is_vet and not unit.unit_type == UnitType.Giant:
                valid_actions[8][pos] = 1.0
        if valid_actions[8].sum() > 0:
            valid_actions[0][ActionTypes.Upgrade2Vet] = 1.0

        # end turn — always valid
        valid_actions[0][ActionTypes.EndTurn] = 1.0

        return valid_actions
    
        

    def _get_valid_move_locations(self, unit, greedy_search=False):

        if greedy_search:
            return self.game.calc_movement_target_and_shortest_path(unit, greedy_search=True)

        valid_paths = self.game.calc_movement_target_and_shortest_path(unit)
        return [self.game.game_board.tup_to_int[node] for node in valid_paths]


    # ═══════════════════════════════════════════════════════════════════════
    # Rendering — thin orchestrator over env/renderer.BoardRenderer
    # ═══════════════════════════════════════════════════════════════════════

    def _uncovered_set(self, shared_fog):
        if shared_fog:
            uncovered = set()
            for p in range(self.n_players):
                uncovered |= self.game.players[p].uncovered_tile_ids
            return uncovered
        return set(range(self.n_tiles))

    def render(
        self,
        *,
        figsize=None,
        shared_fog=True,
        show_action_overlay=True,
        save_path=None,
        show=True,
        # — trajectory inputs (optional) —
        action=None,
        joint_probs=None,
        traj_actions=None,
        critic_value=None,
        # — hidden-tile estimator overlay (optional) —
        show_hidden=False,
        hidden_estimate=None,
    ):
        """Universal renderer for V2.0.

        See `env/renderer.py` for the drawing implementation. All optional
        kwargs are no-ops when omitted; callers that previously used the
        non-keyword `render(figsize, shared_fog, ...)` signature must move
        to keyword-only — `figsize` etc. are now keyword-only by design.
        """
        from env.renderer import BoardRenderer

        if show_hidden and not (
            isinstance(hidden_estimate, tuple) and len(hidden_estimate) == 2
        ):
            raise ValueError(
                "render(show_hidden=True) requires "
                "`hidden_estimate=(est_a, est_b)` as a tuple of two numpy "
                "arrays of shape (N_tiles, REDUCED_FEAT_DIM). Use "
                "`policy.estimate_hidden_dual(env._get_obs())` to build one."
            )

        renderer = BoardRenderer(self)
        prob_overlay, atype_probs = renderer.compute_prob_overlay(
            action, joint_probs, traj_actions,
        )
        if not (joint_probs is not None and traj_actions is not None):
            atype_probs = None  # info panel suppresses bar chart when no probs given

        fig, axes = renderer.build_figure(figsize=figsize, dual=show_hidden)

        if not show_hidden:
            renderer.draw(
                ax=axes['board'],
                ax_info=axes['info'],
                uncovered=self._uncovered_set(shared_fog),
                prob_overlay=prob_overlay,
                atype_probs=atype_probs,
                show_action_overlay=show_action_overlay,
                critic_value=critic_value,
                info_horizontal=False,
            )
        else:
            est_a = np.asarray(hidden_estimate[0])
            est_b = np.asarray(hidden_estimate[1])
            renderer.draw_dual_pov(
                ax_pov_a=axes['pov_a'],
                ax_pov_b=axes['pov_b'],
                ax_info=axes['info'],
                hidden_estimate_pov_a=est_a,
                hidden_estimate_pov_b=est_b,
                prob_overlay=prob_overlay,
                atype_probs=atype_probs,
                critic_value=critic_value,
                show_action_overlay=show_action_overlay,
            )

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=120)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
