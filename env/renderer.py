"""Visualization layer for EnvWrapper.

The `BoardRenderer` class fully owns the matplotlib drawing surface; it is
constructed per `EnvWrapper.render()` call and reads all state directly off
the live `EnvWrapper`. No persistent renderer state lives between calls.

Rendering modes
---------------
- **Single POV** (`show_hidden=False`): one board axis (current player's
  fog view) + a vertical info panel on the right.
- **Dual POV** (`show_hidden=True`): two boards side-by-side (P0 POV +
  P1 POV) + a horizontal info banner on top. Hidden tiles in each POV are
  drawn from a per-tile estimator distribution.

Trajectory and per-action-type probabilities are overlaid when the caller
passes `action`, `joint_probs`, `traj_actions`. The hidden-tile estimator
overlay is opt-in via `show_hidden=True` + `hidden_estimate=...`.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

from game.enums import (
    ActionTypes, UnitType, UnitState,
    NODE_FEAT_DIM, N_CITY_TYPES, N_UNIT_TYPES,
    _TILE_TYPE_START, _PLAYER_CTRL_START, _CITY_START, _ROAD_START,
    UNIT_STATE_SLICE, TILE_TYPE_SLICE, ROAD_SLICE, PLAYER_CTRL_SLICE,
    CITY_SLICE, OWN_TYPE_SLICE, OPP_TYPE_SLICE,
    TileType, DefenseBonus,
    REDUCED_TILE_TYPE_SLICE, REDUCED_ROAD_SLICE, REDUCED_OPP_CTRL_SLICE,
    REDUCED_CITY_SLICE, REDUCED_OPP_UNIT_SLICE,
    MAX_CITY_LEVEL_HIDDEN,
)


# ── Palette and glyph map ────────────────────────────────────────────────
_P_COLORS     = ('#1f77b4', '#d62728')                      # P0 blue, P1 red
_P_VET        = ('#0d3c66', '#6a1315')                      # darker variants
_P_COLORS_RGB = ((0.25, 0.41, 0.88), (0.85, 0.15, 0.15))   # trajectory tints
_HIDDEN_SHADOW = '#A0A0A0'

_UNIT_GLYPH = {
    UnitType.Warrior:  ('♟', None),     # pawn
    UnitType.Rider:    ('♞', None),     # knight
    UnitType.Archer:   ('♝', None),     # bishop
    UnitType.Knight:   ('♛', '♞'), # queen over chess-knight
    UnitType.Catapult: ('♚', None),     # king
    UnitType.Giant:    ('♜', None),     # rook
    UnitType.Sword:    ('♟', 'sword'),  # pawn + sword overlay
    UnitType.Defender: ('♟', 'shield'), # pawn + shield overlay
}

_TERRAIN_PALETTE = {
    int(TileType.deep_water): '#00008B',
    int(TileType.water):      '#4169E1',
    int(TileType.field):      '#90EE90',
    int(TileType.mountain):   '#8A8A80',
}
_TERRAIN_FALLBACK = '#F5F5DC'


def _terrain_color(tile_row):
    """Pick the terrain color from a single (NODE_FEAT_DIM,) feature row."""
    for tt_idx, hexcolor in _TERRAIN_PALETTE.items():
        if tile_row[_TILE_TYPE_START + tt_idx] > 0:
            return hexcolor
    return _TERRAIN_FALLBACK


class BoardRenderer:
    """Per-call matplotlib renderer driven by a live `EnvWrapper`.

    Construct, draw, throw away. `BoardRenderer` is stateless across calls.
    """

    def __init__(self, env):
        self.env          = env
        self.Nx           = env.Nx
        self.Ny           = env.Ny
        self.game         = env.game
        self.n_players    = env.n_players
        self.n_decisions  = env.n_decisions
        self.last_action  = env.last_action
        self._overlay_ctx = env._overlay_ctx

    # ── Geometry helper ─────────────────────────────────────────────────
    def tile_center(self, tile_id):
        row = tile_id // self.Ny
        col = tile_id %  self.Ny
        return col + 0.5, (self.Nx - 1 - row) + 0.5

    # ── Figure construction ─────────────────────────────────────────────
    def build_figure(self, *, figsize, dual):
        if dual:
            fs = figsize if figsize is not None else (18, 8)
            fig = plt.figure(figsize=fs)
            gs  = fig.add_gridspec(
                2, 2,
                height_ratios=[1.0, 6.0],
                width_ratios=[self.Ny, self.Ny],
                hspace=0.08, wspace=0.10,
            )
            ax_info  = fig.add_subplot(gs[0, :])
            ax_pov_a = fig.add_subplot(gs[1, 0])
            ax_pov_b = fig.add_subplot(gs[1, 1])
            return fig, {'info': ax_info, 'pov_a': ax_pov_a, 'pov_b': ax_pov_b}
        else:
            fs = figsize if figsize is not None else (13, 7)
            fig = plt.figure(figsize=fs)
            gs  = fig.add_gridspec(1, 2, width_ratios=[self.Ny, 4.2], wspace=0.08)
            ax      = fig.add_subplot(gs[0])
            ax_info = fig.add_subplot(gs[1])
            return fig, {'board': ax, 'info': ax_info}

    # ── Public single-POV entry ─────────────────────────────────────────
    def draw(
        self, *, ax, ax_info, uncovered,
        prob_overlay=None, atype_probs=None,
        hidden_estimate=None, hidden_tile_ids=None,
        show_action_overlay=True,
        info_horizontal=False, critic_value=None,
        pov_pid=None,
        title='Board State',
    ):
        if pov_pid is None:
            pov_pid = self.game.player_go_id
        rng = np.random.default_rng(self.n_decisions)
        self._render_board(
            ax, uncovered,
            prob_overlay=prob_overlay,
            hidden_estimate=hidden_estimate,
            hidden_tile_ids=hidden_tile_ids,
            rng=rng,
            pov_pid=pov_pid,
        )
        self._render_control_perimeter(ax, uncovered)
        self._render_units(ax, uncovered)
        if show_action_overlay and self.last_action is not None:
            self._render_action_overlay(ax)
        # Board axis frame
        ax.set_xlim(0, self.Ny); ax.set_ylim(0, self.Nx)
        ax.set_aspect('equal')
        ax.set_xticks(range(self.Ny + 1)); ax.set_yticks(range(self.Nx + 1))
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=4)

        if ax_info is not None:
            self._render_info_panel(
                ax_info,
                atype_probs=atype_probs,
                critic_value=critic_value,
                horizontal=info_horizontal,
            )

    # ── Public dual-POV entry ───────────────────────────────────────────
    def draw_dual_pov(
        self, *, ax_pov_a, ax_pov_b, ax_info,
        hidden_estimate_pov_a, hidden_estimate_pov_b,
        prob_overlay=None, atype_probs=None,
        critic_value=None, show_action_overlay=True,
    ):
        # Fixed layout: P0 always left, P1 always right — independent of whose
        # turn it currently is. The two estimates come in (current_pov, opp_pov)
        # ordering from `policy.estimate_hidden_dual()`; remap them to (P0, P1).
        go_id = self.game.player_go_id
        if go_id == 0:
            est_p0, est_p1 = hidden_estimate_pov_a, hidden_estimate_pov_b
        else:
            est_p0, est_p1 = hidden_estimate_pov_b, hidden_estimate_pov_a

        unc_p0 = set(self.game.players[0].uncovered_tile_ids)
        unc_p1 = set(self.game.players[1].uncovered_tile_ids)
        hidden_p0 = [t for t in range(self.Nx * self.Ny) if t not in unc_p0]
        hidden_p1 = [t for t in range(self.Nx * self.Ny) if t not in unc_p1]

        # Left: P0's POV — opponent on hidden tiles is P1 (red).
        self.draw(
            ax=ax_pov_a, ax_info=None,
            uncovered=unc_p0,
            prob_overlay=prob_overlay if go_id == 0 else None,
            hidden_estimate=est_p0,
            hidden_tile_ids=hidden_p0,
            show_action_overlay=show_action_overlay and go_id == 0,
            pov_pid=0,
            title="Player 0 POV",
        )
        # Right: P1's POV — opponent on hidden tiles is P0 (blue).
        self.draw(
            ax=ax_pov_b, ax_info=None,
            uncovered=unc_p1,
            prob_overlay=prob_overlay if go_id == 1 else None,
            hidden_estimate=est_p1,
            hidden_tile_ids=hidden_p1,
            show_action_overlay=show_action_overlay and go_id == 1,
            pov_pid=1,
            title="Player 1 POV",
        )
        self._render_info_panel(
            ax_info,
            atype_probs=atype_probs,
            critic_value=critic_value,
            horizontal=True,
        )

    # ── Pass 1: terrain / fog / road / city / hidden estimator ─────────
    def _render_board(
        self, ax, uncovered, *,
        prob_overlay=None, hidden_estimate=None, hidden_tile_ids=None, rng=None,
        pov_pid=None,
    ):
        if pov_pid is None:
            pov_pid = self.game.player_go_id
        Nx, Ny = self.Nx, self.Ny
        state_grid = self.game.game_board.board_graph.reshape(Nx, Ny, NODE_FEAT_DIM)
        board = self.game.game_board.board

        hidden_set = set(hidden_tile_ids) if hidden_tile_ids is not None else None

        for i in range(Nx):
            for j in range(Ny):
                tile_row = state_grid[i, j]
                tile_id  = i * Ny + j
                x, y     = j, Nx - 1 - i

                # Hidden tile branch
                if tile_id not in uncovered:
                    if hidden_estimate is not None:
                        self._draw_hidden_tile(
                            ax, x, y, hidden_estimate[tile_id], rng,
                            pov_pid=pov_pid,
                        )
                    else:
                        ax.add_patch(Rectangle(
                            (x, y), 1, 1,
                            facecolor='#707070', edgecolor='#404040', linewidth=0.5,
                        ))
                    continue

                # Visible tile: terrain
                fc = _terrain_color(tile_row)
                ax.add_patch(Rectangle(
                    (x, y), 1, 1, facecolor=fc,
                    edgecolor='black', linewidth=0.5,
                ))

                # Road cross
                if tile_row[_ROAD_START] > 0:
                    ax.plot([x + 0.02, x + 0.98], [y + 0.02, y + 0.98],
                            color='#8B4513', lw=1.1, zorder=2,
                            solid_capstyle='round', alpha=0.25)
                    ax.plot([x + 0.02, x + 0.98], [y + 0.98, y + 0.02],
                            color='#8B4513', lw=1.1, zorder=2,
                            solid_capstyle='round', alpha=0.25)

                # City marker (live Tile.city)
                city_obj = board[tile_id].city
                if city_obj is not None:
                    if city_obj.player_id is None:
                        ax.add_patch(Circle((x + 0.5, y + 0.22), 0.09,
                            facecolor='#8B4513', edgecolor='black',
                            linewidth=0.8, zorder=3))
                    else:
                        c = _P_COLORS[int(city_obj.player_id)]
                        ax.add_patch(Rectangle((x + 0.38, y + 0.10), 0.24, 0.14,
                            facecolor=c, edgecolor='black',
                            linewidth=0.8, zorder=3))
                        ax.text(x + 0.50, y + 0.17,
                                str(city_obj.times_upgraded),
                                ha='center', va='center',
                                fontsize=6, color='white', zorder=4)

                # Trajectory move-target tint
                if prob_overlay and tile_id in prob_overlay:
                    alpha = float(np.clip(prob_overlay[tile_id], 0.0, 0.92))
                    pcolor_rgb = _P_COLORS_RGB[self.game.player_go_id]
                    ax.add_patch(Rectangle(
                        (x, y), 1, 1,
                        facecolor=pcolor_rgb, alpha=alpha,
                        edgecolor='none', zorder=3,
                    ))

    # ── Hidden-tile estimator decode (single tile) ──────────────────────
    def _draw_hidden_tile(self, ax, x, y, est_row, rng, *, pov_pid=None):
        """Draw a single hidden tile from its estimator distribution row.

        `est_row` is a 1-D numpy array of length REDUCED_FEAT_DIM, already
        through HiddenTileEstimator.predict_proba — i.e. softmax groups
        are softmaxed and the road / opp_ctrl bits are sigmoid'd.

        Reduced layout decoded here:
            - tile_type : softmax over TileType                  → terrain color
            - road      : sigmoid bit                            → road cross alpha
            - opp_ctrl  : sigmoid bit (0 = unowned/own, 1 = opp) → opponent shadow
            - city      : softmax {None, Village, L1..L_cap}     → city marker
            - opp_unit  : softmax {None, UnitType.*}             → unit silhouette

        `pov_pid` identifies which player's POV is being drawn — the opponent
        on hidden tiles is the *other* player. Defaults to the current player
        in the single-POV path.
        """
        cx, cy = x + 0.5, y + 0.5
        if pov_pid is None:
            pov_pid = self.game.player_go_id
        opp_pid = (pov_pid + 1) % self.n_players

        # Layer 0 — beige background so partial alphas always blend onto a known color.
        ax.add_patch(Rectangle((x, y), 1, 1,
                               facecolor=_TERRAIN_FALLBACK,
                               edgecolor='#404040', linewidth=0.5))

        # Layer 1 — terrain color from argmax of REDUCED_TILE_TYPE_SLICE
        tt_block = np.asarray(est_row[REDUCED_TILE_TYPE_SLICE])
        if tt_block.size > 0 and tt_block.sum() > 0:
            tt_idx   = int(np.argmax(tt_block))
            tt_alpha = float(np.clip(tt_block[tt_idx], 0.0, 1.0))
            tt_color = _TERRAIN_PALETTE.get(tt_idx, _TERRAIN_FALLBACK)
            if tt_alpha > 0.02:
                ax.add_patch(Rectangle(
                    (x, y), 1, 1,
                    facecolor=tt_color, alpha=tt_alpha,
                    edgecolor='none', zorder=1,
                ))

        # Layer 2 — opponent control shadow (single sigmoid bit).
        opp_ctrl_p = float(np.clip(est_row[REDUCED_OPP_CTRL_SLICE.start], 0.0, 1.0))
        ctrl_alpha = opp_ctrl_p * 0.4
        if ctrl_alpha > 0.02:
            ax.add_patch(Rectangle(
                (x, y), 1, 1,
                facecolor=_P_COLORS[opp_pid], alpha=ctrl_alpha,
                edgecolor='none', zorder=1.5,
            ))

        # Layer 3 — road cross at alpha = sigmoid'd road bit
        road_alpha = float(np.clip(est_row[REDUCED_ROAD_SLICE.start], 0.0, 1.0))
        if road_alpha > 0.05:
            ax.plot([x + 0.02, x + 0.98], [y + 0.02, y + 0.98],
                    color='#8B4513', lw=1.1, zorder=2,
                    solid_capstyle='round', alpha=road_alpha * 0.6)
            ax.plot([x + 0.02, x + 0.98], [y + 0.98, y + 0.02],
                    color='#8B4513', lw=1.1, zorder=2,
                    solid_capstyle='round', alpha=road_alpha * 0.6)

        # Layer 4 — city marker.  argmax over the reduced city block:
        #     0     → None       (no marker)
        #     1     → Village    (brown circle)
        #     2..K  → opponent city of level (idx - 1), labelled (or "8+" for cap)
        city_block = np.asarray(est_row[REDUCED_CITY_SLICE])
        if city_block.size > 0:
            c_idx   = int(np.argmax(city_block))
            c_alpha = float(np.clip(city_block[c_idx], 0.0, 1.0))
            if c_idx >= 1 and c_alpha > 0.05:
                if c_idx == 1:
                    ax.add_patch(Circle(
                        (cx, y + 0.22), 0.09,
                        facecolor='#8B4513', edgecolor='black',
                        linewidth=0.8, zorder=3, alpha=c_alpha,
                    ))
                else:
                    level = c_idx - 1
                    label = f"{level}+" if level >= MAX_CITY_LEVEL_HIDDEN else str(level)
                    ax.add_patch(Rectangle(
                        (x + 0.38, y + 0.10), 0.24, 0.14,
                        facecolor=_P_COLORS[opp_pid], edgecolor='black',
                        linewidth=0.8, zorder=3, alpha=c_alpha,
                    ))
                    ax.text(cx, y + 0.17, label,
                            ha='center', va='center',
                            fontsize=6, color='white', zorder=4,
                            alpha=c_alpha)

        # Layer 5 — opponent unit silhouettes.  Draw EVERY unit type whose
        # softmax probability is non-trivial, with `alpha = p(class)` so the
        # rendered overlay is the full posterior over unit type, not just the
        # argmax. Each silhouette gets its own jitter offset so the different
        # candidates don't all stack on the tile center.
        unit_block = np.asarray(est_row[REDUCED_OPP_UNIT_SLICE])
        if unit_block.size > 0:
            color = _P_COLORS[opp_pid]
            for u_idx in range(1, unit_block.size):  # skip idx 0 = None
                u_alpha = float(np.clip(unit_block[u_idx], 0.0, 1.0))
                if u_alpha < 0.05:
                    continue
                try:
                    ut = UnitType(u_idx - 1)
                except ValueError:
                    continue
                if ut not in _UNIT_GLYPH:
                    continue
                if rng is not None:
                    jx = float(rng.uniform(-0.18, 0.18))
                    jy = float(rng.uniform(-0.14, 0.14))
                else:
                    jx, jy = 0.0, 0.0
                self._draw_unit_glyph(
                    ax, ut, cx + jx, cy + jy, color,
                    alpha=u_alpha, outline=True,
                    fontsize_main=18, fontsize_pair=12,
                    linewidth=1.4, zorder=5,
                )

        # Layer 6 — global hidden-tile shadow on top
        ax.add_patch(Rectangle(
            (x, y), 1, 1,
            facecolor=_HIDDEN_SHADOW, alpha=0.20,
            edgecolor='none', zorder=6,
        ))

    # ── Pass 2: units ───────────────────────────────────────────────────
    def _render_units(self, ax, uncovered):
        Nx, Ny = self.Nx, self.Ny
        board  = self.game.game_board.board
        for player in self.game.players:
            for _, unit in player.units_under_control.items():
                tile_id = unit.tile.id
                if tile_id not in uncovered:
                    continue
                row = tile_id // Ny
                col = tile_id %  Ny
                x, y = col, Nx - 1 - row
                tile_obj = board[tile_id]
                walled = (tile_obj.city is not None
                          and len(tile_obj.city.choices) >= 2
                          and tile_obj.city.choices[1] == 1)
                self._draw_unit(ax, unit, x, y, walled)

    # ── Pass 3: action overlays ─────────────────────────────────────────
    def _render_action_overlay(self, ax):
        self._draw_action_overlay(ax, self.tile_center)

    # ── Player-control perimeter outline (replaces alpha fill) ─────────
    def _render_control_perimeter(self, ax, uncovered):
        Nx, Ny = self.Nx, self.Ny
        state_grid = self.game.game_board.board_graph.reshape(Nx, Ny, NODE_FEAT_DIM)

        owner = -np.ones((Nx, Ny), dtype=np.int8)
        for i in range(Nx):
            for j in range(Ny):
                tid = i * Ny + j
                if tid not in uncovered:
                    continue
                if   state_grid[i, j, _PLAYER_CTRL_START]     > 0: owner[i, j] = 0
                elif state_grid[i, j, _PLAYER_CTRL_START + 1] > 0: owner[i, j] = 1

        # (di, dj, x0, y0, x1, y1) — edge offsets relative to tile (x, y)
        edges = [
            (-1,  0, 0.0, 1.0, 1.0, 1.0),  # top
            ( 1,  0, 0.0, 0.0, 1.0, 0.0),  # bottom
            ( 0, -1, 0.0, 0.0, 0.0, 1.0),  # left
            ( 0,  1, 1.0, 0.0, 1.0, 1.0),  # right
        ]
        LW = 2.5
        for i in range(Nx):
            for j in range(Ny):
                o = int(owner[i, j])
                if o == -1:
                    continue
                x, y  = j, Nx - 1 - i
                color = _P_COLORS[o]
                for di, dj, xa, ya, xb, yb in edges:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < Nx and 0 <= nj < Ny:
                        neigh = int(owner[ni, nj])
                    else:
                        neigh = -1
                    if neigh != o:
                        ax.plot([x + xa, x + xb], [y + ya, y + yb],
                                color=color, lw=LW,
                                solid_capstyle='round', zorder=2.5)

    # ── Trajectory probability bookkeeping ──────────────────────────────
    def compute_prob_overlay(self, action, joint_probs, traj_actions):
        """Return (prob_overlay: dict[tid→float], atype_probs: dict[name→float])."""
        atype_probs = {at.name: 0.0 for at in ActionTypes}
        prob_overlay: dict[int, float] = {}
        if joint_probs is None or traj_actions is None:
            return prob_overlay, atype_probs

        try:
            probs_np = joint_probs.detach().cpu().numpy()
        except AttributeError:
            probs_np = np.asarray(joint_probs)

        for traj, p in zip(traj_actions, probs_np):
            atype_probs[ActionTypes(traj[0]).name] += float(p)

        if action is not None:
            sampled_atype = ActionTypes(action[0])
            if sampled_atype == ActionTypes.MoveUnit and len(action) >= 2:
                sampled_uid = action[1]
                mask = np.array([
                    (ActionTypes(t[0]) == ActionTypes.MoveUnit
                     and len(t) >= 3 and t[1] == sampled_uid)
                    for t in traj_actions
                ], dtype=bool)
                if mask.any():
                    move_probs = probs_np[mask]
                    targets    = [traj_actions[i][2] for i in np.where(mask)[0]]
                    total = move_probs.sum()
                    if total > 0:
                        move_probs = move_probs / total
                    for tid, alpha in zip(targets, move_probs):
                        prob_overlay[int(tid)] = float(alpha)

        return prob_overlay, atype_probs

    # ════════════════════════════════════════════════════════════════════
    # Helpers moved verbatim from EnvWrapper (rendering atoms)
    # ════════════════════════════════════════════════════════════════════

    def _draw_unit_glyph(self, ax, unit_type, cx, cy, color, *,
                         alpha=1.0, outline=False, effects=None,
                         fontsize_main=22, fontsize_pair=14,
                         linewidth=1.4, zorder=5):
        """Render a unit's glyph (and any composite extras) at tile-center
        ``(cx, cy)``. Used both for live units and hidden-tile silhouettes.

        ``outline=True`` switches every text glyph to a `TextPath` traced by
        a `PathPatch(fill=False)` — guaranteed outline-only regardless of
        backend. The sword/shield decorations are skipped in outline mode.
        """
        top_glyph, extra = _UNIT_GLYPH[unit_type]

        def _draw_text(text, x, y, sz):
            if outline:
                fp = FontProperties(family='DejaVu Sans', size=sz)
                tp = TextPath((0, 0), text, prop=fp)
                bb = tp.get_extents()
                target_h = 0.55 * (sz / 22.0)
                scale = target_h / max(bb.height, 1e-6)
                tx = x - (bb.x0 + bb.width  / 2) * scale
                ty = y - (bb.y0 + bb.height / 2) * scale
                ax.add_patch(PathPatch(
                    tp.transformed(Affine2D().scale(scale).translate(tx, ty)),
                    fill=False, edgecolor=color, linewidth=linewidth,
                    alpha=alpha, joinstyle='round', capstyle='round',
                    zorder=zorder,
                ))
            else:
                ax.text(x, y, text,
                        ha='center', va='center', fontsize=sz, color=color,
                        fontfamily='DejaVu Sans', zorder=zorder,
                        alpha=alpha, path_effects=effects)

        if extra is None:
            _draw_text(top_glyph, cx, cy + 0.04, fontsize_main)
        elif extra == 'sword':
            _draw_text(top_glyph, cx - 0.05, cy + 0.04, fontsize_main)
            if not outline:
                ax.plot([cx + 0.10, cx + 0.24], [cy - 0.10, cy + 0.16],
                        color='#C0C0C0', lw=2.2, zorder=zorder + 1,
                        solid_capstyle='round', path_effects=effects, alpha=alpha)
                ax.plot([cx + 0.07, cx + 0.17], [cy - 0.04, cy - 0.14],
                        color='#8B4513', lw=2.0, zorder=zorder + 1,
                        solid_capstyle='round', path_effects=effects, alpha=alpha)
        elif extra == 'shield':
            _draw_text(top_glyph, cx - 0.05, cy + 0.04, fontsize_main)
            if not outline:
                shield = Polygon(
                    [(cx + 0.10, cy + 0.18), (cx + 0.26, cy + 0.18),
                     (cx + 0.26, cy - 0.02), (cx + 0.18, cy - 0.16),
                     (cx + 0.10, cy - 0.02)],
                    closed=True, facecolor='#B0B0B0',
                    edgecolor='#7A4A1A', linewidth=1.2, zorder=zorder + 1, alpha=alpha,
                )
                if effects is not None:
                    shield.set_path_effects(effects)
                ax.add_patch(shield)
        else:
            # composite: queen atop a chess-knight
            _draw_text(top_glyph, cx, cy + 0.16, fontsize_pair)
            _draw_text(extra,     cx, cy - 0.12, fontsize_pair)

    def _draw_unit(self, ax, unit, x, y, walled):
        cx, cy = x + 0.5, y + 0.5
        pid = int(unit.player_id)
        color = _P_VET[pid] if unit.is_vet else _P_COLORS[pid]

        # silhouette glow: outline stroke that traces the glyph/icon shapes
        if unit.turn_state in (UnitState.ready, UnitState.escaping, UnitState.can_hit):
            effects = [pe.withStroke(linewidth=5, foreground='white', alpha=0.55)]
        else:
            effects = None

        # defensive shield (left of the glyph; larger for walled cities)
        if unit.def_bonus != DefenseBonus.NoBonus:
            size = 0.12 if (unit.def_bonus == DefenseBonus.Wall or walled) else 0.07
            self._draw_shield(ax, x + 0.14, cy + 0.02, size)

        self._draw_unit_glyph(
            ax, unit.unit_type, cx, cy, color,
            outline=False, effects=effects,
            fontsize_main=22, fontsize_pair=14, zorder=5,
        )

        # HP text (top of tile so it never collides with the city marker) + heal glow
        hp_y = y + 0.88
        healed = (self.last_action is not None
                  and self.last_action["type"] == ActionTypes.HealUnit
                  and self._overlay_ctx.get("healed_unit_id") == unit.unit_id)
        if healed:
            ax.add_patch(Circle((cx, hp_y), 0.13,
                facecolor='#90EE90', edgecolor='none', alpha=0.85, zorder=6))
        hp_text = f"{int(round(unit.current_hp))}/{int(unit.hp)}"
        ax.text(cx, hp_y, hp_text,
                ha='center', va='center', fontsize=6.5,
                fontweight='bold', color='black', zorder=7,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          edgecolor='none', alpha=0.7))

    def _draw_shield(self, ax, cx, cy, s):
        pts = np.array([
            [cx - s,        cy + s * 1.1],
            [cx + s,        cy + s * 1.1],
            [cx + s,        cy - s * 0.2],
            [cx,            cy - s * 1.3],
            [cx - s,        cy - s * 0.2],
        ])
        ax.add_patch(Polygon(pts, facecolor='#D3D3D3',
            edgecolor='#404040', linewidth=0.8, zorder=5))
        ax.plot([cx, cx], [cy - s * 0.9, cy + s * 0.9],
                color='#404040', lw=0.6, zorder=6)
        ax.plot([cx - s * 0.7, cx + s * 0.7], [cy + s * 0.2, cy + s * 0.2],
                color='#404040', lw=0.6, zorder=6)

    def _draw_sword_icon(self, ax, mx, my, angle, s=0.22):
        bx0 = mx + np.cos(angle) * s * 0.7
        by0 = my + np.sin(angle) * s * 0.7
        bx1 = mx - np.cos(angle) * s * 0.7
        by1 = my - np.sin(angle) * s * 0.7
        ax.plot([bx0, bx1], [by0, by1],
                color='#C0C0C0', lw=3.0, solid_capstyle='round', zorder=10)
        perp = angle + np.pi / 2
        gx0 = mx + np.cos(perp) * s * 0.30
        gy0 = my + np.sin(perp) * s * 0.30
        gx1 = mx - np.cos(perp) * s * 0.30
        gy1 = my - np.sin(perp) * s * 0.30
        ax.plot([gx0, gx1], [gy0, gy1],
                color='#8B4513', lw=2.4, solid_capstyle='round', zorder=10)

    def _draw_action_overlay(self, ax, tile_center):
        ta = self.last_action
        if ta is None:
            return
        atype = ta["type"]
        ctx = self._overlay_ctx

        if atype == ActionTypes.MoveUnit:
            path = ctx.get("path", [])
            if len(path) < 2:
                return
            sx, sy = tile_center(path[0])
            dx, dy = tile_center(path[-1])
            ax.add_patch(mpatches.FancyArrowPatch(
                (sx, sy), (dx, dy),
                arrowstyle='-|>', mutation_scale=18,
                color='black', lw=1.8, zorder=10,
                shrinkA=6, shrinkB=10))

        elif atype == ActionTypes.Attack:
            atk_x, atk_y = tile_center(ctx["attacker_tile_id"])
            def_x, def_y = tile_center(ctx["defender_tile_id"])
            ranged = ctx.get("attacker_range", 1) > 1
            died   = ctx.get("defender_died", False)

            if ranged:
                ax.add_patch(Circle((def_x, def_y), 0.38,
                    facecolor='none', edgecolor='red',
                    lw=1.8, linestyle='--', zorder=10))
                ax.add_patch(Circle((def_x, def_y), 0.18,
                    facecolor='none', edgecolor='red',
                    lw=1.4, zorder=10))
                ax.plot([atk_x, def_x], [atk_y, def_y],
                        color='red', lw=1.3, linestyle='--', zorder=10)

            angle = np.arctan2(def_y - atk_y, def_x - atk_x)
            if died:
                mx = atk_x + 0.35 * (def_x - atk_x)
                my = atk_y + 0.35 * (def_y - atk_y)
                self._draw_sword_icon(ax, mx, my, angle)
            elif not ranged:
                mx = (atk_x + def_x) / 2
                my = (atk_y + def_y) / 2
                self._draw_sword_icon(ax, mx, my, angle)

    # ── Info panel ──────────────────────────────────────────────────────
    def _render_info_panel(self, ax_info, *, atype_probs=None,
                           critic_value=None, horizontal=False):
        if horizontal:
            self._draw_info_panel_horizontal(ax_info, atype_probs, critic_value)
        else:
            self._draw_info_panel_vertical(ax_info, atype_probs, critic_value)

    def _draw_info_panel_vertical(self, ax_info, atype_probs, critic_value):
        ax_info.axis('off')
        ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)

        pid = self.game.player_go_id
        pcolor_hx = '#1E6FD9' if pid == 0 else '#D92B1E'

        badge = FancyBboxPatch((0.05, 0.88), 0.90, 0.10,
            boxstyle="round,pad=0.02",
            facecolor=pcolor_hx, edgecolor='none', alpha=0.85)
        ax_info.add_patch(badge)
        ax_info.text(0.50, 0.93, f"▶  Player {pid}'s Turn",
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='white')

        def _row(y, label, value, vc='#222222'):
            ax_info.text(0.08, y, label, ha='left',  va='top',
                fontsize=9, color='#555555')
            ax_info.text(0.92, y, str(value), ha='right', va='top',
                fontsize=9, fontweight='bold', color=vc)

        _row(0.82, "Turn", self.game.turn)
        _row(0.76, "Decisions", self.n_decisions)
        if critic_value is not None:
            v  = critic_value.item() if hasattr(critic_value, 'item') else float(critic_value)
            vc = '#1a7a1a' if v >= 0 else '#cc2200'
            _row(0.70, "Critic V̂", f"{v:+.3f}", vc=vc)

        sep_y = 0.66 if critic_value is not None else 0.72
        ax_info.plot([0.05, 0.95], [sep_y, sep_y],
            color='#CCCCCC', linewidth=0.6)

        for p_idx, player in enumerate(self.game.players):
            y_lbl = sep_y - 0.06 - p_idx * 0.07
            c = _P_COLORS[p_idx]
            ax_info.text(0.08, y_lbl, f"P{p_idx}",
                ha='left', va='top', fontsize=9,
                fontweight='bold', color=c)
            ax_info.text(0.92, y_lbl,
                f"★ {player.stars} (+{player.current_stars_per_turn})",
                ha='right', va='top', fontsize=9, color='#222222')

        # Last action
        last_y = sep_y - 0.06 - 2 * 0.07 - 0.02
        ax_info.plot([0.05, 0.95], [last_y, last_y],
            color='#CCCCCC', linewidth=0.6)
        ax_info.text(0.08, last_y - 0.04, "Last action:",
            ha='left', va='top', fontsize=9, color='#555555')
        ax_info.text(0.08, last_y - 0.10, self._fmt_last_action(),
            ha='left', va='top', fontsize=8, color='#222222')

        # Action-type probability bars
        if atype_probs is not None:
            bars_top = last_y - 0.16
            ax_info.plot([0.05, 0.95], [bars_top, bars_top],
                color='#CCCCCC', linewidth=0.6)
            ax_info.text(0.50, bars_top - 0.025, "Action Probabilities",
                ha='center', va='top', fontsize=8.5, fontweight='bold',
                color='#333333')
            row_y = bars_top - 0.06
            row_h = 0.028
            available = max(row_y - 0.02, 0.05)
            row_step = available / max(len(ActionTypes), 1)
            for at in ActionTypes:
                p = atype_probs.get(at.name, 0.0)
                bar_w = float(p) * 0.60
                ax_info.add_patch(FancyBboxPatch(
                    (0.08, row_y - row_h), 0.60, row_h,
                    boxstyle="round,pad=0.002",
                    facecolor='#EEEEEE', edgecolor='none', clip_on=False, zorder=2))
                if bar_w > 0:
                    ax_info.add_patch(FancyBboxPatch(
                        (0.08, row_y - row_h), bar_w, row_h,
                        boxstyle="round,pad=0.002",
                        facecolor=pcolor_hx, edgecolor='none', alpha=0.75,
                        clip_on=False, zorder=3))
                ax_info.text(0.08, row_y - row_h * 0.5, at.name,
                    ha='left', va='center', fontsize=7, color='#333333', zorder=4)
                ax_info.text(0.70, row_y - row_h * 0.5, f"{float(p)*100:.1f}%",
                    ha='left', va='center', fontsize=7, fontweight='bold',
                    color='#333333', zorder=4)
                row_y -= row_step

    def _draw_info_panel_horizontal(self, ax_info, atype_probs, critic_value):
        ax_info.axis('off')
        ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)

        pid = self.game.player_go_id
        pcolor_hx = '#1E6FD9' if pid == 0 else '#D92B1E'

        # Player chip on the left
        chip = FancyBboxPatch((0.005, 0.55), 0.13, 0.40,
            boxstyle="round,pad=0.02",
            facecolor=pcolor_hx, edgecolor='none', alpha=0.85)
        ax_info.add_patch(chip)
        ax_info.text(0.07, 0.75, f"▶  P{pid}'s Turn",
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='white')

        # Mid columns: stats
        ax_info.text(0.20, 0.82, "Turn", fontsize=8, color='#555555')
        ax_info.text(0.20, 0.62, str(self.game.turn), fontsize=11,
                     fontweight='bold', color='#222222')

        ax_info.text(0.30, 0.82, "Decisions", fontsize=8, color='#555555')
        ax_info.text(0.30, 0.62, str(self.n_decisions), fontsize=11,
                     fontweight='bold', color='#222222')

        col_x = 0.42
        for p_idx, player in enumerate(self.game.players):
            c = _P_COLORS[p_idx]
            ax_info.text(col_x, 0.82, f"P{p_idx}",
                fontsize=8, fontweight='bold', color=c)
            ax_info.text(col_x, 0.62,
                f"★ {player.stars} (+{player.current_stars_per_turn})",
                fontsize=10, color='#222222')
            col_x += 0.10

        if critic_value is not None:
            v  = critic_value.item() if hasattr(critic_value, 'item') else float(critic_value)
            vc = '#1a7a1a' if v >= 0 else '#cc2200'
            ax_info.text(col_x, 0.82, "V̂", fontsize=8, color='#555555')
            ax_info.text(col_x, 0.62, f"{v:+.3f}",
                fontsize=10, fontweight='bold', color=vc)
            col_x += 0.10

        # Last action
        ax_info.text(0.20, 0.32, "Last action:", fontsize=8, color='#555555')
        ax_info.text(0.20, 0.10, self._fmt_last_action(),
                     fontsize=8.5, color='#222222')

        # Action-prob bars on the right half
        if atype_probs is not None:
            bar_x0   = 0.74
            bar_w_max = 0.22
            ax_info.text(bar_x0 + bar_w_max / 2, 0.92, "Action Probabilities",
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='#333333')
            n = max(len(ActionTypes), 1)
            row_h = 0.85 / n
            for k, at in enumerate(ActionTypes):
                p = atype_probs.get(at.name, 0.0)
                bw = float(p) * bar_w_max
                ry = 0.86 - (k + 1) * row_h
                ax_info.add_patch(FancyBboxPatch(
                    (bar_x0, ry), bar_w_max, row_h * 0.8,
                    boxstyle="round,pad=0.002",
                    facecolor='#EEEEEE', edgecolor='none', clip_on=False, zorder=2))
                if bw > 0:
                    ax_info.add_patch(FancyBboxPatch(
                        (bar_x0, ry), bw, row_h * 0.8,
                        boxstyle="round,pad=0.002",
                        facecolor=pcolor_hx, edgecolor='none', alpha=0.75,
                        clip_on=False, zorder=3))
                ax_info.text(bar_x0 - 0.005, ry + row_h * 0.4, at.name,
                    ha='right', va='center', fontsize=6.5,
                    color='#333333', zorder=4)
                ax_info.text(bar_x0 + bar_w_max + 0.005, ry + row_h * 0.4,
                    f"{float(p)*100:.1f}%", ha='left', va='center',
                    fontsize=6.5, fontweight='bold', color='#333333', zorder=4)

    def _fmt_last_action(self):
        ta = self.last_action
        if ta is None:
            return '(none yet)'
        atype = ta["type"]
        ctx = self._overlay_ctx
        if atype == ActionTypes.MoveUnit:
            return f"MoveUnit  tile {ctx.get('src_tile_id')} → {ctx.get('dst_tile_id')}"
        if atype == ActionTypes.Attack:
            flag = "  [KILL]" if ctx.get("defender_died") else ""
            return f"Attack  tile {ctx.get('attacker_tile_id')} → {ctx.get('defender_tile_id')}{flag}"
        if atype == ActionTypes.CreateUnit:
            ut = ta.get("unit_type")
            return f"CreateUnit  {ut.name if ut is not None else '?'}  city={ta.get('city')}"
        if atype == ActionTypes.CaptureCity:
            return f"CaptureCity  u={ta.get('unit_id')}"
        if atype == ActionTypes.HealUnit:
            return f"HealUnit  u={ta.get('unit_id')}"
        if atype == ActionTypes.UpgradeCity:
            return f"UpgradeCity  city={ta.get('city')}  choice={ta.get('choice')}"
        if atype == ActionTypes.PlaceRoad:
            return f"PlaceRoad  tile={ta.get('tile_id')}"
        if atype == ActionTypes.Upgrade2Vet:
            return f"Upgrade2Vet  u={ta.get('unit_id')}"
        if atype == ActionTypes.EndTurn:
            return "EndTurn"
        return atype.name
