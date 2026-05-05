"""
scenarios/editor.py
──────────────────────────────────────────────────────────────────────────────
Interactive WYSIWYG editor for game-state scenarios.

Run:  python -m scenarios.editor          (blank canvas)
      python -m scenarios                 (alias for the same)

The user clicks on a board canvas; clicks dispatch by mode (Tile / Unit /
City / Road / Fog / Inspect) and mutate the in-memory `Scenario`. The board
is repainted by the existing `BoardRenderer` after every edit, so edit and
view share the same surface.

Save / Load buttons land in commit 4 of the scenario-editor plan; this commit
covers the full editing surface against `Scenario.blank()`.
"""

from __future__ import annotations

import sys
from typing import Optional, Tuple

# Backend selection — must come before any pyplot import. TkAgg is in Python's
# stdlib via tkinter on Windows/macOS/Linux; falls back silently if absent.
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.widgets import (
    Button,
    CheckButtons,
    RadioButtons,
    TextBox,
)

from game.enums import (
    BoardType,
    CityType,
    DefenseBonus,
    PlayerId,
    TileType,
    UnitState,
    UnitType,
)
from game.game     import Game
from env.renderer  import BoardRenderer

from scenarios.scenario import (
    Scenario,
    ScenarioCity,
    ScenarioPlayer,
    ScenarioUnit,
)


# ══════════════════════════════════════════════════════════════════════════════
# Renderer adapter
# ══════════════════════════════════════════════════════════════════════════════

class _EditorEnv:
    """
    Minimal duck-type adapter for `BoardRenderer`, which expects the live
    EnvWrapper interface. We only need a handful of attributes — the editor
    has no rollouts, no last_action, no overlays in v1.
    """
    def __init__(self, game, Nx, Ny):
        self.game         = game
        self.Nx           = Nx
        self.Ny           = Ny
        self.n_players    = game.n_players
        self.n_decisions  = 0
        self.last_action  = None
        self._overlay_ctx = None


# ══════════════════════════════════════════════════════════════════════════════
# Layout constants — figure-relative coordinates
# ══════════════════════════════════════════════════════════════════════════════

# Left column (canvas + info panel)
_BOARD_RECT     = [0.025, 0.30, 0.62, 0.66]
_INFO_RECT      = [0.025, 0.02, 0.62, 0.26]

# Right column
_MODE_RECT      = [0.66,  0.66, 0.32, 0.30]
_SUB_REGION     = (0.66,  0.25, 0.32, 0.39)   # (l, b, w, h) — sub-controls
_TOOLBAR_RECT   = (0.66,  0.155, 0.32, 0.09)   # Generate / Save / Load (3 rows)
_PERSIST_REGION = (0.66,  0.02, 0.32, 0.12)


# Modes (order = display order in RadioButtons)
_MODES = ("Tile", "Unit", "City", "Road", "Fog", "Inspect")


# Per-mode default selections in the sub-control widgets.
_DEFAULT_TERRAIN  = TileType.field
_DEFAULT_UNIT     = UnitType.Warrior
_DEFAULT_STATE    = UnitState.ready
_DEFAULT_CITY_LVL = CityType.lvl1


# ══════════════════════════════════════════════════════════════════════════════
# EditorApp
# ══════════════════════════════════════════════════════════════════════════════

class EditorApp:
    """
    Top-level editor controller. Holds the live `Scenario`, builds the figure,
    and dispatches widget callbacks + canvas clicks back to scenario mutators.
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, scenario: Optional[Scenario] = None):
        self.scenario      = scenario or Scenario.blank()
        self.current_mode  = "Tile"
        self.active_player = 0
        self._dirty        = False

        # Per-mode "current spec" — what gets placed on the next click.
        self.tile_terrain   = _DEFAULT_TERRAIN
        self.unit_type      = _DEFAULT_UNIT
        self.unit_hp_str    = ""               # empty → use class default
        self.unit_state         = _DEFAULT_STATE
        self.unit_is_vet        = False
        self.unit_fortify       = False
        self.unit_def_bonus     = DefenseBonus.NoBonus
        self.unit_kills_str     = ""               # empty → 0
        self.city_level         = _DEFAULT_CITY_LVL
        self.city_capital       = False
        self.city_choices       = ""               # comma-separated string
        self.city_border_extended = False          # radius 1 (False) or 2 (True)

        # Widget storage (must keep refs alive; matplotlib GCs unreferenced widgets).
        self._widgets:   dict = {}
        self._sub_axes:  list = []     # axes that get rebuilt on mode change

        # Re-entry guard for programmatic widget updates. Matplotlib's
        # RadioButtons.set_active() fires the click callback, and
        # TextBox.set_val() fires both `change` and `submit`. Without this
        # flag, _sync_widgets_to_scenario would re-fire every handler and
        # mistakenly mark the just-loaded scenario as dirty (and recursively
        # re-trigger active-player ↔ owner mirroring).
        self._syncing: bool = False

        # Build figure & all persistent widgets.
        self._build_figure()
        self._build_mode_radio()
        self._build_persist_panel()
        self._build_toolbar()
        self._build_sub_controls(self.current_mode)

        # Canvas click handler.
        self.fig.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        # First paint.
        self._redraw()

    # ── Figure scaffold ──────────────────────────────────────────────────────

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title("Scenario Editor")
        self.ax_board = self.fig.add_axes(_BOARD_RECT)
        self.ax_info  = self.fig.add_axes(_INFO_RECT)
        self.ax_mode  = self.fig.add_axes(_MODE_RECT)
        self.ax_mode.set_title("Mode", fontsize=10, fontweight="bold")

    def _build_mode_radio(self) -> None:
        radio = RadioButtons(self.ax_mode, _MODES, active=_MODES.index(self.current_mode))
        radio.on_clicked(self._on_mode_change)
        self._widgets["mode"] = radio

    # ── Persistent panel (always visible) ─────────────────────────────────────

    def _build_persist_panel(self) -> None:
        l, b, w, h = _PERSIST_REGION

        # Derived stats line (top of persist region; refreshed in _redraw).
        # Score = player_score_official (the @property fixed in this commit);
        # SPT = sum of each city's city_stars_per_turn.
        ax_stats = self.fig.add_axes([l, b + 0.100, w, 0.020])
        ax_stats.axis("off")
        self._stats_text = ax_stats.text(
            0, 0.5, "", fontsize=8, va="center", fontfamily="monospace",
        )

        # Active-player radio (no title — "P0"/"P1" is self-explanatory).
        ax_active = self.fig.add_axes([l, b + 0.030, 0.10, 0.060])
        radio_active = RadioButtons(
            ax_active,
            tuple(f"P{i}" for i in range(self.scenario.n_players)),
            active=self.active_player,
        )
        radio_active.on_clicked(self._on_active_player_change)
        self._widgets["active_player"] = radio_active

        # P0 stars TextBox
        ax_p0_stars = self.fig.add_axes([l + 0.13, b + 0.060, 0.06, 0.030])
        tb_p0 = TextBox(ax_p0_stars, "P0★ ",
                        initial=str(self.scenario.players[0].stars),
                        textalignment="center")
        tb_p0.on_submit(lambda txt: self._set_player_stars(0, txt))
        self._widgets["p0_stars"] = tb_p0

        # P1 stars TextBox
        ax_p1_stars = self.fig.add_axes([l + 0.24, b + 0.060, 0.06, 0.030])
        tb_p1 = TextBox(ax_p1_stars, "P1★ ",
                        initial=str(self.scenario.players[1].stars),
                        textalignment="center")
        tb_p1.on_submit(lambda txt: self._set_player_stars(1, txt))
        self._widgets["p1_stars"] = tb_p1

        # Turn TextBox
        ax_turn = self.fig.add_axes([l + 0.13, b + 0.020, 0.06, 0.030])
        tb_turn = TextBox(ax_turn, "Turn ", initial=str(self.scenario.turn),
                          textalignment="center")
        tb_turn.on_submit(self._set_turn)
        self._widgets["turn"] = tb_turn

        # Status text (bottom of persist region; transient mode/edit feedback)
        self.ax_status = self.fig.add_axes([l, b - 0.010, w, 0.020])
        self.ax_status.axis("off")
        self._status_text = self.ax_status.text(
            0, 0.5, "", fontsize=8, va="center", fontfamily="monospace",
        )

    # ── Toolbar (Save / Load / Reset) ─────────────────────────────────────────

    def _build_toolbar(self) -> None:
        """3-row layout: Generate (row 1) / Save+Reset (row 2) / Load (row 3)."""
        l, b, w, h = _TOOLBAR_RECT

        # Row 1 — Generate TextBox + Generate Button
        ax_gen_tb = self.fig.add_axes([l + 0.07, b + 0.060, 0.14, 0.025])
        tb_gen = TextBox(ax_gen_tb, "Generate: ", initial="",
                         textalignment="left")
        tb_gen.on_submit(lambda _txt: self._on_generate_click(None))
        self._widgets["generate"] = tb_gen

        ax_gen_btn = self.fig.add_axes([l + 0.225, b + 0.060, 0.085, 0.025])
        btn_gen = Button(ax_gen_btn, "Generate")
        btn_gen.on_clicked(self._on_generate_click)
        self._widgets["generate_btn"] = btn_gen

        # Row 2 — Save TextBox + Reset Button
        ax_save = self.fig.add_axes([l + 0.05, b + 0.030, 0.16, 0.025])
        tb_save = TextBox(ax_save, "Save: ", initial="", textalignment="left")
        tb_save.on_submit(self._on_save_submit)
        self._widgets["save"] = tb_save

        ax_reset = self.fig.add_axes([l + 0.235, b + 0.030, 0.075, 0.025])
        btn_reset = Button(ax_reset, "Reset")
        btn_reset.on_clicked(self._on_reset_click)
        self._widgets["reset"] = btn_reset

        # Row 3 — Load TextBox
        ax_load = self.fig.add_axes([l + 0.05, b + 0.000, 0.26, 0.025])
        tb_load = TextBox(ax_load, "Load: ", initial="", textalignment="left")
        tb_load.on_submit(self._on_load_submit)
        self._widgets["load"] = tb_load

    # ── Save / Load / Reset callbacks ─────────────────────────────────────────

    _DEFAULT_SAVE_DIR = "scenarios/scenarios"

    def _resolve_yaml_path(self, raw: str) -> str:
        """Apply default directory + .yaml extension to a user-typed path."""
        name = raw.strip()
        if not name:
            return ""
        if "/" not in name and "\\" not in name:
            name = f"{self._DEFAULT_SAVE_DIR}/{name}"
        if not name.endswith(".yaml"):
            name = name + ".yaml"
        return name

    def _on_save_submit(self, raw: str) -> None:
        path = self._resolve_yaml_path(raw)
        if not path:
            return
        try:
            self.scenario.to_yaml(path)
        except Exception as e:
            self._set_status(f"Save failed: {type(e).__name__}: {e}")
            return
        self._dirty = False
        self._reset_armed_flags()
        # Mirror the resolved name so the user sees what was actually saved.
        self._widgets["save"].set_val("")
        self._set_status(f"Saved → {path}")
        self._redraw()

    def _on_load_submit(self, raw: str) -> None:
        path = self._resolve_yaml_path(raw)
        if not path:
            return
        # Confirm-on-discard: arm the flag, force a second Enter.
        if self._dirty and not getattr(self, "_load_armed", False):
            self._load_armed = True
            self._set_status(
                "Unsaved changes — press Enter again in Load to discard."
            )
            return
        try:
            new_scenario = Scenario.from_yaml(path)
        except Exception as e:
            self._set_status(f"Load failed: {type(e).__name__}: {e}")
            return
        self.scenario       = new_scenario
        self._dirty         = False
        self._reset_armed_flags()
        self._widgets["load"].set_val("")
        self._sync_widgets_to_scenario()
        self._set_status(f"Loaded ← {path}")
        self._redraw()

    def _on_reset_click(self, _event) -> None:
        if self._dirty and not getattr(self, "_reset_armed", False):
            self._reset_armed = True
            self._set_status(
                "Unsaved changes — click Reset again to discard."
            )
            return
        self.scenario       = Scenario.blank()
        self._dirty         = False
        self._reset_armed_flags()
        self._sync_widgets_to_scenario()
        self._set_status("Reset to blank scenario.")
        self._redraw()

    # ── Generate (random board) ───────────────────────────────────────────────

    _BOARD_SIZE_MIN = 5
    _BOARD_SIZE_MAX = 25

    def _parse_generate(self, raw: str) -> Tuple[BoardType, int]:
        """
        Parse the Generate TextBox.
        Accepts: "<type> <N>", "<N> <type>", or just "<N>" (defaults to Drylands).
        Raises ValueError with a user-facing message on bad input.
        """
        toks = raw.strip().split()
        if not toks:
            raise ValueError("empty input")
        bt: Optional[BoardType] = None
        N:  Optional[int]       = None

        for tok in toks:
            # Try int first; otherwise treat as board-type name.
            try:
                v = int(tok)
                if N is not None:
                    raise ValueError(f"two integers in input: '{raw}'")
                N = v
                continue
            except ValueError:
                pass
            try:
                cand = BoardType[tok.capitalize()]
            except KeyError:
                raise ValueError(
                    f"unknown token '{tok}'; expected '<type> <N>' "
                    f"with type in Drylands/Lakes/Archipelago"
                )
            if bt is not None:
                raise ValueError(f"two board types in input: '{raw}'")
            bt = cand

        if N is None:
            raise ValueError(f"missing size N in '{raw}'")
        if bt is None:
            bt = BoardType.Drylands
        if not (self._BOARD_SIZE_MIN <= N <= self._BOARD_SIZE_MAX):
            raise ValueError(
                f"size {N} out of bounds [{self._BOARD_SIZE_MIN}..{self._BOARD_SIZE_MAX}]"
            )
        return bt, N

    def _on_generate_click(self, _event) -> None:
        if self._dirty and not getattr(self, "_generate_armed", False):
            self._generate_armed = True
            self._set_status(
                "Unsaved changes — click Generate again to discard."
            )
            return
        raw = self._widgets["generate"].text
        try:
            bt, N = self._parse_generate(raw)
        except ValueError as e:
            self._set_status(f"Generate: {e}")
            return
        try:
            cfg = {
                "board_size": [N, N],
                "board_type": bt,
                "n_players":  self.scenario.n_players,
            }
            game = Game(cfg, list(self.scenario.player_tribes))
            game.reset_game()
            new_scenario = Scenario.from_game(game, name=f"random_{bt.name}_{N}")
        except Exception as e:
            self._set_status(f"Generate failed: {type(e).__name__}: {e}")
            return
        self.scenario = new_scenario
        self._dirty   = False
        self._reset_armed_flags()
        self._widgets["generate"].set_val("")
        self._sync_widgets_to_scenario()
        self._set_status(f"Generated random {bt.name} {N}x{N}.")
        self._redraw()

    def _reset_armed_flags(self) -> None:
        """Clear arming flags whenever an action other than the armed one fires."""
        self._load_armed         = False
        self._reset_armed        = False
        self._generate_armed     = False
        self._unit_delete_armed  = False
        self._city_delete_armed  = False

    def _sync_widgets_to_scenario(self) -> None:
        """Reflect the (possibly freshly-loaded) Scenario into all editable widgets.

        Wrapped in `self._syncing = True` so the set_val / set_active callbacks
        fired by matplotlib don't re-mark the just-loaded scenario as dirty.
        """
        self.active_player = int(self.scenario.current_player)
        self._syncing = True
        try:
            for pid in range(self.scenario.n_players):
                tb = self._widgets.get(f"p{pid}_stars")
                if tb is not None:
                    tb.set_val(str(self.scenario.players[pid].stars))
            if "turn" in self._widgets:
                self._widgets["turn"].set_val(str(self.scenario.turn))
            if "active_player" in self._widgets:
                self._widgets["active_player"].set_active(self.active_player)
        finally:
            self._syncing = False

    # ── Sub-controls (per mode) ───────────────────────────────────────────────

    def _build_sub_controls(self, mode: str) -> None:
        # Tear down anything from the previous mode.
        for ax in self._sub_axes:
            ax.remove()
        self._sub_axes = []
        # Drop sub-control widget refs (preserve persistent ones).
        for k in list(self._widgets.keys()):
            if k.startswith("sub_"):
                del self._widgets[k]

        builder = {
            "Tile":    self._build_sub_tile,
            "Unit":    self._build_sub_unit,
            "City":    self._build_sub_city,
            "Road":    self._build_sub_road,
            "Fog":     self._build_sub_fog,
            "Inspect": self._build_sub_inspect,
        }[mode]
        builder()

    def _add_sub_axis(self, rect):
        ax = self.fig.add_axes(rect)
        self._sub_axes.append(ax)
        return ax

    # Tile mode: terrain radio
    def _build_sub_tile(self) -> None:
        l, b, w, h = _SUB_REGION
        ax_t = self._add_sub_axis([l, b + 0.20, w, 0.26])
        ax_t.set_title("Terrain", fontsize=9, fontweight="bold")
        terrain_names = ("field", "water", "deep_water", "mountain")
        radio = RadioButtons(ax_t, terrain_names, active=0)
        radio.on_clicked(lambda lbl: setattr(self, "tile_terrain", TileType[lbl]))
        self._widgets["sub_terrain"] = radio

    # Unit mode: type radio + hp + state + checks + owner + delete button
    def _build_sub_unit(self) -> None:
        l, b, w, h = _SUB_REGION
        # Type (8 options, top section). Top edge sits at b+0.39 = top of region.
        ax_type = self._add_sub_axis([l, b + 0.29, w, 0.10])
        ax_type.set_title("Unit type", fontsize=9, fontweight="bold")
        type_names = tuple(t.name for t in UnitType)
        radio_t = RadioButtons(ax_type, type_names,
                               active=type_names.index(self.unit_type.name))
        radio_t.on_clicked(lambda lbl: setattr(self, "unit_type", UnitType[lbl]))
        self._widgets["sub_unit_type"] = radio_t

        # HP TextBox
        ax_hp = self._add_sub_axis([l + 0.05, b + 0.275, 0.10, 0.025])
        tb_hp = TextBox(ax_hp, "HP ", initial=self.unit_hp_str,
                        textalignment="center")
        tb_hp.on_submit(lambda txt: setattr(self, "unit_hp_str", txt.strip()))
        self._widgets["sub_unit_hp"] = tb_hp

        # def_bonus radio (3 options)
        ax_def = self._add_sub_axis([l, b + 0.17, 0.18, 0.09])
        ax_def.set_title("def_bonus", fontsize=8)
        bonus_names = tuple(db.name for db in DefenseBonus)
        radio_db = RadioButtons(ax_def, bonus_names,
                                active=bonus_names.index(self.unit_def_bonus.name))
        radio_db.on_clicked(
            lambda lbl: setattr(self, "unit_def_bonus", DefenseBonus[lbl])
        )
        self._widgets["sub_unit_def"] = radio_db

        # kills TextBox
        ax_k = self._add_sub_axis([l + 0.22, b + 0.245, 0.08, 0.025])
        tb_k = TextBox(ax_k, "kills ", initial=self.unit_kills_str,
                       textalignment="center")
        tb_k.on_submit(lambda txt: setattr(self, "unit_kills_str", txt.strip()))
        self._widgets["sub_unit_kills"] = tb_k

        # State radio (4 options)
        ax_state = self._add_sub_axis([l, b + 0.08, 0.16, 0.07])
        ax_state.set_title("State", fontsize=8)
        state_names = tuple(s.name for s in UnitState)
        radio_s = RadioButtons(ax_state, state_names,
                               active=state_names.index(self.unit_state.name))
        radio_s.on_clicked(lambda lbl: setattr(self, "unit_state", UnitState[lbl]))
        self._widgets["sub_unit_state"] = radio_s

        # Vet/Fortify checkboxes
        ax_chk = self._add_sub_axis([l + 0.16, b + 0.08, 0.16, 0.07])
        ax_chk.set_title("Flags", fontsize=8)
        chk = CheckButtons(ax_chk, ("vet", "fortify"),
                           (self.unit_is_vet, self.unit_fortify))
        chk.on_clicked(self._on_unit_flag_toggle)
        self._widgets["sub_unit_chk"] = chk

        # Owner radio
        ax_own = self._add_sub_axis([l, b + 0.020, 0.10, 0.05])
        radio_o = RadioButtons(ax_own, ("P0", "P1"), active=self.active_player)
        radio_o.on_clicked(self._on_owner_change)
        self._widgets["sub_unit_owner"] = radio_o

        # Delete button
        ax_del = self._add_sub_axis([l + 0.12, b + 0.030, 0.18, 0.030])
        btn = Button(ax_del, "Delete unit at click")
        btn.on_clicked(self._on_unit_delete_armed)
        self._widgets["sub_unit_del"] = btn

    # City mode
    def _build_sub_city(self) -> None:
        l, b, w, h = _SUB_REGION
        # Level radio (15 options including 'village' — village = unclaimed,
        # owner/is_capital/extended-border are silently ignored when chosen).
        ax_lvl = self._add_sub_axis([l, b + 0.16, w, 0.23])
        ax_lvl.set_title("City level", fontsize=9, fontweight="bold")
        level_names = tuple(c.name for c in CityType)
        radio_l = RadioButtons(ax_lvl, level_names,
                               active=level_names.index(self.city_level.name))
        radio_l.on_clicked(lambda lbl: setattr(self, "city_level", CityType[lbl]))
        self._widgets["sub_city_lvl"] = radio_l

        # Capital checkbox
        ax_cap = self._add_sub_axis([l, b + 0.10, 0.13, 0.05])
        chk_cap = CheckButtons(ax_cap, ("is_capital",), (self.city_capital,))
        chk_cap.on_clicked(
            lambda _lbl: setattr(self, "city_capital", not self.city_capital)
        )
        self._widgets["sub_city_capital"] = chk_cap

        # Extended border (radius 2) checkbox
        ax_brd = self._add_sub_axis([l + 0.14, b + 0.10, 0.18, 0.05])
        chk_brd = CheckButtons(
            ax_brd, ("Extended border (r=2)",), (self.city_border_extended,),
        )
        chk_brd.on_clicked(
            lambda _lbl: setattr(
                self, "city_border_extended", not self.city_border_extended
            )
        )
        self._widgets["sub_city_brd"] = chk_brd

        # Owner radio
        ax_own = self._add_sub_axis([l, b + 0.04, 0.08, 0.055])
        radio_o = RadioButtons(ax_own, ("P0", "P1"), active=self.active_player)
        radio_o.on_clicked(self._on_owner_change)
        self._widgets["sub_city_owner"] = radio_o

        # Choices TextBox
        ax_ch = self._add_sub_axis([l + 0.10, b + 0.045, 0.20, 0.030])
        tb_ch = TextBox(ax_ch, "choices ", initial=self.city_choices,
                        textalignment="left")
        tb_ch.on_submit(lambda txt: setattr(self, "city_choices", txt.strip()))
        self._widgets["sub_city_ch"] = tb_ch

        # Delete button
        ax_del = self._add_sub_axis([l + 0.10, b + 0.005, 0.20, 0.030])
        btn = Button(ax_del, "Delete city / village at click")
        btn.on_clicked(self._on_city_delete_armed)
        self._widgets["sub_city_del"] = btn

    # Road mode — no sub-controls
    def _build_sub_road(self) -> None:
        l, b, w, h = _SUB_REGION
        ax = self._add_sub_axis([l, b + 0.20, w, 0.20])
        ax.axis("off")
        ax.text(0.5, 0.5,
                "Click a tile to toggle its road.",
                ha="center", va="center", fontsize=10, fontstyle="italic")

    # Fog mode — Cover all / Uncover all buttons
    def _build_sub_fog(self) -> None:
        l, b, w, h = _SUB_REGION
        # Hint text
        ax_hint = self._add_sub_axis([l, b + 0.32, w, 0.10])
        ax_hint.axis("off")
        ax_hint.text(
            0.5, 0.5,
            "Click a tile to toggle its visibility\nfor the active player.",
            ha="center", va="center", fontsize=9, fontstyle="italic",
        )

        # Cover all
        ax_cov = self._add_sub_axis([l + 0.02, b + 0.18, 0.13, 0.06])
        btn_cov = Button(ax_cov, "Cover all")
        btn_cov.on_clicked(lambda _e: self._mutate(
            lambda: self.scenario.cover_all(self.active_player)))
        self._widgets["sub_fog_cov"] = btn_cov

        # Uncover all
        ax_unc = self._add_sub_axis([l + 0.17, b + 0.18, 0.13, 0.06])
        btn_unc = Button(ax_unc, "Uncover all")
        btn_unc.on_clicked(lambda _e: self._mutate(
            lambda: self.scenario.uncover_all(self.active_player)))
        self._widgets["sub_fog_unc"] = btn_unc

    # Inspect mode — text display
    def _build_sub_inspect(self) -> None:
        l, b, w, h = _SUB_REGION
        ax = self._add_sub_axis([l, b, w, h])
        ax.axis("off")
        self._inspect_text = ax.text(
            0.02, 0.95, "Click a tile to inspect it.",
            ha="left", va="top", fontsize=9, fontfamily="monospace",
            transform=ax.transAxes,
        )

    # ── Click dispatcher (board canvas) ───────────────────────────────────────

    def _on_canvas_click(self, event):
        if event.inaxes is not self.ax_board:
            return
        if event.xdata is None or event.ydata is None:
            return
        Nx, Ny = self.scenario.board_size
        # Renderer maps (row=i, col=j) → display (x=j, y=Nx-1-i).
        # Inverse: col = floor(xdata); row_from_top = Nx-1-floor(ydata).
        col = int(event.xdata)
        row = (Nx - 1) - int(event.ydata)
        if not (0 <= col < Ny and 0 <= row < Nx):
            return
        x, y = col, row    # YAML coords: pos=[x,y]

        mode = self.current_mode

        if mode == "Tile":
            self._mutate(lambda: self.scenario.set_terrain(x, y, self.tile_terrain))

        elif mode == "Unit":
            if getattr(self, "_unit_delete_armed", False):
                self._unit_delete_armed = False
                self._mutate(lambda: self.scenario.delete_unit(x, y))
            else:
                self._mutate(lambda: self._place_unit(x, y))

        elif mode == "City":
            if getattr(self, "_city_delete_armed", False):
                self._city_delete_armed = False
                self._mutate(lambda: self.scenario.delete_city(x, y))
            else:
                self._mutate(lambda: self._place_city(x, y))

        elif mode == "Road":
            self._mutate(lambda: self.scenario.toggle_road(x, y))

        elif mode == "Fog":
            self._mutate(lambda: self.scenario.toggle_fog(x, y, self.active_player))

        elif mode == "Inspect":
            # Read-only — does NOT mutate. Just refresh the inspect panel.
            self._update_inspect(x, y)

    def _place_unit(self, x: int, y: int) -> None:
        """Translate the current sub-control state into a ScenarioUnit."""
        try:
            hp = float(self.unit_hp_str) if self.unit_hp_str else None
        except ValueError:
            hp = None
        try:
            kills = int(self.unit_kills_str) if self.unit_kills_str else 0
        except ValueError:
            kills = 0
        self.scenario.set_unit(
            x, y, owner=self.active_player,
            type     = self.unit_type,
            hp       = hp,
            state    = self.unit_state,
            is_vet   = self.unit_is_vet,
            fortify  = self.unit_fortify if self.unit_fortify else None,
            def_bonus= self.unit_def_bonus,
            kills    = kills,
        )

    def _place_city(self, x: int, y: int) -> None:
        # Village → top-level scenario.villages, NOT a player's cities.
        # is_capital / owner / extended-border are silently ignored.
        if self.city_level == CityType.village:
            self.scenario.set_village(x, y)
            return
        # Parse choices: "0,1,0" → [0, 1, 0]; empty → []
        try:
            choices = [int(c.strip()) for c in self.city_choices.split(",") if c.strip()]
        except ValueError:
            choices = []
        self.scenario.set_city(
            x, y, owner=self.active_player,
            level         = self.city_level,
            is_capital    = self.city_capital,
            choices       = choices,
            border_radius = 2 if self.city_border_extended else 1,
        )

    # ── Persistent-panel callbacks ────────────────────────────────────────────

    def _on_mode_change(self, label: str) -> None:
        self.current_mode = label
        self._build_sub_controls(label)
        # Reset arming flags between modes
        self._unit_delete_armed = False
        self._city_delete_armed = False
        self._set_status(f"Mode → {label}")
        self.fig.canvas.draw_idle()

    def _on_active_player_change(self, label: str) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.active_player = int(label[1:])     # "P0" → 0
            self._set_status(f"Active player → {label}")
            # Owner sub-radios in Unit/City modes don't auto-sync — switch them.
            for k in ("sub_unit_owner", "sub_city_owner"):
                if k in self._widgets:
                    self._widgets[k].set_active(self.active_player)
        finally:
            self._syncing = False
        self._redraw()  # redraws with this player's POV (changes fog rendering)

    def _set_player_stars(self, pid: int, txt: str) -> None:
        if self._syncing:           # programmatic set_val — ignore
            return
        try:
            self.scenario.players[pid].stars = int(txt)
        except ValueError:
            self._set_status(f"P{pid} stars: invalid integer '{txt}'")
            return
        self._dirty = True
        self._redraw()

    def _set_turn(self, txt: str) -> None:
        if self._syncing:
            return
        try:
            self.scenario.turn = int(txt)
        except ValueError:
            self._set_status(f"turn: invalid integer '{txt}'")
            return
        self._dirty = True
        self._redraw()

    def _on_unit_flag_toggle(self, label: str) -> None:
        if label == "vet":
            self.unit_is_vet = not self.unit_is_vet
        elif label == "fortify":
            self.unit_fortify = not self.unit_fortify

    def _on_owner_change(self, label: str) -> None:
        if self._syncing:
            return
        # Owner sub-radio in Unit/City modes also drives the active player so
        # subsequent placements + fog POV stay consistent.
        self._syncing = True
        try:
            new_pid = int(label[1:])
            self.active_player = new_pid
            if "active_player" in self._widgets:
                self._widgets["active_player"].set_active(new_pid)
        finally:
            self._syncing = False
        self._redraw()

    def _on_unit_delete_armed(self, _event) -> None:
        self._unit_delete_armed = True
        self._set_status("Click a tile to delete its unit.")

    def _on_city_delete_armed(self, _event) -> None:
        self._city_delete_armed = True
        self._set_status("Click a tile to delete its city.")

    # ── Mutation + redraw boundary ────────────────────────────────────────────

    def _mutate(self, fn) -> None:
        """Run `fn` (a scenario mutator), mark dirty, and redraw.

        Any successful edit cancels pending Reset / Load arming — the user
        clearly moved on to something else.
        """
        try:
            fn()
        except Exception as e:
            self._set_status(f"Edit failed: {type(e).__name__}: {e}")
            return
        self._load_armed     = False
        self._reset_armed    = False
        self._generate_armed = False
        self._dirty = True
        self._redraw()

    def _redraw(self) -> None:
        """Rebuild Game from Scenario and repaint the board + info panel."""
        try:
            game = self.scenario.to_game()
        except Exception as e:
            self._set_status(f"to_game() failed: {type(e).__name__}: {e}")
            return

        env = _EditorEnv(game, *self.scenario.board_size)
        renderer = BoardRenderer(env)
        active_player_obj = game.players[self.active_player]
        uncovered = set(active_player_obj.uncovered_tile_ids)

        self.ax_board.clear()
        self.ax_info.clear()

        renderer.draw(
            ax        = self.ax_board,
            ax_info   = self.ax_info,
            uncovered = uncovered,
            title     = f"Editing: {self.scenario.name}"
                        + ("  *" if self._dirty else ""),
        )
        # Update the persistent stars textboxes if the values are stale (e.g.
        # after a fresh Scenario was assigned via Load). Guard with _syncing
        # so set_val's auto-fire of submit doesn't re-mark dirty.
        was_syncing = self._syncing
        self._syncing = True
        try:
            for pid in (0, 1):
                tb = self._widgets.get(f"p{pid}_stars")
                if tb is not None and tb.text != str(self.scenario.players[pid].stars):
                    tb.set_val(str(self.scenario.players[pid].stars))
        finally:
            self._syncing = was_syncing

        # Refresh the derived-stats line.
        parts = []
        for pid in range(self.scenario.n_players):
            p = game.players[pid]
            parts.append(
                f"P{pid} score={p.player_score_official} "
                f"spt={p.current_stars_per_turn}"
            )
        self._stats_text.set_text("  |  ".join(parts))

        self.fig.canvas.draw_idle()

    # ── Inspect mode helper ───────────────────────────────────────────────────

    def _update_inspect(self, x: int, y: int) -> None:
        s   = self.scenario
        tid = y * s.Ny + x
        terrain = s.map_grid[y][x].name
        road    = any(
            tuple(o["pos"]) == (x, y) and o.get("road")
            for o in s.tile_overrides
        )
        owner_u, unit = s.unit_at(x, y)
        owner_c, city = s.city_at(x, y)

        lines = [
            f"Tile ({x:2d}, {y:2d}) tile_id={tid}",
            f"  terrain : {terrain}",
            f"  road    : {road}",
        ]
        if unit is not None:
            lines.append(
                f"  unit    : P{owner_u} {unit.type.name} "
                f"hp={unit.hp if unit.hp is not None else '(default)'} "
                f"state={unit.state.name} vet={unit.is_vet}"
            )
        else:
            lines.append("  unit    : (none)")
        if city == "village":
            # city_at signals an unclaimed village with the literal "village" sentinel.
            lines.append("  city    : village (unclaimed)")
        elif city is not None:
            lines.append(
                f"  city    : P{owner_c} {city.level.name} "
                f"capital={city.is_capital} "
                f"choices={city.choices} "
                f"border_radius={city.border_radius}"
            )
        else:
            lines.append("  city    : (none)")

        # P0/P1 fog status for this tile
        for pid in range(s.n_players):
            unc = s.players[pid].uncovered
            if unc == "all":
                visible = True
            elif unc == "none":
                visible = False
            else:
                visible = (x, y) in [tuple(p) for p in unc]
            lines.append(f"  P{pid} sees : {visible}")

        if hasattr(self, "_inspect_text"):
            self._inspect_text.set_text("\n".join(lines))
            self.fig.canvas.draw_idle()

    # ── Status line ───────────────────────────────────────────────────────────

    def _set_status(self, text: str) -> None:
        self._status_text.set_text(text)
        self.fig.canvas.draw_idle()

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self) -> None:
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main(argv: Optional[list] = None) -> None:
    """Launch the editor on a blank scenario.

    A Load button (commit 4) will be the path for re-opening saved files.
    For commit 3, `python -m scenarios.editor` always starts blank.
    """
    EditorApp().run()


if __name__ == "__main__":
    main(sys.argv[1:])
