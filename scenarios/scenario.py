"""
scenarios/scenario.py
──────────────────────────────────────────────────────────────────────────────
In-memory representation of a hand-authored game scenario.

A `Scenario` is the editable source of truth. The editor mutates it directly;
`to_game()` materialises a real `Game` for the renderer (and, later, for the
eval harness). `to_yaml()` / `from_yaml()` round-trip the Scenario through a
human-readable file format used by the editor's Save/Load buttons and by the
(future) eval harness's fixture loader.

Coordinate convention
─────────────────────
Throughout the YAML schema and the dataclass API, positions are written as
`pos: [x, y]` where:

    x = column index, 0..Ny-1   (left-to-right on the rendered board)
    y = row index,    0..Nx-1   (top-to-bottom in the ASCII map; y=0 is top)

Internally, the game indexes tiles as `tile_id = y * Ny + x`. The map grid is
stored row-major: `map_grid[y][x]`.

`board_size` is `[Nx, Ny] = [n_rows, n_cols]`, matching the game's
`board.board_size` convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Tuple

import yaml

from game.enums import (
    BoardType,
    CityType,
    DefenseBonus,
    PlayerId,
    TileType,
    Tribes,
    UnitState,
    UnitType,
)
from game.game import Game
from game.components.city import City
from game.components.units import (
    Archer,
    Catapult,
    Defender,
    Giant,
    Knight,
    Rider,
    Sword,
    Warrior,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helper tables
# ══════════════════════════════════════════════════════════════════════════════

# UnitType → concrete subclass. Building units bypasses CreateUnit logic so the
# editor can place arbitrary configurations directly.
_UNIT_CLASS_MAP: dict[UnitType, type] = {
    UnitType.Warrior:  Warrior,
    UnitType.Rider:    Rider,
    UnitType.Archer:   Archer,
    UnitType.Knight:   Knight,
    UnitType.Catapult: Catapult,
    UnitType.Giant:    Giant,
    UnitType.Sword:    Sword,
    UnitType.Defender: Defender,
}


def _level_to_times_upgraded(level: CityType) -> int:
    """
    Default times_upgraded for a given CityType, assuming the city was built
    via the standard upgrade chain. Editor authors can override per-city.
    """
    return {
        CityType.village:          0,
        CityType.lvl1:             0,
        CityType.lvl2_workshop:    1,
        CityType.lvl2_explorer:    1,
        CityType.lvl3_resources:   2,
        CityType.lvl3_wall:        2,
        CityType.lvl4_popgrwth:    3,
        CityType.lvl4_bordergrwth: 3,
        CityType.lvl5_su:          4,
        CityType.lvl5_park:        4,
        CityType.lvl6_su:          5,
        CityType.lvl6_park:        5,
        CityType.lvl7_su:          6,
        CityType.lvl7_park:        6,
        CityType.lvl8plus:         7,
    }[level]


# ══════════════════════════════════════════════════════════════════════════════
# Nested dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioUnit:
    pos:       Tuple[int, int]
    type:      UnitType
    hp:        Optional[float] = None        # None → unit class max HP
    state:     UnitState        = UnitState.ready
    is_vet:    bool             = False
    fortify:   Optional[bool]   = None       # None → unit class default
    def_bonus: DefenseBonus     = DefenseBonus.NoBonus
    kills:     int              = 0


@dataclass
class ScenarioCity:
    pos:            Tuple[int, int]
    level:          CityType        = CityType.lvl1
    is_capital:     bool            = False
    choices:        List[int]       = field(default_factory=list)
    times_upgraded: Optional[int]   = None   # None → derived from level
    border_radius:  int             = 1      # 1 = standard; 2 = lvl4_bordergrwth+


# `uncovered` accepts: 'all', 'none', or list of [x, y] pairs
UncoveredSpec = Union[str, List[Tuple[int, int]]]


@dataclass
class ScenarioPlayer:
    id:        int
    stars:     int                    = 0
    uncovered: UncoveredSpec          = "all"
    cities:    List[ScenarioCity]     = field(default_factory=list)
    units:     List[ScenarioUnit]     = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Scenario
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    name:            str                          = "untitled"
    description:     str                          = ""
    board_size:      Tuple[int, int]              = (11, 11)        # (Nx, Ny)
    default_terrain: TileType                     = TileType.field
    n_players:       int                          = 2
    player_tribes:   List[Tribes]                 = field(
        default_factory=lambda: [Tribes.Omaji, Tribes.Imperius]
    )
    current_player:  int                          = 0
    turn:            int                          = 0
    map_grid:        List[List[TileType]]         = field(default_factory=list)
    tile_overrides:  List[dict]                   = field(default_factory=list)
    players:         List[ScenarioPlayer]         = field(default_factory=list)
    # Unclaimed villages — Top-level because villages have no player owner
    # (`tile.city.player_id is None`) and don't belong in any player's
    # `cities_under_control`. Stored as a list of `(x, y)` positions.
    villages:        List[Tuple[int, int]]        = field(default_factory=list)

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def blank(cls, size: Tuple[int, int] = (11, 11), n_players: int = 2) -> "Scenario":
        """A blank canvas: full default-terrain board, two empty players."""
        Nx, Ny = size
        return cls(
            board_size=size,
            n_players=n_players,
            map_grid=[[TileType.field for _ in range(Ny)] for _ in range(Nx)],
            players=[ScenarioPlayer(id=i) for i in range(n_players)],
        )

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def Nx(self) -> int: return self.board_size[0]

    @property
    def Ny(self) -> int: return self.board_size[1]

    def _tile_id(self, x: int, y: int) -> int:
        return y * self.Ny + x

    # ── Snapshot a Game back into a Scenario ─────────────────────────────────

    @classmethod
    def from_game(
        cls,
        game:        Game,
        name:        str = "captured",
        description: str = "",
    ) -> "Scenario":
        """
        Inverse of `to_game`. Walks the live `Game` and reconstructs an
        equivalent `Scenario`. Used by:

          • the editor's "Generate random" feature (a freshly randomised Game
            is snapshotted into a Scenario the user can then edit)
          • later: the eval harness's "snapshot any in-progress game" path

        Caveats:
          • `border_radius` is always emitted as 1 — we do not reverse-engineer
            the painted radius from `controlled_tile_ids`.
          • `tile_overrides` only carries roads on tiles that do NOT also hold
            a city (cities always carry road; surfacing those would be noise).
        """
        board    = game.game_board
        Nx, Ny   = board.board_size
        n_pl     = game.n_players

        # Terrain grid (row-major, indexed by y → row).
        map_grid: List[List[TileType]] = [
            [TileType.field for _ in range(Ny)] for _ in range(Nx)
        ]
        for tile in board.board:
            r, c = board.int_to_tup[tile.id]
            map_grid[r][c] = tile.tile_type

        # Per-tile road overrides (only tiles without an owned/village city).
        tile_overrides: List[dict] = []
        for tile in board.board:
            if tile.has_road and tile.city is None:
                r, c = board.int_to_tup[tile.id]
                tile_overrides.append({"pos": [c, r], "road": True})

        # Unclaimed villages (player_id is None).
        villages: List[Tuple[int, int]] = []
        for tile in board.board:
            if tile.city is not None and tile.city.player_id is None:
                r, c = board.int_to_tup[tile.id]
                villages.append((c, r))

        all_tiles = Nx * Ny
        sps: List[ScenarioPlayer] = []
        for pid in range(n_pl):
            player = game.players[pid]
            # uncovered: collapse to 'all'/'none' if applicable
            unc_set = set(player.uncovered_tile_ids)
            if len(unc_set) == all_tiles:
                uncovered: UncoveredSpec = "all"
            elif len(unc_set) == 0:
                uncovered = "none"
            else:
                uncovered = sorted(
                    [(tid % Ny, tid // Ny) for tid in unc_set]
                )

            sc_cities: List[ScenarioCity] = []
            for city in player.cities_under_control:
                r, c = board.int_to_tup[city.tile_id]
                sc_cities.append(ScenarioCity(
                    pos            = (c, r),
                    level          = city.lvl,
                    is_capital     = bool(city.is_capital),
                    choices        = list(city.choices),
                    times_upgraded = int(city.times_upgraded),
                    border_radius  = 1,        # not reverse-engineered
                ))

            sc_units: List[ScenarioUnit] = []
            for unit in player.units_under_control.values():
                r, c = board.int_to_tup[unit.tile.id]
                # Don't fold the +5 vet bump back into su.hp — leave it
                # implicit so to_game's vet bump reproduces the same max.
                # If the unit was damaged below class default, capture it;
                # otherwise leave hp=None (means "use new max").
                explicit_hp: Optional[float] = None
                base_hp = unit.hp - 5.0 if unit.is_vet else unit.hp
                if unit.current_hp != unit.hp:
                    explicit_hp = float(unit.current_hp)
                sc_units.append(ScenarioUnit(
                    pos       = (c, r),
                    type      = unit.unit_type,
                    hp        = explicit_hp,
                    state     = unit.turn_state,
                    is_vet    = bool(unit.is_vet),
                    fortify   = bool(unit.fortify),
                    def_bonus = unit.def_bonus,
                    kills     = int(unit.kills),
                ))

            sps.append(ScenarioPlayer(
                id        = pid,
                stars     = int(player.stars),
                uncovered = uncovered,
                cities    = sc_cities,
                units     = sc_units,
            ))

        return cls(
            name            = name,
            description     = description,
            board_size      = (Nx, Ny),
            default_terrain = TileType.field,
            n_players       = n_pl,
            player_tribes   = [p.tribe for p in game.players],
            current_player  = int(game.player_go_id),
            turn            = int(game.turn),
            map_grid        = map_grid,
            tile_overrides  = tile_overrides,
            players         = sps,
            villages        = villages,
        )

    # ── Build a real Game from the scenario ───────────────────────────────────

    def to_game(self) -> Game:
        """
        Materialise this scenario as a live `Game` instance ready for the
        renderer or, later, the eval harness.

        Build sequence:
          1. Construct Game(board_config, tribes); call reset_game(). This
             produces a fully-randomised board (capitals, terrain, starting
             units) we will overwrite.
          2. Wipe random units, cities, control, roads from every tile and
             every player's collections.
          3. Apply the scenario's terrain grid → tile.tile_type.
          4. Apply tile overrides → tile.has_road.
          5. For each ScenarioPlayer: stars, uncovered_tile_ids, cities (with
             default 1-radius borders), units (registered in
             game._used_unit_ids).
          6. Set game.player_go_id = current_player; game.turn = turn.
          7. Rebuild board_graph + each player.partial_graph.

        Returns the live Game. Does not mutate `self`.
        """
        Nx, Ny = self.Nx, self.Ny

        # 1. Build a Game and let it self-initialise (BoardType is a placeholder
        #    — terrain is overwritten in step 3 below).
        board_config = {
            "board_size": [Nx, Ny],
            "board_type": BoardType.Drylands,
            "n_players":  self.n_players,
        }
        game = Game(board_config, list(self.player_tribes))
        game.reset_game()
        board = game.game_board

        # 2. Wipe everything reset_game() generated.
        for tile in board.board:
            tile.unit        = None
            tile.city        = None
            tile.cntrl       = None
            tile.has_road    = False
            # tile.tile_status retained at no_status default
        for player in game.players:
            player.units_under_control  = {}
            player.cities_under_control = []
        game._used_unit_ids = set()

        # 3. Apply terrain grid (map_grid is row-major; row index = y).
        for y in range(Nx):
            for x in range(Ny):
                board.board[self._tile_id(x, y)].tile_type = self.map_grid[y][x]

        # 4. Apply tile overrides (currently: roads).
        for ov in self.tile_overrides:
            x, y = ov["pos"]
            tid = self._tile_id(x, y)
            if "road" in ov:
                board.board[tid].has_road = bool(ov["road"])

        # 5. Per-player state.
        for sp in self.players:
            if sp.id >= len(game.players):
                # Scenario carries more players than the Game allocated; ignore.
                continue
            player = game.players[sp.id]
            player.stars = int(sp.stars)
            player.uncovered_tile_ids = self._resolve_uncovered(sp.uncovered)

            # Cities — paint tile.cntrl according to each city's border_radius.
            for sc in sp.cities:
                self._place_city(sc, sp.id, player, game)

            # Units — depend on cities existing (for the home_city pointer).
            for su in sp.units:
                self._place_unit(su, sp.id, player, game)

        # 5b. Unclaimed villages (player_id=None, no border, no cntrl paint).
        for pos in self.villages:
            self._place_village(pos, game)

        # 6. Game-level state.
        game.player_go_id = int(self.current_player)
        game.turn         = int(self.turn)

        # 7. Rebuild graphs for renderer / policy consumption.
        board._update_road_edge_weights()
        board.create_board_graph_from_board_state(game.all_tile_ids)
        for player in game.players:
            player.construct_partial_graph_2players(board)

        return game

    # ── Helpers used by to_game() ─────────────────────────────────────────────

    def _resolve_uncovered(self, spec: UncoveredSpec) -> set:
        """Translate the YAML-friendly uncovered spec into a set of tile_ids."""
        if spec == "all":
            return set(range(self.Nx * self.Ny))
        if spec == "none":
            return set()
        # Otherwise: explicit list of [x, y] pairs.
        return {self._tile_id(int(x), int(y)) for x, y in spec}

    def _place_city(
        self,
        sc:     ScenarioCity,
        pid:    int,
        player,
        game:   Game,
    ) -> None:
        x, y       = sc.pos
        tid        = self._tile_id(x, y)
        board      = game.game_board
        tile       = board.board[tid]

        city = City(
            tile_id   = tid,
            player_id = PlayerId(pid),
            is_capital= sc.is_capital,
        )
        # Override post-init defaults.
        city.lvl     = sc.level
        city.choices = list(sc.choices)
        city.times_upgraded = (
            sc.times_upgraded
            if sc.times_upgraded is not None
            else _level_to_times_upgraded(sc.level)
        )
        # Pad choices with 0 (workshop / SU) up to times_upgraded so
        # `city_stars_per_turn` does not index past the end. Authors who care
        # about specific choice values can specify them explicitly.
        while len(city.choices) < city.times_upgraded:
            city.choices.append(0)

        tile.city     = city
        tile.has_road = True       # cities are always on roads in the codebase
        player.cities_under_control.append(city)

        if sc.is_capital:
            player.capital_id              = tid
            board.capital_tile_ids[pid]    = tid

        # Border painting — radius from sc.border_radius (1 = standard,
        # 2 = lvl4_bordergrwth+). Only paints tiles with no existing cntrl
        # to avoid overwriting an adjacent city's territory.
        r = max(1, int(sc.border_radius))
        row, col = tid // self.Ny, tid % self.Ny
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.Nx and 0 <= nc < self.Ny:
                    ntid = nr * self.Ny + nc
                    if board.board[ntid].cntrl is None:
                        board.board[ntid].cntrl = PlayerId(pid)
                        city.controlled_tile_ids.append(ntid)

    def _place_village(self, pos: Tuple[int, int], game: Game) -> None:
        """
        Place an unclaimed village at `pos`. Villages have `player_id=None`,
        do not paint any cntrl, and are not added to any player's
        `cities_under_control` (matching `Board.initialize`'s behaviour).
        """
        x, y  = pos
        tid   = self._tile_id(x, y)
        tile  = game.game_board.board[tid]
        # Skip if a player city already occupies this tile — owned cities win.
        if tile.city is not None and tile.city.player_id is not None:
            return
        tile.city     = City(tile_id=tid, player_id=None, is_capital=False)
        tile.has_road = True

    def _place_unit(
        self,
        su:     ScenarioUnit,
        pid:    int,
        player,
        game:   Game,
    ) -> None:
        x, y     = su.pos
        tid      = self._tile_id(x, y)
        tile     = game.game_board.board[tid]

        unit_cls = _UNIT_CLASS_MAP[su.type]
        uid      = game._new_unit_id()
        # Attach to first owned city, or None if player has none — the unit
        # class only needs the reference for default-state purposes.
        home_city = (
            player.cities_under_control[0]
            if player.cities_under_control else None
        )

        unit = unit_cls(
            player_id = PlayerId(pid),
            city      = home_city,
            tile      = tile,
            unit_id   = uid,
        )

        # Veteran promotion bumps the unit's max HP by +5 (game convention).
        # Apply BEFORE honoring an explicit `su.hp`, so a user-provided
        # current HP value still wins.
        unit.is_vet = bool(su.is_vet)
        if unit.is_vet:
            unit.hp        = unit.hp + 5.0
            unit.current_hp = unit.hp           # default to new max if no override

        # Override class defaults from scenario spec.
        if su.hp is not None:
            unit.current_hp = float(su.hp)
        unit.turn_state = su.state
        if su.fortify is not None:
            unit.fortify = bool(su.fortify)
        unit.def_bonus  = su.def_bonus
        unit.kills      = int(su.kills)

        tile.unit = unit
        player.units_under_control[uid] = unit

    # ── Editor mutators (live state edits) ────────────────────────────────────

    def _scenario_player(self, player_id: int) -> "ScenarioPlayer":
        for p in self.players:
            if p.id == player_id:
                return p
        raise KeyError(f"No ScenarioPlayer with id={player_id}")

    # Terrain ──────────────────────────────────────────────────────────────────

    def set_terrain(self, x: int, y: int, terrain: TileType) -> None:
        self.map_grid[y][x] = terrain

    # Roads ────────────────────────────────────────────────────────────────────

    def toggle_road(self, x: int, y: int) -> None:
        """Toggle a road overlay on tile (x, y). Maintained in tile_overrides."""
        pos = (int(x), int(y))
        for ov in self.tile_overrides:
            if tuple(ov["pos"]) == pos:
                ov["road"] = not ov.get("road", False)
                # If the override carries no useful info, drop it.
                if not ov.get("road") and len(ov) == 2:   # {pos, road}
                    self.tile_overrides.remove(ov)
                return
        self.tile_overrides.append({"pos": list(pos), "road": True})

    # Units ────────────────────────────────────────────────────────────────────

    def set_unit(
        self,
        x:     int,
        y:     int,
        owner: int,
        **unit_fields,
    ) -> None:
        """Place (or replace) a unit at (x, y) owned by `owner`.

        Any pre-existing unit on (x, y), regardless of owner, is removed first.
        `unit_fields` is forwarded to `ScenarioUnit(...)`.
        """
        self.delete_unit(x, y)
        self._scenario_player(owner).units.append(
            ScenarioUnit(pos=(int(x), int(y)), **unit_fields)
        )

    def delete_unit(self, x: int, y: int) -> None:
        pos = (int(x), int(y))
        for sp in self.players:
            sp.units = [u for u in sp.units if u.pos != pos]

    def unit_at(self, x: int, y: int) -> Tuple[Optional[int], Optional[ScenarioUnit]]:
        """Return (owner_id, ScenarioUnit) at (x, y), or (None, None)."""
        pos = (int(x), int(y))
        for sp in self.players:
            for u in sp.units:
                if u.pos == pos:
                    return sp.id, u
        return None, None

    # Cities ───────────────────────────────────────────────────────────────────

    def set_city(
        self,
        x:     int,
        y:     int,
        owner: int,
        **city_fields,
    ) -> None:
        self.delete_city(x, y)
        self._scenario_player(owner).cities.append(
            ScenarioCity(pos=(int(x), int(y)), **city_fields)
        )

    def delete_city(self, x: int, y: int) -> None:
        """Remove any city OR village at (x, y), regardless of owner."""
        pos = (int(x), int(y))
        for sp in self.players:
            sp.cities = [c for c in sp.cities if c.pos != pos]
        self.villages = [v for v in self.villages if tuple(v) != pos]

    def city_at(self, x: int, y: int) -> Tuple[Optional[int], Optional[ScenarioCity]]:
        """
        Return (owner_id, ScenarioCity) for an owned city at (x, y), or
        (None, None) if empty. Unclaimed villages are signalled by
        `(None, "village")` so callers can tell them apart from "nothing".
        """
        pos = (int(x), int(y))
        for sp in self.players:
            for c in sp.cities:
                if c.pos == pos:
                    return sp.id, c
        if pos in [tuple(v) for v in self.villages]:
            return None, "village"   # sentinel: village w/o owner
        return None, None

    # Villages ─────────────────────────────────────────────────────────────────

    def set_village(self, x: int, y: int) -> None:
        """Place (or de-duplicate) an unclaimed village at (x, y).

        Removes any pre-existing owned city at this position first, so
        switching a tile from "P0 city" to "village" is a single call.
        """
        pos = (int(x), int(y))
        # Drop owned cities at this position (delete_city also drops villages,
        # so we re-add after).
        self.delete_city(x, y)
        self.villages.append(pos)

    def delete_village(self, x: int, y: int) -> None:
        pos = (int(x), int(y))
        self.villages = [v for v in self.villages if tuple(v) != pos]

    # Fog of war ───────────────────────────────────────────────────────────────

    def toggle_fog(self, x: int, y: int, player_id: int) -> None:
        """Flip the uncovered status of tile (x, y) for `player_id`.

        The 'all'/'none' shorthand is materialised on demand so the toggle
        only ever flips a single tile.
        """
        sp  = self._scenario_player(player_id)
        pos = (int(x), int(y))
        if sp.uncovered == "all":
            sp.uncovered = [
                (c, r)
                for r in range(self.Nx)
                for c in range(self.Ny)
                if (c, r) != pos
            ]
        elif sp.uncovered == "none":
            sp.uncovered = [pos]
        else:
            current = list(sp.uncovered)
            if pos in current:
                current.remove(pos)
            else:
                current.append(pos)
            sp.uncovered = current

    def cover_all(self, player_id: int) -> None:
        self._scenario_player(player_id).uncovered = "none"

    def uncover_all(self, player_id: int) -> None:
        self._scenario_player(player_id).uncovered = "all"

    # ── YAML round-trip ───────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Scenario":
        """Load a Scenario from a YAML file (see schema in module docstring)."""
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        return _scenario_from_dict(doc)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Persist this Scenario to YAML. Uses block style with literal-block
        ASCII map. Defaults are omitted to keep files small."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = _scenario_to_dict(self)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                doc, f,
                Dumper          = _ScenarioYamlDumper,
                sort_keys       = False,
                allow_unicode   = True,
                default_flow_style = False,
                width           = 200,         # avoid hard-wrapping list lines
            )


# ══════════════════════════════════════════════════════════════════════════════
# YAML serialisation helpers
# ══════════════════════════════════════════════════════════════════════════════

# ASCII glyphs for terrain in the `map:` block.
_TILE_GLYPHS: dict[TileType, str] = {
    TileType.field:      ".",
    TileType.water:      "~",
    TileType.deep_water: "=",
    TileType.mountain:   "^",
}
_GLYPH_TO_TILE: dict[str, TileType] = {v: k for k, v in _TILE_GLYPHS.items()}


# Custom Dumper:
#   • literal block style for multi-line strings (the map grid)
#   • flow style for short scalar lists (`pos`, `board_size`, `choices`, …)
#     so files stay scannable instead of exploding `pos` onto three lines.
class _ScenarioYamlDumper(yaml.SafeDumper):
    pass


def _str_representer(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def _list_representer(dumper, data):
    # Flow style only for short lists of plain scalars.
    is_short_scalar_list = (
        len(data) <= 6
        and all(isinstance(x, (int, float, bool, str)) for x in data)
    )
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data,
        flow_style=is_short_scalar_list,
    )


_ScenarioYamlDumper.add_representer(str,  _str_representer)
_ScenarioYamlDumper.add_representer(list, _list_representer)


# ── Encoding ──────────────────────────────────────────────────────────────────

def _grid_to_ascii(grid: List[List[TileType]]) -> str:
    """Encode a row-major terrain grid as a multi-line ASCII string."""
    return "\n".join("".join(_TILE_GLYPHS[t] for t in row) for row in grid) + "\n"


def _scenario_unit_to_dict(u: ScenarioUnit) -> dict:
    out: dict = {"pos": list(u.pos), "type": u.type.name}
    if u.hp is not None:
        out["hp"] = float(u.hp)
    if u.state != UnitState.ready:
        out["state"] = u.state.name
    if u.is_vet:
        out["is_vet"] = True
    if u.fortify is not None:
        out["fortify"] = bool(u.fortify)
    if u.def_bonus != DefenseBonus.NoBonus:
        out["def_bonus"] = u.def_bonus.name
    if u.kills:
        out["kills"] = int(u.kills)
    return out


def _scenario_city_to_dict(c: ScenarioCity) -> dict:
    out: dict = {"pos": list(c.pos), "level": c.level.name}
    if c.is_capital:
        out["is_capital"] = True
    if c.choices:
        out["choices"] = list(c.choices)
    if c.times_upgraded is not None:
        out["times_upgraded"] = int(c.times_upgraded)
    if c.border_radius != 1:           # default is 1; only emit overrides
        out["border_radius"] = int(c.border_radius)
    return out


def _scenario_player_to_dict(p: ScenarioPlayer) -> dict:
    out: dict = {"id": p.id, "stars": p.stars}
    if isinstance(p.uncovered, str):
        out["uncovered"] = p.uncovered
    else:
        out["uncovered"] = [list(xy) for xy in p.uncovered]
    if p.cities:
        out["cities"] = [_scenario_city_to_dict(c) for c in p.cities]
    if p.units:
        out["units"] = [_scenario_unit_to_dict(u) for u in p.units]
    return out


def _scenario_to_dict(s: Scenario) -> dict:
    """Serialise a Scenario to a plain-dict structure ready for yaml.dump."""
    out: dict = {
        "name":            s.name,
        "description":     s.description,
        "board_size":      list(s.board_size),
        "default_terrain": s.default_terrain.name,
        "n_players":       s.n_players,
        "player_tribes":   [t.name for t in s.player_tribes],
        "current_player":  s.current_player,
        "turn":            s.turn,
        "map":             _grid_to_ascii(s.map_grid),
    }
    if s.tile_overrides:
        out["tiles"] = [
            {"pos": list(ov["pos"]),
             **{k: v for k, v in ov.items() if k != "pos"}}
            for ov in s.tile_overrides
        ]
    if s.villages:
        out["villages"] = [list(v) for v in s.villages]
    out["players"] = [_scenario_player_to_dict(p) for p in s.players]
    return out


# ── Decoding ──────────────────────────────────────────────────────────────────

def _ascii_to_grid(
    ascii_str: str,
    Nx:        int,
    Ny:        int,
    default:   TileType,
) -> List[List[TileType]]:
    """Inverse of `_grid_to_ascii`. Short rows / unknown chars fall back to default."""
    rows = ascii_str.rstrip("\n").split("\n")
    grid: List[List[TileType]] = []
    for r in range(Nx):
        line = rows[r] if r < len(rows) else ""
        row: List[TileType] = []
        for c in range(Ny):
            ch = line[c] if c < len(line) else None
            if ch is None or ch == " ":
                row.append(default)
            else:
                row.append(_GLYPH_TO_TILE.get(ch, default))
        grid.append(row)
    return grid


def _dict_to_scenario_unit(d: dict) -> ScenarioUnit:
    return ScenarioUnit(
        pos       = tuple(d["pos"]),
        type      = UnitType[d["type"]],
        hp        = float(d["hp"]) if "hp" in d else None,
        state     = UnitState[d.get("state", "ready")],
        is_vet    = bool(d.get("is_vet", False)),
        fortify   = bool(d["fortify"]) if "fortify" in d else None,
        def_bonus = DefenseBonus[d.get("def_bonus", "NoBonus")],
        kills     = int(d.get("kills", 0)),
    )


def _dict_to_scenario_city(d: dict) -> ScenarioCity:
    return ScenarioCity(
        pos            = tuple(d["pos"]),
        level          = CityType[d["level"]],
        is_capital     = bool(d.get("is_capital", False)),
        choices        = list(d.get("choices", [])),
        times_upgraded = int(d["times_upgraded"]) if "times_upgraded" in d else None,
        border_radius  = int(d.get("border_radius", 1)),
    )


def _dict_to_scenario_player(d: dict) -> ScenarioPlayer:
    raw_unc = d.get("uncovered", "all")
    if isinstance(raw_unc, str):
        uncovered: UncoveredSpec = raw_unc
    else:
        uncovered = [tuple(xy) for xy in raw_unc]

    return ScenarioPlayer(
        id        = int(d["id"]),
        stars     = int(d.get("stars", 0)),
        uncovered = uncovered,
        cities    = [_dict_to_scenario_city(c) for c in d.get("cities", [])],
        units     = [_dict_to_scenario_unit(u) for u in d.get("units", [])],
    )


def _scenario_from_dict(d: dict) -> Scenario:
    """Inverse of `_scenario_to_dict`. Tolerates missing optional fields."""
    Nx, Ny     = int(d["board_size"][0]), int(d["board_size"][1])
    default_tt = TileType[d.get("default_terrain", "field")]
    map_grid   = _ascii_to_grid(d.get("map", ""), Nx, Ny, default_tt)

    tile_overrides = [
        {"pos": tuple(ov["pos"]),
         **{k: v for k, v in ov.items() if k != "pos"}}
        for ov in d.get("tiles", [])
    ]

    villages = [tuple(v) for v in d.get("villages", [])]

    return Scenario(
        name            = d.get("name", "untitled"),
        description     = d.get("description", ""),
        board_size      = (Nx, Ny),
        default_terrain = default_tt,
        n_players       = int(d.get("n_players", 2)),
        player_tribes   = [Tribes[t] for t in d.get("player_tribes",
                                                     ["Omaji", "Imperius"])],
        current_player  = int(d.get("current_player", 0)),
        turn            = int(d.get("turn", 0)),
        map_grid        = map_grid,
        tile_overrides  = tile_overrides,
        players         = [_dict_to_scenario_player(p)
                           for p in d.get("players", [])],
        villages        = villages,
    )
