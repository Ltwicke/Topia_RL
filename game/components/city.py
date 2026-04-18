from dataclasses import dataclass, field
from game.enums import CityType, PlayerId


@dataclass
class City:
    """
    City/village data. Held by a Tile; does not reference the tile or any unit.
    Unit occupancy: read tile.unit (tile is the authority).
    Unit count tracking: maintained via current_n_units (units trained at this city).
    """
    tile_id: int
    player_id: PlayerId | None  # None = unclaimed village
    is_capital: bool = False
    under_siege: bool = False
    current_n_units: int = 0
    lvl: CityType = field(init=False)

    def __post_init__(self):
        self.lvl = CityType.village if self.player_id is None else CityType.city

    @property
    def max_unit_cap(self) -> int:
        return {CityType.village: 0, CityType.city: 3, CityType.lvl2_city: 6}[self.lvl]

    def capture(self, new_player_id: PlayerId) -> None:
        self.player_id = new_player_id
        self.lvl = CityType.city
        self.current_n_units = 1
        self.under_siege = False

    def upgrade(self) -> None:
        self.lvl = CityType.lvl2_city

    def seiging(self) -> None:
        pass  # reserved for income mechanics
