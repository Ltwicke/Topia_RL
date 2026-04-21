import numpy as np
from dataclasses import dataclass, field
from game.enums import CityType, PlayerId



_CITY_UPGRADE_COST = {
    CityType.lvl2_workshop :    5,
    CityType.lvl2_explorer :    5,
    CityType.lvl3_resources :   8,
    CityType.lvl3_wall :        8,
    CityType.lvl4_popgrwth :    12,
    CityType.lvl4_bordergrwth : 12,
    CityType.lvl5_su :          16,
    CityType.lvl5_park :        16,
    CityType.lvl6_su :          18,
    CityType.lvl6_park :        18,
    CityType.lvl7_su :          22,
    CityType.lvl7_park :        22,
    CityType.lvl8plus :         30,
}


_CITY_UPGRADES = {
    CityType.lvl1 :             (CityType.lvl2_workshop, CityType.lvl2_explorer),
    CityType.lvl2_workshop :    (CityType.lvl3_resources, CityType.lvl3_wall),
    CityType.lvl2_explorer :    (CityType.lvl3_resources, CityType.lvl3_wall),
    CityType.lvl3_resources :   (CityType.lvl4_popgrwth, CityType.lvl4_bordergrwth),
    CityType.lvl3_wall :        (CityType.lvl4_popgrwth, CityType.lvl4_bordergrwth),
    CityType.lvl4_popgrwth :    (CityType.lvl5_su, CityType.lvl5_park),
    CityType.lvl4_bordergrwth : (CityType.lvl5_su, CityType.lvl5_park),
    CityType.lvl5_su :          (CityType.lvl6_su, CityType.lvl6_park),
    CityType.lvl5_park :        (CityType.lvl6_su, CityType.lvl6_park),
    CityType.lvl6_su :          (CityType.lvl7_su, CityType.lvl7_park),
    CityType.lvl6_park :        (CityType.lvl7_su, CityType.lvl7_park),
    CityType.lvl7_su :          (CityType.lvl8plus, CityType.lvl8plus),
    CityType.lvl7_park :        (CityType.lvl8plus, CityType.lvl8plus),
    CityType.lvl8plus :         (CityType.lvl8plus, CityType.lvl8plus),
}

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
        self.lvl = CityType.village if self.player_id is None else CityType.lvl1
        self.times_upgraded = 0
        self.choices = []
        self.pending_discount = 0
        self.controlled_tile_ids: list[int] = []

    @property
    def max_unit_cap(self) -> int:
        if self.player_id is None:  # village
            return 0
        return self.times_upgraded + 2 # 2 IS THE CORRECT VALUE HERE!

    @property
    def city_stars_per_turn(self) -> int:
        if self.under_siege:
            return 0
        spt = 1
        if self.is_capital:
            spt += 1 #capital produces one more star
        spt += self.times_upgraded
        if self.times_upgraded > 0:
            if self.choices[0] == 0: # workshop
                spt += 1
        if self.times_upgraded > 3: # 4th upgrade (index 3) is the first park/su choice
            spt += np.sum(self.choices[3:]) # park = choice 1; sum from first park choice
        return spt
            

    def capture(self, new_player_id: PlayerId) -> None:
        if self.lvl == CityType.village: # if a village is captured, make it to a lvl1
            self.lvl = CityType.lvl1
        self.player_id = new_player_id
        self.current_n_units = 1
        self.under_siege = False
        self.pending_discount = 0

    def upgrade(self, choice=0) -> None:
        new_lvl = _CITY_UPGRADES[self.lvl][choice]
        self.lvl = new_lvl
        self.times_upgraded += 1
        self.choices.append(choice)

    def seiging(self) -> None:
        ## useless function, just modify city attribute directly...
        self.under_siege = True
