
from game.enums import CityType, PlayerId



class City(object):
    """
    City object. If instantiated with player_id = None it is an unclaimed village. Treat village and city different by checking for player_id.
    hi
    """
    def __init__(self, player_id: PlayerId, tile_id: Int, is_capital=False, unit=None):
        if player_id == None:
            self.lvl = CityType.village
        else:
            self.lvl = CityType.city

        self.player_id = player_id
        self.tile_id = tile_id
        self.is_capital = is_capital
        self.unit = unit
        self.under_seige = False
        self.max_unit_cap = 3
        self.current_n_units = 1

    def capture(self, new_player_id: PlayerId):
        self.player_id = new_player_id
        self.current_n_units = 1
        self.lvl = CityType.city

    def seiging(self):
        # no income
        pass

    def upgrade(self):
        self.lvl = CityType.lvl2_city

    


