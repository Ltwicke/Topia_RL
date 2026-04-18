from game.enums import DefenseBonus, PlayerId, UnitState, UnitType


class Unit(object):
    """
    Base class for all units. unit_id is a random integer (0-9999) assigned by
    Game._new_unit_id(), guaranteed unique and never recycled within a game session.
    """
    def __init__(self, player_id: PlayerId, city, tile, unit_id: int):
        self.unit_id = unit_id
        self.player_id = player_id
        self.city = city
        self.tile = tile

        self.unit_type = None

        self.hp = None
        self.mvpts = None
        self.atk_stat = None
        self.def_stat = None
        self.def_bonus = DefenseBonus.NoBonus
        self.dash = False

        self.vision_range = 1
        self.attack_range = 1

        self.turn_state = UnitState.idle
        self.current_hp = None

    def set_ready(self):
        self.turn_state = UnitState.ready

    def __str__(self):
        return f"{self.unit_type.name} ({self.current_hp}/{self.hp}) in state {self.turn_state.name}"


class Warrior(Unit):
    def __init__(self, player_id: PlayerId, city, tile, unit_id: int):
        super().__init__(player_id, city, tile, unit_id)

        self.unit_type = UnitType.Warrior

        self.hp = 10
        self.atk_stat = 2
        self.def_stat = 2
        self.mvpts = 1

        self.current_hp = 10


class Rider(Unit):
    def __init__(self, player_id: PlayerId, city, tile, unit_id: int):
        super().__init__(player_id, city, tile, unit_id)

        self.unit_type = UnitType.Rider

        self.hp = 10
        self.atk_stat = 2
        self.def_stat = 1
        self.mvpts = 2

        self.current_hp = 10
