from game.enums import DefenseBonus, PlayerId, UnitState, UnitType

_UNIT_COSTS = {
    UnitType.Warrior:  2,
    UnitType.Rider:    4,
    UnitType.Archer:   4,
    UnitType.Knight:   16,
    UnitType.Catapult: 13,
    UnitType.Giant:    20,
    UnitType.Sword:    12,
    UnitType.Defender: 4,
}


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
        self.dash = False # Not needed; handled in advance turn state logic

        self.vision_range = 1
        self.attack_range = 1

        self.fortify = False    # True -> eligible for city Shield/Wall bonus
        self.stiff   = False    # True -> never deals retaliation damage

        self.turn_state = UnitState.idle
        self.current_hp = None
        
        self.is_vet = False
        self.kills = 0

    def set_ready(self):
        self.turn_state = UnitState.ready

    def set_idle(self):
        self.turn_state = UnitState.idle

    def __str__(self):
        return f"{self.unit_type.name} ({self.current_hp}/{self.hp}) in state {self.turn_state.name}"


class Warrior(Unit):
    def __init__(self, player_id: PlayerId, city, tile, unit_id: int):
        super().__init__(player_id, city, tile, unit_id)

        self.unit_type = UnitType.Warrior
        self.cost = 2

        self.hp = 10.0
        self.atk_stat = 2
        self.def_stat = 2
        self.mvpts = 1

        self.current_hp = 10.0

        self.fortify = True


class Rider(Unit):
    def __init__(self, player_id: PlayerId, city, tile, unit_id: int):
        super().__init__(player_id, city, tile, unit_id)

        self.unit_type = UnitType.Rider
        self.cost = 3

        self.hp = 10.0
        self.atk_stat = 2
        self.def_stat = 1
        self.mvpts = 2

        self.current_hp = 10.0

        self.fortify = True



class Knight(Unit):
    def __init__(self, player_id: PlayerId, city, tile, unit_id: int):
        super().__init__(player_id, city, tile, unit_id)

        self.unit_type = UnitType.Knight
        self.cost = 13

        self.hp = 10.0
        self.atk_stat = 3.5
        self.def_stat = 1
        self.mvpts = 3

        self.current_hp = 10.0

        self.fortify = True


class Giant(Unit):
    def __init__(self, player_id: PlayerId, city, tile, unit_id: int):
        super().__init__(player_id, city, tile, unit_id)

        self.unit_type = UnitType.Giant
        self.cost = 20

        self.hp = 40.0
        self.atk_stat = 5.0
        self.def_stat = 4.0
        self.mvpts = 1

        self.current_hp = 40.0



class Archer(Unit):
    def __init__(self, player_id: PlayerId, city, tile, unit_id: int):
        super().__init__(player_id, city, tile, unit_id)

        self.unit_type = UnitType.Archer
        self.cost = 3

        self.attack_range = 2

        self.hp = 10.0
        self.atk_stat = 2.0
        self.def_stat = 1.0
        self.mvpts = 1

        self.current_hp = 10.0

        self.fortify = True


class Catapult(Unit):
    def __init__(self, player_id: PlayerId, city, tile, unit_id: int):
        super().__init__(player_id, city, tile, unit_id)

        self.unit_type = UnitType.Catapult
        self.cost = 12

        self.attack_range = 3

        self.hp = 10.0
        self.atk_stat = 4.0
        self.def_stat = 0.0
        self.mvpts = 1

        self.current_hp = 10.0

        self.stiff = True



class Sword(Unit):
    def __init__(self, player_id: PlayerId, city, tile, unit_id: int):
        super().__init__(player_id, city, tile, unit_id)

        self.unit_type = UnitType.Sword
        self.cost = 5
        
        self.hp = 15.0
        self.atk_stat = 3.0
        self.def_stat = 3.0
        self.mvpts = 1

        self.current_hp = 15.0




class Defender(Unit):
    def __init__(self, player_id: PlayerId, city, tile, unit_id: int):
        super().__init__(player_id, city, tile, unit_id)

        self.unit_type = UnitType.Defender
        self.cost = 4

        self.hp = 15.0
        self.atk_stat = 1.0
        self.def_stat = 3.0
        self.mvpts = 1

        self.current_hp = 15.0

        self.fortify = True

