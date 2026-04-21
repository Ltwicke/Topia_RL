# Current To-Dos and open design choices

This file lays down some important design choices for the todos that should be existentialized.

## Game simulator overhaul and additions

Everything up to date so far!

### Attack interactions with ranged units

- Ranged units get no retaliation damage when attacking a non-ranged unit at distance.
- Ranged units furthermore dont move in the place of a killed defender.
- Include stiff mechanic for units that do not retaliate.

### Defensive Bonus mechanics

- Some units dont have fortify and therefore dont benefit from defensive boni from the city (but still from mountains)

### Road mechanics

- roads cannot be build onto mountains, and mountains end the movement.


## Environment wrapper

### Moving the renderer to its own python file

- Currently the render function is build into the EnvWrapper, but it is probably better to isolate it into its own module. 


### More complex UI on the horizon

- In the long term, the goal is that the user can play against a trained agent. For this, a more refined UI will be needed. But this is something for the future.





