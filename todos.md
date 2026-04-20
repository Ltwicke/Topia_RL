# Current To-Dos and open design choices

This file lays down some important design choices for the todos that should be existentialized.

## Game simulator overhaul and additions

### Blocked city income during enemy seige

- When an enemy unit occupies a city, the income of this city is set to 0 as long as the unit occupies it

### Road mechanic

- Roads cannot be placed in enemy controlled territory
- Roads can be placed in visible territory for a cost of 4 stars
- If two adjacent tiles (also diagonal) both have roads, the edge weight between them gets updated to 0.5 
- a players units cannot use roads that lie within enemy territory (player cntrl) 

 
### Inclusion of score

 - Score is awarded for cities under control, units under control and controlled tiles. 
 - This score is visible to every player in the game.
 
### Addition of multiple new units 
 
 - A giant can never be created in a city other than via the city upgrade mechanic. 


### Upgrade City

 - Workshop vs Explorer; A workshop means one more stars per turn for the city, an explorer is a special unit that is created on upgrade and moves 14 moves in random directions, irregardless of enemy zoc, occupied tiles, only by water DONE
 - after this, the player can choose between a park, or a super_unit (su for short); the park functions exactly like a workshop and choosing su, will invoke CreatUnit and create a Giant on the city (also takes up one space of the city). DONE


## Environment wrapper

### Moving the renderer to its own python file

- Currently the render function is build into the EnvWrapper, but it is probably better to isolate it into its own module. 






