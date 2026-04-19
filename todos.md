# Current To-Dos and open design choices

This file lays down some important design choices for the todos that should be existentialized.

## Game simulator refactorization

### Player partial graph

- The switching of the dimensions of the partial graph generation in the player class is still hard-coded as opposed to being controlled by the slices from the enums.py file 
- Which dimensions to switch is a tricky question and should be human-controlled. Therefore, it would be very interesting to control which dimensions are getting swapped for player 2. 

### Road mechanic

- Roads can be placed on a visible tile by any player. It will therefore be included in the tile featurization.
- If two adjacent tiles (also diagonal) both have roads, the edge weight between them gets updated to 0.5
- Villages and a players city count as a road tile
- a players units cannot use roads that lie withing enemy territory (player cntrl) 
- Placing a road costs 4 stars

### Inclusion of stars and stars per turn

- Creating units and upgrading cities costs stars 
- Owning cities grants stars per turn, which will be awarded to the player at the start of its turn.
 
### Inclusion of score

 - Score is awarded for cities under control, units under control and controlled tiles. 
 - This score is visible to every player in the game.
 
### Addition of multiple new units 
 
 - New units were added and need their logic implemented: Knight, Archer, Catapult, Giant, Sword, 
 - Knights need additional turn_state mechanic: When they kill a unit, they can attack again; this continues until either no unit is in range, or the knight attacked and didnt kill. Use the message argument to identify if the knight made a kill.
 - A giant can never be created in a city other than via the city upgrade mechanic.

### Upgrade to veteran unit mechanic

- Each unit counts the number of enemy unit it killed.
- When 3 kills are reached, the unit can upgrade to veteran status at any time during the players turn.
- upgrading to veteran increases the total hp and current hp (it fully heals the unit) to the original hp + 5

### Healing mechanic

 - Healing units heals 2.0 hp outside of their own territory and 4.0 hp inside their own territory
 - After having healed a unit, it becomes idle
 - A unit can only be healed when in the ready state

### Upgrade City

 - A city may be upgraded anytime during the players turn, if enough stars are available.
 - The player can always choose between 0 and 1 (two choices) for the city upgrade with different effects. 
 - Workshop vs Explorer; A workshop means one more stars per turn for the city, an explorer is a special unit that is created on upgrade and moves 14 moves in random directions, irregardless of enemy zoc, occupied tiles, only by water
 - Wall vs resources; A wall increases the defense bonus, choosing resources will reward 5 stars back to the player after upgrade
 - Popgrwth vs bordergrwth; Choosing popgrwth will reduce the cost of the next city upgrade by 6 stars, choosing bordergrwth instead increases the number of player controlled tiles around the city to a radius of 2 tiles around the city
 - after this, the player can choose between a park, or a super_unit (su for short); the park functions exactly like a workshop and choosing su, will invoke CreatUnit and create a Giant on the city (also takes up one space of the city).


## Environment wrapper

### Moving the renderer to its own python file

- Currently the render function is build into the EnvWrapper, but it is probably better to isolate it into its own module. 

### action mask generation

- Added the new action types to the action mask (HealUnit, UpgradeCity, PlaceRoad, Upgrade2Veteran)
- Need to include the stars cost to every possible action to zero out actions that are too expensive




