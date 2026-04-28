# Current To-Dos and open design choices

This file lays down some important design choices for the todos that should be existentialized.

## Game simulator overhaul and additions

### Mountains give defensive bonus

- When a unit stands on a mountain tile, it gets the defensive bonus
- When a unit is moved to a mountain tile, it uncoveres tiles with a vision range of 2 (not permanently, only on a mountain tile)

### Place roads on water (=bridges)

- Use the same PlaceRoad head and allow to place roads on water tiles, only connecting from south to north and west to east (no diagonal connections)
- Only works if both tiles in either South->North or West->East are field or mountain tiles (any tile to walk on)
- A bridge is treated like a standard road tile once placed.
- Placing a bridge (=A road on a water tile) costs 9 stars instead of 5!

## Environment wrapper

### Notion of center in each tile

- Each tile needs a notion of how much in the center of the map it is and how much on the edge it is. 
- This absolut notion is important because generally, the center of the map is more valuable (like in chess) that the edges and there is no learnable feature that classifies this notion being close to the center.
- What would be the best way to include this to the raw node features?


### Custom game state editor

- For the unit tests after every update of the PPO line, I need a comfortable way to edit user-defined game states.
- A game state is a user-defined state of the game board and player states. From the pre-defined state, the game simulation can unfolds without any furhter disturbances as if it were to happen in a real game.


### Rendering function visual changes

- instead of a grey tile, draw a mountain icon in grey onto the tile (background is same as field tile)


### More complex UI on the horizon
IGNORE FOR NOW!
- In the long term, the goal is that the user can play against a trained agent. For this, a more refined UI will be needed. But this is something for the future.


## RL agent model and PPO pipeline


### Per update unit tests

- There is an array of unit tests which consists of predefined game situations, where after every successful update of the PPO, they should be run multiple times (~20 times per situation) and logged.
- Some tests are logged as pngs. A good naming strategy will be needed here.
- I also need the ability to hardcode some actions which will be taken before the agent gets to decide.


### Entropy fusing strategy

- Currently, entropy is just fused with a linear input projection using nn.Linear onto all of the necessary embeddings.
- Given the objective being to push the network into choosing clear decisions (those with low entropy, essentially telling, this is a must do move), is there a better way to include these low level entropy values?




