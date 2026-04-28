# Current To-Dos and open design choices

This file lays down some important design choices for the todos that should be existentialized.

## Game simulator overhaul and additions

### game board creation logic

- Domains: Split the map in 4 domains (4 quadrants): Each players capital gets assigned to one of the quadrants.
- For the following use the helper function DistanceToEdgeX and DistanceToEdgeY
- 
- Suburb village- placements: 2 villages are associated with each capital and placed within 2 tiles (means: 2 tiles are in between the village and capital tile) from the capital.
- pre-terrain village placements: after suburbs have been placed, follow the density formula for additional villages placed: (([map_width]/3)^2-[#capitals+#suburbs])*0.3 
- Then add mountains and water
- post-terrain village placements: after terrain, add more villages until all eligible spots are filled. They must be 3 tiles from the edge of the map and 2 tiles from any other village.
 

- DRYLANDS: No water tiles
- LAKES: 25%-30% water ("wetness coeficient")
- ARCHIPELAGO: wetness: 60-80%

## Environment wrapper

### Notion of center in each tile

- Each tile needs a notion of how much in the center of the map it is and how much on the edge it is. 
- This absolut notion is important because generally, the center of the map is more valuable (like in chess) that the edges and there is no learnable feature that classifies this notion being close to the center.
- What would be the best way to include this to the raw node features?


### Custom game state editor

- For the unit tests after every update of the PPO line, I need a comfortable way to edit user-defined game states.
- A game state is a user-defined state of the game board and player states. From the pre-defined state, the game simulation can unfolds without any furhter disturbances as if it were to happen in a real game.


### Rendering function including the trajectories and hidden tile estimation

- Remove the little dots in the movement rendering on every potential target tile; they are not needed
- Do not draw any filling for the unit figure shapes for the hidden tiles rendering. It should only be the outline
- Add more jitter to the unit figure shapes.
- Do not switch the position of player 0 and player 1s partial graph in the rendering from left to right, keep them fixed!
- Currently, the coloring scheme of the hidden tiles opponent controll is coded to depend on the player_go_id. However, in the dual_rendering, this is wrong, the color scheme should depend on which players partial graph is shown in the dual view (the red player should always have blue as the hidden player controll estimate color)


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




