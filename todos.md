# Current To-Dos and open design choices

This file lays down some important design choices for the todos that should be existentialized.

## Game simulator overhaul and additions

All done for now.

## Environment wrapper

### Notion of center in each tile

- Each tile needs a notion of how much in the center of the map it is and how much on the edge it is. 
- This absolut notion is important because generally, the center of the map is more valuable (like in chess) that the edges and there is no learnable feature that classifies this notion being close to the center.
- What would be the best way to include this to the raw node features?

### Observation handling refactoring in environmentwrapper

- The function _get_obs collects the observations to be used in the policy agent network DONE
- The function needs to be refactored to include the new features. Specifically, the own current stars, stars per turn and own and opponents scores as well as the turn number. DONE
- Also add the fully uncovered partial graph for the hidden tiles estimator. DONE
- Changes here also need to cascade into make_snapshots in policy.py in the models folder. DONE

### Custom game state editor

- For the unit tests after every update of the PPO line, I need a comfortable way to edit user-defined game states.
- A game state is a user-defined state of the game board and player states. From the pre-defined state, the game simulation can unfolds without any furhter disturbances as if it were to happen in a real game.


### Rendering function including the trajectories and hidden tile estimation

- The current render() function and render_with_trajs() function need to be fused.
- The new resulting render function should include the new visuals of the current render function and include the trajectories of the chosen action and the overall probabilities for each action type.
- To show the trajectories with the red and blue color values on the tiles, change the player conrtoll feature visualization to the total outline of the controlled area (draw a colored border around the controlled space and connect the borders when two borders of the same player run into each other.)
- Additionally, if toggled to true, the render function should also include a visualization for the hidden tile estimation. 
- For this, draw both players pov side by side for every rendering call.
- On the current players pov, draw the estimation in the following way:
- Draw the color of the field type with an alpha value corresponding to the value given by the estimator. 
- Draw the unit type as figure outline shapes with an alpha value corresponding to the value given by the estimator. Add a little noise for the center of the figure shape to wiggle the shapes a bit so they dont lie exactly on top of each other.
- Draw the estimated player tile control as a light blue or red shadow, according to the respective players pov.
- Draw roads with the alpha value according to the sigmoid estimation.
- Draw city positions and levels while ignoring the choices; they are not important in this hidden tile estimation. 
- For all drawings, to signal the fact that the tiles are hidden, draw a light grey shadow over the tiles in the visualization.
- Draw the info box on top and make it horizontally long.

### More complex UI on the horizon
IGNORE FOR NOW!
- In the long term, the goal is that the user can play against a trained agent. For this, a more refined UI will be needed. But this is something for the future.


## RL agent model and PPO pipeline

### New modules for Version 2.0

ALL DONE

- New modules need to be added to the policy network: 
- HealUnit Head (very similar to the CaptureCityHead): Goes over all units, computes pairwise attention and returns a container with the results
- UpgradeCity Head (similar to create city selection head): Could also include a notion of the entropy in the two possible choices, but this doesnt need its own class, just include it into the UpgradeCity Head controlled by an on/ off flag.
- PlaceRoad Head: A simple head that scores the probability to place a road on any visible tile to the player. Here, its also very important to calculate the entropy and bring it into the equation for the final action type head.
- UpgradeUnit Head: (similar to HealUnit Head): Goes over all units, computes pairwise attention and returns a container with the results.
- All of the above new modules need to be bundled in policy.py 

### Pooling strategy for MultiScale convs

- Always pool with the max function, not with the mean function. DONE


### Hidden tile estimator network

- Interfaces the policy agent after the main node embedding layers. DONE
- add the class code into the main_modules.py. This estimator can be either a super minimal single FCNN with a couple of hidden layers or it could be another (smaller) graph transformer. DONE
- Make softmax for the sets of slices, where the values must add to 1 (for example for all unit types, all field types, etc.) DONE
- How this is integrated into the training: The loss will be the sum of each tile estimation loss which will be some form of logistic loss for every vector dimension. The resulting loss will be backpropagated to the node embedding layers. After 5-10 epochs of this, ppo is run.


### Per update unit tests

- There is an array of unit tests which consists of predefined game situations, where after every successful update of the PPO, they should be run multiple times (~20 times per situation) and logged.
- Some tests are logged as pngs. A good naming strategy will be needed here.
- I also need the ability to hardcode some actions which will be taken before the agent gets to decide.


### Entropy fusing strategy

- Currently, entropy is just fused with a linear input projection using nn.Linear onto all of the necessary embeddings.
- Given the objective being to push the network into choosing clear decisions (those with low entropy, essentially telling, this is a must do move), is there a better way to include these low level entropy values?




