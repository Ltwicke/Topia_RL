# Current To-Dos and open design choices

This file lays down some important design choices for the todos that should be existentialized.

## Game simulator overhaul and additions


## Environment wrapper


### Rendering function visual changes

- Currently, there is no visualization for bridges. When roads are rendered, check if the underlying tile is a water tile. If its a water tile, render a bridge with the _bridge_axis direction of the bridge.


### More complex UI on the horizon
IGNORE FOR NOW!
- In the long term, the goal is that the user can play against a trained agent. For this, a more refined UI will be needed. But this is something for the future.


## RL agent model and PPO pipeline



### Entropy fusing strategy

Not important for now!

- Currently, entropy is just fused with a linear input projection using nn.Linear onto all of the necessary embeddings.
- Given the objective being to push the network into choosing clear decisions (those with low entropy, essentially telling, this is a must do move), is there a better way to include these low level entropy values?




